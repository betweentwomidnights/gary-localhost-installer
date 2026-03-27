use crate::manifest::ServiceDef;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

#[derive(Debug, Clone, Serialize)]
pub struct ServiceInfo {
    pub id: String,
    pub display_name: String,
    pub port: u16,
    pub status: String,
    pub pid: Option<u32>,
    pub error: Option<String>,
    pub env_exists: bool,
    pub health_endpoint: Option<String>,
    pub build_status: Option<BuildStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BuildStatus {
    pub building: bool,
    pub current_step: usize,
    pub total_steps: usize,
    pub step_label: String,
    pub log: String,
    pub error: Option<String>,
}

struct RunningService {
    process: Child,
    healthy: bool,
}

pub struct ServiceManager {
    services: Vec<ServiceDef>,
    repo_root: PathBuf,
    running: HashMap<String, RunningService>,
    errors: HashMap<String, String>,
    build_statuses: HashMap<String, BuildStatus>,
}

impl ServiceManager {
    pub fn new(services: Vec<ServiceDef>, repo_root: PathBuf) -> Self {
        Self {
            services,
            repo_root,
            running: HashMap::new(),
            errors: HashMap::new(),
            build_statuses: HashMap::new(),
        }
    }

    fn service_dir(&self, svc: &ServiceDef) -> PathBuf {
        self.repo_root.join(&svc.working_dir)
    }

    fn env_dir(&self, svc: &ServiceDef) -> PathBuf {
        self.service_dir(svc).join("env")
    }

    fn python_exe(&self, svc: &ServiceDef) -> PathBuf {
        self.env_dir(svc).join("Scripts").join("python.exe")
    }

    fn find_service(&self, id: &str) -> Option<&ServiceDef> {
        self.services.iter().find(|s| s.id == id)
    }

    /// Resolve template variables like ${APPDATA}, ${HF_TOKEN}, ${MODELS_DIR}
    fn resolve_env_var(&self, value: &str) -> String {
        let mut result = value.to_string();

        // Resolve ${APPDATA}
        if let Ok(appdata) = std::env::var("APPDATA") {
            result = result.replace("${APPDATA}", &appdata);
        }

        // Resolve ${HF_TOKEN} — check stored file first, then system env
        if result.contains("${HF_TOKEN}") {
            if let Some(hf_token) = crate::read_hf_token() {
                result = result.replace("${HF_TOKEN}", &hf_token);
            }
        }

        // Resolve ${MODELS_DIR} — defaults to %APPDATA%/Gary4JUCE/models
        if result.contains("${MODELS_DIR}") {
            let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| {
                std::env::var("APPDATA")
                    .map(|a| format!("{}\\Gary4JUCE\\models", a))
                    .unwrap_or_default()
            });
            result = result.replace("${MODELS_DIR}", &models_dir);
        }

        result
    }

    /// Check if a running process has exited (crash detection)
    pub fn check_processes(&mut self) {
        let mut exited = Vec::new();

        for (id, running) in &mut self.running {
            match running.process.try_wait() {
                Ok(Some(status)) => {
                    let msg = if status.success() {
                        format!("Process exited cleanly (code 0)")
                    } else {
                        format!(
                            "Process crashed (exit code: {})",
                            status.code().map(|c| c.to_string()).unwrap_or("signal".into())
                        )
                    };
                    log::warn!("{}: {}", id, msg);
                    exited.push((id.clone(), msg));
                }
                Ok(None) => {} // still running
                Err(e) => {
                    log::error!("{}: error checking process: {}", id, e);
                }
            }
        }

        for (id, msg) in exited {
            self.running.remove(&id);
            self.errors.insert(id, msg);
        }
    }

    /// Update health status for a running service
    pub fn set_health(&mut self, service_id: &str, healthy: bool) {
        if let Some(running) = self.running.get_mut(service_id) {
            running.healthy = healthy;
        }
    }

    /// Get list of running services with their health endpoints for polling
    pub fn get_health_targets(&self) -> Vec<(String, u16, String)> {
        self.services
            .iter()
            .filter_map(|svc| {
                if !self.running.contains_key(&svc.id) {
                    return None;
                }
                let endpoint = svc
                    .health_check
                    .as_ref()
                    .map(|h| h.endpoint.clone())
                    .unwrap_or_else(|| "/health".to_string());
                Some((svc.id.clone(), svc.port, endpoint))
            })
            .collect()
    }

    pub fn get_service_info(&self) -> Vec<ServiceInfo> {
        self.services
            .iter()
            .map(|svc| {
                let running = self.running.get(&svc.id);
                let is_running = running.is_some();
                let pid = running.and_then(|r| r.process.id().into());
                let healthy = running.map(|r| r.healthy).unwrap_or(false);
                let error = self.errors.get(&svc.id).cloned();
                let env_exists = self.env_dir(svc).join("Scripts").join("python.exe").exists();

                let health_endpoint = svc
                    .health_check
                    .as_ref()
                    .map(|h| format!("http://localhost:{}{}", svc.port, h.endpoint));

                let status = if is_running && healthy {
                    "running".to_string()
                } else if is_running {
                    "starting".to_string()
                } else if error.is_some() {
                    "failed".to_string()
                } else {
                    "stopped".to_string()
                };

                let build_status = self.build_statuses.get(&svc.id).cloned();

                ServiceInfo {
                    id: svc.id.clone(),
                    display_name: svc.display_name.clone(),
                    port: svc.port,
                    status,
                    pid,
                    error,
                    env_exists,
                    health_endpoint,
                    build_status,
                }
            })
            .collect()
    }

    pub fn start(&mut self, service_id: &str) -> Result<(), String> {
        let svc = self
            .find_service(service_id)
            .ok_or_else(|| format!("Unknown service: {}", service_id))?
            .clone();

        if self.running.contains_key(service_id) {
            return Err(format!("{} is already running", service_id));
        }

        let python = self.python_exe(&svc);
        if !python.exists() {
            return Err(format!(
                "Python venv not found at {}. Build the environment first.",
                python.display()
            ));
        }

        let work_dir = self.service_dir(&svc);
        let log_path = work_dir.join(format!("{}.log", svc.id));

        self.errors.remove(service_id);

        let log_file = std::fs::File::create(&log_path)
            .map_err(|e| format!("Cannot create log file: {}", e))?;
        let log_file_err = log_file
            .try_clone()
            .map_err(|e| format!("Cannot clone log handle: {}", e))?;

        let mut cmd = Command::new(&python);
        cmd.arg(&svc.entry_point)
            .current_dir(&work_dir)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_file_err))
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            // Ensure the working directory is on Python's import path.
            // Critical for services like Carey where the entry point is in a
            // subdirectory (acestep/api_server.py) but imports the parent package.
            .env("PYTHONPATH", &work_dir);

        // Set service-specific env vars (with template resolution)
        for (k, v) in &svc.env {
            let resolved = self.resolve_env_var(v);
            if !resolved.is_empty() && !resolved.contains("${") {
                cmd.env(k, &resolved);
            }
        }

        // Prevent console window on Windows
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let child = cmd.spawn().map_err(|e| format!("Failed to start {}: {}", service_id, e))?;

        log::info!(
            "Started {} (PID {}) from {}",
            service_id,
            child.id(),
            work_dir.display()
        );

        self.running.insert(
            service_id.to_string(),
            RunningService {
                process: child,
                healthy: false,
            },
        );

        Ok(())
    }

    pub fn stop(&mut self, service_id: &str) -> Result<(), String> {
        if let Some(mut running) = self.running.remove(service_id) {
            log::info!("Stopping {}", service_id);
            let _ = running.process.kill();
            let _ = running.process.wait();
            self.errors.remove(service_id);
            Ok(())
        } else {
            Err(format!("{} is not running", service_id))
        }
    }

    pub fn is_running(&self, service_id: &str) -> bool {
        self.running.contains_key(service_id)
    }

    pub fn stop_all(&mut self) {
        let ids: Vec<String> = self.running.keys().cloned().collect();
        for id in ids {
            let _ = self.stop(&id);
        }
    }

    /// Get build info needed to launch an async build
    pub fn get_build_info(&self, service_id: &str) -> Result<BuildInfo, String> {
        let svc = self
            .find_service(service_id)
            .ok_or_else(|| format!("Unknown service: {}", service_id))?;

        if svc.build_steps.is_empty() {
            return Err(format!("No build steps defined for {}", service_id));
        }

        if self.build_statuses.get(service_id).map_or(false, |b| b.building) {
            return Err(format!("{} is already building", service_id));
        }

        Ok(BuildInfo {
            service_id: svc.id.clone(),
            work_dir: self.service_dir(svc),
            env_dir: self.env_dir(svc),
            build_steps: svc.build_steps.clone(),
        })
    }

    /// Mark a build as started
    pub fn set_build_started(&mut self, service_id: &str, total_steps: usize) {
        self.build_statuses.insert(service_id.to_string(), BuildStatus {
            building: true,
            current_step: 0,
            total_steps,
            step_label: "Creating virtual environment...".to_string(),
            log: String::new(),
            error: None,
        });
    }

    /// Update build progress
    pub fn set_build_step(&mut self, service_id: &str, step: usize, label: &str) {
        if let Some(status) = self.build_statuses.get_mut(service_id) {
            status.current_step = step;
            status.step_label = label.to_string();
        }
    }

    /// Append to build log
    pub fn append_build_log(&mut self, service_id: &str, line: &str) {
        if let Some(status) = self.build_statuses.get_mut(service_id) {
            status.log.push_str(line);
            status.log.push('\n');
            // Keep only last 100KB
            if status.log.len() > 100_000 {
                let start = status.log.len() - 80_000;
                let trimmed = status.log[start..].to_string();
                status.log = trimmed;
            }
        }
    }

    /// Mark build as completed
    pub fn set_build_done(&mut self, service_id: &str, error: Option<String>) {
        if let Some(status) = self.build_statuses.get_mut(service_id) {
            status.building = false;
            status.error = error;
            if status.error.is_none() {
                status.step_label = "Build complete".to_string();
                status.current_step = status.total_steps;
            }
        }
    }

    pub fn get_all_build_infos(&self) -> Vec<BuildInfo> {
        self.services.iter().filter_map(|svc| {
            if svc.build_steps.is_empty() { return None; }
            if self.build_statuses.get(&svc.id).map_or(false, |b| b.building) { return None; }
            Some(BuildInfo {
                service_id: svc.id.clone(),
                work_dir: self.service_dir(svc),
                env_dir: self.env_dir(svc),
                build_steps: svc.build_steps.clone(),
            })
        }).collect()
    }

    pub fn get_log(&self, service_id: &str) -> Result<String, String> {
        let svc = self
            .find_service(service_id)
            .ok_or_else(|| format!("Unknown service: {}", service_id))?;

        let is_running = self.running.contains_key(service_id);
        let is_building = self.build_statuses.get(service_id).map_or(false, |b| b.building);

        // Priority:
        // 1. If actively building -> show build log
        // 2. If running (or was running and has a runtime log) -> show runtime log
        // 3. If build completed and no runtime log -> show build log
        // 4. Otherwise -> empty

        if is_building {
            let log = self.build_statuses.get(service_id)
                .map(|b| b.log.clone())
                .unwrap_or_default();
            return Ok(log);
        }

        let log_path = self.service_dir(svc).join(format!("{}.log", svc.id));

        if log_path.exists() && (is_running || !is_building) {
            let runtime_log = self.read_log_file(&log_path)?;
            if !runtime_log.is_empty() {
                return Ok(runtime_log);
            }
        }

        // Fall back to build log if available
        if let Some(build_status) = self.build_statuses.get(service_id) {
            if !build_status.log.is_empty() {
                return Ok(build_status.log.clone());
            }
        }

        Ok(String::new())
    }

    fn read_log_file(&self, log_path: &std::path::Path) -> Result<String, String> {
        let metadata = std::fs::metadata(log_path)
            .map_err(|e| format!("Cannot read log: {}", e))?;
        let file_size = metadata.len();
        let max_read: u64 = 256 * 1024;

        if file_size > max_read {
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(log_path)
                .map_err(|e| format!("Cannot open log: {}", e))?;
            file.seek(SeekFrom::End(-(max_read as i64)))
                .map_err(|e| format!("Seek error: {}", e))?;
            let mut buf = String::new();
            file.read_to_string(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            if let Some(pos) = buf.find('\n') {
                Ok(buf[pos + 1..].to_string())
            } else {
                Ok(buf)
            }
        } else {
            std::fs::read_to_string(log_path)
                .map_err(|e| format!("Cannot read log: {}", e))
        }
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn service_count(&self) -> usize {
        self.services.len()
    }
}

pub struct BuildInfo {
    pub service_id: String,
    pub work_dir: PathBuf,
    pub env_dir: PathBuf,
    pub build_steps: Vec<String>,
}

pub fn install_xformers_shim(env_dir: &PathBuf) -> Result<(), String> {
    let site_packages = env_dir.join("Lib").join("site-packages");
    let xformers_dir = site_packages.join("xformers");
    let ops_dir = xformers_dir.join("ops");

    std::fs::create_dir_all(&ops_dir)
        .map_err(|e| format!("Cannot create xformers shim dir: {}", e))?;

    std::fs::write(
        xformers_dir.join("__init__.py"),
        "__version__ = \"0.0.0+sdpa_shim\"\n",
    )
    .map_err(|e| format!("Cannot write xformers __init__.py: {}", e))?;

    std::fs::write(
        ops_dir.join("__init__.py"),
        r#"import torch
from torch.nn.functional import scaled_dot_product_attention as _sdpa

__all__ = ["memory_efficient_attention", "LowerTriangularMask", "unbind"]

class LowerTriangularMask:
    def __init__(self, *args, **kwargs): pass

def unbind(x, dim=0):
    return torch.unbind(x, dim=dim)

def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)
    causal = isinstance(attn_bias, LowerTriangularMask)
    dropout_p = p if (q.requires_grad and q.is_cuda and q.dtype.is_floating_point) else 0.0
    return _sdpa(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=causal, scale=scale)
"#,
    )
    .map_err(|e| format!("Cannot write xformers ops/__init__.py: {}", e))?;

    log::info!("Installed xformers SDPA shim");
    Ok(())
}
