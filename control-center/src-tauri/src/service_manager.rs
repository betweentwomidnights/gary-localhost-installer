use crate::manifest::{HealthCheck, NativeDef, NativePackage, ServiceDef, ServiceRuntime};
use crate::native_runtime::{self, NativeInstall};
use serde::Serialize;
use std::collections::HashMap;
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

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
    /// "python" or "native".
    pub runtime: String,
    /// For a native service, the backend its installed runtime launches on.
    pub native_backend: Option<String>,
    /// The installed native runtime is older than the one this build pins.
    pub native_update_available: bool,
    /// Why an automatic backend choice had to fall back, if it did.
    pub native_fallback_reason: Option<String>,
    /// Why the service cannot start yet (e.g. its models are missing), shown
    /// in place of a start that would only fail its first request.
    pub start_blocker: Option<String>,
    /// SA3's native migration and reviewed Python cleanup have finished.
    pub sa3_migration_complete: bool,
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
    started_at: Instant,
    last_health_check_at: Option<Instant>,
    _adapter: Option<crate::sa3_adapter::AdapterHandle>,
}

pub struct HealthTarget {
    pub id: String,
    pub port: u16,
    pub endpoint: String,
    pub timeout_seconds: u64,
}

pub struct ServiceManager {
    services: Vec<ServiceDef>,
    native_runtimes: HashMap<String, NativePackage>,
    repo_root: PathBuf,
    running: HashMap<String, RunningService>,
    errors: HashMap<String, String>,
    build_statuses: HashMap<String, BuildStatus>,
    /// Native tools such as training/conversion also hold bundle files open.
    /// Keep those users visible to installation and storage maintenance.
    native_workloads: HashMap<String, HashMap<String, String>>,
    /// Model preparation may replace weights. Keep launches and native tools
    /// from opening them between the preflight check and the final hash check.
    native_model_mutations: HashMap<String, String>,
    /// Weak leases expire even if an installer future is cancelled.
    shared_runtime_mutations: HashMap<String, Weak<String>>,
    sa3_selection: Option<crate::sa3_runtime::Selection>,
    sa3_selection_error: Option<String>,
}

fn health_check_interval(
    health: &HealthCheck,
    healthy: bool,
    elapsed_since_start: Duration,
) -> Duration {
    let normal_interval = Duration::from_secs(health.interval_seconds.max(1));
    let startup_grace = Duration::from_secs(health.startup_grace_seconds);

    if healthy {
        normal_interval
    } else if elapsed_since_start < startup_grace {
        Duration::from_secs(1)
    } else {
        normal_interval.min(Duration::from_secs(5))
    }
}

fn is_successful_health_access_log(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if !lower.contains("/health") {
        return false;
    }

    let is_health_request = [
        "\"get /health",
        "\"head /health",
        " get /health",
        " head /health",
    ]
    .iter()
    .any(|marker| lower.contains(marker));
    if !is_health_request {
        return false;
    }

    ["\" 200", "\" 204", "\" 304", " 200 ", " 204 ", " 304 "]
        .iter()
        .any(|marker| lower.contains(marker) || lower.ends_with(marker.trim_end()))
}

fn filter_successful_health_access_logs(raw: &str) -> String {
    let mut filtered = String::with_capacity(raw.len());
    for line in raw.lines() {
        if is_successful_health_access_log(line) {
            continue;
        }
        if !filtered.is_empty() {
            filtered.push('\n');
        }
        filtered.push_str(line);
    }
    if raw.ends_with('\n') && !filtered.is_empty() {
        filtered.push('\n');
    }
    filtered
}

impl ServiceManager {
    pub fn new(mut services: Vec<ServiceDef>, repo_root: PathBuf) -> Self {
        let (sa3_selection, sa3_selection_error) = match crate::sa3_runtime::read(&repo_root) {
            Ok(selection) => (selection, None),
            Err(error) => (None, Some(error)),
        };
        if sa3_selection.is_some() || sa3_selection_error.is_some() {
            if let Some(service) = services.iter_mut().find(|service| service.id == "sa3") {
                service.runtime = ServiceRuntime::Native;
            }
        }
        Self {
            services,
            native_runtimes: HashMap::new(),
            repo_root,
            running: HashMap::new(),
            errors: HashMap::new(),
            build_statuses: HashMap::new(),
            native_workloads: HashMap::new(),
            native_model_mutations: HashMap::new(),
            shared_runtime_mutations: HashMap::new(),
            sa3_selection,
            sa3_selection_error,
        }
    }

    /// Shared runtimes (e.g. the CUDA redistributables) native backends need.
    pub fn set_native_runtimes(&mut self, runtimes: HashMap<String, NativePackage>) {
        self.native_runtimes = runtimes;
    }

    fn service_dir(&self, svc: &ServiceDef) -> PathBuf {
        self.repo_root.join(&svc.working_dir)
    }

    /// A Python service's venv, or a native service's unpacked runtime.
    fn env_dir(&self, svc: &ServiceDef) -> PathBuf {
        match svc.runtime {
            ServiceRuntime::Python => self.service_dir(svc).join("env"),
            ServiceRuntime::Native => self.native_dir(svc),
        }
    }

    fn native_dir(&self, svc: &ServiceDef) -> PathBuf {
        match svc
            .native
            .as_ref()
            .and_then(|native| native.bundle.as_deref())
        {
            Some(bundle) => self
                .services_dir()
                .join(bundle)
                .join(native_runtime::NATIVE_DIR),
            None => self.service_dir(svc).join(native_runtime::NATIVE_DIR),
        }
    }

    pub fn native_dir_for(&self, service_id: &str) -> Option<PathBuf> {
        let svc = self.find_service(service_id)?;
        svc.native.as_ref()?;
        Some(self.native_dir(svc))
    }

    fn bundle_consumers(&self, svc: &ServiceDef) -> Vec<&ServiceDef> {
        let dir = self.native_dir(svc);
        self.services
            .iter()
            .filter(|candidate| candidate.native.is_some() && self.native_dir(candidate) == dir)
            .collect()
    }

    fn uses_shared_runtime(&self, svc: &ServiceDef, name: &str) -> bool {
        if let Some(install) = self.native_install(svc) {
            if install.version.is_some() {
                return install.runtimes.iter().any(|runtime| runtime == name);
            }
        }
        // A development override or a not-yet-installed bundle has no stamp.
        // Conservatively include every dependency offered on this platform.
        svc.native
            .as_ref()
            .and_then(|native| native.platforms.get(native_runtime::current_platform()))
            .is_some_and(|platform| {
                platform
                    .backends
                    .values()
                    .any(|package| package.requires.iter().any(|runtime| runtime == name))
            })
    }

    fn shared_runtime_blocker(&self, svc: &ServiceDef) -> Option<String> {
        self.shared_runtime_mutations
            .iter()
            .find_map(|(name, lease)| {
                let label = lease.upgrade()?;
                self.uses_shared_runtime(svc, name)
                    .then(|| format!("wait for {label} to finish"))
            })
    }

    /// Reserve only a shared runtime that actually needs replacement. Matching
    /// packages are reused without a lease, so other services can keep running.
    /// The caller owns the lease through its final filesystem mutation.
    pub fn begin_shared_runtime_replacement(&mut self, name: &str) -> Result<Arc<String>, String> {
        if self
            .shared_runtime_mutations
            .get(name)
            .and_then(Weak::upgrade)
            .is_some()
        {
            return Err(format!(
                "Cannot replace {name}: another replacement is in progress."
            ));
        }
        for svc in &self.services {
            if !self.uses_shared_runtime(svc, name) {
                continue;
            }
            if svc.runtime == ServiceRuntime::Native && self.is_running(&svc.id) {
                return Err(format!(
                    "Cannot replace {name}: stop {} first.",
                    svc.display_name
                ));
            }
            if let Some(label) = self.native_model_mutations.get(&svc.id) {
                return Err(format!(
                    "Cannot replace {name}: wait for {label} to finish."
                ));
            }
            if let Some(label) = self
                .native_workloads
                .get(&self.native_dir(svc).to_string_lossy().to_string())
                .and_then(|jobs| jobs.values().next())
            {
                return Err(format!(
                    "Cannot replace {name}: wait for {label} to finish."
                ));
            }
        }
        let lease = Arc::new(format!("{name} replacement"));
        self.shared_runtime_mutations
            .insert(name.into(), Arc::downgrade(&lease));
        Ok(lease)
    }

    /// Called under the manager lock before reserving an install/removal.
    pub fn native_mutation_blocker(&self, service_id: &str) -> Option<String> {
        let svc = self.find_service(service_id)?;
        if let Some(blocker) = self.shared_runtime_blocker(svc) {
            return Some(blocker);
        }
        for consumer in self.bundle_consumers(svc) {
            if let Some(label) = self.native_model_mutations.get(&consumer.id) {
                return Some(format!("wait for {label} to finish"));
            }
            if consumer.runtime == ServiceRuntime::Native && self.is_running(&consumer.id) {
                return Some(format!("stop {} first", consumer.display_name));
            }
            if self.is_building(&consumer.id) {
                return Some(format!(
                    "wait for {}'s install to finish",
                    consumer.display_name
                ));
            }
        }
        if let Some(workloads) = self
            .native_workloads
            .get(&self.native_dir(svc).to_string_lossy().to_string())
        {
            if let Some(label) = workloads.values().next() {
                return Some(format!("wait for {label} to finish"));
            }
        }
        None
    }

    /// Reserve a bundle for a trainer/converter before spawning it, and release
    /// on process exit (including failure/cancellation). Concurrent readers are
    /// fine, but installers must see every reservation.
    pub fn begin_native_workload(
        &mut self,
        service_id: &str,
        job: &str,
        label: &str,
    ) -> Result<(), String> {
        let svc = self
            .find_service(service_id)
            .ok_or_else(|| format!("Unknown service: {service_id}"))?;
        if svc.native.is_none() {
            return Err(format!("{service_id} has no native runtime"));
        }
        if let Some(blocker) = self.shared_runtime_blocker(svc) {
            return Err(format!("Cannot start native tool: {blocker}."));
        }
        if let Some(label) = self.native_model_mutations.get(service_id) {
            return Err(format!("Wait for {label} to finish"));
        }
        for consumer in self.bundle_consumers(svc) {
            if self.is_building(&consumer.id) {
                return Err(format!(
                    "Wait for {}'s install to finish",
                    consumer.display_name
                ));
            }
        }
        let key = self.native_dir(svc).to_string_lossy().to_string();
        self.native_workloads
            .entry(key)
            .or_default()
            .insert(job.to_string(), label.to_string());
        Ok(())
    }

    pub fn end_native_workload(&mut self, service_id: &str, job: &str) {
        let Some(dir) = self.native_dir_for(service_id) else {
            return;
        };
        let key = dir.to_string_lossy().to_string();
        if let Some(workloads) = self.native_workloads.get_mut(&key) {
            workloads.remove(job);
            if workloads.is_empty() {
                self.native_workloads.remove(&key);
            }
        }
    }

    pub fn begin_native_model_mutation(
        &mut self,
        service_id: &str,
        label: &str,
    ) -> Result<(), String> {
        if self.native_service(service_id).is_none() {
            return Err(format!("{service_id} has no native model runtime"));
        }
        if let Some(blocker) = self.native_mutation_blocker(service_id) {
            return Err(format!("Cannot prepare models: {blocker}."));
        }
        self.native_model_mutations
            .insert(service_id.into(), label.into());
        Ok(())
    }

    pub fn end_native_model_mutation(&mut self, service_id: &str) {
        self.native_model_mutations.remove(service_id);
    }

    fn native_install(&self, svc: &ServiceDef) -> Option<NativeInstall> {
        let native = svc.native.as_ref()?;
        native_runtime::installed(&svc.id, &self.native_dir(svc), &native.executable)
    }

    /// A native service's definition and install folder, for the UI.
    pub fn native_service(&self, service_id: &str) -> Option<(NativeDef, PathBuf)> {
        let svc = self.find_service(service_id)?;
        Some((svc.native.clone()?, self.native_dir(svc)))
    }

    pub fn sa3_migration_launch(
        &self,
    ) -> Result<(NativeDef, NativeInstall, Vec<(String, String)>), String> {
        let service = self.find_service("sa3").ok_or("Unknown SA3 service")?;
        let native = service
            .native
            .clone()
            .ok_or("SA3 has no native runtime definition")?;
        let installed = self
            .native_install(service)
            .ok_or("Prepare the SA3 C++ runtime first")?;
        Ok((native, installed, self.native_env_template(service)))
    }

    /// The service's env with every template resolved except
    /// `${NATIVE_BACKEND}`, which depends on the runtime installed.
    fn native_env_template(&self, svc: &ServiceDef) -> Vec<(String, String)> {
        let mut env = svc.env.clone();
        if let Some(native) = &svc.native {
            env.extend(native.env.clone());
        }
        env.iter()
            .map(|(key, value)| (key.clone(), self.resolve_env_var(value)))
            .collect()
    }

    fn services_dir(&self) -> PathBuf {
        self.repo_root.join("services")
    }

    /// Where a service's environment lives (a Python venv, or a native
    /// service's downloaded runtime), for callers that manage storage rather
    /// than processes.
    pub fn env_dir_for(&self, service_id: &str) -> Option<PathBuf> {
        self.find_service(service_id).map(|svc| self.env_dir(svc))
    }

    /// The root every service directory sits under. Storage cleanup uses this
    /// to prove a path it is about to delete is one of ours.
    pub fn managed_services_root(&self) -> PathBuf {
        self.services_dir()
    }

    /// Refresh cleanup journal fields after a transaction without changing the runtime.
    pub fn refresh_sa3_native_selection(&mut self) -> Result<(), String> {
        self.sa3_selection = crate::sa3_runtime::read(&self.repo_root)?;
        Ok(())
    }

    /// Whether a service runs a downloaded native runtime rather than Python.
    pub fn is_native(&self, service_id: &str) -> bool {
        self.find_service(service_id)
            .is_some_and(|svc| svc.runtime == ServiceRuntime::Native)
    }

    /// Called after migration has verified the installed native server. Persist
    /// the profile choice before cleanup, then update this session's launch path.
    pub fn activate_native_sa3(
        &mut self,
        mut selection: crate::sa3_runtime::Selection,
    ) -> Result<crate::sa3_runtime::Selection, String> {
        if self.is_running("sa3") {
            return Err("Stop SA3 before switching its runtime".into());
        }
        if self.native_service("sa3").is_none() {
            return Err("SA3 has no native runtime definition".into());
        }
        if let Some(previous) = &self.sa3_selection {
            selection.cleanup_complete = previous.cleanup_complete;
            selection.cleanup_errors = previous.cleanup_errors.clone();
        }
        crate::sa3_runtime::save(&self.repo_root, &selection)?;
        self.services
            .iter_mut()
            .find(|service| service.id == "sa3")
            .unwrap()
            .runtime = ServiceRuntime::Native;
        self.sa3_selection = Some(selection.clone());
        self.sa3_selection_error = None;
        self.errors.remove("sa3");
        Ok(selection)
    }

    /// Where the shared runtimes native services need are installed.
    pub fn native_runtimes_root(&self) -> PathBuf {
        native_runtime::runtimes_root(&self.repo_root)
    }

    fn models_dir(&self) -> PathBuf {
        crate::storage::models_dir(&self.repo_root)
    }

    fn cache_dir(&self) -> PathBuf {
        crate::storage::cache_dir(&self.repo_root)
    }

    fn hf_home_dir(&self) -> PathBuf {
        crate::storage::effective_hf_home_dir(&self.repo_root)
    }

    fn hf_hub_cache_dir(&self) -> PathBuf {
        crate::storage::effective_hf_hub_cache_dir(&self.repo_root)
    }

    fn python_exe(&self, svc: &ServiceDef) -> PathBuf {
        self.env_dir(svc).join("Scripts").join("python.exe")
    }

    fn find_service(&self, id: &str) -> Option<&ServiceDef> {
        self.services.iter().find(|s| s.id == id)
    }

    /// Resolve template variables like ${GARY_RUNTIME}, ${HF_TOKEN}, ${MODELS_DIR}
    fn resolve_env_var(&self, value: &str) -> String {
        let mut result = value.to_string();

        result = result.replace(
            "${GARY_RUNTIME}",
            &self.repo_root.to_string_lossy().to_string(),
        );
        result = result.replace(
            "${GARY4LOCAL_RUNTIME}",
            &self.repo_root.to_string_lossy().to_string(),
        );
        result = result.replace(
            "${SERVICES_DIR}",
            &self.services_dir().to_string_lossy().to_string(),
        );
        result = result.replace(
            "${MODELS_DIR}",
            &self.models_dir().to_string_lossy().to_string(),
        );
        result = result.replace(
            "${CACHE_DIR}",
            &self.cache_dir().to_string_lossy().to_string(),
        );
        result = result.replace(
            "${HF_HOME}",
            &self.hf_home_dir().to_string_lossy().to_string(),
        );
        result = result.replace(
            "${HF_HUB_CACHE}",
            &self.hf_hub_cache_dir().to_string_lossy().to_string(),
        );

        // Resolve ${APPDATA}
        if let Ok(appdata) = std::env::var("APPDATA") {
            result = result.replace("${APPDATA}", &appdata);
        }

        // Resolve ${HF_TOKEN} — check stored file first, then system env
        if result.contains("${HF_TOKEN}") {
            let hf_token = crate::read_hf_token().unwrap_or_default();
            result = result.replace("${HF_TOKEN}", &hf_token);
        }

        result
    }

    fn apply_runtime_env(&self, cmd: &mut Command) {
        for (key, value) in crate::storage::runtime_env_vars(&self.repo_root) {
            cmd.env(key, value);
        }
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
                            status
                                .code()
                                .map(|c| c.to_string())
                                .unwrap_or("signal".into())
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

    /// Get running services that are due for a health check now.
    pub fn take_due_health_targets(&mut self, now: Instant) -> Vec<HealthTarget> {
        let services = self.services.clone();
        let mut targets = Vec::new();

        for svc in services {
            let Some(health) = svc.health_check.as_ref() else {
                continue;
            };
            let Some(running) = self.running.get_mut(&svc.id) else {
                continue;
            };

            // Probe quickly while a service is still starting so the UI can
            // turn green as soon as the service reports ready. The manifest's
            // normal interval applies after the first successful health check.
            let interval = health_check_interval(
                health,
                running.healthy,
                now.duration_since(running.started_at),
            );
            if let Some(last_health_check_at) = running.last_health_check_at {
                if now.duration_since(last_health_check_at) < interval {
                    continue;
                }
            }

            running.last_health_check_at = Some(now);
            targets.push(HealthTarget {
                id: svc.id,
                port: svc.port,
                endpoint: health.endpoint.clone(),
                timeout_seconds: health.timeout_seconds.max(1),
            });
        }

        targets
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
                let native_install = (svc.runtime == ServiceRuntime::Native)
                    .then(|| self.native_install(svc))
                    .flatten();
                let env_exists = match svc.runtime {
                    ServiceRuntime::Python => self
                        .env_dir(svc)
                        .join("Scripts")
                        .join("python.exe")
                        .exists(),
                    ServiceRuntime::Native => native_install.is_some(),
                };
                let native_update_available = match (&native_install, &svc.native) {
                    (Some(install), Some(native)) => install
                        .version
                        .as_ref()
                        .is_some_and(|version| version != &native.version),
                    _ => false,
                };

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
                    runtime: match svc.runtime {
                        ServiceRuntime::Python => "python".to_string(),
                        ServiceRuntime::Native => "native".to_string(),
                    },
                    native_backend: native_install
                        .as_ref()
                        .map(|install| install.backend.clone()),
                    native_update_available,
                    native_fallback_reason: native_install
                        .and_then(|install| install.fallback_reason),
                    start_blocker: self.start_blocker(svc),
                    sa3_migration_complete: svc.id == "sa3"
                        && svc.runtime == ServiceRuntime::Native
                        && self.sa3_selection.as_ref().is_some_and(|selection| {
                            selection.cleanup_complete && selection.cleanup_errors.is_empty()
                        }),
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
        if let Some(blocker) = self.start_blocker(&svc) {
            return Err(format!("{} {blocker}.",svc.display_name));
        }

        if svc.runtime == ServiceRuntime::Native {
            return self.start_native(&svc);
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

        let mut python_paths = vec![work_dir.clone()];
        if let Some(shared_services_dir) = work_dir.parent() {
            python_paths.push(shared_services_dir.to_path_buf());
        }
        let python_path = std::env::join_paths(&python_paths)
            .map_err(|e| format!("Failed to construct PYTHONPATH: {}", e))?;

        let mut cmd = Command::new(&python);
        cmd.arg(&svc.entry_point)
            .current_dir(&work_dir)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_file_err))
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            // Include both the service directory and the shared services root
            // on Python's import path so packaged builds can resolve helpers
            // like local_session_store.py alongside per-service packages.
            .env("PYTHONPATH", python_path);
        self.apply_runtime_env(&mut cmd);

        // Set service-specific env vars (with template resolution)
        for (k, v) in &svc.env {
            let resolved = self.resolve_env_var(v);
            if !resolved.is_empty() && !resolved.contains("${") {
                cmd.env(k, &resolved);
            }
        }

        if svc.id == "sa3" {
            for (key, value) in crate::sa3_loudness_env() {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    cmd.env(key, trimmed);
                }
            }
            let use_decoder_lora = crate::sa3_use_decoder_lora_enabled();
            cmd.env(
                "SA3_USE_DECODER_LORA",
                if use_decoder_lora { "1" } else { "0" },
            );
        }

        if svc.id == "gary" {
            let use_fp16 = crate::gary_use_fp16_enabled();
            cmd.env("GARY_USE_FP16", if use_fp16 { "1" } else { "0" });
        }

        if svc.id == "melodyflow" {
            let use_flash_attn = crate::melodyflow_use_flash_attn_enabled();
            cmd.env(
                "MELODYFLOW_USE_FLASH_ATTN",
                if use_flash_attn { "1" } else { "0" },
            );
        }

        if svc.id == "carey" {
            let use_xl_models = crate::carey_use_xl_models_enabled();
            let base_config = if use_xl_models {
                "acestep-v15-xl-base"
            } else {
                "acestep-v15-base"
            };
            let sft_config = if use_xl_models {
                "acestep-v15-xl-sft"
            } else {
                "acestep-v15-sft"
            };
            let turbo_config = if use_xl_models {
                "acestep-v15-xl-turbo"
            } else {
                "acestep-v15-turbo"
            };

            cmd.env("ACESTEP_CONFIG_PATH", base_config)
                .env("ACESTEP_BASE_CONFIG_PATH", base_config)
                .env("ACESTEP_SFT_CONFIG_PATH", sft_config)
                .env("ACESTEP_TURBO_CONFIG_PATH", turbo_config)
                .env("ACESTEP_LEGO_CONFIG_PATH", base_config)
                .env("ACESTEP_REGULAR_CONFIG_PATH", base_config)
                .env("ACESTEP_NO_INIT", "true");

            if crate::carey_use_scrag_vae_enabled() {
                cmd.env("ACESTEP_VAE_PATH", "scrag-vae");
            } else {
                cmd.env_remove("ACESTEP_VAE_PATH");
            }
        }

        crate::workload_job::configure_std_command(&mut cmd);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to start {}: {}", service_id, e))?;
        if let Err(error) = crate::workload_job::enroll_std_child(&child) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Failed to enroll {} in the managed workload group: {}",
                service_id, error
            ));
        }

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
                started_at: Instant::now(),
                last_health_check_at: None,
                _adapter: None,
            },
        );

        Ok(())
    }

    fn start_blocker(&self, svc: &ServiceDef) -> Option<String> {
        if svc.id == "sa3" {
            if let Some(jobs) = self
                .native_workloads
                .get(&self.native_dir(svc).to_string_lossy().to_string())
            {
                if let Some((_, label)) = jobs.iter().find(|(job, _)| {
                    matches!(
                        job.as_str(),
                        "legacy-lora-edit"
                            | "lora-import"
                            | "lora-edit"
                            | "checkpoint-selection"
                            | "lora-preparation"
                            | "decoder-preparation"
                    )
                }) {
                    return Some(format!("is waiting for {label} to finish"));
                }
            }
        }
        if svc.id == "sa3"
            && self
                .native_workloads
                .get(&self.native_dir(svc).to_string_lossy().to_string())
                .is_some_and(|jobs| jobs.contains_key("native-migration"))
        {
            return Some("is waiting for native migration validation to finish".into());
        }
        if svc.id == "sa3" {
            if let Some(error) = &self.sa3_selection_error {
                return Some(format!(
                    "cannot use this profile's native selection: {error}"
                ));
            }
        }
        if svc.id == "sa3"
            && self
                .native_workloads
                .get(&self.native_dir(svc).to_string_lossy().to_string())
                .is_some_and(|jobs| jobs.contains_key("native-training"))
        {
            return Some("is waiting for native training to finish".into());
        }
        if svc.runtime == ServiceRuntime::Native {
            if let Some(blocker) = self.shared_runtime_blocker(svc) {
                return Some(blocker);
            }
            if let Some(label) = self.native_model_mutations.get(&svc.id) {
                return Some(format!("is waiting for {label} to finish"));
            }
            for consumer in self.bundle_consumers(svc) {
                if self.is_building(&consumer.id) {
                    return Some(format!(
                        "is waiting for {}'s install to finish",
                        consumer.display_name
                    ));
                }
            }
            if svc.id == "sa3" {
                let encoding = self
                    .sa3_selection
                    .as_ref()
                    .map_or("F16", |selection| selection.encoding.as_str());
                if let Some(missing) = crate::sa3_runtime::missing_models(&self.repo_root, encoding) {
                    return Some(missing);
                }
                if crate::sa3_use_decoder_lora_enabled() {
                    if let Some(blocker) = crate::sa3_decoder::blocker(&self.repo_root) {
                        return Some(blocker);
                    }
                }
            }
        }
        if svc.id == "yuey" {
            return crate::yuey_missing_models(&self.models_dir());
        }
        None
    }

    /// Launch a native service's executable from its installed runtime, with
    /// the backend it was installed for and any shared runtime on PATH.
    fn start_native(&mut self, svc: &ServiceDef) -> Result<(), String> {
        if let Some(blocker) = self.shared_runtime_blocker(svc) {
            return Err(format!("Cannot start {}: {blocker}.", svc.display_name));
        }
        if let Some(label) = self.native_model_mutations.get(&svc.id) {
            return Err(format!("Wait for {label} to finish."));
        }
        for consumer in self.bundle_consumers(svc) {
            if self.is_building(&consumer.id) {
                return Err(format!(
                    "Wait for {}'s runtime install to finish.",
                    consumer.display_name
                ));
            }
        }
        let native = svc
            .native
            .as_ref()
            .ok_or_else(|| format!("{} has no native definition", svc.id))?;
        let install = self.native_install(svc).ok_or_else(|| {
            format!(
                "{}'s runtime is not installed. Install it first.",
                svc.display_name
            )
        })?;
        if let Some(blocker) = self.start_blocker(svc) {
            return Err(format!("{} {blocker}.", svc.display_name));
        }
        let exe = install.dir.join(&native.executable);
        // The manifest keeps SA3 on Python until migration is validated. When
        // native is selected, preserve the public client port while the C++
        // server serves its unified API on the private native port.
        let adapter = if svc.id == "sa3" {
            let template = self.native_env_template(svc);
            let port = template
                .iter()
                .find(|(key, _)| key == "SA3_PORT")
                .ok_or("SA3's native server port is missing")?
                .1
                .parse::<u16>()
                .map_err(|_| "SA3's native server port is invalid")?;
            Some(crate::sa3_adapter::AdapterListener::bind(
                svc.port,
                port,
                self.repo_root.join("sa3/native-inputs"),
                crate::sa3_adapter::client_defaults(&crate::sa3_loudness_env()),
                self.repo_root.clone(),
                template.iter().rev().find(|(key, _)| key == "SA3_DEFAULT_LORA").map(|(_, value)| value.clone()),
            )?)
        } else {
            None
        };

        let work_dir = self.service_dir(svc);
        std::fs::create_dir_all(&work_dir)
            .map_err(|e| format!("Cannot create {}: {}", work_dir.display(), e))?;
        let log_path = work_dir.join(format!("{}.log", svc.id));

        self.errors.remove(&svc.id);

        let log_file = std::fs::File::create(&log_path)
            .map_err(|e| format!("Cannot create log file: {}", e))?;
        let log_file_err = log_file
            .try_clone()
            .map_err(|e| format!("Cannot clone log handle: {}", e))?;

        let mut cmd = Command::new(&exe);
        let args = if svc.id == "sa3" {
            crate::sa3_runtime::launch_args(&native.args, self.sa3_selection.as_ref().map_or("F16", |selection| selection.encoding.as_str()))?
        } else { native.args.clone() };
        cmd.args(&args)
            .current_dir(&install.dir)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_file_err));
        self.apply_runtime_env(&mut cmd);
        for (key, value) in
            native_runtime::launch_env(&self.native_env_template(svc), &install.backend)
        {
            cmd.env(key, value);
        }
        if let Some(path) = native_runtime::path_with_runtimes(&self.repo_root, &install.runtimes) {
            cmd.env("PATH", path);
        }

        if svc.id == "yuey" {
            if let Some(encoding) = crate::yuey_launch_encoding(&self.models_dir(), &install.dir) {
                cmd.env("YUE2_ENCODING", encoding);
            }
            for (key, value) in crate::yuey_generation_env() {
                cmd.env(key, value);
            }
        }

        crate::workload_job::configure_std_command(&mut cmd);

        let mut child = cmd
            .spawn()
            .map_err(|e| native_runtime::spawn_error_message(&svc.display_name, &e))?;
        if let Err(error) = crate::workload_job::enroll_std_child(&child) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "Failed to enroll {} in the managed workload group: {}",
                svc.id, error
            ));
        }

        log::info!(
            "Started {} (PID {}) from {} on {}",
            svc.id,
            child.id(),
            install.dir.display(),
            install.backend
        );

        self.running.insert(
            svc.id.clone(),
            RunningService {
                process: child,
                healthy: false,
                started_at: Instant::now(),
                last_health_check_at: None,
                _adapter: adapter.map(crate::sa3_adapter::AdapterListener::spawn),
            },
        );

        Ok(())
    }

    pub fn stop(&mut self, service_id: &str) -> Result<(), String> {
        if let Some(mut running) = self.running.remove(service_id) {
            log::info!("Stopping {}", service_id);
            // Use taskkill /T to kill the entire process tree on Windows.
            // This ensures subprocesses (e.g. carey_wrapper -> api_server.py) are also killed.
            let pid = running.process.id();
            let _ = std::process::Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid.to_string()])
                .creation_flags(0x08000000) // CREATE_NO_WINDOW
                .output();
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

    pub fn is_building(&self, service_id: &str) -> bool {
        self.build_statuses
            .get(service_id)
            .is_some_and(|status| status.building)
    }

    pub fn any_building(&self) -> bool {
        self.build_statuses.values().any(|status| status.building)
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

        if svc.runtime == ServiceRuntime::Native {
            if let Some(reason) = self.native_mutation_blocker(service_id) {
                return Err(format!(
                    "Cannot install {}'s runtime: {reason}.",
                    svc.display_name
                ));
            }
            return Ok(self.build_info_for(svc));
        }

        if svc.build_steps.is_empty() {
            return Err(format!("No build steps defined for {}", service_id));
        }

        if self
            .build_statuses
            .get(service_id)
            .map_or(false, |b| b.building)
        {
            return Err(format!("{} is already building", service_id));
        }

        Ok(self.build_info_for(svc))
    }

    /// Install the native candidate alongside a legacy Python environment,
    /// without switching the service or deleting any Python files.
    pub fn get_native_build_info(&self, service_id: &str) -> Result<BuildInfo, String> {
        let svc = self
            .find_service(service_id)
            .ok_or_else(|| format!("Unknown service: {service_id}"))?;
        if svc.native.is_none() {
            return Err(format!("{service_id} has no native definition"));
        }
        if let Some(reason) = self.native_mutation_blocker(service_id) {
            return Err(format!(
                "Cannot install {}'s runtime: {reason}.",
                svc.display_name
            ));
        }
        let mut info = self.build_info_for(svc);
        info.runtime = ServiceRuntime::Native;
        info.env_dir = self.native_dir(svc);
        info.native_runtimes = self.native_runtimes.clone();
        info.service_env = self.native_env_template(svc);
        Ok(info)
    }

    fn build_info_for(&self, svc: &ServiceDef) -> BuildInfo {
        let native = svc.runtime == ServiceRuntime::Native;
        BuildInfo {
            service_id: svc.id.clone(),
            runtime_root: self.repo_root.clone(),
            work_dir: self.service_dir(svc),
            env_dir: self.env_dir(svc),
            build_steps: svc.build_steps.clone(),
            runtime: svc.runtime,
            native: svc.native.clone(),
            native_runtimes: if native {
                self.native_runtimes.clone()
            } else {
                HashMap::new()
            },
            service_env: if native {
                self.native_env_template(svc)
            } else {
                Vec::new()
            },
        }
    }

    /// Mark a build as started
    pub fn set_build_started(&mut self, service_id: &str, total_steps: usize) {
        self.build_statuses.insert(
            service_id.to_string(),
            BuildStatus {
                building: true,
                current_step: 0,
                total_steps,
                step_label: "Creating virtual environment...".to_string(),
                log: String::new(),
                error: None,
            },
        );
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
        self.services
            .iter()
            .filter_map(|svc| {
                match svc.runtime {
                    ServiceRuntime::Python if svc.build_steps.is_empty() => return None,
                    // A running native service cannot have its runtime swapped.
                    ServiceRuntime::Native if self.running.contains_key(&svc.id) => return None,
                    _ => {}
                }
                if self
                    .build_statuses
                    .get(&svc.id)
                    .map_or(false, |b| b.building)
                {
                    return None;
                }
                Some(self.build_info_for(svc))
            })
            .collect()
    }

    pub fn get_log(&self, service_id: &str) -> Result<String, String> {
        let svc = self
            .find_service(service_id)
            .ok_or_else(|| format!("Unknown service: {}", service_id))?;

        let is_running = self.running.contains_key(service_id);
        let is_building = self
            .build_statuses
            .get(service_id)
            .map_or(false, |b| b.building);
        let has_failed = self.errors.contains_key(service_id);

        // Priority:
        // 1. If actively building -> show build log
        // 2. If running or failed -> show runtime log (crash output is critical)
        // 3. If stopped with build log -> show build log
        // 4. Otherwise -> show runtime log file if it exists

        if is_building {
            let log = self
                .build_statuses
                .get(service_id)
                .map(|b| b.log.clone())
                .unwrap_or_default();
            return Ok(log);
        }

        let log_path = self.service_dir(svc).join(format!("{}.log", svc.id));

        // If running or just crashed, the runtime log is what the user needs
        if (is_running || has_failed) && log_path.exists() {
            let runtime_log = self.read_log_file(&log_path)?;
            if !runtime_log.is_empty() {
                return Ok(runtime_log);
            }
        }

        // Service is stopped (not failed) — prefer build log over stale runtime log
        if let Some(build_status) = self.build_statuses.get(service_id) {
            if !build_status.log.is_empty() {
                return Ok(build_status.log.clone());
            }
        }

        // Last resort: show runtime log even if stopped (e.g. no build has happened this session)
        if log_path.exists() {
            return self.read_log_file(&log_path);
        }

        Ok(String::new())
    }

    fn read_log_file(&self, log_path: &std::path::Path) -> Result<String, String> {
        let metadata =
            std::fs::metadata(log_path).map_err(|e| format!("Cannot read log: {}", e))?;
        let file_size = metadata.len();
        let max_read: u64 = 256 * 1024;

        if file_size > max_read {
            use std::io::{Read, Seek, SeekFrom};
            let mut file =
                std::fs::File::open(log_path).map_err(|e| format!("Cannot open log: {}", e))?;
            file.seek(SeekFrom::End(-(max_read as i64)))
                .map_err(|e| format!("Seek error: {}", e))?;
            let mut buf = String::new();
            file.read_to_string(&mut buf)
                .map_err(|e| format!("Read error: {}", e))?;
            let trimmed = if let Some(pos) = buf.find('\n') {
                buf[pos + 1..].to_string()
            } else {
                buf
            };
            Ok(filter_successful_health_access_logs(&trimmed))
        } else {
            let raw =
                std::fs::read_to_string(log_path).map_err(|e| format!("Cannot read log: {}", e))?;
            Ok(filter_successful_health_access_logs(&raw))
        }
    }
}

pub struct BuildInfo {
    pub service_id: String,
    pub runtime_root: PathBuf,
    pub work_dir: PathBuf,
    pub env_dir: PathBuf,
    pub build_steps: Vec<String>,
    pub runtime: ServiceRuntime,
    pub native: Option<NativeDef>,
    pub native_runtimes: HashMap<String, NativePackage>,
    /// A native service's env, resolved except for `${NATIVE_BACKEND}`, so
    /// the install check runs the executable exactly as the service will.
    pub service_env: Vec<(String, String)>,
}

impl BuildInfo {
    /// Python builds add uv bootstrap and venv creation to the manifest steps.
    pub fn total_steps(&self) -> usize {
        match self.runtime {
            ServiceRuntime::Python => self.build_steps.len() + 2,
            ServiceRuntime::Native => native_runtime::INSTALL_STEPS,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_profile_selection_survives_restart_and_corruption_never_falls_back_to_python() {
        let root = std::env::temp_dir().join(format!(
            "gary-sa3-manager-selection-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut manifest: crate::manifest::Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json")).unwrap();
        manifest.resolve_native_bundles().unwrap();
        let defs = manifest.services;
        let original = root.join("services/sa3/env/Scripts/python.exe");
        std::fs::create_dir_all(original.parent().unwrap()).unwrap();
        std::fs::write(&original, b"preserved").unwrap();
        let mut manager = ServiceManager::new(defs.clone(), root.clone());
        assert!(!manager.is_native("sa3"));
        manager
            .activate_native_sa3(crate::sa3_runtime::Selection {
                schema_version: 1,
                encoding: "Q4_K_M".into(),
                verified_release: "0.1.2".into(),
                backend: "cuda".into(),
                activated_at: 1,
                cleanup_complete: false,
                cleanup_errors: vec!["locked old environment".into()],
            })
            .unwrap();
        assert!(manager.is_native("sa3"));
        assert!(!manager.is_native("stable-audio"));
        let restarted = ServiceManager::new(defs.clone(), root.clone());
        assert!(restarted.is_native("sa3"));
        assert!(restarted
            .get_service_info()
            .iter()
            .find(|service| service.id == "sa3")
            .unwrap()
            .start_blocker
            .as_deref()
            .unwrap()
            .contains("native models"));
        let fresh = ServiceManager::new(defs.clone(), root.join("other-profile"));
        assert!(!fresh.is_native("sa3"));
        std::fs::write(crate::sa3_runtime::selection_path(&root), "{}").unwrap();
        let mut corrupted = ServiceManager::new(defs, root.clone());
        assert!(corrupted.is_native("sa3"));
        assert!(corrupted
            .start("sa3")
            .unwrap_err()
            .contains("native selection"));
        assert_eq!(std::fs::read(&original).unwrap(), b"preserved");
        crate::remove_managed_path(&root, &std::env::temp_dir().canonicalize().unwrap()).unwrap();
    }

    fn shared_native_manager() -> ServiceManager {
        let mut manifest: crate::manifest::Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .unwrap();
        manifest.resolve_native_bundles().unwrap();
        let native = manifest
            .services
            .iter()
            .find(|svc| svc.id == "sa3")
            .unwrap()
            .native
            .clone()
            .unwrap();
        for svc in &mut manifest.services {
            if ["sa3", "stable-audio", "foundation"].contains(&svc.id.as_str()) {
                svc.native = Some(native.clone());
                svc.runtime = ServiceRuntime::Native;
            }
        }
        let mut manager = ServiceManager::new(manifest.services, PathBuf::from("C:/test-runtime"));
        manager.set_native_runtimes(manifest.native_runtimes);
        manager
    }

    #[test]
    fn stable_audio_services_share_a_bundle_but_yuey_keeps_its_own_ggml() {
        let manager = shared_native_manager();
        assert_eq!(
            manager.native_dir_for("sa3"),
            manager.native_dir_for("foundation")
        );
        assert_eq!(
            manager.native_dir_for("sa3"),
            manager.native_dir_for("stable-audio")
        );
        assert_ne!(
            manager.native_dir_for("sa3"),
            manager.native_dir_for("yuey")
        );
        assert_eq!(manager.env_dir_for("sa3"), manager.native_dir_for("sa3"));
    }

    #[test]
    fn installing_one_consumer_blocks_other_installers_launches_and_tools() {
        let mut manager = shared_native_manager();
        manager.set_build_started("foundation", 5);
        assert!(manager.get_native_build_info("sa3").is_err());
        assert!(manager.get_build_info("stable-audio").is_err());
        assert!(manager.start("sa3").unwrap_err().contains("install"));
        assert!(manager
            .begin_native_workload("sa3", "trainer", "SA3 training")
            .is_err());
        assert!(manager.get_native_build_info("yuey").is_ok());
        manager.set_build_done("foundation", None);
        assert!(manager.get_native_build_info("sa3").is_ok());
    }

    #[test]
    fn native_tools_keep_the_bundle_reserved_until_their_last_job_exits() {
        let mut manager = shared_native_manager();
        manager
            .begin_native_workload("sa3", "trainer", "SA3 training")
            .unwrap();
        manager
            .begin_native_workload("foundation", "converter", "adapter conversion")
            .unwrap();
        assert!(manager.get_native_build_info("stable-audio").is_err());
        manager.end_native_workload("sa3", "trainer");
        assert!(manager.native_mutation_blocker("foundation").is_some());
        manager.end_native_workload("foundation", "converter");
        assert!(manager.native_mutation_blocker("sa3").is_none());
    }

    #[test]
    fn shared_cuda_replacement_waits_for_tools_in_another_bundle() {
        let mut manager = shared_native_manager();
        manager
            .begin_native_workload("sa3", "trainer", "SA3 training")
            .unwrap();
        // Independent bundle installation remains possible when CUDA is reused.
        assert!(manager.get_native_build_info("yuey").is_ok());
        let error = manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .unwrap_err();
        assert!(error.contains("SA3 training"), "{error}");
        manager.end_native_workload("sa3", "trainer");
        assert!(manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .is_ok());
    }

    #[test]
    fn shared_runtime_lease_blocks_new_users_and_expires_on_drop() {
        let mut manager = shared_native_manager();
        let lease = manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .unwrap();
        for service in ["sa3", "foundation", "yuey"] {
            assert!(manager.start(service).unwrap_err().contains("cudart-12.8"));
            assert!(manager
                .begin_native_workload(service, "tool", "native tool")
                .is_err());
            assert!(manager.get_native_build_info(service).is_err());
        }
        assert!(manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .is_err());
        // A native CUDA update doesn't reserve a legacy Python process.
        manager
            .services
            .iter_mut()
            .find(|svc| svc.id == "sa3")
            .unwrap()
            .runtime = ServiceRuntime::Python;
        assert!(!manager.start("sa3").unwrap_err().contains("cudart-12.8"));
        drop(lease);
        assert!(manager.get_native_build_info("yuey").is_ok());
        assert!(manager
            .begin_native_workload("sa3", "tool", "native tool")
            .is_ok());
        manager.end_native_workload("sa3", "tool");
        assert!(manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .is_ok());
    }

    #[test]
    fn shared_cuda_replacement_waits_for_running_native_consumers() {
        let mut manager = shared_native_manager();
        let mut process = Command::new("cmd.exe")
            .args(["/c", "exit /b 0"])
            .creation_flags(0x08000000)
            .spawn()
            .unwrap();
        process.wait().unwrap();
        // Keep a process entry until the regular health poll reaps it.
        manager.running.insert(
            "yuey".into(),
            RunningService {
                process,
                healthy: false,
                started_at: Instant::now(),
                last_health_check_at: None,
                _adapter: None,
            },
        );
        let error = manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .unwrap_err();
        assert!(error.contains("stop"), "{error}");
        // Python processes don't load the shared native CUDA directory.
        manager
            .services
            .iter_mut()
            .find(|svc| svc.id == "yuey")
            .unwrap()
            .runtime = ServiceRuntime::Python;
        assert!(manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .is_ok());
        manager.running.clear();
    }

    #[test]
    fn installed_vulkan_consumer_does_not_reserve_cuda() {
        let mut manager = shared_native_manager();
        let root = std::env::temp_dir().join(format!(
            "gary-shared-vulkan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        manager.repo_root = root.clone();
        let svc = manager.find_service("yuey").unwrap();
        let native_dir = manager.native_dir(svc);
        std::fs::create_dir_all(&native_dir).unwrap();
        std::fs::write(
            native_dir.join(&svc.native.as_ref().unwrap().executable),
            b"fixture",
        )
        .unwrap();
        let stamp = native_runtime::NativeStamp {
            version: "fixture".into(),
            platform: native_runtime::current_platform().into(),
            backend: "vulkan".into(),
            requested_backend: "vulkan".into(),
            fallback_reason: None,
            runtimes: vec![],
        };
        std::fs::write(
            native_dir.join("gary-native.json"),
            serde_json::to_vec(&stamp).unwrap(),
        )
        .unwrap();
        let _lease = manager
            .begin_shared_runtime_replacement("cudart-12.8")
            .unwrap();
        assert!(manager.get_native_build_info("yuey").is_ok());
        assert!(manager
            .begin_native_workload("yuey", "tool", "Vulkan tool")
            .is_ok());
        assert!(manager.get_native_build_info("sa3").is_err());
        crate::remove_managed_path(&root, &std::env::temp_dir().canonicalize().unwrap()).unwrap();
    }

    #[test]
    fn unrelated_shared_runtime_does_not_block_cuda_consumers() {
        let mut manager = shared_native_manager();
        let _lease = manager
            .begin_shared_runtime_replacement("other-runtime")
            .unwrap();
        assert!(manager.get_native_build_info("yuey").is_ok());
        assert!(manager
            .begin_native_workload("sa3", "tool", "native tool")
            .is_ok());
    }

    #[test]
    fn adapter_registry_workloads_block_start_until_their_files_are_committed() {
        let mut manager = shared_native_manager();
        for job in [
            "legacy-lora-edit",
            "lora-import",
            "lora-edit",
            "checkpoint-selection",
            "lora-preparation",
            "decoder-preparation",
        ] {
            manager
                .begin_native_workload("sa3", job, "SA3 adapter update")
                .unwrap();
            for runtime in [ServiceRuntime::Native, ServiceRuntime::Python] {
                manager
                    .services
                    .iter_mut()
                    .find(|svc| svc.id == "sa3")
                    .unwrap()
                    .runtime = runtime;
                assert!(manager.start("sa3").unwrap_err().contains("adapter update"));
            }
            manager.end_native_workload("sa3", job);
            assert!(!manager.start("sa3").unwrap_err().contains("adapter update"));
        }
    }

    #[test]
    fn native_training_blocks_python_and_native_sa3_launches() {
        let mut manager = shared_native_manager();
        manager.begin_native_workload("sa3","native-training","SA3 native training").unwrap();
        assert!(manager.start("sa3").unwrap_err().contains("native training"));
        manager.services.iter_mut().find(|svc| svc.id == "sa3").unwrap().runtime=ServiceRuntime::Python;
        assert!(manager.start("sa3").unwrap_err().contains("native training"));
        manager.end_native_workload("sa3","native-training");
        assert!(!manager.start("sa3").unwrap_err().contains("native training"));
    }

    #[test]
    fn model_preparation_blocks_launches_tools_and_removal_until_released() {
        let mut manager = shared_native_manager();
        manager
            .begin_native_model_mutation("sa3", "SA3 model preparation")
            .unwrap();
        assert!(manager
            .start("sa3")
            .unwrap_err()
            .contains("model preparation"));
        assert!(manager
            .begin_native_workload("sa3", "trainer", "training")
            .is_err());
        assert!(manager
            .begin_native_model_mutation("sa3", "second download")
            .is_err());
        assert!(manager
            .native_mutation_blocker("sa3")
            .unwrap()
            .contains("model preparation"));
        assert!(manager.native_mutation_blocker("yuey").is_none());
        manager.end_native_model_mutation("sa3");
        assert!(manager.native_mutation_blocker("sa3").is_none());
        manager
            .begin_native_workload("sa3", "trainer", "training")
            .unwrap();
        assert!(manager
            .begin_native_model_mutation("sa3", "SA3 models")
            .is_err());
    }

    #[test]
    fn preparing_native_models_leaves_the_python_start_available() {
        let mut manager = shared_native_manager();
        manager
            .services
            .iter_mut()
            .find(|svc| svc.id == "sa3")
            .unwrap()
            .runtime = ServiceRuntime::Python;
        manager
            .begin_native_model_mutation("sa3", "SA3 model preparation")
            .unwrap();
        assert!(manager
            .get_service_info()
            .iter()
            .find(|svc| svc.id == "sa3")
            .unwrap()
            .start_blocker
            .is_none());
    }

    #[test]
    fn preparing_native_does_not_replace_the_python_environment_path() {
        let mut manager = shared_native_manager();
        manager
            .services
            .iter_mut()
            .find(|svc| svc.id == "sa3")
            .unwrap()
            .runtime = ServiceRuntime::Python;
        let python = manager.env_dir_for("sa3").unwrap();
        let candidate = manager.get_native_build_info("sa3").unwrap();
        assert_eq!(python, PathBuf::from("C:/test-runtime/services/sa3/env"));
        assert_eq!(
            candidate.env_dir,
            PathBuf::from("C:/test-runtime/services/sa3/native")
        );
        assert_eq!(candidate.runtime, ServiceRuntime::Native);
        assert_eq!(manager.env_dir_for("sa3").unwrap(), python);
        assert!(candidate
            .service_env
            .iter()
            .any(|(key, value)| key == "SA3_PORT" && value == "18006"));
    }

    fn health(interval_seconds: u64, startup_grace_seconds: u64) -> HealthCheck {
        HealthCheck {
            endpoint: "/health".to_string(),
            interval_seconds,
            timeout_seconds: 5,
            startup_grace_seconds,
        }
    }

    #[test]
    fn unhealthy_services_poll_quickly_during_startup_grace() {
        let health = health(15, 120);

        assert_eq!(
            health_check_interval(&health, false, Duration::from_secs(20)),
            Duration::from_secs(1)
        );
    }

    #[test]
    fn healthy_services_use_manifest_interval() {
        let health = health(15, 120);

        assert_eq!(
            health_check_interval(&health, true, Duration::from_secs(20)),
            Duration::from_secs(15)
        );
    }

    #[test]
    fn unhealthy_services_after_grace_still_retry_without_long_delays() {
        let health = health(15, 120);

        assert_eq!(
            health_check_interval(&health, false, Duration::from_secs(121)),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn successful_health_access_logs_are_filtered() {
        let raw = concat!(
            "loading model\n",
            "127.0.0.1 - - [01/Jul/2026 12:00:00] \"GET /health HTTP/1.1\" 200 -\n",
            "INFO:     127.0.0.1:54000 - \"GET /health HTTP/1.1\" 200 OK\n",
            "model ready\n",
        );

        assert_eq!(
            filter_successful_health_access_logs(raw),
            "loading model\nmodel ready\n"
        );
    }

    #[test]
    fn failed_health_access_logs_are_kept() {
        let raw = "127.0.0.1 - - [01/Jul/2026] \"GET /health HTTP/1.1\" 503 -\n";

        assert_eq!(filter_successful_health_access_logs(raw), raw);
    }
}
