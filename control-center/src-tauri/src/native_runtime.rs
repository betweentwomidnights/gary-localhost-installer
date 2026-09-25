//! Native services: prebuilt GGML servers that gary4local downloads for the
//! machine's GPU instead of building a Python environment.
//!
//! A service's package is a core zip (the executable, ggml, and every CPU
//! variant) plus at most one GPU backend zip unpacked beside it. ggml loads
//! whichever backend DLL sits next to the executable, so an NVIDIA machine and
//! an AMD machine run the same binary with a different DLL. Shared runtimes
//! such as the CUDA redistributables are installed once, outside any service,
//! and put on PATH for the services that need them.

use crate::manifest::{NativeDef, NativePackage};
use crate::service_manager::{BuildInfo, ServiceManager};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::Emitter;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

pub const NATIVE_DIR: &str = "native";
const STAMP_FILE: &str = "gary-native.json";
const PROPS_FILE: &str = "props.json";
const RUNTIMES_DIR: &str = "native-runtimes";
const RUNTIME_STAMP_FILE: &str = "gary-runtime.json";
/// Detect the GPU, fetch the core, fetch the backend, unpack, check it runs.
pub const INSTALL_STEPS: usize = 5;

const NATIVE_APPLICATION_CONTROL_HELP: &str = "Windows got a little overprotective and blocked this service's runtime from starting.\n\nOpen Windows Security -> App & browser control -> Smart App Control settings, temporarily turn Smart App Control off, then install the runtime again. You can turn it back on afterward.";

pub fn current_platform() -> &'static str {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "windows-x64"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "macos-arm64"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "linux-x64"
    } else {
        "unsupported"
    }
}

// ---------------------------------------------------------------------------
// What is installed

/// Written beside the executable once a runtime is unpacked and checked.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NativeStamp {
    pub version: String,
    pub platform: String,
    pub backend: String,
    pub requested_backend: String,
    #[serde(default)]
    pub fallback_reason: Option<String>,
    #[serde(default)]
    pub runtimes: Vec<String>,
}

fn read_stamp(native_dir: &Path) -> Option<NativeStamp> {
    let raw = std::fs::read_to_string(native_dir.join(STAMP_FILE)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// A runnable native service: where it is and how to launch it.
#[derive(Debug, Clone)]
pub struct NativeInstall {
    pub dir: PathBuf,
    /// The value `${NATIVE_BACKEND}` resolves to in the service's env.
    pub backend: String,
    pub runtimes: Vec<String>,
    /// None for a developer build named by `GARY4LOCAL_NATIVE_DIR_<ID>`.
    pub version: Option<String>,
    pub fallback_reason: Option<String>,
}

fn env_suffix(service_id: &str) -> String {
    service_id.to_ascii_uppercase().replace('-', "_")
}

/// A developer can point a service at a local build instead of a downloaded
/// package, e.g. `GARY4LOCAL_NATIVE_DIR_YUEY=C:\dev\yue2.cpp\build-cuda\bin\Release`.
/// `GARY4LOCAL_NATIVE_BACKEND_YUEY` then picks the device; it defaults to auto.
fn dev_override(service_id: &str, executable: &str) -> Option<NativeInstall> {
    let suffix = env_suffix(service_id);
    let dir = std::env::var_os(format!("GARY4LOCAL_NATIVE_DIR_{suffix}")).map(PathBuf::from)?;
    if !dir.join(executable).is_file() {
        return None;
    }
    let backend = std::env::var(format!("GARY4LOCAL_NATIVE_BACKEND_{suffix}"))
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "auto".to_string());
    Some(NativeInstall {
        dir,
        backend,
        runtimes: Vec::new(),
        version: None,
        fallback_reason: None,
    })
}

pub fn installed(service_id: &str, native_dir: &Path, executable: &str) -> Option<NativeInstall> {
    if let Some(install) = dev_override(service_id, executable) {
        return Some(install);
    }
    let stamp = read_stamp(native_dir)?;
    if !native_dir.join(executable).is_file() {
        return None;
    }
    Some(NativeInstall {
        dir: native_dir.to_path_buf(),
        backend: stamp.backend,
        runtimes: stamp.runtimes,
        version: Some(stamp.version),
        fallback_reason: stamp.fallback_reason,
    })
}

pub fn runtime_dir(runtime_root: &Path, name: &str) -> PathBuf {
    runtime_root.join(RUNTIMES_DIR).join(name)
}

/// The service's env with `${NATIVE_BACKEND}` filled in. Anything still
/// holding a template is dropped, as for Python services.
pub fn launch_env(template: &[(String, String)], backend: &str) -> Vec<(String, String)> {
    template
        .iter()
        .map(|(key, value)| (key.clone(), value.replace("${NATIVE_BACKEND}", backend)))
        .filter(|(_, value)| !value.is_empty() && !value.contains("${"))
        .collect()
}

/// PATH with the named shared runtimes in front, so a backend DLL's imports
/// (cudart, cuBLAS) resolve from them. None when nothing needs adding.
pub fn path_with_runtimes(runtime_root: &Path, runtimes: &[String]) -> Option<OsString> {
    if runtimes.is_empty() {
        return None;
    }
    let mut dirs: Vec<PathBuf> = runtimes
        .iter()
        .map(|name| runtime_dir(runtime_root, name))
        .collect();
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    std::env::join_paths(dirs).ok()
}

pub fn spawn_error_message(service: &str, error: &std::io::Error) -> String {
    let message = error.to_string();
    if crate::is_windows_application_control_block(&message) {
        NATIVE_APPLICATION_CONTROL_HELP.to_string()
    } else {
        format!("Failed to start {service}: {message}")
    }
}

// ---------------------------------------------------------------------------
// Choosing a backend

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuAdapter {
    pub name: String,
    pub vendor: GpuVendor,
}

fn vendor_of(pnp_device_id: &str, name: &str) -> GpuVendor {
    let pnp = pnp_device_id.to_ascii_uppercase();
    if let Some(index) = pnp.find("VEN_") {
        match pnp.get(index + 4..index + 8) {
            Some("10DE") => return GpuVendor::Nvidia,
            Some("1002") | Some("1022") => return GpuVendor::Amd,
            Some("8086") => return GpuVendor::Intel,
            _ => {}
        }
    }
    let name = name.to_ascii_lowercase();
    if name.contains("nvidia") || name.contains("geforce") {
        GpuVendor::Nvidia
    } else if name.contains("radeon") || name.contains("amd ") {
        GpuVendor::Amd
    } else if name.contains("intel") {
        GpuVendor::Intel
    } else {
        GpuVendor::Other
    }
}

/// Parse `Get-CimInstance Win32_VideoController | ConvertTo-Json`, which is a
/// bare object for one adapter and an array for several.
fn parse_video_controllers(raw: &str) -> Result<Vec<GpuAdapter>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let value: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|error| format!("could not read the GPU list ({error})"))?;
    let items = match value {
        serde_json::Value::Array(items) => items,
        other => vec![other],
    };
    Ok(items
        .iter()
        .map(|item| {
            let name = item
                .get("Name")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown display adapter")
                .trim()
                .to_string();
            let pnp = item
                .get("PNPDeviceID")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            GpuAdapter {
                vendor: vendor_of(pnp, &name),
                name,
            }
        })
        .collect())
}

#[cfg(target_os = "windows")]
pub async fn detect_gpus() -> Result<Vec<GpuAdapter>, String> {
    let mut cmd = tokio::process::Command::new("powershell");
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance -ClassName Win32_VideoController | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress",
    ]);
    crate::hide_console_window(&mut cmd);
    let output = tokio::time::timeout(Duration::from_secs(30), cmd.output())
        .await
        .map_err(|_| "the GPU query did not answer within 30 seconds".to_string())?
        .map_err(|error| format!("the GPU query could not run: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "the GPU query failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    parse_video_controllers(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(target_os = "macos")]
pub async fn detect_gpus() -> Result<Vec<GpuAdapter>, String> {
    Ok(vec![GpuAdapter {
        name: "Apple GPU".to_string(),
        vendor: GpuVendor::Apple,
    }])
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
pub async fn detect_gpus() -> Result<Vec<GpuAdapter>, String> {
    Ok(Vec::new())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendChoice {
    pub backend: String,
    pub reason: String,
}

/// Used when a service's manifest names no preference.
const DEFAULT_BACKEND_ORDER: &[&str] = &["cuda", "metal", "vulkan"];

/// The backends to try, best first. A backend chosen in settings is the only
/// candidate, so if it fails the install fails loudly. "auto" lists every
/// published backend this machine's GPUs can run, in the service's `prefer`
/// order, so a backend that does not come up falls through to the next.
///
/// There is no CPU runtime. The native models take minutes per song on a GPU,
/// so on a CPU they would only look broken; a machine without a supported GPU
/// is told so instead.
pub fn resolve_backends(
    preference: &str,
    adapters: &[GpuAdapter],
    offered: &[String],
    prefer: &[String],
) -> Result<Vec<BackendChoice>, String> {
    let offers = |backend: &str| offered.iter().any(|name| name == backend);
    let preference = preference.trim().to_ascii_lowercase();
    if !preference.is_empty() && preference != "auto" {
        if offers(&preference) {
            return Ok(vec![BackendChoice {
                backend: preference,
                reason: "chosen in settings".to_string(),
            }]);
        }
        let mut available = offered.to_vec();
        available.sort();
        return Err(format!(
            "the {preference} backend is not published for {} (available: {})",
            current_platform(),
            available.join(", ")
        ));
    }

    let first = |vendor: GpuVendor| adapters.iter().find(|adapter| adapter.vendor == vendor);
    // The GPU each backend would run on here, if any.
    let runs_on = |backend: &str| match backend {
        "cuda" => first(GpuVendor::Nvidia),
        "metal" => first(GpuVendor::Apple),
        "vulkan" => first(GpuVendor::Amd)
            .or_else(|| first(GpuVendor::Nvidia))
            .or_else(|| first(GpuVendor::Intel)),
        _ => None,
    };
    let order: Vec<&str> = if prefer.is_empty() {
        DEFAULT_BACKEND_ORDER.to_vec()
    } else {
        prefer.iter().map(String::as_str).collect()
    };

    let mut choices: Vec<BackendChoice> = Vec::new();
    for &backend in &order {
        if !offers(backend) || choices.iter().any(|choice| choice.backend == backend) {
            continue;
        }
        if let Some(gpu) = runs_on(backend) {
            choices.push(BackendChoice {
                backend: backend.to_string(),
                reason: format!("GPU: {}", gpu.name),
            });
        }
    }
    if choices.is_empty() && adapters.is_empty() {
        // Listing GPUs failed, which says nothing about whether one is here.
        // Try each backend and let the runtime check decide.
        for &backend in &order {
            if offers(backend) && !choices.iter().any(|choice| choice.backend == backend) {
                choices.push(BackendChoice {
                    backend: backend.to_string(),
                    reason: "no GPU was listed, so trying each backend".to_string(),
                });
            }
        }
    }
    if choices.is_empty() {
        let found: Vec<&str> = adapters.iter().map(|adapter| adapter.name.as_str()).collect();
        return Err(format!(
            "this needs an NVIDIA, AMD, or Intel GPU, and none was found (found: {})",
            found.join(", ")
        ));
    }
    Ok(choices)
}

// ---------------------------------------------------------------------------
// Downloading and verifying

pub fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit())
}

/// `sha256sum` output: `<hash>  <name>` per line, name optionally `*`-prefixed.
pub fn parse_sha256sums(raw: &str) -> HashMap<String, String> {
    raw.lines()
        .filter_map(|line| {
            let (hash, name) = line.trim().split_once(char::is_whitespace)?;
            let name = name.trim().trim_start_matches('*');
            (is_sha256(hash) && !name.is_empty())
                .then(|| (name.to_string(), hash.to_ascii_lowercase()))
        })
        .collect()
}

pub async fn sha256_file(path: &Path) -> Result<String, String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await
    .map_err(|error| format!("checksum task failed: {error}"))?
}

fn partial_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    dest.with_file_name(format!("{name}.partial"))
}

/// Download `url` to `dest`, resuming a `.partial` left by an interrupted
/// attempt, and keep it only if its SHA-256 is the expected one. A file
/// already at `dest` with the right hash is kept without a request.
pub async fn download_verified(
    client: &reqwest::Client,
    url: &str,
    headers: reqwest::header::HeaderMap,
    expected_sha256: &str,
    dest: &Path,
    on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<(), String> {
    let expected = expected_sha256.to_ascii_lowercase();
    let name = dest
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| url.to_string());

    if dest.is_file() && sha256_file(dest).await? == expected {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }

    let partial = partial_path(dest);
    let mut resume_from = tokio::fs::metadata(&partial)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let mut request = client.get(url).headers(headers);
    if resume_from > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
    }
    let mut response = request
        .send()
        .await
        .map_err(|error| format!("downloading {name} failed: {error}"))?;
    let status = response.status();

    // 416 means the partial file already holds every byte; it is checked below.
    if status != reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        if !status.is_success() {
            return Err(format!("downloading {name} failed: HTTP {status}"));
        }
        let resumed = resume_from > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
        if !resumed {
            resume_from = 0;
        }
        let total = response.content_length().map(|length| length + resume_from);
        let mut options = tokio::fs::OpenOptions::new();
        options.create(true).write(true);
        if resumed {
            options.append(true);
        } else {
            options.truncate(true);
        }
        let mut file = options
            .open(&partial)
            .await
            .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;

        let mut received = resume_from;
        on_progress(received, total);
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("downloading {name} failed: {error}"))?
        {
            file.write_all(&chunk)
                .await
                .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;
            received += chunk.len() as u64;
            on_progress(received, total);
        }
        file.flush()
            .await
            .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;
    }

    let actual = sha256_file(&partial).await?;
    if actual != expected {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(format!(
            "{name} did not match its published checksum (expected {expected}, got {actual}); the download was discarded"
        ));
    }
    if dest.exists() {
        let _ = tokio::fs::remove_file(dest).await;
    }
    tokio::fs::rename(&partial, dest)
        .await
        .map_err(|error| format!("cannot move {name} into place: {error}"))
}

/// Unpack every entry of `archive` under `into`, refusing any entry whose path
/// would land outside it.
pub fn extract_zip(archive: &Path, into: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive)
        .map_err(|error| format!("cannot open {}: {error}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|error| format!("{} is not a readable zip: {error}", archive.display()))?;
    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .map_err(|error| format!("cannot read {}: {error}", archive.display()))?;
        let relative = entry.enclosed_name().ok_or_else(|| {
            format!(
                "{} contains an unsafe path: {}",
                archive.display(),
                entry.name()
            )
        })?;
        let target = into.join(relative);
        if entry.is_dir() {
            std::fs::create_dir_all(&target)
                .map_err(|error| format!("cannot create {}: {error}", target.display()))?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        let mut out = std::fs::File::create(&target)
            .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
    }
    Ok(())
}

/// Move `staging` to `target`, keeping the old `target` until the new one is
/// in place so a failed swap leaves a working install behind.
fn replace_dir(staging: &Path, target: &Path) -> Result<(), String> {
    let name = target
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    let retired = target.with_file_name(format!("{name}.old"));
    if retired.exists() {
        std::fs::remove_dir_all(&retired)
            .map_err(|error| format!("cannot clear {}: {error}", retired.display()))?;
    }
    if target.exists() {
        std::fs::rename(target, &retired).map_err(|error| {
            format!(
                "cannot replace {}: {error}. Stop anything using it and try again.",
                target.display()
            )
        })?;
    }
    if let Err(error) = std::fs::rename(staging, target) {
        let _ = std::fs::rename(&retired, target);
        return Err(format!("cannot move the new runtime into place: {error}"));
    }
    let _ = std::fs::remove_dir_all(&retired);
    Ok(())
}

// ---------------------------------------------------------------------------
// Checking what came up

/// Run the executable's one-shot `--props` probe, which reports the GGML
/// devices that actually initialised without starting the service.
async fn probe(
    exe: &Path,
    env: &[(String, String)],
    path: Option<OsString>,
) -> Result<serde_json::Value, String> {
    let mut cmd = tokio::process::Command::new(exe);
    cmd.arg("--props")
        .envs(
            env.iter()
                .map(|(key, value)| (key.as_str(), value.as_str())),
        )
        .kill_on_drop(true);
    if let Some(dir) = exe.parent() {
        cmd.current_dir(dir);
    }
    if let Some(path) = path {
        cmd.env("PATH", path);
    }
    crate::hide_console_window(&mut cmd);
    let output = tokio::time::timeout(Duration::from_secs(120), cmd.output())
        .await
        .map_err(|_| "the runtime check did not finish within two minutes".to_string())?
        .map_err(|error| spawn_error_message("the runtime check", &error))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!(
            "the runtime check exited with {}: {}",
            output.status,
            tail(&stderr, 1500)
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        format!(
            "the runtime check printed something other than JSON ({error}): {}",
            tail(&String::from_utf8_lossy(&output.stdout), 500)
        )
    })
}

fn tail(text: &str, max: usize) -> &str {
    let text = text.trim();
    if text.len() <= max {
        return text;
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn devices(props: &serde_json::Value) -> Vec<&serde_json::Value> {
    props
        .get("devices")
        .and_then(|value| value.as_array())
        .map(|devices| devices.iter().collect())
        .unwrap_or_default()
}

/// ggml names its registries CUDA, Vulkan, Metal, and CPU.
fn props_has_backend(props: &serde_json::Value, backend: &str) -> bool {
    devices(props).iter().any(|device| {
        device
            .get("backend")
            .and_then(|value| value.as_str())
            .is_some_and(|name| name.to_ascii_lowercase().contains(backend))
    })
}

fn describe_devices(props: &serde_json::Value) -> Vec<String> {
    devices(props)
        .iter()
        .map(|device| {
            let text = |key: &str| {
                device
                    .get(key)
                    .and_then(|value| value.as_str())
                    .unwrap_or("?")
            };
            let total = device
                .get("memory_total_bytes")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            format!(
                "  {} {} ({}, {:.1} GiB)",
                text("backend"),
                text("description"),
                text("type"),
                total as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// What the UI shows

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeDevice {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub memory_total_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeRuntimeInfo {
    pub service_id: String,
    pub installed: bool,
    pub dev_override: bool,
    pub version: Option<String>,
    pub manifest_version: String,
    pub backend: Option<String>,
    pub preference: String,
    pub fallback_reason: Option<String>,
    pub offered_backends: Vec<String>,
    pub devices: Vec<NativeDevice>,
    pub recommended_encoding: Option<String>,
}

pub fn runtime_info(
    service_id: &str,
    native_dir: &Path,
    def: &NativeDef,
    preference: String,
) -> NativeRuntimeInfo {
    let install = installed(service_id, native_dir, &def.executable);
    let props: Option<serde_json::Value> = install
        .as_ref()
        .and_then(|install| std::fs::read_to_string(install.dir.join(PROPS_FILE)).ok())
        .and_then(|raw| serde_json::from_str(&raw).ok());
    let mut offered_backends: Vec<String> = def
        .platforms
        .get(current_platform())
        .map(|platform| platform.backends.keys().cloned().collect())
        .unwrap_or_default();
    offered_backends.sort();

    let devices = props
        .as_ref()
        .map(|props| {
            devices(props)
                .iter()
                .map(|device| {
                    let text = |key: &str| {
                        device
                            .get(key)
                            .and_then(|value| value.as_str())
                            .unwrap_or("")
                            .to_string()
                    };
                    NativeDevice {
                        name: text("description"),
                        backend: text("backend"),
                        device_type: text("type"),
                        memory_total_bytes: device
                            .get("memory_total_bytes")
                            .and_then(|value| value.as_u64())
                            .unwrap_or(0),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let recommended_encoding = props.as_ref().and_then(|props| {
        props
            .pointer("/hardware/recommended_encoding")
            .and_then(|value| value.as_str())
            .map(str::to_string)
    });

    NativeRuntimeInfo {
        service_id: service_id.to_string(),
        installed: install.is_some(),
        dev_override: install
            .as_ref()
            .is_some_and(|install| install.version.is_none()),
        version: install.as_ref().and_then(|install| install.version.clone()),
        manifest_version: def.version.clone(),
        backend: install.as_ref().map(|install| install.backend.clone()),
        preference,
        fallback_reason: install.and_then(|install| install.fallback_reason),
        offered_backends,
        devices,
        recommended_encoding,
    }
}

/// The encoding the last runtime check recommended for this machine's memory.
pub fn recommended_encoding(native_dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(native_dir.join(PROPS_FILE)).ok()?;
    let props: serde_json::Value = serde_json::from_str(&raw).ok()?;
    props
        .pointer("/hardware/recommended_encoding")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Installing

/// Build-log and status plumbing shared with the Python build pipeline.
#[derive(Clone)]
struct Reporter {
    manager: Arc<Mutex<ServiceManager>>,
    handle: tauri::AppHandle,
    service_id: String,
}

impl Reporter {
    async fn emit(&self) {
        let info = self.manager.lock().await.get_service_info();
        let _ = self.handle.emit("services-updated", &info);
    }

    async fn step(&self, step: usize, label: &str) {
        {
            let mut mgr = self.manager.lock().await;
            mgr.set_build_step(&self.service_id, step, label);
            mgr.append_build_log(&self.service_id, &format!("\n--- {label} ---"));
        }
        self.emit().await;
    }

    async fn log(&self, line: &str) {
        self.manager
            .lock()
            .await
            .append_build_log(&self.service_id, line);
        self.emit().await;
    }

    /// A download callback that rewrites the step label at most a few times a
    /// second. It never waits for the lock: a skipped update costs nothing.
    fn progress(&self, step: usize, label: String) -> impl FnMut(u64, Option<u64>) + Send {
        let reporter = self.clone();
        let mut last = Instant::now() - Duration::from_secs(1);
        move |received, total| {
            if last.elapsed() < Duration::from_millis(250) && total != Some(received) {
                return;
            }
            last = Instant::now();
            let text = match total {
                Some(total) if total > 0 => format!(
                    "{label} {:.0}% ({:.0}/{:.0} MB)",
                    received as f64 * 100.0 / total as f64,
                    received as f64 / 1e6,
                    total as f64 / 1e6
                ),
                _ => format!("{label} ({:.0} MB)", received as f64 / 1e6),
            };
            if let Ok(mut mgr) = reporter.manager.try_lock() {
                mgr.set_build_step(&reporter.service_id, step, &text);
                let info = mgr.get_service_info();
                drop(mgr);
                let _ = reporter.handle.emit("services-updated", &info);
            }
        }
    }
}

/// Where a package comes from and the hash it must have.
struct PackageSource {
    file_name: String,
    url: String,
    sha256: String,
    /// Set when `GARY4LOCAL_NATIVE_PACKAGE_DIR` supplies the file.
    local: Option<PathBuf>,
}

/// `GARY4LOCAL_NATIVE_PACKAGE_DIR` names a folder of packages built by a
/// repo's `ci/package-windows.ps1`, so the install flow can be exercised
/// before a release exists. Hashes then come from that folder's SHA256SUMS.
fn package_source(package: &NativePackage) -> Result<PackageSource, String> {
    let file_name = package
        .url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("the manifest has no file name in {}", package.url))?
        .to_string();

    if let Some(dir) = std::env::var_os("GARY4LOCAL_NATIVE_PACKAGE_DIR").map(PathBuf::from) {
        let sums = std::fs::read_to_string(dir.join("SHA256SUMS"))
            .map(|raw| parse_sha256sums(&raw))
            .map_err(|error| format!("cannot read {}\\SHA256SUMS: {error}", dir.display()))?;
        let sha256 = sums
            .get(&file_name)
            .cloned()
            .ok_or_else(|| format!("{} has no entry for {file_name}", dir.display()))?;
        return Ok(PackageSource {
            local: Some(dir.join(&file_name)),
            url: package.url.clone(),
            file_name,
            sha256,
        });
    }

    if !is_sha256(&package.sha256) {
        return Err(format!(
            "{file_name} has no published checksum in this build's manifest, so it will not be installed. This build was made before that release was pinned."
        ));
    }
    Ok(PackageSource {
        url: package.url.clone(),
        sha256: package.sha256.to_ascii_lowercase(),
        file_name,
        local: None,
    })
}

async fn fetch(
    reporter: &Reporter,
    client: &reqwest::Client,
    source: &PackageSource,
    downloads: &Path,
    step: usize,
    label: &str,
    fetched: &mut Vec<PathBuf>,
) -> Result<PathBuf, String> {
    reporter.step(step, label).await;
    if let Some(local) = &source.local {
        reporter
            .log(&format!("using local package {}", local.display()))
            .await;
        let actual = sha256_file(local).await?;
        if actual != source.sha256 {
            return Err(format!(
                "{} does not match its SHA256SUMS entry",
                local.display()
            ));
        }
        return Ok(local.clone());
    }

    reporter.log(&format!("$ GET {}", source.url)).await;
    let dest = downloads.join(&source.file_name);
    let mut on_progress = reporter.progress(step, label.to_string());
    download_verified(
        client,
        &source.url,
        reqwest::header::HeaderMap::new(),
        &source.sha256,
        &dest,
        &mut on_progress,
    )
    .await?;
    reporter
        .log(&format!(
            "{} verified ({})",
            source.file_name, source.sha256
        ))
        .await;
    fetched.push(dest.clone());
    Ok(dest)
}

fn read_runtime_stamp(dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(dir.join(RUNTIME_STAMP_FILE)).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("sha256")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

#[allow(clippy::too_many_arguments)]
async fn ensure_runtime(
    reporter: &Reporter,
    client: &reqwest::Client,
    name: &str,
    package: &NativePackage,
    runtime_root: &Path,
    downloads: &Path,
    step: usize,
    fetched: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let source = package_source(package)?;
    let dir = runtime_dir(runtime_root, name);
    if read_runtime_stamp(&dir).as_deref() == Some(source.sha256.as_str()) {
        reporter
            .log(&format!("{name} is already installed at {}", dir.display()))
            .await;
        return Ok(());
    }

    let archive = fetch(
        reporter,
        client,
        &source,
        downloads,
        step,
        &format!("Downloading {name}"),
        fetched,
    )
    .await?;
    let staging = dir.with_file_name(format!("{name}.staging"));
    let target = dir.clone();
    let sha256 = source.sha256.clone();
    let runtime_name = name.to_string();
    tokio::task::spawn_blocking(move || {
        if staging.exists() {
            std::fs::remove_dir_all(&staging)
                .map_err(|error| format!("cannot clear {}: {error}", staging.display()))?;
        }
        std::fs::create_dir_all(&staging)
            .map_err(|error| format!("cannot create {}: {error}", staging.display()))?;
        extract_zip(&archive, &staging)?;
        let stamp = serde_json::json!({ "name": runtime_name, "sha256": sha256 });
        std::fs::write(staging.join(RUNTIME_STAMP_FILE), stamp.to_string())
            .map_err(|error| format!("cannot write the runtime stamp: {error}"))?;
        replace_dir(&staging, &target)
    })
    .await
    .map_err(|error| format!("unpacking {name} failed: {error}"))??;
    reporter
        .log(&format!("{name} installed at {}", dir.display()))
        .await;
    Ok(())
}

/// Unpack core plus a backend into a fresh staging folder, stamp it, and
/// swap it in for whatever was installed before.
async fn stage(
    core: &Path,
    backend: &Path,
    stamp: &NativeStamp,
    native_dir: &Path,
) -> Result<(), String> {
    let core = core.to_path_buf();
    let backend = backend.to_path_buf();
    let stamp = serde_json::to_string_pretty(stamp)
        .map_err(|error| format!("cannot write the runtime stamp: {error}"))?;
    let target = native_dir.to_path_buf();
    let staging = native_dir.with_file_name(format!("{NATIVE_DIR}.staging"));
    tokio::task::spawn_blocking(move || {
        if staging.exists() {
            std::fs::remove_dir_all(&staging)
                .map_err(|error| format!("cannot clear {}: {error}", staging.display()))?;
        }
        std::fs::create_dir_all(&staging)
            .map_err(|error| format!("cannot create {}: {error}", staging.display()))?;
        extract_zip(&core, &staging)?;
        extract_zip(&backend, &staging)?;
        std::fs::write(staging.join(STAMP_FILE), stamp)
            .map_err(|error| format!("cannot write the runtime stamp: {error}"))?;
        replace_dir(&staging, &target)
    })
    .await
    .map_err(|error| format!("unpacking failed: {error}"))?
}

/// The whole "install runtime" flow behind a native service's build button.
pub async fn install(
    info: BuildInfo,
    manager: Arc<Mutex<ServiceManager>>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    let reporter = Reporter {
        manager,
        handle,
        service_id: info.service_id.clone(),
    };
    let mut fetched = Vec::new();
    let result = install_inner(&reporter, &info, &mut fetched).await;
    // Packages are unpacked by now or will be fetched again; the verified
    // `.partial` resume path covers an interrupted download either way.
    for path in fetched {
        let _ = tokio::fs::remove_file(&path).await;
    }
    if let Err(error) = &result {
        reporter.log(&format!("\nERROR: {error}")).await;
    }
    result
}

async fn install_inner(
    reporter: &Reporter,
    info: &BuildInfo,
    fetched: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let def = info
        .native
        .as_ref()
        .ok_or_else(|| format!("{} has no native definition", info.service_id))?;
    let platform = current_platform();
    let packages = def.platforms.get(platform).ok_or_else(|| {
        format!(
            "{} {} has no package for {platform}",
            info.service_id, def.version
        )
    })?;
    let offered: Vec<String> = packages.backends.keys().cloned().collect();

    // 1. What GPU is here, and which backend it gets.
    reporter.step(0, "Detecting GPU...").await;
    let adapters = match detect_gpus().await {
        Ok(adapters) => adapters,
        Err(error) => {
            reporter
                .log(&format!(
                    "could not list GPUs ({error}); continuing without"
                ))
                .await;
            Vec::new()
        }
    };
    for adapter in &adapters {
        reporter
            .log(&format!("found {} ({:?})", adapter.name, adapter.vendor))
            .await;
    }
    let preference = crate::native_backend_preference(&info.service_id);
    let candidates = resolve_backends(&preference, &adapters, &offered, &packages.prefer)?;
    reporter
        .log(&format!(
            "backend: {} ({})",
            candidates[0].backend, candidates[0].reason
        ))
        .await;
    if candidates.len() > 1 {
        let rest: Vec<&str> = candidates[1..]
            .iter()
            .map(|choice| choice.backend.as_str())
            .collect();
        reporter
            .log(&format!("if it does not start: {}", rest.join(", then ")))
            .await;
    }

    let client = reqwest::Client::builder()
        .build()
        .map_err(|error| format!("cannot create an HTTP client: {error}"))?;
    let downloads = crate::storage::cache_dir(&info.runtime_root).join("native-downloads");

    // 2. The core runs anywhere by itself.
    let core_source = package_source(&packages.core)?;
    let core = fetch(
        reporter,
        &client,
        &core_source,
        &downloads,
        1,
        &format!("Downloading {} {}", info.service_id, def.version),
        fetched,
    )
    .await?;

    // 3-5. For each candidate in turn: fetch its backend and any shared
    // runtime, unpack it over the core, and check that it came up.
    let exe = info.env_dir.join(&def.executable);
    let mut failures: Vec<String> = Vec::new();
    for (index, choice) in candidates.iter().enumerate() {
        // resolve_backends only returns published backends.
        let package = &packages.backends[&choice.backend];
        let source = package_source(package)?;
        let backend_archive = fetch(
            reporter,
            &client,
            &source,
            &downloads,
            2,
            &format!("Downloading the {} backend", choice.backend),
            fetched,
        )
        .await?;
        for name in &package.requires {
            let runtime = info.native_runtimes.get(name).ok_or_else(|| {
                format!(
                    "the {} backend needs {name}, which the manifest does not define",
                    choice.backend
                )
            })?;
            ensure_runtime(
                reporter,
                &client,
                name,
                runtime,
                &info.runtime_root,
                &downloads,
                2,
                fetched,
            )
            .await?;
        }
        let runtimes = package.requires.clone();

        reporter.step(3, "Unpacking...").await;
        let stamp = NativeStamp {
            version: def.version.clone(),
            platform: platform.to_string(),
            backend: choice.backend.clone(),
            requested_backend: preference.clone(),
            // Loud on purpose: the service works, but not the way it was meant to.
            fallback_reason: (!failures.is_empty()).then(|| failures.join(" ")),
            runtimes,
        };
        stage(&core, &backend_archive, &stamp, &info.env_dir).await?;

        reporter
            .step(4, &format!("Checking the {} backend...", stamp.backend))
            .await;
        match check(reporter, info, &exe, &stamp).await {
            Ok(props) => {
                let recommended = props
                    .pointer("/hardware/recommended_encoding")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown");
                reporter
                    .log(&format!(
                        "\n=== {} {} installed on {} (recommended tier: {recommended}) ===",
                        info.service_id, def.version, stamp.backend
                    ))
                    .await;
                return Ok(());
            }
            Err(error) => {
                let Some(next) = candidates.get(index + 1) else {
                    return Err(error);
                };
                let reason = fallback_reason(&choice.backend, &error, &next.backend);
                reporter.log(&format!("WARNING: {reason}")).await;
                failures.push(reason);
            }
        }
    }
    Err("no backend could be installed".to_string())
}

fn fallback_reason(failed: &str, error: &str, next: &str) -> String {
    let hint = if failed == "cuda" {
        " Updating the NVIDIA driver usually fixes CUDA."
    } else {
        ""
    };
    format!("{failed} did not start on this machine ({error}). Running on {next} instead.{hint}")
}

/// Run the probe with the env the service will launch with, confirm the
/// chosen backend is among the devices, and keep its answer for the UI.
async fn check(
    reporter: &Reporter,
    info: &BuildInfo,
    exe: &Path,
    stamp: &NativeStamp,
) -> Result<serde_json::Value, String> {
    let env = launch_env(&info.service_env, &stamp.backend);
    let path = path_with_runtimes(&info.runtime_root, &stamp.runtimes);
    let props = probe(exe, &env, path).await?;
    for line in describe_devices(&props) {
        reporter.log(&line).await;
    }
    if !props_has_backend(&props, &stamp.backend) {
        return Err(format!(
            "the {} backend did not initialise; the runtime saw only: {}",
            stamp.backend,
            describe_devices(&props).join(";").trim()
        ));
    }
    let serialized = serde_json::to_string_pretty(&props)
        .map_err(|error| format!("cannot keep the runtime check: {error}"))?;
    if let Some(dir) = exe.parent() {
        std::fs::write(dir.join(PROPS_FILE), serialized)
            .map_err(|error| format!("cannot keep the runtime check: {error}"))?;
    }
    Ok(props)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, vendor: GpuVendor) -> GpuAdapter {
        GpuAdapter {
            name: name.to_string(),
            vendor,
        }
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn backends(choices: Vec<BackendChoice>) -> Vec<String> {
        choices.into_iter().map(|choice| choice.backend).collect()
    }

    fn hybrid_laptop() -> [GpuAdapter; 2] {
        [
            gpu("Intel(R) UHD Graphics", GpuVendor::Intel),
            gpu("NVIDIA GeForce RTX 5070 Laptop GPU", GpuVendor::Nvidia),
        ]
    }

    #[test]
    fn without_a_preference_nvidia_tries_cuda_then_vulkan() {
        let choices =
            resolve_backends("auto", &hybrid_laptop(), &names(&["cuda", "vulkan"]), &[]).unwrap();
        assert_eq!(backends(choices), ["cuda", "vulkan"]);
    }

    #[test]
    fn a_service_that_prefers_vulkan_takes_it_on_nvidia_and_keeps_cuda_behind() {
        let choices = resolve_backends(
            "auto",
            &hybrid_laptop(),
            &names(&["cuda", "vulkan"]),
            &names(&["vulkan", "cuda"]),
        )
        .unwrap();
        assert_eq!(backends(choices), ["vulkan", "cuda"]);
    }

    #[test]
    fn amd_and_intel_get_vulkan_alone() {
        for adapter in [
            gpu("AMD Radeon(TM) 8060S Graphics", GpuVendor::Amd),
            gpu("Intel(R) Arc(TM) Graphics", GpuVendor::Intel),
        ] {
            let choices = resolve_backends(
                "",
                &[adapter],
                &names(&["cuda", "vulkan"]),
                &names(&["vulkan", "cuda"]),
            )
            .unwrap();
            assert_eq!(backends(choices), ["vulkan"]);
        }
    }

    #[test]
    fn no_supported_gpu_is_an_error_not_a_cpu_runtime() {
        let adapters = [gpu("Microsoft Basic Render Driver", GpuVendor::Other)];
        let error =
            resolve_backends("auto", &adapters, &names(&["cuda", "vulkan"]), &[]).unwrap_err();
        assert!(error.contains("Microsoft Basic Render Driver"));
        // "cpu" saved in settings by an earlier build is not published either.
        let offered = names(&["cuda", "vulkan"]);
        assert!(resolve_backends("cpu", &hybrid_laptop(), &offered, &[]).is_err());
    }

    #[test]
    fn an_empty_gpu_list_tries_every_backend() {
        let offered = names(&["cuda", "vulkan"]);
        let choices = resolve_backends("auto", &[], &offered, &offered).unwrap();
        assert_eq!(backends(choices), ["cuda", "vulkan"]);
    }

    #[test]
    fn nvidia_without_a_cuda_package_uses_vulkan() {
        let adapters = [gpu("NVIDIA GeForce RTX 3060", GpuVendor::Nvidia)];
        let choices = resolve_backends("auto", &adapters, &names(&["vulkan"]), &[]).unwrap();
        assert_eq!(backends(choices), ["vulkan"]);
    }

    #[test]
    fn a_chosen_backend_is_the_only_candidate() {
        let choices = resolve_backends(
            "CUDA",
            &hybrid_laptop(),
            &names(&["cuda", "vulkan"]),
            &names(&["vulkan", "cuda"]),
        )
        .unwrap();
        assert_eq!(backends(choices), ["cuda"]);
    }

    #[test]
    fn an_unpublished_backend_is_refused() {
        let error = resolve_backends("metal", &[], &names(&["cuda", "vulkan"]), &[]).unwrap_err();
        assert!(error.contains("metal"));
    }

    #[test]
    fn a_failed_cuda_says_what_usually_fixes_it() {
        let reason = fallback_reason("cuda", "no CUDA device", "vulkan");
        assert!(reason.contains("Running on vulkan instead"));
        assert!(reason.contains("NVIDIA driver"));
        assert!(!fallback_reason("vulkan", "no Vulkan device", "cuda").contains("driver"));
    }

    #[test]
    fn video_controllers_parse_from_one_object_or_many() {
        let one = r#"{"Name":"AMD Radeon(TM) 8060S Graphics","PNPDeviceID":"PCI\\VEN_1002&DEV_1586&SUBSYS"}"#;
        let adapters = parse_video_controllers(one).unwrap();
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].vendor, GpuVendor::Amd);

        let many = r#"[{"Name":"Intel(R) UHD Graphics","PNPDeviceID":"PCI\\VEN_8086&DEV_A7A0"},
                      {"Name":"NVIDIA GeForce RTX 5070 Laptop GPU","PNPDeviceID":"PCI\\VEN_10DE&DEV_2F58"},
                      {"Name":"Microsoft Basic Render Driver","PNPDeviceID":"ROOT\\BasicRender\\0000"}]"#;
        let vendors: Vec<GpuVendor> = parse_video_controllers(many)
            .unwrap()
            .into_iter()
            .map(|adapter| adapter.vendor)
            .collect();
        assert_eq!(
            vendors,
            [GpuVendor::Intel, GpuVendor::Nvidia, GpuVendor::Other]
        );
    }

    #[test]
    fn sha256sums_parse_both_text_and_binary_markers() {
        let hash = "a".repeat(64);
        let sums = parse_sha256sums(&format!(
            "{hash}  yuey-core.zip\n{hash} *cudart.zip\nnot a line\n"
        ));
        assert_eq!(sums.get("yuey-core.zip"), Some(&hash));
        assert_eq!(sums.get("cudart.zip"), Some(&hash));
        assert_eq!(sums.len(), 2);
    }

    #[test]
    fn an_unpinned_package_is_refused_before_any_download() {
        std::env::remove_var("GARY4LOCAL_NATIVE_PACKAGE_DIR");
        let package = NativePackage {
            url: "https://example.invalid/yuey-v0.2.0-windows-x64-core.zip".to_string(),
            sha256: "unpublished".to_string(),
            requires: Vec::new(),
        };
        let error = package_source(&package).err().unwrap();
        assert!(error.contains("no published checksum"));
    }

    #[test]
    fn the_backend_template_is_filled_and_unresolved_values_are_dropped() {
        let template = vec![
            ("YUE2_DEVICE".to_string(), "${NATIVE_BACKEND}".to_string()),
            ("YUE2_PORT".to_string(), "8007".to_string()),
            ("LEFT_OVER".to_string(), "${SOMETHING_ELSE}".to_string()),
        ];
        let env: HashMap<_, _> = launch_env(&template, "vulkan").into_iter().collect();
        assert_eq!(env.get("YUE2_DEVICE").map(String::as_str), Some("vulkan"));
        assert_eq!(env.get("YUE2_PORT").map(String::as_str), Some("8007"));
        assert!(!env.contains_key("LEFT_OVER"));
    }

    #[test]
    fn a_probe_counts_a_backend_only_when_a_device_reports_it() {
        let props = serde_json::json!({
            "devices": [
                {"backend": "CPU", "description": "AMD Ryzen", "type": "cpu"},
                {"backend": "Vulkan", "description": "Radeon 8060S", "type": "integrated_gpu"}
            ]
        });
        assert!(props_has_backend(&props, "vulkan"));
        assert!(!props_has_backend(&props, "cuda"));
    }

    fn temp_dir(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("gary4local-native-{label}-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn replacing_a_runtime_keeps_only_the_new_one() {
        let root = temp_dir("replace");
        let target = root.join("native");
        let staging = root.join("native.staging");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("old.dll"), b"old").unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("new.dll"), b"new").unwrap();

        replace_dir(&staging, &target).unwrap();

        assert!(target.join("new.dll").is_file());
        assert!(!target.join("old.dll").exists());
        assert!(!staging.exists());
        assert!(!root.join("native.old").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_stamped_runtime_with_its_executable_is_installed() {
        let root = temp_dir("stamp");
        let native = root.join("native");
        std::fs::create_dir_all(&native).unwrap();
        assert!(installed("stamp-test", &native, "yue2-server.exe").is_none());

        let stamp = NativeStamp {
            version: "v0.2.0".to_string(),
            platform: "windows-x64".to_string(),
            backend: "cuda".to_string(),
            requested_backend: "auto".to_string(),
            fallback_reason: None,
            runtimes: vec!["cudart-12.8".to_string()],
        };
        std::fs::write(
            native.join(STAMP_FILE),
            serde_json::to_string(&stamp).unwrap(),
        )
        .unwrap();
        // A stamp without the executable is a broken install, not an install.
        assert!(installed("stamp-test", &native, "yue2-server.exe").is_none());

        std::fs::write(native.join("yue2-server.exe"), b"").unwrap();
        let install = installed("stamp-test", &native, "yue2-server.exe").unwrap();
        assert_eq!(install.backend, "cuda");
        assert_eq!(install.version.as_deref(), Some("v0.2.0"));
        assert_eq!(install.runtimes, ["cudart-12.8"]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
