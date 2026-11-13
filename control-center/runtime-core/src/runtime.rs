//! Shared native package and launch primitives for desktop and headless hosts.
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Allow a stopped server's TIME_WAIT sockets without sharing an active port.
pub fn bind_loopback(port: u16) -> std::io::Result<std::net::TcpListener> {
    let socket = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )?;
    #[cfg(unix)]
    socket.set_reuse_address(true)?;
    socket.bind(&std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, port)).into())?;
    socket.listen(128)?;
    Ok(socket.into())
}

pub fn current_platform() -> &'static str {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        "windows-x64"
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "macos-arm64"
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        "macos-x64"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "linux-x64"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "linux-arm64"
    } else {
        "unsupported"
    }
}

/// How a native server process starts: program, arguments, folder and every
/// environment variable it gets. The service and any private copy started to
/// verify it are built from the same value, so what a migration proves is what
/// the service then runs. Callers add only their own output handling, process
/// group and, for a private copy, its port.
#[derive(Debug, Clone)]
pub struct NativeLaunch {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub dir: PathBuf,
    /// Applied in order, later entries winning.
    pub env: Vec<(OsString, OsString)>,
}

impl NativeLaunch {
    /// `storage_env` is the profile's runtime variables, then the service's
    /// resolved template with `${NATIVE_BACKEND}` filled in, then PATH with
    /// the install's shared runtimes ahead of the system's.
    pub fn new(
        runtime_root: &Path,
        executable: &str,
        install: &NativeInstall,
        args: Vec<String>,
        storage_env: Vec<(String, String)>,
        template: &[(String, String)],
    ) -> Self {
        let mut env: Vec<(OsString, OsString)> = storage_env
            .into_iter()
            .chain(launch_env(template, &install.backend))
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        if let Some(path) = path_with_runtimes(runtime_root, &install.runtimes) {
            env.push(("PATH".into(), path));
        }
        Self {
            program: install.executable(executable),
            args,
            dir: install.dir.clone(),
            env,
        }
    }

    pub fn std_command(&self) -> std::process::Command {
        let mut command = std::process::Command::new(&self.program);
        command.args(&self.args).current_dir(&self.dir);
        command.envs(self.env.iter().map(|(key, value)| (key, value)));
        command
    }

    pub fn tokio_command(&self) -> tokio::process::Command {
        let mut command = tokio::process::Command::new(&self.program);
        command.args(&self.args).current_dir(&self.dir);
        command.envs(self.env.iter().map(|(key, value)| (key, value)));
        command
    }
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

impl NativeInstall {
    /// Resolve a manifest/tool name the same way as the service launcher.
    /// Windows keeps `.exe`; Linux and macOS packages omit that suffix.
    pub fn executable(&self, name: &str) -> PathBuf {
        self.dir.join(executable_name(name))
    }
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
        .map(|name| runtime_root.join("native-runtimes").join(name))
        .collect();
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    std::env::join_paths(dirs).ok()
}

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
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            // Keep executable bits, without granting setuid/setgid privileges.
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode & 0o777))
                .map_err(|error| {
                    format!("cannot set permissions on {}: {error}", target.display())
                })?;
        }
    }
    Ok(())
}

/// Manifests retain Windows executable names; Unix packages omit that suffix.
pub fn executable_name(name: &str) -> &str {
    if cfg!(windows) {
        name
    } else {
        name.strip_suffix(".exe").unwrap_or(name)
    }
}

/// Some Vulkan software drivers are exposed as GPU devices by ggml. They
/// cannot satisfy an installation that requires GPU generation.
pub fn is_generation_gpu(device: &serde_json::Value) -> bool {
    let kind = device["type"].as_str().unwrap_or("").to_ascii_lowercase();
    if !matches!(kind.as_str(), "gpu" | "igpu" | "integrated_gpu" | "discrete_gpu") {
        return false;
    }
    let description = format!("{} {}", device["name"].as_str().unwrap_or(""),
        device["description"].as_str().unwrap_or("")).to_ascii_lowercase();
    !["llvmpipe", "lavapipe", "swiftshader", "software rasterizer"]
        .iter().any(|name| description.contains(name))
}

#[cfg(test)]
mod executable_tests {
    use super::*;

    #[test]
    fn native_tool_paths_match_the_service_launcher_on_this_platform() {
        let install = NativeInstall {
            dir: PathBuf::from("native runtime with spaces"),
            backend: "vulkan".into(), runtimes: Vec::new(), version: None, fallback_reason: None,
        };
        for name in ["sa3-server", "sa3-train", "sa3-audio-analyze", "sa3-lora-convert", "ace-train", "ace-server", "sat-server", "musicgen-server", "melodyflow-server", "yue2-server"] {
            let manifest_name = format!("{name}.exe");
            let expected = install.dir.join(if cfg!(windows) { manifest_name.clone() } else { name.into() });
            assert_eq!(install.executable(&manifest_name), expected);
            let launch = NativeLaunch::new(Path::new("profile"), &manifest_name, &install, Vec::new(), Vec::new(), &[]);
            assert_eq!(launch.program, expected);
            assert_eq!(install.executable(name), install.dir.join(name));
        }
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::is_generation_gpu;
    use serde_json::json;

    #[test]
    fn software_vulkan_does_not_count_as_a_generation_gpu() {
        for description in ["llvmpipe (LLVM 20.1.2, 256 bits)", "lavapipe", "SwiftShader Device"] {
            assert!(!is_generation_gpu(&json!({"type":"gpu", "backend":"Vulkan", "description":description})));
        }
        assert!(!is_generation_gpu(&json!({"type":"cpu", "description":"AMD Ryzen"})));
        for (kind, description) in [("gpu", "NVIDIA GeForce RTX 5070 Laptop GPU"), ("gpu", "NVIDIA GB10"), ("integrated_gpu", "AMD Radeon 8060S")] {
            assert!(is_generation_gpu(&json!({"type":kind,"description":description})));
        }
    }
}
