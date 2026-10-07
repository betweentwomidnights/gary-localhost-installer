use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub services: Vec<ServiceDef>,
    /// Shared runtime packs that native backends depend on, such as the CUDA
    /// runtime. Installed once and put on PATH for every service that needs one.
    #[serde(default)]
    pub native_runtimes: HashMap<String, NativePackage>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceDef {
    pub id: String,
    pub display_name: String,
    pub port: u16,
    #[serde(default)]
    pub runtime: ServiceRuntime,
    /// The Python script a Python service runs. Native services name their
    /// executable in `native` instead.
    #[serde(default)]
    pub entry_point: String,
    pub working_dir: String,
    #[serde(default)]
    pub build_steps: Vec<String>,
    #[serde(default)]
    pub native: Option<NativeDef>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    pub health_check: Option<HealthCheck>,
}

/// How a service runs: a Python entry point inside a uv venv, or a prebuilt
/// native executable downloaded for the machine's GPU.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceRuntime {
    #[default]
    Python,
    Native,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeDef {
    pub executable: String,
    /// The release the packages below were published under. A different
    /// version in an installed runtime's stamp means it needs updating.
    pub version: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Keyed by platform, e.g. `windows-x64`.
    pub platforms: HashMap<String, NativePlatform>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativePlatform {
    /// The executable, ggml, and every CPU variant. Runs anywhere on its own.
    pub core: NativePackage,
    /// One GPU backend each, e.g. `cuda` or `vulkan`, unpacked over the core.
    #[serde(default)]
    pub backends: HashMap<String, NativePackage>,
    /// The order "auto" tries backends in, among those the machine's GPUs can
    /// run. Measured per service, since the fastest backend is not always the
    /// vendor's own; empty means CUDA, then Metal, then Vulkan.
    #[serde(default)]
    pub prefer: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativePackage {
    pub url: String,
    pub sha256: String,
    /// Names of `nativeRuntimes` entries this package needs on PATH.
    #[serde(default)]
    pub requires: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthCheck {
    pub endpoint: String,
    #[serde(default = "default_interval")]
    pub interval_seconds: u64,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
    #[serde(default = "default_startup_grace")]
    pub startup_grace_seconds: u64,
}

fn default_interval() -> u64 {
    15
}
fn default_timeout() -> u64 {
    5
}
fn default_startup_grace() -> u64 {
    0
}

pub fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;

    let manifest: Manifest =
        serde_json::from_str(&content).map_err(|e| format!("Invalid manifest JSON: {}", e))?;

    log::info!("Loaded {} services from manifest", manifest.services.len());
    for svc in &manifest.services {
        log::info!("  {} ({}) on port {}", svc.display_name, svc.id, svc.port);
    }

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::{Manifest, ServiceRuntime};

    fn bundled() -> Manifest {
        serde_json::from_str(include_str!("../../../services/manifests/services.json"))
            .expect("bundled service manifest should be valid JSON")
    }

    #[test]
    fn python_services_default_to_the_python_runtime() {
        let manifest = bundled();
        let gary = manifest
            .services
            .iter()
            .find(|service| service.id == "gary")
            .expect("bundled manifest should define gary");

        assert_eq!(gary.runtime, ServiceRuntime::Python);
        assert_eq!(gary.entry_point, "g4l_localhost.py");
        assert!(gary.native.is_none());
    }

    #[test]
    fn yuey_is_a_native_service_with_a_package_for_every_backend_it_offers() {
        let manifest = bundled();
        let yuey = manifest
            .services
            .iter()
            .find(|service| service.id == "yuey")
            .expect("bundled manifest should define yuey");

        assert_eq!(yuey.runtime, ServiceRuntime::Native);
        assert_eq!(yuey.port, 8007);
        let native = yuey
            .native
            .as_ref()
            .expect("yuey should carry a native block");
        assert_eq!(native.executable, "yue2-server.exe");

        let windows = native
            .platforms
            .get("windows-x64")
            .expect("yuey should publish a windows-x64 package");
        for backend in ["cuda", "vulkan"] {
            assert!(
                windows.backends.contains_key(backend),
                "yuey should offer a {backend} backend"
            );
        }
        // Measured on an RTX 5070 Laptop: with the decode step replayed as a
        // CUDA graph, CUDA is also the faster backend, and with a DAW open a
        // 170s Vulkan render stalled for over ten minutes where CUDA finished.
        // gary4juce lives in a DAW, so NVIDIA stays on CUDA.
        assert_eq!(windows.prefer, ["cuda", "vulkan"]);

        // Every runtime a backend asks for has to be one the manifest can install.
        for package in windows.backends.values() {
            for runtime in &package.requires {
                assert!(
                    manifest.native_runtimes.contains_key(runtime),
                    "{runtime} is required but not defined"
                );
            }
        }
    }
}
