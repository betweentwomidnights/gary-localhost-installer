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
    #[serde(default = "default_python_version")]
    pub python_version: String,
    #[serde(default = "default_accelerator_profile")]
    pub accelerator_profile: String,
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
fn default_python_version() -> String {
    "3.11".to_string()
}
fn default_accelerator_profile() -> String {
    "cuda-nvidia".to_string()
}

pub fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;

    let manifest: Manifest =
        serde_json::from_str(&content).map_err(|e| format!("Invalid manifest JSON: {}", e))?;

    log::info!("Loaded {} services from manifest", manifest.services.len());
    for svc in &manifest.services {
        log::info!(
            "  {} ({}) on port {} using Python {} ({})",
            svc.display_name,
            svc.id,
            svc.port,
            svc.python_version,
            svc.accelerator_profile
        );
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

#[cfg(test)]
mod rocm_tests {
    use super::Manifest;

    #[test]
    fn carey_rocm_profile_uses_fast_miopen_find_mode() {
        let manifest: Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .expect("bundled service manifest should be valid JSON");

        let carey = manifest
            .services
            .iter()
            .find(|service| service.id == "carey")
            .expect("bundled manifest should define Carey");

        assert!(carey.accelerator_profile.contains("rocm"));
        assert_eq!(
            carey.env.get("MIOPEN_FIND_MODE").map(String::as_str),
            Some("2")
        );
    }

    #[test]
    fn stable_audio_services_use_the_windows_rocm_runtime() {
        let manifest: Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .expect("bundled service manifest should be valid JSON");

        for id in ["stable-audio", "foundation"] {
            let service = manifest
                .services
                .iter()
                .find(|service| service.id == id)
                .unwrap_or_else(|| panic!("bundled manifest should define {id}"));

            assert_eq!(service.python_version, "3.12");
            assert_eq!(service.accelerator_profile, "amd-rocm-windows-7.2.1");
            assert!(service.build_steps.iter().any(|step| step.contains("rocm7.2.1")));
            assert!(service
                .build_steps
                .iter()
                .any(|step| step.contains("torchvision-0.24.1")));
            assert!(!service.build_steps.iter().any(|step| step.contains("flash_attn")));
            assert_eq!(
                service.env.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL").map(String::as_str),
                Some("1")
            );
            assert_eq!(
                service.env.get("MIOPEN_FIND_MODE").map(String::as_str),
                Some("2")
            );
        }
    }

    #[test]
    fn melodyflow_uses_the_windows_rocm_runtime_without_cuda_extensions() {
        let manifest: Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .expect("bundled service manifest should be valid JSON");
        let service = manifest
            .services
            .iter()
            .find(|service| service.id == "melodyflow")
            .expect("bundled manifest should define melodyflow");

        assert_eq!(service.python_version, "3.12");
        assert_eq!(service.accelerator_profile, "amd-rocm-windows-7.2.1");
        assert!(service.build_steps.iter().any(|step| step.contains("rocm7.2.1")));
        assert!(!service.build_steps.iter().any(|step| step.contains("flash_attn")));
        assert!(!service
            .build_steps
            .iter()
            .any(|step| step.contains("install_xformers_shim")));
        assert_eq!(service.env.get("MIOPEN_FIND_MODE").map(String::as_str), Some("2"));

        let requirements = include_str!("../../../services/melodyflow/requirements.txt");
        assert!(requirements.contains("transformers==4.39.3"));
        assert!(requirements.contains("tokenizers==0.15.2"));
    }

    #[test]
    fn gary_uses_the_windows_rocm_runtime_without_cuda_extensions() {
        let manifest: Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .expect("bundled service manifest should be valid JSON");
        let service = manifest
            .services
            .iter()
            .find(|service| service.id == "gary")
            .expect("bundled manifest should define gary");

        assert_eq!(service.python_version, "3.12");
        assert_eq!(service.accelerator_profile, "amd-rocm-windows-7.2.1");
        assert!(service.build_steps.iter().any(|step| step.contains("rocm7.2.1")));
        assert!(!service.build_steps.iter().any(|step| step.contains("flash_attn")));
        assert!(!service
            .build_steps
            .iter()
            .any(|step| step.contains("install_xformers_shim")));
        // Gary installs its own package before requirements, same as the CUDA build.
        assert!(service
            .build_steps
            .iter()
            .any(|step| step.contains("pip install -e . --no-deps")));
        assert_eq!(service.env.get("MIOPEN_FIND_MODE").map(String::as_str), Some("2"));

        // The T5 conditioner pulls dynamo and torch.distributed.fsdp on newer
        // Transformers, which the Windows ROCm wheel cannot import.
        let requirements = include_str!("../../../services/gary/requirements.txt");
        assert!(requirements.contains("transformers==4.39.3"));
        assert!(requirements.contains("tokenizers==0.15.2"));
    }
}
