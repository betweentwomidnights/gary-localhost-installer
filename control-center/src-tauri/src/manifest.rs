use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub services: Vec<ServiceDef>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServiceDef {
    pub id: String,
    pub display_name: String,
    pub port: u16,
    pub entry_point: String,
    pub working_dir: String,
    #[serde(default = "default_python_version")]
    pub python_version: String,
    #[serde(default = "default_accelerator_profile")]
    pub accelerator_profile: String,
    #[serde(default)]
    pub build_steps: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    pub health_check: Option<HealthCheck>,
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

pub fn load_manifest(path: &Path) -> Result<Vec<ServiceDef>, String> {
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

    Ok(manifest.services)
}

#[cfg(test)]
mod tests {
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
