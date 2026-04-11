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

pub fn load_manifest(path: &Path) -> Result<Vec<ServiceDef>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;

    let manifest: Manifest =
        serde_json::from_str(&content).map_err(|e| format!("Invalid manifest JSON: {}", e))?;

    log::info!("Loaded {} services from manifest", manifest.services.len());
    for svc in &manifest.services {
        log::info!("  {} ({}) on port {}", svc.display_name, svc.id, svc.port);
    }

    Ok(manifest.services)
}
