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
    /// One package set can provide multiple executables/services (SA3, SAOS,
    /// Foundation and the trainer). Service entries refer to it by identity.
    #[serde(default)]
    pub native_bundles: HashMap<String, NativeBundle>,
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
    #[serde(default)]
    pub bundle: Option<String>,
    /// The release the packages below were published under. A different
    /// version in an installed runtime's stamp means it needs updating.
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Keyed by platform, e.g. `windows-x64`.
    #[serde(default)]
    pub platforms: HashMap<String, NativePlatform>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeBundle {
    pub version: String,
    pub platforms: HashMap<String, NativePlatform>,
}

pub fn valid_bundle_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == b'-' || ch == b'_')
        && !matches!(id, "con" | "prn" | "aux" | "nul")
        && !(id.len() == 4
            && (id.starts_with("com") || id.starts_with("lpt"))
            && id.as_bytes()[3].is_ascii_digit())
}

impl Manifest {
    /// Resolve references once so install/launch paths consume the same pinned
    /// version and backend packages, with no repeated per-service definitions.
    pub fn resolve_native_bundles(&mut self) -> Result<(), String> {
        for id in self.native_bundles.keys() {
            if !valid_bundle_id(id) {
                return Err(format!("Invalid native bundle identity: {id}"));
            }
        }
        for service in &mut self.services {
            let Some(native) = service.native.as_mut() else {
                continue;
            };
            let Some(id) = native.bundle.as_ref() else {
                continue;
            };
            if !valid_bundle_id(id) {
                return Err(format!("Invalid native bundle identity: {id}"));
            }
            let bundle = self
                .native_bundles
                .get(id)
                .ok_or_else(|| format!("{} references undefined native bundle {id}", service.id))?;
            if (!native.version.is_empty() && native.version != bundle.version)
                || (!native.platforms.is_empty() && native.platforms != bundle.platforms)
            {
                return Err(format!(
                    "{} must use the version/packages from native bundle {id}",
                    service.id
                ));
            }
            native.version = bundle.version.clone();
            native.platforms = bundle.platforms.clone();
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
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

    let mut manifest: Manifest =
        serde_json::from_str(&content).map_err(|e| format!("Invalid manifest JSON: {}", e))?;
    manifest.resolve_native_bundles()?;

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
        let mut manifest: Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .expect("bundled service manifest should be valid JSON");
        manifest
            .resolve_native_bundles()
            .expect("bundle references should resolve");
        manifest
    }

    #[test]
    fn sa3_candidate_uses_one_pinned_bundle_without_switching_python() {
        let mut manifest = bundled();
        let sa3 = manifest
            .services
            .iter()
            .find(|service| service.id == "sa3")
            .unwrap();
        assert_eq!(sa3.runtime, ServiceRuntime::Python);
        let native = sa3.native.as_ref().unwrap();
        assert_eq!(native.bundle.as_deref(), Some("sa3"));
        assert_eq!(native.version, manifest.native_bundles["sa3"].version);
        let cuda = &native.platforms["windows-x64"].backends["cuda"];
        assert_eq!(cuda.requires, ["cudart-12.8"]);
        assert!(crate::native_runtime::is_sha256(&cuda.sha256));
        manifest.resolve_native_bundles().unwrap();
    }

    #[test]
    fn unknown_or_escaping_bundle_references_are_rejected() {
        for id in ["missing", "../sa3", "C:/sa3", "COM1", "con", "lpt2"] {
            let mut manifest = bundled();
            let native = manifest
                .services
                .iter_mut()
                .find(|service| service.id == "sa3")
                .unwrap()
                .native
                .as_mut()
                .unwrap();
            native.bundle = Some(id.to_string());
            assert!(manifest.resolve_native_bundles().is_err(), "{id}");
        }
    }

    #[test]
    fn consumer_cannot_override_shared_package_identity() {
        let mut manifest = bundled();
        manifest
            .services
            .iter_mut()
            .find(|service| service.id == "sa3")
            .unwrap()
            .native
            .as_mut()
            .unwrap()
            .version = "v9.9.9".to_string();
        assert!(manifest.resolve_native_bundles().is_err());
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
