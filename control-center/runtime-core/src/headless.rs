//! Experimental single-service host. No desktop runtime or
//! production-profile migration is involved in this first SA3 path.
use crate::runtime::{self, NativeInstall, NativeLaunch};
use crate::sa3_adapter::{AdapterListener, AdapterOptions};
use crate::service::Service;
use crate::workload_job::ManagedChild;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    pub path: PathBuf,
    pub sha256: String,
    #[serde(skip)]
    url: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub service: Service,
    #[serde(default = "default_root")]
    pub runtime_root: PathBuf,
    #[serde(default)]
    pub models_dir: PathBuf,
    #[serde(default)]
    pub core: Package,
    #[serde(default)]
    pub backend: Package,
    /// Omit for the manifest pinned in this build. Local archive pairs override it.
    #[serde(default)]
    pub manifest: Option<PathBuf>,
    #[serde(skip)]
    expected_version: Option<String>,
    /// Exact GPU name returned by --props, e.g. CUDA0 or Vulkan0.
    pub device: String,
    pub public_port: u16,
    pub model: String,
    pub encoding: String,
    #[serde(default)]
    pub selected_models: std::collections::BTreeMap<String, String>,
    #[serde(default = "f16")]
    pub t5_encoding: String,
    #[serde(default = "f16")]
    pub ae_encoding: String,
    #[serde(default = "threads")]
    pub threads: usize,
    #[serde(default)]
    pub defaults: Map<String, Value>,
}
fn f16() -> String {
    "f16".into()
}
fn threads() -> usize {
    4
}
pub(crate) fn default_root() -> PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base =
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Application Support"));
    #[cfg(not(any(windows, target_os = "macos")))]
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/share")));
    base.map(|p| p.join("gary4local/headless"))
        .unwrap_or_default()
}

#[derive(Debug)]
pub struct ConfigError {
    pub code: &'static str,
    pub message: String,
    pub hint: &'static str,
}
impl std::fmt::Display for ConfigError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(&self.message)
    }
}

impl Config {
    pub fn read(path: &Path) -> Result<Self, String> {
        Self::read_checked(path).map_err(|error| error.to_string())
    }
    pub fn read_checked(path: &Path) -> Result<Self, ConfigError> {
        let raw = std::fs::read(path).map_err(|error| ConfigError {
            code: "config_unreadable",
            message: format!("Cannot read config {}: {error}", path.display()),
            hint: "Check the config path and permissions, or create a config with init.",
        })?;
        Self::from_bytes(&raw)
    }
    pub fn from_bytes(raw: &[u8]) -> Result<Self, ConfigError> {
        let mut config: Self = serde_json::from_slice(raw).map_err(|error| ConfigError {
            code: "config_invalid_json",
            message: format!("Invalid headless config: {error}"),
            hint: "Use schema to inspect the config fields; unknown fields are rejected.",
        })?;
        if config.models_dir.as_os_str().is_empty() {
            config.models_dir = config.managed_models_dir();
        }
        config.resolve_packages().map_err(|message| ConfigError {
            code: "config_package_resolution", message,
            hint: "Inspect capabilities for published backends, or provide a verified local archive pair.",
        })?;
        config.validate().map_err(|message| ConfigError {
            code: "config_invalid", message,
            hint: "Use absolute paths, a dedicated runtime root, a nonzero port and an exact GPU device name.",
        })?;
        Ok(config)
    }
    pub fn log_path(&self) -> PathBuf {
        self.runtime_root
            .join("logs")
            .join(format!("{}-native.log", self.service.id()))
    }
    pub fn managed_models_dir(&self) -> PathBuf {
        self.runtime_root.join("models").join(self.service.id())
    }
    pub fn model_files(&self) -> Result<Vec<&'static crate::model_catalog::PinnedHfFile>, String> {
        if self.service == Service::Sa3 {
            return crate::model_catalog::sa3_medium_files(
                &self.model,
                &self.encoding,
                &self.t5_encoding,
                &self.ae_encoding,
            );
        }
        let mut files =
            crate::model_catalog::service_files(self.service, &self.model, &self.encoding)?;
        let mut names: std::collections::HashSet<_> =
            files.iter().map(|file| file.filename.as_str()).collect();
        for (model, encoding) in &self.selected_models {
            files.extend(
                crate::model_catalog::service_files(self.service, model, encoding)?
                    .into_iter()
                    .filter(|file| names.insert(&file.filename)),
            );
        }
        Ok(files)
    }

    pub fn recommended_tier(&self, model: &str) -> Option<&'static str> {
        use crate::model_catalog::{recommended_model_precision, recommended_yuey_tier};
        let props: Value =
            serde_json::from_slice(&std::fs::read(self.installed_dir().join("props.json")).ok()?)
                .ok()?;
        let device = self.selected_gpu(&props)?;
        let memory = device["memory_total_bytes"]
            .as_u64()
            .filter(|&bytes| bytes > 0)?;
        match self.service {
            Service::Gary | Service::Carey => {
                Some(recommended_model_precision(self.service, model, memory))
                    .filter(|tier| !tier.is_empty())
            }
            Service::Yuey => {
                let hardware = &props["hardware"];
                if hardware["memory_total_bytes"].as_u64() != Some(memory) {
                    return None;
                }
                recommended_yuey_tier(hardware["recommended_encoding"].as_str()?)
            }
            _ => None,
        }
    }

    fn selected_gpu<'a>(&self, props: &'a Value) -> Option<&'a Value> {
        props["devices"]
            .as_array()?
            .iter()
            .filter(|device| {
                device["backend"]
                    .as_str()
                    .is_some_and(|backend| backend.eq_ignore_ascii_case(self.backend_name()))
            })
            .enumerate()
            .find_map(|(index, device)| {
                let matches = device["name"] == self.device
                    || (device["name"].is_null()
                        && format!("{}{}", device["backend"].as_str().unwrap_or(""), index)
                            == self.device);
                (matches && runtime::is_generation_gpu(device)).then_some(device)
            })
    }
    pub fn summary(&self) -> Value {
        json!({"service":self.service, "platform":runtime::current_platform(), "runtime_root":self.runtime_root,
            "models_dir":self.models_dir, "model":self.model, "encoding":self.encoding, "selected_models":self.selected_models,
            "t5_encoding":self.t5_encoding, "ae_encoding":self.ae_encoding,
            "device":self.device, "public_port":self.public_port,"expected_version":self.expected_version,
            "public_url":format!("http://127.0.0.1:{}",self.public_port),
            "runtime_directory":self.installed_dir(),
            "packages":{"core":{"path":self.core.path,"sha256":self.core.sha256,"url":self.core.url},
                "backend":{"path":self.backend.path,"sha256":self.backend.sha256,"url":self.backend.url}}})
    }
    fn resolve_packages(&mut self) -> Result<(), String> {
        let local =
            !self.core.path.as_os_str().is_empty() || !self.backend.path.as_os_str().is_empty();
        if local {
            if self.manifest.is_some() {
                return Err("Choose a manifest or a local archive pair".into());
            }
            return Ok(());
        }
        let manifest = if let Some(path) = &self.manifest {
            if !path.is_absolute() {
                return Err("Use an absolute manifest path".into());
            }
            crate::manifest::load_manifest(path)?
        } else {
            let mut manifest: crate::manifest::Manifest =
                serde_json::from_str(crate::manifest::BUNDLED_JSON).map_err(err)?;
            manifest.resolve_native_bundles()?;
            manifest
        };
        let native = manifest
            .services
            .iter()
            .find(|s| s.id == self.service.id())
            .and_then(|s| s.native.as_ref())
            .ok_or("The manifest lacks this native service")?;
        let platform = runtime::current_platform();
        let packages = native
            .platforms
            .get(platform)
            .ok_or_else(|| format!("No published native package for {platform}"))?;
        let backend = packages.backends.get(self.backend_name()).ok_or_else(|| {
            format!(
                "No published {} package for {platform}; Linux previews currently offer Vulkan",
                self.backend_name()
            )
        })?;
        let resolve = |package: &crate::manifest::NativePackage| -> Result<Package, String> {
            if !crate::download::is_sha256(&package.sha256) {
                return Err("Manifest packages need a full SHA256".into());
            }
            if !package.requires.is_empty() {
                return Err("This headless installer requires system GPU libraries; shared runtime packs are not supported yet".into());
            }
            let url = reqwest::Url::parse(&package.url).map_err(err)?;
            if url.scheme() != "https" {
                return Err("Manifest runtime downloads require HTTPS".into());
            }
            Ok(Package {
                path: self
                    .runtime_root
                    .join("downloads")
                    .join(format!("{}.zip", package.sha256.to_ascii_lowercase())),
                sha256: package.sha256.clone(),
                url: Some(package.url.clone()),
            })
        };
        self.core = resolve(&packages.core)?;
        self.backend = resolve(backend)?;
        self.expected_version = Some(native.version.clone());
        Ok(())
    }
    fn validate(&self) -> Result<(), String> {
        for path in [
            &self.runtime_root,
            &self.models_dir,
            &self.core.path,
            &self.backend.path,
        ] {
            if !path.is_absolute() {
                return Err(format!("Use an absolute path: {}", path.display()));
            }
            if path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err("Use normalized paths without '..'".into());
            }
        }
        if self.runtime_root.parent().is_none() || self.runtime_root == self.models_dir {
            return Err(
                "Use a dedicated runtime root, separate from the existing models directory".into(),
            );
        }
        if !self.selected_models.is_empty() {
            if !matches!(self.service, Service::Gary | Service::Carey)
                || self.models_dir != self.managed_models_dir()
            {
                return Err(
                    "Multiple model selections require managed Gary or Carey storage.".into(),
                );
            }
            if self.selected_models.get(&self.model) != Some(&self.encoding) {
                return Err(
                    "The primary model and encoding must match its selected_models entry.".into(),
                );
            }
            self.model_files()?;
        }
        if self.public_port == 0 {
            return Err("Use a nonzero public loopback port".into());
        }
        if !(self.device.starts_with("CUDA") || self.device.starts_with("Vulkan"))
            || self.threads == 0
            || self.threads > 256
        {
            return Err("Select an exact CUDA/Vulkan GPU name and 1..=256 threads".into());
        }
        for package in [&self.core, &self.backend] {
            if package.sha256.len() != 64 || !package.sha256.bytes().all(|c| c.is_ascii_hexdigit())
            {
                return Err("Both archives require their full SHA256".into());
            }
        }
        Ok(())
    }
    fn backend_name(&self) -> &str {
        if self.device.starts_with("CUDA") {
            "cuda"
        } else {
            "vulkan"
        }
    }
    fn installed_dir(&self) -> PathBuf {
        self.runtime_root.join("runtimes").join(format!(
            "{}-{}-{}",
            self.service.package(),
            self.core.sha256.to_ascii_lowercase(),
            self.backend.sha256.to_ascii_lowercase()
        ))
    }
    fn launch(&self, directory: PathBuf, native_port: u16) -> NativeLaunch {
        let install = NativeInstall {
            dir: directory,
            backend: self.device.clone(),
            runtimes: Vec::new(),
            version: None,
            fallback_reason: None,
        };
        if self.service != Service::Sa3 {
            let mut args = vec![
                "--host".into(),
                "127.0.0.1".into(),
                "--port".into(),
                native_port.to_string(),
                "--device".into(),
                self.device.clone(),
                if self.service == Service::Carey {
                    "--models"
                } else {
                    "--models-dir"
                }
                .into(),
                self.models_dir.to_string_lossy().into_owned(),
            ];
            match self.service {
                Service::StableAudio | Service::Foundation => args.extend([
                    "--model".into(),
                    self.model.clone(),
                    "--encoding".into(),
                    self.encoding.clone(),
                ]),
                Service::Yuey => args.extend(["--encoding".into(), self.encoding.clone()]),
                Service::Carey => args.extend([
                    "--adapters".into(),
                    self.runtime_root
                        .join("adapters")
                        .to_string_lossy()
                        .into_owned(),
                ]),
                Service::Gary | Service::Melodyflow => args.extend([
                    "--adapters-dir".into(),
                    self.runtime_root
                        .join("adapters")
                        .to_string_lossy()
                        .into_owned(),
                ]),
                Service::Sa3 => unreachable!(),
            }
            return NativeLaunch::new(
                &self.runtime_root,
                self.service.executable(),
                &install,
                args,
                vec![
                    (
                        "TMPDIR".into(),
                        self.runtime_root.join("tmp").to_string_lossy().into_owned(),
                    ),
                    (
                        "XDG_CACHE_HOME".into(),
                        self.runtime_root
                            .join("cache")
                            .to_string_lossy()
                            .into_owned(),
                    ),
                ],
                &[],
            );
        }
        NativeLaunch::new(
            &self.runtime_root,
            "sa3-server.exe",
            &install,
            vec![
                "--host".into(),
                "127.0.0.1".into(),
                "--port".into(),
                native_port.to_string(),
                "--models-dir".into(),
                self.models_dir.to_string_lossy().into(),
                "--model".into(),
                self.model.clone(),
                "--encoding".into(),
                self.encoding.clone(),
                "--t5-encoding".into(),
                self.t5_encoding.clone(),
                "--ae-encoding".into(),
                self.ae_encoding.clone(),
                "--threads".into(),
                self.threads.to_string(),
            ],
            vec![
                ("SA3_DEVICE".into(), self.device.clone()),
                (
                    "SA3_ADAPTERS_DIR".into(),
                    self.runtime_root.join("adapters").to_string_lossy().into(),
                ),
                (
                    "SA3_PROMPTS_DIR".into(),
                    self.runtime_root.join("prompts").to_string_lossy().into(),
                ),
                (
                    "TMPDIR".into(),
                    self.runtime_root.join("tmp").to_string_lossy().into(),
                ),
                (
                    "TMP".into(),
                    self.runtime_root.join("tmp").to_string_lossy().into(),
                ),
                (
                    "TEMP".into(),
                    self.runtime_root.join("tmp").to_string_lossy().into(),
                ),
                (
                    "XDG_CACHE_HOME".into(),
                    self.runtime_root.join("cache").to_string_lossy().into(),
                ),
                (
                    "CUDA_CACHE_PATH".into(),
                    self.runtime_root
                        .join("cache/cuda")
                        .to_string_lossy()
                        .into(),
                ),
                (
                    "__GL_SHADER_DISK_CACHE_PATH".into(),
                    self.runtime_root
                        .join("cache/nvidia")
                        .to_string_lossy()
                        .into(),
                ),
            ],
            &[],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(root: PathBuf) -> Config {
        let absolute = std::env::temp_dir().canonicalize().unwrap();
        Config {
            service: Service::Sa3,
            runtime_root: root,
            models_dir: absolute.join("existing-models"),
            core: Package {
                path: absolute.join("core.zip"),
                sha256: "a".repeat(64),
                url: None,
            },
            backend: Package {
                path: absolute.join("vulkan.zip"),
                sha256: "b".repeat(64),
                url: None,
            },
            device: "Vulkan0".into(),
            public_port: 28101,
            model: "medium".into(),
            encoding: "f16".into(),
            selected_models: Default::default(),
            t5_encoding: f16(),
            ae_encoding: f16(),
            threads: 4,
            defaults: Map::new(),
            manifest: None,
            expected_version: None,
        }
    }
    #[test]
    fn refuses_unowned_data_and_prevents_concurrent_owners() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("gary-headless-{}", serial()));
        std::fs::create_dir(&root).unwrap();
        let config = config(root.clone());
        std::fs::write(root.join("keep"), "user data").unwrap();
        assert!(owned_root(&config).is_err());
        assert_eq!(
            std::fs::read_to_string(root.join("keep")).unwrap(),
            "user data"
        );
        std::fs::remove_file(root.join("keep")).unwrap();
        owned_root(&config).unwrap();
        let lease = lock(&config).unwrap();
        assert!(active(&config).unwrap());
        assert!(lock(&config).is_err());
        drop(lease);
        assert!(!active(&config).unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rejects_corrupt_archive_and_cpu_generation_configuration() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("gary-headless-{}", serial()));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("archive.zip");
        std::fs::write(&path, b"corrupt").unwrap();
        assert!(verify_archive(&Package {
            path,
            sha256: "a".repeat(64),
            url: None,
        })
        .is_err());
        let mut config = config(root.clone());
        config.validate().unwrap();
        config.device = "CPU".into();
        assert!(config.validate().is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_resolution_selects_only_pinned_gpu_packages() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("gary-manifest-{}", serial()));
        std::fs::create_dir(&root).unwrap();
        let manifest = root.join("services.json");
        let mut body = json!({"services":[{"id":"sa3","displayName":"SA3","port":8005,"workingDir":"services/sa3",
            "native":{"bundle":"sa3","executable":"sa3-server.exe"}}],
            "nativeBundles":{"sa3":{"version":"v1.2.3","platforms":{runtime::current_platform():{
                "core":{"url":"https://example.invalid/core.zip","sha256":"a".repeat(64)},
                "backends":{"vulkan":{"url":"https://example.invalid/vulkan.zip","sha256":"b".repeat(64)}},"prefer":["vulkan"]}}}}});
        std::fs::write(&manifest, body.to_string()).unwrap();
        let mut cfg = config(root.join("data"));
        cfg.core = Package::default();
        cfg.backend = Package::default();
        cfg.manifest = Some(manifest.clone());
        let mut resolved = cfg.clone();
        resolved.resolve_packages().unwrap();
        resolved.validate().unwrap();
        assert_eq!(resolved.expected_version.as_deref(), Some("v1.2.3"));
        assert_eq!(resolved.core.sha256, "a".repeat(64));
        assert_eq!(
            resolved.backend.url.as_deref(),
            Some("https://example.invalid/vulkan.zip")
        );
        assert!(resolved.core.path.starts_with(root.join("data/downloads")));
        let mut cuda = cfg.clone();
        cuda.device = "CUDA0".into();
        assert!(cuda.resolve_packages().is_err());
        body["nativeBundles"]["sa3"]["platforms"][runtime::current_platform()]["core"]["sha256"] =
            json!("");
        std::fs::write(&manifest, body.to_string()).unwrap();
        assert!(cfg.resolve_packages().is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[tokio::test]
    async fn expected_session_fences_stop_and_uses_no_pid_signals() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("gary-stop-fence-{}", serial()));
        let config = config(root.clone());
        owned_root(&config).unwrap();
        let lease = lock(&config).unwrap();
        let session = lease.session.clone();
        write_json(
            &root.join("state.json"),
            &State {
                service: config.service,
                session: session.clone(),
                pid: std::process::id(),
                native_pid: std::process::id(),
                status: "running".into(),
                public_port: config.public_port,
                native_port: 28102,
                device: config.device.clone(),
            },
        )
        .unwrap();
        assert!(matches!(
            stop(&config, Some("a stale session")).await,
            Err(StopError::SessionMismatch { .. })
        ));
        assert!(active(&config).unwrap());
        assert!(!root.join("stop.request").exists());
        let request = root.join("stop.request");
        let expected = session.clone();
        let owner = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            // Match the serving loop: creation can precede the nonce write.
            while std::fs::read_to_string(&request).ok().as_ref() != Some(&expected) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "Timed out waiting for the matching stop request"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            drop(lease);
        });
        assert_eq!(
            stop(&config, Some(&session)).await.unwrap()["stopped"],
            true
        );
        owner.join().unwrap();
        assert!(!active(&config).unwrap());
        assert_eq!(
            stop(&config, Some(&session)).await.unwrap()["stopped"],
            true
        );
        assert_eq!(
            status(&config).await.unwrap()["running"],
            false,
            "stale state does not imply a live owner"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn readable_owner_record_is_required_and_legacy_unix_nonce_is_supported() {
        let root = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("gary-owner-record-{}", serial()));
        let config = config(root.clone());
        owned_root(&config).unwrap();
        let lease = lock(&config).unwrap();
        assert_eq!(
            owner_session(&config).as_deref(),
            Some(lease.session.as_str())
        );
        std::fs::write(root.join("owner.json"), b"invalid").unwrap();
        assert_eq!(
            owner_session(&config),
            None,
            "malformed owner must not fall back to an old nonce"
        );
        std::fs::remove_file(root.join("owner.json")).unwrap();
        #[cfg(unix)]
        assert_eq!(
            owner_session(&config).as_deref(),
            Some(lease.session.as_str())
        );
        drop(lease);
        std::fs::remove_dir_all(root).unwrap();
    }
}
fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}
fn serial() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

const MARKER: &str = ".gary4local-headless";
fn owned_root(config: &Config) -> Result<(), String> {
    let root = &config.runtime_root;
    if root.exists()
        && !root.join(MARKER).is_file()
        && std::fs::read_dir(root).map_err(err)?.next().is_some()
    {
        return Err(format!(
            "Refusing a nonempty, unowned runtime root: {}",
            root.display()
        ));
    }
    if root.exists() {
        reject_symlinks(root)?;
    }
    std::fs::create_dir_all(root).map_err(err)?;
    if !root.join(MARKER).exists() {
        std::fs::write(root.join(MARKER), "gary4local-server v1\n").map_err(err)?;
    }
    for name in [
        "runtimes",
        "logs",
        "tmp",
        "cache/cuda",
        "cache/nvidia",
        "adapters",
        "prompts",
        "uploads",
    ] {
        std::fs::create_dir_all(root.join(name)).map_err(err)?;
    }
    Ok(())
}

fn reject_symlinks(root: &Path) -> Result<(), String> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let metadata = std::fs::symlink_metadata(&path).map_err(err)?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "Runtime-owned paths cannot be symlinks: {}",
                path.display()
            ));
        }
        if metadata.is_dir() {
            for entry in std::fs::read_dir(path).map_err(err)? {
                pending.push(entry.map_err(err)?.path());
            }
        }
    }
    Ok(())
}

/// Inspect the same ownership boundary as install/serve without creating or
/// adopting anything. Status/stop must not follow an unrelated root's links.
fn inspect_root(config: &Config) -> Result<&'static str, String> {
    let root = &config.runtime_root;
    match std::fs::symlink_metadata(root) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok("missing"),
        Err(error) => return Err(err(error)),
        Ok(metadata) if !metadata.is_dir() || metadata.file_type().is_symlink() => {
            return Err("Runtime root must be a real directory, not a symlink".into())
        }
        _ => {}
    }
    reject_symlinks(root)?;
    if root.join(MARKER).is_file() {
        return Ok("owned");
    }
    if std::fs::read_dir(root).map_err(err)?.next().is_none() {
        return Ok("empty");
    }
    Err(format!(
        "Refusing a nonempty, unowned runtime root: {}",
        root.display()
    ))
}

pub fn model_plan(config: &Config) -> Result<Value, String> {
    let files = config.model_files()?;
    let total: u64 = files.iter().map(|file| file.bytes).sum();
    let missing: Vec<_> = files
        .iter()
        .filter(|file| {
            !std::fs::metadata(config.models_dir.join(&file.filename))
                .is_ok_and(|m| m.is_file() && m.len() == file.bytes)
        })
        .collect();
    let missing_bytes: u64 = missing.iter().map(|file| file.bytes).sum();
    let missing_names: Vec<_> = missing.iter().map(|file| &file.filename).collect();
    Ok(
        json!({"service":config.service,"model":config.model,"missing":missing_names,"missing_bytes":missing_bytes,"models_dir":config.models_dir, "managed": config.models_dir == config.managed_models_dir(),
        "encoding":config.encoding,"selected_models":config.selected_models,"bytes":total,"files":files}),
    )
}

pub async fn download_models(config: &Config) -> Result<Value, String> {
    let files = config.model_files()?;
    if config.models_dir != config.managed_models_dir() {
        return Err("This config uses external weights. To enable managed downloads, omit models_dir in your config.".into());
    }
    inspect_root(config)?;
    owned_root(config)?;
    let _lease = lock(config)?;
    fetch_model_files(config, files).await?;
    model_plan(config)
}

async fn fetch_model_files(
    config: &Config,
    files: Vec<&crate::model_catalog::PinnedHfFile>,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    for file in files {
        eprintln!(
            "Checking {} ({:.2} GiB)",
            file.filename,
            file.bytes as f64 / 1073741824.0
        );
        let mut last = None;
        crate::download::download_verified(
            &client,
            &file.url(),
            Default::default(),
            &file.sha256,
            &config.models_dir.join(&file.filename),
            &mut |done, _| {
                let percent = done.saturating_mul(100) / file.bytes;
                if last != Some(percent / 10) {
                    eprintln!("{}: {percent}%", file.filename);
                    last = Some(percent / 10);
                }
            },
        )
        .await?;
    }
    Ok(())
}

/// Download the selected checkpoint with its shared components, then commit
/// the preference. A failed download or changed config leaves the previous
/// choice intact. The lease prevents racing serve/install/download operations.
pub async fn select_model(
    path: &Path,
    original: &[u8],
    previous: &Config,
    candidate: &Config,
    download: bool,
) -> Result<Config, String> {
    if previous.models_dir != previous.managed_models_dir() {
        return Err("Use managed storage to select a model from the catalog.".into());
    }
    let files = candidate.model_files()?;
    let mut updated = previous.clone();
    updated.model = candidate.model.clone();
    updated.encoding = candidate.encoding.clone();
    if matches!(previous.service, Service::Gary | Service::Carey) {
        if updated.selected_models.is_empty()
            && model_plan(previous).is_ok_and(|plan| plan["missing_bytes"] == 0)
        {
            updated
                .selected_models
                .insert(previous.model.clone(), previous.encoding.clone());
        }
        updated
            .selected_models
            .insert(updated.model.clone(), updated.encoding.clone());
    }
    updated.validate()?;
    inspect_root(previous)?;
    owned_root(previous)?;
    let _lease = lock(previous)?;
    if !std::fs::symlink_metadata(path).map_err(err)?.is_file() {
        return Err("Model selection requires a regular service config file.".into());
    }
    if std::fs::read(path).map_err(err)? != original {
        return Err("The service config changed. Reopen /models and try again.".into());
    }
    if download {
        fetch_model_files(previous, files).await?;
    } else {
        for file in files {
            if crate::download::sha256_file(&previous.models_dir.join(&file.filename)).await?
                != file.sha256
            {
                return Err(format!(
                    "{} failed verification. Use verified model weights and reopen /models.",
                    file.filename
                ));
            }
        }
    }
    if std::fs::read(path).map_err(err)? != original {
        return Err("The service config changed. Reopen /models and try again.".into());
    }
    let mut body: Value = serde_json::from_slice(original).map_err(err)?;
    body["model"] = json!(updated.model);
    body["encoding"] = json!(updated.encoding);
    if !updated.selected_models.is_empty() {
        body["selected_models"] = json!(updated.selected_models);
    }
    write_json(path, &body)?;
    Ok(updated)
}

#[derive(PartialEq, Eq)]
pub struct ModelRemoval {
    files: Vec<(PathBuf, u64)>,
}

impl ModelRemoval {
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|(_, bytes)| bytes).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

pub fn model_removal(config: &Config, model: &str, encoding: &str) -> Result<ModelRemoval, String> {
    if config.models_dir != config.managed_models_dir() {
        return Err("Model removal requires managed storage.".into());
    }
    inspect_root(config)?;
    if !crate::model_catalog::variants(config.service)
        .iter()
        .any(|&(name, tier)| name == model && tier.eq_ignore_ascii_case(encoding))
    {
        return Err("Unknown catalog variant.".into());
    }
    let mut files = Vec::new();
    for file in crate::model_catalog::removable_variant_files(
        config.service,
        model,
        encoding,
        &config.models_dir,
    )? {
        let weight = config.models_dir.join(&file.filename);
        let partial = crate::download::partial_path(&weight);
        for path in [weight, partial] {
            match std::fs::symlink_metadata(&path) {
                Ok(meta) if meta.is_file() => files.push((path, meta.len())),
                Ok(_) => {
                    return Err(format!(
                        "Cannot remove a non-regular weight file at {}",
                        path.display()
                    ))
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(err(e)),
            }
        }
    }
    Ok(ModelRemoval { files })
}

pub fn remove_model(
    path: &Path,
    original: &[u8],
    config: &Config,
    model: &str,
    encoding: &str,
    plan: &ModelRemoval,
) -> Result<(), String> {
    inspect_root(config)?;
    let _lease = lock(config)?;
    if !std::fs::symlink_metadata(path).map_err(err)?.is_file()
        || std::fs::read(path).map_err(err)? != original
        || model_removal(config, model, encoding)? != *plan
    {
        return Err("The model inventory or config changed. Reopen /models and try again.".into());
    }
    let mut updated = config.clone();
    updated
        .selected_models
        .retain(|name, tier| name != model || !tier.eq_ignore_ascii_case(encoding));
    if updated.model == model && updated.encoding.eq_ignore_ascii_case(encoding) {
        if let Some((name, tier)) = updated.selected_models.iter().next() {
            updated.model = name.clone();
            updated.encoding = tier.clone();
        }
    }
    updated.validate()?;
    let mut body: Value = serde_json::from_slice(original).map_err(err)?;
    body["model"] = json!(updated.model);
    body["encoding"] = json!(updated.encoding);
    if updated.selected_models.is_empty() {
        body.as_object_mut()
            .expect("validated config")
            .remove("selected_models");
    } else {
        body["selected_models"] = json!(updated.selected_models);
    }
    // Keep originals recoverable until every move and the config update succeed.
    let staged = config
        .runtime_root
        .join(format!(".remove-model-{}", serial()));
    std::fs::create_dir(&staged).map_err(err)?;
    let mut moved = Vec::new();
    let result = (|| -> Result<(), String> {
        for (original, _) in &plan.files {
            let relative = original.strip_prefix(&config.models_dir).map_err(err)?;
            let saved = staged.join(relative);
            if let Some(parent) = saved.parent() {
                std::fs::create_dir_all(parent).map_err(err)?;
            }
            std::fs::rename(original, &saved).map_err(err)?;
            moved.push((original, saved));
        }
        write_json(path, &body)
    })();
    if let Err(error) = result {
        let mut failures = Vec::new();
        for (original, saved) in moved.into_iter().rev() {
            if let Err(e) = std::fs::rename(&saved, original) {
                failures.push(format!("{} → {}: {e}", saved.display(), original.display()));
            }
        }
        if failures.is_empty() {
            let _ = std::fs::remove_dir_all(&staged);
            return Err(error);
        }
        return Err(format!(
            "{error}\nCould not restore all model files:\n{}",
            failures.join("\n")
        ));
    }
    std::fs::remove_dir_all(&staged).map_err(|e| {
        format!(
            "Removed model from the catalog. Could not clear {}: {e}",
            staged.display()
        )
    })
}

/// Filesystem/ownership/port inspection only. No adoption, downloads, GPU probe,
/// model load or persistent listener. Size checks are not a model hash proof.
pub async fn doctor(config: &Config) -> Result<Value, String> {
    let mut checks = Vec::new();
    let root_state = inspect_root(config);
    let root_safe = root_state.is_ok();
    checks.push(match &root_state {
        Ok(state) => json!({"code":"runtime_root","status":"ok","state":state,"path":config.runtime_root}),
        Err(message) => json!({"code":"runtime_root","status":"error","message":message,"hint":"Choose an empty dedicated runtime root; do not remove unrelated data or forge an ownership marker."}),
    });
    let service = if root_safe {
        status(config).await?
    } else {
        json!({"running":false,"operation_active":false})
    };
    let running = service["running"] == true;
    let busy = service["operation_active"] == true;
    checks.push(json!({"code":"owner_lock","status":if busy {"error"} else {"ok"},
        "message":if running {"This owned serving session is already running; use status or stop, not a second serve."}
            else if busy {"Another operation owns this root; wait for it to finish."} else {"No active owner."}}));

    let runtime = config.installed_dir();
    let installed = if root_safe && runtime.is_dir() {
        let result = metadata(config, &runtime).and_then(|info| {
            let stamp: Value = serde_json::from_slice(&std::fs::read(runtime.join("package-checksums.json")).map_err(err)?).map_err(err)?;
            if stamp != json!({"core":config.core.sha256.to_ascii_lowercase(),"backend":config.backend.sha256.to_ascii_lowercase()}) {
                return Err("Runtime package provenance does not match this config".into());
            }
            if !runtime.join(runtime::executable_name(config.service.executable())).is_file() { return Err("Installed runtime is missing its native server".into()); }
            Ok(info)
        });
        checks.push(match result {
            Ok(info) => json!({"code":"installed_runtime","status":"ok","directory":runtime,"build":info}),
            Err(message) => json!({"code":"installed_runtime","status":"error","message":message,"hint":"Inspect this owned runtime and run install to check the candidate. No files are changed by doctor."}),
        });
        checks.last().unwrap()["status"] == "ok"
    } else {
        checks.push(json!({"code":"installed_runtime","status":"error","directory":runtime,"message":"A matching runtime is not installed or could not be safely inspected.","hint":"After resolving ownership issues, explicitly run install --config FILE."}));
        false
    };

    if !config.models_dir.is_dir() {
        checks.push(json!({"code":"models_directory","status":"error","path":config.models_dir,"hint":"Check your configured model directory. Managed storage can be populated with download-models."}));
    } else {
        checks.push(json!({"code":"models_directory","status":"ok","path":config.models_dir}));
        if let Ok(expected) = config.model_files() {
            let mut files = Vec::new();
            for file in expected {
                let path = config.models_dir.join(&file.filename);
                let metadata = std::fs::metadata(&path)
                    .ok()
                    .filter(|metadata| metadata.is_file());
                files.push(json!({"path":path,"present":metadata.as_ref().is_some_and(|metadata| metadata.len()>0),
                    "actual_bytes":metadata.as_ref().map(|metadata| metadata.len()),"expected_bytes":file.bytes,
                    "sha256":file.sha256,"repo":file.repo,"revision":file.revision}));
            }
            let missing = files.iter().any(|file| file["present"] != true);
            let size_mismatch = files
                .iter()
                .any(|file| file["actual_bytes"] != file["expected_bytes"]);
            checks.push(json!({"code":"model_files","status":if missing || (size_mismatch && config.models_dir==config.managed_models_dir()) {"error"} else if size_mismatch {"warning"} else {"ok"},
                "files":files,"verification":"presence and size only; contents were not hashed or loaded",
                "message":if missing {"Required model files are missing. Provide existing weights or complete model preparation separately."}
                    else if size_mismatch {"This model layout differs from the pinned catalog; compatibility is unverified."}
                    else {"Expected model components are present."}}));
        } else {
            checks.push(json!({"code":"model_files","status":"warning","message":"This model/component encoding combination has no catalog preflight. Native startup remains the compatibility check; no weights have been loaded."}));
        }
    }
    if root_safe && !busy {
        checks.push(match runtime::bind_loopback(config.public_port) {
            Ok(socket) => { drop(socket); json!({"code":"public_port","status":"ok","port":config.public_port}) },
            Err(error) => json!({"code":"public_port","status":"error","port":config.public_port,"message":error.to_string(),"hint":"Choose an unused client port or stop its owner deliberately. doctor does not stop other processes."}),
        });
    }
    checks.push(json!({"code":"gpu_probe","status":"not_checked","message":"doctor never probes the GPU or loads a model. install probes the selected GPU; serve verifies model startup."}));
    let can_attempt_serve = checks.iter().all(|check| check["status"] != "error");
    let mut next_actions = Vec::new();
    if !root_safe {
        next_actions.push(json!({"action":"edit_config","field":"runtime_root","reason":"ownership boundary must be resolved first"}));
    } else if busy {
        next_actions.push(json!({"action":if running {"status_or_stop"} else {"wait_for_owner"},"automatic":false}));
    } else if !installed {
        next_actions.push(json!({"action":"install","requires_explicit_action":true}));
    }
    if checks.iter().any(|check| {
        matches!(
            check["code"].as_str(),
            Some("models_directory" | "model_files")
        ) && check["status"] == "error"
    }) {
        let managed =
            config.models_dir == config.managed_models_dir() && config.model_files().is_ok();
        next_actions.push(json!({"action":if managed {"download-models"} else {"provide_required_model_files"},"model_download_command_available":managed}));
    }
    if checks
        .iter()
        .any(|check| check["code"] == "public_port" && check["status"] == "error")
    {
        next_actions.push(
            json!({"action":"edit_config","field":"public_port","reason":"port is occupied"}),
        );
    }
    if can_attempt_serve {
        next_actions.push(json!({"action":"serve","requires_explicit_action":true,"readiness_reported_only_after_model_load":config.service==Service::Sa3}));
    }
    Ok(
        json!({"can_attempt_serve":can_attempt_serve,"checks":checks,"next_actions":next_actions,
        "service":service,"resolved":config.summary(),"read_only":true,
        "limitations":["Model contents and GPU execution were not verified.","Port availability is a snapshot, not a reservation."]}),
    )
}
struct Lease {
    _file: File,
    session: String,
}
impl Drop for Lease {
    fn drop(&mut self) {
        // A concurrent fork can briefly inherit this open file description
        // before close-on-exec. Explicit unlock releases ownership immediately
        // rather than waiting for every inherited descriptor to close.
        let _ = self._file.unlock();
    }
}
fn lock(config: &Config) -> Result<Lease, String> {
    let mut lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(config.runtime_root.join("manager.lock"))
        .map_err(err)?;
    lock.try_lock()
        .map_err(|_| "This runtime root already has an active operation".to_string())?;
    let session = serial();
    // Write through the locked inode; replacing the lock file would break
    // mutual exclusion. The nonce distinguishes a serving session from a
    // new install/start operation holding the same root after a crash.
    lock.set_len(0).map_err(err)?;
    lock.write_all(
        serde_json::to_string(&json!({"session":session}))
            .map_err(err)?
            .as_bytes(),
    )
    .map_err(err)?;
    lock.sync_data().map_err(err)?;
    // Windows byte-range locks also deny reads through a second handle. Keep
    // the readable nonce separate from the OS-held lock; the matching nonce
    // still distinguishes a new owner from stale state after a crash.
    write_json(
        &config.runtime_root.join("owner.json"),
        &json!({"session":session}),
    )?;
    Ok(Lease {
        _file: lock,
        session,
    })
}
fn owner_session(config: &Config) -> Option<String> {
    let path = config.runtime_root.join("owner.json");
    let raw = match std::fs::read(path) {
        Ok(raw) => raw,
        // Compatibility with older Unix hosts that stored the readable nonce
        // only inside manager.lock. An invalid new record must not fall back.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::read(config.runtime_root.join("manager.lock")).ok()?
        }
        Err(_) => return None,
    };
    let body: Value = serde_json::from_slice(&raw).ok()?;
    body["session"].as_str().map(str::to_owned)
}
fn active(config: &Config) -> Result<bool, String> {
    let path = config.runtime_root.join("manager.lock");
    if !path.exists() {
        return Ok(false);
    }
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(err)?;
    match lock.try_lock() {
        Ok(()) => {
            lock.unlock().map_err(err)?;
            Ok(false)
        }
        Err(std::fs::TryLockError::WouldBlock) => Ok(true),
        Err(error) => Err(err(error)),
    }
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let temporary = path.with_extension(format!("{}.tmp", serial()));
    let bytes = serde_json::to_vec_pretty(value).map_err(err)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temporary).map_err(err)?;
    let written = file
        .write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all());
    drop(file);
    let result = written.and_then(|_| std::fs::rename(&temporary, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(err)
}
fn hash_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(err)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(err)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn verify_archive(package: &Package) -> Result<(), String> {
    if hash_file(&package.path)? != package.sha256.to_ascii_lowercase() {
        return Err(format!("SHA256 mismatch: {}", package.path.display()));
    }
    Ok(())
}
fn clean_command_env(command: &mut std::process::Command, launch: &NativeLaunch) {
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if ["AC_", "ACE_", "YUE2_"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            || name.starts_with("SA3_")
            || name.starts_with("SAT_")
            || name.starts_with("GGML_")
            || name == "LD_LIBRARY_PATH"
        {
            command.env_remove(key);
        }
    }
    command.envs(launch.env.iter().map(|(k, v)| (k, v)));
}

async fn probe(config: &Config, dir: &Path) -> Result<Value, String> {
    let launch = config.launch(dir.to_path_buf(), 0);
    // The installation probe never starts a listener or loads weights.
    let mut command = std::process::Command::new(&launch.program);
    command.arg("--props").current_dir(dir);
    if config.service == Service::Carey {
        command.args(["--device", &config.device]);
    }
    clean_command_env(&mut command, &launch);
    let mut command = tokio::process::Command::from(command);
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    crate::workload_job::configure_tokio_command(&mut command);
    let mut child = command.spawn().map_err(err)?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    let output = tokio::time::timeout(Duration::from_secs(30), child.wait_with_output())
        .await
        .map_err(|_| "GPU probe timed out".to_string())?
        .map_err(err)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = if stderr.contains("createDevice")
            || stderr.contains("InitializationFailedError")
        {
            "Vulkan could not initialize the GPU. If another GPU workload is active, try again after it finishes."
        } else {
            "The GPU probe exited before it could confirm the device."
        };
        let path = config.log_path();
        let result = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .and_then(|mut log| {
                writeln!(
                    log,
                    "\nGPU probe --props failed at {:?} ({}):\n{}\n{}",
                    SystemTime::now(),
                    output.status,
                    stderr,
                    stdout
                )
            });
        let message = format!(
            "{} probe failed ({}). {detail}",
            config.device, output.status
        );
        return Err(match result {
            Ok(()) => format!("{message} Details: {}", path.display()),
            Err(error) => format!(
                "{message} Could not write {}: {error}\n{stderr}",
                path.display()
            ),
        });
    }
    let props: Value = serde_json::from_slice(&output.stdout).map_err(err)?;
    if props["service"] != config.service.probe_id() || config.selected_gpu(&props).is_none() {
        return Err(format!(
            "Selected GPU {} is unavailable in the installed runtime",
            config.device
        ));
    }
    Ok(props)
}
fn metadata(config: &Config, dir: &Path) -> Result<Value, String> {
    let info: Value =
        serde_json::from_slice(&std::fs::read(dir.join("BUILD-INFO.json")).map_err(err)?)
            .map_err(err)?;
    if info["service"] != config.service.package()
        || info["platform"] != runtime::current_platform()
        || info["dirty"] != false
        || info["tests_run"] != true
        || info["commit"].as_str().is_none_or(str::is_empty)
        || info["ggml_commit"].as_str().is_none_or(str::is_empty)
        || config
            .expected_version
            .as_ref()
            .is_some_and(|v| info["version"] != v.as_str())
    {
        return Err(
            "Package metadata does not identify a clean, tested engine build for this platform"
                .into(),
        );
    }
    Ok(info)
}

async fn acquire(package: &Package) -> Result<(), String> {
    if let Some(url) = &package.url {
        eprintln!("Downloading verified runtime package...");
        let client = reqwest::Client::builder().build().map_err(err)?;
        crate::download::download_verified(
            &client,
            url,
            reqwest::header::HeaderMap::new(),
            &package.sha256,
            &package.path,
            &mut |_, _| {},
        )
        .await?;
    }
    verify_archive(package)
}

pub async fn install(config: &Config) -> Result<Value, String> {
    config.validate()?;
    owned_root(config)?;
    let _lock = lock(config)?;
    acquire(&config.core).await?;
    acquire(&config.backend).await?;
    let target = config.installed_dir();
    if target.exists() {
        let stamp: Value = serde_json::from_slice(
            &std::fs::read(target.join("package-checksums.json")).map_err(err)?,
        )
        .map_err(err)?;
        if stamp
            != json!({"core":config.core.sha256.to_ascii_lowercase(),"backend":config.backend.sha256.to_ascii_lowercase()})
        {
            return Err("Existing runtime has different package provenance".into());
        }
        metadata(config, &target)?;
        let props = probe(config, &target).await?;
        write_json(&target.join("props.json"), &props)?;
    } else {
        let staging = config
            .runtime_root
            .join("runtimes")
            .join(format!(".install-{}", serial()));
        std::fs::create_dir(&staging).map_err(err)?;
        let result=async {
            runtime::extract_zip(&config.core.path,&staging)?;
            runtime::extract_zip(&config.backend.path,&staging)?;
            let info=metadata(config,&staging)?;
            let props=probe(config,&staging).await?;
            write_json(&staging.join("props.json"),&props)?;
            write_json(&staging.join("package-checksums.json"),&json!({"core":config.core.sha256.to_ascii_lowercase(),"backend":config.backend.sha256.to_ascii_lowercase()}))?;
            std::fs::rename(&staging,&target).map_err(err)?;
            Ok::<_,String>(info)
        }.await;
        // Only this freshly created staging directory is owned by this operation.
        if staging.exists() {
            let _ = std::fs::remove_dir_all(&staging);
        }
        result?;
    }
    Ok(json!({"service":config.service,"installed":true,"runtime":target,"backend":config.device}))
}

#[derive(Serialize, Deserialize)]
struct State {
    #[serde(default)]
    service: Service,
    session: String,
    pid: u32,
    native_pid: u32,
    status: String,
    public_port: u16,
    native_port: u16,
    device: String,
}
fn state(config: &Config) -> Result<State, String> {
    serde_json::from_slice(&std::fs::read(config.runtime_root.join("state.json")).map_err(err)?)
        .map_err(err)
}
struct StateGuard<'a> {
    config: &'a Config,
    state: State,
}
impl Drop for StateGuard<'_> {
    fn drop(&mut self) {
        self.state.status = "stopped".into();
        let _ = write_json(&self.config.runtime_root.join("state.json"), &self.state);
    }
}
async fn termination() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=term.recv()=>{} }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

// Keep the client port restartable even when the C++ listener does not set
// SO_REUSEADDR. Passing bytes through preserves uploads, audio and streaming.
async fn forward_native(listener: tokio::net::TcpListener, port: u16) -> std::io::Result<()> {
    let mut connections = tokio::task::JoinSet::new();
    loop {
        tokio::select! {
            accepted=listener.accept()=>{
                let (mut client,_) = accepted?;
                connections.spawn(async move {
                    let mut native=tokio::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST,port)).await?;
                    tokio::io::copy_bidirectional(&mut client,&mut native).await
                });
            }
            Some(_)=connections.join_next(), if !connections.is_empty()=>{}
        }
    }
}

struct ModelDirectory(PathBuf);

impl Drop for ModelDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn selected_model_directory(config: &Config) -> Result<Option<ModelDirectory>, String> {
    if config.selected_models.is_empty() {
        return Ok(None);
    }
    let path = config
        .runtime_root
        .join("tmp")
        .join(format!("selected-models-{}", serial()));
    std::fs::create_dir(&path).map_err(err)?;
    let view = ModelDirectory(path);
    let mut files = config.model_files()?;
    if config.service == Service::Carey {
        files.extend(crate::model_catalog::installed_carey_lms(
            &config.models_dir,
        ));
    }
    for file in files {
        let source = config.models_dir.join(&file.filename);
        let dest = view.0.join(&file.filename);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(err)?;
        }
        std::fs::hard_link(&source, &dest)
            .map_err(|e| format!("Cannot expose selected model {}: {e}", source.display()))?;
    }
    Ok(Some(view))
}

pub enum Sa3Startup {
    Cold,
    PrepareModel,
}

pub async fn serve(
    config: &Config,
    sa3_startup: Sa3Startup,
    ready: impl Fn(Value) -> Result<(), String>,
) -> Result<Value, String> {
    config.validate()?;
    owned_root(config)?;
    let lease = lock(config)?;
    if !config.models_dir.is_dir() {
        return Err("The configured model directory is missing".into());
    }
    if config.service != Service::Sa3 && config.models_dir == config.managed_models_dir() {
        let plan = model_plan(config)?;
        if plan["missing_bytes"] != 0 {
            return Err(format!(
                "Missing or incomplete {} weights. Run gary download-models {}.",
                config.service.id(),
                config.service.id()
            ));
        }
    }
    let dir = config.installed_dir();
    eprintln!(
        "Checking the installed {} runtime and GPU...",
        config.service.id()
    );
    metadata(config, &dir)?;
    probe(config, &dir).await?;
    let public = runtime::bind_loopback(config.public_port).map_err(|e| {
        format!(
            "Cannot bind {} on port {}: {e}",
            config.service.id(),
            config.public_port
        )
    })?;
    let reservation = runtime::bind_loopback(0).map_err(err)?;
    let native_port = reservation.local_addr().map_err(err)?.port();
    let view = selected_model_directory(config)?;
    let mut launch_config = config.clone();
    if let Some(view) = &view {
        launch_config.models_dir = view.0.clone();
    }
    let launch = launch_config.launch(dir, native_port);
    let mut command = launch.std_command();
    clean_command_env(&mut command, &launch);
    // Output paths, temporary files and caches belong to this root, not weights.
    command.current_dir(&config.runtime_root);
    let logfile = File::create(config.log_path()).map_err(err)?;
    command
        .stdout(logfile.try_clone().map_err(err)?)
        .stderr(logfile);
    drop(reservation);
    let mut child = ManagedChild::spawn(command)?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(err)?;
    let native_url = format!("http://127.0.0.1:{native_port}/health");
    let grace = config
        .service
        .definition()
        .health_check
        .as_ref()
        .map(|check| check.startup_grace_seconds as u64)
        .unwrap_or(30);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(grace);
    loop {
        if let Some(exit) = child.0.try_wait().map_err(err)? {
            return Err(format!(
                "Native server exited during startup: {exit}; inspect the service log"
            ));
        }
        if let Ok(response) = client.get(&native_url).send().await {
            if response.status().is_success() {
                let health: Value = response.json().await.map_err(err)?;
                if config.service == Service::Sa3 {
                    for capability in crate::sa3_adapter::REQUIRED_CAPABILITIES {
                        if health["capabilities"][capability] != true {
                            return Err(format!(
                                "Native SA3 lacks the required {capability} control"
                            ));
                        }
                    }
                }
                break;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("Native health check timed out; inspect the service log".into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Scripted serve verifies the model before reporting readiness. The
    // interactive console starts the listener cold, like the desktop app.
    if config.service == Service::Sa3 && matches!(sa3_startup, Sa3Startup::PrepareModel) {
        eprintln!("Preparing SA3 from the local model files...");
        let loaded = client
            .post(format!("http://127.0.0.1:{native_port}/load"))
            .timeout(Duration::from_secs(180))
            .send()
            .await
            .map_err(err)?;
        let load_status = loaded.status();
        let loaded: Value = loaded.json().await.map_err(err)?;
        if !load_status.is_success() || loaded["success"] == false {
            return Err(format!("SA3 model preparation failed: {loaded}"));
        }
    }
    let guard = StateGuard {
        config,
        state: State {
            service: config.service,
            session: lease.session.clone(),
            pid: std::process::id(),
            native_pid: child.0.id(),
            status: "running".into(),
            public_port: config.public_port,
            native_port,
            device: config.device.clone(),
        },
    };
    write_json(&config.runtime_root.join("state.json"), &guard.state)?;
    let mut adapter_tasks = tokio::task::JoinSet::new();
    if config.service == Service::Sa3 {
        let adapter = AdapterListener::from_listener(
            public,
            native_port,
            config.runtime_root.join("uploads"),
            config.defaults.clone(),
            AdapterOptions::default(),
        )?;
        adapter_tasks.spawn(adapter.serve());
    } else {
        public.set_nonblocking(true).map_err(err)?;
        let listener = tokio::net::TcpListener::from_std(public).map_err(err)?;
        adapter_tasks.spawn(forward_native(listener, native_port));
    }
    ready(
        json!({"status":"running","service":config.service,"pid":std::process::id(),"public_port":config.public_port,
        "native_port":native_port,"device":config.device,"session":lease.session,
        "public_url":format!("http://127.0.0.1:{}",config.public_port),
        "log_path":config.log_path()}),
    )?;
    let result=async {
        let mut tick=tokio::time::interval(Duration::from_millis(200));
        let termination=termination(); tokio::pin!(termination);
        loop {
            tokio::select! {
                _=&mut termination => return Ok(()),
                result=adapter_tasks.join_next() => return Err(format!("Client adapter stopped unexpectedly: {result:?}")),
                _=tick.tick() => {
                    if let Some(exit)=child.0.try_wait().map_err(err)? { return Err(format!("Native server exited: {exit}")); }
                    let request=std::fs::read_to_string(config.runtime_root.join("stop.request")).unwrap_or_default();
                    if request==guard.state.session { return Ok(()); }
                }
            }
        }
    }.await;
    adapter_tasks.shutdown().await;
    child.stop();
    drop(guard);
    result?;
    Ok(json!({"stopped":true,"service":config.service,"session":lease.session}))
}

pub async fn status(config: &Config) -> Result<Value, String> {
    inspect_root(config)?;
    let operation_active = active(config)?;
    let state = state(config).ok();
    let running = operation_active
        && state.as_ref().is_some_and(|s| {
            s.service == config.service
                && s.status == "running"
                && owner_session(config).as_deref() == Some(&s.session)
        });
    let health = if running {
        if let Some(state) = &state {
            let client = reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(2))
                .build()
                .map_err(err)?;
            match client
                .get(format!("http://127.0.0.1:{}/health", state.public_port))
                .send()
                .await
            {
                Ok(response) => response.json::<Value>().await.ok(),
                Err(_) => None,
            }
        } else {
            None
        }
    } else {
        None
    };
    Ok(
        json!({"service":config.service,"running":running,"operation_active":operation_active,"state":state,"health":health,
        "public_url":format!("http://127.0.0.1:{}",config.public_port),
        "log_path":config.log_path()}),
    )
}

#[derive(Debug)]
pub enum StopError {
    SessionMismatch { expected: String, actual: String },
    Failed(String),
}
impl From<String> for StopError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}
impl From<&str> for StopError {
    fn from(message: &str) -> Self {
        Self::Failed(message.into())
    }
}
impl std::fmt::Display for StopError {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionMismatch { expected, actual } => write!(
                out,
                "Refusing to stop session {actual}: expected {expected}"
            ),
            Self::Failed(message) => out.write_str(message),
        }
    }
}

pub async fn stop(config: &Config, expected_session: Option<&str>) -> Result<Value, StopError> {
    inspect_root(config)?;
    if !active(config)? {
        return Ok(json!({"service":config.service,"running":false,"stopped":true}));
    }
    let current = state(config)?;
    if current.service != config.service
        || current.status != "running"
        || owner_session(config).as_deref() != Some(&current.session)
    {
        return Err("The active operation is not a serving session".into());
    }
    if let Some(expected) = expected_session {
        if current.session != expected {
            return Err(StopError::SessionMismatch {
                expected: expected.into(),
                actual: current.session,
            });
        }
    }
    let session = current.session;
    std::fs::write(config.runtime_root.join("stop.request"), session.as_bytes()).map_err(err)?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while active(config)? {
        if state(config).is_ok_and(|s| s.session != session) {
            return Err("Service session changed while stopping".into());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err("Stop request timed out; inspect the supervisor log".into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(json!({"service":config.service,"stopped":true,"session":session}))
}
