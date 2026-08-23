use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const ACTIVE_RUNTIME_ROOT_ENV: &str = "GARY4LOCAL_ACTIVE_RUNTIME_ROOT";
pub const RUNTIME_ROOT_OVERRIDE_ENV: &str = "GARY4LOCAL_RUNTIME_ROOT";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStorageInfo {
    pub active_root: String,
    pub configured_root: Option<String>,
    pub startup_root: String,
    pub default_root: String,
    pub legacy_root: String,
    pub config_path: String,
    pub pending_restart: bool,
    pub using_legacy_default: bool,
    pub default_root_is_legacy: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StorageConfig {
    #[serde(default)]
    runtime_root: Option<String>,
}

pub fn roaming_app_data_dir() -> PathBuf {
    let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
        let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
        format!("{}\\AppData\\Roaming", home)
    });
    PathBuf::from(appdata)
}

pub fn local_data_root() -> PathBuf {
    let localappdata = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| {
        let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
        format!("{}\\AppData\\Local", home)
    });
    PathBuf::from(localappdata).join("com.betweentwomidnights.gary4local")
}

pub fn legacy_runtime_root() -> PathBuf {
    roaming_app_data_dir().join("Gary4JUCE")
}

pub fn storage_config_path() -> PathBuf {
    local_data_root().join("storage.json")
}

pub fn installed_default_runtime_root() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| local_data_root()))
        .join("gary4local-data")
}

pub fn models_dir(runtime_root: &Path) -> PathBuf {
    runtime_root.join("models")
}

pub fn cache_dir(runtime_root: &Path) -> PathBuf {
    runtime_root.join("cache")
}

pub fn hf_home_dir(runtime_root: &Path) -> PathBuf {
    models_dir(runtime_root).join("huggingface")
}

pub fn hf_hub_cache_dir(runtime_root: &Path) -> PathBuf {
    hf_home_dir(runtime_root).join("hub")
}

fn user_cache_dir() -> PathBuf {
    if let Some(path) = env_path("XDG_CACHE_HOME") {
        return path;
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache")
}

pub fn effective_hf_home_dir(runtime_root: &Path) -> PathBuf {
    if !paths_equivalent(runtime_root, &legacy_runtime_root()) {
        return hf_home_dir(runtime_root);
    }
    env_path("HF_HOME").unwrap_or_else(|| user_cache_dir().join("huggingface"))
}

pub fn effective_hf_hub_cache_dir(runtime_root: &Path) -> PathBuf {
    if !paths_equivalent(runtime_root, &legacy_runtime_root()) {
        return hf_hub_cache_dir(runtime_root);
    }
    env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|| effective_hf_home_dir(runtime_root).join("hub"))
}

pub fn torch_home_dir(runtime_root: &Path) -> PathBuf {
    models_dir(runtime_root).join("torch")
}

pub fn uv_cache_dir(runtime_root: &Path) -> PathBuf {
    cache_dir(runtime_root).join("uv")
}

pub fn uv_python_install_dir(runtime_root: &Path) -> PathBuf {
    runtime_root.join("python")
}

pub fn pip_cache_dir(runtime_root: &Path) -> PathBuf {
    cache_dir(runtime_root).join("pip")
}

pub fn runtime_env_vars(runtime_root: &Path) -> Vec<(String, String)> {
    let runtime = runtime_root.to_string_lossy().to_string();
    let models = models_dir(runtime_root).to_string_lossy().to_string();
    let cache = cache_dir(runtime_root).to_string_lossy().to_string();
    let hf_home = hf_home_dir(runtime_root).to_string_lossy().to_string();
    let hf_hub = hf_hub_cache_dir(runtime_root).to_string_lossy().to_string();
    let torch_home = torch_home_dir(runtime_root).to_string_lossy().to_string();
    let uv_cache = uv_cache_dir(runtime_root).to_string_lossy().to_string();
    let uv_python = uv_python_install_dir(runtime_root)
        .to_string_lossy()
        .to_string();
    let pip_cache = pip_cache_dir(runtime_root).to_string_lossy().to_string();

    let mut env = vec![
        ("GARY4LOCAL_RUNTIME_DIR".to_string(), runtime.clone()),
        ("GARY_RUNTIME".to_string(), runtime.clone()),
        ("GARY4LOCAL_MODELS_DIR".to_string(), models.clone()),
        ("MODELS_DIR".to_string(), models),
        ("GARY4LOCAL_CACHE_DIR".to_string(), cache.clone()),
        ("UV_CACHE_DIR".to_string(), uv_cache),
        ("UV_PYTHON_INSTALL_DIR".to_string(), uv_python),
        ("PIP_CACHE_DIR".to_string(), pip_cache),
    ];

    // Existing installations historically used each library's standard user
    // cache. Preserve that behavior until the user chooses a new runtime root.
    if !paths_equivalent(runtime_root, &legacy_runtime_root()) {
        env.extend([
            ("XDG_CACHE_HOME".to_string(), cache),
            ("HF_HOME".to_string(), hf_home),
            ("HF_HUB_CACHE".to_string(), hf_hub.clone()),
            ("HUGGINGFACE_HUB_CACHE".to_string(), hf_hub.clone()),
            ("TRANSFORMERS_CACHE".to_string(), hf_hub),
            ("TORCH_HOME".to_string(), torch_home),
        ]);
    }

    env
}

pub fn set_active_runtime_root(path: &Path) {
    std::env::set_var(ACTIVE_RUNTIME_ROOT_ENV, path);
}

pub fn active_runtime_root() -> PathBuf {
    if let Some(path) = env_path(ACTIVE_RUNTIME_ROOT_ENV) {
        return path;
    }
    resolve_startup_runtime_root()
}

pub fn resolve_startup_runtime_root() -> PathBuf {
    if let Some(path) = env_path(RUNTIME_ROOT_OVERRIDE_ENV) {
        return path;
    }

    if let Some(path) = read_storage_config()
        .runtime_root
        .and_then(clean_config_path)
    {
        return path;
    }

    automatic_default_runtime_root()
}

fn automatic_default_runtime_root() -> PathBuf {
    let legacy = legacy_runtime_root();
    if legacy_runtime_root_is_populated(&legacy) {
        legacy
    } else {
        installed_default_runtime_root()
    }
}

pub fn storage_info(active_root: &Path) -> RuntimeStorageInfo {
    let config = read_storage_config();
    let configured_root = config.runtime_root.and_then(clean_config_path);
    let startup_root = resolve_startup_runtime_root();
    let legacy_root = legacy_runtime_root();
    let default_root = automatic_default_runtime_root();

    RuntimeStorageInfo {
        active_root: display_path(active_root),
        configured_root: configured_root.as_ref().map(|path| display_path(path)),
        startup_root: display_path(&startup_root),
        default_root: display_path(&default_root),
        legacy_root: display_path(&legacy_root),
        config_path: display_path(&storage_config_path()),
        pending_restart: !paths_equivalent(active_root, &startup_root),
        using_legacy_default: configured_root.is_none()
            && env_path(RUNTIME_ROOT_OVERRIDE_ENV).is_none()
            && paths_equivalent(active_root, &legacy_root),
        default_root_is_legacy: paths_equivalent(&default_root, &legacy_root),
    }
}

pub fn save_runtime_root_config(
    raw_path: &str,
    active_root: &Path,
) -> Result<RuntimeStorageInfo, String> {
    let path = validate_runtime_root(raw_path)?;
    seed_runtime_root_config_files(active_root, &path)?;
    write_storage_config(&StorageConfig {
        runtime_root: Some(display_path(&path)),
    })?;
    Ok(storage_info(active_root))
}

pub fn reset_runtime_root_config(active_root: &Path) -> Result<RuntimeStorageInfo, String> {
    let path = storage_config_path();
    if path.exists() {
        std::fs::remove_file(&path)
            .map_err(|e| format!("Cannot remove storage config {}: {}", path.display(), e))?;
    }
    let startup_root = resolve_startup_runtime_root();
    seed_runtime_root_config_files(active_root, &startup_root)?;
    Ok(storage_info(active_root))
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn clean_config_path(value: String) -> Option<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

fn read_storage_config() -> StorageConfig {
    let path = storage_config_path();
    let Ok(raw) = std::fs::read_to_string(path) else {
        return StorageConfig::default();
    };
    serde_json::from_str::<StorageConfig>(&raw).unwrap_or_default()
}

fn write_storage_config(config: &StorageConfig) -> Result<(), String> {
    let path = storage_config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "Cannot create storage config dir {}: {}",
                parent.display(),
                e
            )
        })?;
    }
    let json = serde_json::to_string_pretty(config)
        .map_err(|e| format!("Cannot serialize storage config: {}", e))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("Cannot save storage config {}: {}", path.display(), e))
}

fn validate_runtime_root(raw_path: &str) -> Result<PathBuf, String> {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return Err("Choose a storage folder first.".to_string());
    }

    let path = PathBuf::from(trimmed);
    if !path.is_absolute() {
        return Err("Storage folder must be an absolute path.".to_string());
    }

    std::fs::create_dir_all(&path)
        .map_err(|e| format!("Cannot create storage folder {}: {}", path.display(), e))?;

    let test_path = path.join(".gary4local-write-test");
    std::fs::write(&test_path, b"ok")
        .map_err(|e| format!("Storage folder is not writable: {}", e))?;
    let _ = std::fs::remove_file(&test_path);

    Ok(path)
}

fn seed_runtime_root_config_files(active_root: &Path, target_root: &Path) -> Result<(), String> {
    for name in ["hf_token.txt", "app_settings.json"] {
        let source = active_root.join(name);
        let target = target_root.join(name);
        if !source.is_file() || target.exists() {
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
        }
        std::fs::copy(&source, &target).map_err(|e| {
            format!(
                "Cannot copy {} to {}: {}",
                source.display(),
                target.display(),
                e
            )
        })?;
    }
    Ok(())
}

fn legacy_runtime_root_is_populated(path: &Path) -> bool {
    [
        "app_settings.json",
        "hf_token.txt",
        "models",
        "services",
        "sa3",
        "carey",
    ]
    .iter()
    .any(|name| path.join(name).exists())
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

pub fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left_key = path_key(left);
    let right_key = path_key(right);
    left_key == right_key
}

fn path_key(path: &Path) -> String {
    let value = path.to_string_lossy().replace('/', "\\");
    if cfg!(windows) {
        value.to_ascii_lowercase()
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::{legacy_runtime_root, runtime_env_vars};
    use std::path::Path;

    #[test]
    fn runtime_env_points_caches_inside_runtime_root() {
        let root = Path::new("D:\\gary4local-data");
        let env = runtime_env_vars(root);
        let lookup = |key: &str| {
            env.iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
                .unwrap()
        };

        assert_eq!(lookup("MODELS_DIR"), "D:\\gary4local-data\\models");
        assert_eq!(
            lookup("HF_HOME"),
            "D:\\gary4local-data\\models\\huggingface"
        );
        assert_eq!(lookup("UV_CACHE_DIR"), "D:\\gary4local-data\\cache\\uv");
    }

    #[test]
    fn legacy_runtime_root_remains_roaming_gary4juce() {
        assert!(legacy_runtime_root()
            .to_string_lossy()
            .contains("Gary4JUCE"));
    }

    #[test]
    fn legacy_runtime_preserves_external_model_cache_defaults() {
        let env = runtime_env_vars(&legacy_runtime_root());
        for key in [
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
        ] {
            assert!(env.iter().all(|(name, _)| name != key), "unexpected {key}");
        }
        assert!(env.iter().any(|(name, _)| name == "UV_CACHE_DIR"));
    }
}
