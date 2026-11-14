use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const ACTIVE_RUNTIME_ROOT_ENV: &str = "GARY4LOCAL_ACTIVE_RUNTIME_ROOT";
pub const RUNTIME_ROOT_OVERRIDE_ENV: &str = "GARY4LOCAL_RUNTIME_ROOT";
const LOCAL_DATA_DIR_NAME: &str = "com.betweentwomidnights.gary4local.rocm";
const LEGACY_RUNTIME_DIR_NAME: &str = "Gary4JUCE-ROCm";

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
    #[cfg(target_os = "linux")]
    {
        let base = linux_xdg_dir("XDG_CONFIG_HOME", ".config");
        return if linux_preview() {
            base.join(linux_storage_id())
        } else {
            base
        };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
            format!("{}\\AppData\\Roaming", home)
        });
        PathBuf::from(appdata)
    }
}

pub fn local_data_root() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        return linux_xdg_dir("XDG_DATA_HOME", ".local/share").join(linux_storage_id());
    }
    #[cfg(not(target_os = "linux"))]
    {
        let localappdata = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
            format!("{}\\AppData\\Local", home)
        });
        PathBuf::from(localappdata).join(LOCAL_DATA_DIR_NAME)
    }
}

#[cfg(target_os = "linux")]
fn linux_preview() -> bool {
    option_env!("GARY4LOCAL_LINUX_PREVIEW") == Some("1")
}

#[cfg(target_os = "linux")]
fn linux_storage_id() -> &'static str {
    if linux_preview() {
        "com.betweentwomidnights.gary4local.preview"
    } else {
        "com.betweentwomidnights.gary4local"
    }
}

#[cfg(target_os = "linux")]
fn xdg_base(value: Option<PathBuf>, home: &Path, fallback: &str) -> PathBuf {
    value.filter(|path| path.is_absolute()).unwrap_or_else(|| home.join(fallback))
}

#[cfg(target_os = "linux")]
fn linux_xdg_dir(name: &str, fallback: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    xdg_base(env_path(name), &home, fallback)
}

#[cfg(target_os = "linux")]
fn preview_runtime_dir(preview_home: Option<PathBuf>) -> PathBuf {
    preview_home.filter(|path| path.is_absolute())
        .map(|path| path.join("data"))
        .unwrap_or_else(|| local_data_root().join("runtime"))
}

pub fn legacy_runtime_root() -> PathBuf {
    roaming_app_data_dir().join(LEGACY_RUNTIME_DIR_NAME)
}

pub fn storage_config_path() -> PathBuf {
    local_data_root().join("storage.json")
}

pub fn installed_default_runtime_root() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        return if linux_preview() {
            preview_runtime_dir(env_path("GARY4LOCAL_PREVIEW_HOME"))
        } else {
            local_data_root().join("runtime")
        };
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| local_data_root()))
            .join("gary4local-data")
    }
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

pub fn ensure_mutations_allowed(root: &Path) -> Result<(), String> {
    if crate::storage_transfer::in_progress() { return Err("Wait for the storage transfer to finish.".into()); }
    if storage_info(root).pending_restart { return Err("Restart to use the selected storage before changing runtime files.".into()); }
    Ok(())
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
    progress: impl FnMut(crate::storage_transfer::Progress),
) -> Result<RuntimeStorageInfo, String> {
    let path = validate_runtime_root(raw_path)?;
    crate::storage_transfer::run(active_root, &path, progress)?;
    seed_runtime_root_config_files(active_root, &path)?;
    write_storage_config(&StorageConfig {
        runtime_root: Some(display_path(&path)),
    })?;
    Ok(storage_info(active_root))
}

pub fn reset_runtime_root_config(active_root: &Path, progress: impl FnMut(crate::storage_transfer::Progress)) -> Result<RuntimeStorageInfo, String> {
    let startup_root = automatic_default_runtime_root();
    std::fs::create_dir_all(&startup_root).map_err(|e| e.to_string())?;
    crate::storage_transfer::run(active_root, &startup_root, progress)?;
    seed_runtime_root_config_files(active_root, &startup_root)?;
    write_storage_config(&StorageConfig::default())?;
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
    crate::sa3_training::save(&path, config)
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
    if let (Ok(left), Ok(right)) = (left.canonicalize(), right.canonicalize()) {
        return path_key(&left) == path_key(&right);
    }
    let left_key = path_key(left);
    let right_key = path_key(right);
    left_key == right_key
}

fn path_key(path: &Path) -> String {
    let value = path.to_string_lossy().replace('/', "\\");
    let value = value.strip_prefix("\\\\?\\").unwrap_or(&value).trim_end_matches('\\');
    if cfg!(windows) {
        value.to_ascii_lowercase()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{legacy_runtime_root, runtime_env_vars};
    #[cfg(target_os = "linux")]
    use std::path::Path;

    #[test]
    fn runtime_env_points_caches_inside_runtime_root() {
        let root = std::env::temp_dir().join("gary4local-data");
        let env = runtime_env_vars(&root);
        let lookup = |key: &str| {
            env.iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
                .unwrap()
        };

        assert_eq!(lookup("MODELS_DIR"), root.join("models").to_string_lossy());
        assert_eq!(
            lookup("HF_HOME"),
            root.join("models").join("huggingface").to_string_lossy()
        );
        assert_eq!(lookup("UV_CACHE_DIR"), root.join("cache").join("uv").to_string_lossy());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_defaults_use_user_data_instead_of_the_installed_binary_directory() {
        assert_eq!(super::installed_default_runtime_root(), super::local_data_root().join("runtime"));
        assert!(super::local_data_root().is_absolute());
        assert!(!super::local_data_root().to_string_lossy().contains("AppData"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn relative_xdg_paths_are_ignored() {
        let home = Path::new("/home/test-user");
        assert_eq!(super::xdg_base(Some("/data/config".into()), home, ".config"), Path::new("/data/config"));
        assert_eq!(super::xdg_base(Some("relative".into()), home, ".config"), home.join(".config"));
        assert_eq!(super::xdg_base(None, home, ".local/share"), home.join(".local/share"));
        assert_eq!(super::preview_runtime_dir(Some("/qa/preview".into())), Path::new("/qa/preview/data"));
        assert_eq!(super::preview_runtime_dir(Some("relative".into())), super::local_data_root().join("runtime"));
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
