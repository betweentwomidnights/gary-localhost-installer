mod manifest;
mod model_manager;
mod service_manager;
mod update;

use model_manager::ModelManager;
use serde::{Deserialize, Serialize};
use service_manager::ServiceManager;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::image::Image;
use tauri::menu::{IconMenuItemBuilder, MenuBuilder, MenuItemBuilder, PredefinedMenuItem};
use tauri::tray::{TrayIconBuilder, TrayIconId};
use tauri::{Emitter, Manager};
use tokio::sync::Mutex;

type ManagerState = Arc<Mutex<ServiceManager>>;
type ModelState = Arc<Mutex<ModelManager>>;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppSettings {
    #[serde(default)]
    melodyflow_use_flash_attn: bool,
    #[serde(default)]
    carey_use_xl_models: bool,
    #[serde(default)]
    close_action_on_x: CloseActionOnX,
    #[serde(default = "default_auto_check_updates")]
    auto_check_updates: bool,
    #[serde(default)]
    skipped_update_version: Option<String>,
    #[serde(default)]
    last_update_check_epoch_ms: Option<u64>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            melodyflow_use_flash_attn: false,
            carey_use_xl_models: false,
            close_action_on_x: CloseActionOnX::Ask,
            auto_check_updates: default_auto_check_updates(),
            skipped_update_version: None,
            last_update_check_epoch_ms: None,
        }
    }
}

fn default_auto_check_updates() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum CloseActionOnX {
    Ask,
    Tray,
    Quit,
}

impl Default for CloseActionOnX {
    fn default() -> Self {
        Self::Ask
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppSettingsPatch {
    #[serde(default)]
    melodyflow_use_flash_attn: Option<bool>,
    #[serde(default)]
    carey_use_xl_models: Option<bool>,
    #[serde(default)]
    close_action_on_x: Option<CloseActionOnX>,
    #[serde(default)]
    auto_check_updates: Option<bool>,
    #[serde(default)]
    skipped_update_version: Option<Option<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CareyLoraCatalogEntry {
    path: String,
    #[serde(default)]
    captions_path: Option<String>,
    #[serde(default = "default_lora_scale")]
    scale: f64,
    #[serde(default = "default_lora_backends")]
    backends: Vec<String>,
    #[serde(default = "default_lora_model_family")]
    model_family: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CareyLoraEntry {
    name: String,
    path: String,
    captions_path: Option<String>,
    resolved_captions_path: Option<String>,
    caption_count: usize,
    scale: f64,
    backends: Vec<String>,
    model_family: String,
    checkpoint_exists: bool,
    registered: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CareyLoraState {
    entries: Vec<CareyLoraEntry>,
    pools: HashMap<String, usize>,
    catalog_path: String,
    registry_path: String,
    captions_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CareyCaptionsBuildResult {
    state: CareyLoraState,
    output: String,
}

fn default_lora_scale() -> f64 {
    1.0
}

fn default_lora_backends() -> Vec<String> {
    vec!["base".to_string(), "turbo".to_string()]
}

fn default_lora_model_family() -> String {
    "standard".to_string()
}

/// Get the path to the stored HF token file.
fn hf_token_path() -> std::path::PathBuf {
    let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
        let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
        format!("{}\\AppData\\Roaming", home)
    });
    std::path::PathBuf::from(appdata)
        .join("Gary4JUCE")
        .join("hf_token.txt")
}

fn app_settings_path() -> std::path::PathBuf {
    gary4juce_runtime_root().join("app_settings.json")
}

fn gary4juce_runtime_root() -> PathBuf {
    let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
        let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
        format!("{}\\AppData\\Roaming", home)
    });
    PathBuf::from(appdata).join("Gary4JUCE")
}

fn runtime_services_dir(runtime_root: &Path) -> PathBuf {
    runtime_root.join("services")
}

fn carey_runtime_dir() -> PathBuf {
    gary4juce_runtime_root().join("carey")
}

fn carey_lora_catalog_path() -> PathBuf {
    carey_runtime_dir().join("lora_catalog.json")
}

fn carey_lora_registry_path() -> PathBuf {
    carey_runtime_dir().join("lora_registry.json")
}

fn carey_captions_path() -> PathBuf {
    carey_runtime_dir().join("captions.json")
}

fn bundled_default_captions_path(runtime_root: &Path) -> PathBuf {
    runtime_root
        .join("services")
        .join("carey")
        .join("default_captions.json")
}

fn resolve_bundle_root_from_resource_dir(resource_dir: &Path) -> Result<PathBuf, String> {
    let direct_root = resource_dir.to_path_buf();
    if direct_root.join("services").is_dir() {
        return Ok(direct_root);
    }

    let nested_root = resource_dir.join("resources");
    if nested_root.join("services").is_dir() {
        return Ok(nested_root);
    }

    Err(format!(
        "Bundled services not found under {} or {}",
        direct_root.join("services").display(),
        nested_root.join("services").display()
    ))
}

fn hide_console_window(cmd: &mut tokio::process::Command) {
    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut entries = Vec::new();
    let reader =
        std::fs::read_dir(dir).map_err(|e| format!("Cannot read {}: {}", dir.display(), e))?;

    for entry in reader {
        let entry = entry.map_err(|e| format!("Cannot read entry in {}: {}", dir.display(), e))?;
        entries.push(entry.path());
    }

    entries.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .cmp(&b.file_name().unwrap_or_default().to_string_lossy())
    });
    Ok(entries)
}

fn hash_dir_recursive(dir: &Path, base: &Path, hasher: &mut impl Hasher) -> Result<(), String> {
    for entry in read_dir_sorted(dir)? {
        let relative = entry
            .strip_prefix(base)
            .map_err(|e| format!("Cannot relativize {}: {}", entry.display(), e))?;
        relative.to_string_lossy().hash(hasher);

        if entry.is_dir() {
            "dir".hash(hasher);
            hash_dir_recursive(&entry, base, hasher)?;
        } else if entry.is_file() {
            "file".hash(hasher);
            let bytes = std::fs::read(&entry)
                .map_err(|e| format!("Cannot read {}: {}", entry.display(), e))?;
            bytes.hash(hasher);
        }
    }

    Ok(())
}

fn compute_bundle_sync_stamp(bundle_root: &Path) -> Result<String, String> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    env!("CARGO_PKG_VERSION").hash(&mut hasher);

    let services_dir = bundle_root.join("services");
    if !services_dir.is_dir() {
        return Err(format!(
            "Bundled services directory is missing: {}",
            services_dir.display()
        ));
    }

    hash_dir_recursive(&services_dir, &services_dir, &mut hasher)?;

    let icon_path = bundle_root.join("icon.png");
    if icon_path.is_file() {
        "icon.png".hash(&mut hasher);
        let icon_bytes = std::fs::read(&icon_path)
            .map_err(|e| format!("Cannot read {}: {}", icon_path.display(), e))?;
        icon_bytes.hash(&mut hasher);
    }

    Ok(format!(
        "{}-{:016x}",
        env!("CARGO_PKG_VERSION"),
        hasher.finish()
    ))
}

fn remove_path(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }

    if path.is_dir() {
        std::fs::remove_dir_all(path)
            .map_err(|e| format!("Cannot remove directory {}: {}", path.display(), e))?;
    } else {
        std::fs::remove_file(path)
            .map_err(|e| format!("Cannot remove file {}: {}", path.display(), e))?;
    }

    Ok(())
}

fn clear_dir_except(dir: &Path, preserve_names: &[&str]) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("Cannot create {}: {}", dir.display(), e))?;

    for entry in read_dir_sorted(dir)? {
        let name = entry
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        if preserve_names.iter().any(|preserve| *preserve == name) {
            continue;
        }

        remove_path(&entry)?;
    }

    Ok(())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| format!("Cannot create {}: {}", dst.display(), e))?;

    for entry in read_dir_sorted(src)? {
        let name = entry.file_name().unwrap_or_default().to_os_string();
        let dst_entry = dst.join(name);

        if entry.is_dir() {
            copy_dir_recursive(&entry, &dst_entry)?;
        } else if entry.is_file() {
            if let Some(parent) = dst_entry.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
            }
            std::fs::copy(&entry, &dst_entry).map_err(|e| {
                format!(
                    "Cannot copy {} to {}: {}",
                    entry.display(),
                    dst_entry.display(),
                    e
                )
            })?;
        }
    }

    Ok(())
}

fn sync_bundled_services_to_runtime(bundle_root: &Path, runtime_root: &Path) -> Result<(), String> {
    const PRESERVED_RUNTIME_NAMES: &[&str] =
        &["env", "checkpoints", ".cache", "gradio_outputs", "riffs"];

    let bundle_services = bundle_root.join("services");
    let runtime_services = runtime_services_dir(runtime_root);
    let stamp_path = runtime_root.join(".services_bundle_stamp");
    let desired_stamp = compute_bundle_sync_stamp(bundle_root)?;
    let current_stamp = std::fs::read_to_string(&stamp_path).ok();

    if current_stamp.as_deref() == Some(desired_stamp.as_str())
        && runtime_services
            .join("manifests")
            .join("services.json")
            .exists()
    {
        log::info!("Runtime services already match bundled resources");
        return Ok(());
    }

    log::info!(
        "Refreshing runtime services at {} from bundled resources {}",
        runtime_services.display(),
        bundle_services.display()
    );

    std::fs::create_dir_all(runtime_root).map_err(|e| {
        format!(
            "Cannot create runtime root {}: {}",
            runtime_root.display(),
            e
        )
    })?;
    std::fs::create_dir_all(&runtime_services).map_err(|e| {
        format!(
            "Cannot create runtime services dir {}: {}",
            runtime_services.display(),
            e
        )
    })?;

    let bundle_entries = read_dir_sorted(&bundle_services)?;
    let bundle_names: std::collections::HashSet<String> = bundle_entries
        .iter()
        .filter_map(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
        .collect();

    for existing in read_dir_sorted(&runtime_services)? {
        let name = existing
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if !bundle_names.contains(&name) {
            remove_path(&existing)?;
        }
    }

    for src in bundle_entries {
        let name = src.file_name().unwrap_or_default().to_os_string();
        let dst = runtime_services.join(name);

        if src.is_dir() {
            let preserve = if src.file_name().unwrap_or_default() == "manifests" {
                &[][..]
            } else {
                PRESERVED_RUNTIME_NAMES
            };
            clear_dir_except(&dst, preserve)?;
            copy_dir_recursive(&src, &dst)?;
        } else if src.is_file() {
            std::fs::copy(&src, &dst).map_err(|e| {
                format!("Cannot copy {} to {}: {}", src.display(), dst.display(), e)
            })?;
        }
    }

    let bundle_icon = bundle_root.join("icon.png");
    let runtime_icon = runtime_root.join("icon.png");
    if bundle_icon.is_file() {
        std::fs::copy(&bundle_icon, &runtime_icon).map_err(|e| {
            format!(
                "Cannot copy {} to {}: {}",
                bundle_icon.display(),
                runtime_icon.display(),
                e
            )
        })?;
    }

    std::fs::write(&stamp_path, desired_stamp)
        .map_err(|e| format!("Cannot write {}: {}", stamp_path.display(), e))?;

    Ok(())
}

/// Read the HF token from our stored file, falling back to system env.
fn read_hf_token() -> Option<String> {
    // Try our stored file first
    let path = hf_token_path();
    if let Ok(token) = std::fs::read_to_string(&path) {
        let trimmed = token.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }
    // Fall back to system environment variable
    std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty())
}

fn read_app_settings() -> AppSettings {
    let path = app_settings_path();
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(_) => return AppSettings::default(),
    };

    serde_json::from_str::<AppSettings>(&raw).unwrap_or_default()
}

fn save_app_settings_file(settings: &AppSettings) -> Result<(), String> {
    let path = app_settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create config dir: {}", e))?;
    }

    let json = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("Cannot serialize app settings: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Cannot save app settings: {}", e))?;
    Ok(())
}

fn merge_app_settings(patch: AppSettingsPatch) -> AppSettings {
    let mut current = read_app_settings();

    if let Some(melodyflow_use_flash_attn) = patch.melodyflow_use_flash_attn {
        current.melodyflow_use_flash_attn = melodyflow_use_flash_attn;
    }

    if let Some(carey_use_xl_models) = patch.carey_use_xl_models {
        current.carey_use_xl_models = carey_use_xl_models;
    }

    if let Some(close_action_on_x) = patch.close_action_on_x {
        current.close_action_on_x = close_action_on_x;
    }

    if let Some(auto_check_updates) = patch.auto_check_updates {
        current.auto_check_updates = auto_check_updates;
    }

    if let Some(skipped_update_version) = patch.skipped_update_version {
        current.skipped_update_version = skipped_update_version
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
    }

    current
}

fn sanitize_lora_name(raw: &str) -> Option<String> {
    let normalized = raw.trim().to_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
    {
        Some(normalized)
    } else {
        None
    }
}

fn sanitize_backend_list(values: Vec<String>) -> Vec<String> {
    let mut cleaned = Vec::new();
    for value in values {
        let text = value.trim().to_lowercase();
        if matches!(text.as_str(), "base" | "turbo") && !cleaned.contains(&text) {
            cleaned.push(text);
        }
    }
    if cleaned.is_empty() {
        default_lora_backends()
    } else {
        cleaned
    }
}

fn normalize_lora_model_family(raw: &str) -> String {
    match raw.trim().to_lowercase().as_str() {
        "xl" => "xl".to_string(),
        _ => "standard".to_string(),
    }
}

fn infer_lora_model_family_from_path(path: &Path) -> String {
    let text = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    if text.contains("xl") {
        "xl".to_string()
    } else {
        "standard".to_string()
    }
}

fn looks_like_lora_checkpoint_dir(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    if dir.join("adapter_config.json").exists()
        || dir.join("adapter_model.safetensors").exists()
        || dir.join("pytorch_lora_weights.safetensors").exists()
    {
        return true;
    }
    match std::fs::read_dir(dir) {
        Ok(entries) => entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .map(|ext| ext.to_string_lossy().eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false)
        }),
        Err(_) => false,
    }
}

fn count_caption_sidecars(dir: &Path) -> usize {
    if !dir.is_dir() {
        return 0;
    }
    match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .flatten()
            .filter(|entry| entry.path().is_file())
            .filter(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                let is_txt = entry
                    .path()
                    .extension()
                    .map(|ext| ext.to_string_lossy().eq_ignore_ascii_case("txt"))
                    .unwrap_or(false);
                is_txt && !name.contains(".v4bak") && !name.ends_with(".v4bak")
            })
            .count(),
        Err(_) => 0,
    }
}

fn read_carey_lora_catalog() -> Result<BTreeMap<String, CareyLoraCatalogEntry>, String> {
    let path = carey_lora_catalog_path();
    if !path.exists() {
        return Ok(BTreeMap::new());
    }

    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
    let mut parsed: BTreeMap<String, CareyLoraCatalogEntry> =
        serde_json::from_str(&raw).map_err(|e| format!("Invalid LoRA catalog JSON: {}", e))?;

    for entry in parsed.values_mut() {
        entry.path = entry.path.trim().to_string();
        entry.captions_path = entry
            .captions_path
            .take()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        entry.backends = sanitize_backend_list(std::mem::take(&mut entry.backends));
        entry.model_family = normalize_lora_model_family(&entry.model_family);
    }

    Ok(parsed)
}

fn save_carey_lora_catalog(catalog: &BTreeMap<String, CareyLoraCatalogEntry>) -> Result<(), String> {
    let path = carey_lora_catalog_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    let json = serde_json::to_string_pretty(catalog)
        .map_err(|e| format!("Cannot serialize LoRA catalog: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Cannot save {}: {}", path.display(), e))?;
    Ok(())
}

fn resolve_carey_captions_source(entry: &CareyLoraCatalogEntry) -> Option<PathBuf> {
    if let Some(captions_path) = entry
        .captions_path
        .as_ref()
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
    {
        return Some(captions_path);
    }

    let checkpoint_dir = PathBuf::from(&entry.path);
    if count_caption_sidecars(&checkpoint_dir) > 0 {
        return Some(checkpoint_dir);
    }

    None
}

fn load_carey_lora_metadata(
    checkpoint_dir: &Path,
    captions_dir: Option<&Path>,
    existing: Option<&CareyLoraCatalogEntry>,
) -> (f64, Vec<String>, String) {
    let mut candidates = vec![checkpoint_dir.join("metadata.json")];
    if let Some(captions_dir) = captions_dir {
        let candidate = captions_dir.join("metadata.json");
        if candidate != candidates[0] {
            candidates.push(candidate);
        }
    }

    for candidate in candidates {
        if !candidate.is_file() {
            continue;
        }

        let Ok(raw) = std::fs::read_to_string(&candidate) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };

        let scale = value
            .get("scale")
            .and_then(|value| value.as_f64())
            .unwrap_or_else(default_lora_scale);
        let backends = value
            .get("backends")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(|text| text.to_string()))
                    .collect::<Vec<_>>()
            })
            .map(sanitize_backend_list)
            .unwrap_or_else(default_lora_backends);
        let model_family = value
            .get("model_family")
            .or_else(|| value.get("family"))
            .and_then(|value| value.as_str())
            .map(normalize_lora_model_family)
            .unwrap_or_else(|| infer_lora_model_family_from_path(checkpoint_dir));
        return (scale, backends, model_family);
    }

    if let Some(existing) = existing {
        return (
            existing.scale,
            sanitize_backend_list(existing.backends.clone()),
            normalize_lora_model_family(&existing.model_family),
        );
    }

    (
        default_lora_scale(),
        default_lora_backends(),
        infer_lora_model_family_from_path(checkpoint_dir),
    )
}

fn build_carey_lora_entries(
    catalog: &BTreeMap<String, CareyLoraCatalogEntry>,
) -> Vec<CareyLoraEntry> {
    catalog
        .iter()
        .map(|(name, entry)| {
            let checkpoint_path = PathBuf::from(&entry.path);
            let checkpoint_exists = looks_like_lora_checkpoint_dir(&checkpoint_path);
            let resolved_captions_path = resolve_carey_captions_source(entry);
            let caption_count = resolved_captions_path
                .as_ref()
                .map(|path| count_caption_sidecars(path))
                .unwrap_or(0);

            CareyLoraEntry {
                name: name.clone(),
                path: entry.path.clone(),
                captions_path: entry.captions_path.clone(),
                resolved_captions_path: resolved_captions_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().to_string()),
                caption_count,
                scale: entry.scale,
                backends: sanitize_backend_list(entry.backends.clone()),
                model_family: normalize_lora_model_family(&entry.model_family),
                checkpoint_exists,
                registered: checkpoint_exists,
            }
        })
        .collect()
}

fn write_carey_lora_registry(entries: &[CareyLoraEntry]) -> Result<(), String> {
    let path = carey_lora_registry_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }

    let mut payload = BTreeMap::<String, serde_json::Value>::new();
    for entry in entries.iter().filter(|entry| entry.registered) {
        payload.insert(
            entry.name.clone(),
            serde_json::json!({
                "path": entry.path,
                "scale": entry.scale,
                "backends": entry.backends,
                "model_family": entry.model_family,
            }),
        );
    }

    let json = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("Cannot serialize LoRA registry: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Cannot save {}: {}", path.display(), e))?;
    Ok(())
}

fn read_carey_caption_pools() -> HashMap<String, usize> {
    let path = carey_captions_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return HashMap::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return HashMap::new();
    };
    let Some(obj) = value.as_object() else {
        return HashMap::new();
    };

    let mut pools = HashMap::new();
    for (name, entries) in obj {
        let count = entries.as_array().map(|items| items.len()).unwrap_or(0);
        pools.insert(name.clone(), count);
    }
    pools
}

fn read_bundled_default_caption_pool(runtime_root: &Path) -> Vec<String> {
    let path = bundled_default_captions_path(runtime_root);
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return Vec::new();
    };
    value
        .get("default")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(|text| text.trim().to_string()))
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn ensure_default_carey_captions(runtime_root: &Path) -> Result<(), String> {
    let default_pool = read_bundled_default_caption_pool(runtime_root);
    if default_pool.is_empty() {
        return Ok(());
    }

    let path = carey_captions_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }

    let mut payload = if path.exists() {
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
        serde_json::from_str::<serde_json::Value>(&raw).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    payload["default"] = serde_json::json!(default_pool);
    let json = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("Cannot serialize default captions: {}", e))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("Cannot save {}: {}", path.display(), e))?;

    Ok(())
}

fn merge_default_captions_into_file(runtime_root: &Path) -> Result<(), String> {
    ensure_default_carey_captions(runtime_root)
}

fn build_carey_lora_state(runtime_root: &Path) -> Result<CareyLoraState, String> {
    ensure_default_carey_captions(runtime_root)?;
    let catalog = read_carey_lora_catalog()?;
    let entries = build_carey_lora_entries(&catalog);
    write_carey_lora_registry(&entries)?;
    Ok(CareyLoraState {
        entries,
        pools: read_carey_caption_pools(),
        catalog_path: carey_lora_catalog_path().to_string_lossy().to_string(),
        registry_path: carey_lora_registry_path().to_string_lossy().to_string(),
        captions_path: carey_captions_path().to_string_lossy().to_string(),
    })
}

async fn try_reload_carey_admin() -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(4))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };

    match client.post("http://localhost:8003/admin/reload").send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

async fn quit_application(handle: &tauri::AppHandle, manager: &ManagerState) {
    let mut m = manager.lock().await;
    m.stop_all();
    handle.exit(0);
}

pub(crate) fn melodyflow_use_flash_attn_enabled() -> bool {
    if option_env!("VITE_ENABLE_MELODYFLOW_FA2_TOGGLE").unwrap_or("1") == "0" {
        return false;
    }
    read_app_settings().melodyflow_use_flash_attn
}

pub(crate) fn carey_use_xl_models_enabled() -> bool {
    read_app_settings().carey_use_xl_models
}

/// Load a PNG file and decode it to RGBA for Tauri Image
fn load_png_as_image(path: &std::path::Path) -> Option<Image<'static>> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(buf.len() / 3 * 4);
            for chunk in buf.chunks(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        _ => return None,
    };

    Some(Image::new_owned(rgba, info.width, info.height))
}

/// Generate a small colored circle image (16x16 RGBA) for tray menu icons.
fn make_dot_icon(r: u8, g: u8, b: u8) -> Image<'static> {
    const SIZE: u32 = 16;
    const CENTER: f32 = 7.5;
    const RADIUS: f32 = 6.0;
    let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];

    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - CENTER;
            let dy = y as f32 - CENTER;
            let dist = (dx * dx + dy * dy).sqrt();
            let idx = ((y * SIZE + x) * 4) as usize;
            if dist <= RADIUS {
                // Smooth edge with 1px anti-aliasing
                let alpha = if dist > RADIUS - 1.0 {
                    ((RADIUS - dist) * 255.0) as u8
                } else {
                    255
                };
                rgba[idx] = r;
                rgba[idx + 1] = g;
                rgba[idx + 2] = b;
                rgba[idx + 3] = alpha;
            }
        }
    }

    Image::new_owned(rgba, SIZE, SIZE)
}

/// Get the dot icon for a service status.
fn status_dot_icon(status: &str) -> Image<'static> {
    match status {
        "running" => make_dot_icon(0x4C, 0xAF, 0x50),  // green
        "starting" => make_dot_icon(0xFF, 0xC1, 0x07), // amber/yellow
        "failed" => make_dot_icon(0xF4, 0x43, 0x36),   // red
        _ => make_dot_icon(0x9E, 0x9E, 0x9E),          // grey
    }
}

fn tray_menu_signature(services: &[service_manager::ServiceInfo]) -> String {
    services
        .iter()
        .map(|svc| format!("{}:{}:{}", svc.id, svc.status, svc.display_name))
        .collect::<Vec<_>>()
        .join("|")
}

/// Rebuild the system tray menu to reflect current service statuses.
/// Call this after acquiring service info (avoids locking the mutex here).
async fn rebuild_tray_menu(app_handle: &tauri::AppHandle, manager: &ManagerState) {
    let services = {
        let mgr = manager.lock().await;
        mgr.get_service_info()
    };
    rebuild_tray_menu_with_info(app_handle, &services);
}

/// Inner sync function that builds the tray menu from pre-fetched service info.
fn rebuild_tray_menu_with_info(
    app_handle: &tauri::AppHandle,
    services: &[service_manager::ServiceInfo],
) {
    let show_item = MenuItemBuilder::with_id("show", "show control center")
        .build(app_handle)
        .unwrap();
    let quit_item = MenuItemBuilder::with_id("quit", "quit (stop all services)")
        .build(app_handle)
        .unwrap();
    let sep1 = PredefinedMenuItem::separator(app_handle).unwrap();
    let sep2 = PredefinedMenuItem::separator(app_handle).unwrap();

    let mut menu_builder = MenuBuilder::new(app_handle).item(&show_item).item(&sep1);

    for svc in services {
        let is_running = svc.status == "running" || svc.status == "starting";
        let action = if is_running { "stop" } else { "start" };
        let label = format!("{} - {}", svc.display_name, action);
        let item_id = format!("svc_{}", svc.id);
        let icon = status_dot_icon(&svc.status);
        let item = IconMenuItemBuilder::with_id(item_id, label)
            .icon(icon)
            .build(app_handle)
            .unwrap();
        menu_builder = menu_builder.item(&item);
    }

    let menu = menu_builder.item(&sep2).item(&quit_item).build().unwrap();

    if let Some(tray) = app_handle.tray_by_id(&TrayIconId::new("main-tray")) {
        let _ = tray.set_menu(Some(menu));
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_dialog::init());
    let builder = if update::app_updater_enabled() {
        builder.plugin(tauri_plugin_updater::Builder::new().build())
    } else {
        builder
    };

    builder.setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            app.manage(update::PendingNativeUpdate::default());

            // --- Resolve runtime root ---
            // In dev, run directly from the repo.
            // In production, sync bundled services into %APPDATA%\Gary4JUCE
            // and run from there so logs/envs/source live in a writable location.
            let bundle_root = if cfg!(debug_assertions) {
                let exe_dir = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.to_path_buf()));
                exe_dir
                    .as_ref()
                    .and_then(|d| d.ancestors().find(|a| a.join("services").is_dir()))
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::env::current_dir().unwrap())
            } else {
                let resource_dir = app.path().resource_dir().map_err(std::io::Error::other)?;
                resolve_bundle_root_from_resource_dir(&resource_dir)
                    .map_err(std::io::Error::other)?
            };

            let runtime_root = if cfg!(debug_assertions) {
                bundle_root.clone()
            } else {
                let runtime_root = gary4juce_runtime_root();
                sync_bundled_services_to_runtime(&bundle_root, &runtime_root)
                    .map_err(std::io::Error::other)?;
                runtime_root
            };

            let manifest_path = runtime_root
                .join("services")
                .join("manifests")
                .join("services.json");
            log::info!("Loading manifest from: {}", manifest_path.display());

            let services = match manifest::load_manifest(&manifest_path) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("Failed to load manifest: {}", e);
                    Vec::new()
                }
            };

            let manager = Arc::new(Mutex::new(ServiceManager::new(
                services,
                runtime_root.clone(),
            )));
            app.manage(manager.clone());

            let model_mgr = Arc::new(Mutex::new(ModelManager::new(runtime_root.clone())));
            app.manage(model_mgr.clone());

            // Store runtime_root for model downloads
            app.manage(runtime_root.clone());

            // --- System tray with Gary icon + per-service controls ---
            let handle = app.handle().clone();
            let tray_manager = manager.clone();

            let icon_path = runtime_root.join("icon.png");
            let tray_icon = if icon_path.exists() {
                load_png_as_image(&icon_path)
            } else {
                None
            };

            // Build tray menu with per-service items
            let show_item = MenuItemBuilder::with_id("show", "show control center").build(app)?;
            let quit_item =
                MenuItemBuilder::with_id("quit", "quit (stop all services)").build(app)?;

            let mut menu_builder = MenuBuilder::new(app)
                .item(&show_item)
                .item(&PredefinedMenuItem::separator(app)?);

            // Add a menu item for each service
            {
                let mgr = manager.blocking_lock();
                let services = mgr.get_service_info();
                for svc in &services {
                    let is_running = svc.status == "running" || svc.status == "starting";
                    let action = if is_running { "stop" } else { "start" };
                    let label = format!("{} - {}", svc.display_name, action);
                    let item_id = format!("svc_{}", svc.id);
                    let icon = status_dot_icon(&svc.status);
                    let item = IconMenuItemBuilder::with_id(item_id, label)
                        .icon(icon)
                        .build(app)?;
                    menu_builder = menu_builder.item(&item);
                }
            }

            let tray_menu = menu_builder
                .item(&PredefinedMenuItem::separator(app)?)
                .item(&quit_item)
                .build()?;

            let mut tray_builder = TrayIconBuilder::with_id("main-tray")
                .tooltip("gary4local")
                .menu(&tray_menu)
                .on_menu_event(
                    move |app_handle: &tauri::AppHandle, event: tauri::menu::MenuEvent| {
                        let event_id = event.id().as_ref().to_string();
                        match event_id.as_str() {
                            "show" => {
                                if let Some(window) = app_handle.get_webview_window("main") {
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                }
                            }
                            "quit" => {
                                let mgr = tray_manager.clone();
                                let handle = app_handle.clone();
                                tauri::async_runtime::spawn(async move {
                                    quit_application(&handle, &mgr).await;
                                });
                            }
                            id if id.starts_with("svc_") => {
                                let service_id = id.strip_prefix("svc_").unwrap().to_string();
                                let mgr = tray_manager.clone();
                                let handle = app_handle.clone();
                                tauri::async_runtime::spawn(async move {
                                    let mut m = mgr.lock().await;
                                    let is_running = m.is_running(&service_id);
                                    if is_running {
                                        let _ = m.stop(&service_id);
                                    } else {
                                        let _ = m.start(&service_id);
                                    }
                                    // Emit updated status
                                    let info = m.get_service_info();
                                    drop(m);
                                    let _ = handle.emit("services-updated", &info);
                                    // Rebuild tray menu to reflect new state
                                    rebuild_tray_menu(&handle, &mgr).await;
                                });
                            }
                            _ => {}
                        }
                    },
                );

            if let Some(icon) = tray_icon {
                tray_builder = tray_builder.icon(icon);
            }

            tray_builder.build(app)?;

            // --- Window close: ask whether to quit or minimize to tray ---
            let close_handle = handle.clone();
            let close_manager = manager.clone();
            {
                let window = app.get_webview_window("main").unwrap();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();

                        match read_app_settings().close_action_on_x {
                            CloseActionOnX::Tray => {
                                if let Some(w) = close_handle.get_webview_window("main") {
                                    let _ = w.hide();
                                }
                            }
                            CloseActionOnX::Quit => {
                                let handle = close_handle.clone();
                                let manager = close_manager.clone();
                                tauri::async_runtime::spawn(async move {
                                    quit_application(&handle, &manager).await;
                                });
                            }
                            CloseActionOnX::Ask => {
                                let _ = close_handle.emit("app-close-requested", ());
                            }
                        }
                    }
                });
            }

            // --- Background: health check + crash detection polling ---
            let poll_manager = manager.clone();
            let poll_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let client = reqwest::Client::builder().build().unwrap();
                let initial_info = {
                    let mgr = poll_manager.lock().await;
                    mgr.get_service_info()
                };
                let mut last_tray_signature = tray_menu_signature(&initial_info);

                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

                    let mut mgr = poll_manager.lock().await;
                    mgr.check_processes();
                    let targets = mgr.take_due_health_targets(std::time::Instant::now());
                    drop(mgr);

                    for target in &targets {
                        let url = format!("http://localhost:{}{}", target.port, target.endpoint);
                        let healthy = match client
                            .get(&url)
                            .timeout(std::time::Duration::from_secs(target.timeout_seconds))
                            .send()
                            .await
                        {
                            Ok(resp) => resp.status().is_success(),
                            Err(_) => false,
                        };

                        let mut mgr = poll_manager.lock().await;
                        mgr.set_health(&target.id, healthy);
                    }

                    let mgr = poll_manager.lock().await;
                    let info = mgr.get_service_info();
                    drop(mgr);
                    let _ = poll_handle.emit("services-updated", &info);

                    // Only rebuild the tray menu when the visible tray state changes.
                    // Replacing the menu every poll tick makes the popup close before
                    // the user can click a service action.
                    let current_tray_signature = tray_menu_signature(&info);
                    if current_tray_signature != last_tray_signature {
                        rebuild_tray_menu_with_info(&poll_handle, &info);
                        last_tray_signature = current_tray_signature;
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_services,
            start_service,
            stop_service,
            restart_service,
            rebuild_env,
            rebuild_all_envs,
            get_service_log,
            get_models,
            download_model,
            get_download_progress,
            fetch_jerry_checkpoints,
            get_carey_lora_state,
            upsert_carey_lora,
            remove_carey_lora,
            build_carey_lora_captions,
            get_hf_token,
            save_hf_token,
            delete_hf_token,
            get_app_settings,
            save_app_settings,
            check_for_app_update,
            install_app_update,
            resolve_close_request,
            open_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
async fn get_services(
    manager: tauri::State<'_, ManagerState>,
) -> Result<Vec<service_manager::ServiceInfo>, String> {
    let mut mgr = manager.lock().await;
    mgr.check_processes();
    Ok(mgr.get_service_info())
}

#[tauri::command]
async fn start_service(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<(), String> {
    let mut mgr = manager.lock().await;
    mgr.start(&service_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn stop_service(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<(), String> {
    let mut mgr = manager.lock().await;
    mgr.stop(&service_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn restart_service(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<(), String> {
    let mut mgr = manager.lock().await;
    mgr.stop(&service_id).ok();
    drop(mgr);
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let mut mgr = manager.lock().await;
    mgr.start(&service_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn rebuild_env(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let build_info = {
        let mut mgr = manager.lock().await;
        let info = mgr.get_build_info(&service_id)?;
        // +2 for "ensure uv" and "create venv" steps
        let total = info.build_steps.len() + 2;
        mgr.set_build_started(&service_id, total);
        info
    };

    let sid = service_id.clone();
    let mgr_clone = manager.inner().clone();
    let handle = app_handle.clone();

    tauri::async_runtime::spawn(async move {
        let result = run_build(build_info, mgr_clone.clone(), handle.clone()).await;

        let mut mgr = mgr_clone.lock().await;
        match result {
            Ok(()) => mgr.set_build_done(&sid, None),
            Err(e) => mgr.set_build_done(&sid, Some(e)),
        }

        let info = mgr.get_service_info();
        let _ = handle.emit("services-updated", &info);
    });

    Ok(())
}

#[tauri::command]
async fn rebuild_all_envs(
    manager: tauri::State<'_, ManagerState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let build_infos = {
        let mgr = manager.lock().await;
        mgr.get_all_build_infos()
    };

    let mgr_clone = manager.inner().clone();
    let handle = app_handle.clone();

    // Run builds sequentially in a single background task,
    // selecting each service in the UI as it starts building.
    tauri::async_runtime::spawn(async move {
        for info in build_infos {
            let sid = info.service_id.clone();

            // Tell the UI to select/highlight this service
            let _ = handle.emit("select-service", &sid);

            {
                let mut mgr = mgr_clone.lock().await;
                let total = info.build_steps.len() + 2;
                mgr.set_build_started(&sid, total);
            }

            let result = run_build(info, mgr_clone.clone(), handle.clone()).await;

            {
                let mut mgr = mgr_clone.lock().await;
                match result {
                    Ok(()) => mgr.set_build_done(&sid, None),
                    Err(e) => mgr.set_build_done(&sid, Some(e.clone())),
                }
                let info = mgr.get_service_info();
                let _ = handle.emit("services-updated", &info);
            }
        }
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Build pipeline: uv bootstrap -> Python 3.11 -> venv -> install steps
// ---------------------------------------------------------------------------

/// Ensure uv is installed. Returns the path to the uv executable.
async fn ensure_uv(
    service_id: &str,
    manager: &Arc<Mutex<ServiceManager>>,
    handle: &tauri::AppHandle,
) -> Result<String, String> {
    {
        let mut mgr = manager.lock().await;
        mgr.set_build_step(service_id, 0, "Checking for uv package manager...");
        mgr.append_build_log(service_id, "$ uv --version");
    }
    emit_status(manager, handle).await;

    // Check if uv is already on PATH
    let mut check_cmd = tokio::process::Command::new("uv");
    check_cmd.arg("--version");
    hide_console_window(&mut check_cmd);
    let check = check_cmd.output().await;

    if let Ok(output) = &check {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(service_id, &format!("uv found: {}", version.trim()));
            }
            return Ok("uv".to_string());
        }
    }

    // Check common install locations on Windows
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    let cargo_uv = format!("{}\\.cargo\\bin\\uv.exe", home);
    let local_uv = format!("{}\\.local\\bin\\uv.exe", home);

    for path in &[&cargo_uv, &local_uv] {
        if std::path::Path::new(path).exists() {
            let mut check_cmd = tokio::process::Command::new(path);
            check_cmd.arg("--version");
            hide_console_window(&mut check_cmd);
            let check = check_cmd.output().await;
            if let Ok(output) = check {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    {
                        let mut mgr = manager.lock().await;
                        mgr.append_build_log(
                            service_id,
                            &format!("uv found at {}: {}", path, version.trim()),
                        );
                    }
                    return Ok(path.to_string());
                }
            }
        }
    }

    // Install uv via PowerShell
    {
        let mut mgr = manager.lock().await;
        mgr.append_build_log(service_id, "uv not found. Installing via PowerShell...");
        mgr.append_build_log(
            service_id,
            "$ powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"",
        );
    }
    emit_status(manager, handle).await;

    let mut install_cmd = tokio::process::Command::new("powershell");
    install_cmd.args([
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "irm https://astral.sh/uv/install.ps1 | iex",
    ]);
    hide_console_window(&mut install_cmd);
    let install = install_cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run PowerShell uv installer: {}", e))?;

    {
        let mut mgr = manager.lock().await;
        let stdout = String::from_utf8_lossy(&install.stdout);
        let stderr = String::from_utf8_lossy(&install.stderr);
        if !stdout.is_empty() {
            mgr.append_build_log(service_id, &stdout);
        }
        if !stderr.is_empty() {
            mgr.append_build_log(service_id, &stderr);
        }
    }

    if !install.status.success() {
        return Err(
            "Failed to install uv. Please install manually: https://docs.astral.sh/uv/".to_string(),
        );
    }

    // After install, uv should be at ~/.local/bin/uv.exe or ~/.cargo/bin/uv.exe
    for path in &[&local_uv, &cargo_uv] {
        if std::path::Path::new(path).exists() {
            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(service_id, &format!("uv installed at {}", path));
            }
            return Ok(path.to_string());
        }
    }

    // Try PATH again (installer may have updated it)
    let mut recheck_cmd = tokio::process::Command::new("uv");
    recheck_cmd.arg("--version");
    hide_console_window(&mut recheck_cmd);
    let recheck = recheck_cmd.output().await;
    if let Ok(output) = recheck {
        if output.status.success() {
            return Ok("uv".to_string());
        }
    }

    Err("uv was installed but cannot be found. Restart the application and try again.".to_string())
}

/// Run the full build pipeline for a service
async fn run_build(
    build_info: service_manager::BuildInfo,
    manager: Arc<Mutex<ServiceManager>>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    let service_id = build_info.service_id.clone();
    let work_dir = &build_info.work_dir;
    let env_dir = &build_info.env_dir;

    // Step 0: Ensure uv is available
    let uv = ensure_uv(&service_id, &manager, &handle).await?;

    // Step 1: Create venv with uv (using Python 3.11)
    if !env_dir.exists() {
        {
            let mut mgr = manager.lock().await;
            mgr.set_build_step(
                &service_id,
                1,
                "Installing Python 3.11 and creating venv...",
            );
            mgr.append_build_log(&service_id, &format!("\n$ {} python install 3.11", uv));
        }
        emit_status(&manager, &handle).await;

        // First ensure Python 3.11 is available via uv
        let py_install = run_command_streamed(
            &uv,
            &["python", "install", "3.11"],
            work_dir,
            &service_id,
            &manager,
        )
        .await;

        if let Err(e) = py_install {
            let mut mgr = manager.lock().await;
            mgr.append_build_log(&service_id, &format!("Warning: uv python install: {}", e));
            // Non-fatal — Python 3.11 might already be on PATH
        }

        // Create the venv
        {
            let mut mgr = manager.lock().await;
            mgr.append_build_log(
                &service_id,
                &format!("\n$ {} venv --python 3.11 --seed env", uv),
            );
        }
        emit_status(&manager, &handle).await;

        let mut venv_cmd = tokio::process::Command::new(&uv);
        venv_cmd
            .args(["venv", "--python", "3.11", "--seed", "env"])
            .current_dir(work_dir);
        hide_console_window(&mut venv_cmd);
        let venv_output = venv_cmd
            .output()
            .await
            .map_err(|e| format!("Failed to create venv: {}", e))?;

        {
            let mut mgr = manager.lock().await;
            let stdout = String::from_utf8_lossy(&venv_output.stdout);
            let stderr = String::from_utf8_lossy(&venv_output.stderr);
            if !stdout.is_empty() {
                mgr.append_build_log(&service_id, &stdout);
            }
            if !stderr.is_empty() {
                mgr.append_build_log(&service_id, &stderr);
            }
        }

        if !venv_output.status.success() {
            return Err("Failed to create virtual environment with uv".to_string());
        }
    } else {
        let mut mgr = manager.lock().await;
        mgr.set_build_step(&service_id, 1, "Virtual environment already exists");
        mgr.append_build_log(&service_id, "Venv already exists, skipping creation");
    }
    emit_status(&manager, &handle).await;

    let python_exe = env_dir.join("Scripts").join("python.exe");

    // Execute build steps — translate "pip install ..." to "uv pip install ... --python ..."
    for (i, step) in build_info.build_steps.iter().enumerate() {
        let step_num = i + 2; // offset by 2 (uv bootstrap + venv creation)

        // Handle xformers shim specially
        if step == "install_xformers_shim" {
            {
                let mut mgr = manager.lock().await;
                mgr.set_build_step(&service_id, step_num, "Installing xformers SDPA shim...");
                mgr.append_build_log(
                    &service_id,
                    &format!(
                        "\n--- Step {}/{}: install_xformers_shim ---",
                        i + 1,
                        build_info.build_steps.len()
                    ),
                );
            }
            emit_status(&manager, &handle).await;

            service_manager::install_xformers_shim(env_dir)?;

            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(&service_id, "xformers SDPA shim installed successfully");
            }
            continue;
        }

        // Translate pip commands to uv pip commands
        let (program, args) = if step.starts_with("pip ") {
            // "pip install ..." -> uv pip install ... --python env/Scripts/python.exe
            let pip_args = &step[4..]; // everything after "pip "
            let mut arg_vec: Vec<String> = vec!["pip".to_string()];
            arg_vec.extend(pip_args.split_whitespace().map(|s| s.to_string()));
            arg_vec.push("--python".to_string());
            arg_vec.push(python_exe.to_str().unwrap().to_string());
            (uv.clone(), arg_vec)
        } else {
            let parts: Vec<&str> = step.splitn(2, ' ').collect();
            let args = parts
                .get(1)
                .unwrap_or(&"")
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            (parts[0].to_string(), args)
        };

        let uv_command_str = format!("{} {}", program, args.join(" "));

        // Truncate step label for UI display
        let short_label = if uv_command_str.len() > 70 {
            format!("{}...", &uv_command_str[..67])
        } else {
            uv_command_str.clone()
        };

        {
            let mut mgr = manager.lock().await;
            mgr.set_build_step(
                &service_id,
                step_num,
                &format!(
                    "Step {}/{}: {}",
                    i + 1,
                    build_info.build_steps.len(),
                    short_label
                ),
            );
            mgr.append_build_log(
                &service_id,
                &format!(
                    "\n--- Step {}/{} ---\n$ {}",
                    i + 1,
                    build_info.build_steps.len(),
                    uv_command_str
                ),
            );
        }
        emit_status(&manager, &handle).await;

        // Run the command with streamed output
        let mut child_cmd = tokio::process::Command::new(&program);
        child_cmd
            .args(&args)
            .current_dir(work_dir)
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        hide_console_window(&mut child_cmd);
        let mut child = child_cmd
            .spawn()
            .map_err(|e| format!("Build step failed '{}': {}", uv_command_str, e))?;

        // Stream stdout and stderr
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let mgr_stdout = manager.clone();
        let sid_stdout = service_id.clone();

        let stdout_task = tauri::async_runtime::spawn(async move {
            if let Some(stdout) = stdout {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let mut mgr = mgr_stdout.lock().await;
                    mgr.append_build_log(&sid_stdout, &line);
                }
            }
        });

        let mgr_stderr = manager.clone();
        let sid_stderr = service_id.clone();

        let stderr_task = tauri::async_runtime::spawn(async move {
            if let Some(stderr) = stderr {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let mut mgr = mgr_stderr.lock().await;
                    mgr.append_build_log(&sid_stderr, &line);
                }
            }
        });

        let _ = stdout_task.await;
        let _ = stderr_task.await;

        let exit_status = child
            .wait()
            .await
            .map_err(|e| format!("Failed to wait for process: {}", e))?;

        if !exit_status.success() {
            let msg = format!(
                "Build step failed (exit code {}): {}",
                exit_status.code().unwrap_or(-1),
                uv_command_str
            );
            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(&service_id, &format!("\nERROR: {}", msg));
            }
            emit_status(&manager, &handle).await;
            return Err(msg);
        }

        {
            let mut mgr = manager.lock().await;
            mgr.append_build_log(&service_id, "OK");
        }
        emit_status(&manager, &handle).await;
    }

    {
        let mut mgr = manager.lock().await;
        mgr.append_build_log(
            &service_id,
            &format!("\n=== Build complete for {} ===", service_id),
        );
    }

    Ok(())
}

/// Run a command and stream its output to the build log (non-fatal helper)
async fn run_command_streamed(
    program: &str,
    args: &[&str],
    work_dir: &std::path::Path,
    service_id: &str,
    manager: &Arc<Mutex<ServiceManager>>,
) -> Result<(), String> {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args).current_dir(work_dir);
    hide_console_window(&mut cmd);
    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run {} {:?}: {}", program, args, e))?;

    {
        let mut mgr = manager.lock().await;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.is_empty() {
            mgr.append_build_log(service_id, &stdout);
        }
        if !stderr.is_empty() {
            mgr.append_build_log(service_id, &stderr);
        }
    }

    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "Command failed with exit code {}",
            output.status.code().unwrap_or(-1)
        ))
    }
}

/// Emit current service status to frontend
async fn emit_status(manager: &Arc<Mutex<ServiceManager>>, handle: &tauri::AppHandle) {
    let mgr = manager.lock().await;
    let info = mgr.get_service_info();
    drop(mgr);
    let _ = handle.emit("services-updated", &info);
}

#[tauri::command]
async fn get_service_log(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<String, String> {
    let mgr = manager.lock().await;
    mgr.get_log(&service_id).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Model management commands
// ---------------------------------------------------------------------------

#[tauri::command]
async fn get_models(
    model_mgr: tauri::State<'_, ModelState>,
) -> Result<Vec<model_manager::ModelEntry>, String> {
    let mgr = model_mgr.lock().await;
    let mut models = mgr.get_gary_models();
    models.extend(mgr.get_jerry_models());
    models.extend(mgr.get_carey_models());
    models.extend(mgr.get_foundation_models());
    Ok(models)
}

#[tauri::command]
async fn download_model(
    model_id: String,
    service_id: Option<String>,
    model_mgr: tauri::State<'_, ModelState>,
    _svc_mgr: tauri::State<'_, ManagerState>,
    repo_root: tauri::State<'_, std::path::PathBuf>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    // Pick the Python env based on which service the model belongs to.
    let svc = service_id.as_deref().unwrap_or("gary");
    let env_dir = match svc {
        "stable-audio" => "stable-audio",
        "carey" => "carey",
        "foundation" => "foundation",
        _ => "gary",
    };
    let python_exe = repo_root
        .join("services")
        .join(env_dir)
        .join("env")
        .join("Scripts")
        .join("python.exe");

    if !python_exe.exists() {
        let label = match env_dir {
            "stable-audio" => "Jerry (Stable Audio)",
            "carey" => "Carey (ACE-Step)",
            "foundation" => "Foundation-1",
            _ => "Gary",
        };
        return Err(format!(
            "{}'s Python environment is not built yet. Build it first.",
            label
        ));
    }

    {
        let mut mgr = model_mgr.lock().await;
        mgr.set_download_started(&model_id);
    }

    let mgr_clone = model_mgr.inner().clone();
    let handle = app_handle.clone();
    let root: std::path::PathBuf = repo_root.to_path_buf();

    // Carey models download to checkpoints/ dir, not HF cache
    if model_id.starts_with("carey::") {
        let checkpoint_dir = root.join("services").join("carey").join("checkpoints");
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_carey_model(
                model_id,
                python_exe,
                checkpoint_dir,
                mgr_clone,
                handle,
            )
            .await;
        });
    } else if model_id.starts_with("foundation::") {
        // Foundation models go to %APPDATA%/Gary4JUCE/models/foundation-1/
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
            format!("{}\\AppData\\Roaming", home)
        });
        let model_dir = std::path::PathBuf::from(appdata)
            .join("Gary4JUCE")
            .join("models")
            .join("foundation-1");
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_foundation_model(
                model_id, python_exe, model_dir, mgr_clone, handle,
            )
            .await;
        });
    } else {
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_model(model_id, python_exe, mgr_clone, handle).await;
        });
    }

    Ok(())
}

#[tauri::command]
async fn fetch_jerry_checkpoints(
    repo: String,
    model_mgr: tauri::State<'_, ModelState>,
    app_handle: tauri::AppHandle,
) -> Result<Vec<String>, String> {
    // Jerry runs on port 8005
    let checkpoints = model_manager::fetch_checkpoints(repo.clone(), 8005).await?;

    // Store in model manager so they appear in the model list
    {
        let mut mgr = model_mgr.lock().await;
        mgr.set_finetune_checkpoints(&repo, checkpoints.clone());
    }

    // Emit updated model list
    let mgr_arc = model_mgr.inner().clone();
    model_manager::emit_model_status(&mgr_arc, &app_handle).await;

    Ok(checkpoints)
}

#[tauri::command]
async fn get_download_progress(
    model_mgr: tauri::State<'_, ModelState>,
) -> Result<Vec<model_manager::DownloadProgress>, String> {
    let mgr = model_mgr.lock().await;
    Ok(mgr.get_download_progress())
}

#[tauri::command]
fn get_carey_lora_state(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<CareyLoraState, String> {
    build_carey_lora_state(repo_root.inner())
}

#[tauri::command]
async fn upsert_carey_lora(
    name: String,
    checkpoint_path: String,
    captions_path: Option<String>,
    model_family: Option<String>,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<CareyLoraState, String> {
    let normalized_name = sanitize_lora_name(&name)
        .ok_or_else(|| "LoRA name must use lowercase letters, numbers, '-' or '_'".to_string())?;
    let checkpoint_dir = PathBuf::from(checkpoint_path.trim());
    if !looks_like_lora_checkpoint_dir(&checkpoint_dir) {
        return Err(format!(
            "{} does not look like a LoRA checkpoint folder",
            checkpoint_dir.display()
        ));
    }

    let normalized_captions_path = captions_path
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let captions_dir = normalized_captions_path.as_ref().map(PathBuf::from);
    if let Some(captions_dir) = captions_dir.as_ref() {
        if !captions_dir.is_dir() {
            return Err(format!(
                "{} is not a valid captions/source folder",
                captions_dir.display()
            ));
        }
    }

    let mut catalog = read_carey_lora_catalog()?;
    let existing = catalog.get(&normalized_name);
    let (scale, backends, inferred_model_family) = load_carey_lora_metadata(
        &checkpoint_dir,
        captions_dir.as_deref(),
        existing,
    );
    let resolved_model_family = model_family
        .as_deref()
        .map(normalize_lora_model_family)
        .unwrap_or(inferred_model_family);

    catalog.insert(
        normalized_name,
        CareyLoraCatalogEntry {
            path: checkpoint_dir.to_string_lossy().to_string(),
            captions_path: normalized_captions_path,
            scale,
            backends,
            model_family: resolved_model_family,
        },
    );
    save_carey_lora_catalog(&catalog)?;

    let state = build_carey_lora_state(repo_root.inner())?;
    let _ = try_reload_carey_admin().await;
    Ok(state)
}

#[tauri::command]
async fn remove_carey_lora(
    name: String,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<CareyLoraState, String> {
    let normalized_name = sanitize_lora_name(&name)
        .ok_or_else(|| "Invalid LoRA name".to_string())?;
    let mut catalog = read_carey_lora_catalog()?;
    catalog.remove(&normalized_name);
    save_carey_lora_catalog(&catalog)?;

    let state = build_carey_lora_state(repo_root.inner())?;
    let _ = try_reload_carey_admin().await;
    Ok(state)
}

#[tauri::command]
async fn build_carey_lora_captions(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<CareyCaptionsBuildResult, String> {
    let initial_state = build_carey_lora_state(repo_root.inner())?;
    let python_exe = repo_root
        .join("services")
        .join("carey")
        .join("env")
        .join("Scripts")
        .join("python.exe");
    if !python_exe.exists() {
        return Err("Carey must be built before captions can be generated.".to_string());
    }

    let script_path = repo_root.join("services").join("carey").join("build_captions.py");
    if !script_path.exists() {
        return Err(format!("Missing {}", script_path.display()));
    }

    if let Some(parent) = carey_captions_path().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }

    let mut cmd = tokio::process::Command::new(&python_exe);
    hide_console_window(&mut cmd);
    cmd.arg(&script_path)
        .current_dir(repo_root.join("services").join("carey"));

    for entry in &initial_state.entries {
        if !entry.registered || entry.caption_count == 0 {
            continue;
        }
        if let Some(resolved_captions_path) = entry.resolved_captions_path.as_ref() {
            cmd.arg("--lora")
                .arg(format!("{}:{}", entry.name, resolved_captions_path));
        }
    }

    cmd.arg("-o").arg(carey_captions_path());

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run build_captions.py: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let combined_output = match (stdout.is_empty(), stderr.is_empty()) {
        (false, false) => format!("{}\n{}", stdout, stderr),
        (false, true) => stdout,
        (true, false) => stderr,
        (true, true) => "No output".to_string(),
    };

    if !output.status.success() {
        return Err(combined_output);
    }

    merge_default_captions_into_file(repo_root.inner())?;
    let _ = try_reload_carey_admin().await;
    let state = build_carey_lora_state(repo_root.inner())?;
    Ok(CareyCaptionsBuildResult {
        state,
        output: combined_output,
    })
}

// ---------------------------------------------------------------------------
// HuggingFace token management
// ---------------------------------------------------------------------------

#[tauri::command]
fn get_hf_token() -> Result<Option<String>, String> {
    Ok(read_hf_token())
}

#[tauri::command]
fn save_hf_token(token: String) -> Result<(), String> {
    let path = hf_token_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("Cannot create config dir: {}", e))?;
    }
    std::fs::write(&path, token.trim()).map_err(|e| format!("Cannot save token: {}", e))?;
    Ok(())
}

#[tauri::command]
fn delete_hf_token() -> Result<(), String> {
    let path = hf_token_path();
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("Cannot delete token: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
fn get_app_settings() -> Result<AppSettings, String> {
    Ok(read_app_settings())
}

#[tauri::command]
fn save_app_settings(settings: AppSettingsPatch) -> Result<AppSettings, String> {
    let merged = merge_app_settings(settings);
    save_app_settings_file(&merged)?;
    Ok(merged)
}

#[tauri::command]
async fn check_for_app_update(
    include_skipped: bool,
    app_handle: tauri::AppHandle,
    pending_update: tauri::State<'_, update::PendingNativeUpdate>,
) -> Result<update::AppUpdateCheck, String> {
    if !update::app_updater_enabled() {
        return Err("App updater is disabled in this build".to_string());
    }

    let current_settings = read_app_settings();
    let mut result = update::check_for_update(
        env!("CARGO_PKG_VERSION"),
        current_settings.skipped_update_version.as_deref(),
        include_skipped,
    )
    .await?;

    if result.update_available {
        match update::check_native_update(
            &app_handle,
            pending_update.inner(),
            &result.latest_version,
        )
        .await
        {
            Ok(available) => {
                result.in_app_install_available = available;
            }
            Err(error) => {
                log::warn!("Native updater check unavailable: {}", error);
                update::clear_pending_native_update(pending_update.inner());
            }
        }
    } else {
        update::clear_pending_native_update(pending_update.inner());
    }

    let mut updated_settings = current_settings;
    updated_settings.last_update_check_epoch_ms = Some(result.checked_at_epoch_ms);
    save_app_settings_file(&updated_settings)?;

    Ok(result)
}

#[tauri::command]
async fn install_app_update(
    app_handle: tauri::AppHandle,
    pending_update: tauri::State<'_, update::PendingNativeUpdate>,
) -> Result<(), String> {
    update::install_native_update(&app_handle, pending_update.inner()).await
}

#[tauri::command]
async fn resolve_close_request(
    action: String,
    remember_choice: bool,
    manager: tauri::State<'_, ManagerState>,
    app_handle: tauri::AppHandle,
) -> Result<AppSettings, String> {
    let close_action = match action.as_str() {
        "tray" => CloseActionOnX::Tray,
        "quit" => CloseActionOnX::Quit,
        "ask" => CloseActionOnX::Ask,
        other => return Err(format!("Unknown close action: {}", other)),
    };

    let settings = if remember_choice {
        let merged = merge_app_settings(AppSettingsPatch {
            close_action_on_x: Some(close_action),
            ..Default::default()
        });
        save_app_settings_file(&merged)?;
        merged
    } else {
        read_app_settings()
    };

    match close_action {
        CloseActionOnX::Tray => {
            if let Some(window) = app_handle.get_webview_window("main") {
                let _ = window.hide();
            }
        }
        CloseActionOnX::Quit => {
            quit_application(&app_handle, manager.inner()).await;
        }
        CloseActionOnX::Ask => {}
    }

    Ok(settings)
}

#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", &url])
            .spawn()
            .map_err(|e| format!("Failed to open URL: {}", e))?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("Failed to open URL: {}", e))?;
    }
    Ok(())
}
