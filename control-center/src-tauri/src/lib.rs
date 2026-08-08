mod manifest;
mod model_manager;
mod service_manager;
mod storage;
mod update;
mod workload_job;

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
struct Sa3LoudnessSettings {
    #[serde(default = "default_sa3_peak_normalize_db")]
    peak_normalize_db: String,
    #[serde(default = "default_sa3_limiter_ceiling_db")]
    limiter_ceiling_db: String,
    #[serde(default = "default_sa3_latent_rescale")]
    latent_rescale: String,
    #[serde(default = "default_sa3_latent_shift")]
    latent_shift: String,
    #[serde(default)]
    latent_target_std: String,
    #[serde(default = "default_sa3_continuation_tail_pad")]
    continuation_tail_pad: String,
}

impl Default for Sa3LoudnessSettings {
    fn default() -> Self {
        Self {
            peak_normalize_db: default_sa3_peak_normalize_db(),
            limiter_ceiling_db: default_sa3_limiter_ceiling_db(),
            latent_rescale: default_sa3_latent_rescale(),
            latent_shift: default_sa3_latent_shift(),
            latent_target_std: String::new(),
            continuation_tail_pad: default_sa3_continuation_tail_pad(),
        }
    }
}

fn default_sa3_peak_normalize_db() -> String {
    "2.0".to_string()
}

fn default_sa3_limiter_ceiling_db() -> String {
    "-0.3".to_string()
}

fn default_sa3_latent_rescale() -> String {
    "1.0".to_string()
}

fn default_sa3_latent_shift() -> String {
    "0.0".to_string()
}

fn default_sa3_continuation_tail_pad() -> String {
    "6".to_string()
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sa3LoudnessSettingsPatch {
    #[serde(default)]
    peak_normalize_db: Option<String>,
    #[serde(default)]
    limiter_ceiling_db: Option<String>,
    #[serde(default)]
    latent_rescale: Option<String>,
    #[serde(default)]
    latent_shift: Option<String>,
    #[serde(default)]
    latent_target_std: Option<String>,
    #[serde(default)]
    continuation_tail_pad: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppSettings {
    #[serde(default)]
    melodyflow_use_flash_attn: bool,
    #[serde(default)]
    carey_use_xl_models: bool,
    #[serde(default)]
    carey_use_scrag_vae: bool,
    #[serde(default)]
    sa3_loudness: Sa3LoudnessSettings,
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
            carey_use_scrag_vae: false,
            sa3_loudness: Sa3LoudnessSettings::default(),
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
    carey_use_scrag_vae: Option<bool>,
    #[serde(default)]
    sa3_loudness: Option<Sa3LoudnessSettingsPatch>,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct CareyAceSidecarFields {
    #[serde(default)]
    caption: String,
    #[serde(default)]
    genre: String,
    #[serde(default)]
    bpm: String,
    #[serde(default)]
    bpm_source: String,
    #[serde(default)]
    lm_bpm: String,
    #[serde(default)]
    local_bpm: String,
    #[serde(default)]
    filename_bpm: String,
    #[serde(default)]
    keyscale: String,
    #[serde(default)]
    key_source: String,
    #[serde(default)]
    lm_keyscale: String,
    #[serde(default)]
    local_keyscale: String,
    #[serde(default)]
    timesignature: String,
    #[serde(default)]
    language: String,
    #[serde(default)]
    is_instrumental: bool,
    #[serde(default)]
    custom_tag: String,
    #[serde(default)]
    lyrics: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CareyAceDatasetSidecarEntry {
    audio_path: String,
    relative_path: String,
    sidecar_path: String,
    exists: bool,
    raw_content: String,
    fields: CareyAceSidecarFields,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CareyAceDatasetSidecarUpdate {
    audio_path: String,
    fields: CareyAceSidecarFields,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CareyAceDatasetSidecarSaveResult {
    saved: usize,
    removed: usize,
    entries: Vec<CareyAceDatasetSidecarEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CareyAceTrainingState {
    #[serde(default)]
    job_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    status: String,
    #[serde(default)]
    phase: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    pid: Option<u32>,
    #[serde(default)]
    owner_pid: Option<u32>,
    #[serde(default)]
    launcher_pid: Option<u32>,
    #[serde(default)]
    child_pid: Option<u32>,
    #[serde(default)]
    run_dir: Option<String>,
    #[serde(default)]
    log_path: Option<String>,
    #[serde(default)]
    cancel_path: Option<String>,
    #[serde(default)]
    final_checkpoint_path: Option<String>,
    #[serde(default)]
    current_step: Option<u32>,
    #[serde(default)]
    max_steps: Option<u32>,
    #[serde(default)]
    current_file: Option<u32>,
    #[serde(default)]
    total_files: Option<u32>,
    #[serde(default)]
    sample_count: Option<u32>,
    #[serde(default)]
    captioned_count: Option<u32>,
    #[serde(default)]
    caption_lm_model: Option<String>,
    #[serde(default)]
    model_family: Option<String>,
    #[serde(default)]
    adapter_type: Option<String>,
    #[serde(default)]
    captions_path: Option<String>,
    #[serde(default)]
    dataset_json_path: Option<String>,
    #[serde(default)]
    training_plan_path: Option<String>,
    #[serde(default)]
    result_path: Option<String>,
    #[serde(default)]
    registered_lora_name: Option<String>,
    #[serde(default)]
    log_tail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sa3TrainingCheckpoint {
    step: u32,
    epoch: u32,
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sa3LoraCatalogEntry {
    path: String,
    #[serde(default)]
    prompts_path: Option<String>,
    #[serde(default = "default_lora_scale")]
    strength: f64,
    #[serde(default)]
    training_job_id: Option<String>,
    #[serde(default)]
    training_checkpoints: Vec<Sa3TrainingCheckpoint>,
    #[serde(default)]
    selected_training_step: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Sa3LoraEntry {
    name: String,
    path: String,
    prompts_path: Option<String>,
    resolved_prompts_path: Option<String>,
    prompt_file_path: String,
    prompt_file_exists: bool,
    prompt_count: usize,
    caption_count: usize,
    strength: f64,
    checkpoint_exists: bool,
    registered: bool,
    training_job_id: Option<String>,
    training_checkpoints: Vec<Sa3TrainingCheckpoint>,
    selected_training_step: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Sa3LoraState {
    entries: Vec<Sa3LoraEntry>,
    pools: HashMap<String, usize>,
    catalog_path: String,
    registry_path: String,
    prompts_dir: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Sa3PromptsBuildResult {
    state: Sa3LoraState,
    output: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LegacyStorageCleanupItem {
    id: String,
    label: String,
    path: String,
    bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LegacyLoraMigrationCandidate {
    service: String,
    name: String,
    source_path: String,
    target_path: String,
    bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LegacyStorageMaintenanceInfo {
    active_root: String,
    legacy_root: String,
    default_hf_cache_root: String,
    cleanup_items: Vec<LegacyStorageCleanupItem>,
    lora_candidates: Vec<LegacyLoraMigrationCandidate>,
    total_cleanup_bytes: u64,
    total_lora_bytes: u64,
    can_cleanup: bool,
    can_migrate_loras: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LegacyStorageMaintenanceResult {
    info: LegacyStorageMaintenanceInfo,
    migrated_loras: usize,
    cleaned_items: usize,
    errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Sa3DatasetSidecarEntry {
    audio_path: String,
    relative_path: String,
    sidecar_path: String,
    content: String,
    exists: bool,
    json_sidecar_exists: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sa3DatasetSidecarUpdate {
    audio_path: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Sa3DatasetSidecarSaveResult {
    saved: usize,
    removed: usize,
    entries: Vec<Sa3DatasetSidecarEntry>,
}

// Mirrors the snake_case JSON that services/sa3/analyze_audio.py prints to stdout.
#[derive(Debug, Deserialize)]
struct Sa3AnalyzerOutput {
    ok: bool,
    #[serde(default)]
    bpm: Option<i64>,
    #[serde(default)]
    keyscale: String,
    #[serde(default)]
    suggestion: String,
    #[serde(default)]
    bpm_confidence: Option<f64>,
    #[serde(default)]
    key_confidence: Option<f64>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Sa3TrackMetadataSuggestion {
    bpm: Option<i64>,
    keyscale: String,
    suggestion: String,
    bpm_confidence: Option<f64>,
    key_confidence: Option<f64>,
}

// Mirrors the status.json that services/carey/sa3_autolabel.py writes via update_status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sa3AutolabelState {
    #[serde(default)]
    status: String, // "" (idle) | starting | running | completed | cancelled | failed
    #[serde(default)]
    phase: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    total: u32,
    #[serde(default)]
    done: u32,
    #[serde(default)]
    current_path: String,
    #[serde(default)]
    style: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    pid: Option<u32>,
    #[serde(default)]
    owner_pid: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Sa3AutolabelAvailability {
    available: bool,
    carey_built: bool,
    captioner_downloaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sa3LoraTrainingState {
    #[serde(default)]
    job_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    status: String,
    #[serde(default)]
    phase: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    pid: Option<u32>,
    #[serde(default)]
    owner_pid: Option<u32>,
    #[serde(default)]
    launcher_pid: Option<u32>,
    #[serde(default)]
    child_pid: Option<u32>,
    #[serde(default)]
    run_dir: Option<String>,
    #[serde(default)]
    log_path: Option<String>,
    #[serde(default)]
    cancel_path: Option<String>,
    #[serde(default)]
    final_checkpoint_path: Option<String>,
    #[serde(default)]
    current_step: Option<u32>,
    #[serde(default)]
    max_steps: Option<u32>,
    #[serde(default)]
    log_tail: String,
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
    gary4juce_runtime_root().join("hf_token.txt")
}

fn legacy_hf_token_path() -> std::path::PathBuf {
    storage::legacy_runtime_root().join("hf_token.txt")
}

fn app_settings_path() -> std::path::PathBuf {
    gary4juce_runtime_root().join("app_settings.json")
}

fn legacy_app_settings_path() -> std::path::PathBuf {
    storage::legacy_runtime_root().join("app_settings.json")
}

fn gary4juce_runtime_root() -> PathBuf {
    storage::active_runtime_root()
}

fn gary4local_local_data_root() -> PathBuf {
    storage::local_data_root()
}

fn startup_diagnostic_log_path() -> PathBuf {
    gary4local_local_data_root()
        .join("logs")
        .join("startup-panic.log")
}

fn append_startup_diagnostic(message: &str) {
    let path = startup_diagnostic_log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "unknown-time".to_string());

    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };

    let _ = std::io::Write::write_all(
        &mut file,
        format!("\n===== {} =====\n{}\n", timestamp, message).as_bytes(),
    );
}

fn install_panic_diagnostics() {
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        let message = format!("gary4local panic: {}\n\n{}", panic_info, backtrace);
        append_startup_diagnostic(&message);
        eprintln!("{}", message);
    }));
}

#[cfg(target_os = "windows")]
fn webview2_user_data_folder() -> PathBuf {
    gary4local_local_data_root().join("WebView2-v2")
}

#[cfg(target_os = "windows")]
fn configure_webview2_user_data_folder() -> Option<PathBuf> {
    if std::env::var_os("WEBVIEW2_USER_DATA_FOLDER").is_some() {
        return None;
    }

    let folder = webview2_user_data_folder();
    if let Err(error) = std::fs::create_dir_all(&folder) {
        eprintln!(
            "Failed to create WebView2 user data folder {}: {}",
            folder.display(),
            error
        );
        return None;
    }

    std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", &folder);
    Some(folder)
}

#[cfg(not(target_os = "windows"))]
fn configure_webview2_user_data_folder() -> Option<PathBuf> {
    None
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

fn carey_training_dir() -> PathBuf {
    carey_runtime_dir().join("training")
}

fn carey_training_jobs_dir() -> PathBuf {
    carey_training_dir().join("jobs")
}

fn carey_training_logs_dir() -> PathBuf {
    carey_training_dir().join("logs")
}

fn carey_training_current_job_path() -> PathBuf {
    carey_training_dir().join("current_job.json")
}

fn carey_checkpoint_dir(runtime_root: &Path) -> PathBuf {
    runtime_root
        .join("services")
        .join("carey")
        .join("checkpoints")
}

fn sa3_runtime_dir() -> PathBuf {
    gary4juce_runtime_root().join("sa3")
}

fn sa3_lora_catalog_path() -> PathBuf {
    sa3_runtime_dir().join("lora_catalog.json")
}

fn sa3_lora_registry_path() -> PathBuf {
    sa3_runtime_dir().join("lora_registry.json")
}

fn sa3_prompts_dir() -> PathBuf {
    sa3_runtime_dir().join("prompts")
}

fn sa3_lora_dir() -> PathBuf {
    sa3_runtime_dir().join("loras")
}

fn sa3_training_dir() -> PathBuf {
    sa3_runtime_dir().join("training")
}

fn sa3_training_models_dir() -> PathBuf {
    sa3_training_dir().join("models")
}

fn sa3_training_jobs_dir() -> PathBuf {
    sa3_training_dir().join("jobs")
}

fn sa3_training_logs_dir() -> PathBuf {
    sa3_training_dir().join("logs")
}

fn sa3_training_current_job_path() -> PathBuf {
    sa3_training_dir().join("current_job.json")
}

fn sa3_autolabel_dir() -> PathBuf {
    sa3_runtime_dir().join("autolabel")
}

fn sa3_autolabel_status_path() -> PathBuf {
    sa3_autolabel_dir().join("status.json")
}

fn sa3_autolabel_cancel_path() -> PathBuf {
    sa3_autolabel_dir().join("cancel.requested")
}

fn sa3_autolabel_log_path() -> PathBuf {
    sa3_autolabel_dir().join("autolabel.log")
}

fn sa3_autolabel_current_job_path() -> PathBuf {
    sa3_autolabel_dir().join("current_job.json")
}

fn bundled_sa3_default_prompts_path(runtime_root: &Path) -> PathBuf {
    runtime_root
        .join("services")
        .join("sa3")
        .join("prompts")
        .join("defaults.json")
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

fn apply_runtime_env(cmd: &mut tokio::process::Command, runtime_root: &Path) {
    for (key, value) in storage::runtime_env_vars(runtime_root) {
        cmd.env(key, value);
    }
}

fn hide_std_console_window(cmd: &mut std::process::Command) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
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
        match std::fs::read(&icon_path) {
            Ok(icon_bytes) => icon_bytes.hash(&mut hasher),
            Err(error) => {
                let message = format!(
                    "Skipping bundled icon hash because {} could not be read: {}",
                    icon_path.display(),
                    error
                );
                log::warn!("{}", message);
                append_startup_diagnostic(&message);
            }
        }
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
        if let Err(error) = std::fs::copy(&bundle_icon, &runtime_icon) {
            let message = format!(
                "Cannot copy {} to {}: {}",
                bundle_icon.display(),
                runtime_icon.display(),
                error
            );
            log::warn!("Non-fatal runtime icon sync warning: {}", message);
            append_startup_diagnostic(&format!("Non-fatal runtime icon sync warning: {}", message));
        }
    }

    std::fs::write(&stamp_path, desired_stamp)
        .map_err(|e| format!("Cannot write {}: {}", stamp_path.display(), e))?;

    Ok(())
}

/// Read the HF token from our stored file, falling back to system env.
fn read_hf_token() -> Option<String> {
    // Try the active runtime root first, then the legacy AppData root so
    // upgrades and storage moves do not strand an already-saved token.
    for path in [hf_token_path(), legacy_hf_token_path()] {
        if let Ok(token) = std::fs::read_to_string(&path) {
            let trimmed = token.trim().to_string();
            if !trimmed.is_empty() {
                return Some(trimmed);
            }
        }
    }
    // Fall back to system environment variables
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .or_else(|| {
            std::env::var("HUGGING_FACE_HUB_TOKEN")
                .ok()
                .filter(|t| !t.is_empty())
        })
}

fn read_app_settings() -> AppSettings {
    for path in [app_settings_path(), legacy_app_settings_path()] {
        if let Ok(raw) = std::fs::read_to_string(&path) {
            return serde_json::from_str::<AppSettings>(&raw).unwrap_or_default();
        }
    }

    AppSettings::default()
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

fn clean_setting_value(value: String) -> String {
    value.trim().to_string()
}

fn merge_sa3_loudness_settings(current: &mut Sa3LoudnessSettings, patch: Sa3LoudnessSettingsPatch) {
    if let Some(value) = patch.peak_normalize_db {
        current.peak_normalize_db = clean_setting_value(value);
    }
    if let Some(value) = patch.limiter_ceiling_db {
        current.limiter_ceiling_db = clean_setting_value(value);
    }
    if let Some(value) = patch.latent_rescale {
        current.latent_rescale = clean_setting_value(value);
    }
    if let Some(value) = patch.latent_shift {
        current.latent_shift = clean_setting_value(value);
    }
    if let Some(value) = patch.latent_target_std {
        current.latent_target_std = clean_setting_value(value);
    }
    if let Some(value) = patch.continuation_tail_pad {
        current.continuation_tail_pad = clean_setting_value(value);
    }
}

fn merge_app_settings(patch: AppSettingsPatch) -> AppSettings {
    let mut current = read_app_settings();

    if let Some(melodyflow_use_flash_attn) = patch.melodyflow_use_flash_attn {
        current.melodyflow_use_flash_attn = melodyflow_use_flash_attn;
    }

    if let Some(carey_use_xl_models) = patch.carey_use_xl_models {
        current.carey_use_xl_models = carey_use_xl_models;
    }

    if let Some(carey_use_scrag_vae) = patch.carey_use_scrag_vae {
        current.carey_use_scrag_vae = carey_use_scrag_vae;
    }

    if let Some(sa3_loudness) = patch.sa3_loudness {
        merge_sa3_loudness_settings(&mut current.sa3_loudness, sa3_loudness);
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

fn is_sa3_dataset_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav" | "flac" | "mp3" | "ogg" | "opus" | "m4a" | "aiff" | "aif"
            )
        })
        .unwrap_or(false)
}

fn user_facing_windows_path(path: &Path) -> String {
    let value = path.to_string_lossy();
    if let Some(unc_path) = value.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{}", unc_path);
    }
    value.strip_prefix(r"\\?\").unwrap_or(&value).to_string()
}

fn canonical_sa3_dataset_root(dataset_path: &str) -> Result<PathBuf, String> {
    let trimmed = dataset_path.trim();
    if trimmed.is_empty() {
        return Err("Choose a dataset folder first".to_string());
    }
    let requested = PathBuf::from(trimmed);
    if !requested.is_dir() {
        return Err(format!(
            "{} is not a valid dataset folder",
            requested.display()
        ));
    }
    requested
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", requested.display(), e))
}

fn collect_sa3_dataset_audio_files(
    root: &Path,
    directory: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let canonical_directory = directory
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", directory.display(), e))?;
    if !canonical_directory.starts_with(root) || !visited.insert(canonical_directory.clone()) {
        return Ok(());
    }

    for entry in read_dir_sorted(&canonical_directory)? {
        let Ok(canonical_entry) = entry.canonicalize() else {
            continue;
        };
        if !canonical_entry.starts_with(root) {
            continue;
        }
        if canonical_entry.is_dir() {
            collect_sa3_dataset_audio_files(root, &canonical_entry, visited, files)?;
        } else if canonical_entry.is_file() && is_sa3_dataset_audio_file(&canonical_entry) {
            files.push(canonical_entry);
        }
    }
    Ok(())
}

fn sa3_dataset_json_sidecar_exists(root: &Path, audio_path: &Path) -> bool {
    let mut candidates = vec![audio_path.with_extension("json")];
    if let (Some(parent), Some(stem)) = (audio_path.parent(), audio_path.file_stem()) {
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join("json").join(stem).with_extension("json"));
        }
    }
    candidates
        .into_iter()
        .any(|candidate| candidate.starts_with(root) && candidate.is_file())
}

fn read_sa3_dataset_sidecars(dataset_path: &str) -> Result<Vec<Sa3DatasetSidecarEntry>, String> {
    let root = canonical_sa3_dataset_root(dataset_path)?;
    let mut files = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_sa3_dataset_audio_files(&root, &root, &mut visited, &mut files)?;
    files.sort_by(|left, right| {
        left.strip_prefix(&root)
            .unwrap_or(left)
            .to_string_lossy()
            .to_lowercase()
            .cmp(
                &right
                    .strip_prefix(&root)
                    .unwrap_or(right)
                    .to_string_lossy()
                    .to_lowercase(),
            )
    });

    Ok(files
        .into_iter()
        .map(|audio_path| {
            let sidecar_path = audio_path.with_extension("txt");
            let exists = sidecar_path.is_file();
            let content = if exists {
                std::fs::read_to_string(&sidecar_path)
                    .unwrap_or_default()
                    .trim_start_matches('\u{feff}')
                    .to_string()
            } else {
                String::new()
            };
            Sa3DatasetSidecarEntry {
                relative_path: audio_path
                    .strip_prefix(&root)
                    .unwrap_or(&audio_path)
                    .to_string_lossy()
                    .to_string(),
                audio_path: user_facing_windows_path(&audio_path),
                sidecar_path: user_facing_windows_path(&sidecar_path),
                content,
                exists,
                json_sidecar_exists: sa3_dataset_json_sidecar_exists(&root, &audio_path),
            }
        })
        .collect())
}

fn save_sa3_dataset_sidecar_updates(
    dataset_path: &str,
    sidecars: Vec<Sa3DatasetSidecarUpdate>,
) -> Result<Sa3DatasetSidecarSaveResult, String> {
    let root = canonical_sa3_dataset_root(dataset_path)?;
    let mut saved = 0;
    let mut removed = 0;

    for sidecar in sidecars {
        let requested_audio = PathBuf::from(sidecar.audio_path.trim());
        let audio_path = requested_audio
            .canonicalize()
            .map_err(|e| format!("Cannot resolve {}: {}", requested_audio.display(), e))?;
        if !audio_path.starts_with(&root)
            || !audio_path.is_file()
            || !is_sa3_dataset_audio_file(&audio_path)
        {
            return Err(format!(
                "{} is not an audio file inside {}",
                audio_path.display(),
                root.display()
            ));
        }

        let sidecar_path = audio_path.with_extension("txt");
        let content = sidecar.content.trim();
        if content.is_empty() {
            if sidecar_path.is_file() {
                std::fs::remove_file(&sidecar_path)
                    .map_err(|e| format!("Cannot remove {}: {}", sidecar_path.display(), e))?;
                removed += 1;
            }
            continue;
        }

        std::fs::write(&sidecar_path, format!("{}\n", content))
            .map_err(|e| format!("Cannot save {}: {}", sidecar_path.display(), e))?;
        saved += 1;
    }

    Ok(Sa3DatasetSidecarSaveResult {
        saved,
        removed,
        entries: read_sa3_dataset_sidecars(dataset_path)?,
    })
}

fn is_carey_ace_dataset_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav" | "flac" | "mp3" | "ogg" | "opus" | "m4a"
            )
        })
        .unwrap_or(false)
}

fn canonical_carey_ace_dataset_root(dataset_path: &str) -> Result<PathBuf, String> {
    let trimmed = dataset_path.trim();
    if trimmed.is_empty() {
        return Err("Choose a dataset folder first".to_string());
    }
    let requested = PathBuf::from(trimmed);
    if !requested.is_dir() {
        return Err(format!(
            "{} is not a valid dataset folder",
            requested.display()
        ));
    }
    requested
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", requested.display(), e))
}

fn collect_carey_ace_dataset_audio_files(
    root: &Path,
    directory: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let canonical_directory = directory
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", directory.display(), e))?;
    if !canonical_directory.starts_with(root) || !visited.insert(canonical_directory.clone()) {
        return Ok(());
    }

    for entry in read_dir_sorted(&canonical_directory)? {
        let Ok(canonical_entry) = entry.canonicalize() else {
            continue;
        };
        if !canonical_entry.starts_with(root) {
            continue;
        }
        if canonical_entry.is_dir() {
            collect_carey_ace_dataset_audio_files(root, &canonical_entry, visited, files)?;
        } else if canonical_entry.is_file() && is_carey_ace_dataset_audio_file(&canonical_entry) {
            files.push(canonical_entry);
        }
    }
    Ok(())
}

fn normalize_carey_ace_sidecar_key(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().replace(' ', "_").as_str() {
        "caption" => Some("caption"),
        "genre" => Some("genre"),
        "bpm" => Some("bpm"),
        "bpm_source" => Some("bpm_source"),
        "lm_bpm" => Some("lm_bpm"),
        "local_bpm" => Some("local_bpm"),
        "filename_bpm" => Some("filename_bpm"),
        "key" | "keyscale" => Some("keyscale"),
        "key_source" => Some("key_source"),
        "lm_keyscale" => Some("lm_keyscale"),
        "local_keyscale" => Some("local_keyscale"),
        "signature" | "timesignature" | "time_signature" => Some("timesignature"),
        "language" => Some("language"),
        "is_instrumental" => Some("is_instrumental"),
        "custom_tag" => Some("custom_tag"),
        "lyrics" => Some("lyrics"),
        _ => None,
    }
}

fn commit_carey_ace_sidecar_value(
    values: &mut BTreeMap<String, String>,
    key: &mut Option<&'static str>,
    lines: &mut Vec<String>,
) {
    if let Some(name) = key.take() {
        values.insert(name.to_string(), lines.join("\n").trim().to_string());
        lines.clear();
    }
}

fn parse_carey_ace_sidecar_content(raw: &str) -> CareyAceSidecarFields {
    let mut values = BTreeMap::<String, String>::new();
    let mut current_key: Option<&'static str> = None;
    let mut current_lines: Vec<String> = Vec::new();
    let mut saw_known_key = false;

    for line in raw.trim_start_matches('\u{feff}').lines() {
        let parsed = line.split_once(':').and_then(|(key, value)| {
            if key
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == ' ')
            {
                normalize_carey_ace_sidecar_key(key).map(|name| (name, value.trim_start()))
            } else {
                None
            }
        });

        if let Some((name, value)) = parsed {
            commit_carey_ace_sidecar_value(&mut values, &mut current_key, &mut current_lines);
            saw_known_key = true;
            current_key = Some(name);
            current_lines.push(value.trim_end().to_string());
        } else if current_key.is_some() {
            current_lines.push(line.trim_end().to_string());
        }
    }
    commit_carey_ace_sidecar_value(&mut values, &mut current_key, &mut current_lines);

    if !saw_known_key && !raw.trim().is_empty() {
        return CareyAceSidecarFields {
            lyrics: raw.trim().to_string(),
            ..CareyAceSidecarFields::default()
        };
    }

    CareyAceSidecarFields {
        caption: values.remove("caption").unwrap_or_default(),
        genre: values.remove("genre").unwrap_or_default(),
        bpm: values.remove("bpm").unwrap_or_default(),
        bpm_source: values.remove("bpm_source").unwrap_or_default(),
        lm_bpm: values.remove("lm_bpm").unwrap_or_default(),
        local_bpm: values.remove("local_bpm").unwrap_or_default(),
        filename_bpm: values.remove("filename_bpm").unwrap_or_default(),
        keyscale: values.remove("keyscale").unwrap_or_default(),
        key_source: values.remove("key_source").unwrap_or_default(),
        lm_keyscale: values.remove("lm_keyscale").unwrap_or_default(),
        local_keyscale: values.remove("local_keyscale").unwrap_or_default(),
        timesignature: values.remove("timesignature").unwrap_or_default(),
        language: values.remove("language").unwrap_or_default(),
        is_instrumental: values
            .remove("is_instrumental")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false),
        custom_tag: values.remove("custom_tag").unwrap_or_default(),
        lyrics: values.remove("lyrics").unwrap_or_default(),
    }
}

fn clean_carey_ace_scalar(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn carey_ace_fields_are_empty(fields: &CareyAceSidecarFields) -> bool {
    fields.caption.trim().is_empty()
        && fields.genre.trim().is_empty()
        && fields.bpm.trim().is_empty()
        && fields.keyscale.trim().is_empty()
        && fields.timesignature.trim().is_empty()
        && fields.language.trim().is_empty()
        && fields.custom_tag.trim().is_empty()
        && fields.lyrics.trim().is_empty()
        && !fields.is_instrumental
}

fn render_carey_ace_sidecar(fields: &CareyAceSidecarFields) -> String {
    let mut lines = Vec::new();
    for (key, value) in [
        ("caption", fields.caption.as_str()),
        ("genre", fields.genre.as_str()),
        ("bpm", fields.bpm.as_str()),
        ("bpm_source", fields.bpm_source.as_str()),
        ("lm_bpm", fields.lm_bpm.as_str()),
        ("local_bpm", fields.local_bpm.as_str()),
        ("filename_bpm", fields.filename_bpm.as_str()),
        ("keyscale", fields.keyscale.as_str()),
        ("key_source", fields.key_source.as_str()),
        ("lm_keyscale", fields.lm_keyscale.as_str()),
        ("local_keyscale", fields.local_keyscale.as_str()),
        ("timesignature", fields.timesignature.as_str()),
        ("language", fields.language.as_str()),
    ] {
        let scalar = clean_carey_ace_scalar(value);
        if !scalar.is_empty() {
            lines.push(format!("{}: {}", key, scalar));
        }
    }
    lines.push(format!(
        "is_instrumental: {}",
        if fields.is_instrumental {
            "true"
        } else {
            "false"
        }
    ));
    let tag = clean_carey_ace_scalar(&fields.custom_tag);
    if !tag.is_empty() {
        lines.push(format!("custom_tag: {}", tag));
    }

    let mut lyrics = fields.lyrics.trim().to_string();
    if lyrics.is_empty() && fields.is_instrumental {
        lyrics = "[Instrumental]".to_string();
    }
    let mut lyric_lines = lyrics.lines();
    let first = lyric_lines.next().unwrap_or("");
    lines.push(format!("lyrics: {}", first));
    lines.extend(lyric_lines.map(|line| line.to_string()));
    format!("{}\n", lines.join("\n"))
}

fn read_carey_ace_dataset_sidecars(
    dataset_path: &str,
) -> Result<Vec<CareyAceDatasetSidecarEntry>, String> {
    let root = canonical_carey_ace_dataset_root(dataset_path)?;
    let mut files = Vec::new();
    let mut visited = std::collections::HashSet::new();
    collect_carey_ace_dataset_audio_files(&root, &root, &mut visited, &mut files)?;
    files.sort_by(|left, right| {
        left.strip_prefix(&root)
            .unwrap_or(left)
            .to_string_lossy()
            .to_lowercase()
            .cmp(
                &right
                    .strip_prefix(&root)
                    .unwrap_or(right)
                    .to_string_lossy()
                    .to_lowercase(),
            )
    });

    Ok(files
        .into_iter()
        .map(|audio_path| {
            let sidecar_path = audio_path.with_extension("txt");
            let exists = sidecar_path.is_file();
            let raw_content = if exists {
                std::fs::read_to_string(&sidecar_path)
                    .unwrap_or_default()
                    .trim_start_matches('\u{feff}')
                    .to_string()
            } else {
                String::new()
            };
            let fields = parse_carey_ace_sidecar_content(&raw_content);
            CareyAceDatasetSidecarEntry {
                relative_path: audio_path
                    .strip_prefix(&root)
                    .unwrap_or(&audio_path)
                    .to_string_lossy()
                    .to_string(),
                audio_path: user_facing_windows_path(&audio_path),
                sidecar_path: user_facing_windows_path(&sidecar_path),
                exists,
                raw_content,
                fields,
            }
        })
        .collect())
}

fn save_carey_ace_dataset_sidecar_updates(
    dataset_path: &str,
    sidecars: Vec<CareyAceDatasetSidecarUpdate>,
) -> Result<CareyAceDatasetSidecarSaveResult, String> {
    let root = canonical_carey_ace_dataset_root(dataset_path)?;
    let mut saved = 0;
    let mut removed = 0;

    for sidecar in sidecars {
        let requested_audio = PathBuf::from(sidecar.audio_path.trim());
        let audio_path = requested_audio
            .canonicalize()
            .map_err(|e| format!("Cannot resolve {}: {}", requested_audio.display(), e))?;
        if !audio_path.starts_with(&root)
            || !audio_path.is_file()
            || !is_carey_ace_dataset_audio_file(&audio_path)
        {
            return Err(format!(
                "{} is not an audio file inside {}",
                audio_path.display(),
                root.display()
            ));
        }

        let sidecar_path = audio_path.with_extension("txt");
        if carey_ace_fields_are_empty(&sidecar.fields) {
            if sidecar_path.is_file() {
                std::fs::remove_file(&sidecar_path)
                    .map_err(|e| format!("Cannot remove {}: {}", sidecar_path.display(), e))?;
                removed += 1;
            }
            continue;
        }

        std::fs::write(&sidecar_path, render_carey_ace_sidecar(&sidecar.fields))
            .map_err(|e| format!("Cannot save {}: {}", sidecar_path.display(), e))?;
        saved += 1;
    }

    Ok(CareyAceDatasetSidecarSaveResult {
        saved,
        removed,
        entries: read_carey_ace_dataset_sidecars(dataset_path)?,
    })
}

fn read_carey_lora_catalog_from(
    path: &Path,
) -> Result<BTreeMap<String, CareyLoraCatalogEntry>, String> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }

    let raw = std::fs::read_to_string(path)
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

fn read_carey_lora_catalog() -> Result<BTreeMap<String, CareyLoraCatalogEntry>, String> {
    read_carey_lora_catalog_from(&carey_lora_catalog_path())
}

fn save_carey_lora_catalog_to(
    path: &Path,
    catalog: &BTreeMap<String, CareyLoraCatalogEntry>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    let json = serde_json::to_string_pretty(catalog)
        .map_err(|e| format!("Cannot serialize LoRA catalog: {}", e))?;
    std::fs::write(path, json).map_err(|e| format!("Cannot save {}: {}", path.display(), e))?;
    Ok(())
}

fn save_carey_lora_catalog(
    catalog: &BTreeMap<String, CareyLoraCatalogEntry>,
) -> Result<(), String> {
    save_carey_lora_catalog_to(&carey_lora_catalog_path(), catalog)
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
    std::fs::write(&path, json).map_err(|e| format!("Cannot save {}: {}", path.display(), e))?;

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

fn looks_like_sa3_lora_checkpoint(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .map(|ext| {
                let ext = ext.to_string_lossy().to_lowercase();
                ext == "ckpt" || ext == "safetensors"
            })
            .unwrap_or(false)
}

fn sa3_prompt_file_path(name: &str) -> PathBuf {
    sa3_prompts_dir().join(format!("{}.json", name))
}

fn read_sa3_lora_catalog_from(
    path: &Path,
) -> Result<BTreeMap<String, Sa3LoraCatalogEntry>, String> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }

    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;
    let mut parsed: BTreeMap<String, Sa3LoraCatalogEntry> =
        serde_json::from_str(&raw).map_err(|e| format!("Invalid SA3 LoRA catalog JSON: {}", e))?;

    for entry in parsed.values_mut() {
        entry.path = entry.path.trim().to_string();
        entry.prompts_path = entry
            .prompts_path
            .take()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        if !entry.strength.is_finite() {
            entry.strength = default_lora_scale();
        }
        entry.training_job_id = entry
            .training_job_id
            .take()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        for checkpoint in &mut entry.training_checkpoints {
            checkpoint.path = checkpoint.path.trim().to_string();
        }
        entry
            .training_checkpoints
            .sort_by(|a, b| (a.step, a.epoch, &a.path).cmp(&(b.step, b.epoch, &b.path)));
        entry
            .training_checkpoints
            .dedup_by(|a, b| a.step == b.step && a.epoch == b.epoch && a.path == b.path);
    }

    Ok(parsed)
}

fn read_sa3_lora_catalog() -> Result<BTreeMap<String, Sa3LoraCatalogEntry>, String> {
    read_sa3_lora_catalog_from(&sa3_lora_catalog_path())
}

fn save_sa3_lora_catalog_to(
    path: &Path,
    catalog: &BTreeMap<String, Sa3LoraCatalogEntry>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    let json = serde_json::to_string_pretty(catalog)
        .map_err(|e| format!("Cannot serialize SA3 LoRA catalog: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Cannot save {}: {}", path.display(), e))?;
    Ok(())
}

fn save_sa3_lora_catalog(catalog: &BTreeMap<String, Sa3LoraCatalogEntry>) -> Result<(), String> {
    save_sa3_lora_catalog_to(&sa3_lora_catalog_path(), catalog)
}

fn resolve_sa3_prompts_source(entry: &Sa3LoraCatalogEntry) -> Option<PathBuf> {
    if let Some(prompts_path) = entry
        .prompts_path
        .as_ref()
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
    {
        return Some(prompts_path);
    }

    let checkpoint_path = PathBuf::from(&entry.path);
    if let Some(parent) = checkpoint_path.parent() {
        if count_caption_sidecars(parent) > 0 {
            return Some(parent.to_path_buf());
        }
    }

    None
}

fn read_sa3_prompt_count(path: &Path) -> usize {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return 0;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return 0;
    };
    let Some(dice) = value.get("dice").and_then(|value| value.as_object()) else {
        return 0;
    };
    dice.values()
        .map(|value| value.as_array().map(|items| items.len()).unwrap_or(0))
        .sum()
}

fn read_sa3_prompt_pools() -> HashMap<String, usize> {
    let mut pools = HashMap::new();
    let dir = sa3_prompts_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return pools;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file()
            || !path
                .extension()
                .map(|ext| ext.to_string_lossy().eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        {
            continue;
        }
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        pools.insert(stem, read_sa3_prompt_count(&path));
    }
    pools
}

fn build_sa3_lora_entries(catalog: &BTreeMap<String, Sa3LoraCatalogEntry>) -> Vec<Sa3LoraEntry> {
    catalog
        .iter()
        .map(|(name, entry)| {
            let checkpoint_path = PathBuf::from(&entry.path);
            let checkpoint_exists = looks_like_sa3_lora_checkpoint(&checkpoint_path);
            let training_checkpoints = entry
                .training_checkpoints
                .iter()
                .filter(|checkpoint| {
                    looks_like_sa3_lora_checkpoint(&PathBuf::from(&checkpoint.path))
                })
                .cloned()
                .collect::<Vec<_>>();
            let selected_training_step = entry.selected_training_step.filter(|step| {
                training_checkpoints
                    .iter()
                    .any(|checkpoint| checkpoint.step == *step)
            });
            let resolved_prompts_path = resolve_sa3_prompts_source(entry);
            let caption_count = resolved_prompts_path
                .as_ref()
                .map(|path| count_caption_sidecars(path))
                .unwrap_or(0);
            let prompt_file = sa3_prompt_file_path(name);
            let prompt_file_exists = prompt_file.is_file();
            let prompt_count = read_sa3_prompt_count(&prompt_file);

            Sa3LoraEntry {
                name: name.clone(),
                path: entry.path.clone(),
                prompts_path: entry.prompts_path.clone(),
                resolved_prompts_path: resolved_prompts_path
                    .as_ref()
                    .map(|path| path.to_string_lossy().to_string()),
                prompt_file_path: prompt_file.to_string_lossy().to_string(),
                prompt_file_exists,
                prompt_count,
                caption_count,
                strength: entry.strength,
                checkpoint_exists,
                registered: checkpoint_exists,
                training_job_id: entry.training_job_id.clone(),
                training_checkpoints,
                selected_training_step,
            }
        })
        .collect()
}

fn write_sa3_lora_registry(entries: &[Sa3LoraEntry]) -> Result<(), String> {
    let path = sa3_lora_registry_path();
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
                "strength": entry.strength,
            }),
        );
    }

    let json = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("Cannot serialize SA3 LoRA registry: {}", e))?;
    std::fs::write(&path, json).map_err(|e| format!("Cannot save {}: {}", path.display(), e))?;
    Ok(())
}

fn ensure_default_sa3_prompts(runtime_root: &Path) -> Result<(), String> {
    let source = bundled_sa3_default_prompts_path(runtime_root);
    if !source.is_file() {
        return Ok(());
    }

    let dest = sa3_prompts_dir().join("defaults.json");
    if dest.is_file() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    std::fs::copy(&source, &dest).map_err(|e| {
        format!(
            "Cannot copy {} to {}: {}",
            source.display(),
            dest.display(),
            e
        )
    })?;
    Ok(())
}

/// Regenerate the on-disk LoRA registry from the catalog.
///
/// The trainer only writes the *catalog*; the registry is what gary4juce reads.
/// Until this was called on training completion, a freshly trained LoRA stayed
/// invisible to gary4juce until the user happened to open the "add LoRAs" modal
/// (which calls build_sa3_lora_state, and that is what actually rewrote the
/// registry). Deliberately lighter than build_sa3_lora_state: no runtime_root and
/// no prompt-defaults work, so it is safe to call from a background task.
fn refresh_sa3_lora_registry() -> Result<(), String> {
    let catalog = read_sa3_lora_catalog()?;
    let entries = build_sa3_lora_entries(&catalog);
    write_sa3_lora_registry(&entries)
}

fn build_sa3_lora_state(runtime_root: &Path) -> Result<Sa3LoraState, String> {
    ensure_default_sa3_prompts(runtime_root)?;
    let catalog = read_sa3_lora_catalog()?;
    let entries = build_sa3_lora_entries(&catalog);
    write_sa3_lora_registry(&entries)?;
    Ok(Sa3LoraState {
        entries,
        pools: read_sa3_prompt_pools(),
        catalog_path: sa3_lora_catalog_path().to_string_lossy().to_string(),
        registry_path: sa3_lora_registry_path().to_string_lossy().to_string(),
        prompts_dir: sa3_prompts_dir().to_string_lossy().to_string(),
    })
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn default_hf_cache_root() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("huggingface")
}

fn known_legacy_hf_repos() -> &'static [&'static str] {
    &[
        "thepatch/vanya_ai_dnb_0.1",
        "thepatch/gary_orchestra_2",
        "thepatch/keygen-gary-v2-small-8",
        "thepatch/keygen-gary-v2-small-12",
        "thepatch/bleeps-medium",
        "thepatch/keygen-gary-medium-12",
        "thepatch/hoenn_lofi",
        "thepatch/bleeps-large-6",
        "thepatch/bleeps-large-8",
        "thepatch/bleeps-large-10",
        "thepatch/bleeps-large-14",
        "thepatch/bleeps-large-20",
        "thepatch/keygen-gary-v2-large-12",
        "thepatch/keygen-gary-v2-large-16",
        "stabilityai/stable-audio-open-small",
        "stabilityai/stable-audio-3-medium",
        "stabilityai/stable-audio-3-medium-base",
    ]
}

fn legacy_hf_cache_folder(hf_root: &Path, repo: &str) -> PathBuf {
    hf_root
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")))
}

fn path_size(path: &Path) -> u64 {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return 0;
    };

    if metadata.is_file() {
        return metadata.len();
    }
    if !metadata.is_dir() {
        return 0;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| path_size(&entry.path()))
        .sum()
}

fn add_cleanup_item(
    items: &mut Vec<LegacyStorageCleanupItem>,
    active_root: &Path,
    id: String,
    label: String,
    path: PathBuf,
) {
    if !path.exists() {
        return;
    }
    if path_is_inside(&path, active_root) || path_is_inside(active_root, &path) {
        return;
    }

    items.push(LegacyStorageCleanupItem {
        id,
        label,
        bytes: path_size(&path),
        path: display_path(&path),
    });
}

fn build_legacy_cleanup_items(
    active_root: &Path,
    legacy_root: &Path,
    default_hf_root: &Path,
) -> Vec<LegacyStorageCleanupItem> {
    let mut items = Vec::new();
    if storage::paths_equivalent(active_root, legacy_root) {
        return items;
    }

    for service_id in [
        "gary",
        "melodyflow",
        "stable-audio",
        "sa3",
        "carey",
        "foundation",
    ] {
        let service_dir = legacy_root.join("services").join(service_id);
        add_cleanup_item(
            &mut items,
            active_root,
            format!("{service_id}-env"),
            format!("{service_id} Python environment"),
            service_dir.join("env"),
        );
        add_cleanup_item(
            &mut items,
            active_root,
            format!("{service_id}-venv"),
            format!("{service_id} virtual environment"),
            service_dir.join(".venv"),
        );
        add_cleanup_item(
            &mut items,
            active_root,
            format!("{service_id}-cache"),
            format!("{service_id} service cache"),
            service_dir.join(".cache"),
        );
    }

    add_cleanup_item(
        &mut items,
        active_root,
        "legacy-cache".to_string(),
        "legacy runtime cache".to_string(),
        legacy_root.join("cache"),
    );
    add_cleanup_item(
        &mut items,
        active_root,
        "legacy-python".to_string(),
        "legacy managed Python installs".to_string(),
        legacy_root.join("python"),
    );
    add_cleanup_item(
        &mut items,
        active_root,
        "legacy-models".to_string(),
        "legacy runtime models".to_string(),
        legacy_root.join("models"),
    );
    add_cleanup_item(
        &mut items,
        active_root,
        "carey-checkpoints".to_string(),
        "legacy Carey model checkpoints".to_string(),
        legacy_root
            .join("services")
            .join("carey")
            .join("checkpoints"),
    );

    for repo in known_legacy_hf_repos() {
        add_cleanup_item(
            &mut items,
            active_root,
            format!("hf-{}", repo.replace('/', "-").replace('.', "-")),
            format!("Hugging Face cache: {repo}"),
            legacy_hf_cache_folder(default_hf_root, repo),
        );
    }

    items
}

fn path_prefix_key(path: &Path) -> String {
    let normalized = if cfg!(windows) {
        path.to_string_lossy()
            .replace('/', "\\")
            .to_ascii_lowercase()
    } else {
        path.to_string_lossy().replace('\\', "/")
    };
    normalized
        .trim_end_matches(|ch| ch == '\\' || ch == '/')
        .to_string()
}

fn path_is_inside(path: &Path, root: &Path) -> bool {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let path_key = path_prefix_key(&path);
    let root_key = path_prefix_key(&root);
    let separator = if cfg!(windows) { "\\" } else { "/" };

    path_key == root_key || path_key.starts_with(&format!("{root_key}{separator}"))
}

fn safe_lora_storage_name(name: &str) -> String {
    if let Some(sanitized) = sanitize_lora_name(name) {
        return sanitized;
    }

    let mut cleaned = String::new();
    for ch in name.trim().to_lowercase().chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-' {
            cleaned.push(ch);
        } else if !cleaned.ends_with('-') {
            cleaned.push('-');
        }
    }

    let cleaned = cleaned.trim_matches('-').to_string();
    if cleaned.is_empty() {
        "lora".to_string()
    } else {
        cleaned
    }
}

fn unique_destination_path(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }

    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let stem = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let extension = path
        .extension()
        .map(|ext| ext.to_string_lossy().to_string())
        .filter(|ext| !ext.is_empty());

    for index in 2.. {
        let file_name = match &extension {
            Some(extension) => format!("{stem}-{index}.{extension}"),
            None => format!("{stem}-{index}"),
        };
        let candidate = parent.join(file_name);
        if !candidate.exists() {
            return candidate;
        }
    }

    path.to_path_buf()
}

fn copy_file_checked(source: &Path, target: &Path) -> Result<(), String> {
    let parent = target
        .parent()
        .ok_or_else(|| format!("{} has no parent folder", target.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;

    let source_len = std::fs::metadata(source)
        .map_err(|e| format!("Cannot inspect {}: {}", source.display(), e))?
        .len();
    let copied = std::fs::copy(source, target).map_err(|e| {
        format!(
            "Cannot copy {} to {}: {}",
            source.display(),
            target.display(),
            e
        )
    })?;
    if copied != source_len {
        let _ = std::fs::remove_file(target);
        return Err(format!(
            "Copy from {} to {} was incomplete: copied {copied} of {source_len} bytes",
            source.display(),
            target.display()
        ));
    }

    Ok(())
}

fn copy_legacy_file_if_missing(source: &Path, target: &Path) -> Result<(), String> {
    if !source.is_file() || target.exists() {
        return Ok(());
    }
    copy_file_checked(source, target)
}

fn migrate_optional_source_dir(
    raw_path: &Option<String>,
    legacy_root: &Path,
    target_base: &Path,
) -> Result<Option<String>, String> {
    let Some(raw_path) = raw_path else {
        return Ok(None);
    };

    let source = PathBuf::from(raw_path);
    if !source.is_dir() || !path_is_inside(&source, legacy_root) {
        return Ok(Some(raw_path.clone()));
    }

    let target = unique_destination_path(target_base);
    copy_dir_recursive(&source, &target)?;
    Ok(Some(display_path(&target)))
}

fn merge_json_object_file_preserve_target(source: &Path, target: &Path) -> Result<(), String> {
    if !source.is_file() {
        return Ok(());
    }
    if !target.is_file() {
        return copy_file_checked(source, target);
    }

    let source_raw = std::fs::read_to_string(source)
        .map_err(|e| format!("Cannot read {}: {}", source.display(), e))?;
    let target_raw = std::fs::read_to_string(target)
        .map_err(|e| format!("Cannot read {}: {}", target.display(), e))?;
    let source_json = serde_json::from_str::<serde_json::Value>(&source_raw)
        .map_err(|e| format!("Invalid JSON in {}: {}", source.display(), e))?;
    let mut target_json = serde_json::from_str::<serde_json::Value>(&target_raw)
        .map_err(|e| format!("Invalid JSON in {}: {}", target.display(), e))?;

    let (Some(source_object), Some(target_object)) =
        (source_json.as_object(), target_json.as_object_mut())
    else {
        return Ok(());
    };

    let mut changed = false;
    for (key, value) in source_object {
        if !target_object.contains_key(key) {
            target_object.insert(key.clone(), value.clone());
            changed = true;
        }
    }
    if !changed {
        return Ok(());
    }

    let json = serde_json::to_string_pretty(&target_json)
        .map_err(|e| format!("Cannot serialize {}: {}", target.display(), e))?;
    std::fs::write(target, json).map_err(|e| format!("Cannot save {}: {}", target.display(), e))
}

fn carey_lora_catalog_path_for(root: &Path) -> PathBuf {
    root.join("carey").join("lora_catalog.json")
}

fn sa3_lora_catalog_path_for(root: &Path) -> PathBuf {
    root.join("sa3").join("lora_catalog.json")
}

fn legacy_sa3_lora_target(active_root: &Path, name: &str, source: &Path) -> PathBuf {
    let safe_name = safe_lora_storage_name(name);
    let extension = source
        .extension()
        .map(|ext| ext.to_string_lossy().to_string())
        .filter(|ext| !ext.is_empty())
        .unwrap_or_else(|| "safetensors".to_string());
    unique_destination_path(
        &active_root
            .join("sa3")
            .join("loras")
            .join(format!("{safe_name}.{extension}")),
    )
}

fn legacy_carey_lora_target(active_root: &Path, name: &str) -> PathBuf {
    unique_destination_path(
        &active_root
            .join("carey")
            .join("loras")
            .join(safe_lora_storage_name(name)),
    )
}

fn active_sa3_catalog_entry_is_valid(entry: &Sa3LoraCatalogEntry) -> bool {
    looks_like_sa3_lora_checkpoint(&PathBuf::from(&entry.path))
}

fn active_carey_catalog_entry_is_valid(entry: &CareyLoraCatalogEntry) -> bool {
    looks_like_lora_checkpoint_dir(&PathBuf::from(&entry.path))
}

fn collect_legacy_lora_candidates(
    active_root: &Path,
    legacy_root: &Path,
) -> Result<Vec<LegacyLoraMigrationCandidate>, String> {
    let mut candidates = Vec::new();
    if storage::paths_equivalent(active_root, legacy_root) {
        return Ok(candidates);
    }

    let active_sa3 = read_sa3_lora_catalog_from(&sa3_lora_catalog_path_for(active_root))?;
    let legacy_sa3 = read_sa3_lora_catalog_from(&sa3_lora_catalog_path_for(legacy_root))?;
    for (name, entry) in legacy_sa3 {
        if active_sa3
            .get(&name)
            .map(active_sa3_catalog_entry_is_valid)
            .unwrap_or(false)
        {
            continue;
        }
        let source = PathBuf::from(&entry.path);
        if !path_is_inside(&source, legacy_root) || !looks_like_sa3_lora_checkpoint(&source) {
            continue;
        }
        let target = legacy_sa3_lora_target(active_root, &name, &source);
        candidates.push(LegacyLoraMigrationCandidate {
            service: "sa3".to_string(),
            name,
            source_path: display_path(&source),
            target_path: display_path(&target),
            bytes: path_size(&source),
        });
    }

    let active_carey = read_carey_lora_catalog_from(&carey_lora_catalog_path_for(active_root))?;
    let legacy_carey = read_carey_lora_catalog_from(&carey_lora_catalog_path_for(legacy_root))?;
    for (name, entry) in legacy_carey {
        if active_carey
            .get(&name)
            .map(active_carey_catalog_entry_is_valid)
            .unwrap_or(false)
        {
            continue;
        }
        let source = PathBuf::from(&entry.path);
        if !path_is_inside(&source, legacy_root) || !looks_like_lora_checkpoint_dir(&source) {
            continue;
        }
        let target = legacy_carey_lora_target(active_root, &name);
        candidates.push(LegacyLoraMigrationCandidate {
            service: "carey".to_string(),
            name,
            source_path: display_path(&source),
            target_path: display_path(&target),
            bytes: path_size(&source),
        });
    }

    Ok(candidates)
}

fn build_legacy_storage_maintenance_info(
    active_root: &Path,
) -> Result<LegacyStorageMaintenanceInfo, String> {
    let legacy_root = storage::legacy_runtime_root();
    let default_hf_root = default_hf_cache_root();
    let cleanup_items = build_legacy_cleanup_items(active_root, &legacy_root, &default_hf_root);
    let lora_candidates = collect_legacy_lora_candidates(active_root, &legacy_root)?;
    let total_cleanup_bytes = cleanup_items.iter().map(|item| item.bytes).sum();
    let total_lora_bytes = lora_candidates.iter().map(|item| item.bytes).sum();
    let legacy_is_active = storage::paths_equivalent(active_root, &legacy_root);

    Ok(LegacyStorageMaintenanceInfo {
        active_root: display_path(active_root),
        legacy_root: display_path(&legacy_root),
        default_hf_cache_root: display_path(&default_hf_root),
        can_cleanup: !legacy_is_active && !cleanup_items.is_empty(),
        can_migrate_loras: !legacy_is_active && !lora_candidates.is_empty(),
        cleanup_items,
        lora_candidates,
        total_cleanup_bytes,
        total_lora_bytes,
    })
}

fn migrate_sa3_catalog_entry(
    name: &str,
    entry: &Sa3LoraCatalogEntry,
    active_root: &Path,
    legacy_root: &Path,
) -> Result<Sa3LoraCatalogEntry, String> {
    let source = PathBuf::from(&entry.path);
    if !path_is_inside(&source, legacy_root) || !looks_like_sa3_lora_checkpoint(&source) {
        return Err(format!(
            "{} is not a legacy SA3 LoRA file",
            source.display()
        ));
    }

    let target = legacy_sa3_lora_target(active_root, name, &source);
    copy_file_checked(&source, &target)?;

    let safe_name = safe_lora_storage_name(name);
    let mut migrated = entry.clone();
    migrated.path = display_path(&target);
    migrated.prompts_path = migrate_optional_source_dir(
        &entry.prompts_path,
        legacy_root,
        &active_root
            .join("sa3")
            .join("lora-sources")
            .join(&safe_name)
            .join("prompts"),
    )?;

    let mut migrated_checkpoints = Vec::new();
    for checkpoint in &entry.training_checkpoints {
        let source = PathBuf::from(&checkpoint.path);
        if source.is_file() && path_is_inside(&source, legacy_root) {
            let file_name = source
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let target = unique_destination_path(
                &active_root
                    .join("sa3")
                    .join("training")
                    .join("migrated")
                    .join(&safe_name)
                    .join(file_name),
            );
            copy_file_checked(&source, &target)?;
            let mut migrated_checkpoint = checkpoint.clone();
            migrated_checkpoint.path = display_path(&target);
            migrated_checkpoints.push(migrated_checkpoint);
        } else if !path_is_inside(&source, legacy_root) {
            migrated_checkpoints.push(checkpoint.clone());
        }
    }
    migrated.training_checkpoints = migrated_checkpoints;
    if let Some(step) = migrated.selected_training_step {
        if !migrated
            .training_checkpoints
            .iter()
            .any(|checkpoint| checkpoint.step == step)
        {
            migrated.selected_training_step = None;
        }
    }

    copy_legacy_file_if_missing(
        &legacy_root
            .join("sa3")
            .join("prompts")
            .join(format!("{name}.json")),
        &active_root
            .join("sa3")
            .join("prompts")
            .join(format!("{name}.json")),
    )?;

    Ok(migrated)
}

fn migrate_carey_catalog_entry(
    name: &str,
    entry: &CareyLoraCatalogEntry,
    active_root: &Path,
    legacy_root: &Path,
) -> Result<CareyLoraCatalogEntry, String> {
    let source = PathBuf::from(&entry.path);
    if !path_is_inside(&source, legacy_root) || !looks_like_lora_checkpoint_dir(&source) {
        return Err(format!(
            "{} is not a legacy Carey LoRA folder",
            source.display()
        ));
    }

    let safe_name = safe_lora_storage_name(name);
    let target = legacy_carey_lora_target(active_root, name);
    copy_dir_recursive(&source, &target)?;

    let mut migrated = entry.clone();
    migrated.path = display_path(&target);
    migrated.captions_path = migrate_optional_source_dir(
        &entry.captions_path,
        legacy_root,
        &active_root
            .join("carey")
            .join("lora-sources")
            .join(&safe_name)
            .join("captions"),
    )?;

    Ok(migrated)
}

fn migrate_legacy_sa3_loras(
    active_root: &Path,
    legacy_root: &Path,
    errors: &mut Vec<String>,
) -> usize {
    let legacy_catalog = match read_sa3_lora_catalog_from(&sa3_lora_catalog_path_for(legacy_root)) {
        Ok(catalog) => catalog,
        Err(error) => {
            errors.push(error);
            return 0;
        }
    };
    let mut active_catalog =
        match read_sa3_lora_catalog_from(&sa3_lora_catalog_path_for(active_root)) {
            Ok(catalog) => catalog,
            Err(error) => {
                errors.push(error);
                return 0;
            }
        };

    let mut prepared = 0;
    for (name, entry) in legacy_catalog {
        if active_catalog
            .get(&name)
            .map(active_sa3_catalog_entry_is_valid)
            .unwrap_or(false)
        {
            continue;
        }
        match migrate_sa3_catalog_entry(&name, &entry, active_root, legacy_root) {
            Ok(migrated) => {
                active_catalog.insert(name, migrated);
                prepared += 1;
            }
            Err(error) => errors.push(format!("SA3 LoRA '{}': {}", name, error)),
        }
    }

    if prepared == 0 {
        return 0;
    }

    if let Err(error) =
        save_sa3_lora_catalog_to(&sa3_lora_catalog_path_for(active_root), &active_catalog)
    {
        errors.push(error);
        return 0;
    }
    if let Err(error) = build_sa3_lora_state(active_root) {
        errors.push(error);
    }

    prepared
}

fn migrate_legacy_carey_loras(
    active_root: &Path,
    legacy_root: &Path,
    errors: &mut Vec<String>,
) -> usize {
    let legacy_catalog =
        match read_carey_lora_catalog_from(&carey_lora_catalog_path_for(legacy_root)) {
            Ok(catalog) => catalog,
            Err(error) => {
                errors.push(error);
                return 0;
            }
        };
    let mut active_catalog =
        match read_carey_lora_catalog_from(&carey_lora_catalog_path_for(active_root)) {
            Ok(catalog) => catalog,
            Err(error) => {
                errors.push(error);
                return 0;
            }
        };

    let mut prepared = 0;
    for (name, entry) in legacy_catalog {
        if active_catalog
            .get(&name)
            .map(active_carey_catalog_entry_is_valid)
            .unwrap_or(false)
        {
            continue;
        }
        match migrate_carey_catalog_entry(&name, &entry, active_root, legacy_root) {
            Ok(migrated) => {
                active_catalog.insert(name, migrated);
                prepared += 1;
            }
            Err(error) => errors.push(format!("Carey LoRA '{}': {}", name, error)),
        }
    }

    if prepared == 0 {
        return 0;
    }

    if let Err(error) = merge_json_object_file_preserve_target(
        &legacy_root.join("carey").join("captions.json"),
        &active_root.join("carey").join("captions.json"),
    ) {
        errors.push(error);
    }
    if let Err(error) =
        save_carey_lora_catalog_to(&carey_lora_catalog_path_for(active_root), &active_catalog)
    {
        errors.push(error);
        return 0;
    }
    if let Err(error) = build_carey_lora_state(active_root) {
        errors.push(error);
    }

    prepared
}

fn migrate_legacy_loras_impl(active_root: &Path) -> LegacyStorageMaintenanceResult {
    let legacy_root = storage::legacy_runtime_root();
    let mut errors = Vec::new();
    let mut migrated_loras = 0;

    if storage::paths_equivalent(active_root, &legacy_root) {
        errors.push("Active storage is still the legacy AppData folder.".to_string());
    } else {
        migrated_loras += migrate_legacy_sa3_loras(active_root, &legacy_root, &mut errors);
        migrated_loras += migrate_legacy_carey_loras(active_root, &legacy_root, &mut errors);
    }

    let info = build_legacy_storage_maintenance_info(active_root).unwrap_or_else(|error| {
        errors.push(error);
        LegacyStorageMaintenanceInfo {
            active_root: display_path(active_root),
            legacy_root: display_path(&legacy_root),
            default_hf_cache_root: display_path(&default_hf_cache_root()),
            cleanup_items: Vec::new(),
            lora_candidates: Vec::new(),
            total_cleanup_bytes: 0,
            total_lora_bytes: 0,
            can_cleanup: false,
            can_migrate_loras: false,
        }
    });

    LegacyStorageMaintenanceResult {
        info,
        migrated_loras,
        cleaned_items: 0,
        errors,
    }
}

fn cleanup_legacy_storage_impl(active_root: &Path) -> LegacyStorageMaintenanceResult {
    let legacy_root = storage::legacy_runtime_root();
    let mut errors = Vec::new();
    let mut cleaned_items = 0;

    let before = match build_legacy_storage_maintenance_info(active_root) {
        Ok(info) => info,
        Err(error) => {
            errors.push(error);
            LegacyStorageMaintenanceInfo {
                active_root: display_path(active_root),
                legacy_root: display_path(&legacy_root),
                default_hf_cache_root: display_path(&default_hf_cache_root()),
                cleanup_items: Vec::new(),
                lora_candidates: Vec::new(),
                total_cleanup_bytes: 0,
                total_lora_bytes: 0,
                can_cleanup: false,
                can_migrate_loras: false,
            }
        }
    };

    if storage::paths_equivalent(active_root, &legacy_root) {
        errors.push("Active storage is still the legacy AppData folder.".to_string());
    } else {
        for item in before.cleanup_items {
            let path = PathBuf::from(&item.path);
            if path.exists() {
                match remove_path(&path) {
                    Ok(()) => cleaned_items += 1,
                    Err(error) => errors.push(error),
                }
            }
        }
    }

    let info = build_legacy_storage_maintenance_info(active_root).unwrap_or_else(|error| {
        errors.push(error);
        LegacyStorageMaintenanceInfo {
            active_root: display_path(active_root),
            legacy_root: display_path(&legacy_root),
            default_hf_cache_root: display_path(&default_hf_cache_root()),
            cleanup_items: Vec::new(),
            lora_candidates: Vec::new(),
            total_cleanup_bytes: 0,
            total_lora_bytes: 0,
            can_cleanup: false,
            can_migrate_loras: false,
        }
    });

    LegacyStorageMaintenanceResult {
        info,
        migrated_loras: 0,
        cleaned_items,
        errors,
    }
}

fn read_text_tail(path: &Path, max_bytes: usize) -> String {
    let Ok(bytes) = std::fs::read(path) else {
        return String::new();
    };
    let start = bytes.len().saturating_sub(max_bytes);
    String::from_utf8_lossy(&bytes[start..]).to_string()
}

fn parse_carey_epoch_progress(log_tail: &str) -> Option<(u32, Option<u32>)> {
    for line in log_tail.lines().rev() {
        let Some((_, raw_progress)) = line.rsplit_once("Epoch ") else {
            continue;
        };
        let progress = raw_progress.trim_start();
        let current_len = progress
            .bytes()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if current_len == 0 {
            continue;
        }
        let current = progress[..current_len].parse::<u32>().ok()?;
        let remainder = progress[current_len..].trim_start();
        let maximum = remainder.strip_prefix('/').and_then(|value| {
            let value = value.trim_start();
            let max_len = value
                .bytes()
                .take_while(|byte| byte.is_ascii_digit())
                .count();
            (max_len > 0)
                .then(|| value[..max_len].parse::<u32>().ok())
                .flatten()
        });
        return Some((current, maximum));
    }
    None
}

fn now_epoch_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn sa3_current_training_status_path() -> Option<PathBuf> {
    std::fs::read_to_string(sa3_training_current_job_path())
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| {
            value
                .get("statusPath")
                .and_then(|value| value.as_str())
                .map(PathBuf::from)
        })
}

fn read_sa3_training_status_file(path: &Path) -> Sa3LoraTrainingState {
    let mut state = std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Sa3LoraTrainingState>(&raw).ok())
        .unwrap_or_else(|| Sa3LoraTrainingState {
            job_id: None,
            name: None,
            status: "idle".to_string(),
            phase: "idle".to_string(),
            message: "No SA3 LoRA training job has been started.".to_string(),
            error: None,
            pid: None,
            owner_pid: None,
            launcher_pid: None,
            child_pid: None,
            run_dir: None,
            log_path: None,
            cancel_path: None,
            final_checkpoint_path: None,
            current_step: None,
            max_steps: None,
            log_tail: String::new(),
        });

    if let Some(log_path) = state.log_path.as_ref().map(PathBuf::from) {
        state.log_tail = read_text_tail(&log_path, 16 * 1024);
    }
    if let Some(run_dir) = state.run_dir.as_ref().map(PathBuf::from) {
        if let Some(step) = read_sa3_training_step(&run_dir) {
            state.current_step = Some(step);
        }
    }
    state
}

fn read_sa3_training_step(run_dir: &Path) -> Option<u32> {
    let path = run_dir.join("demos").join("loss_by_timestep.bin");
    let length = std::fs::metadata(path).ok()?.len();
    const RECORD_BYTES: u64 = 12;
    if length < RECORD_BYTES {
        return None;
    }
    Some((length / RECORD_BYTES).min(u32::MAX as u64) as u32)
}

fn read_sa3_lora_training_state() -> Sa3LoraTrainingState {
    match sa3_current_training_status_path() {
        Some(path) => {
            reconcile_sa3_lora_training_parent_exit(&path, carey_ace_training_parent_is_running)
                .unwrap_or_else(|error| {
                    log::warn!("Could not reconcile SA3 training status: {error}");
                });
            read_sa3_training_status_file(&path)
        }
        None => read_sa3_training_status_file(Path::new("")),
    }
}

fn carey_ace_current_training_status_path() -> Option<PathBuf> {
    std::fs::read_to_string(carey_training_current_job_path())
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| {
            value
                .get("statusPath")
                .and_then(|value| value.as_str())
                .map(PathBuf::from)
        })
}

fn default_carey_ace_training_state() -> CareyAceTrainingState {
    CareyAceTrainingState {
        job_id: None,
        name: None,
        status: "idle".to_string(),
        phase: "idle".to_string(),
        message: "No ACE-Step LoRA training job has been started.".to_string(),
        error: None,
        pid: None,
        owner_pid: None,
        launcher_pid: None,
        child_pid: None,
        run_dir: None,
        log_path: None,
        cancel_path: None,
        final_checkpoint_path: None,
        current_step: None,
        max_steps: None,
        current_file: None,
        total_files: None,
        sample_count: None,
        captioned_count: None,
        caption_lm_model: None,
        model_family: None,
        adapter_type: None,
        captions_path: None,
        dataset_json_path: None,
        training_plan_path: None,
        result_path: None,
        registered_lora_name: None,
        log_tail: String::new(),
    }
}

fn read_carey_ace_training_status_file(path: &Path) -> CareyAceTrainingState {
    let mut state = std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<CareyAceTrainingState>(&raw).ok())
        .unwrap_or_else(default_carey_ace_training_state);

    if let Some(log_path) = state.log_path.as_ref().map(PathBuf::from) {
        state.log_tail = read_text_tail(&log_path, 16 * 1024);
        if let Some((epoch, maximum)) = parse_carey_epoch_progress(&state.log_tail) {
            state.current_step = Some(state.current_step.unwrap_or(0).max(epoch));
            if let Some(maximum) = maximum {
                state.max_steps = Some(maximum);
            }
        }
    }
    state
}

#[cfg(test)]
mod carey_epoch_progress_tests {
    use super::parse_carey_epoch_progress;

    #[test]
    fn parses_training_update_progress() {
        let log = "Epoch 136/150, Step 1360, Loss: 0.42\n[OK] Epoch 137/150 in 16.8s";

        assert_eq!(parse_carey_epoch_progress(log), Some((137, Some(150))));
    }

    #[test]
    fn parses_prompt_mix_epoch_using_status_maximum() {
        let log = "2026-06-18 | INFO | [Gary] Epoch 137 prompt mix: 2/10 tracks use genre";

        assert_eq!(parse_carey_epoch_progress(log), Some((137, None)));
    }

    #[test]
    fn ignores_unrelated_step_and_epoch_path_text() {
        let log =
            "Training checkpoint saved to output/checkpoints/epoch_150 (epoch 150, step 1500)";

        assert_eq!(parse_carey_epoch_progress(log), None);
    }
}

fn read_carey_ace_lora_training_state() -> CareyAceTrainingState {
    match carey_ace_current_training_status_path() {
        Some(path) => {
            reconcile_carey_ace_training_parent_exit(&path, carey_ace_training_parent_is_running)
                .unwrap_or_else(|error| {
                    log::warn!("Could not reconcile ACE-Step training status: {error}");
                });
            read_carey_ace_training_status_file(&path)
        }
        None => read_carey_ace_training_status_file(Path::new("")),
    }
}

fn write_carey_ace_training_launch_state(
    status_path: &Path,
    job_id: &str,
    name: &str,
    run_dir: &Path,
    log_path: &Path,
    cancel_path: &Path,
    max_steps: u32,
    message: &str,
) -> Result<(), String> {
    if let Some(parent) = status_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }

    let payload = serde_json::json!({
        "jobId": job_id,
        "name": name,
        "status": "starting",
        "phase": "starting",
        "message": message,
        "runDir": run_dir.to_string_lossy(),
        "logPath": log_path.to_string_lossy(),
        "cancelPath": cancel_path.to_string_lossy(),
        "currentStep": 0,
        "maxSteps": max_steps,
        "ownerPid": std::process::id(),
        "updatedAt": now_epoch_seconds(),
    });
    let json = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("Cannot serialize ACE training state: {}", e))?;
    std::fs::write(status_path, json)
        .map_err(|e| format!("Cannot save {}: {}", status_path.display(), e))?;

    let current_payload = serde_json::json!({
        "jobId": job_id,
        "statusPath": status_path.to_string_lossy(),
    });
    let current_json = serde_json::to_string_pretty(&current_payload)
        .map_err(|e| format!("Cannot serialize ACE current job: {}", e))?;
    let current_path = carey_training_current_job_path();
    if let Some(parent) = current_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    std::fs::write(&current_path, current_json)
        .map_err(|e| format!("Cannot save {}: {}", current_path.display(), e))?;
    Ok(())
}

fn patch_carey_ace_training_status(
    status_path: &Path,
    updates: &[(&str, serde_json::Value)],
) -> Result<(), String> {
    let mut payload = std::fs::read_to_string(status_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    for (key, value) in updates {
        payload.insert((*key).to_string(), value.clone());
    }
    payload.insert(
        "updatedAt".to_string(),
        serde_json::json!(now_epoch_seconds()),
    );
    let json = serde_json::to_string_pretty(&serde_json::Value::Object(payload))
        .map_err(|e| format!("Cannot serialize ACE training state: {}", e))?;
    std::fs::write(status_path, json)
        .map_err(|e| format!("Cannot save {}: {}", status_path.display(), e))
}

fn write_carey_ace_training_process_id(status_path: &Path, pid: u32) -> Result<(), String> {
    patch_carey_ace_training_status(
        status_path,
        &[
            ("pid", serde_json::json!(pid)),
            ("launcherPid", serde_json::json!(pid)),
        ],
    )
}

fn mark_carey_ace_training_failed(
    status_path: &Path,
    message: &str,
    expected_pid: Option<u32>,
) -> Result<(), String> {
    let state = read_carey_ace_training_status_file(status_path);
    if let Some(expected_pid) = expected_pid {
        let matches_launcher = state.launcher_pid == Some(expected_pid);
        let matches_legacy_pid = state.launcher_pid.is_none() && state.pid == Some(expected_pid);
        if !matches_launcher && !matches_legacy_pid {
            return Ok(());
        }
    }
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(());
    }
    patch_carey_ace_training_status(
        status_path,
        &[
            ("status", serde_json::json!("failed")),
            ("phase", serde_json::json!("failed")),
            ("message", serde_json::json!(message)),
            ("error", serde_json::json!(message)),
            ("pid", serde_json::Value::Null),
            ("ownerPid", serde_json::Value::Null),
            ("launcherPid", serde_json::Value::Null),
            ("childPid", serde_json::Value::Null),
        ],
    )
}

fn reconcile_carey_ace_training_parent_exit<F>(
    status_path: &Path,
    is_process_running: F,
) -> Result<(), String>
where
    F: Fn(u32) -> Option<bool>,
{
    let state = read_carey_ace_training_status_file(status_path);
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(());
    }
    let mut monitored_pids = Vec::new();
    if let Some(pid) = state.owner_pid {
        monitored_pids.push(pid);
    }
    if let Some(pid) = state.launcher_pid {
        if !monitored_pids.contains(&pid) {
            monitored_pids.push(pid);
        }
    }
    if let Some(pid) = state.pid {
        if !monitored_pids.contains(&pid) {
            monitored_pids.push(pid);
        }
    }
    let Some(missing_pid) = monitored_pids
        .into_iter()
        .find(|pid| is_process_running(*pid) == Some(false))
    else {
        return Ok(());
    };

    let warnings = terminate_discovered_carey_ace_training_processes(&state);
    let cleanup_suffix = if warnings.is_empty() {
        String::new()
    } else {
        format!(" Process cleanup warning: {}", warnings.join("; "))
    };
    mark_carey_ace_training_failed(
        status_path,
        &format!(
            "ACE-Step training process {missing_pid} is no longer running. The job may have been stopped outside the app; check the training log for details.{cleanup_suffix}"
        ),
        state.launcher_pid.or(state.pid),
    )
}

#[cfg(target_os = "windows")]
fn carey_ace_training_parent_is_running(pid: u32) -> Option<bool> {
    let mut command = std::process::Command::new("powershell");
    command.args([
        "-NoProfile",
        "-Command",
        &format!(
            "$ErrorActionPreference='SilentlyContinue'; if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ '1' }} else {{ '0' }}"
        ),
    ]);
    hide_std_console_window(&mut command);
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    match String::from_utf8_lossy(&output.stdout).trim() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

#[cfg(not(target_os = "windows"))]
fn carey_ace_training_parent_is_running(_pid: u32) -> Option<bool> {
    None
}

fn reconcile_owned_workload_status<F>(
    status_path: &Path,
    label: &str,
    is_process_running: F,
) -> Result<(), String>
where
    F: Fn(u32) -> Option<bool>,
{
    let mut payload = match std::fs::read_to_string(status_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| value.as_object().cloned())
    {
        Some(payload) => payload,
        None => return Ok(()),
    };
    let active = payload
        .get("status")
        .and_then(|value| value.as_str())
        .map(|status| matches!(status, "starting" | "running"))
        .unwrap_or(false);
    if !active {
        return Ok(());
    }
    let Some(owner_pid) = payload
        .get("ownerPid")
        .and_then(|value| value.as_u64())
        .and_then(|pid| u32::try_from(pid).ok())
    else {
        return Ok(());
    };
    if is_process_running(owner_pid) != Some(false) {
        return Ok(());
    }

    let message = format!(
        "{label} was interrupted because its gary4local owner process {owner_pid} exited unexpectedly."
    );
    payload.insert("status".to_string(), serde_json::json!("failed"));
    payload.insert("phase".to_string(), serde_json::json!("failed"));
    payload.insert("message".to_string(), serde_json::json!(&message));
    payload.insert("error".to_string(), serde_json::json!(&message));
    payload.insert("pid".to_string(), serde_json::Value::Null);
    payload.insert("ownerPid".to_string(), serde_json::Value::Null);
    payload.insert("launcherPid".to_string(), serde_json::Value::Null);
    payload.insert("childPid".to_string(), serde_json::Value::Null);
    payload.insert(
        "updatedAt".to_string(),
        serde_json::json!(now_epoch_seconds()),
    );
    let json = serde_json::to_string_pretty(&serde_json::Value::Object(payload))
        .map_err(|error| format!("Cannot serialize recovered workload state: {error}"))?;
    std::fs::write(status_path, json)
        .map_err(|error| format!("Cannot save {}: {error}", status_path.display()))
}

fn carey_training_required_checkpoint_files(
    checkpoint_dir: &Path,
    model_folder: &str,
    caption_lm_model: Option<&str>,
) -> Vec<PathBuf> {
    let model_dir = checkpoint_dir.join(model_folder);
    let mut required = vec![
        model_dir.join("config.json"),
        model_dir.join("model.safetensors"),
        model_dir.join("silence_latent.pt"),
        checkpoint_dir.join("vae").join("config.json"),
        checkpoint_dir
            .join("vae")
            .join("diffusion_pytorch_model.safetensors"),
        checkpoint_dir
            .join("Qwen3-Embedding-0.6B")
            .join("config.json"),
        checkpoint_dir
            .join("Qwen3-Embedding-0.6B")
            .join("model.safetensors"),
        checkpoint_dir
            .join("Qwen3-Embedding-0.6B")
            .join("tokenizer.json"),
    ];
    if let Some(lm_model) = caption_lm_model {
        let lm_dir = checkpoint_dir.join(lm_model);
        required.push(lm_dir.join("config.json"));
        required.push(lm_dir.join("tokenizer.json"));
        if lm_model == "acestep-5Hz-lm-4B" {
            required.push(lm_dir.join("model.safetensors.index.json"));
            required.push(lm_dir.join("model-00001-of-00002.safetensors"));
            required.push(lm_dir.join("model-00002-of-00002.safetensors"));
        } else {
            required.push(lm_dir.join("model.safetensors"));
        }
    }
    required
}

#[cfg(test)]
mod carey_training_checkpoint_tests {
    use super::{
        carey_training_required_checkpoint_files, mark_carey_ace_training_failed,
        mark_sa3_lora_training_failed, reconcile_carey_ace_training_parent_exit,
        reconcile_owned_workload_status, reconcile_sa3_lora_training_parent_exit,
        write_carey_ace_training_cancelled_state,
    };
    use std::path::Path;

    #[test]
    fn complete_preflight_covers_shared_models_and_sharded_captioner() {
        let root = Path::new("checkpoints");
        let required = carey_training_required_checkpoint_files(
            root,
            "acestep-v15-xl-base",
            Some("acestep-5Hz-lm-4B"),
        );

        assert!(required.contains(&root.join("acestep-v15-xl-base").join("model.safetensors")));
        assert!(required.contains(&root.join("vae").join("diffusion_pytorch_model.safetensors")));
        assert!(required.contains(&root.join("Qwen3-Embedding-0.6B").join("model.safetensors")));
        assert!(required.contains(
            &root
                .join("acestep-5Hz-lm-4B")
                .join("model.safetensors.index.json")
        ));
        assert!(required.contains(
            &root
                .join("acestep-5Hz-lm-4B")
                .join("model-00002-of-00002.safetensors")
        ));
    }

    #[test]
    fn process_watcher_only_fails_the_matching_active_job() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-carey-watcher-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"training","pid":42}"#,
        )
        .expect("write active status");

        mark_carey_ace_training_failed(&status_path, "wrong process", Some(41))
            .expect("ignore stale watcher");
        let unchanged = std::fs::read_to_string(&status_path).expect("read unchanged status");
        assert!(unchanged.contains(r#""status":"running""#));

        mark_carey_ace_training_failed(&status_path, "trainer exited", Some(42))
            .expect("fail matching process");
        let failed = std::fs::read_to_string(&status_path).expect("read failed status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains("trainer exited"));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn process_watcher_keeps_the_launcher_identity_after_python_reexec() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-carey-launcher-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"preprocessing","pid":43,"launcherPid":42,"childPid":44}"#,
        )
        .expect("write active status");

        mark_carey_ace_training_failed(&status_path, "launcher exited", Some(42))
            .expect("fail matching launcher");
        let failed = std::fs::read_to_string(&status_path).expect("read failed status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains(r#""launcherPid": null"#));
        assert!(failed.contains(r#""pid": null"#));
        assert!(failed.contains(r#""childPid": null"#));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn cancellation_clears_both_trainer_process_ids() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-carey-cancel-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        let cancel_path = status_path.with_extension("cancel");
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"training","pid":42,"childPid":43}"#,
        )
        .expect("write active status");

        write_carey_ace_training_cancelled_state(
            &status_path,
            Some(&cancel_path),
            "Training cancelled.",
        )
        .expect("record cancellation");
        let cancelled = std::fs::read_to_string(&status_path).expect("read cancelled status");
        assert!(cancelled.contains(r#""status": "cancelled""#));
        assert!(cancelled.contains(r#""pid": null"#));
        assert!(cancelled.contains(r#""launcherPid": null"#));
        assert!(cancelled.contains(r#""childPid": null"#));
        assert!(cancelled.contains("cancelPath"));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn missing_active_parent_is_recovered_as_a_failed_job() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-carey-recovery-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"training","pid":42,"childPid":43}"#,
        )
        .expect("write active status");

        reconcile_carey_ace_training_parent_exit(&status_path, |_| Some(false))
            .expect("recover stopped parent");
        let failed = std::fs::read_to_string(&status_path).expect("read recovered status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains("no longer running"));
        assert!(failed.contains(r#""pid": null"#));
        assert!(failed.contains(r#""childPid": null"#));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn missing_reexecuted_parent_is_recovered_even_when_launcher_remains() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-carey-reexec-recovery-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"preprocessing","launcherPid":42,"pid":43,"childPid":44}"#,
        )
        .expect("write active status");

        reconcile_carey_ace_training_parent_exit(&status_path, |pid| Some(pid != 43))
            .expect("recover stopped re-executed parent");
        let failed = std::fs::read_to_string(&status_path).expect("read recovered status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains("43 is no longer running"));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn missing_app_owner_recovers_an_orphaned_training_job() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-carey-owner-recovery-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"preprocessing","ownerPid":41,"launcherPid":42,"pid":43,"childPid":44}"#,
        )
        .expect("write active status");

        reconcile_carey_ace_training_parent_exit(&status_path, |pid| Some(pid != 41))
            .expect("recover orphaned job");
        let failed = std::fs::read_to_string(&status_path).expect("read recovered status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains("41 is no longer running"));
        assert!(failed.contains(r#""ownerPid": null"#));
        assert!(failed.contains(r#""launcherPid": null"#));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn shared_recovery_marks_an_orphaned_workload_failed() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-shared-owner-recovery-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"preprocessing","ownerPid":41,"pid":42,"childPid":43}"#,
        )
        .expect("write active status");

        reconcile_owned_workload_status(&status_path, "Test workload", |pid| Some(pid != 41))
            .expect("recover orphaned workload");
        let failed = std::fs::read_to_string(&status_path).expect("read recovered status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains("Test workload was interrupted"));
        assert!(failed.contains(r#""ownerPid": null"#));
        assert!(failed.contains(r#""pid": null"#));
        assert!(failed.contains(r#""childPid": null"#));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn sa3_watcher_matches_the_launcher_after_python_reexec() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-sa3-launcher-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"pre-encoding","pid":43,"launcherPid":42,"childPid":44}"#,
        )
        .expect("write active status");

        mark_sa3_lora_training_failed(&status_path, "launcher exited", Some(42))
            .expect("fail matching launcher");
        let failed = std::fs::read_to_string(&status_path).expect("read failed status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains(r#""launcherPid": null"#));
        assert!(failed.contains(r#""pid": null"#));
        assert!(failed.contains(r#""childPid": null"#));
        let _ = std::fs::remove_file(status_path);
    }

    #[test]
    fn sa3_recovery_detects_a_missing_reexecuted_parent() {
        let status_path = std::env::temp_dir().join(format!(
            "gary-sa3-reexec-recovery-{}-{}.json",
            std::process::id(),
            super::now_epoch_seconds()
        ));
        std::fs::write(
            &status_path,
            r#"{"status":"running","phase":"pre-encoding","launcherPid":42,"pid":43,"childPid":44}"#,
        )
        .expect("write active status");

        reconcile_sa3_lora_training_parent_exit(&status_path, |pid| Some(pid != 43))
            .expect("recover stopped re-executed parent");
        let failed = std::fs::read_to_string(&status_path).expect("read recovered status");
        assert!(failed.contains(r#""status": "failed""#));
        assert!(failed.contains("43 is no longer running"));
        let _ = std::fs::remove_file(status_path);
    }
}

fn write_carey_ace_training_cancelled_state(
    status_path: &Path,
    cancel_path: Option<&Path>,
    message: &str,
) -> Result<(), String> {
    let mut updates = vec![
        ("status", serde_json::json!("cancelled")),
        ("phase", serde_json::json!("cancelled")),
        ("message", serde_json::json!(message)),
        ("error", serde_json::Value::Null),
        ("pid", serde_json::Value::Null),
        ("ownerPid", serde_json::Value::Null),
        ("launcherPid", serde_json::Value::Null),
        ("childPid", serde_json::Value::Null),
    ];
    if let Some(cancel_path) = cancel_path {
        updates.push((
            "cancelPath",
            serde_json::json!(cancel_path.to_string_lossy()),
        ));
    }
    patch_carey_ace_training_status(status_path, &updates)
}

fn write_sa3_training_launch_state(
    status_path: &Path,
    job_id: &str,
    name: &str,
    run_dir: &Path,
    log_path: &Path,
    cancel_path: &Path,
    max_steps: u32,
) -> Result<(), String> {
    if let Some(parent) = status_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }

    let payload = serde_json::json!({
        "jobId": job_id,
        "name": name,
        "status": "starting",
        "phase": "starting",
        "message": "Launching SA3 LoRA training.",
        "runDir": run_dir.to_string_lossy(),
        "logPath": log_path.to_string_lossy(),
        "cancelPath": cancel_path.to_string_lossy(),
        "currentStep": 0,
        "maxSteps": max_steps,
        "ownerPid": std::process::id(),
        "updatedAt": now_epoch_seconds(),
    });
    let json = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("Cannot serialize SA3 training state: {}", e))?;
    std::fs::write(status_path, json)
        .map_err(|e| format!("Cannot save {}: {}", status_path.display(), e))?;

    let current_payload = serde_json::json!({
        "jobId": job_id,
        "statusPath": status_path.to_string_lossy(),
    });
    let current_json = serde_json::to_string_pretty(&current_payload)
        .map_err(|e| format!("Cannot serialize SA3 current job: {}", e))?;
    let current_path = sa3_training_current_job_path();
    if let Some(parent) = current_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    std::fs::write(&current_path, current_json)
        .map_err(|e| format!("Cannot save {}: {}", current_path.display(), e))?;
    Ok(())
}

fn patch_sa3_training_status(
    status_path: &Path,
    updates: &[(&str, serde_json::Value)],
) -> Result<(), String> {
    let mut payload = std::fs::read_to_string(status_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    for (key, value) in updates {
        payload.insert((*key).to_string(), value.clone());
    }
    payload.insert(
        "updatedAt".to_string(),
        serde_json::json!(now_epoch_seconds()),
    );
    let json = serde_json::to_string_pretty(&serde_json::Value::Object(payload))
        .map_err(|e| format!("Cannot serialize SA3 training state: {}", e))?;
    std::fs::write(status_path, json)
        .map_err(|e| format!("Cannot save {}: {}", status_path.display(), e))
}

fn write_sa3_training_process_id(status_path: &Path, pid: u32) -> Result<(), String> {
    patch_sa3_training_status(
        status_path,
        &[
            ("pid", serde_json::json!(pid)),
            ("launcherPid", serde_json::json!(pid)),
        ],
    )
}

fn mark_sa3_lora_training_failed(
    status_path: &Path,
    message: &str,
    expected_pid: Option<u32>,
) -> Result<(), String> {
    let state = read_sa3_training_status_file(status_path);
    if let Some(expected_pid) = expected_pid {
        let matches_launcher = state.launcher_pid == Some(expected_pid);
        let matches_legacy_pid = state.launcher_pid.is_none() && state.pid == Some(expected_pid);
        if !matches_launcher && !matches_legacy_pid {
            return Ok(());
        }
    }
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(());
    }
    patch_sa3_training_status(
        status_path,
        &[
            ("status", serde_json::json!("failed")),
            ("phase", serde_json::json!("failed")),
            ("message", serde_json::json!(message)),
            ("error", serde_json::json!(message)),
            ("pid", serde_json::Value::Null),
            ("ownerPid", serde_json::Value::Null),
            ("launcherPid", serde_json::Value::Null),
            ("childPid", serde_json::Value::Null),
        ],
    )
}

fn reconcile_sa3_lora_training_parent_exit<F>(
    status_path: &Path,
    is_process_running: F,
) -> Result<(), String>
where
    F: Fn(u32) -> Option<bool>,
{
    let state = read_sa3_training_status_file(status_path);
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(());
    }
    let mut monitored_pids = Vec::new();
    for pid in [state.owner_pid, state.launcher_pid, state.pid]
        .into_iter()
        .flatten()
    {
        if !monitored_pids.contains(&pid) {
            monitored_pids.push(pid);
        }
    }
    let Some(missing_pid) = monitored_pids
        .into_iter()
        .find(|pid| is_process_running(*pid) == Some(false))
    else {
        return Ok(());
    };

    let warnings = terminate_discovered_sa3_training_processes(&state);
    let cleanup_suffix = if warnings.is_empty() {
        String::new()
    } else {
        format!(" Process cleanup warning: {}", warnings.join("; "))
    };
    mark_sa3_lora_training_failed(
        status_path,
        &format!(
            "SA3 LoRA training process {missing_pid} is no longer running. The job may have been stopped outside the app; check the training log for details.{cleanup_suffix}"
        ),
        state.launcher_pid.or(state.pid),
    )
}

fn write_sa3_training_cancelled_state(
    status_path: &Path,
    cancel_path: Option<&Path>,
    message: &str,
) -> Result<(), String> {
    let mut updates = vec![
        ("status", serde_json::json!("cancelled")),
        ("phase", serde_json::json!("cancelled")),
        ("message", serde_json::json!(message)),
        ("error", serde_json::Value::Null),
        ("pid", serde_json::Value::Null),
        ("ownerPid", serde_json::Value::Null),
        ("launcherPid", serde_json::Value::Null),
        ("childPid", serde_json::Value::Null),
    ];
    if let Some(cancel_path) = cancel_path {
        updates.push((
            "cancelPath",
            serde_json::json!(cancel_path.to_string_lossy()),
        ));
    }
    patch_sa3_training_status(status_path, &updates)
}

fn write_cancel_marker(cancel_path: &Path) -> Result<(), String> {
    if let Some(parent) = cancel_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    std::fs::write(cancel_path, "cancelled\n")
        .map_err(|e| format!("Cannot write {}: {}", cancel_path.display(), e))
}

fn terminate_process_tree(pid: u32) -> Result<(), String> {
    if pid == 0 {
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        let mut command = std::process::Command::new("taskkill");
        command.args(["/PID", &pid.to_string(), "/T", "/F"]);
        hide_std_console_window(&mut command);
        let output = command
            .output()
            .map_err(|e| format!("Failed to run taskkill for {}: {}", pid, e))?;
        if output.status.success() {
            return Ok(());
        }
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let normalized = text.to_lowercase();
        if normalized.contains("not found")
            || normalized.contains("not running")
            || normalized.contains("no running instance")
        {
            return Ok(());
        }
        Err(format!("taskkill {} failed: {}", pid, text.trim()))
    }

    #[cfg(not(target_os = "windows"))]
    {
        let output = std::process::Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .output()
            .map_err(|e| format!("Failed to run kill for {}: {}", pid, e))?;
        if output.status.success() {
            Ok(())
        } else {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            Err(format!("kill {} failed: {}", pid, text.trim()))
        }
    }
}

#[cfg(target_os = "windows")]
fn discover_sa3_training_pids(state: &Sa3LoraTrainingState) -> Vec<u32> {
    let mut command = std::process::Command::new("powershell");
    command.args([
        "-NoProfile",
        "-Command",
        "$ErrorActionPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'train_lora_job\\.py' -or $_.CommandLine -match 'pre_encode\\.py' -or $_.CommandLine -match 'lora_train\\.py' -or $_.CommandLine -match 'lora_train_lightning\\.py') } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
    ]);
    hide_std_console_window(&mut command);
    let output = command.output();

    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw.trim()) else {
        return Vec::new();
    };

    let mut needles = Vec::new();
    if let Some(job_id) = state.job_id.as_deref().filter(|value| !value.is_empty()) {
        needles.push(job_id.to_lowercase());
    }
    if let Some(run_dir) = state.run_dir.as_deref().filter(|value| !value.is_empty()) {
        needles.push(run_dir.to_lowercase());
    }
    if needles.is_empty() {
        return Vec::new();
    }

    let rows: Vec<serde_json::Value> = match value {
        serde_json::Value::Array(rows) => rows,
        serde_json::Value::Object(_) => vec![value],
        _ => Vec::new(),
    };

    rows.into_iter()
        .filter_map(|row| {
            let command_line = row.get("CommandLine")?.as_str()?.to_lowercase();
            if !needles.iter().any(|needle| command_line.contains(needle)) {
                return None;
            }
            row.get("ProcessId")
                .and_then(|pid| pid.as_u64())
                .and_then(|pid| u32::try_from(pid).ok())
        })
        .collect()
}

fn terminate_discovered_sa3_training_processes(state: &Sa3LoraTrainingState) -> Vec<String> {
    discover_sa3_training_pids(state)
        .into_iter()
        .filter_map(|pid| terminate_process_tree(pid).err())
        .collect()
}

#[cfg(not(target_os = "windows"))]
fn discover_sa3_training_pids(_state: &Sa3LoraTrainingState) -> Vec<u32> {
    Vec::new()
}

#[cfg(target_os = "windows")]
fn discover_carey_ace_training_pids(state: &CareyAceTrainingState) -> Vec<u32> {
    let mut command = std::process::Command::new("powershell");
    command.args([
        "-NoProfile",
        "-Command",
        "$ErrorActionPreference='SilentlyContinue'; Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'train_lora_job\\.py' -or $_.CommandLine -match 'lora_train_lightning\\.py' -or $_.CommandLine -match 'train\\.py') } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
    ]);
    hide_std_console_window(&mut command);
    let output = command.output();

    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let Ok(value) = serde_json::from_str::<serde_json::Value>(raw.trim()) else {
        return Vec::new();
    };

    let mut needles = Vec::new();
    if let Some(job_id) = state.job_id.as_deref().filter(|value| !value.is_empty()) {
        needles.push(job_id.to_lowercase());
    }
    if let Some(run_dir) = state.run_dir.as_deref().filter(|value| !value.is_empty()) {
        needles.push(run_dir.to_lowercase());
    }
    if needles.is_empty() {
        return Vec::new();
    }

    let rows: Vec<serde_json::Value> = match value {
        serde_json::Value::Array(rows) => rows,
        serde_json::Value::Object(_) => vec![value],
        _ => Vec::new(),
    };

    rows.into_iter()
        .filter_map(|row| {
            let command_line = row.get("CommandLine")?.as_str()?.to_lowercase();
            if !needles.iter().any(|needle| command_line.contains(needle)) {
                return None;
            }
            row.get("ProcessId")
                .and_then(|pid| pid.as_u64())
                .and_then(|pid| u32::try_from(pid).ok())
        })
        .collect()
}

fn terminate_discovered_carey_ace_training_processes(state: &CareyAceTrainingState) -> Vec<String> {
    discover_carey_ace_training_pids(state)
        .into_iter()
        .filter_map(|pid| terminate_process_tree(pid).err())
        .collect()
}

#[cfg(not(target_os = "windows"))]
fn discover_carey_ace_training_pids(_state: &CareyAceTrainingState) -> Vec<u32> {
    Vec::new()
}

async fn try_reload_carey_admin() -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(4))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };

    match client
        .post("http://localhost:8003/admin/reload")
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Ask a running SA3 service to re-apply its LoRA adapters.
///
/// SA3 bakes adapters into the pipeline at load time (api.py `load_pipeline` ->
/// `loaded.load_lora(paths)`), so rewriting the catalog/registry on disk does not
/// affect a service that is already running — switching a checkpoint would
/// silently keep serving the previously loaded adapter until a manual restart.
/// `/reload` does `load_pipeline(force=True)`, so this is a full model reload and
/// takes a while; it returns 409 when a generation is in flight. A false return
/// just means "not reloaded" (service down, busy, or timed out) and is not fatal.
async fn try_reload_sa3_admin() -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };

    match client.post("http://localhost:8006/reload").send().await {
        Ok(response) => {
            let status = response.status();
            if status.as_u16() == 409 {
                log::warn!(
                    "SA3 service was busy generating; LoRA adapters were not reloaded. \
                     Restart the SA3 service to apply the change."
                );
            }
            status.is_success()
        }
        Err(_) => false,
    }
}

async fn quit_application(handle: &tauri::AppHandle, manager: &ManagerState) {
    if let Err(error) = cancel_carey_ace_lora_training() {
        log::warn!("Could not cancel ACE-Step training during app quit: {error}");
    }
    if let Err(error) = cancel_sa3_lora_training() {
        log::warn!("Could not cancel SA3 LoRA training during app quit: {error}");
    }
    if let Err(error) = cancel_sa3_autolabel() {
        log::warn!("Could not cancel SA3 auto-label during app quit: {error}");
    }
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

pub(crate) fn carey_use_scrag_vae_enabled() -> bool {
    read_app_settings().carey_use_scrag_vae
}

pub(crate) fn sa3_loudness_env() -> Vec<(&'static str, String)> {
    let settings = read_app_settings().sa3_loudness;
    let tail_pad = settings.continuation_tail_pad;
    vec![
        ("SA3_PEAK_NORMALIZE_DB", settings.peak_normalize_db),
        ("SA3_LIMITER_CEILING_DB", settings.limiter_ceiling_db),
        ("SA3_LATENT_RESCALE", settings.latent_rescale),
        ("SA3_LATENT_SHIFT", settings.latent_shift),
        ("SA3_LATENT_TARGET_STD", settings.latent_target_std),
        ("SA3_TAIL_PAD_SECONDS", tail_pad.clone()),
        ("SA3_CONTINUE_TAIL_PAD", tail_pad),
    ]
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
    if let Err(error) = rebuild_tray_menu_with_info(app_handle, &services) {
        log::error!("Failed to rebuild tray menu: {}", error);
        append_startup_diagnostic(&format!("Failed to rebuild tray menu: {}", error));
    }
}

/// Inner sync function that builds the tray menu from pre-fetched service info.
fn rebuild_tray_menu_with_info(
    app_handle: &tauri::AppHandle,
    services: &[service_manager::ServiceInfo],
) -> Result<(), String> {
    let show_item = MenuItemBuilder::with_id("show", "show control center")
        .build(app_handle)
        .map_err(|error| format!("Cannot build tray show item: {}", error))?;
    let quit_item = MenuItemBuilder::with_id("quit", "quit (stop all services)")
        .build(app_handle)
        .map_err(|error| format!("Cannot build tray quit item: {}", error))?;
    let sep1 = PredefinedMenuItem::separator(app_handle)
        .map_err(|error| format!("Cannot build tray separator: {}", error))?;
    let sep2 = PredefinedMenuItem::separator(app_handle)
        .map_err(|error| format!("Cannot build tray separator: {}", error))?;

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
            .map_err(|error| format!("Cannot build tray service item for {}: {}", svc.id, error))?;
        menu_builder = menu_builder.item(&item);
    }

    let menu = menu_builder
        .item(&sep2)
        .item(&quit_item)
        .build()
        .map_err(|error| format!("Cannot build tray menu: {}", error))?;

    if let Some(tray) = app_handle.tray_by_id(&TrayIconId::new("main-tray")) {
        tray.set_menu(Some(menu))
            .map_err(|error| format!("Cannot set tray menu: {}", error))?;
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    install_panic_diagnostics();

    let webview2_user_data_folder = configure_webview2_user_data_folder();

    let builder = tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_dialog::init());
    let builder = if update::app_updater_enabled() {
        builder.plugin(tauri_plugin_updater::Builder::new().build())
    } else {
        builder
    };

    if let Err(error) = builder
        .setup(move |app| {
            if let Some(folder) = webview2_user_data_folder.as_ref() {
                log::info!("Using WebView2 user data folder: {}", folder.display());
            }

            app.manage(update::PendingNativeUpdate::default());

            // --- Resolve runtime root ---
            // In dev, run directly from the repo.
            // In production, sync bundled services into the selected runtime
            // storage root and run from there so logs/envs/source live in a
            // writable location that can be moved off the system drive.
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
                let runtime_root = storage::resolve_startup_runtime_root();
                if let Err(error) = sync_bundled_services_to_runtime(&bundle_root, &runtime_root) {
                    let message = format!("Runtime service sync failed: {}", error);
                    log::error!("{}", message);
                    append_startup_diagnostic(&message);

                    let manifest_path = runtime_root
                        .join("services")
                        .join("manifests")
                        .join("services.json");
                    if !manifest_path.exists() {
                        return Err(std::io::Error::other(message).into());
                    }
                }
                runtime_root
            };
            storage::set_active_runtime_root(&runtime_root);

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
                let icon = load_png_as_image(&icon_path);
                if icon.is_none() {
                    let message = format!(
                        "Runtime tray icon could not be loaded from {}; continuing without a tray icon",
                        icon_path.display()
                    );
                    log::warn!("{}", message);
                    append_startup_diagnostic(&message);
                }
                icon
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
            if let Some(window) = app.get_webview_window("main") {
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
            } else {
                let message = "Main webview window was not available during setup";
                log::error!("{}", message);
                append_startup_diagnostic(message);
            }

            // --- Background: health check + crash detection polling ---
            let poll_manager = manager.clone();
            let poll_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let client = match reqwest::Client::builder().build() {
                    Ok(client) => client,
                    Err(error) => {
                        log::error!("Failed to build health check client: {}", error);
                        return;
                    }
                };
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
                        if let Err(error) = rebuild_tray_menu_with_info(&poll_handle, &info) {
                            log::error!("Failed to rebuild tray menu: {}", error);
                            append_startup_diagnostic(&format!(
                                "Failed to rebuild tray menu: {}",
                                error
                            ));
                        }
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
            get_carey_ace_dataset_sidecars,
            save_carey_ace_dataset_sidecars,
            get_carey_ace_lora_training_state,
            start_carey_ace_lora_training,
            cancel_carey_ace_lora_training,
            get_sa3_lora_state,
            upsert_sa3_lora,
            activate_sa3_lora_checkpoint,
            remove_sa3_lora,
            build_sa3_lora_prompts,
            get_sa3_dataset_sidecars,
            save_sa3_dataset_sidecars,
            suggest_sa3_track_metadata,
            sa3_autolabel_available,
            get_sa3_autolabel_availability,
            get_sa3_autolabel_state,
            start_sa3_autolabel,
            cancel_sa3_autolabel,
            open_sa3_training_reference,
            get_sa3_lora_training_state,
            start_sa3_lora_training,
            cancel_sa3_lora_training,
            get_hf_token,
            save_hf_token,
            delete_hf_token,
            get_runtime_storage_info,
            save_runtime_storage_root,
            reset_runtime_storage_root,
            get_legacy_storage_maintenance_info,
            migrate_legacy_loras,
            cleanup_legacy_storage,
            get_app_settings,
            save_app_settings,
            check_for_app_update,
            install_app_update,
            resolve_close_request,
            open_url,
            reveal_path,
        ])
        .run(tauri::generate_context!())
    {
        let message = format!("error while running tauri application: {}", error);
        log::error!("{}", message);
        append_startup_diagnostic(&message);
        eprintln!("{}", message);
        std::process::exit(1);
    }
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

const WINDOWS_APPLICATION_CONTROL_HELP: &str = "Windows got a little overprotective and blocked this environment's Python from starting.\n\nOpen Windows Security -> App & browser control -> Smart App Control settings, temporarily turn Smart App Control off, then rebuild the environment. You can turn it back on afterward.";

fn is_windows_application_control_block(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("an application control policy has blocked this file")
        || message.contains("os error 4551")
}

#[cfg(test)]
mod windows_application_control_tests {
    use super::is_windows_application_control_block;

    #[test]
    fn recognizes_application_control_policy_message() {
        assert!(is_windows_application_control_block(
            "An Application Control policy has blocked this file."
        ));
    }

    #[test]
    fn recognizes_windows_error_code() {
        assert!(is_windows_application_control_block(
            "Failed to query Python interpreter (os error 4551)"
        ));
    }

    #[test]
    fn ignores_unrelated_build_errors() {
        assert!(!is_windows_application_control_block(
            "Failed to resolve package dependencies (exit code 1)"
        ));
    }
}

/// Ensure uv is installed. Returns the path to the uv executable.
async fn ensure_uv(
    service_id: &str,
    manager: &Arc<Mutex<ServiceManager>>,
    handle: &tauri::AppHandle,
    runtime_root: &Path,
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
    apply_runtime_env(&mut check_cmd, runtime_root);
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
            apply_runtime_env(&mut check_cmd, runtime_root);
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
    apply_runtime_env(&mut install_cmd, runtime_root);
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
    apply_runtime_env(&mut recheck_cmd, runtime_root);
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
    let runtime_root = &build_info.runtime_root;
    let uv = ensure_uv(&service_id, &manager, &handle, runtime_root).await?;

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
            runtime_root,
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
        apply_runtime_env(&mut venv_cmd, runtime_root);
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
        apply_runtime_env(&mut child_cmd, runtime_root);
        hide_console_window(&mut child_cmd);
        let mut child = child_cmd
            .spawn()
            .map_err(|e| format!("Build step failed '{}': {}", uv_command_str, e))?;

        // Stream stdout and stderr, remembering whether Windows Application
        // Control blocked the environment's Python interpreter.
        let application_control_blocked = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let mgr_stdout = manager.clone();
        let sid_stdout = service_id.clone();
        let stdout_application_control_blocked = application_control_blocked.clone();

        let stdout_task = tauri::async_runtime::spawn(async move {
            if let Some(stdout) = stdout {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if is_windows_application_control_block(&line) {
                        stdout_application_control_blocked
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    let mut mgr = mgr_stdout.lock().await;
                    mgr.append_build_log(&sid_stdout, &line);
                }
            }
        });

        let mgr_stderr = manager.clone();
        let sid_stderr = service_id.clone();
        let stderr_application_control_blocked = application_control_blocked.clone();

        let stderr_task = tauri::async_runtime::spawn(async move {
            if let Some(stderr) = stderr {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if is_windows_application_control_block(&line) {
                        stderr_application_control_blocked
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                    }
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
            let was_application_control_blocked =
                application_control_blocked.load(std::sync::atomic::Ordering::Relaxed);
            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(&service_id, &format!("\nERROR: {}", msg));
                if was_application_control_blocked {
                    mgr.append_build_log(
                        &service_id,
                        &format!("\n{}", WINDOWS_APPLICATION_CONTROL_HELP),
                    );
                }
            }
            emit_status(&manager, &handle).await;
            if was_application_control_blocked {
                return Err(WINDOWS_APPLICATION_CONTROL_HELP.to_string());
            }
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
    runtime_root: &Path,
) -> Result<(), String> {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args).current_dir(work_dir);
    apply_runtime_env(&mut cmd, runtime_root);
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
    models.extend(mgr.get_sa3_models());
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
        "sa3" => "sa3",
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
            "sa3" => "SA3 (Stable Audio 3)",
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
        let runtime_root = root.clone();
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_carey_model(
                model_id,
                python_exe,
                runtime_root,
                checkpoint_dir,
                mgr_clone,
                handle,
            )
            .await;
        });
    } else if model_id.starts_with("foundation::") {
        let runtime_root = root.clone();
        let model_dir = storage::models_dir(&root).join("foundation-1");
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_foundation_model(
                model_id,
                python_exe,
                runtime_root,
                model_dir,
                mgr_clone,
                handle,
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
    let (scale, backends, inferred_model_family) =
        load_carey_lora_metadata(&checkpoint_dir, captions_dir.as_deref(), existing);
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
    let normalized_name =
        sanitize_lora_name(&name).ok_or_else(|| "Invalid LoRA name".to_string())?;
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

    let script_path = repo_root
        .join("services")
        .join("carey")
        .join("build_captions.py");
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

#[tauri::command]
fn get_carey_ace_dataset_sidecars(
    dataset_path: String,
) -> Result<Vec<CareyAceDatasetSidecarEntry>, String> {
    read_carey_ace_dataset_sidecars(&dataset_path)
}

#[tauri::command]
fn save_carey_ace_dataset_sidecars(
    dataset_path: String,
    sidecars: Vec<CareyAceDatasetSidecarUpdate>,
) -> Result<CareyAceDatasetSidecarSaveResult, String> {
    save_carey_ace_dataset_sidecar_updates(&dataset_path, sidecars)
}

#[tauri::command]
fn get_carey_ace_lora_training_state() -> Result<CareyAceTrainingState, String> {
    Ok(read_carey_ace_lora_training_state())
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn start_carey_ace_lora_training(
    name: String,
    dataset_path: String,
    model: String,
    adapter_type: String,
    module_profile: String,
    trigger: String,
    timestep_mu: f64,
    auto_caption: bool,
    caption_lm_model: String,
    overwrite_captions: bool,
    genre_ratio: u32,
    prepare_only: bool,
    include_auto_timesignature: bool,
    bpm_analysis: bool,
    key_analysis: bool,
    analysis_duration: f64,
    rank: u32,
    epochs: u32,
    save_every: u32,
    batch_size: u32,
    gradient_accumulation: u32,
    learning_rate: f64,
    cfg_ratio: f64,
    loss_weighting: String,
    snr_gamma: f64,
    max_duration: f64,
    manager: tauri::State<'_, ManagerState>,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<CareyAceTrainingState, String> {
    let normalized_name = sanitize_lora_name(&name)
        .ok_or_else(|| "LoRA name must use lowercase letters, numbers, '-' or '_'".to_string())?;
    let model = match model.trim() {
        "base" => "base",
        "xl-base" => "xl-base",
        other => return Err(format!("Unknown ACE-Step model: {}", other)),
    };
    let adapter_type = match adapter_type.trim() {
        "lora" => "lora",
        "dora" => "dora",
        other => return Err(format!("Unknown adapter type: {}", other)),
    };
    let module_profile = match module_profile.trim() {
        "attention" => "attention",
        "balanced" => "balanced",
        other => return Err(format!("Unknown adapter module profile: {}", other)),
    };
    let loss_weighting = match loss_weighting.trim() {
        "none" => "none",
        "min_snr" => "min_snr",
        other => return Err(format!("Unknown loss weighting: {}", other)),
    };
    match caption_lm_model.trim() {
        "acestep-5Hz-lm-0.6B" | "acestep-5Hz-lm-1.7B" | "acestep-5Hz-lm-4B" => {}
        other => return Err(format!("Unknown ACE-Step LM model: {}", other)),
    }
    if rank == 0 {
        return Err("LoRA rank must be greater than 0".to_string());
    }
    if epochs == 0 {
        return Err("epochs must be greater than 0".to_string());
    }
    if save_every == 0 {
        return Err("save interval must be greater than 0".to_string());
    }
    if batch_size == 0 || gradient_accumulation == 0 {
        return Err("batch size and gradient accumulation must be greater than 0".to_string());
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be greater than 0".to_string());
    }
    if !timestep_mu.is_finite() {
        return Err("timestep mu must be finite".to_string());
    }
    if !cfg_ratio.is_finite() || !(0.0..1.0).contains(&cfg_ratio) {
        return Err("CFG ratio must be in [0, 1).".to_string());
    }
    if !snr_gamma.is_finite() || snr_gamma <= 0.0 {
        return Err("SNR gamma must be greater than 0".to_string());
    }
    if !max_duration.is_finite() || max_duration <= 0.0 {
        return Err("max duration must be greater than 0".to_string());
    }
    if !analysis_duration.is_finite() || analysis_duration < 0.0 {
        return Err("analysis duration must be 0 or greater".to_string());
    }
    if genre_ratio > 100 {
        return Err("genre ratio must be between 0 and 100".to_string());
    }

    let current_state = read_carey_ace_lora_training_state();
    if matches!(current_state.status.as_str(), "starting" | "running") {
        return Err(format!(
            "ACE-Step LoRA training job '{}' is already {}.",
            current_state
                .name
                .as_deref()
                .unwrap_or(current_state.job_id.as_deref().unwrap_or("current")),
            current_state.phase
        ));
    }

    let dataset_dir = PathBuf::from(dataset_path.trim());
    if !dataset_dir.is_dir() {
        return Err(format!(
            "{} is not a valid dataset folder",
            dataset_dir.display()
        ));
    }

    let python_exe = repo_root
        .join("services")
        .join("carey")
        .join("env")
        .join("Scripts")
        .join("python.exe");
    if !python_exe.exists() {
        return Err("Carey must be built before ACE-Step LoRA training can start.".to_string());
    }

    let script_path = repo_root
        .join("services")
        .join("carey")
        .join("train_lora_job.py");
    if !script_path.exists() {
        return Err(format!("Missing {}", script_path.display()));
    }

    let checkpoint_dir = carey_checkpoint_dir(repo_root.inner());
    let model_folder = if model == "xl-base" {
        "acestep-v15-xl-base"
    } else {
        "acestep-v15-base"
    };
    if !prepare_only || auto_caption {
        let missing = carey_training_required_checkpoint_files(
            &checkpoint_dir,
            model_folder,
            auto_caption.then_some(caption_lm_model.trim()),
        )
        .into_iter()
        .filter(|path| !path.is_file())
        .map(|path| {
            path.strip_prefix(&checkpoint_dir)
                .unwrap_or(&path)
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "Download or repair the selected ACE-Step models from Carey's Models panel before continuing. Missing: {}",
                missing.join(", ")
            ));
        }
    }

    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let job_id = format!("{}-{}", normalized_name, epoch);
    let run_dir = carey_training_jobs_dir().join(&job_id);
    let log_path = carey_training_logs_dir().join(format!("{}.log", job_id));
    let status_path = run_dir.join("status.json");
    let cancel_path = run_dir.join("cancel.requested");

    std::fs::create_dir_all(&run_dir)
        .map_err(|e| format!("Cannot create {}: {}", run_dir.display(), e))?;
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    if let Some(parent) = carey_lora_catalog_path().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }

    let log_file = std::fs::File::create(&log_path)
        .map_err(|e| format!("Cannot create {}: {}", log_path.display(), e))?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|e| format!("Cannot clone log handle: {}", e))?;

    // All non-mutating validation and local file preparation must succeed
    // before stopping the managed inference service.
    {
        let mut mgr = manager.lock().await;
        if mgr.is_running("carey") {
            mgr.stop("carey").map_err(|e| {
                format!(
                    "Could not stop Carey inference before ACE-Step training: {}",
                    e
                )
            })?;
        }
    }

    write_carey_ace_training_launch_state(
        &status_path,
        &job_id,
        &normalized_name,
        &run_dir,
        &log_path,
        &cancel_path,
        epochs,
        if prepare_only {
            "Launching ACE-Step dataset preparation."
        } else {
            "Launching ACE-Step LoRA training."
        },
    )?;

    let mut cmd = tokio::process::Command::new(&python_exe);
    hide_console_window(&mut cmd);
    cmd.arg("-u")
        .arg(&script_path)
        .arg("--job-id")
        .arg(&job_id)
        .arg("--name")
        .arg(&normalized_name)
        .arg("--dataset-dir")
        .arg(&dataset_dir)
        .arg("--checkpoint-dir")
        .arg(&checkpoint_dir)
        .arg("--run-dir")
        .arg(&run_dir)
        .arg("--status-path")
        .arg(&status_path)
        .arg("--current-job-path")
        .arg(carey_training_current_job_path())
        .arg("--cancel-path")
        .arg(&cancel_path)
        .arg("--log-path")
        .arg(&log_path)
        .arg("--model")
        .arg(model)
        .arg("--trigger")
        .arg(trigger.trim())
        .arg("--tag-position")
        .arg("prepend")
        .arg("--genre-ratio")
        .arg(genre_ratio.to_string())
        .arg("--caption")
        .arg(if auto_caption {
            "understand_music"
        } else {
            "skip"
        })
        .arg("--carey-url")
        .arg("http://127.0.0.1:8013")
        .arg("--inference-carey-url")
        .arg("http://127.0.0.1:8003")
        .arg("--caption-lm-model")
        .arg(caption_lm_model.trim())
        .arg("--analysis-duration")
        .arg(analysis_duration.to_string())
        .arg("--rank")
        .arg(rank.to_string())
        .arg("--alpha")
        .arg((rank.saturating_mul(2)).to_string())
        .arg("--learning-rate")
        .arg(learning_rate.to_string())
        .arg("--timestep-mu")
        .arg(timestep_mu.to_string())
        .arg("--cfg-ratio")
        .arg(cfg_ratio.to_string())
        .arg("--loss-weighting")
        .arg(loss_weighting)
        .arg("--snr-gamma")
        .arg(snr_gamma.to_string())
        .arg("--epochs")
        .arg(epochs.to_string())
        .arg("--save-every")
        .arg(save_every.to_string())
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--gradient-accumulation")
        .arg(gradient_accumulation.to_string())
        .arg("--max-duration")
        .arg(max_duration.to_string())
        .arg("--adapter-type")
        .arg(adapter_type)
        .arg("--module-profile")
        .arg(module_profile)
        .arg("--lora-catalog-path")
        .arg(carey_lora_catalog_path())
        .arg("--lora-registry-path")
        .arg(carey_lora_registry_path())
        .arg("--captions-json-path")
        .arg(carey_captions_path());
    if overwrite_captions {
        cmd.arg("--overwrite-captions");
    }
    if prepare_only {
        cmd.arg("--prepare-only");
    }
    if include_auto_timesignature {
        cmd.arg("--include-auto-timesignature");
    }
    if !bpm_analysis {
        cmd.arg("--no-bpm-analysis");
    }
    if !key_analysis {
        cmd.arg("--no-key-analysis");
    }

    cmd.current_dir(repo_root.join("services").join("carey"))
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file_err))
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1");
    workload_job::configure_tokio_command(&mut cmd);

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => {
            let message = format!("Failed to launch ACE-Step LoRA training: {}", error);
            let _ = mark_carey_ace_training_failed(&status_path, &message, None);
            return Err(message);
        }
    };
    if let Err(error) = workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        let message = format!(
            "Failed to enroll ACE-Step training in the managed workload group: {}",
            error
        );
        let _ = mark_carey_ace_training_failed(&status_path, &message, None);
        return Err(message);
    }
    let Some(pid) = child.id() else {
        let _ = child.kill().await;
        let message = "ACE-Step LoRA training launched without a process ID.".to_string();
        let _ = mark_carey_ace_training_failed(&status_path, &message, None);
        return Err(message);
    };
    if let Err(error) = write_carey_ace_training_process_id(&status_path, pid) {
        let _ = child.kill().await;
        let message = format!(
            "Failed to record ACE-Step training process {}: {}",
            pid, error
        );
        let _ = mark_carey_ace_training_failed(&status_path, &message, None);
        return Err(message);
    }

    let watched_status_path = status_path.clone();
    tauri::async_runtime::spawn(async move {
        let exit_result = child.wait().await;
        let state = read_carey_ace_training_status_file(&watched_status_path);
        let warnings = if state.launcher_pid == Some(pid)
            || (state.launcher_pid.is_none() && state.pid == Some(pid))
        {
            terminate_discovered_carey_ace_training_processes(&state)
        } else {
            Vec::new()
        };
        let cleanup_suffix = if warnings.is_empty() {
            String::new()
        } else {
            format!(" Process cleanup warning: {}", warnings.join("; "))
        };
        let message = match exit_result {
            Ok(status) => format!(
                "ACE-Step training launcher {} exited without recording a terminal state ({}). Check the training log for details.{}",
                pid, status, cleanup_suffix
            ),
            Err(error) => format!(
                "Could not monitor ACE-Step training launcher {}: {}. Check the training log for details.{}",
                pid, error, cleanup_suffix
            ),
        };
        let _ = mark_carey_ace_training_failed(&watched_status_path, &message, Some(pid));
    });

    Ok(read_carey_ace_lora_training_state())
}

#[tauri::command]
fn cancel_carey_ace_lora_training() -> Result<CareyAceTrainingState, String> {
    let state = read_carey_ace_lora_training_state();
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(state);
    }

    let cancel_path = state.cancel_path.as_deref().map(PathBuf::from).or_else(|| {
        state
            .run_dir
            .as_deref()
            .map(|run_dir| PathBuf::from(run_dir).join("cancel.requested"))
    });
    if let Some(path) = cancel_path.as_ref() {
        write_cancel_marker(path)?;
    }

    let mut pids = Vec::new();
    if let Some(pid) = state.child_pid {
        pids.push(pid);
    }
    if let Some(pid) = state.pid {
        if !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    if let Some(pid) = state.launcher_pid {
        if !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    for pid in discover_carey_ace_training_pids(&state) {
        if !pids.contains(&pid) {
            pids.push(pid);
        }
    }

    let mut warnings = Vec::new();
    for pid in pids {
        if let Err(error) = terminate_process_tree(pid) {
            warnings.push(error);
        }
    }

    let message = if warnings.is_empty() {
        "Training cancelled.".to_string()
    } else {
        format!(
            "Training cancelled. Process cleanup warning: {}",
            warnings.join("; ")
        )
    };
    if let Some(status_path) = carey_ace_current_training_status_path() {
        write_carey_ace_training_cancelled_state(&status_path, cancel_path.as_deref(), &message)?;
    }

    Ok(read_carey_ace_lora_training_state())
}

#[tauri::command]
fn get_sa3_lora_state(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3LoraState, String> {
    build_sa3_lora_state(repo_root.inner())
}

fn install_managed_sa3_checkpoint(source: &Path, destination: &Path) -> Result<(), String> {
    let parent = destination
        .parent()
        .ok_or_else(|| format!("{} has no parent folder", destination.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;

    let file_name = destination
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let suffix = format!("{}-{nonce}", std::process::id());
    let staged = parent.join(format!(".{file_name}.{suffix}.tmp"));
    let backup = parent.join(format!(".{file_name}.{suffix}.bak"));

    let source_len = std::fs::metadata(source)
        .map_err(|e| format!("Cannot inspect {}: {}", source.display(), e))?
        .len();
    let copied = std::fs::copy(source, &staged).map_err(|e| {
        format!(
            "Cannot stage checkpoint {} at {}: {}",
            source.display(),
            staged.display(),
            e
        )
    })?;
    if copied != source_len {
        let _ = std::fs::remove_file(&staged);
        return Err(format!(
            "Checkpoint copy was incomplete: copied {copied} of {source_len} bytes"
        ));
    }

    let had_destination = destination.is_file();
    if had_destination {
        if let Err(error) = std::fs::rename(destination, &backup) {
            let _ = std::fs::remove_file(&staged);
            return Err(format!(
                "Cannot prepare {} for replacement: {}",
                destination.display(),
                error
            ));
        }
    }

    if let Err(error) = std::fs::rename(&staged, destination) {
        if had_destination {
            let _ = std::fs::rename(&backup, destination);
        }
        let _ = std::fs::remove_file(&staged);
        return Err(format!(
            "Cannot activate checkpoint at {}: {}",
            destination.display(),
            error
        ));
    }

    if had_destination {
        let _ = std::fs::remove_file(&backup);
    }
    Ok(())
}

#[cfg(test)]
mod sa3_lora_checkpoint_tests {
    use super::{install_managed_sa3_checkpoint, Sa3LoraCatalogEntry};

    fn temp_test_dir() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "gary4local-sa3-checkpoint-test-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn manual_catalog_entries_have_no_training_checkpoint_history() {
        let entry: Sa3LoraCatalogEntry = serde_json::from_value(serde_json::json!({
            "path": "C:\\loras\\manual.safetensors",
            "strength": 1.0
        }))
        .expect("manual catalog entry should deserialize");

        assert!(entry.training_job_id.is_none());
        assert!(entry.training_checkpoints.is_empty());
        assert!(entry.selected_training_step.is_none());
    }

    #[test]
    fn managed_checkpoint_install_replaces_the_active_file() {
        let root = temp_test_dir();
        std::fs::create_dir_all(&root).expect("create test folder");
        let source = root.join("step-500.safetensors");
        let destination = root.join("active.safetensors");
        std::fs::write(&source, b"checkpoint-500").expect("write source checkpoint");
        std::fs::write(&destination, b"checkpoint-1000").expect("write active checkpoint");

        install_managed_sa3_checkpoint(&source, &destination)
            .expect("checkpoint activation should succeed");

        assert_eq!(
            std::fs::read(&destination).expect("read active checkpoint"),
            b"checkpoint-500"
        );
        let leftovers = std::fs::read_dir(&root)
            .expect("read test folder")
            .flatten()
            .filter(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                name.ends_with(".tmp") || name.ends_with(".bak")
            })
            .count();
        assert_eq!(leftovers, 0);

        std::fs::remove_dir_all(root).expect("remove test folder");
    }
}

#[tauri::command]
async fn upsert_sa3_lora(
    name: String,
    checkpoint_path: String,
    prompts_path: Option<String>,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3LoraState, String> {
    let normalized_name = sanitize_lora_name(&name)
        .ok_or_else(|| "LoRA name must use lowercase letters, numbers, '-' or '_'".to_string())?;
    let checkpoint_file = PathBuf::from(checkpoint_path.trim());
    if !looks_like_sa3_lora_checkpoint(&checkpoint_file) {
        return Err(format!(
            "{} does not look like an SA3 LoRA checkpoint file",
            checkpoint_file.display()
        ));
    }

    let normalized_prompts_path = prompts_path
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let prompts_dir = normalized_prompts_path.as_ref().map(PathBuf::from);
    if let Some(prompts_dir) = prompts_dir.as_ref() {
        if !prompts_dir.is_dir() {
            return Err(format!(
                "{} is not a valid prompts/source folder",
                prompts_dir.display()
            ));
        }
    }

    let mut catalog = read_sa3_lora_catalog()?;
    let strength = catalog
        .get(&normalized_name)
        .map(|entry| entry.strength)
        .unwrap_or_else(default_lora_scale);
    catalog.insert(
        normalized_name,
        Sa3LoraCatalogEntry {
            path: checkpoint_file.to_string_lossy().to_string(),
            prompts_path: normalized_prompts_path,
            strength,
            training_job_id: None,
            training_checkpoints: Vec::new(),
            selected_training_step: None,
        },
    );
    save_sa3_lora_catalog(&catalog)?;
    let state = build_sa3_lora_state(repo_root.inner())?;
    // Adapters are baked into the pipeline at load time, so a running SA3 service
    // would keep serving the old adapter until restarted. Ask it to re-apply.
    let _ = try_reload_sa3_admin().await;
    Ok(state)
}

#[tauri::command]
async fn activate_sa3_lora_checkpoint(
    name: String,
    step: u32,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3LoraState, String> {
    let normalized_name =
        sanitize_lora_name(&name).ok_or_else(|| "Invalid LoRA name".to_string())?;
    let mut catalog = read_sa3_lora_catalog()?;
    let checkpoint = {
        let entry = catalog
            .get(&normalized_name)
            .ok_or_else(|| format!("LoRA '{}' is not registered", normalized_name))?;
        if entry.training_job_id.is_none() || entry.training_checkpoints.is_empty() {
            return Err(format!(
                "LoRA '{}' was not registered by Gary's SA3 trainer",
                normalized_name
            ));
        }
        entry
            .training_checkpoints
            .iter()
            .find(|checkpoint| checkpoint.step == step)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "Training checkpoint step {} is not registered for '{}'",
                    step, normalized_name
                )
            })?
    };

    let source = PathBuf::from(&checkpoint.path);
    if !looks_like_sa3_lora_checkpoint(&source) {
        return Err(format!(
            "Training checkpoint {} is missing or invalid",
            source.display()
        ));
    }

    let managed_path = sa3_lora_dir().join(format!("{}.safetensors", normalized_name));
    install_managed_sa3_checkpoint(&source, &managed_path)?;

    if let Some(entry) = catalog.get_mut(&normalized_name) {
        entry.path = managed_path.to_string_lossy().to_string();
        entry.selected_training_step = Some(checkpoint.step);
    }
    save_sa3_lora_catalog(&catalog)?;
    let state = build_sa3_lora_state(repo_root.inner())?;
    // Adapters are baked into the pipeline at load time, so a running SA3 service
    // would keep serving the old adapter until restarted. Ask it to re-apply.
    let _ = try_reload_sa3_admin().await;
    Ok(state)
}

#[tauri::command]
async fn remove_sa3_lora(
    name: String,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3LoraState, String> {
    let normalized_name =
        sanitize_lora_name(&name).ok_or_else(|| "Invalid LoRA name".to_string())?;
    let mut catalog = read_sa3_lora_catalog()?;
    catalog.remove(&normalized_name);
    save_sa3_lora_catalog(&catalog)?;
    let state = build_sa3_lora_state(repo_root.inner())?;
    // Adapters are baked into the pipeline at load time, so a running SA3 service
    // would keep serving the old adapter until restarted. Ask it to re-apply.
    let _ = try_reload_sa3_admin().await;
    Ok(state)
}

#[tauri::command]
async fn build_sa3_lora_prompts(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3PromptsBuildResult, String> {
    let initial_state = build_sa3_lora_state(repo_root.inner())?;
    let python_exe = repo_root
        .join("services")
        .join("sa3")
        .join("env")
        .join("Scripts")
        .join("python.exe");
    if !python_exe.exists() {
        return Err("SA3 must be built before prompts can be generated.".to_string());
    }

    let script_path = repo_root
        .join("services")
        .join("sa3")
        .join("build_lora_prompts.py");
    if !script_path.exists() {
        return Err(format!("Missing {}", script_path.display()));
    }

    std::fs::create_dir_all(sa3_prompts_dir())
        .map_err(|e| format!("Cannot create {}: {}", sa3_prompts_dir().display(), e))?;

    let mut outputs = Vec::new();
    for entry in &initial_state.entries {
        if !entry.registered || entry.caption_count == 0 {
            continue;
        }
        let Some(source_path) = entry.resolved_prompts_path.as_ref() else {
            continue;
        };

        let mut cmd = tokio::process::Command::new(&python_exe);
        hide_console_window(&mut cmd);
        cmd.arg(&script_path)
            .arg("--name")
            .arg(&entry.name)
            .arg("--captions-dir")
            .arg(source_path)
            .arg("--out-dir")
            .arg(sa3_prompts_dir())
            .arg("--force")
            .current_dir(repo_root.join("services").join("sa3"));

        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to run build_lora_prompts.py: {}", e))?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let combined_output = match (stdout.is_empty(), stderr.is_empty()) {
            (false, false) => format!("{}\n{}", stdout, stderr),
            (false, true) => stdout,
            (true, false) => stderr,
            (true, true) => format!("{}: no output", entry.name),
        };
        if !output.status.success() {
            return Err(combined_output);
        }
        outputs.push(combined_output);
    }

    if outputs.is_empty() {
        outputs.push("No registered SA3 LoRAs with txt sidecars were found.".to_string());
    }

    ensure_default_sa3_prompts(repo_root.inner())?;
    let state = build_sa3_lora_state(repo_root.inner())?;
    Ok(Sa3PromptsBuildResult {
        state,
        output: outputs.join("\n"),
    })
}

#[tauri::command]
fn get_sa3_dataset_sidecars(dataset_path: String) -> Result<Vec<Sa3DatasetSidecarEntry>, String> {
    read_sa3_dataset_sidecars(&dataset_path)
}

#[tauri::command]
fn save_sa3_dataset_sidecars(
    dataset_path: String,
    sidecars: Vec<Sa3DatasetSidecarUpdate>,
) -> Result<Sa3DatasetSidecarSaveResult, String> {
    save_sa3_dataset_sidecar_updates(&dataset_path, sidecars)
}

#[tauri::command]
async fn suggest_sa3_track_metadata(
    dataset_path: String,
    audio_path: String,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3TrackMetadataSuggestion, String> {
    let root = canonical_sa3_dataset_root(&dataset_path)?;
    let requested = PathBuf::from(audio_path.trim());
    let resolved = requested
        .canonicalize()
        .map_err(|e| format!("Cannot resolve {}: {}", requested.display(), e))?;
    if !resolved.starts_with(&root) || !resolved.is_file() || !is_sa3_dataset_audio_file(&resolved)
    {
        return Err(format!(
            "{} is not an audio file inside {}",
            resolved.display(),
            root.display()
        ));
    }

    let python_exe = repo_root
        .join("services")
        .join("sa3")
        .join("env")
        .join("Scripts")
        .join("python.exe");
    if !python_exe.exists() {
        return Err("SA3 must be built before BPM/key analysis can run.".to_string());
    }
    let script_path = repo_root
        .join("services")
        .join("sa3")
        .join("analyze_audio.py");
    if !script_path.exists() {
        return Err(format!("Missing {}", script_path.display()));
    }

    let mut cmd = tokio::process::Command::new(&python_exe);
    hide_console_window(&mut cmd);
    cmd.arg(&script_path)
        .arg(&resolved)
        .current_dir(repo_root.join("services").join("sa3"));

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run analyze_audio.py: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    // scipy install logs and estimator diagnostics go to stderr; the JSON result is on
    // stdout. Only fall back to stderr when stdout is empty (a hard crash).
    if stdout.is_empty() {
        return Err(if stderr.is_empty() {
            "BPM/key analysis produced no output.".to_string()
        } else {
            stderr
        });
    }
    let parsed: Sa3AnalyzerOutput = serde_json::from_str(&stdout)
        .map_err(|e| format!("Could not parse analyzer output: {} ({})", e, stdout))?;
    if !parsed.ok {
        return Err(parsed
            .error
            .unwrap_or_else(|| "BPM/key analysis failed.".to_string()));
    }
    Ok(Sa3TrackMetadataSuggestion {
        bpm: parsed.bpm,
        keyscale: parsed.keyscale,
        suggestion: parsed.suggestion,
        bpm_confidence: parsed.bpm_confidence,
        key_confidence: parsed.key_confidence,
    })
}

fn read_sa3_autolabel_state() -> Sa3AutolabelState {
    match std::fs::read_to_string(sa3_autolabel_status_path()) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => Sa3AutolabelState::default(),
    }
}

fn read_reconciled_sa3_autolabel_state() -> Sa3AutolabelState {
    let status_path = sa3_autolabel_status_path();
    reconcile_owned_workload_status(
        &status_path,
        "SA3 auto-label",
        carey_ace_training_parent_is_running,
    )
    .unwrap_or_else(|error| {
        log::warn!("Could not reconcile SA3 auto-label status: {error}");
    });
    read_sa3_autolabel_state()
}

fn write_sa3_autolabel_state(status_path: &Path, state: &Sa3AutolabelState) -> Result<(), String> {
    if let Some(parent) = status_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    let raw = serde_json::to_string_pretty(state)
        .map_err(|e| format!("Cannot serialize auto-label state: {}", e))?;
    std::fs::write(status_path, raw)
        .map_err(|e| format!("Cannot write {}: {}", status_path.display(), e))
}

fn write_sa3_autolabel_launch_state(status_path: &Path, style: &str) -> Result<(), String> {
    let state = Sa3AutolabelState {
        status: "starting".to_string(),
        phase: "starting".to_string(),
        message: "Launching auto-label".to_string(),
        style: style.to_string(),
        owner_pid: Some(std::process::id()),
        ..Default::default()
    };
    write_sa3_autolabel_state(status_path, &state)
}

fn write_sa3_autolabel_process_id(status_path: &Path, pid: u32) -> Result<(), String> {
    let mut state = read_sa3_autolabel_state();
    state.pid = Some(pid);
    write_sa3_autolabel_state(status_path, &state)
}

fn mark_sa3_autolabel_failed(message: &str, expected_pid: Option<u32>) -> Result<(), String> {
    let mut state = read_sa3_autolabel_state();
    if let Some(expected_pid) = expected_pid {
        if state.pid != Some(expected_pid) {
            return Ok(());
        }
    }
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(());
    }
    state.status = "failed".to_string();
    state.phase = "error".to_string();
    state.message = message.to_string();
    state.error = Some(message.to_string());
    state.current_path = String::new();
    state.pid = None;
    state.owner_pid = None;
    write_sa3_autolabel_state(&sa3_autolabel_status_path(), &state)
}

fn mark_sa3_autolabel_cancelled() {
    // The python process may be killed before it can record the terminal state.
    let mut state = read_sa3_autolabel_state();
    state.status = "cancelled".to_string();
    state.phase = "cancelled".to_string();
    state.message = "Auto-label cancelled".to_string();
    state.current_path = String::new();
    state.error = None;
    state.pid = None;
    state.owner_pid = None;
    let _ = write_sa3_autolabel_state(&sa3_autolabel_status_path(), &state);
}

fn carey_autolabel_python(repo_root: &Path) -> PathBuf {
    repo_root
        .join("services")
        .join("carey")
        .join("env")
        .join("Scripts")
        .join("python.exe")
}

fn carey_autolabel_script(repo_root: &Path) -> PathBuf {
    repo_root
        .join("services")
        .join("carey")
        .join("sa3_autolabel.py")
}

fn carey_sa3_captioner_downloaded(repo_root: &Path) -> bool {
    let model_dir = repo_root
        .join("services")
        .join("carey")
        .join("checkpoints")
        .join("acestep-5Hz-lm-1.7B");
    model_dir.join("config.json").is_file() && model_dir.join("model.safetensors").is_file()
}

fn build_sa3_autolabel_availability(repo_root: &Path) -> Sa3AutolabelAvailability {
    let carey_built =
        carey_autolabel_python(repo_root).is_file() && carey_autolabel_script(repo_root).is_file();
    let captioner_downloaded = carey_sa3_captioner_downloaded(repo_root);
    Sa3AutolabelAvailability {
        available: carey_built && captioner_downloaded,
        carey_built,
        captioner_downloaded,
    }
}

#[tauri::command]
fn sa3_autolabel_available(repo_root: tauri::State<'_, std::path::PathBuf>) -> bool {
    build_sa3_autolabel_availability(repo_root.inner()).available
}

#[tauri::command]
fn get_sa3_autolabel_availability(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Sa3AutolabelAvailability {
    build_sa3_autolabel_availability(repo_root.inner())
}

#[tauri::command]
fn get_sa3_autolabel_state() -> Sa3AutolabelState {
    read_reconciled_sa3_autolabel_state()
}

#[tauri::command]
async fn start_sa3_autolabel(
    dataset_path: String,
    style: String,
    manager: tauri::State<'_, ManagerState>,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3AutolabelState, String> {
    let style = match style.trim() {
        "bare" => "bare",
        "labeled" => "labeled",
        _ => return Err("style must be 'bare' or 'labeled'".to_string()),
    };
    let root = canonical_sa3_dataset_root(&dataset_path)?;

    let current = read_reconciled_sa3_autolabel_state();
    if matches!(current.status.as_str(), "starting" | "running") {
        return Err("An auto-label job is already running.".to_string());
    }

    let availability = build_sa3_autolabel_availability(repo_root.inner());
    let python_exe = carey_autolabel_python(repo_root.inner());
    if !availability.carey_built {
        return Err("Carey must be built to auto-label — it runs the caption model.".to_string());
    }
    if !availability.captioner_downloaded {
        return Err(
            "Download the ACE-Step 5Hz LM 1.7B captioner from Carey → Models before auto-labeling."
                .to_string(),
        );
    }
    let script_path = carey_autolabel_script(repo_root.inner());
    if !script_path.exists() {
        return Err(format!("Missing {}", script_path.display()));
    }

    let dir = sa3_autolabel_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("Cannot create {}: {}", dir.display(), e))?;
    let status_path = sa3_autolabel_status_path();
    let cancel_path = sa3_autolabel_cancel_path();
    let log_path = sa3_autolabel_log_path();
    let _ = std::fs::remove_file(&cancel_path);

    write_sa3_autolabel_launch_state(&status_path, style)?;

    // Auto-stop the managed Carey inference service so it releases CUDA; the job's own
    // ensure_carey_stopped waits for it to go down before starting the temp LM server.
    {
        let mut mgr = manager.lock().await;
        mgr.stop("carey").ok();
    }

    let log_file = std::fs::File::create(&log_path)
        .map_err(|e| format!("Cannot create {}: {}", log_path.display(), e))?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|e| format!("Cannot clone log handle: {}", e))?;

    let mut cmd = tokio::process::Command::new(&python_exe);
    hide_console_window(&mut cmd);
    cmd.arg("-u")
        .arg(&script_path)
        .arg("--dataset-dir")
        .arg(&root)
        .arg("--style")
        .arg(style)
        .arg("--caption-lm-model")
        .arg("acestep-5Hz-lm-1.7B")
        .arg("--run-dir")
        .arg(&dir)
        .arg("--log-path")
        .arg(&log_path)
        .arg("--status-path")
        .arg(&status_path)
        .arg("--cancel-path")
        .arg(&cancel_path)
        .arg("--current-job-path")
        .arg(sa3_autolabel_current_job_path())
        .current_dir(repo_root.join("services").join("carey"))
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file_err))
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1");
    if let Some(token) = read_hf_token() {
        cmd.env("HF_TOKEN", token);
    }
    workload_job::configure_tokio_command(&mut cmd);

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => {
            let message = format!("Failed to launch auto-label: {}", error);
            let _ = mark_sa3_autolabel_failed(&message, None);
            return Err(message);
        }
    };
    if let Err(error) = workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        let message = format!(
            "Failed to enroll auto-label in the managed workload group: {}",
            error
        );
        let _ = mark_sa3_autolabel_failed(&message, None);
        return Err(message);
    }

    if let Some(pid) = child.id() {
        if let Err(error) = write_sa3_autolabel_process_id(&status_path, pid) {
            let _ = child.kill().await;
            let message = format!("Failed to record auto-label process {}: {}", pid, error);
            let _ = mark_sa3_autolabel_failed(&message, None);
            return Err(message);
        }

        tauri::async_runtime::spawn(async move {
            let exit_result = child.wait().await;
            let message = match exit_result {
                Ok(status) => format!(
                    "Auto-label process {} exited without recording a terminal state ({}). Check the auto-label log for details.",
                    pid, status
                ),
                Err(error) => format!(
                    "Could not monitor auto-label process {}: {}. Check the auto-label log for details.",
                    pid, error
                ),
            };
            let _ = mark_sa3_autolabel_failed(&message, Some(pid));
        });
    }

    Ok(read_sa3_autolabel_state())
}

#[tauri::command]
fn cancel_sa3_autolabel() -> Result<Sa3AutolabelState, String> {
    let state = read_reconciled_sa3_autolabel_state();
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(state);
    }
    write_cancel_marker(&sa3_autolabel_cancel_path())?;
    if let Some(pid) = state.pid {
        let _ = terminate_process_tree(pid);
    }
    mark_sa3_autolabel_cancelled();
    Ok(read_sa3_autolabel_state())
}

#[cfg(test)]
mod sa3_autolabel_availability_tests {
    use super::build_sa3_autolabel_availability;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn requires_built_carey_and_complete_1_7b_captioner() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gary-sa3-autolabel-availability-{}-{}",
            std::process::id(),
            stamp
        ));
        let carey = root.join("services").join("carey");
        let python = carey.join("env").join("Scripts").join("python.exe");
        let script = carey.join("sa3_autolabel.py");
        std::fs::create_dir_all(python.parent().expect("python should have a parent"))
            .expect("create Carey env path");
        std::fs::write(&python, []).expect("create fake Python");
        std::fs::write(&script, []).expect("create fake helper");

        let missing = build_sa3_autolabel_availability(&root);
        assert!(missing.carey_built);
        assert!(!missing.captioner_downloaded);
        assert!(!missing.available);

        let captioner = carey.join("checkpoints").join("acestep-5Hz-lm-1.7B");
        std::fs::create_dir_all(&captioner).expect("create captioner path");
        std::fs::write(captioner.join("config.json"), []).expect("create config");
        assert!(!build_sa3_autolabel_availability(&root).available);

        std::fs::write(captioner.join("model.safetensors"), []).expect("create weights");
        let ready = build_sa3_autolabel_availability(&root);
        assert!(ready.carey_built);
        assert!(ready.captioner_downloaded);
        assert!(ready.available);

        let _ = std::fs::remove_dir_all(root);
    }
}

#[tauri::command]
async fn open_sa3_training_reference(reference: String) -> Result<(), String> {
    let url = match reference.as_str() {
        "underfit" => "https://github.com/dada-bots/underfit#2-optional-add-metadata-for-prompts",
        "prompting" => {
            "https://github.com/Stability-AI/stable-audio-3/blob/main/docs/guides/prompting.md"
        }
        _ => return Err("Unknown SA3 training reference".to_string()),
    };

    let mut command = tokio::process::Command::new("cmd");
    command.args(["/C", "start", "", url]);
    hide_console_window(&mut command);
    let status = command
        .status()
        .await
        .map_err(|e| format!("Failed to open training reference: {}", e))?;
    if !status.success() {
        return Err(format!(
            "Windows could not open the training reference (exit code {:?})",
            status.code()
        ));
    }
    Ok(())
}

#[tauri::command]
fn get_sa3_lora_training_state() -> Result<Sa3LoraTrainingState, String> {
    Ok(read_sa3_lora_training_state())
}

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn start_sa3_lora_training(
    name: String,
    dataset_path: String,
    fixed_prompt: String,
    max_steps: u32,
    rank: u32,
    batch_size: u32,
    checkpoint_every: u32,
    latent_crop_seconds: f64,
    learning_rate: f64,
    loudness_fix_enabled: bool,
    target_latent_rms: f64,
    exclude_seconds_total: bool,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<Sa3LoraTrainingState, String> {
    let normalized_name = sanitize_lora_name(&name)
        .ok_or_else(|| "LoRA name must use lowercase letters, numbers, '-' or '_'".to_string())?;
    if max_steps == 0 {
        return Err("training steps must be greater than 0".to_string());
    }
    if rank == 0 {
        return Err("LoRA rank must be greater than 0".to_string());
    }
    if batch_size == 0 {
        return Err("batch size must be greater than 0".to_string());
    }
    if checkpoint_every == 0 {
        return Err("checkpoint interval must be greater than 0".to_string());
    }
    if !latent_crop_seconds.is_finite() || latent_crop_seconds <= 0.0 {
        return Err("latent crop seconds must be greater than 0".to_string());
    }
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err("learning rate must be greater than 0".to_string());
    }
    if loudness_fix_enabled
        && (!target_latent_rms.is_finite() || !(0.5..=1.3).contains(&target_latent_rms))
    {
        return Err("target latent RMS must be between 0.5 and 1.3".to_string());
    }
    if read_hf_token().is_none() {
        return Err("Save a Hugging Face token before training SA3 LoRAs.".to_string());
    }

    let current_state = read_sa3_lora_training_state();
    if matches!(current_state.status.as_str(), "starting" | "running") {
        return Err(format!(
            "SA3 LoRA training job '{}' is already {}.",
            current_state
                .name
                .as_deref()
                .unwrap_or(current_state.job_id.as_deref().unwrap_or("current")),
            current_state.phase
        ));
    }

    let dataset_dir = PathBuf::from(dataset_path.trim());
    if !dataset_dir.is_dir() {
        return Err(format!(
            "{} is not a valid dataset folder",
            dataset_dir.display()
        ));
    }

    let python_exe = repo_root
        .join("services")
        .join("sa3")
        .join("env")
        .join("Scripts")
        .join("python.exe");
    if !python_exe.exists() {
        return Err("SA3 must be built before LoRA training can start.".to_string());
    }

    let script_path = repo_root
        .join("services")
        .join("sa3")
        .join("train_lora_job.py");
    if !script_path.exists() {
        return Err(format!("Missing {}", script_path.display()));
    }

    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let job_id = format!("{}-{}", normalized_name, epoch);
    let training_root = sa3_training_dir();
    let run_dir = sa3_training_jobs_dir().join(&job_id);
    let log_path = sa3_training_logs_dir().join(format!("{}.log", job_id));
    let status_path = run_dir.join("status.json");
    let cancel_path = run_dir.join("cancel.requested");

    std::fs::create_dir_all(&run_dir)
        .map_err(|e| format!("Cannot create {}: {}", run_dir.display(), e))?;
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {}", parent.display(), e))?;
    }
    std::fs::create_dir_all(sa3_lora_dir())
        .map_err(|e| format!("Cannot create {}: {}", sa3_lora_dir().display(), e))?;
    std::fs::create_dir_all(sa3_prompts_dir())
        .map_err(|e| format!("Cannot create {}: {}", sa3_prompts_dir().display(), e))?;

    write_sa3_training_launch_state(
        &status_path,
        &job_id,
        &normalized_name,
        &run_dir,
        &log_path,
        &cancel_path,
        max_steps,
    )?;

    let log_file = std::fs::File::create(&log_path)
        .map_err(|e| format!("Cannot create {}: {}", log_path.display(), e))?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|e| format!("Cannot clone log handle: {}", e))?;

    let mut cmd = tokio::process::Command::new(&python_exe);
    hide_console_window(&mut cmd);
    cmd.arg("-u")
        .arg(&script_path)
        .arg("--job-id")
        .arg(&job_id)
        .arg("--name")
        .arg(&normalized_name)
        .arg("--dataset-dir")
        .arg(&dataset_dir)
        .arg("--fixed-prompt")
        .arg(fixed_prompt.trim())
        .arg("--training-root")
        .arg(&training_root)
        .arg("--models-dir")
        .arg(sa3_training_models_dir())
        .arg("--run-dir")
        .arg(&run_dir)
        .arg("--lora-dir")
        .arg(sa3_lora_dir())
        .arg("--catalog-path")
        .arg(sa3_lora_catalog_path())
        .arg("--prompts-dir")
        .arg(sa3_prompts_dir())
        .arg("--status-path")
        .arg(&status_path)
        .arg("--current-job-path")
        .arg(sa3_training_current_job_path())
        .arg("--cancel-path")
        .arg(&cancel_path)
        .arg("--log-path")
        .arg(&log_path)
        .arg("--max-steps")
        .arg(max_steps.to_string())
        .arg("--rank")
        .arg(rank.to_string())
        .arg("--alpha")
        .arg("0")
        .arg("--adapter-type")
        .arg("dora")
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--checkpoint-every")
        .arg(checkpoint_every.to_string())
        .arg("--latent-crop-seconds")
        .arg(latent_crop_seconds.to_string())
        .arg("--learning-rate")
        .arg(learning_rate.to_string());
    if loudness_fix_enabled {
        cmd.arg("--per-track-target-latent-rms")
            .arg(target_latent_rms.to_string());
    }
    // Optional: skip the seconds_total conditioner (228 modules instead of 229).
    // Official SA3 docs recommend this on small datasets to prevent "conditioner
    // hijacking", and the sa3.cpp trainer excludes it by default.
    if exclude_seconds_total {
        cmd.arg("--lora-exclude").arg("seconds_total");
    }
    cmd.current_dir(repo_root.join("services").join("sa3"))
        .stdout(std::process::Stdio::from(log_file))
        .stderr(std::process::Stdio::from(log_file_err))
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1");
    if let Some(token) = read_hf_token() {
        cmd.env("HF_TOKEN", token);
    }
    workload_job::configure_tokio_command(&mut cmd);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to launch SA3 LoRA training: {}", e))?;
    if let Err(error) = workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        let message = format!(
            "Failed to enroll SA3 LoRA training in the managed workload group: {}",
            error
        );
        let _ = mark_sa3_lora_training_failed(&status_path, &message, None);
        return Err(message);
    }
    let Some(pid) = child.id() else {
        let _ = child.kill().await;
        let message = "SA3 LoRA training launched without a process ID.".to_string();
        let _ = mark_sa3_lora_training_failed(&status_path, &message, None);
        return Err(message);
    };
    if let Err(error) = write_sa3_training_process_id(&status_path, pid) {
        let _ = child.kill().await;
        let message = format!(
            "Failed to record SA3 LoRA training process {}: {}",
            pid, error
        );
        let _ = mark_sa3_lora_training_failed(&status_path, &message, None);
        return Err(message);
    }

    let watched_status_path = status_path.clone();
    tauri::async_runtime::spawn(async move {
        let exit_result = child.wait().await;
        let state = read_sa3_training_status_file(&watched_status_path);
        // The trainer registered the finished LoRA in the catalog; rebuild the
        // registry now so gary4juce picks it up immediately instead of only after
        // someone opens the "add LoRAs" modal.
        if state.status == "completed" {
            if let Err(error) = refresh_sa3_lora_registry() {
                log::warn!("Could not refresh the SA3 LoRA registry after training: {error}");
            }
        }
        let warnings = if state.launcher_pid == Some(pid)
            || (state.launcher_pid.is_none() && state.pid == Some(pid))
        {
            terminate_discovered_sa3_training_processes(&state)
        } else {
            Vec::new()
        };
        let cleanup_suffix = if warnings.is_empty() {
            String::new()
        } else {
            format!(" Process cleanup warning: {}", warnings.join("; "))
        };
        let message = match exit_result {
            Ok(status) => format!(
                "SA3 LoRA training launcher {} exited without recording a terminal state ({}). Check the training log for details.{}",
                pid, status, cleanup_suffix
            ),
            Err(error) => format!(
                "Could not monitor SA3 LoRA training launcher {}: {}. Check the training log for details.{}",
                pid, error, cleanup_suffix
            ),
        };
        let _ = mark_sa3_lora_training_failed(&watched_status_path, &message, Some(pid));
    });

    Ok(read_sa3_lora_training_state())
}

#[tauri::command]
fn cancel_sa3_lora_training() -> Result<Sa3LoraTrainingState, String> {
    let state = read_sa3_lora_training_state();
    if !matches!(state.status.as_str(), "starting" | "running") {
        return Ok(state);
    }

    let cancel_path = state.cancel_path.as_deref().map(PathBuf::from).or_else(|| {
        state
            .run_dir
            .as_deref()
            .map(|run_dir| PathBuf::from(run_dir).join("cancel.requested"))
    });
    if let Some(path) = cancel_path.as_ref() {
        write_cancel_marker(path)?;
    }

    let mut pids = Vec::new();
    if let Some(pid) = state.child_pid {
        pids.push(pid);
    }
    if let Some(pid) = state.pid {
        if !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    if let Some(pid) = state.launcher_pid {
        if !pids.contains(&pid) {
            pids.push(pid);
        }
    }
    for pid in discover_sa3_training_pids(&state) {
        if !pids.contains(&pid) {
            pids.push(pid);
        }
    }

    let mut warnings = Vec::new();
    for pid in pids {
        if let Err(error) = terminate_process_tree(pid) {
            warnings.push(error);
        }
    }

    let message = if warnings.is_empty() {
        "Training cancelled.".to_string()
    } else {
        format!(
            "Training cancelled. Process cleanup warning: {}",
            warnings.join("; ")
        )
    };
    if let Some(status_path) = sa3_current_training_status_path() {
        write_sa3_training_cancelled_state(&status_path, cancel_path.as_deref(), &message)?;
    }

    Ok(read_sa3_lora_training_state())
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
fn get_runtime_storage_info(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<storage::RuntimeStorageInfo, String> {
    Ok(storage::storage_info(repo_root.inner()))
}

#[tauri::command]
fn save_runtime_storage_root(
    path: String,
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<storage::RuntimeStorageInfo, String> {
    storage::save_runtime_root_config(&path, repo_root.inner())
}

#[tauri::command]
fn reset_runtime_storage_root(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<storage::RuntimeStorageInfo, String> {
    storage::reset_runtime_root_config(repo_root.inner())
}

#[tauri::command]
fn get_legacy_storage_maintenance_info(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<LegacyStorageMaintenanceInfo, String> {
    build_legacy_storage_maintenance_info(repo_root.inner())
}

#[tauri::command]
fn migrate_legacy_loras(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<LegacyStorageMaintenanceResult, String> {
    Ok(migrate_legacy_loras_impl(repo_root.inner()))
}

#[tauri::command]
fn cleanup_legacy_storage(
    repo_root: tauri::State<'_, std::path::PathBuf>,
) -> Result<LegacyStorageMaintenanceResult, String> {
    Ok(cleanup_legacy_storage_impl(repo_root.inner()))
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

#[tauri::command]
fn reveal_path(path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("Path is empty".to_string());
    }

    let target = PathBuf::from(trimmed);
    if !target.exists() {
        return Err(format!("Path does not exist: {}", target.display()));
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;

        let mut command = std::process::Command::new("explorer.exe");
        command.creation_flags(CREATE_NO_WINDOW);
        if target.is_file() {
            command.arg("/select,").arg(&target);
        } else {
            command.arg(&target);
        }
        command
            .spawn()
            .map_err(|error| format!("Failed to reveal path: {}", error))?;
    }

    #[cfg(target_os = "macos")]
    {
        let mut command = std::process::Command::new("open");
        if target.is_file() {
            command.arg("-R");
        }
        command
            .arg(&target)
            .spawn()
            .map_err(|error| format!("Failed to reveal path: {}", error))?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let open_target = if target.is_file() {
            target.parent().unwrap_or(Path::new("."))
        } else {
            target.as_path()
        };
        std::process::Command::new("xdg-open")
            .arg(open_target)
            .spawn()
            .map_err(|error| format!("Failed to reveal path: {}", error))?;
    }

    Ok(())
}
