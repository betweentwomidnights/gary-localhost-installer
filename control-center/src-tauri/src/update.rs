use semver::Version;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri_plugin_updater::UpdaterExt;

pub const DEFAULT_UPDATE_MANIFEST_URL: &str =
    "https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json";

pub fn app_updater_enabled() -> bool {
    option_env!("VITE_ENABLE_APP_UPDATER").unwrap_or("1") != "0"
}

#[derive(Default)]
pub struct PendingNativeUpdate(pub std::sync::Mutex<Option<tauri_plugin_updater::Update>>);

struct NativeUpdaterConfig {
    endpoint: String,
    pubkey: String,
}

fn default_channel() -> String {
    "stable".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpdateManifest {
    #[serde(default = "default_channel")]
    pub channel: String,
    pub latest_version: String,
    #[serde(default)]
    pub download_url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub published_at: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppUpdateCheck {
    pub current_version: String,
    pub manifest_url: String,
    pub checked_at_epoch_ms: u64,
    pub channel: String,
    pub latest_version: String,
    pub update_available: bool,
    pub should_prompt: bool,
    pub in_app_install_available: bool,
    pub download_url: Option<String>,
    pub sha256: Option<String>,
    pub published_at: Option<String>,
    pub notes: Vec<String>,
}

fn native_updater_config() -> Option<NativeUpdaterConfig> {
    let endpoint = std::env::var("GARY4LOCAL_NATIVE_UPDATER_ENDPOINT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())?;
    let pubkey = std::env::var("GARY4LOCAL_NATIVE_UPDATER_PUBKEY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())?;

    Some(NativeUpdaterConfig { endpoint, pubkey })
}

fn manifest_url() -> String {
    std::env::var("GARY4LOCAL_UPDATE_MANIFEST_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            option_env!("GARY4LOCAL_UPDATE_MANIFEST_URL")
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| DEFAULT_UPDATE_MANIFEST_URL.to_string())
}

fn parse_version(value: &str) -> Result<Version, String> {
    let normalized = value.trim().trim_start_matches('v');
    Version::parse(normalized)
        .map_err(|error| format!("Invalid version '{}': {}", value, error))
}

fn versions_match(a: &str, b: &str) -> bool {
    match (parse_version(a), parse_version(b)) {
        (Ok(left), Ok(right)) => left == right,
        _ => a.trim().trim_start_matches('v') == b.trim().trim_start_matches('v'),
    }
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn clear_pending_native_update(pending_update: &PendingNativeUpdate) {
    *pending_update.0.lock().unwrap() = None;
}

pub async fn check_native_update<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    pending_update: &PendingNativeUpdate,
    target_version: &str,
) -> Result<bool, String> {
    let Some(config) = native_updater_config() else {
        clear_pending_native_update(pending_update);
        return Ok(false);
    };

    let endpoint = reqwest::Url::parse(&config.endpoint)
        .map_err(|error| format!("Invalid native updater endpoint: {}", error))?;

    let update = app
        .updater_builder()
        .endpoints(vec![endpoint])
        .map_err(|error| format!("Invalid native updater endpoint config: {}", error))?
        .pubkey(config.pubkey)
        .timeout(Duration::from_secs(6))
        .build()
        .map_err(|error| format!("Failed to build native updater: {}", error))?
        .check()
        .await
        .map_err(|error| format!("Native updater check failed: {}", error))?;

    let matches_target = update
        .as_ref()
        .map(|item| versions_match(&item.version, target_version))
        .unwrap_or(false);

    if matches_target {
        *pending_update.0.lock().unwrap() = update;
        Ok(true)
    } else {
        clear_pending_native_update(pending_update);
        Ok(false)
    }
}

pub async fn install_native_update<R: tauri::Runtime>(
    _app: &tauri::AppHandle<R>,
    pending_update: &PendingNativeUpdate,
) -> Result<(), String> {
    let update = pending_update
        .0
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "No pending in-app update is available".to_string())?;

    update
        .download_and_install(|_, _| {}, || {})
        .await
        .map_err(|error| format!("Failed to install update: {}", error))?;

    #[cfg(not(target_os = "windows"))]
    _app.restart();

    Ok(())
}

pub async fn check_for_update(
    current_version: &str,
    skipped_version: Option<&str>,
    include_skipped: bool,
) -> Result<AppUpdateCheck, String> {
    let manifest_url = manifest_url();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|error| format!("Failed to build update client: {}", error))?;

    let response = client
        .get(&manifest_url)
        .header(reqwest::header::ACCEPT, "application/json")
        .header(reqwest::header::USER_AGENT, format!("gary4local/{}", current_version))
        .send()
        .await
        .map_err(|error| format!("Update check failed: {}", error))?;

    if !response.status().is_success() {
        return Err(format!(
            "Update manifest returned HTTP {}",
            response.status()
        ));
    }

    let manifest: UpdateManifest = response
        .json()
        .await
        .map_err(|error| format!("Invalid update manifest JSON: {}", error))?;

    let current = parse_version(current_version)?;
    let latest = parse_version(&manifest.latest_version)?;
    let update_available = latest > current;

    if update_available
        && manifest
            .download_url
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
    {
        return Err("Update manifest is missing download_url for a newer release".to_string());
    }

    let should_prompt = update_available
        && (include_skipped
            || skipped_version
                .map(str::trim)
                .filter(|value| !value.is_empty())
                != Some(manifest.latest_version.trim()));

    Ok(AppUpdateCheck {
        current_version: current_version.to_string(),
        manifest_url,
        checked_at_epoch_ms: now_epoch_ms(),
        channel: manifest.channel,
        latest_version: manifest.latest_version,
        update_available,
        should_prompt,
        in_app_install_available: false,
        download_url: manifest.download_url,
        sha256: manifest.sha256,
        published_at: manifest.published_at,
        notes: manifest.notes,
    })
}
