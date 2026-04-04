use semver::Version;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const DEFAULT_UPDATE_MANIFEST_URL: &str =
    "https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json";

pub fn app_updater_enabled() -> bool {
    option_env!("VITE_ENABLE_APP_UPDATER").unwrap_or("1") != "0"
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
    pub download_url: Option<String>,
    pub sha256: Option<String>,
    pub published_at: Option<String>,
    pub notes: Vec<String>,
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

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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
        download_url: manifest.download_url,
        sha256: manifest.sha256,
        published_at: manifest.published_at,
        notes: manifest.notes,
    })
}
