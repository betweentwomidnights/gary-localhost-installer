//! Prepared native adapters live beside, never over, legacy checkpoints.
//! Content-addressed files let admitted generations finish using an older
//! adapter while catalog selection moves to a newly converted checkpoint.

use crate::native_runtime::sha256_file;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

// Vendored from sa3.cpp/tools/lora_ckpt_export.py. Keep the two files identical;
// the script's content hash is recorded with each one-time legacy export.
const LEGACY_EXPORTER: &str = include_str!("sa3_helpers/lora_ckpt_export.py");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegacyExport {
    pub source_sha256: String,
    pub exporter_sha256: String,
    pub safetensors_path: String,
    pub safetensors_sha256: String,
    pub config_path: String,
    pub config_sha256: String,
}

static PREPARATION: Mutex<()> = Mutex::const_new(());
static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeLora {
    pub name: String,
    pub source_path: String,
    pub config_path: Option<String>,
    pub source_sha256: Option<String>,
    pub config_sha256: Option<String>,
    pub converter_sha256: Option<String>,
    pub native_path: Option<String>,
    pub native_sha256: Option<String>,
    pub strength: f64,
    pub error: Option<String>,
    #[serde(default)]
    pub prompts_path: Option<String>,
    #[serde(default)]
    pub training_checkpoints: Vec<crate::sa3_training::Checkpoint>,
    #[serde(default)]
    pub legacy_export: Option<LegacyExport>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeLoraState {
    pub entries: Vec<NativeLora>,
    pub preparing: bool,
    pub catalog_path: String,
    pub adapters_dir: String,
}

pub fn adapters_dir(root: &Path) -> PathBuf {
    root.join("sa3/native-loras")
}

pub fn catalog_path(root: &Path) -> PathBuf {
    adapters_dir(root).join("catalog.json")
}

pub fn read_catalog(root: &Path) -> Result<BTreeMap<String, NativeLora>, String> {
    let path = catalog_path(root);
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let dir = adapters_dir(root)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    if !dir.starts_with(&boundary)
        || path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err(
            "Native SA3 adapter catalog points outside the selected runtime folder.".into(),
        );
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("Cannot read native SA3 LoRA catalog: {error}"))?;
    let entries: BTreeMap<String, NativeLora> = serde_json::from_str(&text)
        .map_err(|error| format!("Invalid native SA3 LoRA catalog: {error}"))?;
    for (name, entry) in &entries {
        if crate::sanitize_lora_name(name).as_deref() != Some(name.as_str())
            || entry.name != *name
            || !entry.strength.is_finite()
        {
            return Err("Native SA3 LoRA catalog contains an invalid name or strength.".into());
        }
        if let Some(path) = &entry.native_path {
            checked_native_path(root, Path::new(path))?;
        }
    }
    Ok(entries)
}

/// Catalog paths are local managed files, never arbitrary paths supplied to the
/// native server. Reject external paths even when the referenced file is absent.
pub(crate) fn checked_native_path(root: &Path, path: &Path) -> Result<(), String> {
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let dir = adapters_dir(root)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let parent = path
        .parent()
        .ok_or("Invalid prepared adapter path")?
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !dir.starts_with(&boundary)
        || parent != dir
        || path.extension().and_then(|ext| ext.to_str()) != Some("gguf")
        || (path.exists()
            && path
                .canonicalize()
                .map_err(|error| error.to_string())?
                .parent()
                != Some(dir.as_path()))
    {
        return Err("Prepared SA3 adapter points outside native adapter storage.".into());
    }
    Ok(())
}

struct ConversionStage {
    path: PathBuf,
    boundary: PathBuf,
}

impl Drop for ConversionStage {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = crate::remove_managed_path(&self.path, &self.boundary);
        }
    }
}

fn validate_gguf(path: &Path) -> Result<(), String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut header = [0u8; 24];
    file.read_exact(&mut header)
        .map_err(|_| "Converter did not produce a complete GGUF header")?;
    if &header[..4] != b"GGUF"
        || !matches!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 2 | 3)
        || u64::from_le_bytes(header[8..16].try_into().unwrap()) == 0
    {
        return Err("Converter did not produce a supported GGUF adapter.".into());
    }
    Ok(())
}

pub async fn register_trained(root: &Path, name: &str, source: &Path) -> Result<PathBuf, String> {
    let _reservation = PREPARATION
        .try_lock()
        .map_err(|_| "Native adapter preparation is already running")?;
    if crate::sanitize_lora_name(name).as_deref() != Some(name) {
        return Err("Invalid native adapter name".into());
    }
    validate_gguf(source)?;
    let hash = sha256_file(source).await?;
    let dir = checked_adapters_dir(root)?;
    let dest = dir.join(format!("lora-{name}-{}-f32.gguf", &hash[..24]));
    if !dest.exists() || sha256_file(&dest).await? != hash {
        let stage = dir.join(format!(
            ".trained-{}-{}.partial",
            std::process::id(),
            NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
        ));
        let _cleanup = ConversionStage {
            path: stage.clone(),
            boundary: dir.clone(),
        };
        tokio::fs::copy(source, &stage)
            .await
            .map_err(|error| error.to_string())?;
        if sha256_file(&stage).await? != hash {
            return Err("Trained adapter changed during registration".into());
        }
        if dest.exists() {
            crate::remove_managed_path(&dest, &dir)?;
        }
        std::fs::rename(stage, &dest).map_err(|error| error.to_string())?;
    }
    let mut entries = read_catalog(root)?;
    let previous = entries.get(name);
    let strength = previous.map_or(1.0, |entry| entry.strength);
    let mut prompts_path = previous.and_then(|entry| entry.prompts_path.clone());
    if let Some(dataset) = crate::sa3_training::dataset_for_checkpoint(root, name, source)? {
        prompts_path = Some(dataset.to_string_lossy().into());
        crate::sa3_prompts::build(root, name, &dataset, false)?;
    }
    entries.insert(
        name.into(),
        NativeLora {
            name: name.into(),
            source_path: source.to_string_lossy().into(),
            config_path: None,
            source_sha256: Some(hash.clone()),
            config_sha256: None,
            converter_sha256: None,
            native_path: Some(dest.to_string_lossy().into()),
            native_sha256: Some(hash),
            strength,
            error: None,
            prompts_path,
            training_checkpoints: crate::sa3_training::checkpoints(root, name)?,
            legacy_export: None,
        },
    );
    save_catalog(root, &entries)?;
    Ok(dest)
}

fn checked_adapters_dir(root: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(root).map_err(|error| error.to_string())?;
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let mut dir = boundary.clone();
    for part in ["sa3", "native-loras"] {
        dir.push(part);
        std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        dir = dir.canonicalize().map_err(|error| error.to_string())?;
        if !dir.starts_with(&boundary) {
            return Err(
                "Native SA3 adapter storage points outside the selected runtime folder.".into(),
            );
        }
    }
    Ok(dir)
}

fn save_catalog(root: &Path, entries: &BTreeMap<String, NativeLora>) -> Result<(), String> {
    let dir = checked_adapters_dir(root)?;
    let stage = dir.join(format!(
        ".catalog-{}-{}.json",
        std::process::id(),
        NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage)
        .map_err(|error| format!("Cannot reserve native adapter staging file: {error}"))?;
    let _stage_cleanup = ConversionStage {
        path: stage.clone(),
        boundary: dir.clone(),
    };
    let text = serde_json::to_vec_pretty(entries).map_err(|error| error.to_string())?;
    std::fs::write(&stage, text).map_err(|error| error.to_string())?;
    std::fs::rename(&stage, dir.join("catalog.json"))
        .map_err(|error| format!("Cannot publish native SA3 LoRA catalog: {error}"))
}

pub fn state(root: &Path) -> Result<NativeLoraState, String> {
    let mut entries = read_catalog(root)?;
    let legacy = crate::read_sa3_lora_catalog_from(&root.join("sa3/lora_catalog.json"))?;
    if root.join("sa3/lora_catalog.json").exists() {
        entries.retain(|name, entry| {
            legacy.contains_key(name)
                || Path::new(&entry.source_path)
                    .extension()
                    .is_some_and(|ext| ext == "gguf")
        });
    }
    for (name, source) in legacy {
        if crate::sanitize_lora_name(&name).as_deref() != Some(name.as_str()) {
            return Err(format!(
                "Legacy LoRA '{name}' needs a valid native registry name before preparation."
            ));
        }
        let entry = entries.entry(name.clone()).or_insert_with(|| NativeLora {
            name,
            source_path: source.path.clone(),
            config_path: None,
            source_sha256: None,
            config_sha256: None,
            converter_sha256: None,
            native_path: None,
            native_sha256: None,
            strength: source.strength,
            error: None,
            prompts_path: source.prompts_path.clone(),
            training_checkpoints: Vec::new(),
            legacy_export: None,
        });
        // The Python checkpoint selection may have changed since preparation.
        // Preserve its converted revision in storage but show the pending source.
        if entry.source_path != source.path {
            entry.source_path = source.path;
            entry.native_path = None;
            entry.native_sha256 = None;
            entry.source_sha256 = None;
            entry.error = None;
        }
        entry.strength = source.strength;
        entry.prompts_path = source.prompts_path;
    }
    for entry in entries.values_mut() {
        if Path::new(&entry.source_path)
            .extension()
            .is_some_and(|ext| ext == "gguf")
        {
            entry.training_checkpoints = crate::sa3_training::checkpoints(root, &entry.name)?;
        }
        if entry
            .native_path
            .as_ref()
            .is_some_and(|path| !Path::new(path).is_file())
        {
            entry.native_path = None;
            entry.error = Some("Prepared adapter is missing; prepare it again.".into());
        }
    }
    Ok(NativeLoraState {
        entries: entries.into_values().collect(),
        preparing: PREPARATION.try_lock().is_err(),
        catalog_path: catalog_path(root).to_string_lossy().into(),
        adapters_dir: adapters_dir(root).to_string_lossy().into(),
    })
}

pub(crate) fn embedded_lora_config(path: &Path) -> Result<Option<serde_json::Value>, String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| format!("Invalid safetensors header: {error}"))?;
    let length = u64::from_le_bytes(prefix);
    if !(2..=16 * 1024 * 1024).contains(&length) {
        return Err("Safetensors metadata header is invalid or too large.".into());
    }
    let mut header = vec![0; length as usize];
    file.read_exact(&mut header)
        .map_err(|error| format!("Truncated safetensors header: {error}"))?;
    let header: serde_json::Value =
        serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    header
        .pointer("/__metadata__/lora_config")
        .and_then(|value| value.as_str())
        .map(|config| {
            serde_json::from_str(config)
                .map_err(|error| format!("Invalid embedded LoRA configuration: {error}"))
        })
        .transpose()
}

fn embedded_config(path: &Path) -> Result<bool, String> {
    Ok(embedded_lora_config(path)?.is_some())
}

fn config_for_source(entry: &NativeLora) -> Result<Option<PathBuf>, String> {
    let path = Path::new(&entry.source_path);
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();
    if ext == "ckpt" {
        return Err("This legacy .ckpt needs a safetensors export before native migration. The original is preserved.".into());
    }
    if ext != "safetensors" {
        return Err("Native preparation currently accepts exported .safetensors adapters.".into());
    }
    if let Some(config) = &entry.config_path {
        let path = PathBuf::from(config);
        if !path.is_file() {
            return Err("The adapter's JSON configuration is missing.".into());
        }
        return Ok(Some(path));
    }
    if embedded_config(path)? {
        return Ok(None);
    }
    let config = path.with_extension("json");
    if config.is_file() {
        return Ok(Some(config));
    }
    Err("Adapter has no embedded LoRA configuration or matching JSON sidecar. Export its configuration before migration.".into())
}

/// Validate exported copies separately from the original checkpoint. A missing
/// or damaged copy can be rebuilt while Python still exists; a redirected path
/// is rejected rather than opened outside managed adapter storage.
async fn verified_export(root: &Path, export: &LegacyExport) -> Result<bool, String> {
    let dir = checked_adapters_dir(root)?.join("legacy-exports");
    let mut complete = true;
    for (path, hash) in [
        (&export.safetensors_path, &export.safetensors_sha256),
        (&export.config_path, &export.config_sha256),
    ] {
        let path = Path::new(path);
        if !path.starts_with(&dir)
            || path
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err("Legacy export points outside managed adapter storage.".into());
        }
        if !path.is_file() {
            complete = false;
            continue;
        }
        let boundary = dir.canonicalize().map_err(|error| error.to_string())?;
        if !boundary.starts_with(checked_adapters_dir(root)?)
            || !path
                .canonicalize()
                .map_err(|error| error.to_string())?
                .starts_with(&boundary)
        {
            return Err("Legacy export points outside managed adapter storage.".into());
        }
        if sha256_file(path).await? != *hash {
            complete = false;
        }
    }
    Ok(complete)
}

pub async fn verify_legacy_export(root: &Path, entry: &NativeLora) -> Result<(), String> {
    if Path::new(&entry.source_path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ckpt"))
        && entry.legacy_export.is_none()
    {
        return Err(format!(
            "LoRA '{}' has no verified legacy export; prepare it again",
            entry.name
        ));
    }
    if let Some(export) = &entry.legacy_export {
        if entry.source_sha256.as_deref() != Some(&export.source_sha256)
            || !verified_export(root, export).await?
        {
            return Err(format!(
                "LoRA '{}' legacy export changed; prepare it again",
                entry.name
            ));
        }
    }
    Ok(())
}

async fn cached_legacy_export(
    root: &Path,
    previous: Option<&LegacyExport>,
    source_hash: &str,
) -> Result<Option<LegacyExport>, String> {
    // A verified previous export remains usable after environment cleanup and
    // across app updates. New exports record the current helper's exact hash.
    if let Some(export) = previous {
        if export.source_sha256 == source_hash && verified_export(root, export).await? {
            return Ok(Some(export.clone()));
        }
    }
    // Check older selected checkpoints too. Switching away from an adapter
    // must not make its previously exported revision depend on Python again.
    let adapters = checked_adapters_dir(root)?;
    let exports = adapters.join("legacy-exports");
    if exports.is_dir() {
        let boundary = exports.canonicalize().map_err(|error| error.to_string())?;
        if !boundary.starts_with(&adapters) {
            return Err("Legacy export storage points outside managed adapter storage.".into());
        }
        let mut candidates = std::fs::read_dir(&exports)
            .map_err(|error| error.to_string())?
            .filter_map(Result::ok)
            .map(|entry| entry.path().join("export.json"))
            .collect::<Vec<_>>();
        candidates.sort();
        for path in candidates {
            let Ok(resolved) = path.canonicalize() else {
                continue;
            };
            if !resolved.starts_with(&boundary)
                || resolved.metadata().map_or(true, |info| info.len() > 65536)
            {
                continue;
            }
            let Ok(bytes) = std::fs::read(&resolved) else {
                continue;
            };
            let Ok(export) = serde_json::from_slice::<LegacyExport>(&bytes) else {
                continue;
            };
            if export.source_sha256 == source_hash && verified_export(root, &export).await? {
                return Ok(Some(export));
            }
        }
    }
    Ok(None)
}

/// Cleanup must retain the ability to select every older checkpoint, including
/// CKPT history whose currently selected adapter is already safetensors.
pub async fn verify_legacy_history(root: &Path) -> Result<(), String> {
    let history = crate::read_sa3_lora_catalog_from(&root.join("sa3/lora_catalog.json"))?;
    for (name, legacy) in history {
        for path in std::iter::once(legacy.path).chain(
            legacy
                .training_checkpoints
                .into_iter()
                .map(|checkpoint| checkpoint.path),
        ) {
            let source = Path::new(&path);
            if source
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("ckpt"))
            {
                let hash = sha256_file(source).await?;
                if cached_legacy_export(root, None, &hash).await?.is_none() {
                    return Err(format!("LoRA '{name}' has a legacy checkpoint without a verified export: {path}. Prepare adapters before migration."));
                }
            }
        }
    }
    Ok(())
}

/// Verify an automatically discovered sidecar too, not only explicitly selected
/// configuration files. CKPT exports carry their configuration in the cache.
pub async fn verify_source_config(entry: &NativeLora) -> Result<(), String> {
    let extension = Path::new(&entry.source_path)
        .extension()
        .unwrap_or_default();
    let config = if extension.eq_ignore_ascii_case("safetensors") {
        config_for_source(entry)?
    } else if extension.eq_ignore_ascii_case("ckpt") {
        entry.config_path.as_ref().map(PathBuf::from)
    } else {
        None
    };
    let hash = match config {
        Some(path) => Some(sha256_file(&path).await?),
        None => None,
    };
    if hash != entry.config_sha256 {
        return Err(format!(
            "LoRA '{}' configuration changed since preparation; prepare it again",
            entry.name
        ));
    }
    Ok(())
}

async fn export_legacy_checkpoint(
    root: &Path,
    entry: &mut NativeLora,
    source: &Path,
    source_hash: &str,
) -> Result<PathBuf, String> {
    if let Some(export) =
        cached_legacy_export(root, entry.legacy_export.as_ref(), source_hash).await?
    {
        let input = PathBuf::from(&export.safetensors_path);
        entry.legacy_export = Some(export);
        return Ok(input);
    }
    let adapters = checked_adapters_dir(root)?;
    let python = root.join("services/sa3/env/Scripts/python.exe");
    if !python.is_file() {
        return Err("This .ckpt needs the existing SA3 Python environment for a one-time export. The original is preserved. Prepare it before removing that environment, or import a safetensors export.".into());
    }
    let dir =
        crate::sa3_training::checked_folder(root, &["sa3", "native-loras", "legacy-exports"])?;
    let nonce = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
    let exporter_hash = format!("{:x}", Sha256::digest(LEGACY_EXPORTER.as_bytes()));
    let digest = format!(
        "{:x}",
        Sha256::digest(format!("{source_hash}:{exporter_hash}"))
    );
    let stage = dir.join(format!(".export-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&stage)
        .map_err(|error| format!("Cannot reserve legacy export: {error}"))?;
    let _cleanup = ConversionStage {
        path: stage.clone(),
        boundary: dir.clone(),
    };
    let script = stage.join("exporter.py");
    std::fs::write(&script, LEGACY_EXPORTER).map_err(|error| error.to_string())?;
    let mut cmd = tokio::process::Command::new(python);
    cmd.arg("-I")
        .arg(&script)
        .arg("--ckpt")
        .arg(source)
        .arg("--out")
        .arg(stage.join("adapter"))
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PYTHONIOENCODING", "utf-8")
        .current_dir(&stage)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    crate::workload_job::configure_tokio_command(&mut cmd);
    let mut child = cmd
        .spawn()
        .map_err(|error| format!("Cannot start legacy checkpoint export: {error}"))?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    let result = child
        .wait_with_output()
        .await
        .map_err(|error| error.to_string())?;
    let output = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    crate::sa3_training::save(
        &adapters.join(format!("legacy-export-{}.json", entry.name)),
        &serde_json::json!({"source":source,"success":result.status.success(),"output":output}),
    )?;
    if !result.status.success() {
        let detail = output
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "Legacy checkpoint export failed ({}): {detail}. The original is preserved.",
            result.status
        ));
    }
    let tensors = stage.join("adapter.safetensors");
    let config = stage.join("adapter.json");
    if !embedded_config(&tensors)? {
        return Err("Legacy export has no embedded adapter configuration.".into());
    }
    let config_value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if !config_value
        .as_object()
        .is_some_and(|config| !config.is_empty())
    {
        return Err("Legacy export has an invalid adapter configuration.".into());
    }
    if sha256_file(source).await? != source_hash {
        return Err("Original checkpoint changed during export; prepare it again.".into());
    }
    let dest = dir.join(format!(
        "ckpt-{}-{}-{nonce}",
        &digest[..24],
        std::process::id()
    ));
    let export = LegacyExport {
        source_sha256: source_hash.into(),
        exporter_sha256: exporter_hash,
        safetensors_path: dest.join("adapter.safetensors").to_string_lossy().into(),
        safetensors_sha256: sha256_file(&tensors).await?,
        config_path: dest.join("adapter.json").to_string_lossy().into(),
        config_sha256: sha256_file(&config).await?,
    };
    crate::sa3_training::save(&stage.join("export.json"), &export)?;
    std::fs::rename(&stage, &dest)
        .map_err(|error| format!("Cannot publish legacy export: {error}"))?;
    let path = PathBuf::from(&export.safetensors_path);
    entry.legacy_export = Some(export);
    Ok(path)
}

async fn convert_one(
    root: &Path,
    entry: &mut NativeLora,
    converter: &Path,
    converter_hash: &str,
    runtime_path: Option<&std::ffi::OsStr>,
) -> Result<(), String> {
    if Path::new(&entry.source_path)
        .extension()
        .is_some_and(|ext| ext == "gguf")
    {
        let source = Path::new(&entry.source_path);
        validate_gguf(source)?;
        let hash = sha256_file(source).await?;
        if entry.source_sha256.as_deref() != Some(&hash)
            || entry
                .native_path
                .as_ref()
                .is_none_or(|path| !Path::new(path).is_file())
            || sha256_file(Path::new(entry.native_path.as_ref().unwrap())).await? != hash
        {
            return Err(
                "Native trained adapter needs registration from its training job again.".into(),
            );
        }
        entry.error = None;
        return Ok(());
    }
    let source = Path::new(&entry.source_path)
        .canonicalize()
        .map_err(|error| format!("Cannot read original adapter: {error}"))?;
    let source_hash = sha256_file(&source).await?;
    let previous_export_hash = entry
        .legacy_export
        .as_ref()
        .map(|export| export.safetensors_sha256.clone());
    let (input, config) = if source
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ckpt"))
    {
        let input = export_legacy_checkpoint(root, entry, &source, &source_hash).await?;
        let mut exported = entry.clone();
        exported.source_path = input.to_string_lossy().into();
        (input, config_for_source(&exported)?)
    } else {
        entry.legacy_export = None;
        (source.clone(), config_for_source(entry)?)
    };
    let input_hash = sha256_file(&input).await?;
    let config_hash = match &config {
        Some(path) => Some(sha256_file(path).await?),
        None => None,
    };
    if entry.source_sha256.as_deref() == Some(&source_hash)
        && entry.config_sha256 == config_hash
        && entry.converter_sha256.as_deref() == Some(converter_hash)
        && (entry.legacy_export.is_none() || previous_export_hash.as_deref() == Some(&input_hash))
    {
        if let (Some(path), Some(hash)) = (&entry.native_path, &entry.native_sha256) {
            if Path::new(path).is_file() && sha256_file(Path::new(path)).await? == *hash {
                entry.error = None;
                return Ok(());
            }
        }
    }
    let dir = checked_adapters_dir(root)?;
    let revision = if entry.legacy_export.is_some() {
        format!(
            "v2:{source_hash}:{input_hash}:{}:{converter_hash}",
            config_hash.as_deref().unwrap_or("embedded")
        )
    } else {
        format!(
            "v1:{source_hash}:{}:{converter_hash}",
            config_hash.as_deref().unwrap_or("embedded")
        )
    };
    let digest = format!("{:x}", Sha256::digest(revision));
    let dest = dir.join(format!("lora-{}-{}-f32.gguf", entry.name, &digest[..24]));
    // Each conversion uses a unique unpublished file, never overwriting an
    // adapter a queued job may still open. Old revisions remain recoverable.
    let stage = dir.join(format!(
        ".convert-{}-{}.partial",
        std::process::id(),
        NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage)
        .map_err(|error| format!("Cannot reserve native adapter staging file: {error}"))?;
    let _stage_cleanup = ConversionStage {
        path: stage.clone(),
        boundary: dir.clone(),
    };
    let mut cmd = tokio::process::Command::new(converter);
    cmd.arg("--safetensors")
        .arg(&input)
        .arg("--out")
        .arg(&stage)
        .current_dir(converter.parent().ok_or("Invalid native converter path")?)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    if let Some(config) = &config {
        cmd.arg("--json").arg(config);
    }
    if let Some(path) = runtime_path {
        cmd.env("PATH", path);
    }
    crate::workload_job::configure_tokio_command(&mut cmd);
    let mut child = cmd
        .spawn()
        .map_err(|error| format!("Cannot start native LoRA converter: {error}"))?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    let result = child
        .wait_with_output()
        .await
        .map_err(|error| error.to_string())?;
    let mut output = String::from_utf8_lossy(&result.stdout).into_owned();
    output.push_str(&String::from_utf8_lossy(&result.stderr));
    let log_stage = stage.with_extension("log");
    let _log_cleanup = ConversionStage {
        path: log_stage.clone(),
        boundary: dir.clone(),
    };
    std::fs::write(&log_stage, &output)
        .map_err(|error| format!("Cannot save native conversion log: {error}"))?;
    std::fs::rename(
        &log_stage,
        dir.join(format!("conversion-{}.log", entry.name)),
    )
    .map_err(|error| format!("Cannot publish native conversion log: {error}"))?;
    if !result.status.success() {
        let _ = crate::remove_managed_path(&stage, &dir);
        let detail = output
            .lines()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "Native LoRA conversion failed ({}): {detail}",
            result.status
        ));
    }
    validate_gguf(&stage)?;
    let current_config_hash = match &config {
        Some(path) => Some(sha256_file(path).await?),
        None => None,
    };
    if sha256_file(&source).await? != source_hash
        || sha256_file(&input).await? != input_hash
        || current_config_hash != config_hash
    {
        let _ = crate::remove_managed_path(&stage, &dir);
        return Err(
            "Original adapter or configuration changed during conversion; prepare it again.".into(),
        );
    }
    if let Some(export) = &entry.legacy_export {
        if export.source_sha256 != source_hash || !verified_export(root, export).await? {
            return Err("Legacy export changed during native conversion; prepare it again.".into());
        }
    }
    let native_hash = sha256_file(&stage).await?;
    if let Some(old_hash) = entry
        .native_sha256
        .as_ref()
        .filter(|_| entry.native_path.as_deref() == Some(&dest.to_string_lossy()))
    {
        if dest.is_file() && sha256_file(&dest).await? == *old_hash && *old_hash != native_hash {
            let _ = crate::remove_managed_path(&stage, &dir);
            return Err("Converter produced a different revision for the same source; existing prepared adapter was preserved.".into());
        }
    }
    // If a corrupt cache file occupies this revision, it can be repaired only
    // while no native generation is running (the command's reservation checks).
    if dest.exists() {
        crate::remove_managed_path(&dest, &dir)?;
    }
    std::fs::rename(&stage, &dest)
        .map_err(|error| format!("Cannot publish converted adapter: {error}"))?;
    entry.source_sha256 = Some(source_hash);
    entry.config_sha256 = config_hash;
    entry.converter_sha256 = Some(converter_hash.into());
    entry.native_path = Some(dest.to_string_lossy().into());
    entry.native_sha256 = Some(native_hash);
    entry.error = None;
    Ok(())
}

/// Auxiliary autoencoder adapters use the same immutable conversion machinery,
/// but remain outside the creative LoRA registry.
pub async fn prepare_decoder(
    root: &Path,
    entry: &mut NativeLora,
    converter: &Path,
    runtime_path: Option<&std::ffi::OsStr>,
) -> Result<(), String> {
    let _reservation = PREPARATION
        .try_lock()
        .map_err(|_| "Native adapter preparation is already running")?;
    let source = Path::new(&entry.source_path)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let source_hash = sha256_file(&source).await?;
    let input = if source
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("ckpt"))
    {
        export_legacy_checkpoint(root, entry, &source, &source_hash).await?
    } else {
        source.clone()
    };
    let mut exported = entry.clone();
    exported.source_path = input.to_string_lossy().into();
    let config = config_for_source(&exported)?;
    let (config_value, config_hash) = if let Some(path) = config {
        (
            serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(&path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
            Some(sha256_file(&path).await?),
        )
    } else {
        (
            embedded_lora_config(&input)?.ok_or("Decoder adapter has no configuration")?,
            None,
        )
    };
    if config_value["target"] != "decoder" {
        return Err("The decoder correction adapter must declare target 'decoder'.".into());
    }
    convert_one(
        root,
        entry,
        converter,
        &sha256_file(converter).await?,
        runtime_path,
    )
    .await?;
    if entry.source_sha256.as_deref() != Some(&source_hash) || entry.config_sha256 != config_hash {
        return Err("Decoder source changed during preparation; prepare it again.".into());
    }
    Ok(())
}

/// Every entry gets a recorded result; one incomplete legacy export must not
/// hide successfully prepared adapters or silently permit destructive cleanup.
pub async fn prepare(
    root: &Path,
    converter: &Path,
    runtime_path: Option<&std::ffi::OsStr>,
    mut changed: impl FnMut(NativeLoraState) + Send,
) -> Result<NativeLoraState, String> {
    let _reservation = PREPARATION
        .try_lock()
        .map_err(|_| "SA3 LoRA preparation is already in progress.")?;
    let mut entries: BTreeMap<_, _> = state(root)?
        .entries
        .into_iter()
        .map(|entry| (entry.name.clone(), entry))
        .collect();
    let history = crate::read_sa3_lora_catalog_from(&root.join("sa3/lora_catalog.json"))?;
    let converter_hash = sha256_file(converter).await?;
    let names: Vec<_> = entries.keys().cloned().collect();
    for name in names {
        let entry = entries.get_mut(&name).unwrap();
        let result = async {
            if let Some(legacy) = history.get(&name) {
                for checkpoint in &legacy.training_checkpoints {
                    let path = Path::new(&checkpoint.path);
                    if path
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("ckpt"))
                    {
                        let path = path.canonicalize().map_err(|error| {
                            format!(
                                "Cannot read original checkpoint at step {}: {error}",
                                checkpoint.step
                            )
                        })?;
                        let hash = sha256_file(&path).await?;
                        let mut historical = entry.clone();
                        historical.config_path = None;
                        historical.legacy_export = None;
                        export_legacy_checkpoint(root, &mut historical, &path, &hash)
                            .await
                            .map_err(|error| {
                                format!(
                                    "Cannot preserve legacy checkpoint at step {}: {error}",
                                    checkpoint.step
                                )
                            })?;
                    }
                }
            }
            convert_one(root, entry, converter, &converter_hash, runtime_path).await
        }
        .await;
        if let Err(error) = result {
            // Keep an older good revision in storage, but do not advertise it
            // as the newly selected source's successfully prepared checkpoint.
            entry.native_path = None;
            entry.native_sha256 = None;
            entry.error = Some(error);
        }
        save_catalog(root, &entries)?;
        changed(state(root)?);
    }
    drop(_reservation);
    state(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "gary-native-loras-{label}-{}-{}",
            std::process::id(),
            NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
        ));
        checked_adapters_dir(&root).unwrap();
        root
    }

    fn entry(source: &Path) -> NativeLora {
        NativeLora {
            name: "koan".into(),
            source_path: source.to_string_lossy().into(),
            config_path: None,
            source_sha256: None,
            config_sha256: None,
            converter_sha256: None,
            native_path: None,
            native_sha256: None,
            strength: 0.8,
            error: None,
            prompts_path: None,
            training_checkpoints: Vec::new(),
            legacy_export: None,
        }
    }

    fn safetensors(path: &Path, header: serde_json::Value) {
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn config_uses_embedded_export_then_sidecar_and_reports_legacy_gap() {
        let root = root("config");
        let path = root.join("original.safetensors");
        let mut adapter = entry(&path);
        safetensors(
            &path,
            serde_json::json!({"__metadata__":{"lora_config":"{\"rank\":16}"}}),
        );
        assert_eq!(config_for_source(&adapter).unwrap(), None);
        safetensors(&path, serde_json::json!({"tensor":{}}));
        assert!(config_for_source(&adapter)
            .unwrap_err()
            .contains("configuration"));
        std::fs::write(path.with_extension("json"), "{\"rank\":16}").unwrap();
        assert_eq!(
            config_for_source(&adapter).unwrap(),
            Some(path.with_extension("json"))
        );
        adapter.config_path = Some(root.join("absent.json").to_string_lossy().into());
        assert!(config_for_source(&adapter).unwrap_err().contains("missing"));
        adapter.source_path = root.join("original.ckpt").to_string_lossy().into();
        assert!(config_for_source(&adapter)
            .unwrap_err()
            .contains("original is preserved"));
        std::fs::write(&path, u64::MAX.to_le_bytes()).unwrap();
        assert!(embedded_config(&path).unwrap_err().contains("too large"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn catalog_cannot_register_external_gguf_and_source_changes_clear_readiness() {
        let root = root("catalog");
        let mut adapter = entry(&root.join("first.safetensors"));
        adapter.native_path = Some(root.join("external.gguf").to_string_lossy().into());
        let mut entries = BTreeMap::from([("koan".into(), adapter.clone())]);
        save_catalog(&root, &entries).unwrap();
        assert!(read_catalog(&root).unwrap_err().contains("outside"));
        let native = adapters_dir(&root).join("lora-koan.gguf");
        std::fs::write(&native, b"native").unwrap();
        adapter.native_path = Some(native.to_string_lossy().into());
        entries.insert("koan".into(), adapter);
        save_catalog(&root, &entries).unwrap();
        assert!(state(&root).unwrap().entries[0].native_path.is_some());
        std::fs::write(
            root.join("sa3/lora_catalog.json"),
            serde_json::json!({"koan":{"path":root.join("second.safetensors"),"strength":0.5}})
                .to_string(),
        )
        .unwrap();
        let changed = state(&root).unwrap();
        assert!(changed.entries[0].native_path.is_none());
        assert_eq!(changed.entries[0].strength, 0.5);
        assert!(
            native.is_file(),
            "previous conversion must remain recoverable"
        );
        std::fs::write(root.join("sa3/lora_catalog.json"), "{}").unwrap();
        assert!(
            state(&root).unwrap().entries.is_empty(),
            "unregistered exports must disappear from the native menu"
        );
        assert!(native.is_file(), "unregistering is not destructive cleanup");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn converter_output_must_contain_a_supported_gguf_with_tensors() {
        let root = root("gguf-header");
        let path = root.join("adapter.gguf");
        std::fs::write(&path, b"converter exit zero is insufficient").unwrap();
        assert!(validate_gguf(&path).is_err());
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        std::fs::write(&path, &header).unwrap();
        assert!(validate_gguf(&path).is_err());
        header[8..16].copy_from_slice(&1u64.to_le_bytes());
        std::fs::write(&path, &header).unwrap();
        validate_gguf(&path).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn legacy_exports_survive_selection_changes_without_python_and_reject_tampering() {
        let root = root("legacy-cache");
        let dir = crate::sa3_training::checked_folder(
            &root,
            &["sa3", "native-loras", "legacy-exports", "old-revision"],
        )
        .unwrap();
        let original = root.join("original.ckpt");
        std::fs::write(&original, b"original checkpoint preserved").unwrap();
        let tensors = dir.join("adapter.safetensors");
        safetensors(
            &tensors,
            serde_json::json!({"__metadata__":{"lora_config":"{\"rank\":16}"}}),
        );
        let config = dir.join("adapter.json");
        std::fs::write(&config, b"{\"rank\":16}").unwrap();
        let source_hash = sha256_file(&original).await.unwrap();
        let export = LegacyExport {
            source_sha256: source_hash.clone(),
            exporter_sha256: "older-audited-exporter".into(),
            safetensors_path: tensors.to_string_lossy().into(),
            safetensors_sha256: sha256_file(&tensors).await.unwrap(),
            config_path: config.to_string_lossy().into(),
            config_sha256: sha256_file(&config).await.unwrap(),
        };
        crate::sa3_training::save(&dir.join("export.json"), &export).unwrap();
        let mut adapter = entry(&original);
        let input = export_legacy_checkpoint(&root, &mut adapter, &original, &source_hash)
            .await
            .unwrap();
        assert_eq!(input, tensors);
        adapter.source_sha256 = Some(source_hash.clone());
        verify_legacy_export(&root, &adapter).await.unwrap();
        assert!(!root.join("services/sa3/env").exists());
        std::fs::write(&config, b"changed").unwrap();
        assert!(verify_legacy_export(&root, &adapter)
            .await
            .unwrap_err()
            .contains("changed"));
        assert!(
            export_legacy_checkpoint(&root, &mut adapter, &original, &source_hash)
                .await
                .unwrap_err()
                .contains("one-time export")
        );
        assert_eq!(sha256_file(&original).await.unwrap(), source_hash);
        adapter.legacy_export.as_mut().unwrap().safetensors_path =
            original.to_string_lossy().into();
        assert!(verify_legacy_export(&root, &adapter)
            .await
            .unwrap_err()
            .contains("outside"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn automatically_discovered_sidecar_changes_block_migration() {
        let root = root("sidecar-integrity");
        let source = root.join("original.safetensors");
        safetensors(&source, serde_json::json!({"tensor":{}}));
        let config = source.with_extension("json");
        std::fs::write(&config, b"{\"rank\":16}").unwrap();
        let mut adapter = entry(&source);
        adapter.config_sha256 = Some(sha256_file(&config).await.unwrap());
        assert!(adapter.config_path.is_none());
        verify_source_config(&adapter).await.unwrap();
        std::fs::write(&config, b"{\"rank\":32}").unwrap();
        assert!(verify_source_config(&adapter)
            .await
            .unwrap_err()
            .contains("changed"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn unavailable_converter_does_not_leave_partial_output() {
        let root = root("spawn-failure");
        let source = root.join("original.safetensors");
        safetensors(
            &source,
            serde_json::json!({"__metadata__":{"lora_config":"{\"rank\":16}"}}),
        );
        let before = std::fs::read(&source).unwrap();
        let error = convert_one(
            &root,
            &mut entry(&source),
            &root.join("absent.exe"),
            "hash",
            None,
        )
        .await
        .unwrap_err();
        assert!(error.contains("Cannot start"));
        assert_eq!(std::fs::read(&source).unwrap(), before);
        assert!(std::fs::read_dir(adapters_dir(&root))
            .unwrap()
            .next()
            .is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    #[ignore = "uses a real legacy checkpoint and a junction to an existing Python environment"]
    async fn real_legacy_checkpoint_export_conversion_and_repair_without_python() {
        let converter = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_LORA_CONVERTER").expect("converter path required"),
        );
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_LEGACY_SMOKE_ROOT")
                .expect("isolated fixture root required"),
        );
        assert!(
            root.join("legacy-export-fixture.json").is_file(),
            "isolated fixture marker required"
        );
        assert!(
            !catalog_path(&root).exists(),
            "fresh native catalog required"
        );
        let source = root.join("original.ckpt");
        let source_hash = sha256_file(&source).await.unwrap();
        let historical = root.join("earlier.ckpt");
        let historical_hash = sha256_file(&historical).await.unwrap();
        checked_adapters_dir(&root).unwrap();
        let legacy_catalog = root.join("sa3/lora_catalog.json");
        std::fs::write(
            &legacy_catalog,
            serde_json::json!({"koan":{"path":source,"strength":0.8,"trainingCheckpoints":[{"step":1000,"epoch":1,"path":historical}]}}).to_string(),
        )
        .unwrap();
        let original_catalog = std::fs::read(&legacy_catalog).unwrap();
        assert!(verify_legacy_history(&root)
            .await
            .unwrap_err()
            .contains("without a verified export"));
        let first = prepare(&root, &converter, None, |_| {}).await.unwrap();
        let adapter = first.entries.first().unwrap();
        assert!(adapter.error.is_none(), "{:?}", adapter.error);
        assert_eq!(adapter.source_sha256.as_ref(), Some(&source_hash));
        let export = adapter.legacy_export.as_ref().unwrap();
        assert!(verified_export(&root, export).await.unwrap());
        let native = PathBuf::from(adapter.native_path.as_ref().unwrap());
        validate_gguf(&native).unwrap();
        let native_hash = sha256_file(&native).await.unwrap();
        let native_modified = native.metadata().unwrap().modified().unwrap();
        let exported_modified = Path::new(&export.safetensors_path)
            .metadata()
            .unwrap()
            .modified()
            .unwrap();
        let env = root.join("services/sa3/env");
        let retained = root.join("services/sa3/env-retained");
        assert!(
            !env.canonicalize()
                .unwrap()
                .starts_with(root.canonicalize().unwrap()),
            "fixture must use a junction, not a copied user environment"
        );
        assert!(!retained.exists());
        // Rename only the fixture junction; the environment it targets is untouched.
        std::fs::rename(&env, &retained).unwrap();
        assert!(!env.exists());
        let second = prepare(&root, &converter, None, |_| {}).await.unwrap();
        assert!(
            second.entries[0].error.is_none(),
            "{:?}",
            second.entries[0].error
        );
        assert_eq!(
            native.metadata().unwrap().modified().unwrap(),
            native_modified
        );
        assert_eq!(
            Path::new(&export.safetensors_path)
                .metadata()
                .unwrap()
                .modified()
                .unwrap(),
            exported_modified
        );
        std::fs::write(&native, b"damaged native cache").unwrap();
        let repaired = prepare(&root, &converter, None, |_| {}).await.unwrap();
        assert!(
            repaired.entries[0].error.is_none(),
            "{:?}",
            repaired.entries[0].error
        );
        assert_eq!(sha256_file(&native).await.unwrap(), native_hash);
        // Lost selected-source metadata can recover an older immutable export.
        let mut reset = entry(&source);
        reset.name = "koan".into();
        save_catalog(&root, &BTreeMap::from([("koan".into(), reset)])).unwrap();
        let recovered = prepare(&root, &converter, None, |_| {}).await.unwrap();
        assert!(
            recovered.entries[0].error.is_none(),
            "{:?}",
            recovered.entries[0].error
        );
        assert_eq!(
            recovered.entries[0].native_sha256.as_deref(),
            Some(native_hash.as_str())
        );
        assert_eq!(sha256_file(&source).await.unwrap(), source_hash);
        assert_eq!(std::fs::read(&legacy_catalog).unwrap(), original_catalog);
        assert_eq!(sha256_file(&historical).await.unwrap(), historical_hash);
        verify_legacy_history(&root).await.unwrap();
        let historical_export = cached_legacy_export(&root, None, &historical_hash)
            .await
            .unwrap()
            .unwrap();
        let original_config = std::fs::read(&historical_export.config_path).unwrap();
        std::fs::write(
            &historical_export.config_path,
            b"changed historical configuration",
        )
        .unwrap();
        assert!(verify_legacy_history(&root)
            .await
            .unwrap_err()
            .contains("without a verified export"));
        std::fs::write(&historical_export.config_path, original_config).unwrap();
        assert!(
            !std::fs::read_dir(adapters_dir(&root).join("legacy-exports"))
                .unwrap()
                .any(|item| item
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".export-"))
        );
        println!(
            "Verified legacy checkpoint and native cache without Python: {}",
            native.display()
        );
        std::fs::rename(&retained, &env).unwrap();
    }

    #[tokio::test]
    #[ignore = "uses an installed native converter and an existing exported adapter"]
    async fn real_native_lora_preparation_preserves_sources_and_reuses_verified_output() {
        let converter = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_LORA_CONVERTER").expect("converter path required"),
        );
        let source = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_LORA_SOURCE").expect("source path required"),
        );
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_LORA_SMOKE_ROOT")
                .expect("isolated smoke root required"),
        );
        assert!(!root.exists(), "smoke test must use a new isolated root");
        checked_adapters_dir(&root).unwrap();
        let original_hash = sha256_file(&source).await.unwrap();
        let entries = BTreeMap::from([
            ("koan".into(), entry(&source)),
            (
                "legacy".into(),
                NativeLora {
                    name: "legacy".into(),
                    source_path: root.join("original.ckpt").to_string_lossy().into(),
                    ..entry(&source)
                },
            ),
        ]);
        std::fs::write(root.join("original.ckpt"), b"legacy preserved").unwrap();
        save_catalog(&root, &entries).unwrap();
        let first = prepare(&root, &converter, None, |_| {}).await.unwrap();
        let ready = first
            .entries
            .iter()
            .find(|entry| entry.name == "koan")
            .unwrap();
        assert!(ready.error.is_none(), "{:?}", ready.error);
        let native = PathBuf::from(ready.native_path.as_ref().unwrap());
        validate_gguf(&native).unwrap();
        assert!(first
            .entries
            .iter()
            .find(|entry| entry.name == "legacy")
            .unwrap()
            .error
            .as_ref()
            .unwrap()
            .contains("original is preserved"));
        let modified = native.metadata().unwrap().modified().unwrap();
        let second = prepare(&root, &converter, None, |_| {}).await.unwrap();
        assert_eq!(
            second
                .entries
                .iter()
                .find(|entry| entry.name == "koan")
                .unwrap()
                .native_path,
            ready.native_path
        );
        assert_eq!(
            native.metadata().unwrap().modified().unwrap(),
            modified,
            "verified adapter should be reused"
        );
        assert_eq!(sha256_file(&source).await.unwrap(), original_hash);
        assert_eq!(
            std::fs::read(root.join("original.ckpt")).unwrap(),
            b"legacy preserved"
        );
        // A damaged conversion is repaired from the original export.
        std::fs::write(&native, b"damaged").unwrap();
        let repaired = prepare(&root, &converter, None, |_| {}).await.unwrap();
        assert_eq!(
            repaired
                .entries
                .iter()
                .find(|entry| entry.name == "koan")
                .unwrap()
                .native_sha256,
            ready.native_sha256
        );
        assert_eq!(sha256_file(&source).await.unwrap(), original_hash);
        assert!(!std::fs::read_dir(adapters_dir(&root))
            .unwrap()
            .any(|file| file
                .unwrap()
                .path()
                .extension()
                .is_some_and(|ext| ext == "partial")));
        println!("Verified native adapter: {}", native.display());
    }
}
