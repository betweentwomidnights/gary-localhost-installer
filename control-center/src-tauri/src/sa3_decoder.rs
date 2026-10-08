//! The optional decoder correction is an auxiliary adapter, not a creative LoRA.
use crate::native_runtime::sha256_file;
use crate::sa3_loras::NativeLora;
use serde::Serialize;
use std::path::{Path, PathBuf};

pub const MODEL_ID: &str = "sa3-native::decoder-correction";
const NAME: &str = "squeakfix-v3";

fn custom_path() -> Option<PathBuf> {
    std::env::var_os("SA3_DECODER_LORA_PATH")
        .map(|path| path.to_string_lossy().trim().to_string())
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

fn configuration_blocker() -> Option<String> {
    if custom_path().is_none()
        && [
            ("SA3_DECODER_LORA_REPO", "thepatch/same-l-decoder-lora"),
            ("SA3_DECODER_LORA_FILENAME", "squeakfix_v3.safetensors"),
        ]
        .iter()
        .any(|(key, default)| std::env::var(key).is_ok_and(|value| value != *default))
    {
        Some("Custom decoder repositories need an explicit SA3_DECODER_LORA_PATH for native preparation.".into())
    } else {
        None
    }
}

pub fn source_path(root: &Path) -> PathBuf {
    if let Some(path) = custom_path() {
        if path.is_absolute() {
            return path;
        }
        if let Ok(relative) = path.strip_prefix("~") {
            if let Some(home) = std::env::var_os("USERPROFILE") {
                return PathBuf::from(home).join(relative);
            }
        }
        return root.join("services/sa3").join(path);
    }
    crate::sa3_models::models_dir(root).join("squeakfix_v3.safetensors")
}

fn record_path(root: &Path) -> PathBuf {
    root.join("sa3/native-decoder/adapter.json")
}

pub fn read(root: &Path) -> Result<Option<NativeLora>, String> {
    let path = record_path(root);
    match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Cannot inspect native decoder correction: {error}")),
        Ok(_) => {}
    }
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let dir = path
        .parent()
        .unwrap()
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !dir.starts_with(&boundary)
        || path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err("Native decoder correction record points outside this storage profile.".into());
    }
    let entry: NativeLora =
        serde_json::from_slice(&std::fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("Invalid native decoder correction record: {error}"))?;
    if entry.name != NAME || entry.strength != 1.0 || entry.error.is_some() {
        return Err("Invalid native decoder correction record.".into());
    }
    let native = entry
        .native_path
        .as_deref()
        .ok_or("Native decoder copy is missing")?;
    crate::sa3_loras::checked_native_path(root, Path::new(native))?;
    Ok(Some(entry))
}

pub fn blocker(root: &Path) -> Option<String> {
    if let Some(error) = configuration_blocker() {
        return Some(error);
    }
    match read(root) {
        Err(error) => Some(error),
        Ok(None) => {
            Some("needs prepared native decoder correction; prepare it in SA3 Models".into())
        }
        Ok(Some(entry)) => {
            let selected = source_path(root).canonicalize();
            if selected.ok() != Path::new(&entry.source_path).canonicalize().ok()
                || !Path::new(&entry.source_path).is_file()
            {
                Some("native decoder source changed or is missing; prepare it again".into())
            } else if !entry
                .native_path
                .is_some_and(|path| Path::new(&path).is_file())
            {
                Some("native decoder copy is missing; prepare it again".into())
            } else {
                None
            }
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct State {
    pub enabled: bool,
    pub source_path: String,
    pub source_present: bool,
    pub prepared: bool,
    pub error: Option<String>,
}

pub fn state(root: &Path) -> State {
    let source = source_path(root);
    let error = blocker(root);
    State {
        enabled: crate::sa3_use_decoder_lora_enabled(),
        source_path: source.to_string_lossy().into(),
        source_present: source.is_file(),
        prepared: error.is_none(),
        error,
    }
}

pub async fn prepare(
    root: &Path,
    converter: &Path,
    runtime_path: Option<&std::ffi::OsStr>,
) -> Result<State, String> {
    if let Some(error) = configuration_blocker() {
        return Err(error);
    }
    let source = source_path(root);
    // Managed downloads are pinned. User-supplied decoder sources are retained
    // in place and identified by their own source/configuration hashes.
    if custom_path().is_none() {
        let file = &crate::sa3_models::component(MODEL_ID).unwrap().files[0];
        if sha256_file(&source).await? != file.sha256 {
            return Err(
                "Decoder source failed its pinned checksum; prepare its model component again."
                    .into(),
            );
        }
    }
    let mut entry = read(root)?.unwrap_or(NativeLora {
        native_only: false,
        name: NAME.into(),
        source_path: source.to_string_lossy().into(),
        config_path: None,
        source_sha256: None,
        config_sha256: None,
        converter_sha256: None,
        native_path: None,
        native_sha256: None,
        strength: 1.0,
        error: None,
        prompts_path: None,
        training_checkpoints: Vec::new(),
        legacy_export: None,
    });
    if entry.source_path != source.to_string_lossy() {
        entry.source_path = source.to_string_lossy().into();
        entry.config_path = None;
        entry.native_path = None;
    }
    crate::sa3_loras::prepare_decoder(root, &mut entry, converter, runtime_path).await?;
    let dir = crate::sa3_training::checked_folder(root, &["sa3", "native-decoder"])?;
    crate::sa3_training::save(&dir.join("adapter.json"), &entry)?;
    Ok(state(root))
}

pub async fn verified_path(root: &Path) -> Result<PathBuf, String> {
    if let Some(error) = blocker(root) {
        return Err(error);
    }
    let entry = read(root)?.ok_or("Prepare native decoder correction first")?;
    let path = PathBuf::from(entry.native_path.as_ref().unwrap());
    let source_hash = sha256_file(Path::new(&entry.source_path)).await?;
    if custom_path().is_none()
        && source_hash != crate::sa3_models::component(MODEL_ID).unwrap().files[0].sha256
    {
        return Err("Decoder correction failed its pinned checksum; prepare it again.".into());
    }
    if source_hash.as_str() != entry.source_sha256.as_deref().unwrap_or("")
        || sha256_file(&path).await?.as_str() != entry.native_sha256.as_deref().unwrap_or("")
    {
        return Err("Decoder correction changed since preparation; prepare it again.".into());
    }
    crate::sa3_loras::verify_source_config(&entry).await?;
    crate::sa3_loras::verify_legacy_export(root, &entry).await?;
    Ok(path)
}

pub async fn append_request(
    root: &Path,
    enabled: bool,
    request: &mut serde_json::Value,
) -> Result<(), String> {
    if enabled {
        let path = verified_path(root).await?;
        request["loras"]
            .as_array_mut()
            .ok_or("Native request has no LoRA list")?
            .push(serde_json::json!({"path":path,"strength":1.0}));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn disabled_correction_needs_no_files() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let root = std::env::temp_dir()
                    .join(format!("sa3-disabled-decoder-{}", std::process::id()));
                let mut request = json!({"loras":[]});
                append_request(&root, false, &mut request).await.unwrap();
                assert_eq!(request["loras"], json!([]));
                assert!(!root.exists());
            });
    }

    #[test]
    fn creative_adapter_cannot_be_prepared_as_decoder_correction() {
        let _test = crate::sa3_loras::REGISTRY_TEST.lock().unwrap();
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let root = std::env::temp_dir().join(format!("sa3-decoder-target-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            std::fs::create_dir_all(&root).unwrap();
            let source = root.join("creative.safetensors");
            let header = serde_json::to_vec(&json!({"__metadata__":{"lora_config":json!({"target":"dit","rank":16,"alpha":16}).to_string()}})).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec(); bytes.extend(header);
            std::fs::write(&source, &bytes).unwrap();
            let mut entry = NativeLora { native_only: false, name:NAME.into(), source_path:source.to_string_lossy().into(), config_path:None, source_sha256:None, config_sha256:None, converter_sha256:None, native_path:None, native_sha256:None, strength:1.0, error:None, prompts_path:None, training_checkpoints:vec![], legacy_export:None };
            assert!(crate::sa3_loras::prepare_decoder(&root, &mut entry, &root.join("missing-converter.exe"), None).await.unwrap_err().contains("target 'decoder'"));
            assert_eq!(std::fs::read(&source).unwrap(),bytes);
            assert!(!record_path(&root).exists());
            assert!(!crate::sa3_loras::catalog_path(&root).exists());
            crate::remove_managed_path(&root, &std::env::temp_dir()).unwrap();
        });
    }

    #[test]
    #[ignore = "requires pinned decoder source and installed native converter"]
    fn real_native_decoder_smoke() {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let root = PathBuf::from(std::env::var_os("GARY4LOCAL_SA3_DECODER_SMOKE_ROOT").expect("set isolated decoder root"));
            let converter = PathBuf::from(std::env::var_os("GARY4LOCAL_SA3_DECODER_CONVERTER").expect("set converter"));
            assert!(root.is_absolute());
            assert!(custom_path().is_none(), "use the pinned fixture for this smoke test");
            let source = source_path(&root);
            let source_hash = sha256_file(&source).await.unwrap();
            assert!(prepare(&root, &converter, None).await.unwrap().prepared);
            let native = verified_path(&root).await.unwrap();
            let hash = sha256_file(&native).await.unwrap();
            let modified = std::fs::metadata(&native).unwrap().modified().unwrap();
            assert!(prepare(&root, &converter, None).await.unwrap().prepared);
            assert_eq!(std::fs::metadata(&native).unwrap().modified().unwrap(), modified, "verified cache should be reused");
            for creative in [json!([]), json!([{"path":"creative.gguf","strength":0.8}])] {
                let mut request = json!({"loras":creative});
                let count = request["loras"].as_array().unwrap().len();
                append_request(&root, true, &mut request).await.unwrap();
                assert_eq!(request["loras"].as_array().unwrap().len(), count+1);
                assert_eq!(request["loras"][count]["path"],json!(native));
                assert_eq!(request["loras"][count]["strength"],1.0);
            }
            assert!(!crate::sa3_loras::read_catalog(&root).unwrap().contains_key(NAME));
            std::fs::write(&native, b"damaged native cache").unwrap();
            assert!(verified_path(&root).await.is_err());
            assert!(prepare(&root, &converter, None).await.unwrap().prepared);
            assert_eq!(sha256_file(&native).await.unwrap(), hash);
            assert_eq!(sha256_file(&source).await.unwrap(), source_hash);
            // A modified source must not be accepted as the pinned correction.
            let original = std::fs::read(&source).unwrap();
            std::fs::write(&source, b"damaged source").unwrap();
            assert!(prepare(&root, &converter, None).await.unwrap_err().contains("checksum"));
            assert!(verified_path(&root).await.is_err());
            std::fs::write(&source, original).unwrap();
            verified_path(&root).await.unwrap();
            println!("PASS pinned decoder conversion, auxiliary request composition, cache reuse/repair and original source preservation");
        });
    }
}
