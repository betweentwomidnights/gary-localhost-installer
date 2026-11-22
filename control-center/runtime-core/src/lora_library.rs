//! The desktop LoRA catalogs are also the headless library's source of truth.
use crate::{
    headless::{self, Config},
    service::Service,
    training::{carey, sa3},
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, Weak},
};

#[derive(Clone, Debug)]
pub struct Lease {
    _file: Arc<LockedFile>,
}
#[derive(Debug)]
struct LockedFile(File);
impl Drop for LockedFile {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

/// One process can reenter while its server holds the library. Other processes
/// must stop that service before editing the same adapters or training history.
pub fn lease(root: &Path, service: Service) -> Result<Option<Lease>, String> {
    if !matches!(service, Service::Sa3 | Service::Carey) {
        return Ok(None);
    }
    let dir = checked_dir(root, &[service.id()])?;
    let path = dir.join(".lora-library.lock");
    static OPEN: OnceLock<Mutex<HashMap<PathBuf, Weak<LockedFile>>>> = OnceLock::new();
    let mut open = OPEN
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|e| e.to_string())?;
    if let Some(lease) = open.get(&path).and_then(Weak::upgrade) {
        return Ok(Some(Lease { _file: lease }));
    }
    if fs::symlink_metadata(&path).is_ok_and(|m| !m.is_file()) {
        return Err("LoRA library lock must be a regular file".into());
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| e.to_string())?;
    file.try_lock().map_err(|_| format!("{}'s LoRA library is in use by another Gary process. Stop that service or finish its current operation first.", service.id()))?;
    let locked = Arc::new(LockedFile(file));
    open.insert(path, Arc::downgrade(&locked));
    Ok(Some(Lease { _file: locked }))
}

pub fn checked_dir(root: &Path, parts: &[&str]) -> Result<PathBuf, String> {
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let boundary = root.canonicalize().map_err(|e| e.to_string())?;
    let mut dir = boundary.clone();
    for part in parts {
        dir.push(part);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        dir = dir.canonicalize().map_err(|e| e.to_string())?;
        if !dir.starts_with(&boundary) {
            return Err("LoRA library directory escapes its storage root".into());
        }
    }
    Ok(dir)
}

pub fn desktop_root() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("GARY4LOCAL_RUNTIME_ROOT")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
    {
        return Ok(root);
    }
    #[cfg(target_os = "linux")]
    let local = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/share")))
        .ok_or("Cannot locate the desktop data folder")?
        .join("com.betweentwomidnights.gary4local");
    #[cfg(target_os = "macos")]
    let local = crate::storage_paths::macos_app_data(
        &std::env::var_os("HOME").map(PathBuf::from).filter(|p| p.is_absolute())
            .ok_or("Cannot locate macOS Application Support; HOME must be absolute")?
    );
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let local = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or("Cannot locate desktop storage. Set --library-root explicitly.")?
        .join("com.betweentwomidnights.gary4local");
    let settings = local.join("storage.json");
    match fs::read(&settings) {
        Ok(raw) => {
            let value: Value = serde_json::from_slice(&raw)
                .map_err(|e| format!("Invalid desktop storage settings: {e}"))?;
            if let Some(path) = value["runtimeRoot"]
                .as_str()
                .map(str::trim)
                .filter(|p| !p.is_empty())
            {
                let root = PathBuf::from(path);
                if root.is_absolute() {
                    return Ok(root);
                }
                return Err("Desktop storage must use an absolute path".into());
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("Cannot read desktop storage settings: {e}")),
    }
    #[cfg(target_os = "linux")]
    {
        let config = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")))
            .ok_or("Cannot locate desktop config storage")?;
        let legacy = config.join("Gary4JUCE");
        if [
            "app_settings.json",
            "hf_token.txt",
            "models",
            "services",
            "sa3",
            "carey",
        ]
        .iter()
        .any(|part| legacy.join(part).exists())
        {
            return Ok(legacy);
        }
        Ok(std::env::var_os("GARY4LOCAL_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .map(|p| p.join("data"))
            .unwrap_or_else(|| local.join("runtime")))
    }
    #[cfg(target_os = "macos")]
    { Ok(local.join("runtime")) }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        if let Some(base) = std::env::var_os("APPDATA") {
            let legacy = PathBuf::from(base).join("Gary4JUCE");
            if [
                "app_settings.json",
                "hf_token.txt",
                "models",
                "services",
                "sa3",
                "carey",
            ]
            .iter()
            .any(|part| legacy.join(part).exists())
            {
                return Ok(legacy);
            }
        }
        let installed = local
            .parent()
            .ok_or("Invalid local data folder")?
            .join("gary4local");
        if installed.join("gary4local.exe").is_file() {
            return Ok(installed.join("gary4local-data"));
        }
        Ok(std::env::current_exe()
            .map_err(|e| e.to_string())?
            .parent()
            .ok_or("Cannot locate Gary")?
            .join("gary4local-data"))
    }
}

pub fn adapters_dir(root: &Path, service: Service) -> PathBuf {
    root.join(service.id()).join(if service == Service::Sa3 {
        "native-loras"
    } else {
        "adapters"
    })
}
pub fn catalog_path(root: &Path, service: Service) -> PathBuf {
    if service == Service::Sa3 {
        adapters_dir(root, service).join("catalog.json")
    } else {
        root.join(service.id()).join("lora_catalog.json")
    }
}
pub fn catalog(root: &Path, service: Service) -> Result<Map<String, Value>, String> {
    let path = catalog_path(root, service);
    match fs::read(&path) {
        Ok(raw) => serde_json::from_slice(&raw)
            .map_err(|e| format!("Invalid LoRA catalog {}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Map::new()),
        Err(e) => Err(format!("Cannot read {}: {e}", path.display())),
    }
}
fn write_catalog(
    root: &Path,
    service: Service,
    catalog: &Map<String, Value>,
) -> Result<(), String> {
    let path = catalog_path(root, service);
    crate::sa3_prompts::save_with_policy(&path, catalog, true)
}
fn valid_name(name: &str) -> Result<(), String> {
    if crate::training::sanitize_lora_name(name).as_deref() != Some(name)
        || ["none", "default", "defaults"].contains(&name)
    {
        return Err("Use a valid LoRA name".into());
    }
    Ok(())
}
pub(crate) fn available(config: &Config, name: &str) -> Result<(), String> {
    valid_name(name)?;
    let root = config.library_root.as_ref().ok_or("Missing LoRA library")?;
    if catalog(root, config.service)?.contains_key(name) {
        return Err(format!(
            "A LoRA named {name} already exists in the shared library."
        ));
    }
    if config.service == Service::Sa3 {
        let legacy = root.join("sa3/lora_catalog.json");
        if let Ok(raw) = fs::read(legacy) {
            let entries: Map<String, Value> =
                serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
            if entries.contains_key(name) {
                return Err(format!(
                    "LoRA {name} already exists in the desktop legacy catalog."
                ));
            }
        }
    }
    Ok(())
}

fn fingerprint(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn copy(source: &Path, dest: &Path) -> Result<(), String> {
    fs::copy(source, dest)
        .map(|_| ())
        .map_err(|e| e.to_string())
}
fn prepared_sa3(root: &Path, name: &str, source: &Path) -> Result<(PathBuf, String), String> {
    crate::sa3_gguf::creative_adapter(source)?;
    let hash = fingerprint(source)?;
    let dir = checked_dir(root, &["sa3", "native-loras"])?;
    let dest = dir.join(format!("lora-{name}-{}-f32.gguf", &hash[..24]));
    if !dest.is_file() || fingerprint(&dest)? != hash {
        let stage = dir.join(format!(".adapter-{}.partial", headless::serial()));
        let result =
            copy(source, &stage).and_then(|_| fs::rename(&stage, &dest).map_err(|e| e.to_string()));
        let _ = fs::remove_file(&stage);
        result?;
    }
    Ok((dest, hash))
}

pub fn publish(
    config: &Config,
    name: &str,
    source: &Path,
    family: &str,
    captions: &[String],
    dataset: Option<&Path>,
) -> Result<PathBuf, String> {
    let root = config.library_root.as_ref().ok_or("Missing LoRA library")?;
    let _lease = lease(root, config.service)?;
    available(config, name)?;
    let source = source.canonicalize().map_err(|e| e.to_string())?;
    let mut entries = catalog(root, config.service)?;
    let (dest, entry) = if config.service == Service::Sa3 {
        let (dest, hash) = prepared_sa3(root, name, &source)?;
        let prompts = checked_dir(root, &["sa3", "prompts"])?;
        if let Some(dataset) = dataset {
            crate::sa3_prompts::build(&prompts, name, dataset, false)?;
        }
        let checkpoints = sa3_checkpoints(root, name)?;
        let entry = json!({"nativeOnly":true,"name":name,"sourcePath":source,"configPath":null,
            "sourceSha256":hash,"configSha256":null,"converterSha256":null,"nativePath":dest,
            "nativeSha256":hash,"strength":1.0,"error":null,"promptsPath":dataset,"trainingCheckpoints":checkpoints});
        (dest, entry)
    } else {
        let generation = peft_generation(&source)?;
        let dest = publish_carey(root, name, &generation, family, captions)?;
        let (job, checkpoints) = carey_checkpoints(&source)?;
        let recorded_dataset = source
            .parent()
            .and_then(|job| fs::read(job.join("native-options.json")).ok())
            .and_then(|raw| serde_json::from_slice::<carey::Options>(&raw).ok())
            .map(|options| options.dataset);
        let entry = json!({"path":generation,"captionsPath":dataset.or(recorded_dataset.as_deref()),"scale":1.0,
            "backends":["cpu","mps","cuda"],"modelFamily":family,"trainingJobId":job,
            "trainingCheckpoints":checkpoints,"selectedTrainingCheckpoint": if checkpoints.is_empty() { Value::Null } else { json!("final") }});
        let path = root.join("carey/captions.json");
        let mut pools: Map<String, Value> = match fs::read(&path) {
            Ok(raw) => serde_json::from_slice(&raw).map_err(|e| e.to_string())?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Map::new(),
            Err(e) => return Err(e.to_string()),
        };
        pools.insert(name.into(), json!(captions));
        crate::sa3_prompts::save_with_policy(&path, &pools, true)?;
        (dest, entry)
    };
    entries.insert(name.into(), entry);
    write_catalog(root, config.service, &entries)?;
    Ok(dest)
}

fn peft_generation(source: &Path) -> Result<PathBuf, String> {
    if fs::symlink_metadata(source.join("checkpoint.current")).is_ok() {
        carey::resolve_checkpoint(source)
    } else if source.join("adapter_config.json").is_file()
        && source.join("adapter_model.safetensors").is_file()
    {
        Ok(source.to_owned())
    } else {
        Err("Choose a complete PEFT checkpoint".into())
    }
}
fn publish_carey(
    root: &Path,
    name: &str,
    source: &Path,
    family: &str,
    captions: &[String],
) -> Result<PathBuf, String> {
    let dir = checked_dir(root, &["carey", "adapters"])?;
    let stage = dir.join(format!(".{name}-{}", headless::serial()));
    let dest = dir.join(name);
    fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        for file in ["adapter_model.safetensors", "adapter_config.json"] {
            copy(&source.join(file), &stage.join(file))?;
        }
        let metadata: Value = serde_json::from_slice(
            &fs::read(stage.join("adapter_config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if !metadata["peft_type"]
            .as_str()
            .is_some_and(|s| s.eq_ignore_ascii_case("lora"))
        {
            return Err("Choose a PEFT LoRA or DoRA adapter".into());
        }
        crate::sa3_prompts::save_with_policy(
            &stage.join("adapter_info.json"),
            &json!({"model_family":family,"captions":captions}),
            false,
        )?;
        fs::rename(&stage, &dest).map_err(|e| e.to_string())
    })();
    let _ = fs::remove_dir_all(stage);
    result?;
    Ok(dest)
}

fn sa3_checkpoints(root: &Path, name: &str) -> Result<Vec<Value>, String> {
    let jobs = root.join("sa3/training/jobs");
    let mut result = Vec::new();
    if !jobs.is_dir() {
        return Ok(result);
    }
    for job in crate::dataset::read_dir_sorted(&jobs)? {
        let Ok(raw) = fs::read(job.join("native-options.json")) else {
            continue;
        };
        let Ok(options) = serde_json::from_slice::<sa3::Options>(&raw) else {
            continue;
        };
        if options.name != name {
            continue;
        }
        let run = job.join("run");
        if !run.is_dir() {
            continue;
        }
        for file in crate::dataset::read_dir_sorted(&run)? {
            let number = file
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("adapter-step-"))
                .and_then(|s| s.strip_suffix(".gguf"))
                .and_then(|s| s.parse::<u32>().ok());
            if let Some(step) = number {
                let state = run.join(format!("trainer-state-step-{step}.gguf"));
                if file.is_file() && state.is_file() {
                    result.push(json!({"jobId":job.file_name().unwrap().to_string_lossy(),"step":step,"epoch":null,"path":file,"statePath":state}));
                }
            }
        }
    }
    result.sort_by_key(|v| v["step"].as_u64().unwrap_or(0));
    Ok(result)
}
fn carey_checkpoints(source: &Path) -> Result<(Option<String>, Vec<Value>), String> {
    let Some(job) = source
        .parent()
        .filter(|_| source.file_name().is_some_and(|s| s == "run"))
    else {
        return Ok((None, Vec::new()));
    };
    if !job.join("native-options.json").is_file() {
        return Ok((None, Vec::new()));
    }
    let mut checkpoints = Vec::new();
    for dir in crate::dataset::read_dir_sorted(source)? {
        let epoch = dir
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("checkpoint-epoch-"))
            .and_then(|s| s.parse::<u32>().ok());
        if let Some(epoch) = epoch {
            if let Ok(generation) = carey::resolve_checkpoint(&dir) {
                checkpoints.push(json!({"id":format!("epoch-{epoch}"),"label":format!("epoch {epoch}"),"epoch":epoch,"path":generation}));
            }
        }
    }
    checkpoints.sort_by_key(|v| v["epoch"].as_u64().unwrap_or(0));
    checkpoints.push(
        json!({"id":"final","label":"final epoch","epoch":null,"path":peft_generation(source)?}),
    );
    Ok((
        Some(job.file_name().unwrap().to_string_lossy().into()),
        checkpoints,
    ))
}

pub struct Registry(pub PathBuf);
impl crate::sa3_adapter::AdapterRegistry for Registry {
    fn entries(&self) -> Result<Vec<crate::sa3_adapter::AdapterLora>, String> {
        let dir = adapters_dir(&self.0, Service::Sa3);
        let mut result = Vec::new();
        for (name, entry) in catalog(&self.0, Service::Sa3)? {
            valid_name(&name)?;
            let native = entry["nativePath"].as_str().map(PathBuf::from);
            let mut error = entry["error"].as_str().map(String::from);
            if let Some(path) = &native {
                match path.canonicalize() {
                    Ok(path)
                        if path.parent()
                            == Some(dir.canonicalize().map_err(|e| e.to_string())?.as_path()) => {}
                    Ok(_) => return Err("Prepared LoRA escapes the shared library".into()),
                    Err(e) => error = Some(format!("Prepared LoRA is unavailable: {e}")),
                }
            }
            result.push(crate::sa3_adapter::AdapterLora {
                name,
                source_path: entry["sourcePath"].as_str().unwrap_or_default().into(),
                native_path: native.map(|p| p.to_string_lossy().into_owned()),
                strength: entry["strength"].as_f64().unwrap_or(1.0),
                error,
            });
        }
        Ok(result)
    }
    fn paths(&self) -> (PathBuf, PathBuf) {
        (
            adapters_dir(&self.0, Service::Sa3),
            catalog_path(&self.0, Service::Sa3),
        )
    }
    fn append_decoder<'a>(
        &'a self,
        _: &'a mut Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

pub(crate) fn select(config: &Config, name: &str, weights: &Path) -> Result<(), String> {
    let root = config.library_root.as_ref().ok_or("Missing LoRA library")?;
    let _lease = lease(root, config.service)?;
    let mut entries = catalog(root, config.service)?;
    let entry = entries
        .get_mut(name)
        .ok_or("LoRA is no longer registered")?;
    if config.service == Service::Sa3 {
        let (native, hash) = prepared_sa3(root, name, weights)?;
        entry["sourcePath"] = json!(weights);
        entry["sourceSha256"] = json!(hash);
        entry["nativePath"] = json!(native);
        entry["nativeSha256"] = json!(hash);
        entry["error"] = Value::Null;
    } else {
        let source = weights.parent().ok_or("Invalid PEFT checkpoint")?;
        let dest = adapters_dir(root, config.service)
            .join(name)
            .join("adapter_model.safetensors");
        let stage = dest.with_extension(format!("{}.tmp", headless::serial()));
        let result = copy(weights, &stage)
            .and_then(|_| fs::rename(&stage, &dest).map_err(|e| e.to_string()));
        let _ = fs::remove_file(stage);
        result?;
        entry["path"] = json!(source);
        let selected = entry["trainingCheckpoints"]
            .as_array()
            .and_then(|list| {
                list.iter().find(|v| {
                    v["path"].as_str().is_some_and(|p| {
                        Path::new(p).canonicalize().ok() == source.canonicalize().ok()
                    })
                })
            })
            .map(|v| v["id"].clone())
            .unwrap_or(Value::Null);
        entry["selectedTrainingCheckpoint"] = selected;
    }
    write_catalog(root, config.service, &entries)
}

/// Select only a checkpoint belonging to this registered LoRA's saved history.
pub fn select_checkpoint(config: &Config, name: &str, requested: &Path) -> Result<(), String> {
    let requested = requested.canonicalize().map_err(|e| e.to_string())?;
    let history = crate::lora_history::history(config, name)?
        .ok_or("This LoRA has no saved training history")?;
    let selected = history
        .checkpoints
        .iter()
        .position(|checkpoint| checkpoint.weights.canonicalize().ok().as_ref() == Some(&requested))
        .ok_or("Choose a checkpoint from this LoRA's saved history")?;
    crate::lora_history::activate(config, name, &history, selected)
}

/// Explicitly connect a legacy headless profile to a desktop library. Copy
/// immutable history first and switch the profile only after publication.
pub fn share(config_path: &Path, target: &Path) -> Result<Value, String> {
    let setting = target.to_owned();
    let resolved = if target == Path::new("desktop") {
        desktop_root()?
    } else {
        target.to_owned()
    };
    let target = resolved.as_path();
    if !target.is_absolute()
        || target
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("Choose an absolute normalized library root".into());
    }
    let original = fs::read(config_path).map_err(|e| e.to_string())?;
    let config = Config::from_bytes(&original).map_err(|e| e.to_string())?;
    if !matches!(config.service, Service::Sa3 | Service::Carey) {
        return Err("LoRA libraries are available for SA3 and Carey".into());
    }
    if config.library_root.as_ref().is_some_and(|root| {
        root == target
            || matches!((root.canonicalize(), target.canonicalize()), (Ok(a), Ok(b)) if a == b)
    }) {
        return Ok(json!({"shared":true,"library_root":target,"migrated":[]}));
    }
    if config.library_root.is_some() {
        return Err("This profile already uses a library. Use desktop's storage migration to move that library.".into());
    }
    let _profile = headless::reserve_tool(&config)?;
    let _library = lease(target, config.service)?;
    let root = target.canonicalize().map_err(|e| e.to_string())?;
    let mut updated = config.clone();
    updated.library_root = Some(root.clone());
    updated.validate()?;
    let names = crate::lora_history::registered(&config)?;
    let existing = catalog(&root, config.service)?;
    for name in &names {
        if let Some(entry) = existing.get(name) {
            let old = crate::training_job::adapter_path(&config, name)?;
            let new = if config.service == Service::Sa3 {
                PathBuf::from(
                    entry["nativePath"]
                        .as_str()
                        .ok_or("Existing LoRA has no native path")?,
                )
            } else {
                adapters_dir(&root, config.service)
                    .join(name)
                    .join("adapter_model.safetensors")
            };
            let old = if config.service == Service::Carey {
                old.join("adapter_model.safetensors")
            } else {
                old
            };
            if fingerprint(&old)? != fingerprint(&new)? {
                return Err(format!(
                    "LoRA {name} already exists with different weights in the shared library."
                ));
            }
        } else {
            available(&updated, name)?;
        }
    }
    let old_jobs = config.training_dir().join("jobs");
    let new_jobs = checked_dir(&root, &[config.service.id(), "training", "jobs"])?;
    let mut job_paths = Vec::new();
    if old_jobs.is_dir() {
        for job in crate::dataset::read_dir_sorted(&old_jobs)? {
            if !fs::symlink_metadata(&job).is_ok_and(|m| m.is_dir()) {
                continue;
            }
            let Ok(options) = fs::read(job.join("native-options.json")) else {
                continue;
            };
            let options: Value = serde_json::from_slice(&options).map_err(|e| e.to_string())?;
            if !options["name"]
                .as_str()
                .is_some_and(|name| valid_name(name).is_ok())
            {
                continue;
            }
            let Ok(progress) = sa3::read_progress(&job.join("native-progress.json")) else {
                continue;
            };
            if !matches!(progress.status.as_str(), "completed" | "cancelled") {
                continue;
            }
            let job = job.canonicalize().map_err(|e| e.to_string())?;
            let name = format!(
                "native-headless-{}",
                job.file_name()
                    .ok_or("Invalid training job")?
                    .to_string_lossy()
            );
            let dest = new_jobs.join(name);
            if dest.exists() {
                let origin: Value = serde_json::from_slice(
                    &fs::read(dest.join(".headless-source.json"))
                        .map_err(|e| format!("Cannot verify existing migrated history: {e}"))?,
                )
                .map_err(|e| e.to_string())?;
                if origin["source"] != json!(job)
                    || origin["settings"] != json!(fingerprint(&job.join("native-options.json"))?)
                {
                    return Err("Existing training history belongs to a different migration".into());
                }
                job_paths.push((job, dest));
                continue;
            }
            let stage = new_jobs.join(format!(".sharing-{}", headless::serial()));
            let result = (|| {
                copy_history(&job, &stage)?;
                let progress_path = stage.join("native-progress.json");
                let mut value: Value =
                    serde_json::from_slice(&fs::read(&progress_path).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                for key in ["final_adapter", "adapter_checkpoint", "state_checkpoint"] {
                    if let Some(path) = value[key].as_str().filter(|p| !p.is_empty()) {
                        let path = Path::new(path).canonicalize().map_err(|e| e.to_string())?;
                        let relative = path
                            .strip_prefix(&job)
                            .map_err(|_| "Training progress points outside its job")?;
                        value[key] = json!(dest.join(relative));
                    }
                }
                crate::sa3_prompts::save_with_policy(&progress_path, &value, true)?;
                crate::sa3_prompts::save_with_policy(
                    &stage.join(".headless-source.json"),
                    &json!({"source":job,"settings":fingerprint(&job.join("native-options.json"))?}),
                    false,
                )?;
                fs::rename(&stage, &dest).map_err(|e| e.to_string())
            })();
            let _ = fs::remove_dir_all(&stage);
            result?;
            job_paths.push((job, dest));
        }
    }
    for name in &names {
        if existing.contains_key(name) {
            continue;
        }
        let history = crate::lora_history::history(&config, name)?;
        let source = history
            .as_ref()
            .map(|h| h.checkpoints[h.active].weights.clone())
            .unwrap_or(crate::training_job::adapter_path(&config, name)?);
        let mut shared_source = source.clone();
        for (old, new) in &job_paths {
            if let Ok(relative) = source.strip_prefix(old) {
                shared_source = new.join(relative);
            }
        }
        if config.service == Service::Sa3 {
            let prompts = config.prompts_dir().join(format!("{name}.json"));
            if prompts.is_file() {
                let value: Value =
                    serde_json::from_slice(&fs::read(&prompts).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                let dir = checked_dir(&root, &["sa3", "prompts"])?;
                crate::sa3_prompts::save_with_policy(
                    &dir.join(format!("{name}.json")),
                    &value,
                    false,
                )?;
            }
            let dataset = shared_source
                .parent()
                .and_then(Path::parent)
                .and_then(|job| fs::read(job.join("native-options.json")).ok())
                .and_then(|raw| serde_json::from_slice::<sa3::Options>(&raw).ok())
                .map(|options| options.dataset);
            publish(&updated, name, &shared_source, "", &[], dataset.as_deref())?;
        } else {
            let published = crate::training_job::adapter_path(&config, name)?;
            let info: Value = serde_json::from_slice(
                &fs::read(published.join("adapter_info.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let family = info["model_family"].as_str().unwrap_or("standard");
            let captions: Vec<String> =
                serde_json::from_value(info["captions"].clone()).map_err(|e| e.to_string())?;
            let source = shared_source
                .parent()
                .filter(|_| shared_source.is_file())
                .unwrap_or(&shared_source);
            let run = job_paths
                .iter()
                .find(|(_, path)| source.starts_with(path))
                .map(|(_, path)| path.join("run"));
            publish(
                &updated,
                name,
                run.as_deref().unwrap_or(source),
                family,
                &captions,
                None,
            )?;
            if run.is_some() {
                select(&updated, name, &shared_source)?;
            }
        }
    }
    if fs::read(config_path).map_err(|e| e.to_string())? != original {
        return Err(
            "The profile changed during library sharing. Its original settings were preserved."
                .into(),
        );
    }
    let backup = config_path.with_extension(format!("before-sharing-{}.json", headless::serial()));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(&backup)
        .and_then(|mut f| {
            f.write_all(&original)?;
            f.sync_all()
        })
        .map_err(|e| e.to_string())?;
    let mut body: Value = serde_json::from_slice(&original).map_err(|e| e.to_string())?;
    body["library_root"] = json!(if setting == Path::new("desktop") {
        setting
    } else {
        root.clone()
    });
    headless::write_json(config_path, &body)?;
    Ok(json!({"shared":true,"library_root":root,"migrated":names,"backup":backup}))
}

fn copy_history(source: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir(dest).map_err(|e| e.to_string())?;
    for path in crate::dataset::read_dir_sorted(source)? {
        let name = path.file_name().ok_or("Invalid training file")?;
        let target = dest.join(name);
        let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if meta.is_dir() {
            copy_history(&path, &target)?;
        } else if meta.is_file() {
            if fs::hard_link(&path, &target).is_err() {
                copy(&path, &target)?;
            }
        } else if meta.is_symlink() && name == "checkpoint.current" {
            carey::resolve_checkpoint(path.parent().ok_or("Invalid checkpoint")?)?;
            fs::write(
                target,
                fs::read_link(&path)
                    .map_err(|e| e.to_string())?
                    .to_string_lossy()
                    .as_bytes(),
            )
            .map_err(|e| e.to_string())?;
        } else {
            return Err(format!(
                "Unsupported file in training history {}",
                path.display()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharing_preserves_sources_active_checkpoints_and_unrelated_desktop_entries() {
        for service in [Service::Sa3, Service::Carey] {
            let fixture = std::env::temp_dir().join(format!("gary-share-{}", headless::serial()));
            let host = fixture.join("host");
            let library = fixture.join("desktop");
            let profile = fixture.join("profile.json");
            fs::create_dir_all(&fixture).unwrap();
            headless::write_json(
                &profile,
                &json!({"service":service,"runtime_root":host,
                "device":"Vulkan0","public_port":8006,
                "model":if service == Service::Sa3 { "medium" } else { "base" },
                "encoding":if service == Service::Sa3 { "f16" } else { "bf16" }}),
            )
            .unwrap();
            let original = fs::read(&profile).unwrap();
            let config = Config::read(&profile).unwrap();
            fs::create_dir_all(config.installed_dir()).unwrap();
            fs::write(host.join(".gary4local-headless"), "gary4local-server v1\n").unwrap();
            fs::write(
                config.installed_dir().join("BUILD-INFO.json"),
                json!({
                    "service":service.package(),"platform":crate::runtime::current_platform(),
                    "dirty":false,"tests_run":true,"commit":"fixture","ggml_commit":"fixture",
                    "version":config.summary()["expected_version"]
                })
                .to_string(),
            )
            .unwrap();
            let job = host.join("training/jobs/fixture");
            let run = job.join("run");
            let dataset = fixture.join("dataset");
            fs::create_dir_all(&run).unwrap();
            fs::create_dir_all(&dataset).unwrap();
            fs::create_dir_all(config.adapters_dir()).unwrap();
            fs::create_dir_all(config.prompts_dir()).unwrap();
            let options = if service == Service::Sa3 {
                json!({"name":"fixture","dataset":dataset,"fixedPrompt":"","steps":1000,"rank":64,
                    "batchSize":1,"checkpointEvery":500,"duration":30,"learningRate":0.0001,
                    "targetLatentRms":0.0,"layerScope":"transformer-core","encoding":"F16","resume":null})
            } else {
                json!({"name":"fixture","dataset":dataset,"dit":"base","adapterType":"lora","profile":"default",
                    "rank":64,"epochs":2,"checkpointEvery":1,"batchSize":1,"accumulation":1,"learningRate":0.0001,
                    "cfgDropout":0.0,"timestepMu":0.0,"minSnr":null,"genreRatio":0,"trigger":"",
                    "windowMinSeconds":30,"windowMaxSeconds":60})
            };
            fs::write(job.join("native-options.json"), options.to_string()).unwrap();
            let (earlier, final_adapter, final_weights) = if service == Service::Sa3 {
                let earlier = run.join("adapter-step-500.gguf");
                let final_adapter = run.join("adapter-final.gguf");
                let bytes = crate::sa3_gguf::fixture("sa3-lora", "dit");
                fs::write(&earlier, &bytes).unwrap();
                fs::write(run.join("trainer-state-step-500.gguf"), b"state").unwrap();
                let mut final_bytes = bytes;
                *final_bytes.last_mut().unwrap() ^= 1;
                fs::write(&final_adapter, final_bytes).unwrap();
                (earlier, final_adapter.clone(), final_adapter)
            } else {
                let peft = |checkpoint: &Path, bytes: &[u8]| {
                    let relative = ".checkpoint-generations/generation-1";
                    let generation = checkpoint.join(relative);
                    fs::create_dir_all(&generation).unwrap();
                    fs::write(checkpoint.join("checkpoint.current"), relative).unwrap();
                    fs::write(
                        generation.join("adapter_config.json"),
                        r#"{"peft_type":"LORA"}"#,
                    )
                    .unwrap();
                    let weights = generation.join("adapter_model.safetensors");
                    fs::write(&weights, bytes).unwrap();
                    weights
                };
                let earlier = peft(&run.join("checkpoint-epoch-1"), b"earlier");
                let final_weights = peft(&run, b"final");
                (earlier, run.clone(), final_weights)
            };
            let mut dest = crate::training_job::adapter_path(&config, "fixture").unwrap();
            if service == Service::Carey {
                fs::create_dir_all(&dest).unwrap();
                fs::copy(
                    earlier.with_file_name("adapter_config.json"),
                    dest.join("adapter_config.json"),
                )
                .unwrap();
                fs::write(
                    dest.join("adapter_info.json"),
                    r#"{"model_family":"xl","captions":["Synth melody"]}"#,
                )
                .unwrap();
                dest.push("adapter_model.safetensors");
            } else {
                fs::write(
                    config.prompts_dir().join("fixture.json"),
                    r#"{"dice":{"instrumental":["Curated prompt"]}}"#,
                )
                .unwrap();
            }
            fs::copy(&earlier, &dest).unwrap();
            fs::write(job.join("native-progress.json"), json!({"schema_version":1,"status":"completed","phase":"completed",
                "message":"","error":"","step":1000,"epoch":2,"max_steps":1000,"final_adapter":final_adapter,
                "adapter_checkpoint":earlier}).to_string()).unwrap();
            let old_progress = fs::read(job.join("native-progress.json")).unwrap();
            let mut entries = Map::new();
            entries.insert("untouched".into(), json!({"sentinel":true}));
            checked_dir(
                &library,
                &[
                    service.id(),
                    if service == Service::Sa3 {
                        "native-loras"
                    } else {
                        "adapters"
                    },
                ],
            )
            .unwrap();
            write_catalog(&library, service, &entries).unwrap();

            if service == Service::Sa3 {
                let mut conflicting = entries.clone();
                conflicting.insert("fixture".into(), json!({"nativePath":final_weights}));
                write_catalog(&library, service, &conflicting).unwrap();
                assert!(share(&profile, &library)
                    .unwrap_err()
                    .contains("different weights"));
                assert_eq!(fs::read(&profile).unwrap(), original);
                assert_eq!(catalog(&library, service).unwrap(), conflicting);
                assert_eq!(
                    fs::read(job.join("native-progress.json")).unwrap(),
                    old_progress
                );
                write_catalog(&library, service, &entries).unwrap();
            }
            let shared = share(&profile, &library).unwrap();
            let updated = Config::read(&profile).unwrap();
            assert_eq!(
                fs::read(shared["backup"].as_str().unwrap()).unwrap(),
                original
            );
            assert_eq!(
                fs::read(job.join("native-progress.json")).unwrap(),
                old_progress
            );
            assert_eq!(fs::read(&dest).unwrap(), fs::read(&earlier).unwrap());
            assert_eq!(
                catalog(&library, service).unwrap()["untouched"],
                entries["untouched"]
            );
            let history = crate::lora_history::history(&updated, "fixture")
                .unwrap()
                .unwrap();
            assert_eq!((history.active, history.checkpoints.len()), (1, 2));
            assert!(history
                .checkpoints
                .iter()
                .all(|c| c.weights.starts_with(updated.training_dir())));
            crate::lora_history::activate(&updated, "fixture", &history, 0).unwrap();
            let published = crate::training_job::adapter_path(&updated, "fixture").unwrap();
            let published = if service == Service::Carey {
                published.join("adapter_model.safetensors")
            } else {
                published
            };
            assert_eq!(
                fs::read(published).unwrap(),
                fs::read(&final_weights).unwrap()
            );
            assert_eq!(share(&profile, &library).unwrap()["migrated"], json!([]));
            fs::remove_dir_all(fixture).unwrap();
        }
    }

    #[test]
    fn another_process_cannot_edit_a_leased_library() {
        let root = std::env::temp_dir().join(format!("gary-library-lock-{}", headless::serial()));
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "lora_library::tests::lease_child", "--ignored"])
            .env("GARY_TEST_LIBRARY_LEASE", &root)
            .spawn()
            .unwrap();
        let start = std::time::Instant::now();
        while !root.join("ready").is_file() && start.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let ready = root.join("ready").is_file();
        let blocked = lease(&root, Service::Sa3).is_err();
        fs::write(root.join("release"), b"").unwrap();
        let status = child.wait().unwrap();
        assert!(ready && blocked && status.success());
        let first = lease(&root, Service::Sa3).unwrap();
        let reentrant = lease(&root, Service::Sa3).unwrap();
        drop((first, reentrant));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "launched by the cross-process lease test"]
    fn lease_child() {
        let Some(root) = std::env::var_os("GARY_TEST_LIBRARY_LEASE") else {
            return;
        };
        let root = PathBuf::from(root);
        let _lease = lease(&root, Service::Sa3).unwrap();
        fs::write(root.join("ready"), b"").unwrap();
        let start = std::time::Instant::now();
        while !root.join("release").is_file() && start.elapsed().as_secs() < 15 {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}
