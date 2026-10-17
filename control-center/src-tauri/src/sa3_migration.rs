//! Inventory and native validation for SA3's move from PyTorch.
//! Verification and activation preserve old files. Cleanup remains a separate
//! transaction after runtime and adapter requirements have been satisfied.
//! What SA3's Python install owns is its manifest `legacy` block, applied by
//! the shared engine in `migration_inventory`; what is SA3's alone is here.

use crate::migration_inventory::Extras;
use std::path::{Path, PathBuf};

#[cfg(test)]
pub use crate::migration_inventory::Preview as Sa3MigrationPreview;

fn target() -> crate::migration::Target<'static> {
    crate::sa3_runtime::target()
}

pub fn native_training_base_present(active_root: &Path) -> bool {
    let models = crate::sa3_models::models_dir(active_root);
    ["F16", "Q4_K_M"].iter().any(|encoding| {
        crate::sa3_models::component(&format!("sa3-native::medium-base-{encoding}"))
            .is_some_and(|component| crate::sa3_models::present(component, &models))
    })
}

/// What SA3's tests expect its manifest to declare as the staged PyTorch
/// training cache (train_lora_job.py::stage_base_model). A test checks the
/// manifest still matches, so an edit there cannot silently widen cleanup.
#[cfg(test)]
pub(crate) const STAGED_TRAINING_BASE: &str = "sa3/training/models/sa3-medium/base";
#[cfg(test)]
pub(crate) const STAGED_TRAINING_FILES: &[&str] = &[
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/tokenizer.json",
    "t5gemma-b-b-ul2/tokenizer_config.json",
];

/// SA3's legacy Hugging Face repository ids from its manifest block, or just
/// those with `role`.
pub fn legacy_repos(role: Option<&str>) -> Vec<&'static str> {
    target()
        .def
        .legacy
        .hf_repos
        .iter()
        .filter(|repo| role.is_none() || repo.role.as_deref() == role)
        .map(|repo| repo.repo.as_str())
        .collect()
}

pub fn legacy_training_base_present(active_root: &Path, hf_hub: &Path) -> bool {
    crate::migration_inventory::role_present(target(), active_root, hf_hub, "training-base")
}

#[cfg(test)]
pub fn legacy_runtime_present(active_root: &Path, hf_hub: &Path) -> bool {
    crate::migration_inventory::legacy_present(target(), active_root, hf_hub)
}

#[cfg(test)]
pub fn preview(active_root: &Path, hf_hub: &Path) -> Sa3MigrationPreview {
    crate::migration_inventory::preview(target(), active_root, hf_hub, extras(active_root, hf_hub))
}

/// See `migration_inventory::recheck`.
#[cfg(test)]
pub(crate) fn recheck(active_root: &Path, hf_hub: &Path, path: &Path) -> Sa3MigrationPreview {
    crate::migration_inventory::recheck(
        target(),
        active_root,
        hf_hub,
        path,
        extras(active_root, hf_hub),
    )
}

/// SA3 as the shared migration engine sees it.
pub struct Sa3;

impl crate::migration::Service for Sa3 {
    fn target(&self) -> crate::migration::Target<'_> {
        target()
    }

    fn extras(&self, root: &Path, hub: &Path) -> Extras {
        extras(root, hub)
    }

    fn models(&self) -> &'static crate::native_models::Catalog {
        crate::sa3_models::models()
    }

    fn tier_components(&self, tier: &str) -> Result<Vec<String>, String> {
        crate::sa3_models::preparation_ids(tier, None)
    }

    fn features(&self) -> &'static [&'static str] {
        &["training-base", "decoder-correction", "adapters"]
    }

    /// While decoder correction is on, the server needs its adapter.
    fn implied_components(&self) -> Vec<String> {
        if crate::sa3_use_decoder_lora_enabled() {
            vec![crate::sa3_decoder::MODEL_ID.into()]
        } else {
            Vec::new()
        }
    }

    /// Training and dataset jobs launch through the same lock, and both read
    /// what migration replaces.
    fn exclusive(&self, cleanup: bool) -> Result<Option<Box<dyn std::any::Any + Send>>, String> {
        let launch = crate::sa3_training::LAUNCH
            .try_lock()
            .map_err(|_| "SA3 training or migration is already starting")?;
        let training = matches!(
            crate::read_sa3_lora_training_state().status.as_str(),
            "starting" | "running"
        );
        if training && !cleanup {
            return Err("Wait for SA3 training to finish before switching its runtime".into());
        }
        if cleanup
            && (training
                || matches!(crate::read_sa3_autolabel_state().status.as_str(), "starting" | "running"))
        {
            return Err("Wait for SA3 training and dataset jobs to finish before cleanup.".into());
        }
        Ok(Some(Box::new(launch)))
    }

    /// The PyTorch training base may only go once its native replacement is
    /// on disk; otherwise retiring Python would end SA3 training entirely.
    fn cleanup_blocker(&self, root: &Path, hub: &Path, downloaded: &[String]) -> Option<String> {
        let legacy = legacy_training_base_present(root, hub)
            || legacy_repos(Some("training-base"))
                .iter()
                .any(|repo| downloaded.iter().any(|id| id == repo));
        (legacy && !native_training_base_present(root)).then(|| {
            "Prepare the C++ training model before removing the existing PyTorch training model."
                .into()
        })
    }

    fn verify<'a>(
        &'a self,
        run: crate::migration::Run<'a>,
    ) -> crate::migration::Proof<'a, crate::migration::Selection> {
        Box::pin(async move {
            let decoder_enabled = crate::sa3_use_decoder_lora_enabled();
            let selection = verify_native(
                run.root,
                run.native,
                run.installed,
                run.launch,
                run.encoding,
                decoder_enabled,
                run.progress,
            )
            .await?;
            if decoder_enabled != crate::sa3_use_decoder_lora_enabled() {
                return Err("Decoder correction changed during migration validation; retry with the current setting.".into());
            }
            Ok(selection)
        })
    }

    /// Activation already proved this runtime: hashed every model, generated
    /// privately and trained two steps. Repeat that only if something it
    /// proved has changed since. The adapter checks always run, because they
    /// are what guarantees nothing still needs the Python environment.
    fn reverify<'a>(
        &'a self,
        run: crate::migration::Run<'a>,
        selection: &'a crate::migration::Selection,
    ) -> crate::migration::Proof<'a, Option<String>> {
        Box::pin(async move {
            let decoder_enabled = crate::sa3_use_decoder_lora_enabled();
            let unchanged = runtime_fingerprint(
                run.root,
                run.native,
                run.installed,
                run.encoding,
                decoder_enabled,
            )
            .is_ok_and(|current| selection.runtime_fingerprint.as_deref() == Some(current.as_str()));
            let fingerprint = if unchanged {
                (run.progress)("The C++ runtime is unchanged since it was verified. Checking adapters...");
                verify_adapters(run.root, &run.progress).await?;
                None
            } else {
                verify_native(
                    run.root,
                    run.native,
                    run.installed,
                    run.launch,
                    run.encoding,
                    decoder_enabled,
                    run.progress,
                )
                .await?
                .runtime_fingerprint
            };
            if decoder_enabled != crate::sa3_use_decoder_lora_enabled() {
                return Err("Decoder correction changed during verification; review cleanup again.".into());
            }
            Ok(fingerprint)
        })
    }

    fn authoritative_records(&self) -> Vec<String> {
        ["sa3/lora_catalog.json", "sa3/lora_registry.json", "sa3/native-loras/catalog.json"]
            .map(String::from)
            .to_vec()
    }

    /// Imported adapter configurations keep their exact content, because the
    /// catalog records their hash: the catalog pointer moves, not the file's
    /// contents. That covers explicit configs, legacy-export configs and the
    /// sidecar beside each source.
    fn opaque_files(&self, root: &Path) -> Result<Vec<String>, String> {
        let catalog = crate::sa3_loras::read_catalog(root)?;
        let decoder = crate::sa3_decoder::read(root)?;
        let mut files = Vec::new();
        for entry in catalog.values().chain(decoder.iter()) {
            let sidecar = Path::new(&entry.source_path).with_extension("json");
            files.extend(entry.config_path.iter().cloned());
            files.extend(entry.legacy_export.iter().map(|export| export.config_path.clone()));
            files.push(sidecar.to_string_lossy().into_owned());
        }
        Ok(files)
    }

    fn validate_records(&self, root: &Path) -> Result<(), String> {
        crate::sa3_loras::read_catalog(root).map(|_| ())
    }

    fn blocking_jobs(&self) -> &'static [&'static str] {
        &[
            "native-migration",
            "native-training",
            "legacy-lora-edit",
            "lora-import",
            "lora-edit",
            "checkpoint-selection",
            "lora-preparation",
            "decoder-preparation",
        ]
    }

    /// The selected tier's models, and the decoder correction when enabled.
    fn native_start_blocker(
        &self,
        root: &Path,
        selection: Option<&crate::migration::Selection>,
    ) -> Option<String> {
        let encoding = selection.map_or("F16", |selection| selection.encoding.as_str());
        if let Some(missing) = crate::sa3_runtime::missing_models(root, encoding) {
            return Some(missing);
        }
        if crate::sa3_use_decoder_lora_enabled() {
            return crate::sa3_decoder::blocker(root);
        }
        None
    }

    /// Gary's SA3 HTTP contract on the public port, translated to the native
    /// server's unified API on its private SA3_PORT.
    fn client_adapter(
        &self,
        public_port: u16,
        env: &[(String, String)],
        root: &Path,
    ) -> Result<Option<Box<dyn crate::migration::ClientAdapter>>, String> {
        let native_port = env
            .iter()
            .find(|(key, _)| key == "SA3_PORT")
            .ok_or("SA3's native server port is missing")?
            .1
            .parse::<u16>()
            .map_err(|_| "SA3's native server port is invalid")?;
        let default_lora = env
            .iter()
            .rev()
            .find(|(key, _)| key == "SA3_DEFAULT_LORA")
            .map(|(_, value)| value.clone());
        let listener = crate::sa3_adapter::AdapterListener::bind(
            public_port,
            native_port,
            root.join("sa3/native-inputs"),
            crate::sa3_adapter::client_defaults(&crate::sa3_loudness_env()),
            root.to_path_buf(),
            default_lora,
        )?;
        Ok(Some(Box::new(listener)))
    }
}

/// SA3's contribution to its inventory: its selection, the user data its
/// catalogs, training history and decoder correction point at, and whether a
/// training base exists in either form.
fn extras(root: &Path, hf_hub: &Path) -> Extras {
    let mut protected_paths = Vec::new();
    let mut warnings = Vec::new();
    if let Ok(catalog) =
        crate::read_sa3_lora_catalog_from(&root.join("sa3/lora_catalog.json"))
    {
        for entry in catalog.values() {
            protected_paths.push(PathBuf::from(&entry.path));
            protected_paths.push(PathBuf::from(&entry.path).with_extension("json"));
            if let Some(path) = &entry.prompts_path {
                protected_paths.push(PathBuf::from(path));
            }
            for checkpoint in &entry.training_checkpoints {
                protected_paths.push(PathBuf::from(&checkpoint.path));
                protected_paths.push(PathBuf::from(&checkpoint.path).with_extension("json"));
            }
        }
    } else {
        warnings.push("Cannot read the legacy SA3 LoRA catalog; cleanup cannot safely account for its original adapters.".into());
    }
    if let Ok(catalog) = crate::sa3_loras::read_catalog(root) {
        for entry in catalog.values() {
            protected_paths.push(PathBuf::from(&entry.source_path));
            if let Some(path) = &entry.prompts_path {
                protected_paths.push(PathBuf::from(path));
            }
            if let Some(path) = &entry.config_path {
                protected_paths.push(PathBuf::from(path));
            }
        }
    } else {
        warnings.push("Cannot read the native SA3 LoRA catalog; cleanup cannot safely account for its original adapters.".into());
    }
    // Jobs may precede their first registered checkpoint. Preserve their
    // original dataset and custom prompt configuration as well as job files.
    let jobs = root.join("sa3/training/jobs");
    if jobs.exists() {
        match std::fs::read_dir(&jobs) {
            Ok(entries) => {
                for entry in entries {
                    let Ok(entry) = entry else {
                        warnings.push("Cannot fully inventory SA3 training history.".into());
                        continue;
                    };
                    if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                        continue;
                    }
                    let options = entry.path().join("native-options.json");
                    if !options.exists() {
                        continue;
                    }
                    if !crate::path_is_inside(&options, &jobs) {
                        warnings.push("SA3 training options redirect outside managed history.".into());
                        continue;
                    }
                    match std::fs::read(&options)
                        .ok()
                        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
                    {
                        Some(value) => {
                            if let Some(dataset) = value["dataset"].as_str() {
                                protected_paths.push(PathBuf::from(dataset));
                            } else {
                                warnings.push(
                                    "SA3 native training history has no dataset path.".into(),
                                );
                            }
                            if let Some(config) = value["promptConfig"].as_str() {
                                protected_paths.push(PathBuf::from(config));
                            }
                        }
                        None => warnings.push(
                            "Cannot safely inspect SA3 native training dataset paths.".into(),
                        ),
                    }
                }
            }
            Err(_) => warnings.push("Cannot safely inspect SA3 training history.".into()),
        }
    }
    match crate::sa3_decoder::read(root) {
        Ok(Some(entry)) => {
            protected_paths.push(PathBuf::from(&entry.source_path));
            if let Some(path) = &entry.prompts_path {
                protected_paths.push(PathBuf::from(path));
            }
            if let Some(path) = entry.config_path {
                protected_paths.push(PathBuf::from(path));
            }
        }
        Err(error) => warnings.push(format!("Cannot safely inspect decoder correction: {error}")),
        Ok(None) => {}
    }
    Extras {
        selection: crate::sa3_runtime::read(root),
        protected: protected_paths,
        preserved: vec![(
            "Decoder LoRA source for conversion".into(),
            hf_hub.join("models--thepatch--same-l-decoder-lora"),
        )],
        warnings,
        training_base_present: native_training_base_present(root)
            || legacy_training_base_present(root, hf_hub),
    }
}

/// Verify the actual installed tools and selected weights while Python remains
/// selected. The probe owns a private server; no client port or old file changes.
pub async fn verify_native(
    root: &Path,
    native: &crate::manifest::NativeDef,
    installed: &crate::native_runtime::NativeInstall,
    launch: &crate::native_runtime::NativeLaunch,
    encoding: &str,
    decoder_enabled: bool,
    progress: impl Fn(&str) + Sync,
) -> Result<crate::sa3_runtime::Selection, String> {
    use serde_json::{json, Value};
    use std::time::{Duration, Instant};
    // Taken before any check, so a file that changes while this runs leaves a
    // fingerprint that no longer matches, never one that wrongly does.
    let fingerprint = runtime_fingerprint(root, native, installed, encoding, decoder_enabled)?;
    let runtime_path = crate::native_runtime::path_with_runtimes(root, &installed.runtimes);
    progress("Checking native server capabilities...");
    crate::sa3_runtime::probe_server(
        &installed.dir.join(&native.executable),
        runtime_path.as_deref(),
    )
    .await?;
    progress("Checking native trainer capabilities...");
    crate::sa3_training::probe(
        &installed.dir.join("sa3-train.exe"),
        runtime_path.as_deref(),
    )
    .await?;
    progress("Checking native audio analysis capabilities...");
    crate::sa3_analysis::probe(&installed.dir.join("sa3-audio-analyze.exe")).await?;
    let models = crate::sa3_models::models_dir(root);
    for id in crate::sa3_models::preparation_ids(encoding, None)? {
        let component = crate::sa3_models::component(&id).unwrap();
        if !crate::sa3_models::present(component, &models) {
            return Err(format!("Prepare {} before migration", component.label));
        }
        for file in &component.files {
            progress(&format!("Verifying {}...", file.filename));
            if crate::native_runtime::sha256_file(&models.join(&file.filename)).await?
                != file.sha256
            {
                return Err(format!(
                    "Native model {} failed verification; prepare it again",
                    file.filename
                ));
            }
        }
    }
    verify_adapters(root, &progress).await?;
    progress("Starting the private native test server...");
    let dir = crate::sa3_training::checked_folder(root, &["sa3", "migration-checks"])?;
    let server = crate::migration_verify::PrivateServer::start(
        launch,
        &dir,
        "SA3",
        crate::sa3_runtime::REQUIRED_CAPABILITIES,
    )
    .await?;
    let client = &server.client;
    let base = &server.base;
    progress("Loading the selected native model...");
    client
        .post(format!("{base}/load"))
        .timeout(Duration::from_secs(180))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("Native SA3 model load failed: {error}"))?;
    client
        .get(format!("{base}/ready"))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("Native SA3 model is not ready: {error}"))?;
    progress("Testing native generation...");
    let mut request = json!({"prompt":"a soft instrumental tone","duration":0.25,"crop_duration":0.25,"steps":1,"seed":42,"keep_models":false,"loras":[]});
    crate::sa3_decoder::append_request(root, decoder_enabled, &mut request).await?;
    let submitted: Value = client
        .post(format!("{base}/generate"))
        .json(&request)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    let id = submitted["session_id"]
        .as_str()
        .filter(|id| {
            !id.is_empty()
                && id
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '-')
        })
        .ok_or("Native SA3 returned an invalid migration-check job")?;
    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        let completed: Value = client
            .get(format!("{base}/poll_status/{id}"))
            .send()
            .await
            .map_err(|error| error.to_string())?
            .error_for_status()
            .map_err(|error| error.to_string())?
            .json()
            .await
            .map_err(|error| error.to_string())?;
        match completed["status"].as_str() {
            Some("completed") => {
                use base64::Engine;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(
                        completed["audio_data"]
                            .as_str()
                            .ok_or("Native SA3 check returned no audio")?,
                    )
                    .map_err(|error| error.to_string())?;
                let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
                    .map_err(|error| error.to_string())?;
                if reader.spec().sample_rate != 44100
                    || reader.spec().channels != 2
                    || reader.duration() != 11025
                {
                    return Err("Native SA3 check returned unexpected audio geometry".into());
                }
                break;
            }
            Some("failed") => {
                return Err(format!(
                    "Native SA3 generation check failed: {}",
                    completed["error"]
                ))
            }
            _ => {}
        }
        if Instant::now() >= deadline {
            return Err("Native SA3 generation check timed out".into());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    server.stop().await?;
    // A capability flag alone does not prove that this machine can train or
    // publish checkpoints. Exercise a tiny isolated dataset when a base exists.
    for training_encoding in ["Q4_K_M", "F16"] {
        let component =
            crate::sa3_models::component(&format!("sa3-native::medium-base-{training_encoding}"))
                .unwrap();
        if crate::sa3_models::present(component, &models) {
            for file in &component.files {
                progress(&format!("Verifying training model {}...", file.filename));
                if crate::native_runtime::sha256_file(&models.join(&file.filename)).await?
                    != file.sha256
                {
                    return Err("Native training base failed verification; prepare it again".into());
                }
            }
            verify_training(
                root,
                installed,
                runtime_path.as_deref(),
                training_encoding,
                &progress,
            )
            .await?;
            break;
        }
    }
    Ok(crate::sa3_runtime::Selection {
        schema_version: 1,
        encoding: encoding.into(),
        verified_release: installed
            .version
            .clone()
            .unwrap_or_else(|| "development".into()),
        backend: installed.backend.clone(),
        activated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        cleanup_complete: false,
        cleanup_errors: Vec::new(),
        runtime_fingerprint: Some(fingerprint),
    })
}

/// Everything that must hold before Python can go and that the runtime proof
/// does not cover: every prepared adapter still matches its source, and every
/// legacy .ckpt already has the export only the Python environment can make.
/// Cleanup always runs this, even when it skips the runtime proof.
pub async fn verify_adapters(root: &Path, progress: &impl Fn(&str)) -> Result<(), String> {
    for entry in crate::sa3_loras::state(root)?.entries {
        progress(&format!("Verifying LoRA '{}'...", entry.name));
        let path = entry
            .native_path
            .as_deref()
            .filter(|_| entry.error.is_none())
            .ok_or_else(|| {
                format!(
                    "Prepare LoRA '{}' before migration: {}",
                    entry.name,
                    entry.error.as_deref().unwrap_or("native copy missing")
                )
            })?;
        if crate::native_runtime::sha256_file(Path::new(path))
            .await?
            .as_str()
            != entry.native_sha256.as_deref().unwrap_or("")
            || crate::native_runtime::sha256_file(Path::new(&entry.source_path))
                .await?
                .as_str()
                != entry.source_sha256.as_deref().unwrap_or("")
        {
            return Err(format!(
                "LoRA '{}' changed since preparation; prepare it again",
                entry.name
            ));
        }
        crate::sa3_loras::verify_legacy_export(root, &entry).await?;
        crate::sa3_loras::verify_source_config(&entry).await?;
    }
    crate::sa3_loras::verify_legacy_history(root).await
}

/// Size and modification time of everything `verify_native` proved the C++
/// runtime against: its executables, the selected generation tier, any
/// training base and the decoder correction, plus release, backend and
/// settings. Metadata only, so recomputing it hashes nothing. Cleanup compares
/// it with the one recorded at activation and repeats the full proof (model
/// hashing, a private generation and a training run) only when it differs.
pub fn runtime_fingerprint(
    root: &Path,
    native: &crate::manifest::NativeDef,
    installed: &crate::native_runtime::NativeInstall,
    encoding: &str,
    decoder_enabled: bool,
) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    fn stamp(path: &Path) -> (String, Option<(u64, u128)>) {
        let metadata = std::fs::metadata(path).ok().and_then(|metadata| {
            let modified = metadata.modified().ok()?;
            let nanos = modified.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
            Some((metadata.len(), nanos))
        });
        (path.to_string_lossy().into_owned(), metadata)
    }
    let mut files = Vec::new();
    for executable in [native.executable.as_str(), "sa3-train.exe", "sa3-audio-analyze.exe"] {
        files.push(stamp(&installed.dir.join(executable)));
    }
    let models = crate::sa3_models::models_dir(root);
    let mut ids = crate::sa3_models::preparation_ids(encoding, None)?;
    ids.extend(["Q4_K_M", "F16"].map(|tier| format!("sa3-native::medium-base-{tier}")));
    for id in ids {
        let component = crate::sa3_models::component(&id).ok_or("Unknown SA3 model")?;
        for file in &component.files {
            files.push(stamp(&models.join(&file.filename)));
        }
    }
    let decoder = if decoder_enabled {
        crate::sa3_decoder::read(root)?.map(|entry| {
            let native = entry.native_path.as_deref().map(|path| stamp(Path::new(path)));
            let source = stamp(Path::new(&entry.source_path));
            (source, native, entry.source_sha256, entry.native_sha256)
        })
    } else {
        None
    };
    let identity = (
        1u32,
        &installed.version,
        &installed.backend,
        encoding,
        decoder_enabled,
        crate::sa3_decoder::custom_path().is_some(),
        files,
        decoder,
    );
    let bytes = serde_json::to_vec(&identity).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

async fn verify_training(
    root: &Path,
    installed: &crate::native_runtime::NativeInstall,
    runtime_path: Option<&std::ffi::OsStr>,
    encoding: &str,
    progress: &impl Fn(&str),
) -> Result<(), String> {
    use std::io::Read;
    let checks = crate::sa3_training::checked_folder(root, &["sa3", "migration-checks"])?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = checks.join(format!("training-{nonce}"));
    std::fs::create_dir(&dir).map_err(|error| error.to_string())?;
    let dataset = dir.join("dataset");
    std::fs::create_dir(&dataset).map_err(|error| error.to_string())?;
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = hound::WavWriter::create(dataset.join("check.wav"), spec)
        .map_err(|error| error.to_string())?;
    for frame in 0..44100 {
        let sample =
            ((frame as f64 * 220.0 * std::f64::consts::TAU / 44100.0).sin() * 1600.0) as i16;
        for _ in 0..2 {
            wav.write_sample(sample)
                .map_err(|error| error.to_string())?;
        }
    }
    wav.finalize().map_err(|error| error.to_string())?;
    std::fs::write(dataset.join("check.txt"), "a quiet synthetic tone")
        .map_err(|error| error.to_string())?;
    let output = dir.join("run");
    let status = dir.join("progress.json");
    let models = crate::sa3_models::models_dir(root);
    let log_path = dir.join("training.log");
    let log = std::fs::File::create(&log_path).map_err(|error| error.to_string())?;
    let mut command = tokio::process::Command::new(installed.dir.join("sa3-train.exe"));
    command
        .args([
            "--model",
            "medium",
            "--device",
            &installed.backend,
            "--encoding",
            encoding,
            "--t5-encoding",
            "f16",
            "--ae-encoding",
            "f32",
            "--steps",
            "2",
            "--rank",
            "16",
            "--adapter-type",
            "dora-rows",
            "--batch-size",
            "1",
            "--checkpoint-every",
            "1",
            "--duration",
            "1",
            "--lora-scope",
            "core",
            "--target-latent-rms",
            "0",
        ])
        .arg("--dataset")
        .arg(&dataset)
        .arg("--models-dir")
        .arg(&models)
        .arg("--out")
        .arg(&output)
        .arg("--progress-file")
        .arg(&status)
        .current_dir(&installed.dir)
        .stdout(log.try_clone().map_err(|error| error.to_string())?)
        .stderr(log)
        .kill_on_drop(true);
    for (flag, id, needle) in [
        ("--tok", "sa3-native::text".to_string(), "vocab"),
        ("--t5", "sa3-native::text".to_string(), "encoder"),
        ("--same", "sa3-native::medium-decoder".to_string(), "same-l"),
        (
            "--cond",
            "sa3-native::medium-decoder".to_string(),
            "conditioner",
        ),
        (
            "--dit",
            format!("sa3-native::medium-base-{encoding}"),
            "dit",
        ),
    ] {
        let component =
            crate::sa3_models::component(&id).ok_or("Missing training check component")?;
        let file = component
            .files
            .iter()
            .find(|file| file.filename.contains(needle))
            .ok_or("Missing training check file")?;
        command.arg(flag).arg(models.join(&file.filename));
    }
    if let Some(path) = runtime_path {
        command.env("PATH", path);
    }
    crate::workload_job::configure_tokio_command(&mut command);
    progress("Testing native training and checkpoint saving with a synthetic sample...");
    let mut child = command.spawn().map_err(|error| error.to_string())?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
    let mut last_message = String::new();
    let exit = loop {
        if let Some(exit) = child.try_wait().map_err(|error| error.to_string())? {
            break exit;
        }
        if let Ok(state) = crate::sa3_training::read_progress(&status) {
            let message = format!(
                "Checking training: {} (step {}/2)...",
                state.phase, state.step
            );
            if message != last_message {
                progress(&message);
                last_message = message;
            }
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "Native training check timed out; inspect {}",
                log_path.display()
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    };
    if !exit.success() {
        return Err(format!(
            "Native training check failed with {exit}; inspect {}",
            log_path.display()
        ));
    }
    let state = crate::sa3_training::read_progress(&status)?;
    let metrics: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&status).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if state.status != "completed"
        || state.step != 2
        || !metrics["loss"]
            .as_f64()
            .is_some_and(|loss| loss.is_finite() && loss >= 0.0)
    {
        return Err("Native training check did not finish two finite updates".into());
    }
    let boundary = output.canonicalize().map_err(|error| error.to_string())?;
    for file in [
        &state.final_adapter,
        &state.adapter_checkpoint,
        &state.state_checkpoint,
    ] {
        let path = Path::new(file)
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if path.parent() != Some(boundary.as_path()) {
            return Err("Training check checkpoint escaped its private output".into());
        }
        let mut header = [0u8; 4];
        std::fs::File::open(&path)
            .and_then(|mut file| file.read_exact(&mut header))
            .map_err(|error| error.to_string())?;
        if &header != b"GGUF" {
            return Err("Training check did not save valid GGUF checkpoints".into());
        }
    }
    // Successful checks must not accumulate hundreds of MB of throwaway LoRAs.
    // Keep the log and progress report; retain all artifacts when validation fails.
    let private_boundary = dir.canonicalize().map_err(|error| error.to_string())?;
    for temporary in [&dataset, &output] {
        let resolved = temporary
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if resolved.parent() != Some(private_boundary.as_path()) {
            return Err("Training check cleanup escaped its private folder".into());
        }
        std::fs::remove_dir_all(&resolved)
            .map_err(|error| format!("Cannot remove temporary training check files: {error}"))?;
    }
    progress("Native training, progress updates and checkpoint saving passed.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::preview;
    use std::path::{Path, PathBuf};

    #[tokio::test]
    #[ignore = "runs the automatic two-step migration training check on real CUDA models"]
    async fn real_automatic_training_check() {
        let root = PathBuf::from(std::env::var_os("GARY4LOCAL_AUTOCHECK_ROOT").unwrap());
        assert!(
            root.to_string_lossy().contains("artifacts"),
            "Use an isolated artifact profile"
        );
        let installed = crate::native_runtime::NativeInstall {
            dir: PathBuf::from(std::env::var_os("GARY4LOCAL_AUTOCHECK_NATIVE").unwrap()),
            backend: "cuda".into(),
            runtimes: vec!["cudart-12.8".into()],
            version: Some("v0.1.1".into()),
            fallback_reason: None,
        };
        let runtime_path = crate::native_runtime::path_with_runtimes(&root, &installed.runtimes);
        for encoding in ["Q4_K_M", "F16"] {
            super::verify_training(
                &root,
                &installed,
                runtime_path.as_deref(),
                encoding,
                &|message| println!("{encoding}: {message}"),
            )
            .await
            .unwrap();
        }
    }

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(
                std::env::temp_dir()
                    .join(format!("gary-sa3-migration-{}-{nonce}", std::process::id())),
            )
        }
        fn write(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"keep").unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn sa3_manifest_declares_exactly_the_footprint_its_tests_expect() {
        // Cleanup removes only what the manifest's legacy block names. These
        // are the paths SA3's cleanup tests were written against; a manifest
        // edit that widens or renames any of them must be deliberate.
        let target = super::target();
        let legacy = &target.def.legacy;
        assert_eq!(target.def.data_dir, "sa3");
        assert_eq!(target.models_dir, Some("sa3"));
        assert_eq!(target.label(), "SA3");
        assert_eq!(legacy.env_dirs, ["env", ".venv"]);
        let repos: Vec<_> = legacy
            .hf_repos
            .iter()
            .map(|repo| (repo.repo.as_str(), repo.role.as_deref()))
            .collect();
        assert_eq!(
            repos,
            [
                ("stabilityai/stable-audio-3-medium", None),
                ("stabilityai/stable-audio-3-medium-base", Some("training-base")),
            ]
        );
        let staged = legacy.staged.as_ref().unwrap();
        assert_eq!(staged.dir, super::STAGED_TRAINING_BASE);
        assert_eq!(staged.files, super::STAGED_TRAINING_FILES);
        assert_eq!(staged.role.as_deref(), Some("training-base"));
        assert_eq!(
            legacy.code.packages,
            ["stable_audio_3", "underfit", "dataset_processing", "tests"]
        );
        assert_eq!(legacy.code.files.len(), 11);
        assert!(legacy.code.files.iter().all(|file| !file.contains('/')));
    }

    #[test]
    fn runtime_fingerprint_changes_only_with_what_verification_proved() {
        use super::runtime_fingerprint;
        let fixture = Fixture::new();
        let native: crate::manifest::NativeDef =
            serde_json::from_value(serde_json::json!({"executable": "sa3-server.exe"})).unwrap();
        for executable in ["sa3-server.exe", "sa3-train.exe", "sa3-audio-analyze.exe"] {
            fixture.write(&format!("services/sa3/native/{executable}"));
        }
        let installed = crate::native_runtime::NativeInstall {
            dir: fixture.0.join("services/sa3/native"),
            backend: "cuda".into(),
            runtimes: vec!["cudart-12.8".into()],
            version: Some("v0.1.3".into()),
            fallback_reason: None,
        };
        let models = crate::sa3_models::models_dir(&fixture.0);
        let tier = crate::sa3_models::component("sa3-native::medium-F16").unwrap();
        let model = models.join(&tier.files[0].filename);
        fixture.write(&model.strip_prefix(&fixture.0).unwrap().to_string_lossy());
        let fingerprint = |encoding: &str, decoder: bool| {
            runtime_fingerprint(&fixture.0, &native, &installed, encoding, decoder).unwrap()
        };
        let verified = fingerprint("F16", false);

        assert_eq!(verified, fingerprint("F16", false), "unchanged files must match");
        assert_ne!(verified, fingerprint("Q4_K_M", false), "another tier is another proof");
        assert_ne!(verified, fingerprint("F16", true), "the decoder setting is part of it");
        std::fs::write(&model, b"a replaced, longer model file").unwrap();
        assert_ne!(verified, fingerprint("F16", false), "a replaced model must re-verify");
        let replaced = fingerprint("F16", false);
        let base = crate::sa3_models::component("sa3-native::medium-base-Q4_K_M").unwrap();
        let base = models.join(&base.files[0].filename);
        fixture.write(&base.strip_prefix(&fixture.0).unwrap().to_string_lossy());
        assert_ne!(replaced, fingerprint("F16", false), "a new training base needs its check");
        let mut updated = installed.clone();
        updated.version = Some("v0.1.4".into());
        assert_ne!(
            fingerprint("F16", false),
            runtime_fingerprint(&fixture.0, &native, &updated, "F16", false).unwrap(),
            "a new release must re-verify"
        );
    }

    #[test]
    fn new_install_has_no_cleanup_candidates_and_creates_nothing() {
        let fixture = Fixture::new();
        let result = preview(&fixture.0, &fixture.0.join("models/huggingface/hub"));
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.estimated_cleanup_bytes, 0);
        assert!(!result.legacy_runtime_present);
        assert!(!fixture.0.exists());
    }

    #[test]
    fn bundled_code_and_native_assets_do_not_require_python_runtime_migration() {
        let fixture = Fixture::new();
        let code = fixture.write("runtime/services/sa3/requirements.txt");
        fixture.write("runtime/services/sa3/native/sa3-server.exe");
        fixture.write("runtime/models/sa3/native.gguf");
        fixture.write("other-profile/services/sa3/env/Scripts/python.exe");
        let root = fixture.0.join("runtime");
        crate::migration_code::record_bundle(crate::sa3_runtime::target(), &root, &root, false).unwrap();
        let result = preview(&root, &root.join("models/huggingface/hub"));
        assert!(!result.legacy_runtime_present);
        assert!(result.warnings.is_empty());
        assert_eq!(result.cleanup_candidates.len(), 1);
        assert_eq!(result.cleanup_candidates[0].kind, Some("code"));
        assert!(code.is_file());
        fixture.write("runtime/services/sa3/.venv/Scripts/python.exe");
        assert!(preview(&root, &root.join("models/huggingface/hub")).legacy_runtime_present);
    }

    #[test]
    fn staged_training_inventory_is_narrow_and_includes_cacheless_training() {
        let fixture = Fixture::new();
        let root = fixture.0.join("custom-runtime");
        let hub = fixture.0.join("external-hub");
        let base = root.join(super::STAGED_TRAINING_BASE);
        for file in super::STAGED_TRAINING_FILES {
            fixture.write(&format!("custom-runtime/{}/{file}", super::STAGED_TRAINING_BASE));
        }
        fixture.write(&format!("custom-runtime/{}/user-model.ckpt", super::STAGED_TRAINING_BASE));
        fixture.write("custom-runtime/sa3/training/jobs/old/checkpoint.safetensors");
        let result = preview(&root, &hub);
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
        assert_eq!(result.cleanup_candidates.len(), 6);
        assert!(result.training_base_present && result.legacy_runtime_present);
        assert_eq!(result.estimated_cleanup_bytes, 24);
        assert!(result.cleanup_candidates.iter().all(|entry| entry.kind == Some("weights") && Path::new(&entry.path).starts_with(&base)));
        assert!(!hub.exists());
        let old_review = result.cleanup_token;
        std::fs::write(base.join("model.safetensors"), b"changed staged file").unwrap();
        assert_ne!(preview(&root, &hub).cleanup_token, old_review);
    }

    #[test]
    fn staged_training_cleanup_preserves_catalog_sources_and_dataset_overlaps() {
        for dataset in [false, true] {
            let fixture = Fixture::new();
            let root = &fixture.0;
            let model = fixture.write(&format!("{}/model.safetensors", super::STAGED_TRAINING_BASE));
            if dataset {
                let options = fixture.write("sa3/training/jobs/native-job/native-options.json");
                std::fs::write(options, serde_json::json!({"dataset":root.join("sa3"),"promptConfig":null}).to_string()).unwrap();
            } else {
                let catalog = fixture.write("sa3/lora_catalog.json");
                std::fs::write(catalog, serde_json::json!({"user":{"path":model,"strength":1}}).to_string()).unwrap();
            }
            let result = preview(root, &root.join("hub"));
            assert!(result.cleanup_candidates.is_empty());
            assert_eq!(result.warnings.len(), 1);
            assert!(model.is_file());
        }
    }

    #[test]
    fn every_other_service_folder_is_protected_including_unlisted_ones() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        // A hub living inside another service's folder, one that no list in
        // this module names. SA3's weights there overlap that service.
        let hub = root.join("services/audiocraft/hub");
        let weights = fixture.write(
            "services/audiocraft/hub/models--stabilityai--stable-audio-3-medium/weights",
        );
        fixture.write("services/sa3/env/python.exe");
        let result = preview(root, &hub);
        assert!(result
            .cleanup_candidates
            .iter()
            .all(|candidate| !Path::new(&candidate.path).starts_with(root.join("services/audiocraft"))));
        assert!(result.warnings.iter().any(|warning| warning.contains("overlaps")));
        assert!(weights.is_file());
    }

    #[cfg(windows)]
    #[test]
    fn staged_training_cache_junction_is_not_followed_even_within_the_profile() {
        use std::os::windows::process::CommandExt;
        let fixture = Fixture::new();
        let model = fixture.write("runtime/sa3/user-models/model.safetensors");
        let root = fixture.0.join("runtime");
        let base = root.join(super::STAGED_TRAINING_BASE);
        std::fs::create_dir_all(base.parent().unwrap()).unwrap();
        let output = std::process::Command::new("cmd").raw_arg(format!(
            "/c mklink /J \"{}\" \"{}\"", base.display(), model.parent().unwrap().display()
        )).output().unwrap();
        assert!(output.status.success());
        let result = preview(&root, &root.join("hub"));
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.warnings.len(), 1);
        assert!(model.is_file());
    }

    #[test]
    fn custom_storage_only_inventories_sa3_owned_pytorch_paths() {
        let fixture = Fixture::new();
        let env = fixture.write("runtime/services/sa3/env/python.exe");
        let model = fixture.write(
            "runtime/models/huggingface/hub/models--stabilityai--stable-audio-3-medium/weights",
        );
        let keep = [
            fixture.write("runtime/services/stable-audio/env/python.exe"),
            fixture.write("runtime/services/foundation/env/python.exe"),
            fixture.write("runtime/models/huggingface/hub/models--thepatch--same-l-decoder-lora/squeakfix_v3.safetensors"),
            fixture.write("runtime/sa3/loras/custom.safetensors"),
            fixture.write("runtime/sa3/training/jobs/run/checkpoint.safetensors"),
            fixture.write("runtime/native-runtimes/cudart-12.8/cudart64_12.dll"),
            fixture.write("runtime/cache/uv/package"),
            fixture.write("legacy/services/sa3/env/python.exe"),
        ];
        let root = fixture.0.join("runtime");
        let result = preview(&root, &root.join("models/huggingface/hub"));
        assert_eq!(result.cleanup_candidates.len(), 2);
        assert!(result.legacy_runtime_present);
        assert_eq!(result.estimated_cleanup_bytes, 8);
        for path in keep.iter().chain([&env, &model]) {
            assert!(path.is_file());
        }
    }

    #[test]
    fn legacy_storage_uses_the_effective_external_hugging_face_cache() {
        let fixture = Fixture::new();
        let model =
            fixture.write("external-hub/models--stabilityai--stable-audio-3-medium-base/weights");
        fixture.write(
            "legacy/models/huggingface/hub/models--stabilityai--stable-audio-3-medium/unused",
        );
        let result = preview(&fixture.0.join("legacy"), &fixture.0.join("external-hub"));
        assert_eq!(result.cleanup_candidates.len(), 1);
        assert_eq!(
            Path::new(&result.cleanup_candidates[0].path),
            model.parent().unwrap()
        );
        assert!(model.exists());
        assert!(result.legacy_runtime_present);
    }

    #[test]
    fn a_cache_redirected_into_user_artifacts_is_preserved() {
        let fixture = Fixture::new();
        let model = fixture.write("runtime/sa3/models--stabilityai--stable-audio-3-medium/weights");
        let root = fixture.0.join("runtime");
        let result = preview(&root, &root.join("sa3"));
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.warnings.len(), 1);
        assert!(model.exists());
    }

    #[test]
    fn cleanup_preview_preserves_original_loras_inside_an_owned_model_cache() {
        let fixture = Fixture::new();
        let root = fixture.0.join("runtime");
        let hub = root.join("models/huggingface/hub");
        let source = fixture.write("runtime/models/huggingface/hub/models--stabilityai--stable-audio-3-medium/adapter.safetensors");
        let catalog = fixture.write("runtime/sa3/lora_catalog.json");
        std::fs::write(
            &catalog,
            serde_json::json!({"original":{"path":source,"strength":1}}).to_string(),
        )
        .unwrap();
        let result = preview(&root, &hub);
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.warnings.len(), 1);
        assert!(source.is_file());
    }

    #[cfg(windows)]
    #[test]
    fn an_environment_redirected_into_the_native_bundle_is_preserved() {
        use std::os::windows::process::CommandExt;
        let fixture = Fixture::new();
        let native = fixture.write("runtime/services/sa3/native/sa3-server.exe");
        let root = fixture.0.join("runtime");
        let env = root.join("services/sa3/env");
        let output = std::process::Command::new("cmd")
            .raw_arg(format!(
                "/c mklink /J \"{}\" \"{}\"",
                env.to_string_lossy().replace('/', "\\"),
                native
                    .parent()
                    .unwrap()
                    .to_string_lossy()
                    .replace('/', "\\")
            ))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result = preview(&root, &root.join("models/huggingface/hub"));
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.warnings.len(), 1);
        assert!(native.is_file());
        assert!(result.legacy_runtime_present);
    }

    #[tokio::test]
    #[ignore = "verifies compatible real tools and existing model GGUFs before profile activation"]
    async fn real_native_runtime_verification_keeps_python_until_activation() {
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_MIGRATION_SMOKE_ROOT")
                .expect("new isolated root required"),
        );
        let packaged = std::env::var("GARY4LOCAL_SA3_SMOKE_USE_INSTALLED_PACKAGE")
            .is_ok_and(|value| value == "1");
        if packaged {
            assert!(root.join("services/sa3/native/gary-native.json").is_file());
            assert!(!crate::sa3_runtime::selection_path(&root).exists());
            assert!(!root.join("services/sa3/env").exists());
            assert!(std::env::var_os("GARY4LOCAL_NATIVE_DIR_SA3").is_none());
        } else {
            assert!(!root.exists());
        }
        let source_models = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_SMOKE_MODELS").expect("model folder required"),
        );
        let server = if packaged {
            root.join("services/sa3/native/sa3-server.exe")
        } else {
            PathBuf::from(
                std::env::var_os("GARY4LOCAL_SA3_SMOKE_BINARY")
                    .expect("compatible server required"),
            )
        };
        let target = crate::sa3_models::checked_models_dir(&root).unwrap();
        for id in crate::sa3_models::preparation_ids("F16", None).unwrap() {
            for file in &crate::sa3_models::component(&id).unwrap().files {
                std::fs::hard_link(
                    source_models.join(&file.filename),
                    target.join(&file.filename),
                )
                .unwrap();
            }
        }
        let original = root.join("services/sa3/env/Scripts/python.exe");
        std::fs::create_dir_all(original.parent().unwrap()).unwrap();
        std::fs::write(&original, b"preserved Python environment").unwrap();
        let mut manifest: crate::manifest::Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .unwrap();
        manifest.resolve_native_bundles().unwrap();
        let defs = manifest.services;
        let native = defs
            .iter()
            .find(|service| service.id == "sa3")
            .unwrap()
            .native
            .clone()
            .unwrap();
        let installed = if packaged {
            crate::native_runtime::installed("sa3", server.parent().unwrap(), &native.executable)
                .expect("the production installer must have stamped the package")
        } else {
            crate::native_runtime::NativeInstall {
                dir: server.parent().unwrap().into(),
                backend: "cuda".into(),
                runtimes: Vec::new(),
                version: None,
                fallback_reason: None,
            }
        };
        let decoder_enabled =
            std::env::var("GARY4LOCAL_SA3_SMOKE_DECODER").is_ok_and(|value| value == "1");
        if decoder_enabled {
            let source = PathBuf::from(
                std::env::var_os("GARY4LOCAL_SA3_DECODER_SOURCE")
                    .expect("pinned decoder source required"),
            );
            std::fs::copy(source, crate::sa3_decoder::source_path(&root)).unwrap();
            assert!(
                crate::sa3_decoder::prepare(
                    &root,
                    &installed.dir.join("sa3-lora-convert.exe"),
                    None
                )
                .await
                .unwrap()
                .prepared
            );
        }
        let launch = if packaged {
            let manager = crate::service_manager::ServiceManager::new(defs.clone(), root.clone());
            manager.migration_launch("sa3", "F16").unwrap().2
        } else {
            crate::native_runtime::NativeLaunch::new(
                &root,
                &native.executable,
                &installed,
                crate::migration::Service::launch_args(&crate::sa3_migration::Sa3, &native.args, Some("F16")).unwrap(),
                Vec::new(),
                &[("SA3_MODELS_DIR".into(), target.to_string_lossy().into())],
            )
        };
        let selection = super::verify_native(
            &root,
            &native,
            &installed,
            &launch,
            "F16",
            decoder_enabled,
            |message| println!("{message}"),
        )
        .await
        .unwrap();
        assert!(
            !crate::sa3_runtime::selection_path(&root).exists(),
            "verification must not commit the runtime choice"
        );
        let mut manager = crate::service_manager::ServiceManager::new(defs.clone(), root.clone());
        assert!(!manager.is_native("sa3"));
        manager.activate_native("sa3", selection).unwrap();
        assert!(manager.is_native("sa3"));
        assert!(crate::service_manager::ServiceManager::new(defs, root.clone()).is_native("sa3"));
        assert_eq!(
            std::fs::read(&original).unwrap(),
            b"preserved Python environment"
        );
        println!("PASS native trainer/server/model verification and persistent profile activation; Python preserved");
        if std::env::var("GARY4LOCAL_SA3_SMOKE_CLEANUP").is_ok_and(|value| value == "1") {
            let hub = root.join("models/huggingface/hub");
            let legacy = hub.join("models--stabilityai--stable-audio-3-medium/weights.fixture");
            std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
            std::fs::write(&legacy, b"owned legacy weight fixture").unwrap();
            let mut copied_code = Vec::new();
            if let Some(bundle) = std::env::var_os("GARY4LOCAL_SA3_SMOKE_CODE_BUNDLE") {
                let bundle = PathBuf::from(bundle);
                crate::migration_code::record_bundle(crate::sa3_runtime::target(), &bundle, &root, false).unwrap();
                for dest in crate::migration_code::recorded_paths(crate::sa3_runtime::target(), &root).unwrap() {
                    let relative = dest.strip_prefix(&root).unwrap();
                    let source = bundle.join(relative);
                    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
                    std::fs::copy(&source, &dest).unwrap();
                    copied_code.push((source, dest));
                }
                assert!(!copied_code.is_empty());
                assert!(crate::migration_code::status(crate::sa3_runtime::target(), &root, None).warnings.is_empty());
            }
            let reviewed = super::preview(&root, &hub);
            assert!(reviewed.warnings.is_empty(), "{:?}", reviewed.warnings);
            let result = crate::sa3_cleanup::run(&root, &hub, &reviewed.cleanup_token, |_| {}).unwrap();
            assert!(result.selection.cleanup_complete);
            assert_eq!(result.removed_paths.len(), 2 + copied_code.len());
            for (source, dest) in &copied_code {
                assert!(source.is_file(), "original repository source must survive");
                assert!(!dest.exists(), "owned installed source must retire");
            }
            assert!(!original.exists());
            assert!(!legacy.exists());
            assert!(crate::sa3_models::present(
                crate::sa3_models::component("sa3-native::medium-F16").unwrap(),
                &target
            ));
            if decoder_enabled {
                crate::sa3_decoder::verified_path(&root).await.unwrap();
            }
            manager.refresh_migration("sa3").unwrap();
            assert!(manager.is_native("sa3"));
            println!("PASS reviewed fixture cleanup after real native verification; models, decoder and native selection preserved");
            println!("PASS {} checksummed installed Python source files retired; original repository sources retained", copied_code.len());
        }
    }

    #[tokio::test]
    #[ignore = "launches an activated isolated profile through ServiceManager with an installed package or developer override"]
    async fn real_native_selected_profile_launches_through_service_manager() {
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_MIGRATION_SMOKE_ROOT")
                .expect("existing isolated activation profile required"),
        );
        assert!(crate::sa3_runtime::read(&root).unwrap().is_some());
        assert!(
            std::env::var_os("GARY4LOCAL_NATIVE_DIR_SA3").is_some()
                || root.join("services/sa3/native/gary-native.json").is_file()
        );
        let public = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let private = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let public_port = public.local_addr().unwrap().port();
        let private_port = private.local_addr().unwrap().port();
        let mut manifest: crate::manifest::Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .unwrap();
        manifest.resolve_native_bundles().unwrap();
        let sa3 = manifest
            .services
            .iter_mut()
            .find(|service| service.id == "sa3")
            .unwrap();
        sa3.port = public_port;
        sa3.native
            .as_mut()
            .unwrap()
            .env
            .insert("SA3_PORT".into(), private_port.to_string());
        let mut manager =
            crate::service_manager::ServiceManager::new(manifest.services, root.clone());
        manager.set_native_runtimes(manifest.native_runtimes);
        assert!(manager.is_native("sa3"));
        drop(public);
        drop(private);
        let default_prompts = root.join("sa3/prompts/defaults.json");
        let previous_defaults = std::fs::read(&default_prompts).ok();
        manager.start("sa3").unwrap();
        let defaults: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&default_prompts).unwrap()).unwrap();
        if let Some(previous) = previous_defaults {
            assert_eq!(std::fs::read(&default_prompts).unwrap(), previous);
        } else {
            assert!(!defaults["dice"]["generic"].as_array().unwrap().is_empty());
        }
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .unwrap();
        let base = format!("http://127.0.0.1:{public_port}");
        let checked: Result<serde_json::Value, String> = async {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                if client
                    .get(format!("{base}/health"))
                    .send()
                    .await
                    .is_ok_and(|response| response.status().is_success())
                {
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    return Err("managed native service did not start".into());
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            client
                .post(format!("{base}/load"))
                .timeout(std::time::Duration::from_secs(180))
                .send()
                .await
                .map_err(|error| error.to_string())?
                .error_for_status()
                .map_err(|error| error.to_string())?;
            client
                .get(format!("{base}/ready"))
                .send()
                .await
                .map_err(|error| error.to_string())?
                .error_for_status()
                .map_err(|error| error.to_string())?;
            client
                .get(format!("{base}/health"))
                .send()
                .await
                .map_err(|error| error.to_string())?
                .json()
                .await
                .map_err(|error| error.to_string())
        }
        .await;
        manager.stop("sa3").unwrap();
        let health = checked.unwrap();
        assert_eq!(health["encoding"], "f16");
        assert_eq!(health["model_loaded"], true);
        let python = root.join("services/sa3/env/Scripts/python.exe");
        if crate::sa3_runtime::read(&root)
            .unwrap()
            .unwrap()
            .cleanup_complete
        {
            assert!(
                !python.exists(),
                "cleaned native profile must launch without Python"
            );
        } else {
            assert_eq!(
                std::fs::read(python).unwrap(),
                b"preserved Python environment"
            );
        }
        println!("PASS selected native profile started/stopped through production ServiceManager and public adapter load/readiness; legacy environment state preserved");
    }
}
