//! Inventory and native validation for SA3's move from PyTorch.
//! Verification and activation preserve old files. Cleanup remains a separate
//! transaction after runtime and adapter requirements have been satisfied.

use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationItem {
    pub label: String,
    pub path: String,
    pub bytes: u64,
    pub kind: Option<&'static str>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sa3MigrationPreview {
    pub active_root: String,
    pub hf_hub_cache: String,
    pub cleanup_candidates: Vec<MigrationItem>,
    pub estimated_cleanup_bytes: u64,
    pub preserved_paths: Vec<MigrationItem>,
    pub warnings: Vec<String>,
    pub native_selection: Option<crate::sa3_runtime::Selection>,
    pub cleanup_token: String,
}

fn item(label: &str, path: &Path, kind: &'static str) -> MigrationItem {
    MigrationItem {
        label: label.to_string(),
        path: path.to_string_lossy().to_string(),
        bytes: crate::path_size(path),
        kind: Some(kind),
    }
}

pub fn preview(active_root: &Path, hf_hub: &Path) -> Sa3MigrationPreview {
    let mut result = Sa3MigrationPreview {
        active_root: active_root.to_string_lossy().to_string(),
        hf_hub_cache: hf_hub.to_string_lossy().to_string(),
        cleanup_candidates: Vec::new(),
        estimated_cleanup_bytes: 0,
        preserved_paths: Vec::new(),
        warnings: Vec::new(),
        native_selection: None,
        cleanup_token: String::new(),
    };
    match crate::sa3_runtime::read(active_root) {
        Ok(selection) => result.native_selection = selection,
        Err(error) => result.warnings.push(error),
    }
    let service = active_root.join("services").join("sa3");
    let mut candidates = vec![
        (
            "SA3 Python environment",
            service.join("env"),
            service.clone(),
        ),
        (
            "SA3 alternate Python environment",
            service.join(".venv"),
            service.clone(),
        ),
        (
            "SA3 PyTorch inference weights",
            hf_hub.join("models--stabilityai--stable-audio-3-medium"),
            hf_hub.to_path_buf(),
        ),
        (
            "SA3 PyTorch training base",
            hf_hub.join("models--stabilityai--stable-audio-3-medium-base"),
            hf_hub.to_path_buf(),
        ),
    ];
    let code = crate::sa3_code::status(active_root);
    result.warnings.extend(code.warnings);
    for path in &code.candidates {
        candidates.push(("SA3 bundled Python source", path.clone(), service.clone()));
    }
    for path in &code.retained {
        result.preserved_paths.push(MigrationItem {
            label: "Edited Python code or developer checkout (preserved)".into(),
            path: path.to_string_lossy().into(),
            bytes: 0,
            kind: None,
        });
    }
    let protected = active_root.join("sa3");
    let mut protected_paths = vec![
        protected.clone(),
        active_root.join("models/sa3"),
        active_root.join("native-runtimes"),
        service.join("native"),
    ];
    if let Ok(catalog) =
        crate::read_sa3_lora_catalog_from(&active_root.join("sa3/lora_catalog.json"))
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
        result.warnings.push("Cannot read the legacy SA3 LoRA catalog; cleanup cannot safely account for its original adapters.".into());
    }
    if let Ok(catalog) = crate::sa3_loras::read_catalog(active_root) {
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
        result.warnings.push("Cannot read the native SA3 LoRA catalog; cleanup cannot safely account for its original adapters.".into());
    }
    // Jobs may precede their first registered checkpoint. Preserve their
    // original dataset and custom prompt configuration as well as job files.
    let jobs = active_root.join("sa3/training/jobs");
    if jobs.exists() {
        match std::fs::read_dir(&jobs) {
            Ok(entries) => {
                for entry in entries {
                    let Ok(entry) = entry else {
                        result
                            .warnings
                            .push("Cannot fully inventory SA3 training history.".into());
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
                        result
                            .warnings
                            .push("SA3 training options redirect outside managed history.".into());
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
                                result.warnings.push(
                                    "SA3 native training history has no dataset path.".into(),
                                );
                            }
                            if let Some(config) = value["promptConfig"].as_str() {
                                protected_paths.push(PathBuf::from(config));
                            }
                        }
                        None => result.warnings.push(
                            "Cannot safely inspect SA3 native training dataset paths.".into(),
                        ),
                    }
                }
            }
            Err(_) => result
                .warnings
                .push("Cannot safely inspect SA3 training history.".into()),
        }
    }
    match crate::sa3_decoder::read(active_root) {
        Ok(Some(entry)) => {
            protected_paths.push(PathBuf::from(&entry.source_path));
            if let Some(path) = &entry.prompts_path {
                protected_paths.push(PathBuf::from(path));
            }
            if let Some(path) = entry.config_path {
                protected_paths.push(PathBuf::from(path));
            }
        }
        Err(error) => result
            .warnings
            .push(format!("Cannot safely inspect decoder correction: {error}")),
        Ok(None) => {}
    }
    for id in [
        "gary",
        "melodyflow",
        "stable-audio",
        "carey",
        "foundation",
        "yuey",
    ] {
        protected_paths.push(active_root.join("services").join(id));
    }
    if let Ok(entries) = std::fs::read_dir(hf_hub) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if ![
                "models--stabilityai--stable-audio-3-medium",
                "models--stabilityai--stable-audio-3-medium-base",
            ]
            .contains(&name.to_string_lossy().as_ref())
            {
                protected_paths.push(entry.path());
            }
        }
    }
    let mut seen = Vec::new();
    for (label, path, owner) in candidates {
        if owner == service && !crate::path_is_inside(&owner, active_root) && path.exists() {
            result.warnings.push(format!(
                "SA3 service storage redirects outside the active profile: {}",
                owner.display()
            ));
            continue;
        }

        match crate::resolve_managed_path(&path, &owner) {
            Ok(Some(canonical)) => {
                if (!code.candidates.contains(&path) && canonical.parent() != owner.canonicalize().ok().as_deref())
                    || canonical.file_name() != path.file_name()
                {
                    result.warnings.push(format!("Preserving redirected SA3 cleanup target {}; review it manually.", path.display()));
                    continue;
                }
                // A redirected cache or environment must never turn the
                // proposed cleanup into removal of user-owned SA3 artifacts.
                if protected_paths.iter().any(|protected| crate::path_is_inside(&canonical, protected)
                    || crate::path_is_inside(protected, &canonical))
                {
                    result.warnings.push(format!(
                        "Preserving {} because it overlaps SA3 user data or native runtime files.", path.display()
                    ));
                } else if !seen.contains(&canonical) {
                    let kind = if code.candidates.contains(&path) { "code" } else if owner == service { "environment" } else { "weights" };
                    result.cleanup_candidates.push(item(label, &path, kind));
                    seen.push(canonical);
                }
            }
            Ok(None) => {}
            Err(_) => result.warnings.push(format!(
                "Cannot safely inventory {} within {}. Review this redirected or inaccessible path manually.",
                path.display(), owner.display()
            )),
        }
    }
    result.estimated_cleanup_bytes = result
        .cleanup_candidates
        .iter()
        .map(|entry| entry.bytes)
        .sum();
    // Do not count these recursively: a LoRA checkpoint can be many GB, and
    // they cannot contribute to the space reclaimed by this migration.
    for (label, path) in [
        ("LoRAs, prompts, training jobs and checkpoints", protected),
        (
            "SA3 native runtime, settings and unrecognized service files",
            service,
        ),
        (
            "Shared CUDA and other native runtimes",
            active_root.join("native-runtimes"),
        ),
        (
            "Native model weights",
            active_root.join("models").join("sa3"),
        ),
        (
            "Decoder LoRA source for conversion",
            hf_hub.join("models--thepatch--same-l-decoder-lora"),
        ),
    ] {
        result.preserved_paths.push(MigrationItem {
            label: label.to_string(),
            path: path.to_string_lossy().to_string(),
            bytes: 0,
            kind: None,
        });
    }
    use sha2::{Digest, Sha256};
    let identities: Vec<_> = result
        .cleanup_candidates
        .iter()
        .map(|entry| (&entry.path, Path::new(&entry.path).canonicalize().ok()))
        .collect();
    result.cleanup_token = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                active_root.canonicalize().ok(),
                hf_hub.canonicalize().ok(),
                identities,
                code.review_identity,
                &result.warnings,
            ))
            .unwrap()
        )
    );
    result
}

/// Verify the actual installed tools and selected weights while Python remains
/// selected. The probe owns a private server; no client port or old file changes.
pub async fn verify_native(
    root: &Path,
    native: &crate::manifest::NativeDef,
    installed: &crate::native_runtime::NativeInstall,
    template: &[(String, String)],
    encoding: &str,
    decoder_enabled: bool,
    progress: impl Fn(&str),
) -> Result<crate::sa3_runtime::Selection, String> {
    use serde_json::{json, Value};
    use std::time::{Duration, Instant};
    let runtime_path = crate::native_runtime::path_with_runtimes(root, &installed.runtimes);
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
    crate::sa3_loras::verify_legacy_history(root).await?;
    progress("Starting the private native test server...");
    let dir = crate::sa3_training::checked_folder(root, &["sa3", "migration-checks"])?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let log = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(format!("server-{nonce}.log")))
        .map_err(|error| error.to_string())?;
    let reservation = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|error| error.to_string())?;
    let port = reservation
        .local_addr()
        .map_err(|error| error.to_string())?
        .port();
    let mut command = tokio::process::Command::new(installed.dir.join(&native.executable));
    crate::hide_console_window(&mut command);
    command
        .args(crate::sa3_runtime::launch_args(&native.args, encoding)?)
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&installed.dir)
        .envs(crate::native_runtime::launch_env(
            template,
            &installed.backend,
        ))
        .stdout(log.try_clone().map_err(|error| error.to_string())?)
        .stderr(log)
        .kill_on_drop(true);
    if let Some(path) = &runtime_path {
        command.env("PATH", path);
    }
    crate::workload_job::configure_tokio_command(&mut command);
    drop(reservation);
    let mut child = command.spawn().map_err(|error| error.to_string())?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    let client = reqwest::Client::builder()
        .no_proxy()
        .pool_max_idle_per_host(0)
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|error| error.to_string())?;
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let health: Value = loop {
        if let Some(exit) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!(
                "Native SA3 migration check exited with {exit}; inspect migration-checks logs"
            ));
        }
        if let Ok(response) = client.get(format!("{base}/health")).send().await {
            if response.status().is_success() {
                break response.json().await.map_err(|error| error.to_string())?;
            }
        }
        if Instant::now() >= deadline {
            return Err(
                "Native SA3 migration check did not start; inspect migration-checks logs".into(),
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    for capability in [
        "fixed_prefix",
        "request_splice",
        "conditioning_duration",
        "model_lifecycle",
    ] {
        if health["capabilities"][capability] != true {
            return Err(format!("Installed SA3 runtime lacks {capability}; prepare a compatible release before migration"));
        }
    }
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
    child.kill().await.map_err(|error| error.to_string())?;
    child.wait().await.map_err(|error| error.to_string())?;
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
    })
}

#[cfg(test)]
mod tests {
    use super::preview;
    use std::path::{Path, PathBuf};

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
    fn new_install_has_no_cleanup_candidates_and_creates_nothing() {
        let fixture = Fixture::new();
        let result = preview(&fixture.0, &fixture.0.join("models/huggingface/hub"));
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.estimated_cleanup_bytes, 0);
        assert!(!fixture.0.exists());
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
    }

    #[tokio::test]
    #[ignore = "verifies compatible real tools and existing model GGUFs before profile activation"]
    async fn real_native_runtime_verification_keeps_python_until_activation() {
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_MIGRATION_SMOKE_ROOT")
                .expect("new isolated root required"),
        );
        assert!(!root.exists());
        let source_models = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_SMOKE_MODELS").expect("model folder required"),
        );
        let server = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_SMOKE_BINARY").expect("compatible server required"),
        );
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
            serde_json::from_str(include_str!("../../../services/manifests/services.json")).unwrap();
        manifest.resolve_native_bundles().unwrap();
        let defs = manifest.services;
        let native = defs
            .iter()
            .find(|service| service.id == "sa3")
            .unwrap()
            .native
            .clone()
            .unwrap();
        let installed = crate::native_runtime::NativeInstall {
            dir: server.parent().unwrap().into(),
            backend: "cuda".into(),
            runtimes: Vec::new(),
            version: None,
            fallback_reason: None,
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
                crate::sa3_decoder::prepare(&root, &installed.dir.join("sa3-lora-convert.exe"), None)
                    .await
                    .unwrap()
                    .prepared
            );
        }
        let env = vec![("SA3_MODELS_DIR".into(), target.to_string_lossy().into())];
        let selection = super::verify_native(
            &root,
            &native,
            &installed,
            &env,
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
        manager.activate_native_sa3(selection).unwrap();
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
                crate::sa3_code::record_bundle(&bundle, &root, false).unwrap();
                for dest in crate::sa3_code::recorded_paths(&root).unwrap() {
                    let relative = dest.strip_prefix(&root).unwrap();
                    let source = bundle.join(relative);
                    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
                    std::fs::copy(&source, &dest).unwrap();
                    copied_code.push((source, dest));
                }
                assert!(!copied_code.is_empty());
                assert!(crate::sa3_code::status(&root).warnings.is_empty());
            }
            let reviewed = super::preview(&root, &hub);
            assert!(reviewed.warnings.is_empty(), "{:?}", reviewed.warnings);
            let result = crate::sa3_cleanup::run(&root, &hub, &reviewed.cleanup_token).unwrap();
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
            manager.refresh_sa3_native_selection().unwrap();
            assert!(manager.is_native("sa3"));
            println!("PASS reviewed fixture cleanup after real native verification; models, decoder and native selection preserved");
            println!("PASS {} checksummed installed Python source files retired; original repository sources retained", copied_code.len());
        }
    }

    #[tokio::test]
    #[ignore = "launches an already activated isolated profile through ServiceManager with a native developer override"]
    async fn real_native_selected_profile_launches_through_service_manager() {
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_MIGRATION_SMOKE_ROOT")
                .expect("existing isolated activation profile required"),
        );
        assert!(crate::sa3_runtime::read(&root).unwrap().is_some());
        assert!(std::env::var_os("GARY4LOCAL_NATIVE_DIR_SA3").is_some());
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
        let defaults: serde_json::Value = serde_json::from_slice(&std::fs::read(&default_prompts).unwrap()).unwrap();
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
