//! Retire reviewed, SA3-owned Python environments, code and weight repositories.
//! The command verifies native execution and reserves the service before entering
//! this transaction. The persisted selection journals incomplete cleanup so an
//! interruption or locked file never causes fallback to Python.
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultInfo {
    pub removed_paths: Vec<String>,
    pub estimated_removed_bytes: u64,
    pub errors: Vec<String>,
    pub selection: crate::sa3_runtime::Selection,
}

/// Returns the inventory it checked, so the caller can act on that same plan.
pub fn validate_review(
    root: &Path,
    hub: &Path,
    token: &str,
) -> Result<crate::sa3_migration::Sa3MigrationPreview, String> {
    let current = crate::sa3_migration::preview(root, hub);
    if current.cleanup_token != token {
        return Err("SA3 storage changed since review. Rescan and review cleanup again.".into());
    }
    if !current.warnings.is_empty() {
        return Err(
            "Resolve the SA3 cleanup inventory warnings before removing Python files.".into(),
        );
    }
    if current.native_selection.is_none() {
        return Err("Verify and select C++ before retiring the SA3 Python installation.".into());
    }
    if crate::sa3_migration::legacy_training_base_present(root, hub)
        && !crate::sa3_migration::native_training_base_present(root)
    {
        return Err("Prepare the C++ training model before removing the existing PyTorch training model.".into());
    }
    Ok(current)
}

pub(crate) fn run(
    root: &Path,
    hub: &Path,
    token: &str,
    progress: impl Fn(&str),
) -> Result<ResultInfo, String> {
    run_with_progress(root, hub, token, crate::remove_managed_path, progress)
}

#[cfg(test)]
fn run_with(
    root: &Path,
    hub: &Path,
    token: &str,
    remove: impl FnMut(&Path, &Path) -> Result<bool, String>,
) -> Result<ResultInfo, String> {
    run_with_progress(root, hub, token, remove, |_| {})
}

fn run_with_progress(
    root: &Path,
    hub: &Path,
    token: &str,
    mut remove: impl FnMut(&Path, &Path) -> Result<bool, String>,
    progress: impl Fn(&str),
) -> Result<ResultInfo, String> {
    progress("Checking the reviewed Python files before cleanup...");
    let plan = validate_review(root, hub, token)?;
    let mut selection = crate::sa3_runtime::read(root)?.ok_or("Native SA3 selection is missing")?;
    // Publish the incomplete state before the first deletion. Following a crash,
    // rescanning discovers remaining paths and retry never changes runtime.
    selection.cleanup_complete = false;
    selection.cleanup_errors =
        vec!["SA3 cleanup was interrupted; review remaining files and retry.".into()];
    crate::sa3_runtime::save(root, &selection)?;
    let mut result = ResultInfo {
        removed_paths: Vec::new(),
        estimated_removed_bytes: 0,
        errors: Vec::new(),
        selection,
    };
    let service = root.join("services/sa3");
    let mut allowed: Vec<(PathBuf, PathBuf)> = crate::sa3_migration::legacy_runtime_candidates(root, hub)
        .into_iter().map(|(_, path, owner)| (path, owner)).collect();
    allowed.extend(crate::sa3_code::recorded_paths(root)?.into_iter().map(|path| (path, service.clone())));
    let identities: Vec<_> = plan
        .cleanup_candidates
        .iter()
        .map(|entry| {
            let path = Path::new(&entry.path);
            path
                .canonicalize()
                .map(|canonical| (canonical, crate::sa3_migration::staged_training_identity(root, path)))
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    for (entry, (identity, staged_identity)) in plan.cleanup_candidates.iter().zip(identities) {
        progress(&format!("Removing {}...", entry.label));
        let path = Path::new(&entry.path);
        let current = crate::sa3_migration::recheck(root, hub, path);
        if !current.warnings.is_empty()
            || !current
                .cleanup_candidates
                .iter()
                .any(|candidate| candidate.path == entry.path)
            || path.canonicalize().ok().as_ref() != Some(&identity)
            || crate::sa3_migration::staged_training_identity(root, path) != staged_identity
        {
            result.errors.push(format!("Preserved {} because its ownership or protected paths changed; rescan before retrying.", path.display()));
        } else if let Some((_, owner)) = allowed.iter().find(|(candidate, _)| candidate == path) {
            let owner = owner.as_path();
            match crate::resolve_managed_path(path, owner) {
                Ok(Some(current)) if current == identity => match remove(path, owner) {
                    Ok(true) => {
                        result.removed_paths.push(entry.path.clone());
                        result.estimated_removed_bytes += entry.bytes;
                    }
                    Ok(false) => {}
                    Err(error) => result.errors.push(error),
                },
                _ => result.errors.push(format!(
                    "Preserved {} because its storage boundary changed.",
                    path.display()
                )),
            }
        } else {
            result.errors.push(format!(
                "Refusing unrecognized cleanup target: {}",
                path.display()
            ));
        }
        result.selection.cleanup_errors = result.errors.clone();
        if result.selection.cleanup_errors.is_empty() {
            result
                .selection
                .cleanup_errors
                .push("SA3 cleanup was interrupted; review remaining files and retry.".into());
        }
        crate::sa3_runtime::save(root, &result.selection)?;
    }
    progress("Checking for remaining Python files and saving the cleanup result...");
    let remaining = crate::sa3_migration::preview(root, hub);
    result.errors.extend(remaining.warnings);
    if !remaining.cleanup_candidates.is_empty() && result.errors.is_empty() {
        result
            .errors
            .push("Legacy SA3 files remain. Rescan and review cleanup again.".into());
    }
    result.selection.cleanup_complete =
        result.errors.is_empty() && remaining.cleanup_candidates.is_empty();
    result.selection.cleanup_errors = result.errors.clone();
    crate::sa3_runtime::save(root, &result.selection)?;
    progress(if result.selection.cleanup_complete {
        "Python cleanup complete. SA3 is ready."
    } else {
        "Cleanup needs attention. Review the remaining files below."
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(root: &Path, hub: &Path, token: &str) -> Result<ResultInfo, String> {
        super::run(root, hub, token, |_| {})
    }
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "gary-sa3-cleanup-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                )))
        }
        fn write(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"fixture").unwrap();
            path
        }
        fn select(&self) {
            crate::sa3_runtime::save(
                &self.0,
                &crate::sa3_runtime::Selection {
                    schema_version: 1,
                    encoding: "F16".into(),
                    verified_release: "0.1.2".into(),
                    backend: "cuda".into(),
                    activated_at: 1,
                    cleanup_complete: false,
                    cleanup_errors: Vec::new(),
                },
            )
            .unwrap();
        }
        fn hub(&self) -> PathBuf {
            self.0.join("external-hub")
        }
        fn token(&self) -> String {
            crate::sa3_migration::preview(&self.0, &self.hub()).cleanup_token
        }

        fn code(&self) -> Vec<PathBuf> {
            let bundle = self.0.join("bundle");
            let mut paths = Vec::new();
            for relative in [
                "api.py",
                "stable_audio_3/factory.py",
                "underfit/utils/audio.py",
            ] {
                let source = self.write(&format!("bundle/services/sa3/{relative}"));
                let installed = self.write(&format!("services/sa3/{relative}"));
                assert_eq!(
                    std::fs::read(source).unwrap(),
                    std::fs::read(&installed).unwrap()
                );
                paths.push(installed);
            }
            crate::sa3_code::record_bundle(&bundle, &self.0, false).unwrap();
            paths
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = crate::remove_managed_path(&self.0, &std::env::temp_dir());
        }
    }

    #[test]
    fn cleanup_reports_the_current_item_before_removal_and_distinguishes_partial_failure() {
        for locked_weights in [false, true] {
            let fixture = Fixture::new();
            fixture.write("services/sa3/env/python.exe");
            let weights = fixture.write("external-hub/models--stabilityai--stable-audio-3-medium/weights");
            fixture.select();
            let messages = std::cell::RefCell::new(Vec::<String>::new());
            let result = run_with_progress(
                &fixture.0, &fixture.hub(), &fixture.token(),
                |path, owner| {
                    // Activity must reach the UI before a potentially slow deletion.
                    let current = messages.borrow().last().unwrap().clone();
                    assert!(current.starts_with("Removing SA3 "));
                    if locked_weights && path == weights.parent().unwrap() {
                        assert!(current.contains("inference weights"));
                        Err("weights are locked".into())
                    } else {
                        crate::remove_managed_path(path, owner)
                    }
                },
                |message| messages.borrow_mut().push(message.to_string()),
            ).unwrap();
            let messages = messages.into_inner();
            assert!(messages[0].starts_with("Checking the reviewed"));
            assert!(messages.iter().any(|message| message.contains("Removing SA3 Python environment")));
            assert!(messages.iter().any(|message| message.contains("Removing SA3 PyTorch inference weights")));
            assert!(messages.iter().any(|message| message.starts_with("Checking for remaining")));
            assert_eq!(result.selection.cleanup_complete, !locked_weights);
            if locked_weights {
                assert!(weights.exists());
                assert!(messages.last().unwrap().starts_with("Cleanup needs attention"));
                assert!(!messages.iter().any(|message| message.starts_with("Python cleanup complete")));
            } else {
                assert!(!weights.exists());
                assert!(messages.last().unwrap().starts_with("Python cleanup complete"));
            }
        }
    }

    #[test]
    #[ignore = "uses existing staged PyTorch and GGUF training files as read-only hard-link sources; set GARY4LOCAL_STAGED_BASE_SOURCE and GARY4LOCAL_STAGED_BASE_NATIVE_MODELS"]
    fn real_staged_training_cleanup_preserves_sources_and_retries_locked_weights() {
        use crate::sa3_migration::{STAGED_TRAINING_BASE, STAGED_TRAINING_FILES};
        let source = PathBuf::from(std::env::var_os("GARY4LOCAL_STAGED_BASE_SOURCE").unwrap());
        let native_models = PathBuf::from(std::env::var_os("GARY4LOCAL_STAGED_BASE_NATIVE_MODELS").unwrap());
        if let Some(root) = std::env::var_os("GARY4LOCAL_STAGED_BASE_PROFILE").map(PathBuf::from) {
            assert_eq!(root.join(STAGED_TRAINING_BASE).canonicalize().unwrap(), source.canonicalize().unwrap());
            let preview = crate::sa3_migration::preview(&root, &crate::storage::effective_hf_hub_cache_dir(&root));
            assert!(preview.training_base_present && preview.legacy_runtime_present);
            assert!(preview.warnings.is_empty(), "Read-only source profile warnings: {:?}", preview.warnings);
            let staged: Vec<_> = preview.cleanup_candidates.iter().filter(|item| Path::new(&item.path).starts_with(&source)).collect();
            assert_eq!(staged.len(), 6);
            println!("PASS read-only live-profile preview includes all six staged training files ({} logical bytes) with no warnings", staged.iter().map(|item| item.bytes).sum::<u64>());
        }
        let fixture = Fixture::new();
        assert!(!source.starts_with(&fixture.0) && !native_models.starts_with(&fixture.0));
        let mut originals = Vec::new();
        for file in STAGED_TRAINING_FILES {
            let original = source.join(file);
            let metadata = std::fs::metadata(&original).unwrap();
            originals.push((original.clone(), metadata.len(), metadata.modified().unwrap()));
            let target = fixture.0.join(STAGED_TRAINING_BASE).join(file);
            std::fs::create_dir_all(target.parent().unwrap()).unwrap();
            std::fs::hard_link(original, target).unwrap();
        }
        let keep = [
            fixture.write(&format!("{STAGED_TRAINING_BASE}/user-model.ckpt")),
            fixture.write(&format!("{STAGED_TRAINING_BASE}/t5gemma-b-b-ul2/custom.json")),
            fixture.write("sa3/training/jobs/old/checkpoint.safetensors"),
            fixture.write("sa3/training/runs/old/training.log"),
            fixture.write("sa3/datasets/audio.wav"),
            fixture.write("sa3/loras/user.safetensors"),
        ];
        let env = fixture.write("services/sa3/env/python.exe");
        fixture.select();
        let preview = crate::sa3_migration::preview(&fixture.0, &fixture.hub());
        assert!(preview.training_base_present && preview.legacy_runtime_present);
        assert_eq!(preview.cleanup_candidates.len(), 7);
        assert!(validate_review(&fixture.0, &fixture.hub(), &preview.cleanup_token).unwrap_err().contains("C++ training model"));
        let native = crate::sa3_models::component("sa3-native::medium-base-F16").unwrap();
        let dir = crate::sa3_models::checked_models_dir(&fixture.0).unwrap();
        for file in &native.files {
            std::fs::hard_link(native_models.join(&file.filename), dir.join(&file.filename)).unwrap();
        }
        let locked = fixture.0.join(STAGED_TRAINING_BASE).join("model.safetensors");
        let first = run_with(&fixture.0, &fixture.hub(), &fixture.token(), |path, owner| {
            if path == locked { Err("locked staged training weights".into()) }
            else { crate::remove_managed_path(path, owner) }
        }).unwrap();
        assert!(!first.selection.cleanup_complete && locked.is_file() && !env.exists());
        assert_eq!(first.errors, ["locked staged training weights"]);
        let retry = run(&fixture.0, &fixture.hub(), &fixture.token()).unwrap();
        assert!(retry.selection.cleanup_complete);
        assert!(!crate::sa3_migration::legacy_runtime_present(&fixture.0, &fixture.hub()));
        assert!(crate::sa3_migration::preview(&fixture.0, &fixture.hub()).cleanup_candidates.is_empty());
        for file in STAGED_TRAINING_FILES { assert!(!fixture.0.join(STAGED_TRAINING_BASE).join(file).exists()); }
        for path in &keep { assert!(path.is_file(), "{}", path.display()); }
        assert!(crate::sa3_models::present(native, &dir));
        for (original, bytes, modified) in &originals {
            let current = std::fs::metadata(original).unwrap();
            assert_eq!(current.len(), *bytes);
            assert_eq!(current.modified().unwrap(), *modified);
        }
        println!("PASS six staged files ({} logical bytes) included without HF cache; native replacement required; lock journal/retry; original staged/native sources, custom files, jobs, datasets and LoRAs preserved", originals.iter().map(|(_, bytes, _)| bytes).sum::<u64>());
    }

    #[test]
    fn staged_training_cleanup_requires_a_native_replacement_before_any_removal() {
        let fixture = Fixture::new();
        let env = fixture.write("services/sa3/env/python.exe");
        let model = fixture.write(&format!("{}/model.safetensors", crate::sa3_migration::STAGED_TRAINING_BASE));
        fixture.select();
        assert!(run(&fixture.0, &fixture.hub(), &fixture.token()).unwrap_err().contains("C++ training model"));
        assert!(env.is_file() && model.is_file());
        assert!(crate::sa3_migration::preview(&fixture.0, &fixture.hub()).training_base_present);
    }

    #[test]
    fn cleanup_requires_native_selection_and_current_review() {
        let fixture = Fixture::new();
        let original = fixture.write("services/sa3/env/python.exe");
        assert!(run(&fixture.0, &fixture.hub(), &fixture.token())
            .unwrap_err()
            .contains("select C++"));
        fixture.select();
        let token = fixture.token();
        fixture.write("services/sa3/.venv/python.exe");
        assert!(run(&fixture.0, &fixture.hub(), &token)
            .unwrap_err()
            .contains("changed"));
        assert!(original.is_file());
    }

    #[test]
    fn code_cleanup_requires_current_contents_and_keeps_custom_files() {
        let fixture = Fixture::new();
        let env = fixture.write("services/sa3/env/python.exe");
        let code = fixture.code();
        let settings = fixture.write("services/sa3/defaults.ini");
        let unknown = fixture.write("services/sa3/user.py");
        let native = fixture.write("services/sa3/native/sa3-server.exe");
        fixture.select();
        let stale = fixture.token();
        std::fs::write(&code[0], b"edited user code").unwrap();
        assert!(run(&fixture.0, &fixture.hub(), &stale)
            .unwrap_err()
            .contains("changed"));
        assert!(env.is_file());
        let result = run(&fixture.0, &fixture.hub(), &fixture.token()).unwrap();
        assert!(result.selection.cleanup_complete);
        assert_eq!(result.removed_paths.len(), 3);
        assert!(!env.exists());
        assert!(!code[1].exists());
        assert!(!code[2].exists());
        for path in [&code[0], &settings, &unknown, &native] {
            assert!(path.is_file());
        }
        assert_eq!(std::fs::read(&code[0]).unwrap(), b"edited user code");
    }

    #[test]
    fn code_edited_during_cleanup_is_still_caught_by_the_per_item_recheck() {
        let fixture = Fixture::new();
        let env = fixture.write("services/sa3/env/python.exe");
        let code = fixture.code();
        fixture.select();
        let edited = code[2].clone();
        let result = run_with(&fixture.0, &fixture.hub(), &fixture.token(), |path, owner| {
            // The environment goes first; edit a later file after review.
            if path == env.parent().unwrap() {
                std::fs::write(&edited, b"edited mid-cleanup").unwrap();
            }
            crate::remove_managed_path(path, owner)
        })
        .unwrap();
        assert!(!env.exists() && !code[0].exists() && !code[1].exists());
        assert_eq!(std::fs::read(&edited).unwrap(), b"edited mid-cleanup");
        assert!(!result.selection.cleanup_complete);
        assert!(result.errors.iter().any(|error| error.contains("Preserved")));
    }

    #[test]
    fn recheck_classifies_only_the_path_it_was_asked_about() {
        let fixture = Fixture::new();
        fixture.write("services/sa3/env/python.exe");
        let code = fixture.code();
        let focused = crate::sa3_migration::recheck(&fixture.0, &fixture.hub(), &code[1]);
        assert_eq!(focused.cleanup_candidates.len(), 1);
        assert_eq!(Path::new(&focused.cleanup_candidates[0].path), code[1]);
        assert_eq!(focused.cleanup_candidates[0].bytes, 0);
        let full = crate::sa3_migration::preview(&fixture.0, &fixture.hub());
        assert_eq!(full.cleanup_candidates.len(), 4);
    }

    #[test]
    fn locked_code_is_journaled_and_retry_only_removes_remaining_owned_files() {
        let fixture = Fixture::new();
        let code = fixture.code();
        fixture.select();
        let first = run_with(
            &fixture.0,
            &fixture.hub(),
            &fixture.token(),
            |path, owner| {
                if path == code[1] {
                    Err("locked Python source fixture".into())
                } else {
                    crate::remove_managed_path(path, owner)
                }
            },
        )
        .unwrap();
        assert!(!first.selection.cleanup_complete);
        assert_eq!(first.errors, ["locked Python source fixture"]);
        assert!(!code[0].exists());
        assert!(code[1].is_file());
        assert!(!code[2].exists());
        let retry = run(&fixture.0, &fixture.hub(), &fixture.token()).unwrap();
        assert!(retry.selection.cleanup_complete);
        assert_eq!(retry.removed_paths.len(), 1);
        assert!(!code[1].exists());
    }

    #[test]
    fn replaced_bundle_ownership_requires_a_new_review_even_for_the_same_paths() {
        let fixture = Fixture::new();
        let code = fixture.code();
        fixture.select();
        let stale = fixture.token();
        std::fs::write(&code[0], b"new bundled source").unwrap();
        std::fs::write(
            fixture.0.join("bundle/services/sa3/api.py"),
            b"new bundled source",
        )
        .unwrap();
        crate::sa3_code::record_bundle(&fixture.0.join("bundle"), &fixture.0, false).unwrap();
        assert_ne!(stale, fixture.token());
        assert!(run(&fixture.0, &fixture.hub(), &stale)
            .unwrap_err()
            .contains("changed"));
        assert!(code.iter().all(|path| path.is_file()));
    }

    #[test]
    fn code_changes_after_first_removal_are_preserved_and_require_review() {
        let fixture = Fixture::new();
        let code = fixture.code();
        fixture.select();
        let first = run_with(
            &fixture.0,
            &fixture.hub(),
            &fixture.token(),
            |path, owner| {
                if path == code[0] {
                    std::fs::write(&code[1], b"modified during cleanup").unwrap();
                }
                crate::remove_managed_path(path, owner)
            },
        )
        .unwrap();
        assert!(!first.selection.cleanup_complete);
        assert!(first.errors.iter().any(|error| error.contains("changed")));
        assert_eq!(std::fs::read(&code[1]).unwrap(), b"modified during cleanup");
        assert!(
            run(&fixture.0, &fixture.hub(), &fixture.token())
                .unwrap()
                .selection
                .cleanup_complete
        );
        assert!(code[1].is_file());
    }

    #[test]
    fn registered_caption_paths_inside_owned_code_folders_block_retirement() {
        let fixture = Fixture::new();
        let code = fixture.code();
        let catalog = fixture.write("sa3/lora_catalog.json");
        let source = fixture.write("sa3/loras/user.safetensors");
        std::fs::write(
            catalog,
            serde_json::json!({"user":{
                "path":source,"strength":1,"promptsPath":code[1].parent().unwrap()
            }})
            .to_string(),
        )
        .unwrap();
        fixture.select();
        let review = crate::sa3_migration::preview(&fixture.0, &fixture.hub());
        assert!(!review.warnings.is_empty());
        assert!(run(&fixture.0, &fixture.hub(), &review.cleanup_token)
            .unwrap_err()
            .contains("warnings"));
        assert!(code.iter().all(|path| path.is_file()));
    }

    #[test]
    fn partial_failure_is_journaled_and_retry_keeps_native_selection() {
        let fixture = Fixture::new();
        let env = fixture.write("services/sa3/env/python.exe");
        let weights =
            fixture.write("external-hub/models--stabilityai--stable-audio-3-medium/weights");
        let keep = [
            fixture.write("services/sa3/api.py"),
            fixture.write("sa3/loras/user.ckpt"),
            fixture.write("sa3/training/jobs/job/checkpoint.gguf"),
            fixture.write("models/sa3/native.gguf"),
            fixture.write("services/foundation/env/python.exe"),
            fixture.write("native-runtimes/cudart-12.8/runtime.dll"),
            fixture.write("external-hub/models--other--shared/weights"),
            fixture.write("cache/uv/shared"),
            fixture.write("services/sa3/native/sa3-server.exe"),
        ];
        fixture.select();
        let result = run_with(
            &fixture.0,
            &fixture.hub(),
            &fixture.token(),
            |path, owner| {
                if path.ends_with("env") {
                    Err("locked Python fixture".into())
                } else {
                    crate::remove_managed_path(path, owner)
                }
            },
        )
        .unwrap();
        assert_eq!(result.errors, vec!["locked Python fixture"]);
        assert!(!result.selection.cleanup_complete);
        assert!(env.is_file());
        assert!(!weights.exists());
        let saved = crate::sa3_runtime::read(&fixture.0).unwrap().unwrap();
        assert_eq!(saved.encoding, "F16");
        assert_eq!(saved.cleanup_errors, result.errors);
        let retry = run(&fixture.0, &fixture.hub(), &fixture.token()).unwrap();
        assert!(retry.selection.cleanup_complete);
        assert!(!env.exists());
        for path in keep {
            assert!(path.is_file(), "{}", path.display());
        }
        assert!(
            run(&fixture.0, &fixture.hub(), &fixture.token())
                .unwrap()
                .selection
                .cleanup_complete
        );
    }

    #[test]
    fn interrupted_cleanup_retries_from_persisted_native_journal() {
        let fixture = Fixture::new();
        let env = fixture.write("services/sa3/env/python.exe");
        let weights =
            fixture.write("external-hub/models--stabilityai--stable-audio-3-medium/weights");
        fixture.select();
        let stopped = std::panic::catch_unwind(|| {
            run_with(
                &fixture.0,
                &fixture.hub(),
                &fixture.token(),
                |path, owner| {
                    let saved = crate::sa3_runtime::read(&fixture.0).unwrap().unwrap();
                    assert!(!saved.cleanup_complete);
                    assert!(saved
                        .cleanup_errors
                        .iter()
                        .any(|error| error.contains("interrupted")));
                    if path.ends_with("models--stabilityai--stable-audio-3-medium") {
                        panic!("simulated interruption");
                    }
                    crate::remove_managed_path(path, owner)
                },
            )
            .unwrap();
        });
        assert!(stopped.is_err());
        assert!(!env.exists());
        assert!(weights.is_file());
        assert!(
            !crate::sa3_runtime::read(&fixture.0)
                .unwrap()
                .unwrap()
                .cleanup_complete
        );
        assert!(
            run(&fixture.0, &fixture.hub(), &fixture.token())
                .unwrap()
                .selection
                .cleanup_complete
        );
        assert!(!weights.exists());
    }

    #[test]
    fn unfinished_training_dataset_inside_an_env_is_preserved() {
        let fixture = Fixture::new();
        let dataset = fixture.write("services/sa3/env/dataset/audio.wav");
        let options = fixture.write("sa3/training/jobs/native-job/native-options.json");
        fixture.select();
        std::fs::write(
            options,
            serde_json::json!({"dataset":dataset.parent().unwrap(),"promptConfig":null})
                .to_string(),
        )
        .unwrap();
        assert!(run(&fixture.0, &fixture.hub(), &fixture.token())
            .unwrap_err()
            .contains("warnings"));
        assert!(dataset.is_file());
    }

    #[cfg(windows)]
    #[test]
    fn deleting_an_env_does_not_follow_nested_dataset_junctions() {
        use std::os::windows::process::CommandExt;
        let fixture = Fixture::new();
        let outside = Fixture::new();
        let dataset = outside.write("dataset/audio.wav");
        let env = fixture.write("services/sa3/env/python.exe");
        fixture.select();
        let link = env.parent().unwrap().join("dataset-link");
        let created = std::process::Command::new("cmd")
            .raw_arg(format!(
                "/c mklink /J \"{}\" \"{}\"",
                link.display(),
                dataset.parent().unwrap().display()
            ))
            .output()
            .unwrap();
        assert!(
            created.status.success(),
            "{}",
            String::from_utf8_lossy(&created.stderr)
        );
        assert!(
            run(&fixture.0, &fixture.hub(), &fixture.token())
                .unwrap()
                .selection
                .cleanup_complete
        );
        assert!(!env.exists());
        assert_eq!(std::fs::read(dataset).unwrap(), b"fixture");
    }

    #[test]
    fn catalog_and_dataset_protections_block_cleanup() {
        let fixture = Fixture::new();
        let source = fixture.write("services/sa3/env/user.safetensors");
        let catalog = fixture.write("sa3/lora_catalog.json");
        fixture.select();
        std::fs::write(
            catalog,
            serde_json::json!({"user":{"path":source,"strength":1}}).to_string(),
        )
        .unwrap();
        assert!(run(&fixture.0, &fixture.hub(), &fixture.token())
            .unwrap_err()
            .contains("warnings"));
        assert!(source.is_file());
    }
}
