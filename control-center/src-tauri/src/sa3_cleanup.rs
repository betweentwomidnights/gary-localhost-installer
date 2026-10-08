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

pub fn validate_review(root: &Path, hub: &Path, token: &str) -> Result<(), String> {
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
    Ok(())
}

pub(crate) fn run(root: &Path, hub: &Path, token: &str) -> Result<ResultInfo, String> {
    run_with(root, hub, token, crate::remove_managed_path)
}

fn run_with(
    root: &Path,
    hub: &Path,
    token: &str,
    mut remove: impl FnMut(&Path, &Path) -> Result<bool, String>,
) -> Result<ResultInfo, String> {
    validate_review(root, hub, token)?;
    let plan = crate::sa3_migration::preview(root, hub);
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
    let mut allowed = vec![
        service.join("env"),
        service.join(".venv"),
        hub.join("models--stabilityai--stable-audio-3-medium"),
        hub.join("models--stabilityai--stable-audio-3-medium-base"),
    ];
    allowed.extend(crate::sa3_code::recorded_paths(root)?);
    let identities: Vec<PathBuf> = plan
        .cleanup_candidates
        .iter()
        .map(|entry| {
            Path::new(&entry.path)
                .canonicalize()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    for (entry, identity) in plan.cleanup_candidates.iter().zip(identities) {
        let path = Path::new(&entry.path);
        let current = crate::sa3_migration::preview(root, hub);
        if !current.warnings.is_empty()
            || !current
                .cleanup_candidates
                .iter()
                .any(|candidate| candidate.path == entry.path)
            || path.canonicalize().ok().as_ref() != Some(&identity)
        {
            result.errors.push(format!("Preserved {} because its ownership or protected paths changed; rescan before retrying.", path.display()));
        } else if let Some(index) = allowed.iter().position(|candidate| candidate == path) {
            let owner = if index < 2 || index >= 4 {
                service.as_path()
            } else {
                hub
            };
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
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
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
