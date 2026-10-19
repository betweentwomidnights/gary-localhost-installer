//! Retire a migrated service's reviewed Python environments, code and weight
//! repositories. The command verifies native execution and reserves the service
//! before entering this transaction. The persisted selection journals
//! incomplete cleanup so an interruption or a locked file never causes fallback
//! to Python.
use crate::migration::{Selection, Service};
use crate::migration_inventory::Preview;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultInfo {
    pub removed_paths: Vec<String>,
    pub estimated_removed_bytes: u64,
    pub errors: Vec<String>,
    pub selection: Selection,
}

pub fn preview(service: &dyn Service, root: &Path, hub: &Path) -> Preview {
    crate::migration_inventory::preview(service.target(), root, hub, service.extras(root, hub))
}

fn read_selection(service: &dyn Service, root: &Path) -> Result<Option<Selection>, String> {
    crate::migration::read(root, service.target().def, |selection| service.check_selection(selection))
}

fn save_selection(service: &dyn Service, root: &Path, selection: &Selection) -> Result<(), String> {
    crate::migration::save(root, service.target().def, selection, |selection| {
        service.check_selection(selection)
    })
}

/// Returns the inventory it checked, so the caller can act on that same plan.
pub fn validate_review(
    service: &dyn Service,
    root: &Path,
    hub: &Path,
    token: &str,
) -> Result<Preview, String> {
    let label = service.target().label();
    let current = preview(service, root, hub);
    if current.cleanup_token != token {
        return Err(format!("{label} storage changed since review. Rescan and review cleanup again."));
    }
    if !current.warnings.is_empty() {
        return Err(format!(
            "Resolve the {label} cleanup inventory warnings before removing Python files."
        ));
    }
    if current.native_selection.is_none() {
        return Err(format!(
            "Verify and select C++ before retiring the {label} Python installation."
        ));
    }
    // What is on disk; callers that know which legacy models the model
    // manager holds as downloaded check those before reviewing.
    if let Some(blocker) = service.retirement_blocker(root, hub, &[]) {
        return Err(blocker);
    }
    Ok(current)
}

pub(crate) fn run(
    service: &dyn Service,
    root: &Path,
    hub: &Path,
    token: &str,
    progress: impl Fn(&str),
) -> Result<ResultInfo, String> {
    run_with_progress(service, root, hub, token, crate::remove_managed_path, progress)
}

pub(crate) fn run_with_progress(
    service: &dyn Service,
    root: &Path,
    hub: &Path,
    token: &str,
    mut remove: impl FnMut(&Path, &Path) -> Result<bool, String>,
    progress: impl Fn(&str),
) -> Result<ResultInfo, String> {
    let target = service.target();
    let label = target.label();
    let interrupted = format!("{label} cleanup was interrupted; review remaining files and retry.");
    progress("Checking the reviewed Python files before cleanup...");
    let plan = validate_review(service, root, hub, token)?;
    let mut selection = read_selection(service, root)?
        .ok_or_else(|| format!("Native {label} selection is missing"))?;
    // Publish the incomplete state before the first deletion. Following a crash,
    // rescanning discovers remaining paths and retry never changes runtime.
    selection.cleanup_complete = false;
    selection.cleanup_errors = vec![interrupted.clone()];
    save_selection(service, root, &selection)?;
    let mut result = ResultInfo {
        removed_paths: Vec::new(),
        estimated_removed_bytes: 0,
        errors: Vec::new(),
        selection,
    };
    let allowed = crate::migration_inventory::allowed(target, root, hub)?;
    let identities: Vec<_> = plan
        .cleanup_candidates
        .iter()
        .map(|entry| {
            let path = Path::new(&entry.path);
            path.canonicalize()
                .map(|canonical| {
                    (canonical, crate::migration_inventory::staged_identity(target, root, path))
                })
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    for (entry, (identity, staged_identity)) in plan.cleanup_candidates.iter().zip(identities) {
        progress(&format!("Removing {}...", entry.label));
        let path = Path::new(&entry.path);
        let current = crate::migration_inventory::recheck(
            target,
            root,
            hub,
            path,
            service.extras(root, hub),
        );
        if !current.warnings.is_empty()
            || !current
                .cleanup_candidates
                .iter()
                .any(|candidate| candidate.path == entry.path)
            || path.canonicalize().ok().as_ref() != Some(&identity)
            || crate::migration_inventory::staged_identity(target, root, path) != staged_identity
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
            result.selection.cleanup_errors.push(interrupted.clone());
        }
        save_selection(service, root, &result.selection)?;
    }
    progress("Checking for remaining Python files and saving the cleanup result...");
    let remaining = preview(service, root, hub);
    result.errors.extend(remaining.warnings);
    if !remaining.cleanup_candidates.is_empty() && result.errors.is_empty() {
        result
            .errors
            .push(format!("Legacy {label} files remain. Rescan and review cleanup again."));
    }
    result.selection.cleanup_complete =
        result.errors.is_empty() && remaining.cleanup_candidates.is_empty();
    result.selection.cleanup_errors = result.errors.clone();
    save_selection(service, root, &result.selection)?;
    if result.selection.cleanup_complete {
        progress(&format!("Python cleanup complete. {label} is ready."));
    } else {
        progress("Cleanup needs attention. Review the remaining files below.");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::MigrationDef;
    use crate::migration::Target;
    use crate::migration_inventory::Extras;
    use std::path::PathBuf;

    /// A service the engine has never seen, described only by its block.
    struct Other(MigrationDef);

    impl Service for Other {
        fn target(&self) -> Target<'_> {
            Target { id: "foundation", def: &self.0, models_dir: Some("foundation") }
        }

        fn extras(&self, root: &Path, _hub: &Path) -> Extras {
            Extras {
                selection: crate::migration::read(root, &self.0, |_| Ok(())),
                protected: Vec::new(),
                preserved: Vec::new(),
                warnings: Vec::new(),
                training_base_present: false,
            }
        }

        fn models(&self) -> &'static crate::native_models::Catalog {
            static NONE: std::sync::OnceLock<crate::native_models::Catalog> = std::sync::OnceLock::new();
            NONE.get_or_init(|| crate::native_models::Catalog::bundled("foundation", "[]"))
        }

        fn tier_components(&self, _tier: &str) -> Result<Vec<String>, String> {
            Ok(Vec::new())
        }

        fn verify<'a>(
            &'a self,
            _run: crate::migration::Run<'a>,
        ) -> crate::migration::Proof<'a, crate::migration::Selection> {
            Box::pin(async { Err("cleanup tests never verify".to_string()) })
        }
    }

    fn write(path: PathBuf) -> PathBuf {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"fixture").unwrap();
        path
    }

    #[test]
    fn another_service_migrates_from_nothing_but_its_manifest_block() {
        let root = std::env::temp_dir().join(format!(
            "gary-generic-cleanup-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let hub = root.join("models/huggingface/hub");
        let service = Other(
            serde_json::from_value(serde_json::json!({
                "dataDir": "foundation",
                "label": "Foundation-1",
                "legacy": {
                    "envDirs": ["env"],
                    "hfRepos": [{"repo": "org/foundation-1", "label": "PyTorch weights"}]
                }
            }))
            .unwrap(),
        );
        let env = write(root.join("services/foundation/env/Scripts/python.exe"));
        let weights = write(hub.join("models--org--foundation-1/snapshots/abc/model.safetensors"));
        let kept = [
            write(root.join("foundation/presets/user.json")),
            write(root.join("services/foundation/native/foundation-server.exe")),
            write(root.join("services/foundation/defaults.ini")),
            write(root.join("models/foundation/foundation-F16.gguf")),
            write(hub.join("models--org--someone-else/snapshots/abc/model.safetensors")),
            write(root.join("services/gary/env/Scripts/python.exe")),
        ];

        let before = preview(&service, &root, &hub);
        assert!(before.legacy_runtime_present);
        assert!(before.warnings.is_empty(), "{:?}", before.warnings);
        let labels: Vec<_> = before.cleanup_candidates.iter().map(|item| item.label.as_str()).collect();
        assert_eq!(labels, ["Foundation-1 Python environment", "Foundation-1 PyTorch weights"]);
        assert!(validate_review(&service, &root, &hub, &before.cleanup_token)
            .unwrap_err()
            .contains("select C++"));

        let selection = Selection {
            schema_version: 1,
            encoding: "F16".into(),
            verified_release: "v0.1.0".into(),
            backend: "cuda".into(),
            activated_at: 1,
            cleanup_complete: false,
            cleanup_errors: Vec::new(),
            runtime_fingerprint: None,
        };
        crate::migration::save(&root, &service.0, &selection, |_| Ok(())).unwrap();
        let token = preview(&service, &root, &hub).cleanup_token;
        let result = run(&service, &root, &hub, &token, |_| {}).unwrap();

        assert!(result.selection.cleanup_complete, "{:?}", result.errors);
        assert!(!env.exists() && !weights.exists());
        assert!(!crate::migration_inventory::legacy_present(service.target(), &root, &hub));
        for path in &kept {
            assert!(path.is_file(), "{} must be kept", path.display());
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
