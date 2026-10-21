//! What a migrating service's Python install owns, and which of it cleanup may
//! remove. Candidates come only from the manifest's `legacy` block; everything
//! else is protected by default. Each candidate must still resolve, without
//! following redirects, to exactly where the manifest says it lives, and must
//! not overlap user data, native files, another service or another model.
use crate::migration::{Selection, Target};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Item {
    pub label: String,
    pub path: String,
    pub bytes: u64,
    pub kind: Option<&'static str>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Preview {
    pub active_root: String,
    pub hf_hub_cache: String,
    pub cleanup_candidates: Vec<Item>,
    pub legacy_runtime_present: bool,
    /// Whether the service's replaced artifact (SA3's training base) exists
    /// in either form, native or legacy. Always false for a service without one.
    pub training_base_present: bool,
    pub estimated_cleanup_bytes: u64,
    pub preserved_paths: Vec<Item>,
    pub warnings: Vec<String>,
    pub native_selection: Option<Selection>,
    pub cleanup_token: String,
    /// Why cleanup cannot remove Python whatever the review finds: features
    /// that still run only there.
    pub python_still_needed: Option<String>,
}

/// What only the service knows, gathered by the service before an inventory.
pub struct Extras {
    /// Its selection, read with its own encoding check.
    pub selection: Result<Option<Selection>, String>,
    /// User data the manifest cannot name: catalogued adapters, datasets.
    pub protected: Vec<PathBuf>,
    /// Shown as kept, after the shared preserved paths.
    pub preserved: Vec<(String, PathBuf)>,
    pub warnings: Vec<String>,
    pub training_base_present: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Environment,
    Weights,
    Code,
}

struct Candidate {
    label: String,
    path: PathBuf,
    owner: PathBuf,
    kind: Kind,
}

fn path_present(path: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Ok(_) => true,
        Err(error) => error.kind() != std::io::ErrorKind::NotFound,
    }
}

/// Everything the manifest says the Python install owns, except its code.
fn legacy_candidates(target: Target, root: &Path, hub: &Path) -> Vec<Candidate> {
    let label = target.label();
    let legacy = &target.def.legacy;
    let service = target.service_dir(root);
    let mut candidates = Vec::new();
    for (index, env) in legacy.env_dirs.iter().enumerate() {
        let name = if index == 0 { "Python environment" } else { "alternate Python environment" };
        candidates.push(Candidate {
            label: format!("{label} {name}"),
            path: service.join(env),
            owner: service.clone(),
            kind: Kind::Environment,
        });
    }
    for repo in &legacy.hf_repos {
        candidates.push(Candidate {
            label: format!("{label} {}", repo.label),
            path: hub.join(repo.cache_dir_name()),
            owner: hub.to_path_buf(),
            kind: Kind::Weights,
        });
    }
    if let Some(staged) = &legacy.staged {
        let dir = root.join(&staged.dir);
        for file in &staged.files {
            candidates.push(Candidate {
                label: format!("{label} {}", staged.label),
                path: dir.join(file),
                owner: dir.clone(),
                kind: Kind::Weights,
            });
        }
    }
    candidates
}

/// One of the exact files the Python service staged in its data folder.
pub fn staged_file(target: Target, root: &Path, path: &Path) -> bool {
    target.def.legacy.staged.as_ref().is_some_and(|staged| {
        let dir = root.join(&staged.dir);
        staged.files.iter().any(|file| path == dir.join(file))
    })
}

/// Size and modification time of a staged file, so cleanup can tell it is
/// still the file that was reviewed. None for anything else.
pub fn staged_identity(target: Target, root: &Path, path: &Path) -> Option<(u64, Option<u128>)> {
    if !staged_file(target, root, path) {
        return None;
    }
    std::fs::symlink_metadata(path).ok().map(|metadata| {
        (
            metadata.len(),
            metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|time| time.as_nanos()),
        )
    })
}

/// Whether any legacy artifact tagged with `role` is still on disk.
pub fn role_present(target: Target, root: &Path, hub: &Path, role: &str) -> bool {
    let legacy = &target.def.legacy;
    legacy
        .hf_repos
        .iter()
        .filter(|repo| repo.role.as_deref() == Some(role))
        .any(|repo| path_present(&hub.join(repo.cache_dir_name())))
        || legacy.staged.as_ref().is_some_and(|staged| {
            staged.role.as_deref() == Some(role)
                && staged.files.iter().any(|file| path_present(&root.join(&staged.dir).join(file)))
        })
}

/// A downgrade can recreate legacy files after cleanup was recorded as complete.
/// Bounded for service-status polling; the full inventory runs in preview.
/// Bundled source alone is not evidence of an installed Python runtime.
pub fn legacy_present(target: Target, root: &Path, hub: &Path) -> bool {
    legacy_candidates(target, root, hub)
        .iter()
        .any(|candidate| path_present(&candidate.path))
}

/// Every path the manifest lets cleanup remove, with the folder it must sit
/// directly in. Cleanup refuses anything not on this list, whatever a preview
/// says.
pub fn allowed(target: Target, root: &Path, hub: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    let mut allowed: Vec<_> = legacy_candidates(target, root, hub)
        .into_iter()
        .map(|candidate| (candidate.path, candidate.owner))
        .collect();
    let service = target.service_dir(root);
    allowed.extend(
        crate::migration_code::recorded_paths(target, root)?
            .into_iter()
            .map(|path| (path, service.clone())),
    );
    Ok(allowed)
}

pub fn preview(target: Target, root: &Path, hub: &Path, extras: Extras) -> Preview {
    inventory(target, root, hub, None, extras)
}

/// The same ownership and protection decision as `preview`, for one path.
/// Cleanup calls this right before each removal, so it skips sizing (a walk of
/// the whole Python environment) and hashes only that path if it is code.
/// The full preview still runs before and after the cleanup transaction.
pub fn recheck(target: Target, root: &Path, hub: &Path, path: &Path, extras: Extras) -> Preview {
    inventory(target, root, hub, Some(path), extras)
}

fn inventory(
    target: Target,
    root: &Path,
    hub: &Path,
    focus: Option<&Path>,
    extras: Extras,
) -> Preview {
    let label = target.label();
    let mut result = Preview {
        active_root: root.to_string_lossy().to_string(),
        hf_hub_cache: hub.to_string_lossy().to_string(),
        cleanup_candidates: Vec::new(),
        legacy_runtime_present: legacy_present(target, root, hub),
        training_base_present: extras.training_base_present,
        estimated_cleanup_bytes: 0,
        preserved_paths: Vec::new(),
        warnings: Vec::new(),
        native_selection: None,
        cleanup_token: String::new(),
        python_still_needed: None,
    };
    match extras.selection {
        Ok(selection) => result.native_selection = selection,
        Err(error) => result.warnings.push(error),
    }
    let service = target.service_dir(root);
    let mut candidates = legacy_candidates(target, root, hub);
    if let Some(focus) = focus {
        candidates.retain(|candidate| candidate.path == focus);
    }
    let code = crate::migration_code::status(target, root, focus);
    result.warnings.extend(code.warnings);
    for path in &code.candidates {
        candidates.push(Candidate {
            label: format!("{label} bundled Python source"),
            path: path.clone(),
            owner: service.clone(),
            kind: Kind::Code,
        });
    }
    for path in &code.retained {
        result.preserved_paths.push(Item {
            label: "Edited Python code or developer checkout (preserved)".into(),
            path: path.to_string_lossy().into(),
            bytes: 0,
            kind: None,
        });
    }
    result.warnings.extend(extras.warnings);

    // The service's own data folder. Kept out of `protected_paths` because a
    // staged legacy file inside it is the one thing cleanup may remove from
    // it; anything else listed below protects whatever it names, including
    // this folder if, say, a training dataset is the whole of it.
    let data = target.data_dir(root);
    let models = target
        .models_dir
        .map(|dir| crate::storage::models_dir(root).join(dir));
    let mut protected_paths = vec![root.join("native-runtimes"), service.join("native")];
    protected_paths.extend(models.clone());
    protected_paths.extend(extras.protected);
    // Everything else under services/ belongs to another service or to the
    // bundle itself. Read it from disk rather than naming services, so one
    // added later is protected without anyone remembering to list it here.
    match std::fs::read_dir(root.join("services")) {
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Ok(entry) if entry.file_name() != target.id => protected_paths.push(entry.path()),
                    Ok(_) => {}
                    Err(_) => result
                        .warnings
                        .push("Cannot fully inventory the other services' folders.".into()),
                }
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => result
            .warnings
            .push("Cannot inspect the other services' folders.".into()),
    }
    // Every other repository in the hub is another model's.
    let own_repos: Vec<String> = target
        .def
        .legacy
        .hf_repos
        .iter()
        .map(|repo| repo.cache_dir_name())
        .collect();
    if let Ok(entries) = std::fs::read_dir(hub) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !own_repos.contains(&name) {
                protected_paths.push(entry.path());
            }
        }
    }

    let mut seen = Vec::new();
    for candidate in candidates {
        let Candidate { label: name, path, owner, kind } = candidate;
        let staged = staged_file(target, root, &path);
        if owner == service && !crate::path_is_inside(&owner, root) && path.exists() {
            result.warnings.push(format!(
                "{label} service storage redirects outside the active profile: {}",
                owner.display()
            ));
            continue;
        }

        match crate::resolve_managed_path(&path, &owner) {
            Ok(Some(canonical)) => {
                if (staged
                    && (!std::fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.is_file())
                        || root
                            .canonicalize()
                            .ok()
                            .map(|root_dir| root_dir.join(path.strip_prefix(root).unwrap()))
                            != Some(canonical.clone())))
                    || (!staged
                        && kind != Kind::Code
                        && canonical.parent() != owner.canonicalize().ok().as_deref())
                    || canonical.file_name() != path.file_name()
                {
                    result.warnings.push(format!(
                        "Preserving redirected {label} cleanup target {}; review it manually.",
                        path.display()
                    ));
                    continue;
                }
                // A redirected cache or environment must never turn the
                // proposed cleanup into removal of user-owned artifacts.
                let overlaps = |protected_path: &PathBuf| {
                    crate::path_is_inside(&canonical, protected_path)
                        || crate::path_is_inside(protected_path, &canonical)
                };
                // Staged legacy files are exempt from the data folder alone;
                // every catalog, dataset and native protection still applies.
                if (!staged && overlaps(&data)) || protected_paths.iter().any(overlaps) {
                    result.warnings.push(format!(
                        "Preserving {} because it overlaps {label} user data or native runtime files.",
                        path.display()
                    ));
                } else if !seen.contains(&canonical) {
                    let kind = match kind {
                        Kind::Code => "code",
                        Kind::Environment => "environment",
                        Kind::Weights => "weights",
                    };
                    result.cleanup_candidates.push(Item {
                        label: name,
                        path: path.to_string_lossy().to_string(),
                        bytes: if focus.is_none() { crate::path_size(&path) } else { 0 },
                        kind: Some(kind),
                    });
                    seen.push(canonical);
                }
            }
            Ok(None) => {}
            Err(_) => result.warnings.push(format!(
                "Cannot safely inventory {} within {}. Review this redirected or inaccessible path manually.",
                path.display(),
                owner.display()
            )),
        }
    }
    result.estimated_cleanup_bytes = result
        .cleanup_candidates
        .iter()
        .map(|entry| entry.bytes)
        .sum();
    // Not counted recursively: a LoRA checkpoint can be many GB, and none of
    // these contribute to the space this migration reclaims.
    let data_label = target
        .def
        .data_label
        .clone()
        .unwrap_or_else(|| format!("{label} user data"));
    let mut preserved = vec![
        (data_label, data),
        (format!("{label} native runtime, settings and unrecognized service files"), service),
        ("Shared CUDA and other native runtimes".to_string(), root.join("native-runtimes")),
    ];
    if let Some(models) = models {
        preserved.push(("Native model weights".to_string(), models));
    }
    preserved.extend(extras.preserved);
    for (label, path) in preserved {
        result.preserved_paths.push(Item {
            label,
            path: path.to_string_lossy().to_string(),
            bytes: 0,
            kind: None,
        });
    }
    use sha2::{Digest, Sha256};
    let identities: Vec<_> = result
        .cleanup_candidates
        .iter()
        .map(|entry| {
            (
                &entry.path,
                Path::new(&entry.path).canonicalize().ok(),
                staged_identity(target, root, Path::new(&entry.path)),
            )
        })
        .collect();
    result.cleanup_token = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                root.canonicalize().ok(),
                hub.canonicalize().ok(),
                identities,
                code.review_identity,
                &result.warnings,
            ))
            .unwrap()
        )
    );
    result
}
