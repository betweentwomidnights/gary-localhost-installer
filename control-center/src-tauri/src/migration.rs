//! A service's move from its Python environment to a native runtime.
//!
//! The switch is recorded per storage profile, in the service's data folder,
//! as its native selection. Once a selection exists the service runs native,
//! and an interrupted cleanup must never silently restore Python, so the same
//! record journals cleanup progress. What is service-specific about a
//! migration lives with the service; this module holds what every one shares.
use crate::manifest::{MigrationDef, ServiceDef};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const SELECTION_FILE: &str = "native-runtime.json";

/// What only a migrating service can answer. The shared inventory and cleanup
/// engine asks these; everything else it takes from the manifest block.
pub trait Service {
    fn target(&self) -> Target<'_>;

    /// The service's contribution to an inventory: its selection, the user
    /// data its own records point at, and anything it shows as kept.
    fn extras(&self, root: &Path, hub: &Path) -> crate::migration_inventory::Extras;

    /// Whether `selection.encoding` names one of the service's tiers.
    fn check_selection(&self, _selection: &Selection) -> Result<(), String> {
        Ok(())
    }

    /// Why cleanup must not run yet, e.g. a legacy artifact whose native
    /// replacement has not been prepared.
    fn cleanup_blocker(&self, _root: &Path, _hub: &Path) -> Option<String> {
        None
    }
}

/// A migrating service as the shared engine sees it: its id, which names its
/// service folder, and its manifest block.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    pub id: &'a str,
    pub def: &'a MigrationDef,
}

impl<'a> Target<'a> {
    pub fn service_dir(&self, root: &Path) -> PathBuf {
        root.join("services").join(self.id)
    }

    pub fn data_dir(&self, root: &Path) -> PathBuf {
        root.join(&self.def.data_dir)
    }

    /// The data folder as path components, for `checked_folder`.
    pub fn data_folders(&self) -> Vec<&'a str> {
        self.def.data_dir.split('/').collect()
    }

    pub fn label(&self) -> &'a str {
        &self.def.label
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Selection {
    pub schema_version: u32,
    pub encoding: String,
    pub verified_release: String,
    pub backend: String,
    pub activated_at: u64,
    #[serde(default)]
    pub cleanup_complete: bool,
    #[serde(default)]
    pub cleanup_errors: Vec<String>,
    /// What migration verification proved the runtime against; see
    /// `sa3_migration::runtime_fingerprint`. Absent on older selections, which
    /// only means the next cleanup verifies the runtime in full again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_fingerprint: Option<String>,
}

impl Selection {
    /// Python is retired: cleanup finished with nothing left to review.
    pub fn retired(&self) -> bool {
        self.cleanup_complete && self.cleanup_errors.is_empty()
    }

    /// Cleanup has started, so a bundle must no longer refresh the service's
    /// Python files, even if cleanup was interrupted.
    pub fn cleanup_started(&self) -> bool {
        self.cleanup_complete || !self.cleanup_errors.is_empty()
    }

    /// The checks every service shares. Whether `encoding` names one of its
    /// tiers is the service's own question; `read` and `save` take it as `check`.
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.verified_release.is_empty()
            || !["cuda", "vulkan", "metal", "cpu", "auto"].contains(&self.backend.as_str())
        {
            return Err("Invalid or unsupported native runtime selection".into());
        }
        Ok(())
    }
}

/// A service's migration as this session knows it: its selection, or why it
/// could not be read. A selection that exists but cannot be read still means
/// the service has moved, so it runs native and reports the error instead.
#[derive(Debug, Clone, Default)]
pub struct State {
    pub selection: Option<Selection>,
    pub error: Option<String>,
}

impl State {
    pub fn load(root: &Path, svc: &ServiceDef, migration: &MigrationDef) -> Self {
        match read(root, migration, |selection| check_for(svc, selection)) {
            Ok(selection) => Self { selection, error: None },
            Err(error) => Self { selection: None, error: Some(error) },
        }
    }

    /// Whether the service has left Python, whatever state cleanup is in.
    pub fn switched(&self) -> bool {
        self.selection.is_some() || self.error.is_some()
    }
}

/// The services that implement their half of a migration, by id. A service
/// can have a manifest block before it has an implementation here; the shared
/// parts still work for it, and its selection's encoding goes unchecked.
pub fn service(id: &str) -> Option<&'static dyn Service> {
    match id {
        "sa3" => Some(&crate::sa3_migration::Sa3),
        _ => None,
    }
}

/// The service-specific half of validating a selection.
pub fn check_for(svc: &ServiceDef, selection: &Selection) -> Result<(), String> {
    service(&svc.id).map_or(Ok(()), |service| service.check_selection(selection))
}

pub fn selection_path(root: &Path, migration: &MigrationDef) -> PathBuf {
    root.join(&migration.data_dir).join(SELECTION_FILE)
}

pub fn read(
    root: &Path,
    migration: &MigrationDef,
    check: impl Fn(&Selection) -> Result<(), String>,
) -> Result<Option<Selection>, String> {
    let path = selection_path(root, migration);
    match std::fs::symlink_metadata(&path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Cannot inspect native runtime selection: {error}")),
    }
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let dir = root
        .join(&migration.data_dir)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !dir.starts_with(&boundary)
        || path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err("Native runtime selection points outside the active storage profile".into());
    }
    let selection: Selection =
        serde_json::from_slice(&std::fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("Cannot read native runtime selection: {error}"))?;
    selection.validate()?;
    check(&selection)?;
    Ok(Some(selection))
}

pub fn save(
    root: &Path,
    migration: &MigrationDef,
    selection: &Selection,
    check: impl Fn(&Selection) -> Result<(), String>,
) -> Result<(), String> {
    selection.validate()?;
    check(selection)?;
    let folders: Vec<&str> = migration.data_dir.split('/').collect();
    let dir = crate::sa3_training::checked_folder(root, &folders)?;
    let path = dir.join(SELECTION_FILE);
    if std::fs::symlink_metadata(&path).is_ok()
        && path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err("Native runtime selection points outside the active storage profile".into());
    }
    crate::sa3_training::save(&path, selection)
}
