//! A service's move from its Python environment to a native runtime.
//!
//! The switch is recorded per storage profile, in the service's data folder,
//! as its native selection. Once a selection exists the service runs native,
//! and an interrupted cleanup must never silently restore Python, so the same
//! record journals cleanup progress. What is service-specific about a
//! migration lives with the service; this module holds what every one shares.
use crate::manifest::{MigrationDef, NativeDef, ServiceDef};
use crate::native_models::Catalog;
use crate::native_runtime::{NativeInstall, NativeLaunch};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

pub const SELECTION_FILE: &str = "native-runtime.json";

/// An asynchronous migration proof.
pub type Proof<'a, T> = Pin<Box<dyn Future<Output = Result<T, String>> + Send + 'a>>;

/// What a runtime proof works from: the installed runtime, launched exactly as
/// the service would be, but at the tier being proved.
pub struct Run<'a> {
    pub root: &'a Path,
    pub native: &'a NativeDef,
    pub installed: &'a NativeInstall,
    pub launch: &'a NativeLaunch,
    pub encoding: &'a str,
    pub progress: &'a (dyn Fn(&str) + Sync),
}

/// What the migration panel needs to know about a service.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub service_id: String,
    pub label: String,
    pub data_label: Option<String>,
    /// Generation tiers, best quality first; the first is the default.
    pub tiers: Vec<String>,
    /// The components each tier needs, which is what preparing it downloads.
    pub tier_components: std::collections::BTreeMap<String, Vec<String>>,
    /// Service-specific steps the panel offers, such as SA3's adapters.
    pub features: Vec<String>,
    /// Optional components each tier can add, such as jerry's finetunes.
    pub tier_options: std::collections::BTreeMap<String, Vec<String>>,
    /// What the panel calls those options.
    pub options_label: Option<String>,
}

/// Model choices that suit this machine's GPU, from what the runtime's install
/// probe measured.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct Recommendations {
    pub tier: Option<String>,
    pub components: Vec<String>,
}

/// What only a migrating service can answer. The shared inventory and cleanup
/// engine asks these; everything else it takes from the manifest block.
pub trait Service: Sync {
    fn target(&self) -> Target<'_>;

    /// The native models this service downloads, and where they live.
    fn models(&self) -> &'static Catalog;

    /// Generation tiers, best quality first; the first is the default.
    fn tiers(&self) -> &'static [&'static str] {
        &["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]
    }

    /// The model components generating at `tier` needs.
    fn tier_components(&self, tier: &str) -> Result<Vec<String>, String>;

    /// Optional components that work with `tier` without being needed by it,
    /// such as finetunes of the tier's base model.
    fn tier_options(&self, _tier: &str) -> Vec<String> {
        Vec::new()
    }

    /// What the models panel calls the tier options.
    fn options_label(&self) -> Option<&'static str> {
        None
    }

    /// What suits a GPU with `memory_bytes` of memory: the tier to prepare
    /// and the components to badge as recommended. By default nothing.
    fn recommendations(&self, _memory_bytes: u64) -> Recommendations {
        Recommendations::default()
    }

    /// Service-specific steps the migration panel offers beyond runtime,
    /// models and checks.
    fn features(&self) -> &'static [&'static str] {
        &[]
    }

    /// Components every model plan includes whatever the panel asked for,
    /// such as a correction the service currently applies.
    fn implied_components(&self) -> Vec<String> {
        Vec::new()
    }

    /// Keeps conflicting work out while a migration step runs, for as long as
    /// the returned guard lives. `cleanup` says which step it is.
    fn exclusive(&self, _cleanup: bool) -> Result<Option<Box<dyn std::any::Any + Send>>, String> {
        Ok(None)
    }

    /// Prove the installed runtime privately at `run.encoding`, returning the
    /// selection activation records.
    fn verify<'a>(&'a self, run: Run<'a>) -> Proof<'a, Selection>;

    /// What cleanup proves again before removing Python. Returns the
    /// fingerprint to record when it repeated the full proof; by default it
    /// always does.
    fn reverify<'a>(&'a self, run: Run<'a>, _selection: &'a Selection) -> Proof<'a, Option<String>> {
        Box::pin(async move { Ok(self.verify(run).await?.runtime_fingerprint) })
    }

    /// The ids the model manager knows the Python install's models by: those
    /// cleanup waits on while downloading and forgets afterwards. By default
    /// the manifest's legacy Hugging Face repositories.
    fn legacy_model_ids(&self) -> Vec<String> {
        self.target().def.legacy.hf_repos.iter().map(|repo| repo.repo.clone()).collect()
    }

    /// The service's contribution to an inventory: its selection, the user
    /// data its own records point at, and anything it shows as kept.
    fn extras(&self, root: &Path, hub: &Path) -> crate::migration_inventory::Extras;

    /// Whether `selection.encoding` names one of the service's tiers.
    fn check_selection(&self, selection: &Selection) -> Result<(), String> {
        self.tier_components(&selection.encoding).map(|_| ())
    }

    /// What still runs only in the Python service, such as its training, as
    /// a phrase for messages. While there is any, a fresh install still sets
    /// Python up and cleanup refuses to remove it, so switching to C++ never
    /// loses a feature.
    fn python_only_features(&self) -> Option<&'static str> {
        None
    }

    /// Why cleanup must not run yet, e.g. a legacy artifact whose native
    /// replacement has not been prepared. `downloaded` holds the legacy model
    /// ids the model manager reports as downloaded.
    fn cleanup_blocker(&self, _root: &Path, _hub: &Path, _downloaded: &[String]) -> Option<String> {
        None
    }

    /// Why the Python installation cannot be removed now: what only Python
    /// runs, then the service's own blocker. Cleanup asks this, never
    /// `cleanup_blocker` alone.
    fn retirement_blocker(&self, root: &Path, hub: &Path, downloaded: &[String]) -> Option<String> {
        self.python_still_needed().or_else(|| self.cleanup_blocker(root, hub, downloaded))
    }

    /// Why the Python installation stays whatever is on disk, for the
    /// migration preview and cleanup alike.
    fn python_still_needed(&self) -> Option<String> {
        self.python_only_features().map(|features| {
            format!(
                "{}'s {features} still run in Python, so its Python installation stays until they run in C++ too.",
                self.target().label()
            )
        })
    }

    /// Records in the data folder, relative to the runtime root, that storage
    /// transfer must refuse to carry if they no longer parse.
    fn authoritative_records(&self) -> Vec<String> {
        Vec::new()
    }

    /// Files the service's records point at and hash, so storage transfer
    /// copies them byte for byte instead of relocating paths inside them.
    fn opaque_files(&self, _root: &Path) -> Result<Vec<String>, String> {
        Ok(Vec::new())
    }

    /// Check that the records the service depends on parse in a profile that
    /// storage transfer has just copied them into.
    fn validate_records(&self, _root: &Path) -> Result<(), String> {
        Ok(())
    }

    /// Native jobs that must finish before the service may start: anything
    /// that rewrites what its server loads, or validates it.
    fn blocking_jobs(&self) -> &'static [&'static str] {
        &[]
    }

    /// Why the native runtime cannot start with this selection yet: by
    /// default, models its tier needs that are not on disk.
    fn native_start_blocker(&self, root: &Path, selection: Option<&Selection>) -> Option<String> {
        let tier = selection.map_or(self.tiers()[0], |selection| selection.encoding.as_str());
        match self.tier_components(tier) {
            Ok(ids) => self.models().missing(root, &ids),
            Err(error) => Some(error),
        }
    }

    /// The manifest's launch arguments adjusted for `encoding`: the selected
    /// tier, or the one being verified before it is selected.
    fn launch_args(&self, args: &[String], encoding: Option<&str>) -> Result<Vec<String>, String> {
        let tier = encoding.unwrap_or(self.tiers()[0]);
        self.tier_components(tier)?;
        with_encoding_arg(args, tier)
    }

    /// The panel's view of this service.
    fn profile(&self) -> Profile {
        let target = self.target();
        Profile {
            service_id: target.id.to_string(),
            label: target.label().to_string(),
            data_label: target.def.data_label.clone(),
            tiers: self.tiers().iter().map(|tier| tier.to_string()).collect(),
            tier_components: self
                .tiers()
                .iter()
                .filter_map(|tier| Some((tier.to_string(), self.tier_components(tier).ok()?)))
                .collect(),
            features: self.features().iter().map(|feature| feature.to_string()).collect(),
            tier_options: self
                .tiers()
                .iter()
                .map(|tier| (tier.to_string(), self.tier_options(tier)))
                .filter(|(_, options)| !options.is_empty())
                .collect(),
            options_label: self.options_label().map(str::to_string),
        }
    }

    /// Whatever keeps the Python service's client contract on its public port
    /// while the native server listens privately, bound before the server
    /// starts so a taken port fails the start instead of the first request.
    fn client_adapter(
        &self,
        _public_port: u16,
        _env: &[(String, String)],
        _root: &Path,
    ) -> Result<Option<Box<dyn ClientAdapter>>, String> {
        Ok(None)
    }

    /// Bring the files the native server reads at startup up to date, just
    /// before it starts.
    fn prepare_launch(&self, _root: &Path) -> Result<(), String> {
        Ok(())
    }
}

/// `args` with `--encoding` set to `tier`, added when the manifest left it out.
pub fn with_encoding_arg(args: &[String], tier: &str) -> Result<Vec<String>, String> {
    let mut args = args.to_vec();
    let indices: Vec<usize> = args
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value == "--encoding").then_some(index))
        .collect();
    match indices.as_slice() {
        [] => args.extend(["--encoding".into(), tier.to_ascii_lowercase()]),
        [index] if index + 1 < args.len() && !args[index + 1].starts_with("--") => {
            args[index + 1] = tier.to_ascii_lowercase();
        }
        _ => return Err("The native launch has an invalid encoding argument".into()),
    }
    Ok(args)
}

/// A bound client adapter, waiting for its native server to start.
pub trait ClientAdapter: Send {
    /// Start serving. The handle keeps the adapter alive, and it stops when
    /// the handle is dropped with the running service.
    fn start(self: Box<Self>) -> Box<dyn std::any::Any + Send>;
}

/// A migrating service as the shared engine sees it: its id, which names its
/// service folder, and its manifest block.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    pub id: &'a str,
    pub def: &'a MigrationDef,
    /// The native runtime's models folder (`native.modelsDir`), protected
    /// from cleanup.
    pub models_dir: Option<&'a str>,
}

impl<'a> Target<'a> {
    /// The target for a manifest entry, if it has a migration block.
    pub fn of(service: &'a ServiceDef) -> Option<Self> {
        Some(Self {
            id: &service.id,
            def: service.migration.as_ref()?,
            models_dir: service.native.as_ref().and_then(|native| native.models_dir.as_deref()),
        })
    }
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
        "foundation" => Some(&crate::foundation_migration::Foundation),
        "stable-audio" => Some(&crate::stable_audio_migration::StableAudio),
        "carey" => Some(&crate::carey_migration::Carey),
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
