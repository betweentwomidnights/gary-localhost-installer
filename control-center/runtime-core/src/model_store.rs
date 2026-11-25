//! Model storage shared between headless and desktop gary4local.
//!
//! Standard headless profiles keep their weights in desktop's `models/<service>`
//! folder, so one machine never holds two copies. A profile created before that
//! kept its own copy under `runtime_root/models/<service>`; the first run after
//! updating merges it in.
use crate::download::{sha256_file, DestinationLock};
use crate::headless::Config;
use std::path::{Path, PathBuf};

/// Desktop's selected storage, plus `models`. On a machine without desktop this
/// is desktop's default location, so a later desktop install finds the weights.
pub fn desktop_models_root() -> Result<PathBuf, String> {
    crate::lora_library::desktop_root().map(|root| root.join("models"))
}

/// A folder on the same file system as the profile's weights, for hard-linked
/// views and staged removals; neither works across file systems.
pub fn scratch_dir(config: &Config) -> PathBuf {
    match &config.models_root {
        Some(root) if config.shares_models() => root.join(".headless"),
        _ => config.runtime_root.join("tmp"),
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct MergeReport {
    /// Files moved into the shared folder, and their bytes.
    pub moved: (usize, u64),
    /// Copies the shared folder already had, deleted, and the bytes freed.
    pub duplicates: (usize, u64),
    /// Files that differ from the shared copy and from the catalog, set aside.
    pub conflicts: Vec<PathBuf>,
}

impl MergeReport {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    pub fn summary(&self, service: &str, shared: &Path) -> String {
        let gb = |bytes: u64| bytes as f64 / 1e9;
        let mut text = format!("{service} now uses the models folder shared with desktop: {}", shared.display());
        if self.moved.0 > 0 {
            text += &format!("\n  moved {} file(s), {:.1} GB", self.moved.0, gb(self.moved.1));
        }
        if self.duplicates.0 > 0 {
            text += &format!(
                "\n  removed {} duplicate(s) desktop already had, freeing {:.1} GB",
                self.duplicates.0,
                gb(self.duplicates.1)
            );
        }
        for path in &self.conflicts {
            text += &format!("\n  kept a copy that differs from desktop's: {}", path.display());
        }
        text
    }
}

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

/// Every regular file under `root`, relative to it. Symlinks are not followed
/// and are reported, so nothing outside the profile is moved.
fn files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(relative) = pending.pop() {
        for entry in std::fs::read_dir(root.join(&relative)).map_err(err)? {
            let entry = entry.map_err(err)?;
            let kind = entry.file_type().map_err(err)?;
            let path = relative.join(entry.file_name());
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                out.push(path);
            } else {
                return Err(format!("Cannot merge models: {} is not a regular file", root.join(path).display()));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The catalog hash for a file name, when this service publishes one.
fn catalog_sha256(config: &Config, relative: &Path) -> Option<String> {
    let name = relative.to_string_lossy().replace('\\', "/");
    crate::model_catalog::all_files(config.service)
        .find(|file| file.filename == name)
        .map(|file| file.sha256.to_ascii_lowercase())
}

/// Move `from` to `to`, by rename on one file system or by a verified copy.
async fn relocate(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).map_err(err)?;
    }
    match std::fs::rename(from, to) {
        Ok(()) => return Ok(()),
        Err(error) if error.kind() != std::io::ErrorKind::CrossesDevices => return Err(err(error)),
        Err(_) => {}
    }
    let staged = crate::download::partial_path(to).with_extension("merge");
    std::fs::copy(from, &staged).map_err(err)?;
    if sha256_file(&staged).await? != sha256_file(from).await? {
        let _ = std::fs::remove_file(&staged);
        return Err(format!("Copying {} changed its contents", from.display()));
    }
    std::fs::rename(&staged, to).map_err(err)?;
    std::fs::remove_file(from).map_err(err)
}

fn is_partial(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "partial" || ext == "lock")
}

/// Merge this profile's own weights into the shared folder. Identical copies
/// are deleted, missing ones moved, and a copy that differs is kept only if it
/// is the one matching the catalog; otherwise it is set aside under
/// `runtime_root/models-unmerged`. Each file is claimed against concurrent
/// downloads in desktop or another profile while it is handled.
pub async fn merge_private_models(config: &Config) -> Result<MergeReport, String> {
    let mut report = MergeReport::default();
    let private = config.private_models_dir();
    let shared = config.managed_models_dir();
    if !config.shares_models() || private == shared || !private.is_dir() {
        return Ok(report);
    }
    std::fs::create_dir_all(&shared).map_err(err)?;
    for relative in files(&private)? {
        let from = private.join(&relative);
        let bytes = std::fs::metadata(&from).map_err(err)?.len();
        if is_partial(&relative) {
            // An interrupted download: resume desktop's if it has one, else carry this one over.
            let to = shared.join(&relative);
            let weight = to.with_extension("");
            if relative.extension().is_some_and(|ext| ext == "lock") || to.exists() || weight.exists() {
                std::fs::remove_file(&from).map_err(err)?;
            } else {
                let _claim = DestinationLock::acquire(&weight).await?;
                relocate(&from, &to).await?;
            }
            continue;
        }
        let to = shared.join(&relative);
        let claim = DestinationLock::acquire(&to).await?;
        if !to.exists() {
            relocate(&from, &to).await?;
            report.moved.0 += 1;
            report.moved.1 += bytes;
            claim.finish();
            continue;
        }
        let ours = sha256_file(&from).await?;
        let theirs = sha256_file(&to).await?;
        let expected = catalog_sha256(config, &relative);
        if ours == theirs || expected.as_deref() == Some(theirs.as_str()) {
            std::fs::remove_file(&from).map_err(err)?;
            report.duplicates.0 += 1;
            report.duplicates.1 += bytes;
        } else if expected.as_deref() == Some(ours.as_str()) {
            // Desktop's copy is damaged and ours is the published file.
            std::fs::remove_file(&to).map_err(err)?;
            relocate(&from, &to).await?;
            report.moved.0 += 1;
            report.moved.1 += bytes;
        } else {
            let aside = config
                .runtime_root
                .join("models-unmerged")
                .join(config.service.id())
                .join(&relative);
            relocate(&from, &aside).await?;
            report.conflicts.push(aside);
        }
        claim.finish();
    }
    remove_empty_dirs(&private);
    Ok(report)
}

fn remove_empty_dirs(root: &Path) {
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                remove_empty_dirs(&entry.path());
            }
        }
    }
    let _ = std::fs::remove_dir(root);
}

/// The first run of a standard profile after updating: point it at desktop's
/// models and merge what it downloaded on its own. A profile with an explicit
/// models_dir or models_root keeps its choice. Returns what was merged, or
/// None when there was nothing to change. A running service defers this to the
/// next run, because its lease is held.
pub async fn adopt_desktop_models(path: &Path) -> Result<Option<String>, String> {
    let original = match std::fs::read(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(err(error)),
    };
    let mut body: serde_json::Value = serde_json::from_slice(&original).map_err(err)?;
    if body.get("models_dir").is_some() || body.get("models_root").is_some() {
        return Ok(None);
    }
    if desktop_models_root().is_err() {
        return Ok(None);
    }
    body["models_root"] = serde_json::json!("desktop");
    let config = Config::from_bytes(&serde_json::to_vec(&body).map_err(err)?).map_err(err)?;
    // Without a runtime root there is nothing of its own to merge, and no lease to take.
    let _lease = if config.runtime_root.exists() {
        match crate::headless::lock(&config) {
            Ok(lease) => Some(lease),
            Err(_) => return Ok(None),
        }
    } else {
        None
    };
    let report = merge_private_models(&config).await?;
    if std::fs::read(path).map_err(err)? != original {
        return Err("The service config changed while its models were merged. Run the command again.".into());
    }
    crate::headless::write_json(path, &body)?;
    Ok(Some(report.summary(config.service.id(), &config.managed_models_dir())))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(PathBuf);
    impl Temp {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
                "gary-model-store-{name}-{}",
                crate::headless::serial()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn config(runtime: &Path, models_root: Option<&Path>) -> Config {
        let mut body = serde_json::json!({"runtime_root":runtime,"device":"Vulkan0","public_port":28106,
            "model":"medium","encoding":"f16"});
        if let Some(root) = models_root {
            body["models_root"] = serde_json::json!(root);
        }
        Config::from_bytes(&serde_json::to_vec(&body).unwrap()).unwrap()
    }

    fn write(path: &Path, bytes: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn shared_profiles_resolve_to_the_shared_folder_and_its_scratch_space() {
        let temp = Temp::new("resolve");
        let shared = temp.0.join("desktop/models");
        let config = config(&temp.0.join("runtime"), Some(&shared));
        assert_eq!(config.models_dir, shared.join("sa3"));
        assert!(config.shares_models());
        assert_eq!(scratch_dir(&config), shared.join(".headless"));
        assert_eq!(config.private_models_dir(), temp.0.join("runtime/models/sa3"));

        let own = super::tests::config(&temp.0.join("runtime"), None);
        assert!(!own.shares_models());
        assert_eq!(own.models_dir, own.private_models_dir());
        assert_eq!(scratch_dir(&own), temp.0.join("runtime/tmp"));
    }

    #[test]
    fn models_root_refuses_external_weights_and_overlap_with_the_runtime_root() {
        let temp = Temp::new("refuse");
        let runtime = temp.0.join("runtime");
        let both = serde_json::json!({"runtime_root":runtime,"models_dir":temp.0.join("weights"),
            "models_root":temp.0.join("shared"),"device":"Vulkan0","public_port":28106,"model":"medium","encoding":"f16"});
        assert!(Config::from_bytes(&serde_json::to_vec(&both).unwrap()).is_err());
        let inside = serde_json::json!({"runtime_root":runtime,"models_root":runtime.join("models"),
            "device":"Vulkan0","public_port":28106,"model":"medium","encoding":"f16"});
        assert!(Config::from_bytes(&serde_json::to_vec(&inside).unwrap()).is_err());
    }

    #[tokio::test]
    async fn merging_moves_new_files_deletes_duplicates_and_sets_conflicts_aside() {
        let temp = Temp::new("merge");
        let runtime = temp.0.join("runtime");
        let shared_root = temp.0.join("desktop/models");
        let config = config(&runtime, Some(&shared_root));
        let private = config.private_models_dir();
        let shared = config.managed_models_dir();
        write(&private.join("only-headless.gguf"), "headless weights");
        write(&private.join("nested/also-headless.gguf"), "nested weights");
        write(&private.join("both.gguf"), "same bytes");
        write(&shared.join("both.gguf"), "same bytes");
        write(&private.join("differs.gguf"), "headless edition");
        write(&shared.join("differs.gguf"), "desktop edition");
        write(&private.join("finished.gguf.partial"), "half");
        write(&shared.join("finished.gguf"), "whole");
        write(&private.join("fresh.gguf.partial"), "resume me");

        let report = merge_private_models(&config).await.unwrap();

        assert_eq!(report.moved, (2, ("headless weights".len() + "nested weights".len()) as u64));
        assert_eq!(report.duplicates, (1, "same bytes".len() as u64));
        let aside = runtime.join("models-unmerged/sa3/differs.gguf");
        assert_eq!(report.conflicts, vec![aside.clone()]);
        assert_eq!(std::fs::read_to_string(shared.join("only-headless.gguf")).unwrap(), "headless weights");
        assert_eq!(std::fs::read_to_string(shared.join("nested/also-headless.gguf")).unwrap(), "nested weights");
        assert_eq!(std::fs::read_to_string(shared.join("differs.gguf")).unwrap(), "desktop edition");
        assert_eq!(std::fs::read_to_string(&aside).unwrap(), "headless edition");
        assert_eq!(std::fs::read_to_string(shared.join("finished.gguf")).unwrap(), "whole");
        assert!(!shared.join("finished.gguf.partial").exists());
        assert_eq!(std::fs::read_to_string(shared.join("fresh.gguf.partial")).unwrap(), "resume me");
        assert!(!private.exists(), "the merged private folder is removed");
        // A carried-over partial keeps its claim file for the download that finishes it.
        let leftover_locks: Vec<_> = files(&shared).unwrap().into_iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "lock"))
            .filter(|path| path != Path::new("fresh.gguf.partial.lock")).collect();
        assert!(leftover_locks.is_empty(), "{leftover_locks:?}");

        // Nothing left to do on the next run.
        assert!(merge_private_models(&config).await.unwrap().is_empty());
    }

    #[test]
    fn a_download_claim_excludes_other_claims_until_it_finishes() {
        let temp = Temp::new("claim");
        let dest = temp.0.join("weights.gguf");
        let first = DestinationLock::try_acquire(&dest).unwrap();
        let busy = DestinationLock::try_acquire(&dest).err().unwrap();
        assert!(busy.contains("being downloaded by another gary4local process"), "{busy}");
        first.finish();
        assert!(!temp.0.join("weights.gguf.partial.lock").exists());
        DestinationLock::try_acquire(&dest).unwrap().finish();
    }
}
