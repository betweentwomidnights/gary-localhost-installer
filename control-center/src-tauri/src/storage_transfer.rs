//! Copy a native installation before committing a storage change. Source files
//! remain recoverable; retries reuse byte-identical destination files. Never
//! follow redirects or silently merge unrelated storage profiles.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Instant, SystemTime};

static COPYING: AtomicBool = AtomicBool::new(false);
static NEXT: AtomicU64 = AtomicU64::new(0);
const RECEIPT: &str = ".gary-native-storage-transfer.json";
const PROFILE_ID: &str = ".gary-native-profile.json";
const JSON_LIMIT: u64 = 32 * 1024 * 1024;

pub struct Guard;
impl Guard {
    pub fn acquire() -> Result<Self, String> {
        COPYING
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| "A storage transfer is already running.")?;
        Ok(Self)
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        COPYING.store(false, Ordering::SeqCst);
    }
}
pub fn in_progress() -> bool {
    COPYING.load(Ordering::SeqCst)
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub message: String,
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Receipt {
    schema_version: u32,
    source_root: String,
    complete: bool,
    #[serde(default)]
    partial_file: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    relative: PathBuf,
    bytes: u64,
    modified: SystemTime,
}

struct Reporter<F> {
    callback: F,
    completed: u64,
    total: u64,
    last: Instant,
}
impl<F: FnMut(Progress)> Reporter<F> {
    fn report(&mut self, message: &str, bytes: u64, force: bool) {
        self.completed = self.completed.saturating_add(bytes).min(self.total);
        if force || self.last.elapsed().as_millis() >= 100 {
            (self.callback)(Progress {
                message: message.into(),
                completed_bytes: self.completed,
                total_bytes: self.total,
            });
            self.last = Instant::now();
        }
    }
}

fn metadata(path: &Path) -> Result<fs::Metadata, String> {
    fs::symlink_metadata(path).map_err(|e| format!("Cannot inspect {}: {e}", path.display()))
}

// Canonical equality also rejects Windows junctions that lead to another
// location *inside* the root, not merely ones that escape it.
fn checked(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("Invalid storage transfer path.".into());
    }
    let mut path = root.to_path_buf();
    for component in relative.components() {
        path.push(component);
        match fs::symlink_metadata(&path) {
            Ok(meta) => {
                if meta.file_type().is_symlink()
                    || path_key(&path.canonicalize().map_err(|e| e.to_string())?) != path_key(&path)
                {
                    return Err(format!(
                        "Storage transfer cannot follow a redirected path: {}",
                        path.display()
                    ));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("Cannot inspect {}: {e}", path.display())),
        }
    }
    Ok(path)
}

fn visit(root: &Path, relative: &Path, entries: &mut Vec<Entry>) -> Result<(), String> {
    let path = checked(root, relative)?;
    let meta = match fs::symlink_metadata(&path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.to_string()),
    };
    if meta.is_dir() {
        for child in fs::read_dir(&path).map_err(|e| e.to_string())? {
            let child = child.map_err(|e| e.to_string())?;
            visit(root, &relative.join(child.file_name()), entries)?;
        }
    } else if meta.is_file() {
        entries.push(Entry {
            relative: relative.into(),
            bytes: meta.len(),
            modified: meta.modified().map_err(|e| e.to_string())?,
        });
    } else {
        return Err(format!("Unsupported storage entry: {}", path.display()));
    }
    Ok(())
}

fn inventory(root: &Path) -> Result<Vec<Entry>, String> {
    let mut entries = Vec::new();
    // Only native assets and SA3's user data. Python environments are not
    // portable and remain governed by the existing Python storage workflow.
    for relative in [
        "models/sa3",
        "models/yuey",
        "native-runtimes",
        "sa3",
        "app_settings.json",
        "hf_token.txt",
    ] {
        visit(root, Path::new(relative), &mut entries)?;
    }
    let services = checked(root, Path::new("services"))?;
    if services.is_dir() {
        for child in fs::read_dir(services).map_err(|e| e.to_string())? {
            let child = child.map_err(|e| e.to_string())?;
            // A Python-only service may itself live on a redirected drive.
            // It is outside this transfer unless it contains native assets.
            if !child.path().join("native").exists() {
                continue;
            }
            let relative = Path::new("services").join(child.file_name()).join("native");
            visit(root, &relative, &mut entries)?;
        }
    }
    // Datasets and imported sources may be managed under another directory in
    // this storage profile. Carry referenced user files as well as the standard
    // SA3 layout, leaving genuinely external sources at their original paths.
    let mut index = 0;
    let mut followed = std::collections::BTreeSet::new();
    while index < entries.len() {
        let entry = entries[index].clone();
        index += 1;
        if !entry.relative.starts_with("sa3")
            || entry.relative.extension().and_then(|e| e.to_str()) != Some("json")
            || entry.bytes > JSON_LIMIT
        {
            continue;
        }
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(
            &fs::read(checked(root, &entry.relative)?).map_err(|e| e.to_string())?,
        ) {
            let mut references = Vec::new();
            referenced_paths(&value, root, &mut references);
            for relative in references {
                if relative.as_os_str().is_empty() {
                    return Err("A dataset uses the whole runtime folder. Choose a dedicated dataset folder before moving storage.".into());
                }
                if followed.insert(relative.clone()) {
                    visit(root, &relative, &mut entries)?;
                    if ["safetensors", "ckpt", "gguf"].iter().any(|e| {
                        relative
                            .extension()
                            .and_then(|s| s.to_str())
                            .is_some_and(|s| s.eq_ignore_ascii_case(e))
                    }) {
                        visit(root, &relative.with_extension("json"), &mut entries)?;
                    }
                }
            }
        }
    }
    entries.sort_by_key(|e| e.relative.clone());
    entries.dedup_by(|a, b| a.relative == b.relative);
    // Put the selection last: incomplete copying must not advertise a migrated
    // destination. The startup setting is committed only after all verification.
    entries.sort_by_key(|e| {
        (
            e.relative == Path::new("sa3/native-runtime.json"),
            e.relative.clone(),
        )
    });
    Ok(entries)
}

fn referenced_paths(value: &serde_json::Value, root: &Path, output: &mut Vec<PathBuf>) {
    match value {
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                if [
                    "dataset",
                    "promptConfig",
                    "sourcePath",
                    "configPath",
                    "promptsPath",
                    "path",
                    "statePath",
                    "safetensorsPath",
                ]
                .contains(&key.as_str())
                {
                    if let Some(text) = value.as_str() {
                        let prefix = path_key(root);
                        let key = path_key(Path::new(text));
                        if key == prefix
                            || key
                                .strip_prefix(&prefix)
                                .is_some_and(|s| s.starts_with('/'))
                        {
                            let normalized = text.replace('\\', "/");
                            let normalized = normalized.strip_prefix("//?/").unwrap_or(&normalized);
                            output.push(PathBuf::from(
                                normalized
                                    .get(prefix.len()..)
                                    .unwrap_or("")
                                    .trim_start_matches('/'),
                            ));
                        }
                    }
                }
                referenced_paths(value, root, output);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                referenced_paths(value, root, output);
            }
        }
        _ => {}
    }
}

fn path_key(path: &Path) -> String {
    let value = path.to_string_lossy().replace('\\', "/");
    let value = value
        .strip_prefix("//?/")
        .unwrap_or(&value)
        .trim_end_matches('/');
    if cfg!(windows) {
        value.to_ascii_lowercase()
    } else {
        value.into()
    }
}

fn rebase(value: &mut serde_json::Value, source: &Path, target: &Path) {
    match value {
        serde_json::Value::String(text) => {
            let key = path_key(Path::new(text));
            let prefix = path_key(source);
            if key == prefix
                || key
                    .strip_prefix(&prefix)
                    .is_some_and(|suffix| suffix.starts_with('/'))
            {
                // Preserve original suffix casing, including canonical Windows
                // extended-length paths. Never rewrite neighbouring/external roots.
                let normalized = text.replace('\\', "/");
                let normalized = normalized.strip_prefix("//?/").unwrap_or(&normalized);
                let suffix = normalized
                    .get(prefix.len()..)
                    .unwrap_or("")
                    .trim_start_matches('/');
                *text = target.join(suffix).to_string_lossy().into_owned();
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                rebase(value, source, target);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                rebase(value, source, target);
            }
        }
        _ => {}
    }
}

fn opaque_configs(root: &Path) -> Result<std::collections::BTreeSet<PathBuf>, String> {
    let catalog = crate::sa3_loras::read_catalog(root)?;
    let decoder = crate::sa3_decoder::read(root)?;
    let mut result = std::collections::BTreeSet::new();
    for entry in catalog.values().chain(decoder.iter()) {
        let sidecar = Path::new(&entry.source_path).with_extension("json");
        for text in entry
            .config_path
            .iter()
            .cloned()
            .chain(entry.legacy_export.iter().map(|e| e.config_path.clone()))
            .chain(std::iter::once(sidecar.to_string_lossy().into_owned()))
        {
            let mut references = Vec::new();
            referenced_paths(
                &serde_json::json!({"configPath":text}),
                root,
                &mut references,
            );
            result.extend(references);
        }
    }
    Ok(result)
}

fn transformed(
    entry: &Entry,
    source: &Path,
    target: &Path,
    opaque: &std::collections::BTreeSet<PathBuf>,
) -> Result<Option<Vec<u8>>, String> {
    // Imported adapter configurations retain their exact content and recorded
    // hash. Relocate the catalog pointer, not the configuration's contents.
    if opaque.contains(&entry.relative) {
        return Ok(None);
    }
    // JSON under SA3 is catalog/history/settings, including trainer sidecars.
    // Model metadata and bundled runtime JSON are byte-for-byte artifacts.
    if (!entry.relative.starts_with("sa3") && entry.relative != Path::new("app_settings.json"))
        || entry.relative.extension().and_then(|e| e.to_str()) != Some("json")
    {
        return Ok(None);
    }
    if entry.bytes > JSON_LIMIT {
        return Err(format!(
            "SA3 metadata is too large to relocate: {}",
            entry.relative.display()
        ));
    }
    let bytes = fs::read(checked(source, &entry.relative)?).map_err(|e| e.to_string())?;
    // User data can include non-JSON files with a .json suffix. Preserve those
    // verbatim, but never silently carry a malformed authoritative catalog.
    let mut value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(e) => {
            if [
                "sa3/native-runtime.json",
                "sa3/lora_catalog.json",
                "sa3/lora_registry.json",
                "sa3/native-loras/catalog.json",
            ]
            .iter()
            .any(|p| entry.relative == Path::new(p))
            {
                return Err(format!(
                    "Cannot relocate invalid SA3 catalog {}: {e}",
                    entry.relative.display()
                ));
            }
            return Ok(None);
        }
    };
    let original = value.clone();
    rebase(&mut value, source, target);
    if value == original {
        Ok(Some(bytes))
    } else {
        serde_json::to_vec_pretty(&value)
            .map(Some)
            .map_err(|e| e.to_string())
    }
}

fn digest<F: FnMut(Progress)>(
    path: &Path,
    reporter: &mut Reporter<F>,
    label: &str,
) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        reporter.report(label, count as u64, false);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let part = path.with_file_name(format!(
        ".gary-transfer-{}-{}.part",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    atomic_write_at(path, bytes, &part)
}

fn atomic_write_at(path: &Path, bytes: &[u8], part: &Path) -> Result<(), String> {
    let mut created = false;
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&part)
            .map_err(|e| e.to_string())?;
        created = true;
        file.write_all(bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        // Persist replacement atomically, including Windows destination files.
        drop(file);
        fs::rename(&part, path).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if result.is_err() && created {
        let _ = fs::remove_file(&part);
    }
    result
}

fn receipt(
    target: &Path,
    source: &Path,
    complete: bool,
    partial_file: Option<&Path>,
) -> Result<(), String> {
    let path = checked(target, Path::new(RECEIPT))?;
    atomic_write(
        &path,
        &serde_json::to_vec_pretty(&Receipt {
            schema_version: 1,
            source_root: source.to_string_lossy().into(),
            complete,
            partial_file: partial_file.map(|p| p.to_string_lossy().into()),
        })
        .map_err(|e| e.to_string())?,
    )
}

fn profile_id(root: &Path) -> Result<Option<String>, String> {
    let path = checked(root, Path::new(PROFILE_ID))?;
    if !path.exists() {
        return Ok(None);
    }
    let id: String = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Invalid native storage identity: {e}"))?;
    if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid native storage identity.".into());
    }
    Ok(Some(id))
}

#[cfg(windows)]
fn check_space(target: &Path, bytes: u64) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    let path: Vec<u16> = target.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0;
    let result = unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            path.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if result == 0 {
        return Err(format!(
            "Cannot check destination free space: {}",
            std::io::Error::last_os_error()
        ));
    }
    if available < bytes {
        return Err(format!("Not enough free space in the selected folder. The transfer needs about {:.1} GB; {:.1} GB is available. Current storage is unchanged.", bytes as f64 / 1e9, available as f64 / 1e9));
    }
    Ok(())
}
#[cfg(not(windows))]
fn check_space(_target: &Path, _bytes: u64) -> Result<(), String> {
    Ok(())
}

pub fn run(source: &Path, target: &Path, callback: impl FnMut(Progress)) -> Result<(), String> {
    let source = source
        .canonicalize()
        .map_err(|e| format!("Cannot open current storage: {e}"))?;
    let target = target
        .canonicalize()
        .map_err(|e| format!("Cannot open selected storage: {e}"))?;
    if source == target {
        return Ok(());
    }
    if source.starts_with(&target) || target.starts_with(&source) {
        return Err("Choose a storage folder outside the current storage; parent and nested folders cannot be transferred safely.".into());
    }
    let mut reporter = Reporter {
        callback,
        completed: 0,
        total: 0,
        last: Instant::now(),
    };
    reporter.report(
        "Checking native runtimes, models and LoRA history…",
        0,
        true,
    );
    let entries = inventory(&source)?;
    if entries.is_empty() {
        return Ok(());
    }
    let source_selection = crate::sa3_runtime::read(&source)?;
    let opaque = opaque_configs(&source)?;
    let target_selection = crate::sa3_runtime::read(&target)?;
    let source_id = profile_id(&source)?.unwrap_or_else(|| {
        format!(
            "{:x}",
            Sha256::digest(
                format!(
                    "{}-{:?}-{}-{}",
                    source.display(),
                    SystemTime::now(),
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                )
                .as_bytes()
            )
        )
    });
    let target_id = profile_id(&target)?;
    let same_profile = target_id.as_ref() == Some(&source_id)
        || (target_id.is_none()
            && source_selection
                .as_ref()
                .zip(target_selection.as_ref())
                .is_some_and(|(a, b)| a.activated_at == b.activated_at));
    if target_id.is_some() && !same_profile {
        return Err("The selected folder belongs to a different native storage profile. Choose an empty folder.".into());
    }
    if target_selection.is_some() && !same_profile {
        return Err("The selected folder belongs to a different SA3 migration. Choose an empty folder to keep both installations safe.".into());
    }
    let receipt_path = checked(&target, Path::new(RECEIPT))?;
    if receipt_path.exists() {
        let prior: Receipt =
            serde_json::from_slice(&fs::read(receipt_path).map_err(|e| e.to_string())?)
                .map_err(|e| format!("Cannot read storage transfer journal: {e}"))?;
        if prior.schema_version != 1
            || (!crate::storage::paths_equivalent(Path::new(&prior.source_root), &source)
                && !same_profile)
        {
            return Err("The selected folder contains a transfer from another storage profile. Choose an empty folder.".into());
        }
        if let Some(partial) = prior.partial_file {
            let partial = Path::new(&partial);
            let name = partial
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or("Invalid interrupted storage transfer path.")?;
            if !name.starts_with(".gary-transfer-") || !name.ends_with(".part") {
                return Err("Invalid interrupted storage transfer path.".into());
            }
            let partial = checked(&target, partial)?;
            if partial.exists() {
                if !metadata(&partial)?.is_file() {
                    return Err("Interrupted transfer file is not a regular file.".into());
                }
                fs::remove_file(partial)
                    .map_err(|e| format!("Cannot clear interrupted storage copy: {e}"))?;
            }
        }
    }
    // Check *every* conflict before creating files. A destination from the same
    // SA3 lineage may update its JSON catalogs/settings; immutable data cannot
    // be overwritten. Incomplete same-source retries reuse verified files.
    let mut target_before = std::collections::BTreeMap::new();
    let mut required = JSON_LIMIT;
    for entry in &entries {
        let destination = checked(&target, &entry.relative)?;
        if destination.exists() && !metadata(&destination)?.is_file() {
            return Err(format!(
                "Storage destination conflicts with a file: {}",
                destination.display()
            ));
        }
        if destination.is_file() {
            let meta = metadata(&destination)?;
            target_before.insert(
                entry.relative.clone(),
                Some((meta.len(), meta.modified().map_err(|e| e.to_string())?)),
            );
            let bytes = transformed(entry, &source, &target, &opaque)?;
            let expected = if let Some(bytes) = &bytes {
                format!("{:x}", Sha256::digest(bytes))
            } else {
                digest(
                    &checked(&source, &entry.relative)?,
                    &mut reporter,
                    "Checking existing files…",
                )?
            };
            let actual = digest(&destination, &mut reporter, "Checking existing files…")?;
            if actual != expected
                && !(same_profile
                    && !opaque.contains(&entry.relative)
                    && ((entry.relative.starts_with("sa3")
                        && entry.relative.extension().and_then(|e| e.to_str()) == Some("json"))
                        || entry.relative == Path::new("app_settings.json")))
            {
                return Err(format!("The selected folder contains a different copy of {}. Choose an empty folder; Gary will not overwrite it.", entry.relative.display()));
            }
        } else {
            target_before.insert(entry.relative.clone(), None);
            required = required.saturating_add(entry.bytes);
        }
    }
    check_space(&target, required)?;
    reporter.completed = 0;
    reporter.total = entries
        .iter()
        .map(|e| e.bytes)
        .sum::<u64>()
        .saturating_mul(2);
    atomic_write(
        &checked(&source, Path::new(PROFILE_ID))?,
        &serde_json::to_vec(&source_id).map_err(|e| e.to_string())?,
    )?;
    receipt(&target, &source, false, None)?;
    let mut verified_target = Vec::new();
    for entry in &entries {
        let original = checked(&source, &entry.relative)?;
        let destination = checked(&target, &entry.relative)?;
        let current = match fs::symlink_metadata(&destination) {
            Ok(meta) => Some((meta.len(), meta.modified().map_err(|e| e.to_string())?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.to_string()),
        };
        if target_before[&entry.relative] != current {
            return Err("The selected folder changed during transfer. Close other Gary instances and retry.".into());
        }
        fs::create_dir_all(destination.parent().unwrap())
            .map_err(|e| format!("Cannot create storage destination: {e}"))?;
        checked(&target, &entry.relative)?;
        let label = format!("Copying {}", entry.relative.display());
        reporter.report(&label, 0, true);
        let bytes = transformed(entry, &source, &target, &opaque)?;
        let expected = if let Some(bytes) = &bytes {
            let expected = format!("{:x}", Sha256::digest(bytes));
            if !destination.exists() || digest(&destination, &mut reporter, &label)? != expected {
                atomic_write(&destination, bytes)?;
            }
            reporter.report(&label, entry.bytes, false);
            expected
        } else if destination.exists() {
            digest(&original, &mut reporter, &label)?
        } else {
            let part = destination.with_file_name(format!(
                ".gary-transfer-{}-{}.part",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut created = false;
            let result: Result<String, String> = (|| {
                let mut input = File::open(&original).map_err(|e| e.to_string())?;
                let mut output = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&part)
                    .map_err(|e| e.to_string())?;
                created = true;
                receipt(
                    &target,
                    &source,
                    false,
                    Some(part.strip_prefix(&target).map_err(|e| e.to_string())?),
                )?;
                let mut hash = Sha256::new();
                let mut buffer = vec![0; 1024 * 1024];
                loop {
                    let count = input.read(&mut buffer).map_err(|e| e.to_string())?;
                    if count == 0 {
                        break;
                    }
                    output.write_all(&buffer[..count]).map_err(|e| {
                        format!(
                            "Cannot copy {} (check free space): {e}",
                            entry.relative.display()
                        )
                    })?;
                    hash.update(&buffer[..count]);
                    reporter.report(&label, count as u64, false);
                }
                output.sync_all().map_err(|e| e.to_string())?;
                drop(output);
                // create_new staging and conflict preflight prevent replacing a
                // pre-existing file. The destination is rechecked before rename.
                if destination.exists() {
                    return Err("Storage destination changed during transfer.".into());
                }
                fs::rename(&part, &destination).map_err(|e| e.to_string())?;
                Ok(format!("{:x}", hash.finalize()))
            })();
            if result.is_err() && created {
                let _ = fs::remove_file(part);
            }
            result?
        };
        reporter.report(&format!("Verifying {}", entry.relative.display()), 0, true);
        if digest(&destination, &mut reporter, "Verifying copied files…")? != expected {
            return Err(format!(
                "Copied file verification failed: {}. Current storage is unchanged.",
                entry.relative.display()
            ));
        }
        let verified = metadata(&destination)?;
        verified_target.push((
            entry.relative.clone(),
            verified.len(),
            verified.modified().map_err(|e| e.to_string())?,
        ));
        let after = metadata(&original)?;
        if after.len() != entry.bytes
            || after.modified().map_err(|e| e.to_string())? != entry.modified
        {
            return Err(
                "Current storage changed during transfer. Close other Gary instances and retry."
                    .into(),
            );
        }
    }
    if inventory(&source)? != entries {
        return Err(
            "Current storage changed during transfer. Retry before switching storage.".into(),
        );
    }
    for (relative, bytes, modified) in verified_target {
        let current = metadata(&checked(&target, &relative)?)?;
        if current.len() != bytes || current.modified().map_err(|e| e.to_string())? != modified {
            return Err(
                "Copied files changed before storage could be switched. Retry the transfer.".into(),
            );
        }
    }
    crate::sa3_runtime::read(&target)?;
    crate::sa3_loras::read_catalog(&target)?;
    atomic_write(
        &checked(&target, Path::new(PROFILE_ID))?,
        &serde_json::to_vec(&source_id).map_err(|e| e.to_string())?,
    )?;
    receipt(&target, &source, true, None)?;
    reporter.completed = reporter.total;
    reporter.report(
        "Native runtimes, models and LoRAs copied and verified. Restart to use the new storage.",
        0,
        true,
    );
    Ok(())
}

/// Generic legacy cleanup predates GGUF storage. Never offer blanket model
/// deletion while a native model tree is present in the former profile.
pub fn contains_native_models(root: &Path) -> bool {
    ["models/sa3", "models/yuey"]
        .iter()
        .any(|p| root.join(p).exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "gary-storage-transfer-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir(&path).unwrap();
            Self(path.canonicalize().unwrap())
        }
        fn root(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::create_dir_all(&path).unwrap();
            path
        }
        fn write(root: &Path, relative: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
            let path = root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, bytes).unwrap();
            path
        }
        fn native(root: &Path) {
            crate::sa3_runtime::save(
                root,
                &crate::sa3_runtime::Selection {
                    schema_version: 1,
                    encoding: "F16".into(),
                    verified_release: "v0.1.3".into(),
                    backend: "vulkan".into(),
                    activated_at: 123,
                    cleanup_complete: true,
                    cleanup_errors: vec![],
                },
            )
            .unwrap();
            Self::write(root, "models/sa3/training-base.gguf", b"training weights");
            Self::write(root, "models/yuey/q4.gguf", b"quantized weights");
            Self::write(
                root,
                "native-runtimes/cudart-12.8/cudart.dll",
                b"shared runtime",
            );
            Self::write(root, "services/sa3/native/sa3-server.exe", b"sa3");
            Self::write(root, "services/yuey/native/yue2-server.exe", b"yuey");
            let adapter = Self::write(root, "sa3/native-loras/test.gguf", b"adapter");
            let source = Self::write(
                root,
                "sa3/training/jobs/native-test/run/adapter-step-1.gguf",
                b"adapter",
            );
            let state = Self::write(
                root,
                "sa3/training/jobs/native-test/run/trainer-state-step-1.gguf",
                b"optimizer",
            );
            Self::write(root, "sa3/native-loras/catalog.json", json!({"test": {
                "nativeOnly":true, "name":"test", "sourcePath":source, "configPath":null,
                "sourceSha256":null,"configSha256":null,"converterSha256":null,
                "nativePath":adapter,"nativeSha256":null,"strength":0.5,"error":null,
                "promptsPath":null,"trainingCheckpoints":[{"jobId":"native-test","step":1,"epoch":1,"path":source,"statePath":state}],"legacyExport":null
            }}).to_string());
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn transfer_carries_native_training_history_models_shared_pack_and_managed_sources() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::native(&source);
        let dataset = Fixture::write(&source, "datasets/audio/song.wav", b"audio")
            .parent()
            .unwrap()
            .to_path_buf();
        Fixture::write(&source, "datasets/audio/song.txt", b"caption");
        let external = fixture.root("external").join("prompts");
        let options = json!({"dataset":dataset,"promptConfig":external,"resume":source.join("sa3/training/jobs/native-test/run/trainer-state-step-1.gguf")});
        Fixture::write(
            &source,
            "sa3/training/jobs/native-test/native-options.json",
            options.to_string(),
        );
        Fixture::write(
            &source,
            "app_settings.json",
            json!({"datasetPath":dataset}).to_string(),
        );
        Fixture::write(&source, "services/sa3/env/python.exe", b"not portable");
        let before = inventory(&source).unwrap();
        let mut progress = Vec::new();
        run(&source, &target, |p| progress.push(p)).unwrap();
        let selection = crate::sa3_runtime::read(&target).unwrap().unwrap();
        assert!(selection.cleanup_complete);
        assert_eq!(selection.backend, "vulkan");
        assert!(target.join("models/sa3/training-base.gguf").is_file());
        assert!(target
            .join("native-runtimes/cudart-12.8/cudart.dll")
            .is_file());
        assert!(target.join("models/yuey/q4.gguf").is_file());
        assert!(!target.join("services/sa3/env").exists());
        assert_eq!(inventory(&source).unwrap(), before, "source remains intact");
        let catalog = crate::sa3_loras::read_catalog(&target).unwrap();
        assert_eq!(
            Path::new(catalog["test"].native_path.as_ref().unwrap()),
            target.join("sa3/native-loras/test.gguf")
        );
        assert!(Path::new(&catalog["test"].source_path).starts_with(&target));
        assert!(
            Path::new(&catalog["test"].training_checkpoints[0].state_path).starts_with(&target)
        );
        let relocated: serde_json::Value = serde_json::from_slice(
            &fs::read(target.join("sa3/training/jobs/native-test/native-options.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            Path::new(relocated["dataset"].as_str().unwrap()),
            target.join("datasets/audio")
        );
        assert_eq!(
            Path::new(relocated["promptConfig"].as_str().unwrap()),
            external
        );
        assert!(target.join("datasets/audio/song.txt").is_file());
        let last = progress.last().unwrap();
        assert!(last.total_bytes > 0);
        assert_eq!(last.completed_bytes, last.total_bytes);
        let mut manifest: crate::manifest::Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .unwrap();
        manifest.resolve_native_bundles().unwrap();
        let manager = crate::service_manager::ServiceManager::new(manifest.services, target);
        assert!(
            manager
                .get_service_info()
                .iter()
                .find(|s| s.id == "sa3")
                .unwrap()
                .sa3_migration_complete
        );
    }

    #[test]
    fn repeat_transfer_and_return_to_default_reuse_files_and_update_same_profile_metadata() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        let third = fixture.root("custom-2");
        Fixture::native(&source);
        run(&source, &target, |_| {}).unwrap();
        let weight_time = metadata(&target.join("models/sa3/training-base.gguf"))
            .unwrap()
            .modified()
            .unwrap();
        run(&source, &target, |_| {}).unwrap();
        assert_eq!(
            metadata(&target.join("models/sa3/training-base.gguf"))
                .unwrap()
                .modified()
                .unwrap(),
            weight_time
        );
        let mut selection = crate::sa3_runtime::read(&target).unwrap().unwrap();
        selection.backend = "cuda".into();
        crate::sa3_runtime::save(&target, &selection).unwrap();
        run(&target, &third, |_| {}).unwrap();
        run(&third, &source, |_| {}).unwrap();
        assert_eq!(
            crate::sa3_runtime::read(&source).unwrap().unwrap().backend,
            "cuda"
        );
        assert!(
            Path::new(&crate::sa3_loras::read_catalog(&source).unwrap()["test"].source_path)
                .starts_with(&source)
        );
    }

    #[test]
    fn transfer_rejects_conflicts_unrelated_profiles_and_overlapping_roots() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::native(&source);
        Fixture::write(
            &target,
            "models/sa3/training-base.gguf",
            b"different user file",
        );
        assert!(run(&source, &target, |_| {})
            .unwrap_err()
            .contains("different copy"));
        assert!(!target.join(RECEIPT).exists());
        assert_eq!(
            fs::read(target.join("models/sa3/training-base.gguf")).unwrap(),
            b"different user file"
        );
        let mut selection = crate::sa3_runtime::read(&source).unwrap().unwrap();
        selection.activated_at = 999;
        crate::sa3_runtime::save(&target, &selection).unwrap();
        assert!(run(&source, &target, |_| {})
            .unwrap_err()
            .contains("different SA3"));
        let nested = source.join("nested");
        fs::create_dir(&nested).unwrap();
        assert!(run(&source, &nested, |_| {})
            .unwrap_err()
            .contains("nested"));
        assert!(run(&source, &fixture.0, |_| {}).is_err());
        run(&source, &source, |_| {}).unwrap();
    }

    #[test]
    fn transfer_detects_source_changes_and_corrupt_destination_before_commit() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::native(&source);
        let mut changed = false;
        assert!(run(&source, &target, |p| {
            if !changed && p.message.starts_with("Copying") {
                changed = true;
                Fixture::write(&source, "sa3/late-user-file.txt", b"new");
            }
        })
        .unwrap_err()
        .contains("changed"));
        let receipt: Receipt =
            serde_json::from_slice(&fs::read(target.join(RECEIPT)).unwrap()).unwrap();
        assert!(!receipt.complete);
        run(&source, &target, |_| {}).unwrap();
        assert!(target.join("sa3/late-user-file.txt").exists());
        let target2 = fixture.root("custom-2");
        let mut corrupted = false;
        let error = run(&source, &target2, |p| {
            if !corrupted && p.message.starts_with("Verifying models") {
                corrupted = true;
                Fixture::write(&target2, "models/sa3/training-base.gguf", b"corrupt");
            }
        })
        .unwrap_err();
        assert!(error.contains("verification failed"), "{error}");
        assert!(!target2.join("sa3/native-runtime.json").exists());
        assert_eq!(
            fs::read(source.join("models/sa3/training-base.gguf")).unwrap(),
            b"training weights"
        );
    }

    #[test]
    fn rebasing_handles_windows_prefixes_case_and_preserves_neighbouring_roots() {
        let mut value = json!({"path":"\\\\?\\C:\\Gary\\sa3\\LoRA.gguf","dataset":"c:/gary/dataset","external":"C:/Gary-other/song.wav","prompt":"music with C:/Gary in the middle"});
        rebase(&mut value, Path::new("C:/Gary"), Path::new("D:/New"));
        assert_eq!(
            path_key(Path::new(value["path"].as_str().unwrap())),
            "d:/new/sa3/lora.gguf"
        );
        assert_eq!(value["external"], "C:/Gary-other/song.wav");
        assert_eq!(value["prompt"], "music with C:/Gary in the middle");
    }

    #[cfg(windows)]
    #[test]
    fn locked_source_can_be_retried_and_junctions_are_rejected() {
        use std::os::windows::fs::OpenOptionsExt;
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::native(&source);
        let locked = OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(source.join("models/sa3/training-base.gguf"))
            .unwrap();
        assert!(run(&source, &target, |_| {}).is_err());
        drop(locked);
        run(&source, &target, |_| {}).unwrap();
        let external = fixture.root("external");
        Fixture::write(&external, "do-not-touch", b"protected");
        let junction = source.join("sa3/redirect");
        let result = std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(&junction)
            .arg(&external)
            .output()
            .unwrap();
        assert!(result.status.success());
        let target2 = fixture.root("custom-2");
        assert!(run(&source, &target2, |_| {})
            .unwrap_err()
            .contains("redirected"));
        fs::remove_dir(junction).unwrap();
        assert_eq!(
            fs::read(external.join("do-not-touch")).unwrap(),
            b"protected"
        );
    }

    #[test]
    fn legacy_cleanup_excludes_native_model_tree() {
        let fixture = Fixture::new();
        let legacy = fixture.root("legacy");
        let active = fixture.root("active");
        let hf = fixture.root("hf");
        Fixture::native(&legacy);
        let items = crate::build_legacy_cleanup_items(&active, &legacy, &hf);
        assert!(!items.iter().any(|e| e.id == "legacy-models"));
        assert!(legacy.join("models/sa3/training-base.gguf").is_file());
    }

    #[test]
    fn interrupted_copy_discards_only_journaled_partial_and_preserves_unknown_files() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::native(&source);
        let partial = Path::new("models/sa3/.gary-transfer-interrupted.part");
        Fixture::write(&target, partial.to_str().unwrap(), b"incomplete");
        let unknown = Fixture::write(&target, "models/sa3/.gary-transfer-user.part", b"keep me");
        receipt(&target, &source, false, Some(partial)).unwrap();
        run(&source, &target, |_| {}).unwrap();
        assert!(!target.join(partial).exists());
        assert_eq!(fs::read(unknown).unwrap(), b"keep me");
        let journal: Receipt =
            serde_json::from_slice(&fs::read(target.join(RECEIPT)).unwrap()).unwrap();
        assert!(journal.complete);
        assert!(journal.partial_file.is_none());
    }

    #[test]
    fn temporary_name_collision_never_removes_an_existing_file() {
        let fixture = Fixture::new();
        let existing = Fixture::write(&fixture.0, ".gary-transfer-existing.part", b"user file");
        let destination = fixture.0.join("catalog.json");
        assert!(atomic_write_at(&destination, b"new metadata", &existing).is_err());
        assert_eq!(fs::read(existing).unwrap(), b"user file");
        assert!(!destination.exists());
    }

    #[test]
    fn yuey_only_profile_can_return_to_default_with_updated_settings() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::write(&source, "models/yuey/q8.gguf", b"weights");
        Fixture::write(&source, "app_settings.json", b"{\"value\":1}");
        run(&source, &target, |_| {}).unwrap();
        Fixture::write(&target, "app_settings.json", b"{\"value\":2}");
        run(&target, &source, |_| {}).unwrap();
        assert_eq!(
            fs::read(source.join("app_settings.json")).unwrap(),
            b"{\"value\":2}"
        );
        assert!(crate::sa3_runtime::read(&target).unwrap().is_none());
    }

    #[test]
    fn imported_adapter_sidecar_is_carried_without_changing_its_recorded_hash() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::native(&source);
        let adapter = Fixture::write(&source, "imports/adapter.safetensors", b"tensor contents");
        let configuration =
            json!({"rank":64,"originalTrainingFolder":source.join("sa3/training")}).to_string();
        Fixture::write(&source, "imports/adapter.json", &configuration);
        let path = source.join("sa3/native-loras/catalog.json");
        let mut catalog: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        catalog["test"]["sourcePath"] = json!(adapter);
        catalog["test"]["configSha256"] =
            json!(format!("{:x}", Sha256::digest(configuration.as_bytes())));
        fs::write(&path, catalog.to_string()).unwrap();
        run(&source, &target, |_| {}).unwrap();
        assert_eq!(
            fs::read_to_string(target.join("imports/adapter.json")).unwrap(),
            configuration
        );
        assert!(target.join("imports/adapter.safetensors").is_file());
        let relocated = crate::sa3_loras::read_catalog(&target).unwrap();
        assert_eq!(
            relocated["test"].config_sha256,
            Some(format!("{:x}", Sha256::digest(configuration.as_bytes())))
        );
    }

    #[cfg(windows)]
    #[test]
    fn destination_junction_and_existing_metadata_race_are_rejected() {
        let fixture = Fixture::new();
        let source = fixture.root("default");
        let target = fixture.root("custom");
        Fixture::native(&source);
        let external = fixture.root("external");
        let junction = target.join("models");
        assert!(std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(&junction)
            .arg(&external)
            .output()
            .unwrap()
            .status
            .success());
        assert!(run(&source, &target, |_| {})
            .unwrap_err()
            .contains("redirected"));
        fs::remove_dir(junction).unwrap();
        run(&source, &target, |_| {}).unwrap();
        let mut changed = false;
        let error = run(&source, &target, |p| {
            if !changed && p.message.starts_with("Copying") {
                changed = true;
                Fixture::write(&target, "sa3/native-runtime.json", b"{\"other\":true}");
            }
        })
        .unwrap_err();
        assert!(error.contains("changed"), "{error}");
        assert_eq!(
            fs::read(target.join("sa3/native-runtime.json")).unwrap(),
            b"{\"other\":true}"
        );
    }

    #[test]
    fn completed_storage_change_blocks_launches_and_installs_until_restart() {
        let fixture = Fixture::new();
        let root = fixture.root("runtime");
        let mut manifest: crate::manifest::Manifest =
            serde_json::from_str(include_str!("../../../services/manifests/services.json"))
                .unwrap();
        manifest.resolve_native_bundles().unwrap();
        let mut manager = crate::service_manager::ServiceManager::new(manifest.services, root);
        manager.set_storage_change_pending(true);
        assert!(manager.get_service_info().iter().all(|s| s
            .start_blocker
            .as_deref()
            .is_some_and(|s| s.contains("restart"))));
        assert!(manager.get_build_info("gary").is_err());
        assert!(manager.get_native_build_info("sa3").is_err());
        assert!(manager
            .begin_native_workload("sa3", "test", "training")
            .is_err());
        manager.set_storage_change_pending(false);
        assert!(manager.get_build_info("gary").is_ok());
    }

    #[tokio::test]
    #[ignore = "copies a real native profile snapshot; original profile is read-only"]
    async fn real_profile_snapshot_transfers_models_adapters_and_runnable_native_tools() {
        let real = PathBuf::from(std::env::var_os("GARY4LOCAL_STORAGE_TRANSFER_SOURCE").unwrap())
            .canonicalize()
            .unwrap();
        let before = inventory(&real).unwrap();
        let opaque = opaque_configs(&real).unwrap();
        let fixture = Fixture::new();
        let source = fixture.root("snapshot");
        let target = fixture.root("custom");
        let original_id = profile_id(&real).unwrap();
        let start = Instant::now();
        for entry in &before {
            let destination = source.join(&entry.relative);
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            if let Some(bytes) = transformed(entry, &real, &source, &opaque).unwrap() {
                fs::write(destination, bytes).unwrap();
            } else {
                fs::hard_link(real.join(&entry.relative), destination).unwrap();
            }
        }
        println!(
            "Isolated hard-link snapshot: {} files, {} bytes",
            before.len(),
            before.iter().map(|e| e.bytes).sum::<u64>()
        );
        let mut last = Instant::now();
        run(&source, &target, |p| {
            if last.elapsed().as_secs() >= 15 {
                println!(
                    "{} / {} bytes: {}",
                    p.completed_bytes, p.total_bytes, p.message
                );
                last = Instant::now();
            }
        })
        .unwrap();
        assert!(
            crate::sa3_runtime::read(&target)
                .unwrap()
                .unwrap()
                .cleanup_complete
        );
        assert!(crate::sa3_migration::native_training_base_present(&target));
        let entries = crate::sa3_loras::read_catalog(&target).unwrap();
        assert_eq!(
            entries.len(),
            crate::sa3_loras::read_catalog(&real).unwrap().len()
        );
        for entry in entries.values() {
            crate::sa3_loras::verify_source_config(entry).await.unwrap();
            crate::sa3_loras::verify_legacy_export(&target, entry)
                .await
                .unwrap();
            if let (Some(path), Some(hash)) = (&entry.native_path, &entry.native_sha256) {
                assert_eq!(
                    &crate::native_runtime::sha256_file(Path::new(path))
                        .await
                        .unwrap(),
                    hash
                );
            }
        }
        crate::sa3_loras::verify_legacy_history(&target)
            .await
            .unwrap();
        crate::sa3_decoder::read(&target).unwrap();
        let runtime_path =
            crate::native_runtime::path_with_runtimes(&target, &["cudart-12.8".into()]);
        crate::sa3_runtime::probe_server(
            &target.join("services/sa3/native/sa3-server.exe"),
            runtime_path.as_deref(),
        )
        .await
        .unwrap();
        assert_eq!(
            inventory(&real).unwrap(),
            before,
            "production source files unchanged"
        );
        assert_eq!(
            profile_id(&real).unwrap(),
            original_id,
            "production identity unchanged"
        );
        println!("Real snapshot transfer, {} native adapters and offline native server probe passed in {:.2}s; production source unchanged", entries.len(), start.elapsed().as_secs_f64());
    }
}
