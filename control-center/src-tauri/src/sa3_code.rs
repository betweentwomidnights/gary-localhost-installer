//! Installed Python code is removable only with recorded bundle ownership.
//! Edited/unrecognized files, configuration and developer checkouts survive.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};

const LIMIT: u64 = 16 * 1024 * 1024;
const INVENTORY: &str = "legacy-code-inventory.json";

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OwnedFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Inventory {
    schema_version: u32,
    files: Vec<OwnedFile>,
}

#[derive(Default)]
pub struct Status {
    pub candidates: Vec<PathBuf>,
    pub retained: Vec<PathBuf>,
    pub warnings: Vec<String>,
    pub review_identity: Option<String>,
}

fn code_path(path: &str) -> bool {
    let parts: Vec<_> = path.split('/').collect();
    if parts.iter().any(|part| {
        part.is_empty()
            || *part == "."
            || *part == ".."
            || !part
                .bytes()
                .all(|ch| ch.is_ascii_alphanumeric() || b"._-".contains(&ch))
    }) {
        return false;
    }
    if parts.len() == 1 {
        return [
            "api.py",
            "analyze_audio.py",
            "bpm_analysis.py",
            "key_analysis.py",
            "build_lora_prompts.py",
            "lora_train.py",
            "lora_train_lightning.py",
            "train_lora_job.py",
            "train_memory.py",
            "setup.py",
            "requirements.txt",
        ]
        .contains(&path);
    }
    ["stable_audio_3", "underfit", "dataset_processing", "tests"].contains(&parts[0])
        && path.ends_with(".py")
        && !parts.contains(&"__pycache__")
}

fn checksum(path: &Path, expected_bytes: u64) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut reader = file.take(expected_bytes + 1);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut total = 0;
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        total += count as u64;
        hasher.update(&buffer[..count]);
    }
    if total != expected_bytes {
        return Err("Python code changed during inventory".into());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn inventory_path(root: &Path) -> PathBuf {
    root.join("sa3").join(INVENTORY)
}

fn read(root: &Path) -> Result<Option<Inventory>, String> {
    let path = inventory_path(root);
    match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
        Ok(metadata) if metadata.len() > 1024 * 1024 || !metadata.is_file() => {
            return Err("Invalid SA3 Python code inventory file".into())
        }
        _ => {}
    }
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let dir = root
        .join("sa3")
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !dir.starts_with(&boundary)
        || path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err("SA3 Python code inventory escapes managed storage".into());
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|error| error.to_string())?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > 1024 * 1024 {
        return Err("Oversized SA3 Python code inventory".into());
    }
    let inventory: Inventory = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid SA3 Python code inventory: {error}"))?;
    if inventory.schema_version != 1 || inventory.files.len() > 4096 {
        return Err("Unsupported or oversized SA3 Python code inventory".into());
    }
    let mut names = BTreeSet::new();
    for file in &inventory.files {
        if !code_path(&file.path)
            || !names.insert(&file.path)
            || file.bytes > LIMIT
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("Invalid file in SA3 Python code inventory".into());
        }
    }
    Ok(Some(inventory))
}

/// Used after a bundle copy, or to seed ownership for an older same-stamp
/// install. Bundle checksums never substitute for edited runtime contents.
pub fn record_bundle(bundle_root: &Path, root: &Path, only_if_missing: bool) -> Result<(), String> {
    if only_if_missing && read(root)?.is_some() {
        return Ok(());
    }
    let source = bundle_root.join("services/sa3");
    if !source.is_dir() {
        return Ok(());
    }
    fn collect(source: &Path, dir: &Path, files: &mut Vec<OwnedFile>) -> Result<(), String> {
        for entry in std::fs::read_dir(dir).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let kind = entry.file_type().map_err(|error| error.to_string())?;
            if kind.is_symlink() {
                continue;
            }
            let path = entry.path();
            let relative = path
                .strip_prefix(source)
                .map_err(|error| error.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            if kind.is_dir()
                && ["stable_audio_3", "underfit", "dataset_processing", "tests"]
                    .contains(&relative.split('/').next().unwrap_or(""))
            {
                collect(source, &path, files)?;
            } else if kind.is_file() && code_path(&relative) {
                let bytes = entry.metadata().map_err(|error| error.to_string())?.len();
                if bytes > LIMIT {
                    return Err("Bundled SA3 Python source is oversized".into());
                }
                files.push(OwnedFile {
                    path: relative,
                    bytes,
                    sha256: checksum(&path, bytes)?,
                });
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    collect(&source, &source, &mut files)?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    if files.len() > 4096 {
        return Err("Bundled SA3 Python inventory is oversized".into());
    }
    let dir = crate::sa3_training::checked_folder(root, &["sa3"])?;
    let path = dir.join(INVENTORY);
    if path.exists()
        && path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err("SA3 Python code inventory escapes managed storage".into());
    }
    crate::sa3_training::save(
        &path,
        &Inventory {
            schema_version: 1,
            files,
        },
    )
}

pub fn status(root: &Path) -> Status {
    let mut status = Status::default();
    let inventory = match read(root) {
        Ok(Some(inventory)) => inventory,
        Ok(None) => return status,
        Err(error) => {
            status.warnings.push(error);
            return status;
        }
    };
    let service = root.join("services/sa3");
    status.review_identity = Some(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&inventory).unwrap())
    ));
    let developer = root.join(".git").exists() || service.join(".git").exists();
    let boundary = root.canonicalize().ok();
    let owner = service.canonicalize().ok();
    for file in inventory.files {
        let path = service.join(&file.path);
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                status
                    .warnings
                    .push(format!("Cannot inspect {}: {error}", path.display()));
                continue;
            }
        };
        let valid_path = boundary
            .as_ref()
            .zip(owner.as_ref())
            .is_some_and(|(boundary, owner)| {
                owner.starts_with(boundary)
                    && path.canonicalize().ok() == Some(owner.join(&file.path))
            });
        if !valid_path || metadata.file_type().is_symlink() || !metadata.is_file() {
            status.warnings.push(format!(
                "Preserving redirected SA3 Python file {}; review it manually.",
                path.display()
            ));
        } else if developer || metadata.len() != file.bytes {
            status.retained.push(path);
        } else {
            match checksum(&path, file.bytes) {
                Ok(hash) if hash == file.sha256 => status.candidates.push(path),
                Ok(_) => status.retained.push(path),
                Err(error) => status
                    .warnings
                    .push(format!("Cannot verify {}: {error}", path.display())),
            }
        }
    }
    status
}

/// The cleanup transaction independently recognizes each recorded code path;
/// it still requires a fresh checksum/protection preview before removal.
pub fn recorded_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    Ok(read(root)?
        .map(|inventory| {
            inventory
                .files
                .into_iter()
                .map(|file| root.join("services/sa3").join(file.path))
                .collect()
        })
        .unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_inventory_cannot_admit_native_files_for_deletion() {
        let root = std::env::temp_dir().join(format!("sa3-code-invalid-{}", std::process::id()));
        let native = root.join("services/sa3/native/sa3-server.exe");
        std::fs::create_dir_all(native.parent().unwrap()).unwrap();
        std::fs::write(&native, b"native file").unwrap();
        let dir = crate::sa3_training::checked_folder(&root, &["sa3"]).unwrap();
        for path in ["../outside.py", "native/sa3-server.exe", "env/api.py"] {
            std::fs::write(
                dir.join(INVENTORY),
                serde_json::json!({"schemaVersion":1,"files":[{
                    "path":path,"bytes":11,"sha256":"0".repeat(64)
                }]})
                .to_string(),
            )
            .unwrap();
            let result = status(&root);
            assert!(!result.warnings.is_empty());
            assert!(result.candidates.is_empty());
            assert!(recorded_paths(&root).is_err());
            assert!(native.is_file());
        }
        crate::remove_managed_path(&root, &std::env::temp_dir()).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn owned_code_cannot_follow_a_runtime_package_junction_outside_the_profile() {
        use std::os::windows::process::CommandExt;
        let root = std::env::temp_dir().join(format!("sa3-code-junction-{}", std::process::id()));
        let bundle = root.join("bundle");
        let runtime = root.join("runtime");
        let outside = root.join("outside");
        let source = bundle.join("services/sa3/stable_audio_3/factory.py");
        std::fs::create_dir_all(source.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(source, b"bundled source").unwrap();
        std::fs::write(outside.join("factory.py"), b"bundled source").unwrap();
        let service = runtime.join("services/sa3");
        std::fs::create_dir_all(&service).unwrap();
        record_bundle(&bundle, &runtime, false).unwrap();
        let created = std::process::Command::new("cmd")
            .raw_arg(format!(
                "/c mklink /J \"{}\" \"{}\"",
                service.join("stable_audio_3").display(),
                outside.display()
            ))
            .output()
            .unwrap();
        assert!(created.status.success());
        let result = status(&runtime);
        assert!(result.candidates.is_empty());
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.contains("redirected")));
        assert_eq!(
            std::fs::read(outside.join("factory.py")).unwrap(),
            b"bundled source"
        );
        crate::remove_managed_path(&root, &std::env::temp_dir()).unwrap();
    }

    #[test]
    fn ownership_preserves_edits_unknown_files_settings_and_developer_sources() {
        let root = std::env::temp_dir().join(format!("sa3-code-owned-{}", std::process::id()));
        let bundle = root.join("bundle");
        let runtime = root.join("runtime");
        for relative in [
            "api.py",
            "requirements.txt",
            "stable_audio_3/factory.py",
            "defaults.ini",
            "prompts/defaults.json",
        ] {
            for base in [&bundle, &runtime] {
                let path = base.join("services/sa3").join(relative);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, b"bundled source").unwrap();
            }
        }
        record_bundle(&bundle, &runtime, false).unwrap();
        assert_eq!(status(&runtime).candidates.len(), 3);
        std::fs::write(runtime.join("services/sa3/api.py"), b"customized code").unwrap();
        std::fs::write(runtime.join("services/sa3/user.py"), b"unrecognized source").unwrap();
        let result = status(&runtime);
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.retained.len(), 1);
        assert!(result.warnings.is_empty());
        std::fs::write(runtime.join(".git"), b"gitdir: worktree").unwrap();
        assert!(status(&runtime).candidates.is_empty());
        assert_eq!(status(&runtime).retained.len(), 3);
        for path in [
            "../api.py",
            "native/attack.py",
            "env/script.py",
            "stable_audio_3/../../attack.py",
            "stable_audio_3/a.py:stream",
            "stable_audio_3/config.json",
        ] {
            assert!(!code_path(path));
        }
        crate::remove_managed_path(&root, &std::env::temp_dir()).unwrap();
    }
}
