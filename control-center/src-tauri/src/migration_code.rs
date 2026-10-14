//! A migrating service's installed Python code is removable only with recorded
//! bundle ownership: the inventory written when the bundle was copied in.
//! Edited or unrecognized files, configuration and developer checkouts survive.
//! Which files count as code is the manifest's `legacy.code`.
use crate::manifest::LegacyCode;
use crate::migration::Target;
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

fn code_path(code: &LegacyCode, path: &str) -> bool {
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
        return code.files.iter().any(|file| file == path);
    }
    code.packages.iter().any(|package| package == parts[0])
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

fn inventory_path(target: Target, root: &Path) -> PathBuf {
    target.data_dir(root).join(INVENTORY)
}

fn read(target: Target, root: &Path) -> Result<Option<Inventory>, String> {
    let label = target.label();
    let path = inventory_path(target, root);
    match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
        Ok(metadata) if metadata.len() > 1024 * 1024 || !metadata.is_file() => {
            return Err(format!("Invalid {label} Python code inventory file"))
        }
        _ => {}
    }
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let dir = target
        .data_dir(root)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !dir.starts_with(&boundary)
        || path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err(format!("{label} Python code inventory escapes managed storage"));
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|error| error.to_string())?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > 1024 * 1024 {
        return Err(format!("Oversized {label} Python code inventory"));
    }
    let inventory: Inventory = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid {label} Python code inventory: {error}"))?;
    if inventory.schema_version != 1 || inventory.files.len() > 4096 {
        return Err(format!("Unsupported or oversized {label} Python code inventory"));
    }
    let mut names = BTreeSet::new();
    for file in &inventory.files {
        if !code_path(&target.def.legacy.code, &file.path)
            || !names.insert(&file.path)
            || file.bytes > LIMIT
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!("Invalid file in {label} Python code inventory"));
        }
    }
    Ok(Some(inventory))
}

/// Used after a bundle copy, or to seed ownership for an older same-stamp
/// install. Bundle checksums never substitute for edited runtime contents.
pub fn record_bundle(
    target: Target,
    bundle_root: &Path,
    root: &Path,
    only_if_missing: bool,
) -> Result<(), String> {
    if only_if_missing && read(target, root)?.is_some() {
        return Ok(());
    }
    let label = target.label();
    let source = target.service_dir(bundle_root);
    if !source.is_dir() {
        return Ok(());
    }
    fn collect(
        code: &LegacyCode,
        label: &str,
        source: &Path,
        dir: &Path,
        files: &mut Vec<OwnedFile>,
    ) -> Result<(), String> {
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
            let package = relative.split('/').next().unwrap_or("");
            if kind.is_dir() && code.packages.iter().any(|known| known == package) {
                collect(code, label, source, &path, files)?;
            } else if kind.is_file() && code_path(code, &relative) {
                let bytes = entry.metadata().map_err(|error| error.to_string())?.len();
                if bytes > LIMIT {
                    return Err(format!("Bundled {label} Python source is oversized"));
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
    collect(&target.def.legacy.code, label, &source, &source, &mut files)?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    if files.len() > 4096 {
        return Err(format!("Bundled {label} Python inventory is oversized"));
    }
    let dir = crate::sa3_training::checked_folder(root, &target.data_folders())?;
    let path = dir.join(INVENTORY);
    if path.exists()
        && path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err(format!("{label} Python code inventory escapes managed storage"));
    }
    crate::sa3_training::save(
        &path,
        &Inventory {
            schema_version: 1,
            files,
        },
    )
}

/// Classify every recorded file, or with `only` just that installed path,
/// skipping the hashing of the rest. Cleanup re-checks each file right before
/// removing it; hashing every recorded file for each of them made cleanup
/// quadratic in the size of the bundle.
pub fn status(target: Target, root: &Path, only: Option<&Path>) -> Status {
    let mut status = Status::default();
    let inventory = match read(target, root) {
        Ok(Some(inventory)) => inventory,
        Ok(None) => return status,
        Err(error) => {
            status.warnings.push(error);
            return status;
        }
    };
    let service = target.service_dir(root);
    status.review_identity = Some(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&inventory).unwrap())
    ));
    let developer = root.join(".git").exists() || service.join(".git").exists();
    let boundary = root.canonicalize().ok();
    let owner = service.canonicalize().ok();
    for file in inventory.files {
        let path = service.join(&file.path);
        if only.is_some_and(|only| only != path) {
            continue;
        }
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
                "Preserving redirected {} Python file {}; review it manually.",
                target.label(),
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
pub fn recorded_paths(target: Target, root: &Path) -> Result<Vec<PathBuf>, String> {
    let service = target.service_dir(root);
    Ok(read(target, root)?
        .map(|inventory| {
            inventory
                .files
                .into_iter()
                .map(|file| service.join(file.path))
                .collect()
        })
        .unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sa3() -> Target<'static> {
        crate::sa3_runtime::target()
    }
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
            let result = status(sa3(), &root, None);
            assert!(!result.warnings.is_empty());
            assert!(result.candidates.is_empty());
            assert!(recorded_paths(sa3(), &root).is_err());
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
        record_bundle(sa3(), &bundle, &runtime, false).unwrap();
        let created = std::process::Command::new("cmd")
            .raw_arg(format!(
                "/c mklink /J \"{}\" \"{}\"",
                service.join("stable_audio_3").display(),
                outside.display()
            ))
            .output()
            .unwrap();
        assert!(created.status.success());
        let result = status(sa3(), &runtime, None);
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
        record_bundle(sa3(), &bundle, &runtime, false).unwrap();
        assert_eq!(status(sa3(), &runtime, None).candidates.len(), 3);
        std::fs::write(runtime.join("services/sa3/api.py"), b"customized code").unwrap();
        std::fs::write(runtime.join("services/sa3/user.py"), b"unrecognized source").unwrap();
        let result = status(sa3(), &runtime, None);
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.retained.len(), 1);
        assert!(result.warnings.is_empty());
        std::fs::write(runtime.join(".git"), b"gitdir: worktree").unwrap();
        assert!(status(sa3(), &runtime, None).candidates.is_empty());
        assert_eq!(status(sa3(), &runtime, None).retained.len(), 3);
        for path in [
            "../api.py",
            "native/attack.py",
            "env/script.py",
            "stable_audio_3/../../attack.py",
            "stable_audio_3/a.py:stream",
            "stable_audio_3/config.json",
        ] {
            assert!(!code_path(&sa3().def.legacy.code, path));
        }
        crate::remove_managed_path(&root, &std::env::temp_dir()).unwrap();
    }
}
