//! Caption dice pools must remain available after the Python environment retires.
use gary4local_core::sa3_prompts::save_with_policy;
use std::path::Path;

/// New native profiles get the same bundled dice pool as Python. Existing
/// user-edited defaults are preserved, including an intentionally empty pool.
pub fn ensure_defaults(root: &Path) -> Result<(), String> {
    let dir = crate::sa3_training::checked_folder(root, &["sa3", "prompts"])?;
    let path = dir.join("defaults.json");
    match std::fs::symlink_metadata(&path) {
        Ok(_) => {
            if !path.is_file() {
                return Err("Default prompt pool must be a file".into());
            }
            if path
                .canonicalize()
                .map_err(|error| error.to_string())?
                .parent()
                != Some(dir.as_path())
            {
                return Err("Default prompt pool escapes managed storage".into());
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let defaults: serde_json::Value =
                serde_json::from_str(include_str!("../../../services/sa3/prompts/defaults.json"))
                    .map_err(|error| error.to_string())?;
            save_with_policy(&path, &defaults, false)
        }
        Err(error) => Err(error.to_string()),
    }
}

pub fn build(root: &Path, name: &str, dataset: &Path, force: bool) -> Result<String, String> {
    if crate::sanitize_lora_name(name).as_deref() != Some(name) {
        return Err("Invalid prompt pool name".into());
    }
    let dir = crate::sa3_training::checked_folder(root, &["sa3", "prompts"])?;
    gary4local_core::sa3_prompts::build(&dir, name, dataset, force)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn native_profiles_seed_defaults_and_preserve_user_edits() {
        let root = std::env::temp_dir().join(format!("sa3-defaults-{}", std::process::id()));
        ensure_defaults(&root).unwrap();
        let path = root.join("sa3/prompts/defaults.json");
        let defaults: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert!(!defaults["dice"]["generic"].as_array().unwrap().is_empty());
        let edited = br#"{"dice":{"generic":["my prompt"]}}"#;
        std::fs::write(&path, edited).unwrap();
        ensure_defaults(&root).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), edited);
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(ensure_defaults(&root).unwrap_err().contains("file"));
        crate::remove_managed_path(&root, &std::env::temp_dir()).unwrap();
    }

    #[test]
    fn pools_deduplicate_and_preserve_curated_prompts() {
        let root = std::env::temp_dir().join(format!("gary-native-prompts-{}", std::process::id()));
        std::fs::create_dir_all(root.join("dataset/sub")).unwrap();
        std::fs::write(root.join("dataset/a.txt"), "\u{feff}Dnb, 174 bpm, A minor").unwrap();
        std::fs::write(root.join("dataset/sub/b.txt"), "dnb, Key: C min").unwrap();
        build(&root, "test", &root.join("dataset"), false).unwrap();
        let path = root.join("sa3/prompts/test.json");
        let pool: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(pool["dice"]["instrumental"], json!(["Dnb"]));
        std::fs::write(&path, b"curated").unwrap();
        save_with_policy(&path, &json!({"replacement":true}), false).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"curated",
            "atomic publication must not replace a concurrently created pool"
        );
        build(&root, "test", &root.join("dataset"), false).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"curated");
        assert!(build(&root, "../escape", &root.join("dataset"), true).is_err());
        crate::remove_managed_path(&root, &std::env::temp_dir().canonicalize().unwrap()).unwrap();
    }
}
