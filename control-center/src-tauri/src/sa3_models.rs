//! Published SA3 model components, kept separate from the legacy HF cache.
//! Inference and training share the encoder, tokenizer, decoder and conditioner.

use crate::native_models::Catalog;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub use crate::native_models::CatalogComponent as Component;

pub fn models() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| Catalog::bundled("sa3", include_str!("../sa3-models.json")))
}

#[cfg(test)]
pub fn catalog() -> &'static [Component] {
    models().components()
}

pub fn component(id: &str) -> Option<&'static Component> {
    models().component(id)
}

pub fn models_dir(runtime_root: &Path) -> PathBuf {
    models().models_dir(runtime_root)
}

pub fn checked_models_dir(runtime_root: &Path) -> Result<PathBuf, String> {
    models().checked_models_dir(runtime_root)
}

pub fn preparation_ids(encoding: &str, training_base: Option<&str>) -> Result<Vec<String>, String> {
    if !["F16", "Q8_0", "Q5_K_M", "Q4_K_M"].contains(&encoding) {
        return Err("Choose an SA3 inference encoding: F16, Q8_0, Q5_K_M or Q4_K_M.".into());
    }
    if training_base.is_some_and(|tier| !["F16", "Q4_K_M"].contains(&tier)) {
        return Err("Choose an SA3 training base: F16 or Q4_K_M.".into());
    }
    let mut ids = vec![
        "sa3-native::text".into(),
        "sa3-native::medium-decoder".into(),
        format!("sa3-native::medium-{encoding}"),
    ];
    if let Some(tier) = training_base {
        ids.push(format!("sa3-native::medium-base-{tier}"));
    }
    Ok(ids)
}

pub fn present(entry: &Component, dir: &Path) -> bool {
    Catalog::present(entry, dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn published_components_are_disjoint_and_pinned() {
        let mut ids = HashSet::new();
        let mut files = HashSet::new();
        for entry in catalog() {
            assert!(ids.insert(&entry.id));
            assert!(entry.id.starts_with("sa3-native::"));
            assert!(!entry.files.is_empty());
            for file in &entry.files {
                file.validate().unwrap();
                assert!(
                    files.insert(&file.filename),
                    "shared files belong in one component"
                );
            }
        }
    }

    #[test]
    fn quantized_inference_does_not_change_decoder_or_training_precision() {
        let ids = preparation_ids("Q8_0", Some("F16")).unwrap();
        assert!(ids.contains(&"sa3-native::medium-base-F16".into()));
        let files: Vec<_> = ids
            .iter()
            .flat_map(|id| &component(id).unwrap().files)
            .collect();
        assert!(files
            .iter()
            .any(|file| file.filename.ends_with("same-l-v1.0-F32.gguf")));
        assert!(files
            .iter()
            .any(|file| file.filename.ends_with("encoder-0.3B-v1.0-F16.gguf")));
        assert_eq!(
            files
                .iter()
                .filter(|file| file.filename.contains("vocab"))
                .count(),
            1
        );
        assert!(preparation_ids("../F16", None).is_err());
        assert!(preparation_ids("F16", Some("Q8_0")).is_err());
        assert!(component("sa3-native::../../user-file").is_none());
    }

    #[test]
    fn partial_or_wrong_size_components_are_not_installed() {
        let dir = std::env::temp_dir().join(format!("gary-sa3-catalog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut entry = catalog()[0].clone();
        entry.files.truncate(1);
        entry.files[0].filename = "fixture.gguf".into();
        entry.files[0].bytes = 4;
        assert!(!present(&entry, &dir));
        std::fs::write(dir.join("fixture.gguf.partial"), b"1234").unwrap();
        assert!(!present(&entry, &dir));
        std::fs::write(dir.join("fixture.gguf"), b"123").unwrap();
        assert!(!present(&entry, &dir));
        std::fs::write(dir.join("fixture.gguf"), b"1234").unwrap();
        assert!(present(&entry, &dir));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn completed_download_state_does_not_hide_a_missing_component() {
        let root = std::env::temp_dir().join(format!("gary-sa3-missing-{}", std::process::id()));
        let mut manager = crate::model_manager::ModelManager::new(root);
        manager.set_download_started("sa3-native::text");
        manager.set_download_done("sa3-native::text", None);
        let models = manager.get_native_models("sa3");
        assert_eq!(
            models
                .iter()
                .find(|model| model.id == "sa3-native::text")
                .unwrap()
                .status,
            crate::model_manager::ModelStatus::Available
        );
    }

    #[test]
    fn model_preparation_resolves_the_selected_storage_boundary() {
        let root = std::env::temp_dir().join(format!("gary-sa3-storage-{}", std::process::id()));
        let dir = checked_models_dir(&root).unwrap();
        assert_eq!(dir, root.canonicalize().unwrap().join("models/sa3"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
