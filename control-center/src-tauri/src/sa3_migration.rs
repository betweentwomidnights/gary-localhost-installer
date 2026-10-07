//! Read-only inventory for SA3's move from PyTorch to the native runtime.
//! This is deliberately separate from cleanup: installation, API parity and
//! LoRA conversion must be validated before any of these paths can be retired.

use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MigrationItem {
    pub label: String,
    pub path: String,
    pub bytes: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sa3MigrationPreview {
    pub active_root: String,
    pub hf_hub_cache: String,
    pub cleanup_candidates: Vec<MigrationItem>,
    pub estimated_cleanup_bytes: u64,
    pub preserved_paths: Vec<MigrationItem>,
    pub warnings: Vec<String>,
}

fn item(label: &str, path: &Path) -> MigrationItem {
    MigrationItem {
        label: label.to_string(),
        path: path.to_string_lossy().to_string(),
        bytes: crate::path_size(path),
    }
}

pub fn preview(active_root: &Path, hf_hub: &Path) -> Sa3MigrationPreview {
    let mut result = Sa3MigrationPreview {
        active_root: active_root.to_string_lossy().to_string(),
        hf_hub_cache: hf_hub.to_string_lossy().to_string(),
        cleanup_candidates: Vec::new(),
        estimated_cleanup_bytes: 0,
        preserved_paths: Vec::new(),
        warnings: Vec::new(),
    };
    let service = active_root.join("services").join("sa3");
    let candidates = [
        (
            "SA3 Python environment",
            service.join("env"),
            service.clone(),
        ),
        (
            "SA3 alternate Python environment",
            service.join(".venv"),
            service.clone(),
        ),
        (
            "SA3 PyTorch inference weights",
            hf_hub.join("models--stabilityai--stable-audio-3-medium"),
            hf_hub.to_path_buf(),
        ),
        (
            "SA3 PyTorch training base",
            hf_hub.join("models--stabilityai--stable-audio-3-medium-base"),
            hf_hub.to_path_buf(),
        ),
    ];
    let protected = active_root.join("sa3");
    let mut seen = Vec::new();
    for (label, path, owner) in candidates {
        match crate::resolve_managed_path(&path, &owner) {
            Ok(Some(canonical)) => {
                // A redirected cache or environment must never turn the
                // proposed cleanup into removal of user-owned SA3 artifacts.
                if crate::path_is_inside(&canonical, &protected)
                    || crate::path_is_inside(&protected, &canonical)
                {
                    result.warnings.push(format!(
                        "Preserving {} because it overlaps SA3 user data.", path.display()
                    ));
                } else if !seen.contains(&canonical) {
                    result.cleanup_candidates.push(item(label, &path));
                    seen.push(canonical);
                }
            }
            Ok(None) => {}
            Err(_) => result.warnings.push(format!(
                "Cannot safely inventory {} within {}. Review this redirected or inaccessible path manually.",
                path.display(), owner.display()
            )),
        }
    }
    result.estimated_cleanup_bytes = result
        .cleanup_candidates
        .iter()
        .map(|entry| entry.bytes)
        .sum();
    // Do not count these recursively: a LoRA checkpoint can be many GB, and
    // they cannot contribute to the space reclaimed by this migration.
    for (label, path) in [
        ("LoRAs, prompts, training jobs and checkpoints", protected),
        ("SA3 service code and helper scripts", service),
        (
            "Shared CUDA and other native runtimes",
            active_root.join("native-runtimes"),
        ),
        (
            "Native model weights",
            active_root.join("models").join("sa3"),
        ),
        (
            "Decoder LoRA source for conversion",
            hf_hub.join("models--thepatch--same-l-decoder-lora"),
        ),
    ] {
        result.preserved_paths.push(MigrationItem {
            label: label.to_string(),
            path: path.to_string_lossy().to_string(),
            bytes: 0,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::preview;
    use std::path::{Path, PathBuf};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(
                std::env::temp_dir()
                    .join(format!("gary-sa3-migration-{}-{nonce}", std::process::id())),
            )
        }
        fn write(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"keep").unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn new_install_has_no_cleanup_candidates_and_creates_nothing() {
        let fixture = Fixture::new();
        let result = preview(&fixture.0, &fixture.0.join("models/huggingface/hub"));
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.estimated_cleanup_bytes, 0);
        assert!(!fixture.0.exists());
    }

    #[test]
    fn custom_storage_only_inventories_sa3_owned_pytorch_paths() {
        let fixture = Fixture::new();
        let env = fixture.write("runtime/services/sa3/env/python.exe");
        let model = fixture.write(
            "runtime/models/huggingface/hub/models--stabilityai--stable-audio-3-medium/weights",
        );
        let keep = [
            fixture.write("runtime/services/stable-audio/env/python.exe"),
            fixture.write("runtime/services/foundation/env/python.exe"),
            fixture.write("runtime/models/huggingface/hub/models--thepatch--same-l-decoder-lora/squeakfix_v3.safetensors"),
            fixture.write("runtime/sa3/loras/custom.safetensors"),
            fixture.write("runtime/sa3/training/jobs/run/checkpoint.safetensors"),
            fixture.write("runtime/native-runtimes/cudart-12.8/cudart64_12.dll"),
            fixture.write("runtime/cache/uv/package"),
            fixture.write("legacy/services/sa3/env/python.exe"),
        ];
        let root = fixture.0.join("runtime");
        let result = preview(&root, &root.join("models/huggingface/hub"));
        assert_eq!(result.cleanup_candidates.len(), 2);
        assert_eq!(result.estimated_cleanup_bytes, 8);
        for path in keep.iter().chain([&env, &model]) {
            assert!(path.is_file());
        }
    }

    #[test]
    fn legacy_storage_uses_the_effective_external_hugging_face_cache() {
        let fixture = Fixture::new();
        let model =
            fixture.write("external-hub/models--stabilityai--stable-audio-3-medium-base/weights");
        fixture.write(
            "legacy/models/huggingface/hub/models--stabilityai--stable-audio-3-medium/unused",
        );
        let result = preview(&fixture.0.join("legacy"), &fixture.0.join("external-hub"));
        assert_eq!(result.cleanup_candidates.len(), 1);
        assert_eq!(
            Path::new(&result.cleanup_candidates[0].path),
            model.parent().unwrap()
        );
        assert!(model.exists());
    }

    #[test]
    fn a_cache_redirected_into_user_artifacts_is_preserved() {
        let fixture = Fixture::new();
        let model = fixture.write("runtime/sa3/models--stabilityai--stable-audio-3-medium/weights");
        let root = fixture.0.join("runtime");
        let result = preview(&root, &root.join("sa3"));
        assert!(result.cleanup_candidates.is_empty());
        assert_eq!(result.warnings.len(), 1);
        assert!(model.exists());
    }
}
