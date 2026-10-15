//! Foundation-1's move from its Python service to sa3.cpp's sat-server.
//! sat-server keeps the Python service's HTTP contract on the same port, so
//! there is no client adapter; what is Foundation's own is which models a tier
//! needs and how its runtime is proved.

use crate::migration::{Proof, Run, Selection, Service, Target};
use crate::migration_inventory::Extras;
use std::path::Path;

fn foundation() -> &'static crate::manifest::ServiceDef {
    crate::manifest::bundled()
        .services
        .iter()
        .find(|service| service.id == "foundation")
        .expect("the bundled manifest has a Foundation entry")
}

fn target() -> Target<'static> {
    Target::of(foundation()).expect("Foundation's manifest entry has a migration block")
}

/// Foundation-1 as the shared migration engine sees it.
pub struct Foundation;

impl Service for Foundation {
    fn target(&self) -> Target<'_> {
        target()
    }

    fn extras(&self, root: &Path, _hub: &Path) -> Extras {
        Extras {
            selection: crate::migration::read(root, target().def, |selection| {
                self.check_selection(selection)
            }),
            protected: Vec::new(),
            preserved: Vec::new(),
            warnings: Vec::new(),
            training_base_present: false,
        }
    }

    fn models(&self) -> &'static crate::native_models::Catalog {
        crate::foundation_models::models()
    }

    fn tiers(&self) -> &'static [&'static str] {
        crate::foundation_models::TIERS
    }

    fn tier_components(&self, tier: &str) -> Result<Vec<String>, String> {
        crate::foundation_models::tier_components(tier)
    }

    /// Python downloaded its model into a plain folder, not the Hugging Face
    /// cache, under this model-manager id.
    fn legacy_model_ids(&self) -> Vec<String> {
        vec![crate::model_manager::FOUNDATION_MODEL_ID.into()]
    }

    fn blocking_jobs(&self) -> &'static [&'static str] {
        &["native-migration"]
    }

    fn verify<'a>(&'a self, run: Run<'a>) -> Proof<'a, Selection> {
        Box::pin(verify_native(run))
    }

    /// Activation hashed the models and generated privately; cleanup repeats
    /// that only when the runtime or models changed since.
    fn reverify<'a>(&'a self, run: Run<'a>, selection: &'a Selection) -> Proof<'a, Option<String>> {
        Box::pin(async move {
            let unchanged = fingerprint(run.root, run.native, run.installed, run.encoding)
                .is_ok_and(|current| selection.runtime_fingerprint.as_deref() == Some(current.as_str()));
            if unchanged {
                (run.progress)("The C++ runtime is unchanged since it was verified.");
                return Ok(None);
            }
            Ok(verify_native(run).await?.runtime_fingerprint)
        })
    }
}

/// Size and modification time of what `verify_native` proves: the server, the
/// tier's models and the encoder, with release, backend and tier. Metadata
/// only, so recomputing it hashes nothing.
fn fingerprint(
    root: &Path,
    native: &crate::manifest::NativeDef,
    installed: &crate::native_runtime::NativeInstall,
    encoding: &str,
) -> Result<String, String> {
    let models = crate::foundation_models::models();
    let dir = models.models_dir(root);
    let mut paths = vec![installed.dir.join(&native.executable)];
    for id in crate::foundation_models::tier_components(encoding)? {
        let component = models.component(&id).ok_or("Unknown Foundation-1 model")?;
        paths.extend(component.files.iter().map(|file| dir.join(&file.filename)));
    }
    Ok(crate::migration_verify::fingerprint(
        &paths,
        &serde_json::json!([1, installed.version, installed.backend, encoding]),
    ))
}

/// Prove the installed runtime with the tier being verified, on a private
/// port: hash the models, then exercise every route gary4juce calls.
async fn verify_native(run: Run<'_>) -> Result<Selection, String> {
    use base64::Engine;
    use serde_json::json;
    // Taken before any check, so a file that changes while this runs leaves a
    // fingerprint that no longer matches, never one that wrongly does.
    let fingerprint = fingerprint(run.root, run.native, run.installed, run.encoding)?;
    let ids = crate::foundation_models::tier_components(run.encoding)?;
    crate::foundation_models::models()
        .verify(run.root, &ids, run.progress)
        .await?;
    (run.progress)("Starting the private native test server...");
    let dir = crate::sa3_training::checked_folder(run.root, &["foundation", "migration-checks"])?;
    let server = crate::migration_verify::PrivateServer::start(run.launch, &dir, "Foundation-1", &[]).await?;

    (run.progress)("Testing preset randomization...");
    let preset: serde_json::Value = server
        .client
        .post(format!("{}/randomize", server.base))
        .json(&json!({"seed": 42}))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("Foundation-1 /randomize failed: {error}"))?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    if preset["family"].as_str().map_or(true, str::is_empty) {
        return Err("Foundation-1 /randomize returned no preset".into());
    }

    // Four bars at a trained tempo: the exact sample count proves the timing
    // profile as well as generation.
    const SAMPLES: u32 = 352_800;
    (run.progress)("Testing native generation...");
    let generated = crate::migration_verify::run_audio_job(
        &server,
        "/generate",
        &json!({"prompt": "Synth, Pad, warm", "bars": 4, "bpm": 120, "seed": 42, "steps": 4}),
        "Foundation-1 generation",
    )
    .await?;
    crate::migration_verify::check_wav(&generated, SAMPLES, "Foundation-1 generation")?;

    (run.progress)("Testing audio-to-audio...");
    let transformed = crate::migration_verify::run_audio_job(
        &server,
        "/audio2audio",
        &json!({
            "audio_data": base64::engine::general_purpose::STANDARD.encode(&generated),
            "host_bpm": 120, "bars": 4, "prompt": "Keys, Rhodes Piano", "seed": 42,
            "steps": 4, "init_noise_level": 0.6,
        }),
        "Foundation-1 audio-to-audio",
    )
    .await?;
    crate::migration_verify::check_wav(&transformed, SAMPLES, "Foundation-1 audio-to-audio")?;
    server.stop().await?;

    Ok(Selection {
        schema_version: 1,
        encoding: run.encoding.into(),
        verified_release: run
            .installed
            .version
            .clone()
            .unwrap_or_else(|| "development".into()),
        backend: run.installed.backend.clone(),
        activated_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        cleanup_complete: false,
        cleanup_errors: Vec::new(),
        runtime_fingerprint: Some(fingerprint),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "gary-foundation-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    /// Every file the bundle ships at the top of services/foundation is
    /// declared, so a new one cannot be left behind by cleanup unnoticed, and
    /// nothing declared is missing from it.
    #[test]
    fn the_manifest_declares_exactly_the_bundled_python_code() {
        let bundled = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../services/foundation");
        let code = &target().def.legacy.code;
        for file in &code.files {
            assert!(bundled.join(file).is_file(), "{file} is declared but not bundled");
        }
        for package in &code.packages {
            assert!(bundled.join(package).is_dir(), "{package} is declared but not bundled");
        }
        let output = std::process::Command::new("git")
            .args(["ls-files", "."])
            .current_dir(&bundled)
            .output()
            .unwrap();
        for tracked in String::from_utf8(output.stdout).unwrap().lines() {
            let top = tracked.split('/').next().unwrap();
            assert!(
                code.files.iter().any(|file| file == top) || code.packages.iter().any(|package| package == top),
                "services/foundation/{top} is bundled but not declared in Foundation's migration block"
            );
        }
    }

    #[test]
    fn sat_server_serves_the_python_contract_on_the_same_port() {
        let service = foundation();
        let native = service.native.as_ref().unwrap();
        assert_eq!(native.bundle.as_deref(), Some("sa3"));
        assert_eq!(native.executable, "sat-server.exe");
        assert_eq!(native.models_dir.as_deref(), Some("foundation"));
        // No client adapter: the native server takes the public port itself.
        assert_eq!(native.env.get("SA3_PORT").map(String::as_str), Some(service.port.to_string().as_str()));
        assert!(Foundation.client_adapter(service.port, &[], Path::new(".")).unwrap().is_none());
        let args = Foundation.launch_args(&native.args, Some("Q4_K_M")).unwrap();
        assert_eq!(args, ["--model", "foundation-1", "--encoding", "q4_k_m"]);
        assert!(Foundation.launch_args(&native.args, Some("../x")).is_err());
        let profile = Foundation.profile();
        assert_eq!(profile.tiers, ["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]);
        assert_eq!(profile.tier_components["Q5_K_M"], ["foundation-native::Q5_K_M", "foundation-native::encoder"]);
        assert!(profile.features.is_empty());
    }

    #[test]
    fn a_tier_cannot_start_until_its_bundle_and_the_encoder_are_downloaded() {
        let root = temp_root("start");
        let blocker = Foundation.native_start_blocker(&root, None).unwrap();
        assert!(blocker.contains("Foundation-1 F16") && blocker.contains("Audio-to-audio encoder"), "{blocker}");
        let dir = crate::foundation_models::models().models_dir(&root);
        std::fs::create_dir_all(&dir).unwrap();
        for id in Foundation.tier_components("F16").unwrap() {
            for file in &crate::foundation_models::models().component(&id).unwrap().files {
                // A sparse file of the pinned size is "present" to an inventory check.
                std::fs::File::create(dir.join(&file.filename)).unwrap().set_len(file.bytes).unwrap();
            }
        }
        assert!(Foundation.native_start_blocker(&root, None).is_none());
        let q4 = Selection {
            schema_version: 1,
            encoding: "Q4_K_M".into(),
            verified_release: "v0.1.5".into(),
            backend: "vulkan".into(),
            activated_at: 1,
            cleanup_complete: false,
            cleanup_errors: vec![],
            runtime_fingerprint: None,
        };
        assert!(Foundation.native_start_blocker(&root, Some(&q4)).unwrap().contains("Foundation-1 Q4_K_M"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cleanup_retires_the_python_install_and_keeps_native_models() {
        let root = temp_root("cleanup");
        let hub = root.join("hf-cache");
        let write = |relative: &str| {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"fixture").unwrap();
            path
        };
        let env = write("services/foundation/env/Scripts/python.exe");
        let weights = write("models/foundation-1/Foundation_1.safetensors");
        let config = write("models/foundation-1/model_config.json");
        let native_model = write("models/foundation/foundation-1-dit-1.1B-v1.0-F16.gguf");
        let shared_runtime = write("services/sa3/native/sat-server.exe");
        let preview = crate::migration_cleanup::preview(&Foundation, &root, &hub);
        let candidates: Vec<_> = preview.cleanup_candidates.iter().map(|item| std::path::PathBuf::from(&item.path)).collect();
        assert!(preview.legacy_runtime_present);
        for expected in [env.parent().unwrap().parent().unwrap(), weights.as_path(), config.as_path()] {
            assert!(candidates.iter().any(|path| path == expected), "{} is not a cleanup candidate: {candidates:?}", expected.display());
        }
        for kept in [&native_model, &shared_runtime] {
            assert!(!candidates.iter().any(|path| kept.starts_with(path)), "{} would be removed", kept.display());
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
