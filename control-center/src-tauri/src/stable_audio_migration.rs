//! Jerry's move from its Python stable-audio service to sa3.cpp's sat-server.
//! A SAOS sat-server keeps the Python service's HTTP contract on the same
//! port and serves the base model and every finetune downloaded at its tier,
//! so there is no client adapter; what is jerry's own is which models a tier
//! needs and offers, and how its runtime is proved.

use crate::migration::{Proof, Run, Selection, Service, Target};
use crate::migration_inventory::Extras;
use std::path::Path;

fn stable_audio() -> &'static crate::manifest::ServiceDef {
    crate::manifest::bundled()
        .services
        .iter()
        .find(|service| service.id == "stable-audio")
        .expect("the bundled manifest has a stable-audio entry")
}

fn target() -> Target<'static> {
    Target::of(stable_audio()).expect("stable-audio's manifest entry has a migration block")
}

/// Jerry as the shared migration engine sees it.
pub struct StableAudio;

impl Service for StableAudio {
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
        crate::stable_audio_models::models()
    }

    fn tiers(&self) -> &'static [&'static str] {
        crate::stable_audio_models::TIERS
    }

    fn tier_components(&self, tier: &str) -> Result<Vec<String>, String> {
        crate::stable_audio_models::tier_components(tier)
    }

    fn tier_options(&self, tier: &str) -> Vec<String> {
        crate::stable_audio_models::finetunes(tier)
    }

    fn options_label(&self) -> Option<&'static str> {
        Some("finetunes")
    }

    fn blocking_jobs(&self) -> &'static [&'static str] {
        &["native-migration"]
    }

    fn verify<'a>(&'a self, run: Run<'a>) -> Proof<'a, Selection> {
        Box::pin(verify_native(run))
    }

    /// Activation hashed the models and generated privately; cleanup repeats
    /// that only when the runtime or base models changed since.
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

/// Size and modification time of what `verify_native` proves: the server and
/// the tier's base bundle, with release, backend and tier. Metadata only, so
/// recomputing it hashes nothing.
fn fingerprint(
    root: &Path,
    native: &crate::manifest::NativeDef,
    installed: &crate::native_runtime::NativeInstall,
    encoding: &str,
) -> Result<String, String> {
    let models = crate::stable_audio_models::models();
    let dir = models.models_dir(root);
    let mut paths = vec![installed.dir.join(&native.executable)];
    for id in crate::stable_audio_models::tier_components(encoding)? {
        let component = models.component(&id).ok_or("Unknown jerry model")?;
        paths.extend(component.files.iter().map(|file| dir.join(&file.filename)));
    }
    Ok(crate::migration_verify::fingerprint(
        &paths,
        &serde_json::json!([1, installed.version, installed.backend, encoding]),
    ))
}

/// Prove the installed runtime with the tier being verified, on a private
/// port: hash the base bundle, then exercise every route gary4juce's jerry
/// tab calls, with the base model and each finetune already downloaded.
async fn verify_native(run: Run<'_>) -> Result<Selection, String> {
    use crate::migration_verify::{check_wav, run_sync_audio};
    use serde_json::json;
    // Taken before any check, so a file that changes while this runs leaves a
    // fingerprint that no longer matches, never one that wrongly does.
    let fingerprint = fingerprint(run.root, run.native, run.installed, run.encoding)?;
    let ids = crate::stable_audio_models::tier_components(run.encoding)?;
    crate::stable_audio_models::models()
        .verify(run.root, &ids, run.progress)
        .await?;
    (run.progress)("Starting the private native test server...");
    let dir = crate::sa3_training::checked_folder(run.root, &["stable-audio", "migration-checks"])?;
    let server = crate::migration_verify::PrivateServer::start(run.launch, &dir, "jerry", &[]).await?;

    (run.progress)("Listing jerry's models...");
    let response = server
        .client
        .get(format!("{}/models/status", server.base))
        .send()
        .await
        .map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        return Err("The installed runtime does not serve jerry's model list; update it before migration".into());
    }
    let status: serde_json::Value = response.json().await.map_err(|error| error.to_string())?;
    let details = status["model_details"].as_object().cloned().unwrap_or_default();
    if details.get("standard_saos").and_then(|base| base["source"].as_str())
        != Some("stabilityai/stable-audio-open-small")
    {
        return Err("jerry's native model list is missing the base model".into());
    }

    // gary4juce asks for whole seconds and for loops in bars; exact sample
    // counts prove the crop as well as generation.
    let base = json!({
        "model_type": "standard", "sampler_type": "pingpong", "steps": 8, "cfg_scale": 1.0,
        "seed": 42, "return_format": "base64",
    });
    let with = |extra: serde_json::Value| {
        let mut body = base.clone();
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        body
    };
    (run.progress)("Testing native generation...");
    let (generated, _) = run_sync_audio(
        &server,
        "/generate",
        &with(json!({"prompt": "warm synth pad 120bpm", "seconds_total": 4})),
        "jerry generation",
    )
    .await?;
    check_wav(&generated, 4 * 44_100, "jerry generation")?;

    (run.progress)("Testing a smart loop...");
    let (looped, metadata) = run_sync_audio(
        &server,
        "/generate/loop",
        &with(json!({"prompt": "funk drum loop 120bpm", "loop_type": "drums"})),
        "jerry loop",
    )
    .await?;
    // Four bars at 120 BPM: the longest loop that leaves a second of the canvas spare.
    check_wav(&looped, 352_800, "jerry loop")?;
    if metadata["bars"] != 4 {
        return Err(format!("jerry loop returned {} bars; expected 4", metadata["bars"]));
    }

    // Each finetune already downloaded at this tier, the way gary4juce names
    // it: by the repo and checkpoint in its source.
    for (key, detail) in &details {
        if detail["type"] != "finetune" {
            continue;
        }
        let source = detail["source"].as_str().unwrap_or_default();
        let Some((repo, checkpoint)) = source.rsplit_once('/') else {
            return Err(format!("jerry lists {key} with an invalid source"));
        };
        (run.progress)(&format!("Testing the {repo} finetune..."));
        let what = format!("{repo} generation");
        let (audio, _) = run_sync_audio(
            &server,
            "/generate",
            &with(json!({
                "prompt": "dusty drums and bass 100bpm", "seconds_total": 4, "model_type": "finetune",
                "finetune_repo": repo, "finetune_checkpoint": checkpoint,
                "sampler_type": "euler", "steps": 8, "cfg_scale": 4.0,
            })),
            &what,
        )
        .await?;
        check_wav(&audio, 4 * 44_100, &what)?;
    }

    (run.progress)("Testing the prompt bank...");
    let prompts: serde_json::Value = server
        .client
        .get(format!("{}/models/prompts", server.base))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("jerry /models/prompts failed: {error}"))?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    if prompts["success"] != true || !prompts["prompts"].is_object() {
        return Err("jerry /models/prompts returned no prompt bank".into());
    }
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
            "gary-stable-audio-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    /// Every file the bundle ships at the top of services/stable-audio is
    /// declared, so a new one cannot be left behind by cleanup unnoticed, and
    /// nothing declared is missing from it.
    #[test]
    fn the_manifest_declares_exactly_the_bundled_python_code() {
        let bundled = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../services/stable-audio");
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
                "services/stable-audio/{top} is bundled but not declared in jerry's migration block"
            );
        }
    }

    #[test]
    fn sat_server_serves_the_python_contract_on_the_same_port() {
        let service = stable_audio();
        let native = service.native.as_ref().unwrap();
        assert_eq!(native.bundle.as_deref(), Some("sa3"));
        assert_eq!(native.executable, "sat-server.exe");
        assert_eq!(native.models_dir.as_deref(), Some("stable-audio"));
        // No client adapter: the native server takes the public port itself.
        assert_eq!(native.env.get("SA3_PORT").map(String::as_str), Some(service.port.to_string().as_str()));
        assert!(StableAudio.client_adapter(service.port, &[], Path::new(".")).unwrap().is_none());
        let args = StableAudio.launch_args(&native.args, Some("Q4_K_M")).unwrap();
        assert_eq!(args, ["--model", "saos", "--encoding", "q4_k_m"]);
        assert!(StableAudio.launch_args(&native.args, Some("../x")).is_err());
        let profile = StableAudio.profile();
        assert_eq!(profile.tiers, ["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]);
        assert_eq!(profile.tier_components["Q5_K_M"], ["stable-audio-native::Q5_K_M"]);
        assert_eq!(
            profile.tier_options["Q5_K_M"],
            ["stable-audio-native::kickbass::Q5_K_M", "stable-audio-native::jerry-grunge::Q5_K_M"]
        );
        assert_eq!(profile.options_label.as_deref(), Some("finetunes"));
    }

    #[test]
    fn a_tier_starts_with_its_base_bundle_and_no_finetunes() {
        let root = temp_root("start");
        let blocker = StableAudio.native_start_blocker(&root, None).unwrap();
        assert!(blocker.contains("Stable Audio Open Small F16"), "{blocker}");
        let models = crate::stable_audio_models::models();
        let dir = models.models_dir(&root);
        std::fs::create_dir_all(&dir).unwrap();
        for file in &models.component("stable-audio-native::F16").unwrap().files {
            // A sparse file of the pinned size is "present" to an inventory check.
            std::fs::File::create(dir.join(&file.filename)).unwrap().set_len(file.bytes).unwrap();
        }
        assert!(StableAudio.native_start_blocker(&root, None).is_none());
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
        let env = write("services/stable-audio/env/Scripts/python.exe");
        let base = write("hf-cache/models--stabilityai--stable-audio-open-small/snapshots/x/model.safetensors");
        let finetune = write("hf-cache/models--thepatch--jerry_grunge/snapshots/x/jerry.ckpt");
        let native_model = write("models/stable-audio/finetunes/kickbass/kickbass-v1-e257-dit-0.3B-v1.0-F16.gguf");
        let shared_runtime = write("services/sa3/native/sat-server.exe");
        let preview = crate::migration_cleanup::preview(&StableAudio, &root, &hub);
        let candidates: Vec<_> = preview.cleanup_candidates.iter().map(|item| std::path::PathBuf::from(&item.path)).collect();
        assert!(preview.legacy_runtime_present);
        let repo_of = |path: &std::path::Path| path.ancestors().nth(3).unwrap().to_path_buf();
        for expected in [env.parent().unwrap().parent().unwrap().to_path_buf(), repo_of(&base), repo_of(&finetune)] {
            assert!(candidates.iter().any(|path| *path == expected), "{} is not a cleanup candidate: {candidates:?}", expected.display());
        }
        for kept in [&native_model, &shared_runtime] {
            assert!(!candidates.iter().any(|path| kept.starts_with(path)), "{} would be removed", kept.display());
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
