//! gary's and terry's move from their Python services to audiocraft.cpp's
//! musicgen-server and melodyflow-server. Both servers keep the Python
//! services' routes and reply shapes on the same ports, so there is no client
//! adapter; what is each service's own is which models it needs and how its
//! runtime is proved. They share one runtime bundle, as jerry, SA3 and
//! foundation-1 share sa3.cpp's.

use crate::manifest::ServiceDef;
use crate::migration::{Proof, Run, Selection, Service, Target};
use crate::migration_inventory::Extras;
use crate::native_models::Catalog;
use std::path::Path;
use std::sync::OnceLock;

/// The one tier each service ships: F16 weights, with the F32 codecs they
/// were converted with.
const TIERS: &[&str] = &["F16"];

/// The finetune gary's C++ service starts with. gary4juce sends the repo id
/// as `model_name`, and musicgen-server lists and accepts it by that id.
const GARY_MODEL: &str = "thepatch/vanya_ai_dnb_0.1";
const GARY_SHARED: &str = "gary-native::shared";
const GARY_VANYA: &str = "gary-native::vanya_ai_dnb_0.1";
const TERRY_F16: &str = "melodyflow-native::F16";

/// The finetunes gary can add, each published at these precisions, most
/// faithful first. They are musicgen-large: 6.6 GB at F16, too much for an
/// 8 GB GPU beside the KV cache, so they ship quantized.
const GARY_FINETUNES: &[&str] = &["hoenn_lofi", "keygen-gary-v2-large-12"];
const FINETUNE_TIERS: &[&str] = &["Q8_0", "Q4_K_M"];

/// What generating needs beside a finetune's weights: a 30-second KV cache
/// for each guidance stream, the graph, and what the driver keeps.
const GARY_HEADROOM_BYTES: u64 = 3 << 30;

fn finetune_component(finetune: &str, tier: &str) -> String {
    format!("gary-native::{finetune}::{tier}")
}

fn component_bytes(id: &str) -> u64 {
    gary_models().component(id).map_or(0, |component| component.files.iter().map(|file| file.bytes).sum())
}

/// The precision of `finetune` to badge for a GPU with `memory_bytes`: the
/// most faithful one that fits beside the headroom, else the smallest.
fn recommended_tier(finetune: &str, memory_bytes: u64) -> &'static str {
    FINETUNE_TIERS
        .iter()
        .copied()
        .find(|tier| component_bytes(&finetune_component(finetune, tier)) + GARY_HEADROOM_BYTES <= memory_bytes)
        .unwrap_or(FINETUNE_TIERS[FINETUNE_TIERS.len() - 1])
}

/// The preset terry's proof transforms with.
const TERRY_VARIATION: &str = "accordion_folk";

fn def(id: &str) -> &'static ServiceDef {
    crate::manifest::bundled()
        .services
        .iter()
        .find(|service| service.id == id)
        .expect("the bundled manifest has gary and melodyflow entries")
}

fn target(id: &str) -> Target<'static> {
    Target::of(def(id)).expect("gary and melodyflow have migration blocks")
}

fn gary_models() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| Catalog::bundled("gary", include_str!("../gary-models.json")))
}

fn terry_models() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| Catalog::bundled("melodyflow", include_str!("../melodyflow-models.json")))
}

fn check_tier(tier: &str, components: &[&str]) -> Result<Vec<String>, String> {
    if !TIERS.contains(&tier) {
        return Err("audiocraft.cpp's services run F16 models.".into());
    }
    Ok(components.iter().map(|id| id.to_string()).collect())
}

/// gary (MusicGen) as the shared migration engine sees it.
pub struct Gary;

/// terry (MelodyFlow) as the shared migration engine sees it.
pub struct Terry;

impl Service for Gary {
    fn target(&self) -> Target<'_> {
        target("gary")
    }

    fn extras(&self, root: &Path, hub: &Path) -> Extras {
        // Python downloads any of gary4juce's finetunes on demand; C++ gary
        // lists only what has been converted, which is what has a -GGUF repo
        // in the catalog. Say so before cleanup takes the Python copies away.
        let converted: Vec<&str> = gary_models()
            .components()
            .iter()
            .flat_map(|component| &component.files)
            .filter_map(|file| file.repo.strip_suffix("-GGUF"))
            .collect();
        let others: Vec<&str> = self
            .target()
            .def
            .legacy
            .hf_repos
            .iter()
            .filter(|repo| repo.repo.starts_with("thepatch/") && !converted.contains(&repo.repo.as_str()))
            .filter(|repo| hub.join(repo.cache_dir_name()).exists())
            .map(|repo| repo.repo.trim_start_matches("thepatch/"))
            .collect();
        let warnings = if others.is_empty() {
            Vec::new()
        } else {
            vec![format!(
                "C++ gary runs the finetunes converted so far. These ones you downloaded for Python aren't converted yet: {}.",
                others.join(", ")
            )]
        };
        extras(self, root, warnings)
    }

    fn models(&self) -> &'static Catalog {
        gary_models()
    }

    fn tiers(&self) -> &'static [&'static str] {
        TIERS
    }

    fn tier_components(&self, tier: &str) -> Result<Vec<String>, String> {
        check_tier(tier, &[GARY_SHARED, GARY_VANYA])
    }

    fn tier_options(&self, tier: &str) -> Vec<String> {
        if !TIERS.contains(&tier) {
            return Vec::new();
        }
        GARY_FINETUNES
            .iter()
            .flat_map(|finetune| FINETUNE_TIERS.iter().map(move |precision| finetune_component(finetune, precision)))
            .collect()
    }

    fn options_label(&self) -> Option<&'static str> {
        Some("finetunes")
    }

    fn recommendations(&self, memory_bytes: u64) -> crate::migration::Recommendations {
        crate::migration::Recommendations {
            tier: Some(TIERS[0].into()),
            components: GARY_FINETUNES
                .iter()
                .map(|finetune| finetune_component(finetune, recommended_tier(finetune, memory_bytes)))
                .collect(),
        }
    }

    fn launch_args(&self, args: &[String], encoding: Option<&str>) -> Result<Vec<String>, String> {
        self.tier_components(encoding.unwrap_or(TIERS[0]))?;
        Ok(args.to_vec())
    }

    fn blocking_jobs(&self) -> &'static [&'static str] {
        &["native-migration"]
    }

    fn verify<'a>(&'a self, run: Run<'a>) -> Proof<'a, Selection> {
        Box::pin(verify_gary(run))
    }

    fn reverify<'a>(&'a self, run: Run<'a>, selection: &'a Selection) -> Proof<'a, Option<String>> {
        Box::pin(async move {
            if unchanged(self, &run, selection) {
                return Ok(None);
            }
            Ok(verify_gary(run).await?.runtime_fingerprint)
        })
    }
}

impl Service for Terry {
    fn target(&self) -> Target<'_> {
        target("melodyflow")
    }

    fn extras(&self, root: &Path, _hub: &Path) -> Extras {
        extras(self, root, Vec::new())
    }

    fn models(&self) -> &'static Catalog {
        terry_models()
    }

    fn tiers(&self) -> &'static [&'static str] {
        TIERS
    }

    fn tier_components(&self, tier: &str) -> Result<Vec<String>, String> {
        check_tier(tier, &[TERRY_F16])
    }

    fn launch_args(&self, args: &[String], encoding: Option<&str>) -> Result<Vec<String>, String> {
        self.tier_components(encoding.unwrap_or(TIERS[0]))?;
        Ok(args.to_vec())
    }

    fn blocking_jobs(&self) -> &'static [&'static str] {
        &["native-migration"]
    }

    fn verify<'a>(&'a self, run: Run<'a>) -> Proof<'a, Selection> {
        Box::pin(verify_terry(run))
    }

    fn reverify<'a>(&'a self, run: Run<'a>, selection: &'a Selection) -> Proof<'a, Option<String>> {
        Box::pin(async move {
            if unchanged(self, &run, selection) {
                return Ok(None);
            }
            Ok(verify_terry(run).await?.runtime_fingerprint)
        })
    }
}

fn extras(service: &dyn Service, root: &Path, warnings: Vec<String>) -> Extras {
    Extras {
        selection: crate::migration::read(root, service.target().def, |selection| {
            service.check_selection(selection)
        }),
        protected: Vec::new(),
        preserved: Vec::new(),
        warnings,
        training_base_present: false,
    }
}

/// Activation hashed the models and ran every route privately; cleanup
/// repeats that only when the runtime or models changed since.
fn unchanged(service: &dyn Service, run: &Run<'_>, selection: &Selection) -> bool {
    let unchanged = fingerprint(service, run)
        .is_ok_and(|current| selection.runtime_fingerprint.as_deref() == Some(current.as_str()));
    if unchanged {
        (run.progress)("The C++ runtime is unchanged since it was verified.");
    }
    unchanged
}

/// Size and modification time of what a proof covers: the server and the
/// tier's models, with release, backend and tier. Metadata only.
fn fingerprint(service: &dyn Service, run: &Run<'_>) -> Result<String, String> {
    let models = service.models();
    let dir = models.models_dir(run.root);
    let mut paths = vec![run.installed.dir.join(&run.native.executable)];
    for id in service.tier_components(run.encoding)? {
        let component = models.component(&id).ok_or("Unknown audiocraft model")?;
        paths.extend(component.files.iter().map(|file| dir.join(&file.filename)));
    }
    Ok(crate::migration_verify::fingerprint(
        &paths,
        &serde_json::json!([1, run.installed.version, run.installed.backend, run.encoding]),
    ))
}

/// Hash the tier's models, then start the installed server on a private port.
async fn start(
    service: &dyn Service,
    run: &Run<'_>,
    label: &str,
) -> Result<(String, crate::migration_verify::PrivateServer), String> {
    // Taken before any check, so a file that changes while this runs leaves a
    // fingerprint that no longer matches, never one that wrongly does.
    let fingerprint = fingerprint(service, run)?;
    let ids = service.tier_components(run.encoding)?;
    service.models().verify(run.root, &ids, run.progress).await?;
    (run.progress)("Starting the private native test server...");
    let data_dir = service.target().def.data_dir.as_str();
    let dir = crate::sa3_training::checked_folder(run.root, &[data_dir, "migration-checks"])?;
    let server = crate::migration_verify::PrivateServer::start(run.launch, &dir, label, &[]).await?;
    Ok((fingerprint, server))
}

fn verified(run: &Run<'_>, fingerprint: String) -> Selection {
    Selection {
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
    }
}

/// A plain tone at the host's rate, as gary4juce sends a recording.
fn test_recording(seconds: u32) -> Vec<u8> {
    const RATE: u32 = 44_100;
    let spec = hound::WavSpec { channels: 2, sample_rate: RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut wav = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut wav, spec).expect("an in-memory WAV");
    for i in 0..RATE * seconds {
        let sample = (8000.0 * (2.0 * std::f32::consts::PI * 110.0 * i as f32 / RATE as f32).sin()) as i16;
        writer.write_sample(sample).expect("an in-memory WAV");
        writer.write_sample(sample).expect("an in-memory WAV");
    }
    writer.finalize().expect("an in-memory WAV");
    wav.into_inner()
}

async fn get_json(server: &crate::migration_verify::PrivateServer, path: &str, what: &str) -> Result<serde_json::Value, String> {
    server
        .client
        .get(format!("{}{path}", server.base))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("{what} failed: {error}"))?
        .json()
        .await
        .map_err(|error| error.to_string())
}

/// Whether gary's model list offers `path`, alone or as a checkpoint in a
/// group, in any size.
fn lists(models: &serde_json::Value, path: &str) -> bool {
    models["models"].as_object().is_some_and(|sizes| {
        sizes.values().filter_map(|entries| entries.as_array()).flatten().any(|entry| {
            entry["path"] == path
                || entry["checkpoints"]
                    .as_array()
                    .is_some_and(|checkpoints| checkpoints.iter().any(|checkpoint| checkpoint["path"] == path))
        })
    })
}

/// The memory of the GPU the server will use, from its own `--props` probe:
/// the largest device of the runtime's backend, as the server prefers it.
async fn gpu_memory(run: &Run<'_>) -> Option<u64> {
    let mut command = run.launch.tokio_command();
    crate::hide_console_window(&mut command);
    let output = command.arg("--props").kill_on_drop(true).output().await.ok()?;
    let props: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    props["devices"]
        .as_array()?
        .iter()
        .filter(|device| device["type"] != "cpu")
        .filter(|device| {
            device["backend"].as_str().is_some_and(|name| name.to_ascii_lowercase().contains(&run.installed.backend))
        })
        .filter_map(|device| device["memory_total_bytes"].as_u64())
        .max()
}

/// Run every route gary4juce calls on gary: the model list, a generation
/// from each end of the recording, and a retry of one.
async fn verify_gary(run: Run<'_>) -> Result<Selection, String> {
    use base64::Engine;
    use crate::migration_verify::{check_wav_spec, run_session_job};
    use serde_json::json;
    const POLL: &str = "/api/juce/poll_status";
    // Eight seconds of 32 kHz mono, whichever end the prompt comes from.
    const SAMPLES: u32 = 8 * 32_000;
    let (fingerprint, server) = start(&Gary, &run, "gary").await?;

    (run.progress)("Listing gary's models...");
    let listed = get_json(&server, "/api/models", "gary /api/models").await?;
    // gary4juce builds its dropdown from objects only, and sends `path` back.
    let offered = listed["models"]["small"]
        .as_array()
        .is_some_and(|models| models.iter().any(|model| model["type"] == "single" && model["path"] == GARY_MODEL));
    if !offered {
        return Err("gary's C++ server does not list vanya_ai_dnb_0.1 the way gary4juce reads it. Update the runtime.".into());
    }

    let recording = base64::engine::general_purpose::STANDARD.encode(test_recording(6));
    (run.progress)("Testing a generation from the start of the recording...");
    let (session, generated) = run_session_job(
        &server,
        "/api/juce/process_audio",
        POLL,
        &json!({"audio_data": recording, "model_name": GARY_MODEL, "prompt_duration": 4, "duration": 8, "seed": 42}),
        "gary process_audio",
    )
    .await?;
    check_wav_spec(&generated, 32_000, 1, SAMPLES, "gary process_audio")?;

    // A description turns on text guidance, which the other two leave off.
    (run.progress)("Testing a continuation with a description...");
    let (_, continued) = run_session_job(
        &server,
        "/api/juce/continue_music",
        POLL,
        &json!({"audio_data": recording, "model_name": GARY_MODEL, "prompt_duration": 4, "duration": 8, "seed": 42, "description": "drum and bass"}),
        "gary continue_music",
    )
    .await?;
    check_wav_spec(&continued, 32_000, 1, SAMPLES, "gary continue_music")?;

    (run.progress)("Testing a retry...");
    let (_, retried) = run_session_job(
        &server,
        "/api/juce/retry_music",
        POLL,
        // gary4juce's retry fields; it sends no duration, so a retry is 30 s
        // there as on Python. Eight keeps the proof quick.
        &json!({
            "session_id": session, "prompt_duration": 4, "model_name": GARY_MODEL, "top_k": 250,
            "temperature": 1.0, "cfg_coef": 3.0, "description": "", "seed": 7, "duration": 8,
        }),
        "gary retry_music",
    )
    .await?;
    check_wav_spec(&retried, 32_000, 1, SAMPLES, "gary retry_music")?;

    // Each downloaded finetune, at the precision the server runs: its most
    // faithful downloaded one. One too big for this GPU is the user's choice
    // to have downloaded, not a broken runtime, so it is listed but not run.
    let memory = gpu_memory(&run).await;
    for finetune in GARY_FINETUNES {
        let Some(component) = FINETUNE_TIERS
            .iter()
            .map(|precision| finetune_component(finetune, precision))
            .find(|id| gary_models().missing(run.root, std::slice::from_ref(id)).is_none())
        else {
            continue;
        };
        let repo = format!("thepatch/{finetune}");
        if !lists(&listed, &repo) {
            return Err(format!("gary's C++ server does not list {finetune}"));
        }
        if memory.is_some_and(|memory| component_bytes(&component) + GARY_HEADROOM_BYTES > memory) {
            (run.progress)(&format!("Skipping {finetune}: it needs more memory than this GPU has."));
            continue;
        }
        (run.progress)(&format!("Testing {finetune}..."));
        let what = format!("gary {finetune}");
        let (_, generated) = run_session_job(
            &server,
            "/api/juce/process_audio",
            POLL,
            &json!({"audio_data": recording, "model_name": repo, "prompt_duration": 4, "duration": 6, "seed": 42}),
            &what,
        )
        .await?;
        check_wav_spec(&generated, 32_000, 1, 6 * 32_000, &what)?;
    }
    server.stop().await?;
    Ok(verified(&run, fingerprint))
}

/// Run every route gary4juce calls on terry: a transform and its undo, after
/// checking the preset it uses is offered.
async fn verify_terry(run: Run<'_>) -> Result<Selection, String> {
    use base64::Engine;
    use crate::migration_verify::{check_wav_at, run_session_job};
    use serde_json::json;
    let (fingerprint, server) = start(&Terry, &run, "terry").await?;

    (run.progress)("Listing terry's presets...");
    let presets = get_json(&server, "/variations", "terry /variations").await?;
    if presets["variations"][TERRY_VARIATION]["prompt"].as_str().map_or(true, str::is_empty) {
        return Err(format!("terry's C++ server does not offer the {TERRY_VARIATION} preset"));
    }

    // MelodyFlow edits at 48 kHz stereo and keeps the recording's length.
    let recording = test_recording(4);
    (run.progress)("Testing a transform...");
    let (session, transformed) = run_session_job(
        &server,
        "/api/juce/transform_audio",
        "/api/juce/poll_status",
        &json!({
            "audio_data": base64::engine::general_purpose::STANDARD.encode(&recording),
            "variation": TERRY_VARIATION, "seed": 42,
        }),
        "terry transform_audio",
    )
    .await?;
    check_wav_at(&transformed, 48_000, 4 * 48_000, "terry transform_audio")?;

    (run.progress)("Testing undo...");
    let undone: serde_json::Value = server
        .client
        .post(format!("{}/api/juce/undo_transform", server.base))
        .json(&json!({"session_id": session}))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("terry undo_transform failed: {error}"))?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    let restored = base64::engine::general_purpose::STANDARD
        .decode(undone["audio_data"].as_str().unwrap_or_default())
        .map_err(|error| error.to_string())?;
    if restored != recording {
        return Err("terry undo_transform did not return the audio that was sent".into());
    }
    server.stop().await?;
    Ok(verified(&run, fingerprint))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "gary-audiocraft-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn services() -> [(&'static str, &'static dyn Service); 2] {
        [("gary", &Gary), ("melodyflow", &Terry)]
    }

    #[test]
    fn both_servers_serve_the_python_contract_on_the_same_ports() {
        for (id, service, executable, port_var) in [
            ("gary", &Gary as &dyn Service, "musicgen-server.exe", "AC_MUSICGEN_PORT"),
            ("melodyflow", &Terry, "melodyflow-server.exe", "AC_MELODYFLOW_PORT"),
        ] {
            let svc = def(id);
            let native = svc.native.as_ref().unwrap();
            assert_eq!(native.bundle.as_deref(), Some("audiocraft"));
            assert_eq!(native.executable, executable);
            assert_eq!(native.models_dir.as_deref(), Some(id));
            assert_eq!(native.env.get(port_var).map(String::as_str), Some(svc.port.to_string().as_str()));
            assert_eq!(native.env.get("AC_MODELS_DIR").map(String::as_str), Some(format!("${{MODELS_DIR}}/{id}").as_str()));
            // No client adapter: the native server takes the public port itself.
            assert!(service.client_adapter(svc.port, &[], Path::new(".")).unwrap().is_none());
            // The servers take no --encoding; the default would add one.
            assert!(service.launch_args(&native.args, None).unwrap().is_empty());
            assert!(service.launch_args(&native.args, Some("Q8_0")).is_err());
            let profile = service.profile();
            assert_eq!(profile.tiers, ["F16"]);
            assert!(profile.features.is_empty());
            assert!(service.python_only_features().is_none());
        }
        assert_eq!(Gary.profile().tier_components["F16"], [GARY_SHARED, GARY_VANYA]);
        assert_eq!(Terry.profile().tier_components["F16"], [TERRY_F16]);
    }

    #[test]
    fn every_model_is_pinned_and_each_bundle_is_complete() {
        // Each finetune comes from its own repo; the shared files and vanya
        // from vanya's, and terry's from MelodyFlow's.
        let repo_of = |id: &str| match id.split("::").nth(1) {
            Some("shared") | Some("vanya_ai_dnb_0.1") => "thepatch/vanya_ai_dnb_0.1-GGUF".to_string(),
            Some("F16") => "thepatch/melodyflow-t24-30secs-GGUF".to_string(),
            Some(finetune) => format!("thepatch/{finetune}-GGUF"),
            None => unreachable!(),
        };
        for catalog in [gary_models(), terry_models()] {
            for entry in catalog.components() {
                for file in &entry.files {
                    file.validate().unwrap();
                    assert_eq!(file.repo, repo_of(&entry.id), "{}", entry.id);
                }
            }
        }
        let files = |catalog: &Catalog, ids: Vec<String>| -> Vec<String> {
            ids.iter()
                .flat_map(|id| catalog.component(id).unwrap().files.iter().map(|file| file.filename.clone()))
                .collect()
        };
        // musicgen-server finds its codec and T5 by these names, and lists
        // every musicgen-*.gguf beside them.
        assert_eq!(files(gary_models(), Gary.tier_components("F16").unwrap()), [
            "encodec-32khz-v1.0-F32.gguf",
            "t5-base-encoder-0.1B-v1.0-F16.gguf",
            "musicgen-vanya-dnb-0.4B-v1.0-F16.gguf",
        ]);
        assert_eq!(files(terry_models(), Terry.tier_components("F16").unwrap()), [
            "melodyflow-dit-t24-1.0B-v1.0-F16.gguf",
            "melodyflow-vae-48khz-v1.0-F32.gguf",
            "t5-base-encoder-0.1B-v1.0-F16.gguf",
        ]);
        // The same T5 file in both folders.
        let t5 = |catalog: &Catalog, id: &str| {
            catalog.component(id).unwrap().files.iter().find(|file| file.filename.starts_with("t5-")).unwrap().sha256.clone()
        };
        assert_eq!(t5(gary_models(), GARY_SHARED), t5(terry_models(), TERRY_F16));
        assert!(Gary.tier_components("../F16").is_err());
    }

    /// Every file each service bundles at the top of its folder is declared,
    /// so a new one cannot be left behind by cleanup unnoticed, and nothing
    /// declared is missing from it.
    #[test]
    fn the_manifest_declares_exactly_the_bundled_python_code() {
        for (id, service) in services() {
            let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../services").join(id);
            let code = &service.target().def.legacy.code;
            for file in &code.files {
                assert!(bundled.join(file).is_file(), "{id}/{file} is declared but not bundled");
            }
            for package in &code.packages {
                assert!(bundled.join(package).is_dir(), "{id}/{package} is declared but not bundled");
            }
            let output = std::process::Command::new("git").args(["ls-files", "."]).current_dir(&bundled).output().unwrap();
            for tracked in String::from_utf8(output.stdout).unwrap().lines() {
                let top = tracked.split('/').next().unwrap();
                assert!(
                    code.files.iter().any(|file| file == top) || code.packages.iter().any(|package| package == top),
                    "services/{id}/{top} is bundled but not declared in its migration block"
                );
            }
        }
    }

    #[test]
    fn cleanup_retires_the_python_install_and_keeps_native_models() {
        let root = temp_root("cleanup");
        let hub = root.join("hf-cache");
        let write = |path: std::path::PathBuf| {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"fixture").unwrap();
            path
        };
        let shared_runtime = write(root.join("services/audiocraft/native/musicgen-server.exe"));
        let t5 = write(hub.join("models--t5-base/snapshots/a/model.safetensors"));
        for (id, service, repo, native_model) in [
            ("gary", &Gary as &dyn Service, "models--thepatch--vanya_ai_dnb_0.1", "models/gary/musicgen-vanya-dnb-0.4B-v1.0-F16.gguf"),
            ("melodyflow", &Terry, "models--facebook--melodyflow-t24-30secs", "models/melodyflow/melodyflow-dit-t24-1.0B-v1.0-F16.gguf"),
        ] {
            let env = write(root.join(format!("services/{id}/env/Scripts/python.exe")));
            let weights = write(hub.join(repo).join("snapshots/a/state_dict.bin"));
            let native_model = write(root.join(native_model));
            let preview = crate::migration_cleanup::preview(service, &root, &hub);
            let candidates: Vec<_> = preview.cleanup_candidates.iter().map(|item| std::path::PathBuf::from(&item.path)).collect();
            assert!(preview.legacy_runtime_present, "{id}");
            for expected in [env.parent().unwrap().parent().unwrap(), &hub.join(repo)] {
                assert!(candidates.iter().any(|path| path == expected), "{id}: {} is not a cleanup candidate: {candidates:?}", expected.display());
            }
            assert!(weights.starts_with(hub.join(repo)));
            // T5 is shared with the other Python services, so neither claims it.
            for kept in [&native_model, &shared_runtime, &t5] {
                assert!(!candidates.iter().any(|path| kept.starts_with(path)), "{id}: {} would be removed", kept.display());
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gary_warns_about_finetunes_it_cannot_run_yet() {
        let root = temp_root("finetunes");
        let hub = root.join("hf-cache");
        std::fs::create_dir_all(hub.join("models--thepatch--vanya_ai_dnb_0.1")).unwrap();
        assert!(Gary.extras(&root, &hub).warnings.is_empty());
        std::fs::create_dir_all(hub.join("models--thepatch--bleeps-large-6")).unwrap();
        std::fs::create_dir_all(hub.join("models--thepatch--hoenn_lofi")).unwrap();
        let warnings = Gary.extras(&root, &hub).warnings;
        assert_eq!(warnings.len(), 1);
        // hoenn_lofi is converted now, so only bleeps-large-6 is named.
        assert!(warnings[0].ends_with("converted yet: bleeps-large-6."), "{warnings:?}");
        std::fs::remove_dir_all(root).unwrap();
    }

    /// GARY4LOCAL_AUDIOCRAFT_SMOKE_ROOT (a new folder), _MODELS (a folder
    /// holding the GGUFs) and _BIN (the folder with both servers); optionally
    /// _BACKEND (cuda by default) and _OPTIONS (comma-separated gary finetune
    /// components to link and prove as well).
    #[tokio::test]
    #[ignore = "verifies real audiocraft.cpp servers against hard-linked model GGUFs"]
    async fn real_native_runtime_verification() {
        let var = |name: &str| std::env::var(format!("GARY4LOCAL_AUDIOCRAFT_SMOKE_{name}")).ok();
        let root = std::path::PathBuf::from(var("ROOT").expect("a new isolated root is required"));
        assert!(!root.exists(), "the smoke root must not exist yet");
        let source = std::path::PathBuf::from(var("MODELS").expect("a model folder is required"));
        let bin = std::path::PathBuf::from(var("BIN").expect("the servers' folder is required"));
        let backend = var("BACKEND").unwrap_or_else(|| "cuda".into());
        let options: Vec<String> = var("OPTIONS").map_or_else(Vec::new, |list| list.split(',').map(str::to_string).collect());
        for (id, service) in services() {
            let models = service.models();
            let dir = models.checked_models_dir(&root).unwrap();
            let mut components = service.tier_components("F16").unwrap();
            components.extend(options.iter().filter(|option| models.component(option).is_some()).cloned());
            for component in components {
                for file in &models.component(&component).unwrap().files {
                    std::fs::hard_link(source.join(&file.filename), dir.join(&file.filename)).unwrap();
                }
            }
            let native = def(id).native.clone().unwrap();
            let installed = crate::native_runtime::NativeInstall {
                dir: bin.clone(),
                backend: backend.clone(),
                runtimes: Vec::new(),
                version: None,
                fallback_reason: None,
            };
            let launch = crate::native_runtime::NativeLaunch::new(
                &root,
                &native.executable,
                &installed,
                service.launch_args(&native.args, None).unwrap(),
                Vec::new(),
                &[
                    ("AC_MODELS_DIR".into(), dir.to_string_lossy().into()),
                    ("AC_DEVICE".into(), backend.clone()),
                ],
            );
            let progress = |message: &str| println!("{id}: {message}");
            let run = Run { root: &root, native: &native, installed: &installed, launch: &launch, encoding: "F16", progress: &progress };
            let selection = service.verify(run).await.unwrap();
            assert_eq!(selection.encoding, "F16");
            assert!(selection.runtime_fingerprint.is_some());
            assert!(!crate::migration::selection_path(&root, service.target().def).exists(), "verification must not select C++");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_large_finetunes_are_optional_at_each_precision() {
        let options = Gary.tier_options("F16");
        assert_eq!(options, [
            "gary-native::hoenn_lofi::Q8_0",
            "gary-native::hoenn_lofi::Q4_K_M",
            "gary-native::keygen-gary-v2-large-12::Q8_0",
            "gary-native::keygen-gary-v2-large-12::Q4_K_M",
        ]);
        for id in &options {
            let component = gary_models().component(id).unwrap();
            let [file] = component.files.as_slice() else { panic!("{id} is one file") };
            // musicgen-server names a model by the part before its size label
            // and ranks the -<ENCODING> suffix.
            let precision = id.rsplit("::").next().unwrap();
            assert!(file.filename.starts_with("musicgen-") && file.filename.ends_with(&format!("-3.3B-v1.0-{precision}.gguf")), "{}", file.filename);
        }
        assert!(Gary.tier_options("Q8_0").is_empty());
        assert!(Terry.tier_options("F16").is_empty());
        assert_eq!(Gary.options_label(), Some("finetunes"));
    }

    #[test]
    fn a_finetune_is_recommended_at_the_best_precision_that_fits() {
        const MIB: u64 = 1 << 20;
        // The RTX 5070 Laptop GPU reports 8151 MiB, and a guided 30-second
        // hoenn_lofi continuation peaked at 5,853 MiB over idle with Q8_0.
        let laptop = Gary.recommendations(8151 * MIB);
        assert_eq!(laptop.tier.as_deref(), Some("F16"));
        assert_eq!(laptop.components, ["gary-native::hoenn_lofi::Q8_0", "gary-native::keygen-gary-v2-large-12::Q8_0"]);
        // A 6 GB card gets Q4_K_M (peak 4,609 MiB over idle).
        assert_eq!(Gary.recommendations(6 * 1024 * MIB).components, [
            "gary-native::hoenn_lofi::Q4_K_M",
            "gary-native::keygen-gary-v2-large-12::Q4_K_M",
        ]);
        // Smaller still is offered the smallest rather than nothing.
        assert_eq!(recommended_tier("hoenn_lofi", 2 * 1024 * MIB), "Q4_K_M");
    }

    #[test]
    fn a_finetune_is_listed_alone_or_as_a_checkpoint() {
        let listed = serde_json::json!({"models": {
            "small": [{"name": "vanya_ai_dnb_0.1", "path": "thepatch/vanya_ai_dnb_0.1", "type": "single"}],
            "medium": [],
            "large": [
                {"name": "hoenn_lofi", "path": "thepatch/hoenn_lofi", "type": "single"},
                {"name": "keygen-gary-v2-large", "type": "group", "checkpoints": [
                    {"name": "keygen-gary-v2-large-12", "path": "thepatch/keygen-gary-v2-large-12", "epoch": 12}
                ]}
            ]
        }});
        for path in [GARY_MODEL, "thepatch/hoenn_lofi", "thepatch/keygen-gary-v2-large-12"] {
            assert!(lists(&listed, path), "{path}");
        }
        assert!(!lists(&listed, "thepatch/keygen-gary-v2-large-16"));
        assert!(!lists(&serde_json::json!({"models": {"small": ["thepatch/hoenn_lofi"]}}), "thepatch/hoenn_lofi"));
    }

    #[test]
    fn the_test_recording_is_what_gary4juce_sends() {
        let wav = test_recording(2);
        let reader = hound::WavReader::new(std::io::Cursor::new(&wav)).unwrap();
        assert_eq!((reader.spec().sample_rate, reader.spec().channels, reader.duration()), (44_100, 2, 88_200));
    }
}
