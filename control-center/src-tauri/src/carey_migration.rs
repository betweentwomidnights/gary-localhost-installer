//! Carey's move from its Python ACE-Step service to acestep.cpp's ace-server.
//! ace-server serves its own job API rather than Python Carey's routes, and
//! gary4juce moves to that API separately, so there is no client adapter and
//! the proof exercises ace-server's contract directly: the tasks gary4juce's
//! Carey tab will send, on the models this profile has downloaded.
//!
//! Python Carey also owns LoRA training, its LoRA registry and the caption
//! helpers, none of which run natively in gary4local yet. Until they do, a
//! fresh install still sets Python up and cleanup refuses to remove it, so
//! switching never loses them.

use crate::carey_models::{self, DEFAULT_DIT};
use crate::migration::{Proof, Run, Selection, Service, Target};
use crate::migration_inventory::Extras;
use std::path::Path;
use std::time::Duration;

const SAMPLE_RATE: u32 = 48_000;

fn carey() -> &'static crate::manifest::ServiceDef {
    crate::manifest::bundled()
        .services
        .iter()
        .find(|service| service.id == "carey")
        .expect("the bundled manifest has a carey entry")
}

fn target() -> Target<'static> {
    Target::of(carey()).expect("carey's manifest entry has a migration block")
}

/// Carey as the shared migration engine sees it.
pub struct Carey;

impl Service for Carey {
    fn target(&self) -> Target<'_> {
        target()
    }

    fn extras(&self, root: &Path, _hub: &Path) -> Extras {
        Extras {
            selection: crate::migration::read(root, target().def, |selection| self.check_selection(selection)),
            protected: Vec::new(),
            preserved: Vec::new(),
            warnings: Vec::new(),
            training_base_present: false,
        }
    }

    fn models(&self) -> &'static crate::native_models::Catalog {
        carey_models::models()
    }

    fn tiers(&self) -> &'static [&'static str] {
        carey_models::TIERS
    }

    fn tier_components(&self, tier: &str) -> Result<Vec<String>, String> {
        carey_models::tier_components(tier)
    }

    fn tier_options(&self, tier: &str) -> Vec<String> {
        carey_models::options(tier)
    }

    fn options_label(&self) -> Option<&'static str> {
        Some("more models")
    }

    fn recommendations(&self, memory_bytes: u64) -> crate::migration::Recommendations {
        carey_models::recommendations(memory_bytes)
    }

    fn blocking_jobs(&self) -> &'static [&'static str] {
        &["native-migration"]
    }

    fn python_only_features(&self) -> Option<&'static str> {
        Some("LoRA training, LoRAs and caption helpers")
    }

    /// ace-server chooses a model per request, so the tier only decides what
    /// is downloaded; it takes no encoding argument.
    fn launch_args(&self, args: &[String], encoding: Option<&str>) -> Result<Vec<String>, String> {
        self.tier_components(encoding.unwrap_or(self.tiers()[0]))?;
        Ok(args.to_vec())
    }

    fn verify<'a>(&'a self, run: Run<'a>) -> Proof<'a, Selection> {
        Box::pin(verify_native(run))
    }

    /// Activation hashed the models and generated privately; cleanup repeats
    /// that only when the runtime or the tier's models changed since.
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

/// Size and modification time of what `verify_native` always proves: the
/// server and the tier's models, with release, backend and tier.
fn fingerprint(
    root: &Path,
    native: &crate::manifest::NativeDef,
    installed: &crate::native_runtime::NativeInstall,
    encoding: &str,
) -> Result<String, String> {
    let models = carey_models::models();
    let dir = models.models_dir(root);
    let mut paths = vec![installed.dir.join(&native.executable)];
    for id in carey_models::tier_components(encoding)? {
        let component = models.component(&id).ok_or("Unknown Carey model")?;
        paths.extend(component.files.iter().map(|file| dir.join(&file.filename)));
    }
    Ok(crate::migration_verify::fingerprint(
        &paths,
        &serde_json::json!([1, installed.version, installed.backend, encoding]),
    ))
}

/// The catalog components on disk beyond the tier's own, DiTs first: every
/// one of them is something gary4juce may ask the server for.
fn downloaded_options(root: &Path, encoding: &str) -> Result<Vec<String>, String> {
    let models = carey_models::models();
    let needed = carey_models::tier_components(encoding)?;
    let mut options: Vec<String> = models
        .components()
        .iter()
        .map(|component| component.id.clone())
        .filter(|id| !needed.contains(id) && models.missing(root, std::slice::from_ref(id)).is_none())
        .collect();
    options.sort_by_key(|id| id.starts_with("carey-native::lm-"));
    Ok(options)
}

fn file_of(id: &str) -> Result<String, String> {
    let component = carey_models::models().component(id).ok_or("Unknown Carey model")?;
    let [file] = component.files.as_slice() else {
        return Err(format!("{id} is not a single model file"));
    };
    Ok(file.filename.clone())
}

/// Check a WAV is 48 kHz stereo and within `tolerance` samples of `samples`.
fn check_wav_near(wav: &[u8], samples: u32, tolerance: u32, what: &str) -> Result<(), String> {
    let reader = hound::WavReader::new(std::io::Cursor::new(wav)).map_err(|error| error.to_string())?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE || spec.channels != 2 || reader.duration().abs_diff(samples) > tolerance {
        return Err(format!(
            "{what} check returned {} Hz, {} channels, {} samples; expected {SAMPLE_RATE} Hz stereo, about {samples}",
            spec.sample_rate,
            spec.channels,
            reader.duration()
        ));
    }
    Ok(())
}

/// A short, fixed request: enough to load and run a model end to end.
fn request(synth_model: &str, extra: serde_json::Value) -> serde_json::Value {
    let mut body = serde_json::json!({
        "caption": "warm analog synth pad with soft drums",
        "lyrics": "[Instrumental]",
        "duration": 4,
        "bpm": 120,
        "keyscale": "C major",
        "timesignature": "4",
        "seed": 42,
        "inference_steps": 8,
        "synth_model": synth_model,
        "output_format": "wav16",
    });
    body.as_object_mut().unwrap().extend(extra.as_object().cloned().unwrap_or_default());
    body
}

/// Prove the installed runtime with the tier being verified, on a private
/// port: hash the tier's models, then run the tasks gary4juce will send on
/// the base model, and generate once with every other model downloaded.
async fn verify_native(run: Run<'_>) -> Result<Selection, String> {
    use crate::ace_client::{synth_audio, Body};
    use crate::migration_verify::check_wav_at;
    use serde_json::json;
    // Taken before any check, so a file that changes while this runs leaves a
    // fingerprint that no longer matches, never one that wrongly does.
    let fingerprint = fingerprint(run.root, run.native, run.installed, run.encoding)?;
    let ids = carey_models::tier_components(run.encoding)?;
    carey_models::models().verify(run.root, &ids, run.progress).await?;
    let options = downloaded_options(run.root, run.encoding)?;
    (run.progress)("Starting the private native test server...");
    let dir = crate::sa3_training::checked_folder(run.root, &["carey", "migration-checks"])?;
    let server = crate::migration_verify::PrivateServer::start(run.launch, &dir, "Carey", &[]).await?;
    let (client, base) = (&server.client, server.base.as_str());
    let wait = Duration::from_secs(600);

    (run.progress)("Listing Carey's models...");
    let props: serde_json::Value = client
        .get(format!("{base}/props"))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("Carey /props failed: {error}"))?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    let listed = |bucket: &str, file: &str| {
        props["models"][bucket]
            .as_array()
            .is_some_and(|names| names.iter().any(|name| name.as_str() == Some(file)))
    };
    let dit = carey_models::dit_file(DEFAULT_DIT, run.encoding);
    let shared = carey_models::models()
        .component(carey_models::SHARED)
        .ok_or("Carey's catalog has no shared models")?;
    if !listed("dit", &dit)
        || !listed("embedding", &shared.files[0].filename)
        || !listed("vae", &shared.files[1].filename)
    {
        return Err("Carey's native server does not list the tier's models".into());
    }

    (run.progress)("Testing native generation...");
    let generated = synth_audio(client, base, Body::Json(&request(&dit, json!({}))), wait, "Carey generation").await?;
    check_wav_at(&generated, SAMPLE_RATE, 4 * SAMPLE_RATE, "Carey generation")?;

    // A continuation regenerates from the end of the source to the requested
    // length, keeping the source as it was.
    (run.progress)("Testing a continuation...");
    let continued = synth_audio(
        client,
        base,
        Body::WithAudio(
            &request(&dit, json!({"task_type": "repaint", "duration": 8, "repainting_start": 4, "repainting_end": 8})),
            &generated,
        ),
        wait,
        "Carey continuation",
    )
    .await?;
    check_wav_at(&continued, SAMPLE_RATE, 8 * SAMPLE_RATE, "Carey continuation")?;

    (run.progress)("Testing a lego stem...");
    let stem = synth_audio(
        client,
        base,
        Body::WithAudio(&request(&dit, json!({"task_type": "lego", "track": "drums"})), &generated),
        wait,
        "Carey lego",
    )
    .await?;
    check_wav_at(&stem, SAMPLE_RATE, 4 * SAMPLE_RATE, "Carey lego")?;

    for id in options {
        let file = file_of(&id)?;
        let label = carey_models::models().component(&id).map_or(id.clone(), |component| component.label.clone());
        (run.progress)(&format!("Testing {label}..."));
        if id.starts_with("carey-native::lm-") {
            // Standard generation: the LM writes the audio codes the base DiT
            // renders, and the codes, not the request, decide the length.
            let lm = request(&dit, json!({"lm_model": file}));
            let (written, _) =
                crate::ace_client::run(client, base, "/lm", Body::Json(&lm), wait, &label).await?;
            let written: serde_json::Value = serde_json::from_slice(&written).map_err(|error| error.to_string())?;
            let first = written.get(0).cloned().ok_or_else(|| format!("{label} wrote no request"))?;
            if first["audio_codes"].as_str().unwrap_or_default().is_empty() {
                return Err(format!("{label} wrote no audio codes"));
            }
            let audio = synth_audio(client, base, Body::Json(&first), wait, &label).await?;
            check_wav_near(&audio, 4 * SAMPLE_RATE, SAMPLE_RATE / 2, &label)?;
        } else {
            let audio = synth_audio(client, base, Body::Json(&request(&file, json!({}))), wait, &label).await?;
            check_wav_at(&audio, SAMPLE_RATE, 4 * SAMPLE_RATE, &label)?;
        }
    }
    server.stop().await?;

    Ok(Selection {
        schema_version: 1,
        encoding: run.encoding.into(),
        verified_release: run.installed.version.clone().unwrap_or_else(|| "development".into()),
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
            "gary-carey-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn ace_server_takes_its_model_from_each_request_not_an_encoding_argument() {
        let args = vec!["--keep-loaded".to_string()];
        assert_eq!(Carey.launch_args(&args, Some("Q4_K_M")).unwrap(), args);
        assert_eq!(Carey.launch_args(&[], None).unwrap(), Vec::<String>::new());
        assert!(Carey.launch_args(&args, Some("../x")).is_err());
    }

    #[test]
    fn python_stays_until_training_and_loras_run_natively() {
        let blocker = Carey.retirement_blocker(Path::new("."), Path::new("."), &[]).unwrap();
        assert!(blocker.starts_with("Carey's LoRA training"), "{blocker}");
    }

    #[test]
    fn a_tier_starts_with_its_base_model_and_the_shared_components() {
        let root = temp_root("start");
        let blocker = Carey.native_start_blocker(&root, None).unwrap();
        assert!(blocker.contains("2B base BF16"), "{blocker}");
        let models = carey_models::models();
        let dir = models.models_dir(&root);
        std::fs::create_dir_all(&dir).unwrap();
        for id in carey_models::tier_components("BF16").unwrap() {
            for file in &models.component(&id).unwrap().files {
                // A sparse file of the pinned size is "present" to an inventory check.
                std::fs::File::create(dir.join(&file.filename)).unwrap().set_len(file.bytes).unwrap();
            }
        }
        assert!(Carey.native_start_blocker(&root, None).is_none());
        // Other models on disk are options: each is proved, none is required.
        let xl = models.component("carey-native::xl-base::Q8_0").unwrap();
        std::fs::File::create(dir.join(&xl.files[0].filename)).unwrap().set_len(xl.files[0].bytes).unwrap();
        assert_eq!(downloaded_options(&root, "BF16").unwrap(), ["carey-native::xl-base::Q8_0"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// GARY4LOCAL_CAREY_SMOKE_ROOT (a new folder), _MODELS (a folder holding
    /// the GGUFs) and _BINARY (an ace-server.exe); optionally _TIER and
    /// _OPTIONS (comma-separated component ids to link and prove as well).
    #[tokio::test]
    #[ignore = "verifies a real ace-server against hard-linked model GGUFs"]
    async fn real_native_runtime_verification() {
        let var = |name: &str| std::env::var(format!("GARY4LOCAL_CAREY_SMOKE_{name}")).ok();
        let root = std::path::PathBuf::from(var("ROOT").expect("a new isolated root is required"));
        assert!(!root.exists(), "the smoke root must not exist yet");
        let source = std::path::PathBuf::from(var("MODELS").expect("a model folder is required"));
        let server = std::path::PathBuf::from(var("BINARY").expect("an ace-server is required"));
        let tier = var("TIER").unwrap_or_else(|| "BF16".into());
        let mut ids = carey_models::tier_components(&tier).unwrap();
        ids.extend(var("OPTIONS").into_iter().flat_map(|list| {
            list.split(',').map(str::to_string).collect::<Vec<_>>()
        }));
        let models = carey_models::models();
        let dir = models.checked_models_dir(&root).unwrap();
        for id in &ids {
            for file in &models.component(id).unwrap().files {
                std::fs::hard_link(source.join(&file.filename), dir.join(&file.filename)).unwrap();
            }
        }
        let native = carey().native.clone().unwrap();
        let installed = crate::native_runtime::NativeInstall {
            dir: server.parent().unwrap().into(),
            backend: "cuda".into(),
            runtimes: Vec::new(),
            version: None,
            fallback_reason: None,
        };
        let launch = crate::native_runtime::NativeLaunch::new(
            &root,
            &native.executable,
            &installed,
            Carey.launch_args(&native.args, Some(&tier)).unwrap(),
            Vec::new(),
            &[("ACE_MODELS_DIR".into(), dir.to_string_lossy().into())],
        );
        let progress = |message: &str| println!("{message}");
        let run = Run {
            root: &root,
            native: &native,
            installed: &installed,
            launch: &launch,
            encoding: &tier,
            progress: &progress,
        };
        let selection = Carey.verify(run).await.unwrap();
        assert_eq!(selection.encoding, tier);
        assert!(selection.runtime_fingerprint.is_some());
        assert!(!crate::migration::selection_path(&root, target().def).exists(), "verification must not select C++");
        std::fs::remove_dir_all(root).unwrap();
    }
}
