//! Desktop caption model discovery and job adapters.
use std::path::{Path, PathBuf};
use crate::migration_verify::PrivateServer;
use gary4local_core::carey_caption::{self as shared, CaptionEndpoint};
pub use shared::Options;
#[cfg(test)]
use shared::{sa3_genre, sa3_sidecar, sidecar_fields, filename_bpm, choose_bpm, choose_key, normalize_keyscale, quality_error};
/// The downloaded build of the `size` LM ("0.6B", "1.7B", "4B") to caption
/// with, best precision first.
pub fn lm_file(root: &Path, size: &str) -> Result<String, String> {
    let models = crate::carey_models::models();
    crate::carey_models::TIERS
        .iter()
        .filter_map(|tier| models.component(&format!("carey-native::lm-{size}::{tier}")))
        .find(|component| models.missing(root, std::slice::from_ref(&component.id)).is_none())
        .map(|component| component.files[0].filename.clone())
        .ok_or_else(|| format!("Download the {size} language model from Carey's models to caption with it."))
}

/// The shared VAE every Carey tier downloads.
pub fn vae_file() -> Result<String, String> {
    crate::carey_models::models()
        .component(crate::carey_models::SHARED)
        .map(|shared| shared.files[1].filename.clone())
        .ok_or_else(|| "Carey's catalog has no shared models".into())
}

/// Captioning needs the audio VAE, but not the shared text encoder. Inspect
/// its pinned file independently so an LM-only download is reported accurately.
pub fn caption_vae(root: &Path) -> Result<String, String> {
    let models = crate::carey_models::models();
    let shared = models.component(crate::carey_models::SHARED)
        .ok_or("Carey's catalog has no shared models")?;
    let filename = vae_file()?;
    let file = shared.files.iter().find(|file| file.filename == filename)
        .ok_or("Carey's catalog has no audio VAE")?;
    let present = std::fs::symlink_metadata(models.models_dir(root).join(&filename))
        .is_ok_and(|metadata| metadata.file_type().is_file() && metadata.len() == file.bytes);
    if !present { return Err("Download Carey's shared VAE from Carey → Models before auto-labeling.".into()); }
    Ok(filename)
}

/// A downloaded DiT whose audio tokenizer feeds the LM, for callers that do
/// not choose one: the 2B models before XL, best precision first. Every
/// Carey tier downloads 2B base, so one is there once Carey is migrated.
pub fn caption_dit(root: &Path) -> Result<String, String> {
    use crate::carey_models::{DEFAULT_DIT, SIBLING_DITS, TIERS, XL_DITS};
    let models = crate::carey_models::models();
    std::iter::once(DEFAULT_DIT)
        .chain(SIBLING_DITS.iter().copied())
        .chain(XL_DITS.iter().copied())
        .flat_map(|variant| TIERS.iter().map(move |tier| crate::carey_models::dit_component(variant, tier)))
        .filter_map(|id| models.component(&id))
        .find(|component| models.missing(root, std::slice::from_ref(&component.id)).is_none())
        .map(|component| component.files[0].filename.clone())
        .ok_or_else(|| "Download one of Carey's models to caption with.".into())
}

pub async fn caption_dataset(server: &PrivateServer, options: &Options, scratch: &Path, progress: impl FnMut(shared::Progress<'_>), cancelled: impl Fn() -> bool) -> Result<(usize, usize), String> {
    shared::caption_dataset(&CaptionEndpoint { client: &server.client, base: &server.base }, options, scratch, progress, cancelled).await
}
pub async fn label_sa3_dataset(server: &PrivateServer, options: &Options, files: &[PathBuf], labeled: bool, scratch: &Path, progress: impl FnMut(shared::Progress<'_>), cancelled: impl Fn() -> bool) -> Result<(), String> {
    shared::label_sa3_dataset(&CaptionEndpoint { client: &server.client, base: &server.base }, options, files, labeled, scratch, progress, cancelled).await
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;
    use crate::ace_client::{self, Body};

    #[test]
    fn bpm_follows_the_python_precedence() {
        assert_eq!(filename_bpm("break_bpm92_take2.wav"), Some(92));
        assert_eq!(filename_bpm("bpm-174.flac"), Some(174));
        assert_eq!(filename_bpm("subpm120.wav"), None);
        assert_eq!(choose_bpm(Some(92), Some(140.0), Some((90.0, 3.0))).source, "filename");
        assert_eq!(choose_bpm(None, Some(120.0), None).bpm, Some(120));
        assert_eq!(choose_bpm(None, None, None).source, "missing");
        assert_eq!(choose_bpm(None, None, Some((97.6, 2.0))).bpm, Some(98));
        // Half-time agrees: the LM's value stands.
        let half = choose_bpm(None, Some(87.0), Some((174.0, 2.0)));
        assert_eq!((half.bpm, half.source), (Some(87), "lm_agrees_with_local"));
        let low = choose_bpm(None, Some(120.0), Some((97.0, 1.1)));
        assert_eq!((low.bpm, low.source), (Some(120), "lm_local_low_confidence"));
        let overrode = choose_bpm(None, Some(120.0), Some((97.0, 1.5)));
        assert_eq!((overrode.bpm, overrode.source), (Some(97), "local_overrode_lm"));
    }

    #[test]
    fn keys_normalize_and_follow_the_python_precedence() {
        assert_eq!(normalize_keyscale("bb maj"), "A# major");
        assert_eq!(normalize_keyscale("F♯_minor"), "F# minor");
        assert_eq!(normalize_keyscale("N/A"), "");
        assert_eq!(normalize_keyscale("C dorian"), "");
        assert_eq!(choose_key("F minor", Some(("F minor", 0.01))).source, "lm_agrees_with_local");
        let overrode = choose_key("C major", Some(("A minor", 0.2)));
        assert_eq!((overrode.keyscale.as_str(), overrode.source), ("A minor", "local_overrode_lm"));
        let low = choose_key("C major", Some(("A minor", 0.1)));
        assert_eq!((low.keyscale.as_str(), low.source), ("C major", "lm_local_low_confidence"));
        assert_eq!(choose_key("", None).source, "missing");
    }

    #[test]
    fn unusable_captions_are_caught() {
        assert!(quality_error("A warm lo-fi beat with dusty drums and a mellow Rhodes.", "caption").is_none());
        assert!(quality_error("  ", "caption").unwrap().contains("empty"));
        assert!(quality_error("<|audio_code_12|><|audio_code_4|>", "caption").is_some());
        assert!(quality_error(&"a".repeat(30), "caption").unwrap().contains("repeated characters"));
        assert!(quality_error(&format!("great track{}", "!".repeat(25)), "caption").unwrap().contains("punctuation"));
        assert!(quality_error("A song. bpm: 120, keyscale: C major", "caption").unwrap().contains("embedded"));
        assert!(quality_error("12 34 56 78 90 12 34 56 78 90 12 34 56 78 90 12 34 56 78 90", "caption").is_some());
    }

    #[test]
    fn a_caption_result_becomes_a_carey_sidecar() {
        let options = Options {
            dataset: PathBuf::new(),
            lm: String::new(),
            dit: String::new(),
            vae: String::new(),
            overwrite: false,
            bpm_analysis: true,
            key_analysis: true,
            trigger: " my-tag ".into(),
        };
        let result = json!({
            "caption": " Dusty boom bap with a mellow Rhodes. ", "genre": "", "is_instrumental": null,
            "lyrics": "[verse]\n[chorus]", "bpm": 120, "keyscale": "C major", "language": "en", "time_signature": "3",
            "local_analysis": {"bpm": 92, "bpm_confidence": 1.8, "keyscale": "F minor", "key_confidence": 0.3},
        });
        let fields = sidecar_fields(Path::new("song.wav"), &result, &options);
        assert_eq!(fields.caption, "Dusty boom bap with a mellow Rhodes.");
        assert_eq!((fields.bpm.as_str(), fields.bpm_source.as_str()), ("92", "local_overrode_lm"));
        assert_eq!((fields.keyscale.as_str(), fields.key_source.as_str()), ("F minor", "local_overrode_lm"));
        assert!(fields.is_instrumental, "section markers alone are an instrumental");
        assert_eq!(fields.language, "unknown");
        assert_eq!(fields.custom_tag, "my-tag");
        assert_eq!(fields.timesignature, "3");
        assert!(fields.lyrics.is_empty());
        let staged = crate::carey_training::metadata(&fields, "").unwrap();
        assert!(staged.contains("bpm: 92\nkey: F minor\nsignature: 3\n") && staged.ends_with("lyrics: [Instrumental]\n"), "{staged}");
        let vocal = json!({"caption": "A sad piano ballad with soft vocals.", "lyrics": "I walked alone", "bpm": 70, "language": "en"});
        let without_analysis = Options { bpm_analysis: false, key_analysis: false, ..options };
        let fields = sidecar_fields(Path::new("ballad_bpm68.wav"), &vocal, &without_analysis);
        assert_eq!((fields.bpm.as_str(), fields.bpm_source.as_str()), ("68", "filename"));
        assert!(!fields.is_instrumental && fields.lyrics.is_empty(), "vocal lyrics are the user's to write");
    }

    #[test]
    fn a_caption_result_becomes_an_sa3_sidecar_as_the_python_helper_wrote_it() {
        assert_eq!(sa3_genre(&json!({"genre": " lo-fi hip hop "})).unwrap(), "lo-fi hip hop");
        for unusable in [json!({}), json!({"genre": "N/A"}), json!({"genre": "unknown"}), json!({"genre": "<|audio_code_12|>"})] {
            assert!(sa3_genre(&unusable).is_err(), "{unusable}");
        }
        assert_eq!(sa3_sidecar(false, "techno", Some(145), "C minor"), "techno, 145 bpm, C minor");
        assert_eq!(sa3_sidecar(false, "techno", None, ""), "techno");
        assert_eq!(
            sa3_sidecar(true, "techno", Some(145), "C minor"),
            "TrackType: Music, VocalType: Instrumental, Genre: techno, BPM: 145, Key: C minor"
        );
        assert_eq!(sa3_sidecar(true, "techno", None, ""), "TrackType: Music, VocalType: Instrumental, Genre: techno");
    }

    /// GARY4LOCAL_CAREY_CAPTION_SMOKE_ROOT (a new folder), _MODELS (a folder
    /// with the 2B base, the shared GGUFs and an LM: _LM names it, the 0.6B
    /// Q8_0 by default) and _SERVER (an ace-server.exe; _BACKEND, cuda by
    /// default, names its build). Renders two songs, captions them as Carey
    /// would, stages the sidecars it wrote for training, then auto-labels the
    /// same songs as SA3 would.
    #[tokio::test]
    #[ignore = "captions real renders with a real ace-server and LM"]
    async fn real_native_captioning_writes_trainable_sidecars() {
        let var = |name: &str| std::env::var(format!("GARY4LOCAL_CAREY_CAPTION_SMOKE_{name}")).ok();
        let root = PathBuf::from(var("ROOT").expect("a new isolated root is required"));
        assert!(!root.exists(), "the smoke root must not exist yet");
        let models = PathBuf::from(var("MODELS").expect("a model folder is required"));
        let server_exe = PathBuf::from(var("SERVER").expect("an ace-server is required"));
        std::fs::create_dir_all(root.join("dataset/nested")).unwrap();
        let installed = crate::native_runtime::NativeInstall {
            dir: server_exe.parent().unwrap().into(),
            backend: var("BACKEND").unwrap_or_else(|| "cuda".into()),
            runtimes: Vec::new(),
            version: None,
            fallback_reason: None,
        };
        let launch = crate::native_runtime::NativeLaunch::new(
            &root,
            "ace-server.exe",
            &installed,
            Vec::new(),
            Vec::new(),
            &[("ACE_MODELS_DIR".into(), models.to_string_lossy().into())],
        );
        let server = PrivateServer::start(&launch, &root, "Carey captioner", &[]).await.unwrap();
        let dit = "acestep-v15-base-BF16.gguf";
        for (path, caption) in [
            (root.join("dataset/groove_bpm96.wav"), "funky drum groove with slap bass and clavinet"),
            (root.join("dataset/nested/pad.wav"), "slow ambient synth pad with soft piano"),
        ] {
            let request = json!({"caption": caption, "lyrics": "[Instrumental]", "duration": 20, "bpm": 96,
                "seed": 7, "inference_steps": 30, "synth_model": dit, "output_format": "wav16"});
            let audio = ace_client::synth_audio(&server.client, &server.base, Body::Json(&request),
                Duration::from_secs(600), "render").await.unwrap();
            std::fs::write(path, audio).unwrap();
        }
        let options = Options {
            dataset: root.join("dataset"),
            lm: var("LM").unwrap_or_else(|| "acestep-5Hz-lm-0.6B-Q8_0.gguf".into()),
            dit: dit.into(),
            vae: "vae-BF16.gguf".into(),
            overwrite: false,
            bpm_analysis: true,
            key_analysis: true,
            trigger: "smoke".into(),
        };
        let (captioned, total) = caption_dataset(&server, &options, &root, |progress| {
            println!("[{}/{}] {}", progress.index, progress.total, progress.name)
        }, || false)
        .await
        .unwrap();
        assert_eq!((captioned, total), (2, 2));
        for sidecar in ["dataset/groove_bpm96.txt", "dataset/nested/pad.txt"] {
            let raw = std::fs::read_to_string(root.join(sidecar)).unwrap();
            println!("--- {sidecar}\n{raw}");
            let fields = crate::parse_carey_ace_sidecar_content(&raw);
            assert!(quality_error(&fields.caption, "caption").is_none());
            assert_eq!(fields.custom_tag, "smoke");
            if fields.is_instrumental {
                crate::carey_training::metadata(&fields, "").unwrap();
            }
        }
        let groove = crate::parse_carey_ace_sidecar_content(&std::fs::read_to_string(root.join("dataset/groove_bpm96.txt")).unwrap());
        assert_eq!((groove.bpm.as_str(), groove.bpm_source.as_str()), ("96", "filename"));

        // SA3's auto-label overwrites the same sidecars with one line each.
        let files = [root.join("dataset/groove_bpm96.wav"), root.join("dataset/nested/pad.wav")];
        label_sa3_dataset(&server, &Options { overwrite: true, trigger: String::new(), ..options }, &files, false, &root, |_| {}, || false)
            .await
            .unwrap();
        server.stop().await.unwrap();
        for sidecar in ["dataset/groove_bpm96.txt", "dataset/nested/pad.txt"] {
            let raw = std::fs::read_to_string(root.join(sidecar)).unwrap();
            println!("--- SA3 {sidecar}: {raw}");
            assert_eq!(raw.lines().count(), 1, "{raw}");
        }
        let groove = std::fs::read_to_string(root.join("dataset/groove_bpm96.txt")).unwrap();
        assert!(groove.contains(", 96 bpm"), "the filename BPM wins: {groove}");
    }
}
