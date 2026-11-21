//! Shared configuration and progress for the bundled native trainers.
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};

pub fn sanitize_lora_name(raw: &str) -> Option<String> {
    let normalized = raw.trim().to_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
    {
        Some(normalized)
    } else {
        None
    }
}

pub fn resolve_sa3_lora_layer_scope(
    layer_scope: &str,
) -> Result<(Option<&'static str>, Option<&'static str>), String> {
    match layer_scope.trim() {
        // Seven adapters per block: self-attention (2), cross-attention (3),
        // and feed-forward (2). 24 medium-DiT blocks * 7 = 168.
        "transformer-core" => Ok((Some("transformer.layers"), Some("to_local_embed"))),
        "full" => Ok((None, None)),
        "full-no-seconds" => Ok((None, Some("seconds_total"))),
        _ => Err(format!("Unknown SA3 LoRA layer scope: {layer_scope}")),
    }
}

pub mod sa3 {
    use super::*;

    // The desktop full-track option uses the native 3072-token window.
    pub(crate) const FULL_TRACK_SECONDS: f64 = 285.35;
    pub(crate) const DURATION_SECONDS: std::ops::RangeInclusive<f64> = 0.1..=380.0;

    #[derive(Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Options {
        pub name: String,
        pub dataset: PathBuf,
        pub fixed_prompt: String,
        pub steps: u32,
        pub rank: u32,
        pub batch_size: u32,
        pub checkpoint_every: u32,
        pub duration: f64,
        pub learning_rate: f64,
        pub target_latent_rms: f64,
        pub layer_scope: String,
        pub encoding: String,
        pub resume: Option<PathBuf>,
        #[serde(default)]
        pub prompt_config: Option<PathBuf>,
    }

    #[derive(Debug, Deserialize)]
    pub struct Progress {
        pub schema_version: u32,
        pub status: String,
        pub phase: String,
        pub message: String,
        pub error: String,
        pub step: u32,
        /// Written by ace-train, which counts epochs; sa3-train counts steps only.
        #[serde(default)]
        pub epoch: u32,
        pub max_steps: u32,
        pub final_adapter: String,
        pub adapter_checkpoint: String,
        #[serde(default)]
        pub state_checkpoint: String,
    }

    pub fn read_progress(path: &Path) -> Result<Progress, String> {
        let data = std::fs::read(path).map_err(|error| error.to_string())?;
        let progress: Progress =
            serde_json::from_slice(&data).map_err(|error| error.to_string())?;
        if progress.schema_version != 1 {
            return Err("Unsupported native trainer progress schema".into());
        }
        Ok(progress)
    }

    pub fn validate(options: &Options) -> Result<(), String> {
        if ["none", "default", "defaults"].contains(&options.name.as_str())
            || sanitize_lora_name(&options.name).as_deref() != Some(options.name.as_str())
            || !options.dataset.is_dir()
            || options.steps == 0
            || options.steps > i32::MAX as u32
            || !(1..=256).contains(&options.rank)
            || !(1..=64).contains(&options.batch_size)
            || options.checkpoint_every == 0
            || options.checkpoint_every > i32::MAX as u32
            || !DURATION_SECONDS.contains(&options.duration)
            || !options.learning_rate.is_finite()
            || options.learning_rate <= 0.0
            || options.learning_rate > 1.0
            || !options.target_latent_rms.is_finite()
            || (options.target_latent_rms != 0.0
                && !(0.5..=1.3).contains(&options.target_latent_rms))
            || !["F16", "Q4_K_M"].contains(&options.encoding.as_str())
        {
            return Err("Invalid native SA3 training settings or dataset folder".into());
        }
        resolve_sa3_lora_layer_scope(&options.layer_scope)?;
        Ok(())
    }

    pub fn caption_prompt_config(trigger: &str) -> serde_json::Value {
        let trigger = trigger.trim();
        json!({"prompt_config": {
            "use_tags": true, "use_paths": false, "use_fixed": false,
            "fixed_text": "", "balance": {"tags": 40}, "tag_keys": ["prompt"],
            "hide_tag_names": true, "shuffle": false,
            "trigger": trigger, "trigger_pct": if trigger.is_empty() { 0 } else { 100 }
        }})
    }

    pub fn trainer_args(
        command: &mut tokio::process::Command,
        options: &Options,
        backend: &str,
        models: &Path,
        output: &Path,
        cache: &Path,
        progress_path: &Path,
        cancel_path: &Path,
    ) -> Result<(), String> {
        command
            .args([
                "--model",
                "medium",
                "--device",
                backend,
                "--encoding",
                &options.encoding,
                "--t5-encoding",
                "f16",
                "--ae-encoding",
                "f32",
            ])
            .arg("--dataset")
            .arg(&options.dataset)
            .arg("--models-dir")
            .arg(models)
            .arg("--out")
            .arg(output)
            .arg("--steps")
            .arg(options.steps.to_string())
            .arg("--rank")
            .arg(options.rank.to_string())
            .args(["--adapter-type", "dora-rows", "--alpha", "0"])
            .arg("--batch-size")
            .arg(options.batch_size.to_string())
            .arg("--checkpoint-every")
            .arg(options.checkpoint_every.to_string())
            .arg("--duration")
            .arg(options.duration.to_string())
            .arg("--lr")
            .arg(options.learning_rate.to_string())
            .arg("--target-latent-rms")
            .arg(options.target_latent_rms.to_string())
            .arg("--latents-cache-dir")
            .arg(cache)
            .arg("--progress-file")
            .arg(progress_path)
            .arg("--cancel-file")
            .arg(cancel_path);
        let text =
            crate::model_catalog::component(crate::service::Service::Sa3, "sa3-native::text")
                .unwrap();
        let decoder = crate::model_catalog::component(
            crate::service::Service::Sa3,
            "sa3-native::medium-decoder",
        )
        .unwrap();
        let base = crate::model_catalog::component(
            crate::service::Service::Sa3,
            &format!("sa3-native::medium-base-{}", options.encoding),
        )
        .unwrap();
        for (flag, component, needle) in [
            ("--tok", text, "vocab"),
            ("--t5", text, "encoder"),
            ("--same", decoder, "same-l"),
            ("--cond", decoder, "conditioner"),
            ("--dit", base, "dit"),
        ] {
            let file = component
                .files
                .iter()
                .find(|file| file.filename.contains(needle))
                .ok_or("Invalid native training model catalog")?;
            command.arg(flag).arg(models.join(&file.filename));
        }
        match options.layer_scope.as_str() {
            "transformer-core" => {
                command.args(["--lora-scope", "core"]);
            }
            "full" => {
                command.args(["--lora-scope", "full"]);
            }
            "full-no-seconds" => {
                command.args(["--lora-scope", "full", "--lora-exclude", "seconds_total"]);
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

pub mod carey {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Options {
        pub name: String,
        pub dataset: PathBuf,
        /// The base DiT to train on, as its catalog component.
        pub dit: String,
        /// "lora" or "dora" as the dialog names them.
        pub adapter_type: String,
        pub profile: String,
        pub rank: u32,
        pub epochs: u32,
        pub checkpoint_every: u32,
        pub batch_size: u32,
        pub accumulation: u32,
        pub learning_rate: f64,
        pub cfg_dropout: f64,
        pub timestep_mu: f64,
        pub min_snr: Option<f64>,
        pub genre_ratio: u32,
        pub trigger: String,
        pub window_min_seconds: f64,
        pub window_max_seconds: f64,
        /// The earlier checkpoint this run continues from, if any.
        #[serde(default)]
        pub resume: Option<Resume>,
    }

    /// How a run continues from an earlier run's checkpoint.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub enum ResumeMode {
        /// From its full trainer state: the original settings and epoch budget,
        /// continuing at the update where it stopped. ace-train refuses anything
        /// else, since its cosine schedule spans the whole budget.
        Exact,
        /// From its weights alone, under new settings: the optimizer, warmup and
        /// schedule start over.
        Weights,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Resume {
        /// An earlier run's checkpoint folder: its `run`, or a `checkpoint-epoch-N`.
        pub checkpoint: PathBuf,
        pub mode: ResumeMode,
    }

    pub const TRAINABLE_DITS: &[&str] = &["base", "xl-base"];

    pub fn validate(options: &Options) -> Result<(), String> {
        let trainable = TRAINABLE_DITS.iter().any(|variant| {
            crate::model_catalog::PRECISIONS
                .iter()
                .any(|tier| format!("carey-native::{variant}::{tier}") == options.dit)
        });
        if ["none", "default", "defaults"].contains(&options.name.as_str())
            || sanitize_lora_name(&options.name).as_deref() != Some(options.name.as_str())
        {
            return Err("LoRA name must use lowercase letters, numbers, '-' or '_'".into());
        }
        if !trainable {
            return Err("Train on the 2B or XL base model.".into());
        }
        if !["lora", "dora"].contains(&options.adapter_type.as_str())
            || !["attention", "balanced"].contains(&options.profile.as_str())
        {
            return Err("Unknown adapter type or module profile.".into());
        }
        if !(1..=256).contains(&options.rank)
            || !(1..=100_000).contains(&options.epochs)
            || options.checkpoint_every == 0
            || options.batch_size != 1
            || !(1..=256).contains(&options.accumulation)
            || options.genre_ratio > 100
        {
            return Err(
            "Invalid training settings. Native training uses batch size 1: raise gradient accumulation instead.".into(),
        );
        }
        if !options.learning_rate.is_finite()
            || options.learning_rate <= 0.0
            || options.learning_rate > 1.0
        {
            return Err("learning rate must be greater than 0".into());
        }
        if !options.cfg_dropout.is_finite() || !(0.0..1.0).contains(&options.cfg_dropout) {
            return Err("CFG ratio must be in [0, 1).".into());
        }
        if !options.timestep_mu.is_finite()
            || options
                .min_snr
                .is_some_and(|gamma| !gamma.is_finite() || gamma <= 0.0)
        {
            return Err("timestep mu and SNR gamma must be finite, and gamma positive.".into());
        }
        if !options.window_min_seconds.is_finite()
            || !options.window_max_seconds.is_finite()
            || options.window_min_seconds < 1.0
            || options.window_max_seconds < options.window_min_seconds
            || options.window_max_seconds > 600.0
        {
            return Err("Training windows must be between 1 and 600 seconds, the minimum no longer than the maximum.".into());
        }
        Ok(())
    }

    pub fn trainer_args(options: &Options) -> Result<Vec<String>, String> {
        let shared =
            crate::model_catalog::component(crate::service::Service::Carey, "carey-native::shared")
                .ok_or("Carey's catalog has no shared models")?;
        let dit = crate::model_catalog::component(crate::service::Service::Carey, &options.dit)
            .ok_or("Unknown training model")?;
        let mut args: Vec<String> = vec![
            "--dit".into(),
            dit.files[0].filename.clone(),
            "--text-encoder".into(),
            shared.files[0].filename.clone(),
            "--vae".into(),
            shared.files[1].filename.clone(),
            "--adapter-type".into(),
            if options.adapter_type == "dora" {
                "dora-rows"
            } else {
                "lora"
            }
            .into(),
            "--profile".into(),
            options.profile.clone(),
            "--rank".into(),
            options.rank.to_string(),
            "--alpha".into(),
            (options.rank * 2).to_string(),
            "--epochs".into(),
            options.epochs.to_string(),
            "--checkpoint-every".into(),
            options.checkpoint_every.to_string(),
            "--batch-size".into(),
            options.batch_size.to_string(),
            "--accumulation".into(),
            options.accumulation.to_string(),
            "--learning-rate".into(),
            options.learning_rate.to_string(),
            // The validated 8 GB profile: 100 warmup updates into a cosine decay.
            "--warmup-steps".into(),
            "100".into(),
            "--schedule".into(),
            "cosine".into(),
            "--cfg-dropout".into(),
            options.cfg_dropout.to_string(),
            "--timestep-mu".into(),
            options.timestep_mu.to_string(),
            "--genre-ratio".into(),
            options.genre_ratio.to_string(),
            "--tag-position".into(),
            "prepend".into(),
            "--window-min-seconds".into(),
            options.window_min_seconds.to_string(),
            "--window-max-seconds".into(),
            options.window_max_seconds.to_string(),
        ];
        if let Some(gamma) = options.min_snr {
            args.extend(["--min-snr".into(), gamma.to_string()]);
        }
        Ok(args)
    }

    pub fn resolve_checkpoint(dir: &Path) -> Result<PathBuf, String> {
        let pointer = dir.join("checkpoint.current");
        let target = match std::fs::read_link(&pointer) {
            Ok(target) => target,
            Err(_) => PathBuf::from(
                std::fs::read_to_string(&pointer)
                    .map_err(|error| {
                        format!("{} has no checkpoint pointer: {error}", dir.display())
                    })?
                    .trim(),
            ),
        };
        let mut parts = target.components();
        let valid = !target.is_absolute()
            && parts
                .next()
                .is_some_and(|part| part.as_os_str() == ".checkpoint-generations")
            && parts.next().is_some_and(|part| {
                part.as_os_str()
                    .to_string_lossy()
                    .starts_with("generation-")
            })
            && parts.next().is_none();
        if !valid {
            return Err(format!(
                "{} points outside its checkpoint generations",
                dir.display()
            ));
        }
        let generation = dir.join(target);
        if !generation.join("adapter_model.safetensors").is_file()
            || !generation.join("adapter_config.json").is_file()
        {
            return Err(format!(
                "{} is not a complete adapter",
                generation.display()
            ));
        }
        Ok(generation)
    }
}
