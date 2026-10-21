//! Native Carey LoRA training through acestep.cpp's ace-train. Carey's
//! datasets are nested folders of mixed audio with Carey sidecars; ace-train
//! reads a flat folder of WAVs paired with its own metadata files, so each job
//! stages one. The trainer owns progress, cancellation at an optimizer-update
//! boundary and its checkpoint generations; Gary owns the job record, the LoRA
//! catalog and the adapters folder ace-server loads.

use crate::carey_models;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

pub static LAUNCH: Mutex<()> = Mutex::const_new(());
static NEXT_JOB: AtomicU64 = AtomicU64::new(0);

/// What the training dialog chose, recorded beside each job.
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

/// A checkpoint a new run can continue from.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumePoint {
    pub name: String,
    pub job_id: String,
    pub path: String,
    /// Epochs the checkpoint had completed, out of its run's budget.
    pub epoch: u32,
    pub epochs: u32,
    /// Saved part way through an epoch, as cancelling does.
    pub mid_epoch: bool,
    /// Whether it holds the trainer state an exact resume needs, with epochs
    /// left to train.
    pub exact: bool,
}

/// The DiTs a LoRA can train on: lego, extract and complete need a base model,
/// and turbo and sft adapters are not what Carey trains.
pub const TRAINABLE_DITS: &[&str] = &["base", "xl-base"];
/// GPU memory a training run needs beside its frozen DiT's weights: adapter
/// weights and AdamW moments, rotating-window activations and preprocessing.
/// XL base Q4_K_M with 45-60 s windows peaked at 7.1 GB on an 8151 MiB RTX
/// 5070 while preprocessing, 4.0 GB once warm.
pub const TRAINING_HEADROOM_BYTES: u64 = 5 << 30;

/// The best precision of `variant` to train on a GPU with `memory_bytes`.
pub fn recommended_training_tier(variant: &str, memory_bytes: u64) -> &'static str {
    carey_models::TIERS
        .iter()
        .copied()
        .find(|tier| {
            carey_models::models()
                .component(&carey_models::dit_component(variant, tier))
                .is_some_and(|component| {
                    component.files.iter().map(|file| file.bytes).sum::<u64>() + TRAINING_HEADROOM_BYTES
                        <= memory_bytes
                })
        })
        .unwrap_or(carey_models::TIERS[carey_models::TIERS.len() - 1])
}

fn validate(options: &Options) -> Result<(), String> {
    let trainable = TRAINABLE_DITS.iter().any(|variant| {
        carey_models::TIERS
            .iter()
            .any(|tier| carey_models::dit_component(variant, tier) == options.dit)
    });
    if ["none", "default", "defaults"].contains(&options.name.as_str())
        || crate::sanitize_lora_name(&options.name).as_deref() != Some(options.name.as_str())
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
    if !options.learning_rate.is_finite() || options.learning_rate <= 0.0 || options.learning_rate > 1.0 {
        return Err("learning rate must be greater than 0".into());
    }
    if !options.cfg_dropout.is_finite() || !(0.0..1.0).contains(&options.cfg_dropout) {
        return Err("CFG ratio must be in [0, 1).".into());
    }
    if !options.timestep_mu.is_finite() || options.min_snr.is_some_and(|gamma| !gamma.is_finite() || gamma <= 0.0) {
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

/// One sidecar's value: trimmed, on one line, as ace-train's parser reads it.
fn scalar(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A beats-per-bar numerator from "4", "4/4" or "6/8"; ACE-Step's own default
/// of four when the sidecar has none.
fn signature(value: &str) -> Result<u32, String> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(4);
    }
    value
        .split('/')
        .next()
        .and_then(|numerator| numerator.trim().parse::<u32>().ok())
        .filter(|beats| (1..=32).contains(beats))
        .ok_or_else(|| format!("time signature \"{value}\" is not like 4/4"))
}

fn bpm(value: &str) -> Result<u32, String> {
    let value = value.trim();
    value
        .parse::<f64>()
        .ok()
        .filter(|bpm| bpm.is_finite() && *bpm >= 1.0 && *bpm <= 1000.0)
        .map(|bpm| bpm.round() as u32)
        .ok_or_else(|| if value.is_empty() { "no BPM".into() } else { format!("BPM \"{value}\" is not a number") })
}

/// One Carey sidecar as ace-train metadata. ace-train requires every field
/// to be present, lyrics last; Carey writes `keyscale` and `timesignature` and
/// omits empty fields. The trigger, when set, is every example's custom tag,
/// as the Python job writes it.
pub fn metadata(fields: &crate::CareyAceSidecarFields, trigger: &str) -> Result<String, String> {
    let caption = scalar(&fields.caption);
    if caption.is_empty() {
        return Err("no caption".into());
    }
    let bpm = bpm(&fields.bpm)?;
    let beats = signature(&fields.timesignature)?;
    // Empty lyrics on a vocal song train as empty lyric conditioning, as in
    // PyTorch Carey; only an instrumental gets the [Instrumental] marker.
    let mut lyrics = fields.lyrics.trim().replace("\r\n", "\n");
    if lyrics.is_empty() && fields.is_instrumental {
        lyrics = "[Instrumental]".into();
    }
    let tag = if trigger.trim().is_empty() { scalar(&fields.custom_tag) } else { scalar(trigger) };
    Ok(format!(
        "caption: {caption}\ngenre: {}\nbpm: {bpm}\nkey: {}\nsignature: {beats}\nis_instrumental: {}\ncustom_tag: {tag}\nlyrics: {lyrics}\n",
        scalar(&fields.genre),
        scalar(&fields.keyscale),
        fields.is_instrumental,
    ))
}

/// Whether ace-train reads this WAV as it is: 16-bit PCM or 32-bit float.
fn readable_wav(path: &Path) -> bool {
    hound::WavReader::open(path).is_ok_and(|reader| {
        let spec = reader.spec();
        matches!(
            (spec.sample_format, spec.bits_per_sample),
            (hound::SampleFormat::Int, 16) | (hound::SampleFormat::Float, 32)
        )
    })
}

async fn stage_audio(source: &Path, dest: &Path) -> Result<(), String> {
    let wav = source.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
    if wav && readable_wav(source) {
        // Training reads its inputs without changing them, so a link is enough.
        if std::fs::hard_link(source, dest).is_ok() {
            return Ok(());
        }
        return std::fs::copy(source, dest).map(|_| ()).map_err(|error| error.to_string());
    }
    let mut command = tokio::process::Command::new("ffmpeg");
    crate::hide_console_window(&mut command);
    command
        .args(["-nostdin", "-v", "error", "-y", "-i"])
        .arg(source)
        .args(["-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "wav"])
        .arg(dest)
        .kill_on_drop(true);
    let output = command
        .output()
        .await
        .map_err(|error| format!("FFmpeg is needed to train on {}: {error}", source.display()))?;
    if !output.status.success() {
        return Err(format!(
            "Could not decode {}: {}",
            source.display(),
            String::from_utf8_lossy(&output.stderr).chars().take(500).collect::<String>()
        ));
    }
    Ok(())
}

/// What staging produced: the flat dataset ace-train reads and the captions
/// Carey's caption pool takes from it.
pub struct Staged {
    pub count: usize,
    pub captions: Vec<String>,
}

/// "no BPM: a.wav, b.wav and 3 more; no caption: c.wav"
fn problem_summary(problems: &[(String, Vec<String>)]) -> String {
    problems
        .iter()
        .map(|(kind, songs)| {
            let shown = songs.iter().take(2).cloned().collect::<Vec<_>>().join(", ");
            match songs.len() {
                0..=2 => format!("{kind}: {shown}"),
                more => format!("{kind}: {shown} and {} more", more - 2),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Stage `dataset` into `dest` as numbered WAV and metadata pairs. Every
/// sidecar problem is reported at once, before any audio is decoded.
pub async fn stage(dataset: &Path, dest: &Path, trigger: &str) -> Result<Staged, String> {
    let entries = crate::read_carey_ace_dataset_sidecars(&dataset.to_string_lossy())?;
    if entries.is_empty() {
        return Err(format!("{} has no audio files to train on.", dataset.display()));
    }
    // Problems grouped by kind, each naming a few songs: one short line.
    let mut problems: Vec<(String, Vec<String>)> = Vec::new();
    let mut staged = Vec::new();
    for entry in &entries {
        match metadata(&entry.fields, trigger) {
            Ok(text) => staged.push((PathBuf::from(&entry.audio_path), text, scalar(&entry.fields.caption))),
            Err(problem) => match problems.iter_mut().find(|(kind, _)| *kind == problem) {
                Some((_, songs)) => songs.push(entry.relative_path.clone()),
                None => problems.push((problem, vec![entry.relative_path.clone()])),
            },
        }
    }
    if !problems.is_empty() {
        return Err(format!("Fix these sidecars in the editor first: {}.", problem_summary(&problems)));
    }
    std::fs::create_dir_all(dest).map_err(|error| error.to_string())?;
    let mut manifest = Vec::new();
    let mut captions = Vec::new();
    for (index, (audio, text, caption)) in staged.iter().enumerate() {
        let stem = format!("{index:04}");
        stage_audio(audio, &dest.join(format!("{stem}.wav"))).await?;
        std::fs::write(dest.join(format!("{stem}.txt")), text).map_err(|error| error.to_string())?;
        manifest.push(json!({"staged": stem, "source": audio}));
        if !captions.contains(caption) {
            captions.push(caption.clone());
        }
    }
    crate::sa3_training::save(&dest.join("staging.json"), &manifest)?;
    Ok(Staged { count: staged.len(), captions })
}

/// The generation folder a native checkpoint's `checkpoint.current` names:
/// a text file or a link, relative and inside `.checkpoint-generations`.
pub fn resolve_checkpoint(dir: &Path) -> Result<PathBuf, String> {
    let pointer = dir.join("checkpoint.current");
    let target = match std::fs::read_link(&pointer) {
        Ok(target) => target,
        Err(_) => PathBuf::from(
            std::fs::read_to_string(&pointer)
                .map_err(|error| format!("{} has no checkpoint pointer: {error}", dir.display()))?
                .trim(),
        ),
    };
    let mut parts = target.components();
    let valid = !target.is_absolute()
        && parts.next().is_some_and(|part| part.as_os_str() == ".checkpoint-generations")
        && parts.next().is_some_and(|part| part.as_os_str().to_string_lossy().starts_with("generation-"))
        && parts.next().is_none();
    if !valid {
        return Err(format!("{} points outside its checkpoint generations", dir.display()));
    }
    let generation = dir.join(target);
    if !generation.join("adapter_model.safetensors").is_file() || !generation.join("adapter_config.json").is_file() {
        return Err(format!("{} is not a complete adapter", generation.display()));
    }
    Ok(generation)
}

/// Every adapter a run saved, oldest first: periodic epochs, then its root
/// (the final epoch, or where a cancelled run stopped).
pub fn run_checkpoints(run: &Path, final_label: &str) -> Vec<crate::CareyTrainingCheckpoint> {
    let mut epochs: Vec<(u32, PathBuf)> = std::fs::read_dir(run)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let epoch = entry.file_name().to_str()?.strip_prefix("checkpoint-epoch-")?.parse().ok()?;
            Some((epoch, entry.path()))
        })
        .collect();
    epochs.sort_by_key(|(epoch, _)| *epoch);
    let mut checkpoints: Vec<_> = epochs
        .into_iter()
        .filter_map(|(epoch, dir)| {
            Some(crate::CareyTrainingCheckpoint {
                id: format!("epoch-{epoch}"),
                label: format!("epoch {epoch}"),
                epoch: Some(epoch),
                path: resolve_checkpoint(&dir).ok()?.to_string_lossy().into_owned(),
            })
        })
        .collect();
    if let Ok(path) = resolve_checkpoint(run) {
        checkpoints.push(crate::CareyTrainingCheckpoint {
            id: "final".into(),
            label: final_label.into(),
            epoch: None,
            path: path.to_string_lossy().into_owned(),
        });
    }
    checkpoints
}

/// The checkpoint at `dir`, one of `options`' run's saves, as a resume point.
fn resume_point(dir: &Path, options: &Options, job_id: &str) -> Option<ResumePoint> {
    let generation = resolve_checkpoint(dir).ok()?;
    let state: serde_json::Value = std::fs::read(generation.join("trainer_state.json"))
        .ok()
        .and_then(|raw| serde_json::from_slice(&raw).ok())?;
    let epoch = u32::try_from(state["completed_epochs"].as_u64()?).ok()?;
    let mid_epoch = state["continuation"]["next_batch"].as_u64().unwrap_or(0) > 0;
    let full = generation.join("checkpoint_manifest.json").is_file()
        && generation.join("optimizer_state.safetensors").is_file();
    Some(ResumePoint {
        name: options.name.clone(),
        job_id: job_id.into(),
        path: dir.to_string_lossy().into_owned(),
        epoch,
        epochs: options.epochs,
        mid_epoch,
        exact: full && epoch < options.epochs,
    })
}

/// Every checkpoint of every C++ training run, cancelled and failed ones
/// included, with the options its run recorded.
fn resume_sources(root: &Path) -> Vec<(ResumePoint, Options, PathBuf)> {
    let jobs = root.join("carey").join("training").join("jobs");
    let mut sources = Vec::new();
    for entry in std::fs::read_dir(jobs).into_iter().flatten().flatten() {
        let job = entry.path();
        let Some(job_id) = entry.file_name().to_str().filter(|id| is_native_job(Some(id))).map(String::from) else {
            continue;
        };
        let Some(options) = std::fs::read(job.join("native-options.json"))
            .ok()
            .and_then(|raw| serde_json::from_slice::<Options>(&raw).ok())
        else {
            continue;
        };
        let run = job.join("run");
        let mut dirs: Vec<(u32, PathBuf)> = std::fs::read_dir(&run)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let epoch = entry.file_name().to_str()?.strip_prefix("checkpoint-epoch-")?.parse().ok()?;
                Some((epoch, entry.path()))
            })
            .collect();
        dirs.sort_by_key(|(epoch, _)| *epoch);
        dirs.push((u32::MAX, run));
        for (_, dir) in dirs {
            if let Some(point) = resume_point(&dir, &options, &job_id) {
                sources.push((point, options.clone(), job.clone()));
            }
        }
    }
    sources.sort_by(|(a, ..), (b, ..)| {
        (&a.name, &a.job_id, a.epoch, a.mid_epoch).cmp(&(&b.name, &b.job_id, b.epoch, b.mid_epoch))
    });
    sources
}

/// The checkpoints a new run can continue from, oldest first per LoRA.
pub fn resume_points(root: &Path) -> Vec<ResumePoint> {
    resume_sources(root).into_iter().map(|(point, ..)| point).collect()
}

/// The resume point at `checkpoint`, its run's options and its job folder.
fn find_resume_source(root: &Path, checkpoint: &Path) -> Result<(ResumePoint, Options, PathBuf), String> {
    let requested = checkpoint.canonicalize().map_err(|error| error.to_string())?;
    resume_sources(root)
        .into_iter()
        .find(|(point, ..)| Path::new(&point.path).canonicalize().is_ok_and(|path| path == requested))
        .ok_or_else(|| "Choose a checkpoint from Carey's C++ training history.".into())
}

/// The options the run that saved `checkpoint` trained with, for the dialog
/// to restore.
pub fn resume_options(root: &Path, checkpoint: &Path) -> Result<Options, String> {
    find_resume_source(root, checkpoint).map(|(_, options, _)| Options { resume: None, ..options })
}

/// Hard-link `source` to `dest`, or copy it where the volume cannot link.
fn link_or_copy(source: &Path, dest: &Path) -> Result<(), String> {
    if std::fs::hard_link(source, dest).is_err() {
        std::fs::copy(source, dest).map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Give a new job an earlier job's staged dataset, which an exact resume needs
/// unchanged: ace-train checks every file's contents against its plan.
fn link_dataset(source: &Path, dest: &Path) -> Result<Staged, String> {
    std::fs::create_dir_all(dest).map_err(|error| error.to_string())?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(source)
        .map_err(|error| format!("The earlier run's staged dataset is gone: {error}"))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    files.sort();
    let (mut count, mut captions) = (0, Vec::new());
    for file in &files {
        link_or_copy(file, &dest.join(file.file_name().ok_or("Invalid staged file")?))?;
        match file.extension().and_then(|extension| extension.to_str()) {
            Some("wav") => count += 1,
            Some("txt") => {
                let text = std::fs::read_to_string(file).map_err(|error| error.to_string())?;
                if let Some(caption) = text.lines().find_map(|line| line.strip_prefix("caption: ")) {
                    if !captions.iter().any(|known| known == caption) {
                        captions.push(caption.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    if count == 0 {
        return Err("The earlier run's staged dataset has no audio.".into());
    }
    Ok(Staged { count, captions })
}

/// Expose `generation` to ace-server as the adapter `name`: its folder under
/// the adapters folder ace-server scans, holding the generation's weights and
/// config. Links when the volume allows, since generations never change.
pub fn publish_adapter(root: &Path, name: &str, generation: &Path) -> Result<PathBuf, String> {
    let adapters = crate::sa3_training::checked_folder(root, &["carey", "adapters"])?;
    let dest = adapters.join(name);
    let stage = adapters.join(format!(".{name}.{}.tmp", NEXT_JOB.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&stage).map_err(|error| error.to_string())?;
    let result = (|| {
        for file in ["adapter_model.safetensors", "adapter_config.json"] {
            link_or_copy(&generation.join(file), &stage.join(file))?;
        }
        if dest.exists() {
            std::fs::remove_dir_all(&dest).map_err(|error| error.to_string())?;
        }
        std::fs::rename(&stage, &dest).map_err(|error| error.to_string())
    })();
    if stage.exists() {
        let _ = std::fs::remove_dir_all(&stage);
    }
    result.map(|_| dest)
}

/// Take `name` out of the adapters folder ace-server scans, if it is there.
pub fn unpublish_adapter(root: &Path, name: &str) -> Result<(), String> {
    let dest = root.join("carey").join("adapters").join(name);
    match std::fs::remove_dir_all(&dest) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("Cannot remove {}: {error}. Stop Carey first.", dest.display())),
    }
}

/// Whether a catalog entry came from the C++ trainer, whose adapters ace-server
/// loads from the adapters folder.
pub fn is_native_job(job_id: Option<&str>) -> bool {
    job_id.is_some_and(|id| id.starts_with("native-"))
}

/// The family Carey's catalog and gary4juce know an adapter's base by.
fn model_family(dit: &str) -> &'static str {
    if dit.starts_with("carey-native::xl-") { "xl" } else { "standard" }
}

/// Register a finished run's adapter as `name`: its catalog entry with every
/// checkpoint, the caption pool, and the adapters folder ace-server loads.
/// `earlier` are the checkpoints an exact resume continued, listed first.
pub fn register(
    root: &Path,
    options: &Options,
    job_id: &str,
    run: &Path,
    captions: &[String],
    cancelled: bool,
    earlier: &[crate::CareyTrainingCheckpoint],
) -> Result<PathBuf, String> {
    let final_label = if cancelled { "where training stopped" } else { "final epoch" };
    let mut checkpoints = earlier.to_vec();
    checkpoints.extend(run_checkpoints(run, final_label));
    let last = checkpoints.last().ok_or("The trainer saved no adapter to register")?;
    let (selected, selected_id) = (last.path.clone(), last.id.clone());
    let generation = PathBuf::from(&selected);
    publish_adapter(root, &options.name, &generation)?;
    let catalog_path = crate::carey_lora_catalog_path_for(root);
    let mut catalog = crate::read_carey_lora_catalog_from(&catalog_path)?;
    catalog.insert(
        options.name.clone(),
        crate::CareyLoraCatalogEntry {
            path: selected.clone(),
            captions_path: Some(options.dataset.to_string_lossy().into_owned()),
            scale: 1.0,
            backends: crate::default_lora_backends(),
            model_family: model_family(&options.dit).into(),
            training_job_id: Some(job_id.into()),
            training_checkpoints: checkpoints,
            selected_training_checkpoint: Some(selected_id),
        },
    );
    crate::save_carey_lora_catalog_to(&catalog_path, &catalog)?;
    let pools_path = root.join("carey").join("captions.json");
    let mut pools: serde_json::Map<String, serde_json::Value> = std::fs::read(&pools_path)
        .ok()
        .and_then(|raw| serde_json::from_slice(&raw).ok())
        .unwrap_or_default();
    pools.insert(options.name.clone(), json!(captions));
    crate::sa3_training::save(&pools_path, &pools)?;
    Ok(generation)
}

/// The ace-train arguments for `options`, before the job's own paths.
pub fn trainer_args(options: &Options) -> Result<Vec<String>, String> {
    let shared = carey_models::models()
        .component(carey_models::SHARED)
        .ok_or("Carey's catalog has no shared models")?;
    let dit = carey_models::models().component(&options.dit).ok_or("Unknown training model")?;
    let mut args: Vec<String> = vec![
        "--dit".into(),
        dit.files[0].filename.clone(),
        "--text-encoder".into(),
        shared.files[0].filename.clone(),
        "--vae".into(),
        shared.files[1].filename.clone(),
        "--adapter-type".into(),
        if options.adapter_type == "dora" { "dora-rows" } else { "lora" }.into(),
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

/// Whether the installed ace-train speaks the schema-1 file controls.
async fn probe(executable: &Path, runtime_path: Option<&std::ffi::OsStr>) -> Result<(), String> {
    let mut command = tokio::process::Command::new(executable);
    crate::hide_console_window(&mut command);
    command.arg("--control-info").kill_on_drop(true);
    if let Some(path) = runtime_path {
        command.env("PATH", path);
    }
    let result = tokio::time::timeout(std::time::Duration::from_secs(15), command.output())
        .await
        .map_err(|_| "Native trainer capability check timed out")?
        .map_err(|error| error.to_string())?;
    let info: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap_or_default();
    if !result.status.success()
        || info["schema_version"] != 1
        || info["cooperative_cancel_file"] != true
        || info["atomic_progress_file"] != true
    {
        return Err("The installed Carey runtime's trainer lacks progress and cancellation support. Update the runtime first.".into());
    }
    Ok(())
}

pub struct Job {
    child: tokio::process::Child,
    root: PathBuf,
    id: String,
    status_path: PathBuf,
    progress_path: PathBuf,
    cancel_path: PathBuf,
    run: PathBuf,
    options: Options,
    captions: Vec<String>,
    earlier: Vec<crate::CareyTrainingCheckpoint>,
}

/// `path` without the `\\?\` prefix canonicalizing adds. ace-train up to
/// v0.1.2 appends checkpoint folders with `/`, which a `\\?\` path rejects.
fn plain(path: PathBuf) -> PathBuf {
    PathBuf::from(crate::user_facing_windows_path(&path))
}

/// What a resume continues from, once checked against its run.
struct Continuation {
    mode: ResumeMode,
    point: ResumePoint,
    /// The job folder of the run that saved the checkpoint.
    source: PathBuf,
}

/// The options a run actually trains with. An exact resume takes its source
/// run's; continuing from weights keeps what the weights' shapes depend on.
fn continuation(root: &Path, options: Options) -> Result<(Options, Option<Continuation>), String> {
    let Some(resume) = options.resume.clone() else {
        return Ok((options, None));
    };
    let (point, original, source) = find_resume_source(root, &resume.checkpoint)?;
    if original.name != options.name {
        return Err(format!("A resumed run keeps its LoRA's name, {}.", original.name));
    }
    let options = match resume.mode {
        ResumeMode::Exact if !point.exact => {
            return Err("That checkpoint has no epochs left to resume: continue from its weights instead.".into());
        }
        ResumeMode::Exact => Options { resume: Some(resume.clone()), ..original },
        ResumeMode::Weights => {
            if (&original.dit, &original.adapter_type, &original.profile, original.rank)
                != (&options.dit, &options.adapter_type, &options.profile, options.rank)
            {
                return Err("Continuing from weights keeps the original model, adapter type, coverage and rank.".into());
            }
            options
        }
    };
    Ok((options, Some(Continuation { mode: resume.mode, point, source })))
}

/// Launch a native training job: stage the dataset, then start ace-train on
/// `backend` with the job's progress and cancel files. The job record is
/// Carey's usual status file, so the training dialog follows it unchanged.
pub async fn start(
    root: &Path,
    executable: &Path,
    backend: &str,
    runtime_path: Option<&std::ffi::OsStr>,
    options: Options,
) -> Result<Job, String> {
    let (options, continuation) = continuation(root, options)?;
    validate(&options)?;
    probe(executable, runtime_path).await?;
    let models = carey_models::models();
    let needed = [options.dit.clone(), carey_models::SHARED.to_string()];
    if let Some(missing) = models.missing(root, &needed) {
        return Err(format!("Download {missing} from Carey's models before training."));
    }
    let model_dir = plain(models.checked_models_dir(root)?);
    let jobs = plain(crate::sa3_training::checked_folder(root, &["carey", "training", "jobs"])?);
    let id = format!(
        "native-{}-{}-{}",
        options.name,
        crate::now_epoch_seconds() as u64,
        NEXT_JOB.fetch_add(1, Ordering::Relaxed)
    );
    let job = jobs.join(&id);
    std::fs::create_dir(&job).map_err(|error| error.to_string())?;
    let status_path = job.join("status.json");
    let progress_path = job.join("native-progress.json");
    let cancel_path = job.join("cancel.requested");
    let log_path = job.join("training.log");
    let run = job.join("run");
    crate::write_carey_ace_training_launch_state(
        &status_path,
        &id,
        &options.name,
        &job,
        &log_path,
        &cancel_path,
        0,
        "Staging the dataset for C++ training.",
    )?;
    crate::patch_carey_ace_training_status(
        &status_path,
        &[
            ("runtime", json!("acestep.cpp")),
            ("phase", json!("staging")),
            ("modelFamily", json!(model_family(&options.dit))),
            ("adapterType", json!(options.adapter_type)),
        ],
    )?;
    let fail = |message: String| {
        let _ = crate::mark_carey_ace_training_failed(&status_path, &message, None);
        message
    };
    let staged = match &continuation {
        Some(Continuation { mode: ResumeMode::Exact, source, .. }) => {
            link_dataset(&source.join("dataset"), &job.join("dataset")).map_err(fail)?
        }
        _ => stage(&options.dataset, &job.join("dataset"), &options.trigger).await.map_err(fail)?,
    };
    // ace-train reads full trainer state from a checkpoint folder, and only
    // weights from a folder holding nothing else.
    let resume_from = match &continuation {
        Some(Continuation { mode: ResumeMode::Exact, point, .. }) => Some(PathBuf::from(&point.path)),
        Some(Continuation { mode: ResumeMode::Weights, point, .. }) => {
            let generation = resolve_checkpoint(Path::new(&point.path)).map_err(fail)?;
            let weights = job.join("resume-weights");
            std::fs::create_dir(&weights).map_err(|error| fail(error.to_string()))?;
            for file in ["adapter_model.safetensors", "adapter_config.json"] {
                link_or_copy(&generation.join(file), &weights.join(file)).map_err(fail)?;
            }
            Some(weights)
        }
        None => None,
    };
    // An exact resume's earlier saves stay in its catalog entry.
    let earlier = match &continuation {
        Some(Continuation { mode: ResumeMode::Exact, point, source }) => run_checkpoints(&source.join("run"), "")
            .into_iter()
            .filter(|checkpoint| checkpoint.epoch.is_some_and(|epoch| epoch <= point.epoch))
            .collect(),
        _ => Vec::new(),
    };
    crate::sa3_training::save(&job.join("native-options.json"), &options).map_err(fail)?;
    // Shared across jobs: ace-train keys every entry by its inputs and settings.
    let cache = plain(crate::sa3_training::checked_folder(root, &["carey", "training", "native-cache"]).map_err(fail)?);
    let mut command = tokio::process::Command::new(executable);
    crate::hide_console_window(&mut command);
    command
        .arg("--models")
        .arg(&model_dir)
        .arg("--dataset")
        .arg(job.join("dataset"))
        .arg("--output")
        .arg(&run)
        .args(trainer_args(&options).map_err(fail)?)
        .arg("--preprocess-cache-dir")
        .arg(&cache)
        .arg("--progress-file")
        .arg(&progress_path)
        .arg("--cancel-file")
        .arg(&cancel_path)
        .args(resume_from.iter().flat_map(|dir| [std::ffi::OsString::from("--resume"), plain(dir.clone()).into()]))
        .env("GGML_BACKEND", backend)
        .current_dir(executable.parent().ok_or("Invalid native trainer path").map_err(|e| fail(e.into()))?)
        .kill_on_drop(true);
    if let Some(path) = runtime_path {
        command.env("PATH", path);
    }
    let log = std::fs::File::create(&log_path).map_err(|error| fail(error.to_string()))?;
    command
        .stdout(log.try_clone().map_err(|error| fail(error.to_string()))?)
        .stderr(log);
    crate::workload_job::configure_tokio_command(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| fail(format!("Could not start ace-train: {error}")))?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(fail(error));
    }
    crate::patch_carey_ace_training_status(
        &status_path,
        &[
            ("pid", json!(child.id())),
            ("launcherPid", json!(child.id())),
            ("runDir", json!(crate::user_facing_windows_path(&run))),
            ("phase", json!("loading")),
            ("totalFiles", json!(staged.count)),
            (
                "message",
                json!(match &continuation {
                    Some(Continuation { mode: ResumeMode::Exact, point, .. }) => {
                        format!("Resuming from epoch {} of {} with ace-train.", point.epoch, point.epochs)
                    }
                    Some(Continuation { mode: ResumeMode::Weights, point, .. }) => format!(
                        "Training {} epochs on {} songs from the epoch {} weights with ace-train.",
                        options.epochs, staged.count, point.epoch
                    ),
                    None => format!("Training on {} songs with ace-train.", staged.count),
                }),
            ),
        ],
    )
    .map_err(fail)?;
    Ok(Job {
        child,
        root: root.to_path_buf(),
        id,
        status_path,
        progress_path,
        cancel_path,
        run,
        options,
        captions: staged.captions,
        earlier,
    })
}

impl Job {
    pub fn status_path(&self) -> &Path {
        &self.status_path
    }

    fn update(&self) -> Result<(), String> {
        let Ok(progress) = crate::sa3_training::read_progress(&self.progress_path) else {
            // Not written yet, or caught mid-replacement; the next poll reads it.
            return Ok(());
        };
        let cancelling = self.cancel_path.exists();
        crate::patch_carey_ace_training_status(
            &self.status_path,
            &[
                ("status", json!("running")),
                ("phase", json!(progress.phase)),
                (
                    "message",
                    json!(if cancelling {
                        "Cancellation requested: finishing the current update and saving the adapter.".to_string()
                    } else {
                        progress.message
                    }),
                ),
                // The dialog reports epochs, as Python Carey's log does.
                ("currentStep", json!(progress.epoch)),
                ("maxSteps", json!(self.options.epochs)),
            ],
        )
    }

    /// Follow the trainer until it exits, then record the outcome and
    /// register whatever adapter it saved.
    pub async fn monitor(mut self) -> Result<(), String> {
        loop {
            match self.child.try_wait() {
                Ok(Some(exit)) => return self.finish(exit),
                Ok(None) => {
                    self.update()?;
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                Err(error) => return Err(error.to_string()),
            }
        }
    }

    fn finish(&self, exit: std::process::ExitStatus) -> Result<(), String> {
        let terminal = crate::sa3_training::read_progress(&self.progress_path).ok();
        let status = terminal.as_ref().map_or_else(|| "failed".to_string(), |progress| progress.status.clone());
        let status = status.as_str();
        if !exit.success() || !["completed", "cancelled"].contains(&status) {
            let error = terminal
                .as_ref()
                .map(|progress| progress.error.clone())
                .filter(|error| !error.is_empty())
                .unwrap_or_else(|| format!("ace-train exited with {exit}; see the training log."));
            crate::patch_carey_ace_training_status(
                &self.status_path,
                &[
                    ("status", json!("failed")),
                    ("phase", json!("failed")),
                    ("message", json!(error)),
                    ("error", json!(error)),
                    ("pid", serde_json::Value::Null),
                    ("ownerPid", serde_json::Value::Null),
                    ("launcherPid", serde_json::Value::Null),
                ],
            )?;
            return Ok(());
        }
        let terminal = terminal.expect("a completed or cancelled run has progress");
        let cancelled = status == "cancelled";
        let registered = if terminal.final_adapter.is_empty() {
            None
        } else {
            let output = PathBuf::from(&terminal.final_adapter).canonicalize().map_err(|error| error.to_string())?;
            if output != self.run.canonicalize().map_err(|error| error.to_string())? {
                return Err("ace-train reported an adapter outside its run folder".into());
            }
            Some(register(&self.root, &self.options, &self.id, &self.run, &self.captions, cancelled, &self.earlier)?)
        };
        let message = match (&registered, cancelled) {
            (Some(_), false) => format!("Training completed; {} is registered for Carey.", self.options.name),
            (Some(_), true) => format!(
                "Training cancelled at update {}; that adapter is registered as {}.",
                terminal.step, self.options.name
            ),
            (None, _) => "Training cancelled before the first update; no adapter was saved.".to_string(),
        };
        crate::patch_carey_ace_training_status(
            &self.status_path,
            &[
                ("status", json!(status)),
                ("phase", json!(status)),
                ("message", json!(message)),
                ("currentStep", json!(terminal.epoch)),
                ("maxSteps", json!(self.options.epochs)),
                ("finalCheckpointPath", json!(registered.as_deref().map(crate::user_facing_windows_path))),
                ("registeredLoraName", json!(registered.as_ref().map(|_| self.options.name.clone()))),
                ("pid", serde_json::Value::Null),
                ("ownerPid", serde_json::Value::Null),
                ("launcherPid", serde_json::Value::Null),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(raw: &str) -> crate::CareyAceSidecarFields {
        crate::parse_carey_ace_sidecar_content(raw)
    }

    #[test]
    fn a_carey_sidecar_becomes_complete_ace_train_metadata() {
        let sidecar = fields(
            "caption: dusty boom bap\nbpm: 92.4\nkeyscale: F minor\ntimesignature: 4/4\nis_instrumental: true\nlyrics:\n",
        );
        assert_eq!(
            metadata(&sidecar, "  my-trigger ").unwrap(),
            "caption: dusty boom bap\ngenre: \nbpm: 92\nkey: F minor\nsignature: 4\nis_instrumental: true\ncustom_tag: my-trigger\nlyrics: [Instrumental]\n"
        );
        let vocal = fields("caption: sad song\nbpm: 120\nis_instrumental: false\nlyrics: [verse]\nline one\nline two\n");
        let text = metadata(&vocal, "").unwrap();
        assert!(text.ends_with("custom_tag: \nlyrics: [verse]\nline one\nline two\n"), "{text}");
        assert!(text.contains("signature: 4\n"), "a missing signature is ACE-Step's 4/4");
        // A captioned vocal song arrives with no lyrics, and trains on none.
        let unlyrical = metadata(&fields("caption: whispered pop\nbpm: 90\nis_instrumental: false\n"), "").unwrap();
        assert!(unlyrical.ends_with("is_instrumental: false\ncustom_tag: \nlyrics: \n"), "{unlyrical}");
    }

    #[test]
    fn incomplete_sidecars_are_named_not_guessed() {
        assert_eq!(metadata(&fields("bpm: 120\n"), "").unwrap_err(), "no caption");
        assert_eq!(metadata(&fields("caption: x\n"), "").unwrap_err(), "no BPM");
        assert!(metadata(&fields("caption: x\nbpm: fast\n"), "").unwrap_err().contains("not a number"));
        let summary = problem_summary(&[
            ("no BPM".into(), vec!["a.wav".into(), "b.wav".into(), "c.wav".into(), "d.wav".into()]),
            ("no caption".into(), vec!["e.wav".into()]),
        ]);
        assert_eq!(summary, "no BPM: a.wav, b.wav and 2 more; no caption: e.wav");
        assert!(metadata(&fields("caption: x\nbpm: 90\ntimesignature: waltz\nis_instrumental: true\n"), "").is_err());
        assert_eq!(signature("6/8").unwrap(), 6);
    }

    #[test]
    fn training_recommends_xl_q4_on_an_8_gb_card() {
        let laptop = 8151u64 << 20;
        assert_eq!(recommended_training_tier("xl-base", laptop), "Q4_K_M");
        assert_eq!(recommended_training_tier("base", laptop), "Q8_0");
        assert_eq!(recommended_training_tier("xl-base", 16 << 30), "BF16");
    }

    fn options() -> Options {
        Options {
            name: "my-lora".into(),
            dataset: PathBuf::from("."),
            dit: "carey-native::xl-base::Q4_K_M".into(),
            adapter_type: "dora".into(),
            profile: "balanced".into(),
            rank: 64,
            epochs: 150,
            checkpoint_every: 25,
            batch_size: 1,
            accumulation: 1,
            learning_rate: 3e-4,
            cfg_dropout: 0.15,
            timestep_mu: -0.4,
            min_snr: Some(5.0),
            genre_ratio: 20,
            trigger: String::new(),
            window_min_seconds: 45.0,
            window_max_seconds: 60.0,
            resume: None,
        }
    }

    #[test]
    fn the_dialog_maps_onto_ace_train_exactly() {
        let args = trainer_args(&options()).unwrap();
        let value = |flag: &str| args[args.iter().position(|arg| arg == flag).unwrap() + 1].clone();
        assert_eq!(value("--dit"), "acestep-v15-xl-base-Q4_K_M.gguf");
        assert_eq!(value("--text-encoder"), "Qwen3-Embedding-0.6B-Q8_0.gguf");
        assert_eq!(value("--adapter-type"), "dora-rows");
        assert_eq!(value("--alpha"), "128");
        assert_eq!(value("--min-snr"), "5");
        assert_eq!(value("--window-max-seconds"), "60");
        assert!(validate(&options()).is_ok());
        assert!(validate(&Options { dit: "carey-native::turbo::BF16".into(), ..options() }).is_err());
        assert!(validate(&Options { batch_size: 2, ..options() }).is_err());
        assert!(validate(&Options { window_min_seconds: 70.0, ..options() }).is_err());
        let flat = Options { min_snr: None, adapter_type: "lora".into(), ..options() };
        assert!(!trainer_args(&flat).unwrap().contains(&"--min-snr".to_string()));
    }

    #[test]
    fn checkpoints_resolve_through_their_pointers() {
        let run = std::env::temp_dir().join(format!("gary-carey-run-{}-{}", std::process::id(), NEXT_JOB.fetch_add(1, Ordering::Relaxed)));
        let generation = |dir: &Path, n: u32| {
            let path = dir.join(".checkpoint-generations").join(format!("generation-{n}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("adapter_model.safetensors"), b"weights").unwrap();
            std::fs::write(path.join("adapter_config.json"), b"{}").unwrap();
            std::fs::write(dir.join("checkpoint.current"), format!(".checkpoint-generations/generation-{n}")).unwrap();
        };
        generation(&run.join("checkpoint-epoch-25"), 1);
        generation(&run.join("checkpoint-epoch-50"), 1);
        generation(&run, 2);
        let checkpoints = run_checkpoints(&run, "final epoch");
        let ids: Vec<_> = checkpoints.iter().map(|checkpoint| checkpoint.id.as_str()).collect();
        assert_eq!(ids, ["epoch-25", "epoch-50", "final"]);
        assert!(checkpoints[2].path.ends_with("generation-2"));
        std::fs::write(run.join("checkpoint.current"), "../elsewhere").unwrap();
        assert!(resolve_checkpoint(&run).is_err());
        std::fs::remove_dir_all(run).unwrap();
    }

    /// A runtime root holding one ten-epoch job that saved epoch 5 and then
    /// its final epoch; `final_state` writes the final save's trainer state.
    fn resume_root(final_state: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("gary-carey-resume-{}-{}", std::process::id(), NEXT_JOB.fetch_add(1, Ordering::Relaxed)));
        let job = root.join("carey/training/jobs/native-tones-1-0");
        let save = |dir: &Path, n: u32, state: &str| {
            let path = dir.join(".checkpoint-generations").join(format!("generation-{n}"));
            std::fs::create_dir_all(&path).unwrap();
            for file in ["adapter_model.safetensors", "adapter_config.json", "checkpoint_manifest.json", "optimizer_state.safetensors"] {
                std::fs::write(path.join(file), file).unwrap();
            }
            std::fs::write(path.join("trainer_state.json"), state).unwrap();
            std::fs::write(dir.join("checkpoint.current"), format!(".checkpoint-generations/generation-{n}")).unwrap();
        };
        save(&job.join("run/checkpoint-epoch-5"), 1, r#"{"completed_epochs":5,"continuation":{"next_batch":0}}"#);
        save(&job.join("run"), 2, final_state);
        std::fs::create_dir_all(job.join("dataset")).unwrap();
        for (stem, caption) in [("0000", "soft tones"), ("0001", "soft tones"), ("0002", "loud tones")] {
            std::fs::write(job.join(format!("dataset/{stem}.wav")), stem).unwrap();
            std::fs::write(job.join(format!("dataset/{stem}.txt")), format!("caption: {caption}\ngenre: ambient\n")).unwrap();
        }
        let options = Options { name: "tones".into(), epochs: 10, ..options() };
        crate::sa3_training::save(&job.join("native-options.json"), &options).unwrap();
        (root, job)
    }

    #[test]
    fn every_save_of_every_run_is_a_resume_point() {
        let (root, job) = resume_root(r#"{"completed_epochs":7,"continuation":{"next_batch":4}}"#);
        let points = resume_points(&root);
        let summary: Vec<_> = points.iter().map(|point| (point.epoch, point.mid_epoch, point.exact)).collect();
        // Epoch 5, then where the cancelled run stopped part way through epoch 8.
        assert_eq!(summary, [(5, false, true), (7, true, true)]);
        assert!(points.iter().all(|point| point.name == "tones" && point.job_id == "native-tones-1-0" && point.epochs == 10));
        assert_eq!(resume_options(&root, &job.join("run")).unwrap().epochs, 10);
        assert!(resume_options(&root, &root).is_err());
        std::fs::remove_dir_all(root).unwrap();

        // A run that finished its budget continues only from its weights.
        let (root, _) = resume_root(r#"{"completed_epochs":10,"continuation":{"next_batch":0}}"#);
        assert!(!resume_points(&root).last().unwrap().exact);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_resumes_take_the_original_settings_and_weights_keep_their_shape() {
        let (root, job) = resume_root(r#"{"completed_epochs":10,"continuation":{"next_batch":0}}"#);
        let resume = |checkpoint: &Path, mode| Some(Resume { checkpoint: checkpoint.to_path_buf(), mode });
        let asked = Options { name: "tones".into(), epochs: 40, learning_rate: 1e-4, ..options() };

        let epoch_5 = job.join("run/checkpoint-epoch-5");
        let (exact, from) = continuation(&root, Options { resume: resume(&epoch_5, ResumeMode::Exact), ..asked.clone() }).unwrap();
        assert_eq!((exact.epochs, exact.learning_rate), (10, 3e-4), "an exact resume ignores changed settings");
        assert_eq!(from.unwrap().point.epoch, 5);
        assert!(continuation(&root, Options { resume: resume(&job.join("run"), ResumeMode::Exact), ..asked.clone() }).is_err(), "nothing left to resume");

        let (weights, _) = continuation(&root, Options { resume: resume(&job.join("run"), ResumeMode::Weights), ..asked.clone() }).unwrap();
        assert_eq!((weights.epochs, weights.learning_rate), (40, 1e-4));
        for changed in [Options { rank: 32, ..asked.clone() }, Options { name: "other".into(), ..asked.clone() }] {
            assert!(continuation(&root, Options { resume: resume(&job.join("run"), ResumeMode::Weights), ..changed }).is_err());
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_exact_resume_reuses_the_staged_dataset_and_its_captions() {
        let (root, job) = resume_root(r#"{"completed_epochs":7,"continuation":{"next_batch":0}}"#);
        let dest = root.join("new-dataset");
        let staged = link_dataset(&job.join("dataset"), &dest).unwrap();
        assert_eq!(staged.count, 3);
        assert_eq!(staged.captions, ["soft tones", "loud tones"]);
        assert_eq!(std::fs::read(dest.join("0002.wav")).unwrap(), b"0002");
        std::fs::remove_dir_all(root).unwrap();
    }

    /// GARY4LOCAL_CAREY_TRAIN_SMOKE_ROOT (a new folder), _MODELS (a folder
    /// holding the 2B base and shared GGUFs) and _TRAINER (an ace-train.exe).
    /// Stages a nested, mixed-format Carey dataset, trains two short epochs on
    /// CUDA and registers the result under the smoke root.
    #[tokio::test]
    #[ignore = "trains with a real ace-train on real model GGUFs"]
    async fn real_native_training_stages_trains_and_registers() {
        let var = |name: &str| std::env::var(format!("GARY4LOCAL_CAREY_TRAIN_SMOKE_{name}")).ok();
        let root = PathBuf::from(var("ROOT").expect("a new isolated root is required"));
        assert!(!root.exists(), "the smoke root must not exist yet");
        let models = PathBuf::from(var("MODELS").expect("a model folder is required"));
        let trainer = PathBuf::from(var("TRAINER").expect("an ace-train is required"));
        let dataset = root.join("dataset");
        std::fs::create_dir_all(dataset.join("nested")).unwrap();
        // A 16-bit WAV at 44.1 kHz links as it is; a 24-bit WAV needs FFmpeg.
        for (path, bits, rate) in [(dataset.join("first.wav"), 16, 44_100), (dataset.join("nested/second.wav"), 24, 48_000)] {
            let spec = hound::WavSpec { channels: 2, sample_rate: rate, bits_per_sample: bits, sample_format: hound::SampleFormat::Int };
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            let scale = (1i32 << (bits - 2)) as f32;
            for i in 0..rate * 4 {
                let t = i as f32 / rate as f32;
                let sample = ((t * 220.0 * std::f32::consts::TAU).sin() * scale) as i32;
                writer.write_sample(sample).unwrap();
                writer.write_sample(sample / 2).unwrap();
            }
            writer.finalize().unwrap();
            // The nested song is a vocal one with no lyrics, as the captioner leaves it.
            let instrumental = bits == 16;
            std::fs::write(
                path.with_extension("txt"),
                format!("caption: soft sustained tones\nbpm: 120\nkeyscale: C major\ntimesignature: 4/4\nis_instrumental: {instrumental}\nlyrics:\n"),
            )
            .unwrap();
        }
        let staged_dir = root.join("job/dataset");
        let staged = stage(&dataset, &staged_dir, "smoke-tag").await.unwrap();
        assert_eq!(staged.count, 2);
        assert_eq!(staged.captions, ["soft sustained tones"]);
        assert!(std::fs::read_to_string(staged_dir.join("0001.txt"))
            .unwrap()
            .ends_with("is_instrumental: false\ncustom_tag: smoke-tag\nlyrics: \n"));
        let options = Options {
            dit: "carey-native::base::BF16".into(),
            epochs: 2,
            checkpoint_every: 1,
            rank: 8,
            window_min_seconds: 2.0,
            window_max_seconds: 2.0,
            dataset: dataset.clone(),
            ..options()
        };
        let run = root.join("job/run");
        let progress = root.join("job/progress.json");
        let status = tokio::process::Command::new(&trainer)
            .arg("--models")
            .arg(&models)
            .arg("--dataset")
            .arg(&staged_dir)
            .arg("--output")
            .arg(&run)
            .args(trainer_args(&options).unwrap())
            .arg("--preprocess-cache-dir")
            .arg(root.join("cache"))
            .arg("--progress-file")
            .arg(&progress)
            .arg("--cancel-file")
            .arg(root.join("job/cancel"))
            .env("GGML_BACKEND", "cuda")
            .status()
            .await
            .unwrap();
        assert!(status.success());
        let terminal = crate::sa3_training::read_progress(&progress).unwrap();
        assert_eq!(terminal.status, "completed");
        let generation = register(&root, &options, "smoke-job", &run, &staged.captions, false, &[]).unwrap();
        let catalog = crate::read_carey_lora_catalog_from(&crate::carey_lora_catalog_path_for(&root)).unwrap();
        let entry = &catalog["my-lora"];
        assert_eq!(entry.path, generation.to_string_lossy());
        assert_eq!(entry.selected_training_checkpoint.as_deref(), Some("final"));
        assert_eq!(entry.training_checkpoints.iter().map(|checkpoint| checkpoint.id.as_str()).collect::<Vec<_>>(), ["epoch-1", "final"]);
        assert!(root.join("carey/adapters/my-lora/adapter_model.safetensors").is_file());
        let pools: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("carey/captions.json")).unwrap()).unwrap();
        assert_eq!(pools["my-lora"], json!(["soft sustained tones"]));
    }

    /// GARY4LOCAL_CAREY_RESUME_SMOKE_ROOT (a new folder on the models' volume),
    /// _MODELS (a folder holding the 2B base BF16 and shared GGUFs), _TRAINER
    /// (an installed ace-train.exe) and optionally _BACKEND (default vulkan).
    /// Through `start`, trains three epochs, resumes exactly from epoch 1 and
    /// must reproduce the uninterrupted final adapter, then continues one epoch
    /// from the final weights.
    #[tokio::test]
    #[ignore = "trains with a real ace-train on real model GGUFs"]
    async fn real_native_training_resumes_exactly_and_from_weights() {
        let var = |name: &str| std::env::var(format!("GARY4LOCAL_CAREY_RESUME_SMOKE_{name}")).ok();
        let root = PathBuf::from(var("ROOT").expect("a new isolated root is required"));
        assert!(!root.exists(), "the smoke root must not exist yet");
        let models = PathBuf::from(var("MODELS").expect("a model folder is required"));
        let trainer = PathBuf::from(var("TRAINER").expect("an ace-train is required"));
        let backend = var("BACKEND").unwrap_or_else(|| "vulkan".into());
        let dit = "carey-native::base::BF16";
        std::fs::create_dir_all(root.join("models/carey")).unwrap();
        for id in [dit, carey_models::SHARED] {
            for file in &carey_models::models().component(id).unwrap().files {
                std::fs::hard_link(models.join(&file.filename), root.join("models/carey").join(&file.filename)).unwrap();
            }
        }
        let dataset = root.join("dataset");
        std::fs::create_dir_all(&dataset).unwrap();
        for (name, frequency) in [("low", 220.0), ("high", 440.0)] {
            let spec = hound::WavSpec { channels: 2, sample_rate: 48_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut writer = hound::WavWriter::create(dataset.join(format!("{name}.wav")), spec).unwrap();
            for i in 0..48_000 * 4 {
                let sample = ((i as f32 / 48_000.0 * frequency * std::f32::consts::TAU).sin() * 8_000.0) as i16;
                writer.write_sample(sample).unwrap();
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
            std::fs::write(
                dataset.join(format!("{name}.txt")),
                format!("caption: {name} sustained tone\nbpm: 120\nkeyscale: C major\ntimesignature: 4/4\nis_instrumental: true\nlyrics:\n"),
            )
            .unwrap();
        }
        let options = Options {
            name: "resume-smoke".into(),
            dataset: dataset.clone(),
            dit: dit.into(),
            epochs: 3,
            checkpoint_every: 1,
            rank: 8,
            window_min_seconds: 2.0,
            window_max_seconds: 2.0,
            ..options()
        };
        let train = |options: Options| {
            let (root, trainer, backend) = (root.clone(), trainer.clone(), backend.clone());
            async move {
                let job = start(&root, &trainer, &backend, None, options).await.unwrap();
                let status = job.status_path().to_path_buf();
                job.monitor().await.unwrap();
                let status: serde_json::Value = serde_json::from_slice(&std::fs::read(&status).unwrap()).unwrap();
                assert_eq!(status["status"], "completed", "{status}");
                status["finalCheckpointPath"].as_str().unwrap().to_string()
            }
        };
        let adapter = |generation: &str| std::fs::read(Path::new(generation).join("adapter_model.safetensors")).unwrap();

        let uninterrupted = train(options.clone()).await;
        let points = resume_points(&root);
        assert_eq!(points.iter().map(|point| (point.epoch, point.exact)).collect::<Vec<_>>(), [(1, true), (2, true), (3, false)]);

        // Changed settings are ignored: an exact resume trains as its run did.
        let resumed = train(Options {
            epochs: 40,
            resume: Some(Resume { checkpoint: PathBuf::from(&points[0].path), mode: ResumeMode::Exact }),
            ..options.clone()
        })
        .await;
        assert_eq!(adapter(&resumed), adapter(&uninterrupted), "an exact resume must reproduce the uninterrupted run");
        let catalog = crate::read_carey_lora_catalog_from(&crate::carey_lora_catalog_path_for(&root)).unwrap();
        let ids: Vec<_> = catalog["resume-smoke"].training_checkpoints.iter().map(|checkpoint| checkpoint.id.as_str()).collect();
        assert_eq!(ids, ["epoch-1", "epoch-2", "final"], "the resumed entry keeps the epoch it continued");

        let continued = train(Options {
            epochs: 1,
            resume: Some(Resume { checkpoint: PathBuf::from(&points[2].path), mode: ResumeMode::Weights }),
            ..options.clone()
        })
        .await;
        assert_ne!(adapter(&continued), adapter(&uninterrupted), "one more epoch changes the weights");
        let log = std::fs::read_to_string(Path::new(&continued).ancestors().nth(3).unwrap().join("training.log")).unwrap();
        assert!(log.contains("PEFT weight-only restart"), "continuing from weights restarts the optimizer");
        println!("PASS exact resume reproduced the uninterrupted adapter; weights continued for one epoch");
    }
}
