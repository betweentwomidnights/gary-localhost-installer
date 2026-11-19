use super::{answer, model_path, model_size, say, service_label, Console, Input};
use crate::{
    dataset,
    service::Service,
    training::{self, carey, sa3},
    training_job::{self, Settings},
};
use std::{
    fmt::Display,
    fs,
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
};

const SUPPORT: &str = "https://discord.gg/jBxGPtw3xa";

async fn yes(input: &mut Input, question: &str) -> Result<bool, String> {
    Ok(matches!(
        answer(input, question).await?.to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

async fn setting<T: FromStr + Display>(
    input: &mut Input,
    label: &str,
    default: T,
) -> Result<T, String> {
    loop {
        let value = answer(input, &format!("{label} [{default}]: ")).await?;
        if value.is_empty() {
            return Ok(default);
        }
        if let Ok(parsed) = value.parse() {
            return Ok(parsed);
        }
        say("Enter a valid value.")?;
    }
}

async fn sa3_window(input: &mut Input) -> Result<f64, String> {
    say(format!(
        "Full tracks use SA3’s {0}-second window. Longer tracks use random crops of {0} seconds. Shorter windows use less VRAM.",
        sa3::FULL_TRACK_SECONDS
    ))?;
    loop {
        let value = answer(input, "Training window (full or seconds) [47]: ").await?;
        match value.to_ascii_lowercase().as_str() {
            "" => return Ok(47.0),
            "full" | "full track" | "full tracks" | "full-track" | "full-tracks" => {
                return Ok(sa3::FULL_TRACK_SECONDS);
            }
            _ => {
                if let Ok(seconds) = value.parse::<f64>() {
                    if sa3::DURATION_SECONDS.contains(&seconds) {
                        return Ok(seconds);
                    }
                }
            }
        }
        say("Enter full or a window between 0.1 and 380 seconds.")?;
    }
}

async fn learning_rate(input: &mut Input, default: f64) -> Result<f64, String> {
    say("Learning rate sets the size of each update. Compared with 1e-4, 3e-4 takes larger steps and 1e-5 takes smaller steps.")?;
    loop {
        let value = answer(input, &format!("Learning rate [{default:e}]: ")).await?;
        if value.is_empty() {
            return Ok(default);
        }
        if let Ok(rate) = value.parse::<f64>() {
            if rate > 0.0 && rate <= 1.0 {
                return Ok(rate);
            }
        }
        say("Enter a learning rate greater than 0 and at most 1, such as 1e-4.")?;
    }
}

async fn rank(input: &mut Input, default: u32) -> Result<u32, String> {
    say("Higher rank (for example, 64–128) gives the adapter more capacity within the selected layers and uses more memory.")?;
    setting(input, "Rank", default).await
}

async fn lora_name(input: &mut Input) -> Result<String, String> {
    let input = answer(input, "LoRA name: ").await?;
    training::sanitize_lora_name(&input)
        .filter(|s| !["none", "default", "defaults"].contains(&s.as_str()))
        .ok_or_else(|| "Use letters, numbers, hyphens, and underscores for the LoRA name.".into())
}

async fn settings(
    input: &mut Input,
    service: Service,
    dataset: PathBuf,
    name: String,
) -> Result<Settings, String> {
    if service == Service::Sa3 {
        let value = sa3::Options {
            dataset,
            name,
            fixed_prompt: answer(input, "Trigger text [none]: ").await?,
            encoding: setting(
                input,
                "Training base precision (F16 or Q4_K_M)",
                "F16".to_owned(),
            )
            .await?
            .to_uppercase(),
            steps: setting(input, "Training steps", 2000).await?,
            checkpoint_every: setting(input, "Save every N steps", 500).await?,
            duration: sa3_window(input).await?,
            rank: 16,
            batch_size: 1,
            learning_rate: 1e-4,
            target_latent_rms: 0.0,
            layer_scope: "transformer-core".into(),
            resume: None,
            prompt_config: None,
        };
        let mut value = value;
        if yes(
            input,
            "Customize learning rate, rank, and advanced settings? [y/N] ",
        )
        .await?
        {
            value.learning_rate = learning_rate(input, value.learning_rate).await?;
            value.rank = rank(input, value.rank).await?;
            value.batch_size = setting(input, "Batch size", value.batch_size).await?;
            value.layer_scope = setting(
                input,
                "Layer scope (transformer-core, full, full-no-seconds)",
                value.layer_scope,
            )
            .await?;
            value.target_latent_rms = setting(
                input,
                "Target latent RMS (0 disables)",
                value.target_latent_rms,
            )
            .await?;
        }
        Ok(Settings::Sa3(value))
    } else {
        let model = setting(
            input,
            "Training checkpoint (base or xl-base)",
            "base".to_owned(),
        )
        .await?;
        let tier = setting(input, "Training precision", "BF16".to_owned())
            .await?
            .to_uppercase();
        let mut value = carey::Options {
            name,
            dataset,
            dit: format!("carey-native::{model}::{tier}"),
            trigger: answer(input, "Trigger text [none]: ").await?,
            epochs: setting(input, "Epochs", 150).await?,
            checkpoint_every: setting(input, "Save every N epochs", 25).await?,
            window_min_seconds: setting(input, "Minimum window in seconds", 45.0).await?,
            window_max_seconds: setting(input, "Maximum window in seconds", 60.0).await?,
            adapter_type: "dora".into(),
            profile: "balanced".into(),
            rank: 64,
            batch_size: 1,
            accumulation: 1,
            learning_rate: 3e-4,
            cfg_dropout: 0.15,
            timestep_mu: -0.4,
            min_snr: Some(5.0),
            genre_ratio: 20,
            resume: None,
        };
        if yes(
            input,
            "Customize learning rate, rank, and advanced settings? [y/N] ",
        )
        .await?
        {
            value.learning_rate = learning_rate(input, value.learning_rate).await?;
            value.rank = rank(input, value.rank).await?;
            value.adapter_type =
                setting(input, "Adapter type (lora or dora)", value.adapter_type).await?;
            value.profile =
                setting(input, "Modules (attention or balanced)", value.profile).await?;
            value.accumulation =
                setting(input, "Gradient accumulation", value.accumulation).await?;
            value.cfg_dropout = setting(input, "CFG dropout", value.cfg_dropout).await?;
            value.timestep_mu = setting(input, "Timestep mu", value.timestep_mu).await?;
            value.genre_ratio =
                setting(input, "Genre prompt percentage", value.genre_ratio).await?;
            let snr = setting(
                input,
                "Min SNR gamma (0 disables)",
                value.min_snr.unwrap_or(0.0),
            )
            .await?;
            value.min_snr = (snr != 0.0).then_some(snr);
        }
        Ok(Settings::Carey(value))
    }
}

struct Track {
    relative: String,
    audio: PathBuf,
    sidecar: PathBuf,
    has_json: bool,
}

fn tracks(service: Service, path: &Path) -> Result<Vec<Track>, String> {
    let folder = path.to_str().ok_or("Use a UTF-8 dataset path.")?;
    let rows: Vec<_> = if service == Service::Sa3 {
        dataset::read_sa3_dataset_sidecars(folder)?
            .into_iter()
            .map(|row| Track {
                relative: row.relative_path,
                audio: row.audio_path.into(),
                sidecar: row.sidecar_path.into(),
                has_json: row.json_sidecar_exists,
            })
            .collect()
    } else {
        dataset::read_carey_ace_dataset_sidecars(folder)?
            .into_iter()
            .map(|row| Track {
                relative: row.relative_path,
                audio: row.audio_path.into(),
                sidecar: row.sidecar_path.into(),
                has_json: false,
            })
            .collect()
    };
    if rows.is_empty() {
        return Err("The dataset folder contains no supported audio files.".into());
    }
    let mut names = std::collections::HashSet::new();
    for row in &rows {
        if !names.insert(&row.sidecar) {
            return Err(format!(
                "Multiple audio files share {}. Give the audio files distinct names.",
                row.sidecar.display()
            ));
        }
        if fs::symlink_metadata(&row.sidecar).is_ok_and(|file| !file.is_file()) {
            return Err(format!(
                "Use a regular sidecar file at {}",
                row.sidecar.display()
            ));
        }
    }
    Ok(rows)
}

fn safe_text(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\t'))
        .collect()
}

async fn review(input: &mut Input, rows: &[Track]) -> Result<(), String> {
    loop {
        for (index, row) in rows.iter().enumerate() {
            let status = if row.sidecar.is_file() {
                "text sidecar"
            } else if row.has_json {
                "JSON metadata"
            } else {
                "missing sidecar"
            };
            say(format!(
                "{:>3}  {}  ·  {status}",
                index + 1,
                safe_text(&row.relative)
            ))?;
        }
        let choice = answer(
            input,
            "Track number to review or edit [Enter for training settings]: ",
        )
        .await?;
        if choice.is_empty() {
            return Ok(());
        }
        let Some(row) = choice
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| rows.get(n))
        else {
            say("Choose a listed track number.")?;
            continue;
        };
        let original = match fs::read_to_string(&row.sidecar) {
            Ok(content) => Some(content),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.to_string()),
        };
        say(row.sidecar.display())?;
        say(safe_text(
            original.as_deref().unwrap_or("No text sidecar yet."),
        ))?;
        if !yes(input, "Replace this text sidecar? [y/N] ").await? {
            continue;
        }
        say("Paste or type the sidecar. Enter /save on its own line to save, or /cancel to leave it as is.")?;
        let mut replacement = String::new();
        loop {
            let line = input.recv().await.ok_or("End of terminal input.")??;
            match line.trim() {
                "/cancel" => break,
                "/save" => {
                    if replacement.trim().is_empty() {
                        say("Enter some sidecar text or /cancel.")?;
                        continue;
                    }
                    if fs::symlink_metadata(&row.sidecar).is_ok_and(|file| !file.is_file())
                        || fs::read_to_string(&row.sidecar).ok() != original
                    {
                        return Err("The sidecar changed. Reopen the dataset review.".into());
                    }
                    let stage = row
                        .sidecar
                        .with_extension(format!("{}.tmp", crate::headless::serial()));
                    let mut file = fs::File::create_new(&stage).map_err(|e| e.to_string())?;
                    let result = file
                        .write_all(replacement.as_bytes())
                        .and_then(|_| file.sync_all());
                    drop(file);
                    let result = result.and_then(|_| {
                        if original.is_none() {
                            fs::hard_link(&stage, &row.sidecar)
                        } else {
                            fs::rename(&stage, &row.sidecar)
                        }
                    });
                    let _ = fs::remove_file(&stage);
                    result.map_err(|e| e.to_string())?;
                    say("Sidecar saved.")?;
                    break;
                }
                _ => {
                    replacement.push_str(line.trim_end_matches('\r'));
                    replacement.push('\n');
                }
            }
        }
    }
}

impl Console {
    pub(super) async fn add_lora(&self, input: &mut Input, service: Service) -> Result<(), String> {
        let config = self.config(service)?;
        if self.state(service).await?["operation_active"] == true {
            return Err(format!(
                "Stop {} before adding a LoRA.",
                service_label(service)
            ));
        }
        let source = model_path(
            &answer(
                input,
                if service == Service::Sa3 {
                    "Path to the SA3 GGUF or safetensors LoRA: "
                } else {
                    "Path to the Carey PEFT folder: "
                },
            )
            .await?,
        )
        .canonicalize()
        .map_err(|e| format!("Cannot open the adapter: {e}"))?;
        let name = lora_name(input).await?;
        let mut meta = None;
        let mut family = String::new();
        if service == Service::Carey {
            family = setting(input, "Model family (2b or xl)", "2b".to_owned()).await?;
            family = match family.as_str() {
                "2b" => "standard".into(),
                "xl" => "xl".into(),
                _ => return Err("Choose 2b or xl.".into()),
            };
        } else if source
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
            let value = answer(input, "Path to LoRA JSON [Enter if embedded]: ").await?;
            if !value.is_empty() {
                meta = Some(
                    model_path(&value)
                        .canonicalize()
                        .map_err(|e| e.to_string())?,
                );
            }
        }
        let added =
            training_job::add_lora(&config, &name, &source, meta.as_deref(), &family).await?;
        say(format!(
            "Added {name} at {}. Start {} to use it.",
            added.display(),
            service.id()
        ))
    }

    pub(super) async fn train_lora(
        &mut self,
        input: &mut Input,
        service: Service,
    ) -> Result<(), String> {
        if self.training.is_some() {
            return Err(
                "A training job is already active here. Use /training for progress or /cancel to stop it.".into(),
            );
        }
        let config = self.config(service)?;
        if self.state(service).await?["operation_active"] == true {
            return Err(format!(
                "Stop {} before training a LoRA.",
                service_label(service)
            ));
        }
        let folder = answer(input, "Where’s your dataset? ").await?;
        if folder.is_empty() {
            return Err("Enter a dataset folder.".into());
        }
        let folder = model_path(&folder)
            .canonicalize()
            .map_err(|e| format!("Cannot open the dataset: {e}"))?;
        let rows = tracks(service, &folder)?;
        say(format!(
            "Found {} tracks in {}",
            rows.len(),
            folder.display()
        ))?;
        if yes(input, "Auto-label the dataset? [y/N] ").await? {
            let result = self.label_dataset(input, service, &folder, &rows).await;
            if let Err(error) = result {
                say(format!(
                    "Auto-labeling stopped: {error}\nYou can review or edit sidecars below."
                ))?;
            }
        }
        review(input, &rows).await?;
        say(format!(
            "Training settings help · \x1b]8;;{SUPPORT}\x1b\\{SUPPORT}\x1b]8;;\x1b\\ @kev"
        ))?;
        let name = lora_name(input).await?;
        let settings = settings(input, service, folder, name).await?;
        settings.validate()?;
        let files = settings.files()?;
        let missing = training_job::missing_bytes(&config, &files);
        if missing > 0 {
            if !yes(
                input,
                &format!(
                    "Download {} of training weights? [y/N] ",
                    model_size(missing)
                ),
            )
            .await?
            {
                return say("Training cancelled.");
            }
            training_job::fetch(&config, files).await?;
        }
        if !yes(input, "Start training? [y/N] ").await? {
            return say("Training cancelled.");
        }
        let job = training_job::start(&config, settings).await?;
        say(format!(
            "Training started. Type /cancel to stop safely.\n/training shows progress and log paths.\nLog {}",
            job.log.display()
        ))?;
        if service == Service::Sa3 {
            say(
                "SA3 pre-encodes the audio before step 1. Progress updates as each track finishes.",
            )?;
        }
        self.training = Some(job);
        Ok(())
    }

    async fn label_dataset(
        &self,
        input: &mut Input,
        service: Service,
        folder: &Path,
        rows: &[Track],
    ) -> Result<(), String> {
        let overwrite = rows.iter().any(|row| row.sidecar.is_file())
            && yes(input, "Replace existing text sidecars too? [y/N] ").await?;
        let files: Vec<_> = rows
            .iter()
            .filter(|row| overwrite || (!row.sidecar.exists() && !row.has_json))
            .map(|row| row.audio.clone())
            .collect();
        if files.is_empty() {
            return say("All tracks already have sidecars.");
        }
        let carey = self.config(Service::Carey)?;
        if self.state(Service::Carey).await?["operation_active"] == true {
            return Err("Stop Carey before running the captioner.".into());
        }
        let caption = training_job::caption_files(&carey)?;
        let missing = training_job::missing_bytes(&carey, &caption);
        if missing > 0 {
            if !yes(
                input,
                &format!(
                    "Download {} of Carey captioning weights? [y/N] ",
                    model_size(missing)
                ),
            )
            .await?
            {
                return Ok(());
            }
            training_job::fetch(&carey, caption).await?;
        }
        training_job::auto_label(&carey, service, folder, &files, overwrite).await?;
        say("Auto-labeling finished. Review the generated sidecars before training.")
    }
}
