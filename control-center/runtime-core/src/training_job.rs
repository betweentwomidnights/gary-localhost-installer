//! Trainer ownership and native adapter publication for the headless console.
use crate::{
    carey_caption::{self, CaptionEndpoint},
    headless::{self, Config, Lease},
    model_catalog::{component, installed_carey_lms, PinnedHfFile},
    runtime,
    service::Service,
    training::{carey, sa3},
    workload_job::ManagedChild,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

pub(crate) enum Settings {
    Sa3(sa3::Options),
    Carey(carey::Options),
}

impl Settings {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Sa3(options) => sa3::validate(options),
            Self::Carey(options) => carey::validate(options),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Sa3(options) => &options.name,
            Self::Carey(options) => &options.name,
        }
    }

    pub fn files(&self) -> Result<Vec<&'static PinnedHfFile>, String> {
        let (service, ids) = match self {
            Self::Sa3(options) => (
                Service::Sa3,
                vec![
                    "sa3-native::text".into(),
                    "sa3-native::medium-decoder".into(),
                    format!("sa3-native::medium-base-{}", options.encoding),
                ],
            ),
            Self::Carey(options) => (
                Service::Carey,
                vec!["carey-native::shared".into(), options.dit.clone()],
            ),
        };
        let mut files = Vec::new();
        for id in ids {
            let item =
                component(service, &id).ok_or_else(|| format!("Unknown training model {id}"))?;
            for file in &item.files {
                file.validate()?;
                files.push(file);
            }
        }
        Ok(files)
    }
}

pub(crate) fn missing_bytes(config: &Config, files: &[&PinnedHfFile]) -> u64 {
    files
        .iter()
        .filter(|file| {
            !fs::symlink_metadata(config.models_dir.join(&file.filename))
                .is_ok_and(|meta| meta.is_file() && meta.len() == file.bytes)
        })
        .map(|file| file.bytes)
        .sum()
}

pub(crate) async fn fetch(
    config: &Config,
    files: Vec<&'static PinnedHfFile>,
) -> Result<(), String> {
    if config.models_dir != config.managed_models_dir() {
        return Err("Download requires managed model storage. Add the required files to the configured external weights folder.".into());
    }
    let _lease = headless::reserve_tool(config)?;
    headless::fetch_model_files(config, files).await
}

pub(crate) fn adapter_path(config: &Config, name: &str) -> Result<PathBuf, String> {
    let valid = crate::training::sanitize_lora_name(name).as_deref() == Some(name)
        && !["none", "default", "defaults"].contains(&name);
    if !valid {
        return Err("Use a LoRA name with lowercase letters, numbers, '-' or '_'.".into());
    }
    let leaf = if config.service == Service::Sa3 {
        format!("lora-{name}.gguf")
    } else {
        name.to_owned()
    };
    Ok(config.runtime_root.join("adapters").join(leaf))
}

fn adapter_dest(config: &Config, name: &str) -> Result<PathBuf, String> {
    let path = adapter_path(config, name)?;
    if path.exists() {
        return Err(format!(
            "A LoRA named {name} already exists. Choose another name."
        ));
    }
    Ok(path)
}

pub(crate) struct Scratch(pub PathBuf);
impl Scratch {
    pub fn new(config: &Config, kind: &str) -> Result<Self, String> {
        let path = config
            .runtime_root
            .join("tmp")
            .join(format!("{kind}-{}", headless::serial()));
        fs::create_dir(&path).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn publish(
    config: &Config,
    name: &str,
    source: &Path,
    family: &str,
    captions: &[String],
    sa3_dataset: Option<&Path>,
) -> Result<PathBuf, String> {
    let dest = adapter_dest(config, name)?;
    let stage = Scratch::new(config, "adapter")?;
    if config.service == Service::Sa3 {
        let file = stage.0.join("adapter.gguf");
        fs::copy(source, &file).map_err(|e| e.to_string())?;
        crate::sa3_gguf::creative_adapter(&file)?;
        if let Some(dataset) = sa3_dataset {
            crate::sa3_prompts::build(&config.runtime_root.join("prompts"), name, dataset, false)
                .map_err(|error| format!("Cannot build dice prompts for {name}: {error}"))?;
        }
        fs::rename(&file, &dest).map_err(|e| e.to_string())?;
    } else {
        let pointer = source.join("checkpoint.current");
        let source = match fs::symlink_metadata(&pointer) {
            Ok(meta) if meta.is_file() || meta.is_symlink() => carey::resolve_checkpoint(source)?,
            Ok(_) => {
                return Err(format!(
                    "Invalid checkpoint pointer at {}",
                    pointer.display()
                ))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => source.to_owned(),
            Err(error) => return Err(format!("Cannot inspect {}: {error}", pointer.display())),
        };
        for name in ["adapter_model.safetensors", "adapter_config.json"] {
            let src = source.join(name);
            if !fs::metadata(&src).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0) {
                return Err(format!(
                    "Expected {} in the PEFT adapter folder.",
                    src.display()
                ));
            }
            fs::copy(src, stage.0.join(name)).map_err(|e| e.to_string())?;
        }
        let metadata: Value = serde_json::from_slice(
            &fs::read(stage.0.join("adapter_config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if !metadata["peft_type"]
            .as_str()
            .is_some_and(|kind| kind.eq_ignore_ascii_case("lora"))
        {
            return Err("Choose a PEFT LoRA or DoRA folder.".into());
        }
        headless::write_json(
            &stage.0.join("adapter_info.json"),
            &json!({"model_family":family,"captions":captions}),
        )?;
        fs::rename(&stage.0, &dest).map_err(|e| e.to_string())?;
    }
    Ok(dest)
}

pub(crate) async fn add_lora(
    config: &Config,
    name: &str,
    source: &Path,
    metadata: Option<&Path>,
    family: &str,
) -> Result<PathBuf, String> {
    let _lease = headless::reserve_tool(config)?;
    adapter_dest(config, name)?;
    if config.service != Service::Sa3
        || source
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return publish(config, name, source, family, &[], None);
    }
    if !source
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
    {
        return Err("Choose a GGUF or safetensors SA3 LoRA.".into());
    }
    let temp = Scratch::new(config, "conversion")?;
    let output = temp.0.join("converted.gguf");
    let mut command = tool(config, "sa3-lora-convert")?;
    command
        .arg("--safetensors")
        .arg(source)
        .arg("--out")
        .arg(&output);
    if let Some(meta) = metadata {
        command.arg("--json").arg(meta);
    }
    let result = command
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(format!(
            "SA3 conversion failed: {}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    publish(config, name, &output, family, &[], None)
}

fn tool(config: &Config, name: &str) -> Result<tokio::process::Command, String> {
    let dir = config.installed_dir();
    let executable = dir.join(runtime::executable_name(name));
    if !executable.is_file() {
        return Err(format!(
            "Installed runtime is missing {name}. Run /install {}.",
            config.service.id()
        ));
    }
    let launch = config.launch(dir.clone(), config.public_port);
    let mut command = std::process::Command::new(executable);
    headless::clean_command_env(&mut command, &launch);
    command
        .current_dir(dir)
        .env("GGML_BACKEND", &config.device)
        .stdin(Stdio::null());
    crate::workload_job::configure_std_command(&mut command);
    Ok(command.into())
}

pub(crate) struct Job {
    config: Config,
    child: tokio::process::Child,
    directory: PathBuf,
    output: PathBuf,
    name: String,
    family: &'static str,
    captions: Vec<String>,
    sa3_dataset: Option<PathBuf>,
    pub progress: PathBuf,
    pub log: PathBuf,
    last_progress: Option<String>,
    _lease: Lease,
}

fn progress_line(service: Service, value: &sa3::Progress) -> String {
    if service == Service::Sa3 && value.phase == "pre-encode" {
        // The native trainer updates its message for each track before any steps advance.
        let detail = value
            .message
            .trim()
            .strip_prefix("[pre-encode] ")
            .unwrap_or_default()
            .chars()
            .filter(|ch| !ch.is_control())
            .collect::<String>();
        if !detail.is_empty() {
            return format!("sa3 training · pre-encode · {detail}");
        }
        return "sa3 training · pre-encode".into();
    }
    format!(
        "{} training · {} · step {}/{} · epoch {}",
        service.id(),
        value.phase,
        value.step,
        value.max_steps,
        value.epoch
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sa3_adapter::AdapterRegistry;

    #[test]
    fn uses_bf16_for_new_carey_caption_model_downloads() {
        let models =
            std::env::temp_dir().join(format!("gary-caption-default-{}", headless::serial()));
        let config: Config = serde_json::from_value(json!({
            "service": "carey", "models_dir": models, "device": "Vulkan0",
            "public_port": 8003, "model": "base", "encoding": "bf16"
        }))
        .unwrap();
        assert_eq!(
            caption_files(&config).unwrap()[0].filename,
            "acestep-5Hz-lm-1.7B-BF16.gguf"
        );
    }

    #[test]
    fn published_sa3_training_has_a_registry_entry_and_dice_prompts() {
        let root =
            std::env::temp_dir().join(format!("gary-sa3-publication-{}", headless::serial()));
        for dir in ["tmp", "adapters", "prompts", "dataset"] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        let config: Config = serde_json::from_value(json!({
            "runtime_root": &root, "device": "Vulkan0", "public_port": 28309,
            "model": "medium", "encoding": "f16"
        }))
        .unwrap();
        let checkpoint = root.join("adapter-final.gguf");
        fs::write(&checkpoint, crate::sa3_gguf::fixture("sa3-lora", "dit")).unwrap();
        let dataset = root.join("dataset");
        fs::write(dataset.join("a.txt"), "Dusty drum break, 95 bpm, A minor").unwrap();
        fs::write(dataset.join("b.txt"), "dusty drum break, Key: A min").unwrap();

        let published = publish(&config, "trained", &checkpoint, "", &[], Some(&dataset)).unwrap();
        let registry = AdapterDirectory(root.join("adapters")).entries().unwrap();
        assert_eq!(registry.len(), 1);
        assert_eq!(registry[0].name, "trained");
        assert_eq!(registry[0].source_path, published.to_string_lossy());
        let pool: Value =
            serde_json::from_slice(&fs::read(root.join("prompts/trained.json")).unwrap()).unwrap();
        assert_eq!(pool["dice"]["instrumental"], json!(["Dusty drum break"]));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publishes_careys_checkpoint_generation_with_dice_prompts() {
        let root =
            std::env::temp_dir().join(format!("gary-carey-publication-{}", headless::serial()));
        let checkpoint = root.join("run");
        let relative = ".checkpoint-generations/generation-1";
        let generation = checkpoint.join(relative);
        for dir in [root.join("adapters"), root.join("tmp"), generation.clone()] {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(generation.join("adapter_model.safetensors"), b"latest").unwrap();
        fs::write(
            generation.join("adapter_config.json"),
            b"{\"peft_type\":\"LORA\"}",
        )
        .unwrap();
        fs::write(checkpoint.join("adapter_model.safetensors"), b"previous").unwrap();
        fs::write(
            checkpoint.join("adapter_config.json"),
            b"{\"peft_type\":\"LORA\"}",
        )
        .unwrap();
        let config: Config = serde_json::from_value(json!({
            "service": "carey", "runtime_root": &root, "device": "Vulkan0",
            "public_port": 8003, "model": "xl-base", "encoding": "bf16"
        }))
        .unwrap();
        let pointer = checkpoint.join("checkpoint.current");
        fs::write(&pointer, relative).unwrap();
        #[cfg(unix)]
        {
            fs::remove_file(&pointer).unwrap();
            std::os::unix::fs::symlink(relative, &pointer).unwrap();
        }
        let captions = ["Quiet synth chords".into(), "ambient, drone".into()];
        let published = publish(&config, "trained", &checkpoint, "xl", &captions, None).unwrap();
        assert_eq!(
            fs::read(published.join("adapter_model.safetensors")).unwrap(),
            b"latest"
        );
        let info: Value =
            serde_json::from_slice(&fs::read(published.join("adapter_info.json")).unwrap())
                .unwrap();
        assert_eq!(info, json!({"model_family":"xl","captions":captions}));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sa3_pre_encode_updates_before_the_first_step() {
        let mut progress: sa3::Progress = serde_json::from_value(json!({
            "schema_version": 1,
            "status": "running",
            "phase": "pre-encode",
            "message": "[pre-encode] track-a: 90.0s -> 969 frames, gain 1.0000\n",
            "error": "",
            "step": 0,
            "max_steps": 2000,
            "final_adapter": "",
            "adapter_checkpoint": ""
        }))
        .unwrap();
        assert_eq!(
            progress_line(Service::Sa3, &progress),
            "sa3 training · pre-encode · track-a: 90.0s -> 969 frames, gain 1.0000"
        );
        progress.message = "[pre-encode] track-b: reused 468 frames\n".into();
        assert_eq!(
            progress_line(Service::Sa3, &progress),
            "sa3 training · pre-encode · track-b: reused 468 frames"
        );
        progress.phase = "training".into();
        progress.step = 1;
        assert_eq!(
            progress_line(Service::Sa3, &progress),
            "sa3 training · training · step 1/2000 · epoch 0"
        );
    }
}

impl Job {
    pub fn cancel(&self) -> Result<(), String> {
        fs::write(self.directory.join("cancel.requested"), b"cancel\n").map_err(|e| e.to_string())
    }

    pub fn progress_summary(&self) -> Option<String> {
        sa3::read_progress(&self.progress)
            .ok()
            .map(|value| progress_line(self.config.service, &value))
    }

    pub fn poll(&mut self) -> Result<Option<String>, String> {
        if let Some(exit) = self.child.try_wait().map_err(|e| e.to_string())? {
            let result = sa3::read_progress(&self.progress).map_err(|e| {
                format!(
                    "Trainer exited with {exit}. {e}. Log {}",
                    self.log.display()
                )
            })?;
            if result.status == "cancelled" {
                return Ok(Some(format!(
                    "Training cancelled. Checkpoints are in {}",
                    self.output.display()
                )));
            }
            if !exit.success() || result.status != "completed" {
                return Err(format!(
                    "Training failed: {}. Log {}",
                    result.error,
                    self.log.display()
                ));
            }
            let final_adapter = Path::new(&result.final_adapter)
                .canonicalize()
                .map_err(|e| format!("Missing final adapter: {e}"))?;
            let output = self.output.canonicalize().map_err(|e| e.to_string())?;
            if !final_adapter.starts_with(&output) {
                return Err("Trainer returned a checkpoint outside its run.".into());
            }
            let dest = publish(
                &self.config,
                &self.name,
                &final_adapter,
                self.family,
                &self.captions,
                self.sa3_dataset.as_deref(),
            )
            .map_err(|e| format!("{e}\nTraining output is in {}", self.output.display()))?;
            return Ok(Some(format!(
                "Training complete. LoRA added at {}{}",
                dest.display(),
                if self.sa3_dataset.is_some() || !self.captions.is_empty() {
                    "\nDice prompts saved for this LoRA."
                } else {
                    ""
                }
            )));
        }
        if let Some(line) = self.progress_summary() {
            if self.last_progress.as_ref() != Some(&line) {
                println!("{line}");
                self.last_progress = Some(line);
            }
        }
        Ok(None)
    }

    pub async fn cancel_and_wait(&mut self) -> Result<(), String> {
        self.cancel()?;
        tokio::time::timeout(Duration::from_secs(30), self.child.wait())
            .await
            .map_err(|_| {
                format!(
                    "Trainer did not stop within 30 seconds. Checkpoints and logs are in {}",
                    self.directory.display()
                )
            })?
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

pub(crate) async fn start(config: &Config, settings: Settings) -> Result<Job, String> {
    settings.validate()?;
    let lease = headless::reserve_tool(config)?;
    if missing_bytes(config, &settings.files()?) > 0 {
        return Err("Training weights are missing or incomplete.".into());
    }
    let name = settings.name().to_owned();
    adapter_dest(config, &name)?;
    let (executable, family) = match &settings {
        Settings::Sa3(_) => ("sa3-train", ""),
        Settings::Carey(value) if value.dit.contains("::xl-base::") => ("ace-train", "xl"),
        Settings::Carey(_) => ("ace-train", "standard"),
    };
    let info = tokio::time::timeout(
        Duration::from_secs(15),
        tool(config, executable)?
            .arg("--control-info")
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| "Trainer capability probe timed out")?
    .map_err(|e| e.to_string())?;
    let caps: Value = serde_json::from_slice(&info.stdout).unwrap_or_default();
    if !info.status.success()
        || caps["schema_version"] != 1
        || caps["cooperative_cancel_file"] != true
        || caps["atomic_progress_file"] != true
    {
        return Err(
            "Update the runtime to a release with trainer progress and cancellation support."
                .into(),
        );
    }
    let directory = config
        .runtime_root
        .join("training")
        .join("jobs")
        .join(format!("{name}-{}", headless::serial()));
    fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let output = directory.join("run");
    let progress = directory.join("native-progress.json");
    let cancel = directory.join("cancel.requested");
    let log = directory.join("training.log");
    let cache = config.runtime_root.join("training").join("cache");
    fs::create_dir_all(&cache).map_err(|e| e.to_string())?;
    let mut command = tool(config, executable)?;
    let (captions, sa3_dataset) = match settings {
        Settings::Sa3(options) => {
            let prompts = directory.join("prompt-config.json");
            headless::write_json(&prompts, &sa3::caption_prompt_config(&options.fixed_prompt))?;
            headless::write_json(&directory.join("native-options.json"), &options)?;
            sa3::trainer_args(
                &mut command,
                &options,
                &config.device,
                &config.models_dir,
                &output,
                &cache,
                &progress,
                &cancel,
            )?;
            command.arg("--prompt-config").arg(prompts);
            (Vec::new(), Some(options.dataset))
        }
        Settings::Carey(options) => {
            let staged = directory.join("dataset");
            let data =
                crate::carey_dataset::stage(&options.dataset, &staged, &options.trigger).await?;
            headless::write_json(&directory.join("native-options.json"), &options)?;
            command
                .arg("--models")
                .arg(&config.models_dir)
                .arg("--dataset")
                .arg(staged)
                .arg("--output")
                .arg(&output)
                .args(carey::trainer_args(&options)?)
                .arg("--preprocess-cache-dir")
                .arg(&cache)
                .arg("--progress-file")
                .arg(&progress)
                .arg("--cancel-file")
                .arg(cancel);
            (data.captions, None)
        }
    };
    let writer = fs::File::create(&log).map_err(|e| e.to_string())?;
    command
        .stdout(writer.try_clone().map_err(|e| e.to_string())?)
        .stderr(writer)
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|e| e.to_string())?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    Ok(Job {
        _lease: lease,
        config: config.clone(),
        child,
        directory,
        output,
        name,
        family,
        captions,
        sa3_dataset,
        progress,
        log,
        last_progress: None,
    })
}

pub(crate) fn caption_files(config: &Config) -> Result<Vec<&'static PinnedHfFile>, String> {
    let all = component(Service::Carey, "carey-native::shared").ok_or("Carey has no audio VAE")?;
    let vae = all.files.get(1).ok_or("Carey has no audio VAE")?;
    let dit = crate::model_catalog::variants(Service::Carey)
        .into_iter()
        .filter_map(|(model, tier)| {
            component(Service::Carey, &format!("carey-native::{model}::{tier}"))
        })
        .find(|item| missing_bytes(config, &item.files.iter().collect::<Vec<_>>()) == 0)
        .or_else(|| component(Service::Carey, "carey-native::base::Q4_K_M"))
        .ok_or("Carey's caption catalog has no DiT")?;
    let default_lm = component(Service::Carey, "carey-native::lm-1.7B::BF16")
        .and_then(|item| item.files.first())
        .ok_or("Carey's caption catalog has no default LM")?;
    let lm = installed_carey_lms(&config.models_dir)
        .find(|file| file.filename == default_lm.filename)
        .or_else(|| {
            installed_carey_lms(&config.models_dir).find(|file| file.filename.contains("1.7B"))
        })
        .or_else(|| installed_carey_lms(&config.models_dir).next())
        .unwrap_or(default_lm);
    Ok(vec![lm, &dit.files[0], vae])
}

pub(crate) async fn auto_label(
    config: &Config,
    service: Service,
    dataset: &Path,
    files: &[PathBuf],
    overwrite: bool,
) -> Result<(), String> {
    let _lease = headless::reserve_tool(config)?;
    let models = caption_files(config)?;
    if missing_bytes(config, &models) > 0 {
        return Err("Caption weights are missing or incomplete.".into());
    }
    let scratch = Scratch::new(config, "caption")?;
    let socket = runtime::bind_loopback(0).map_err(|e| e.to_string())?;
    let port = socket.local_addr().map_err(|e| e.to_string())?.port();
    let launch = config.launch(config.installed_dir(), port);
    let mut command = launch.std_command();
    headless::clean_command_env(&mut command, &launch);
    let log_path = config.runtime_root.join("logs/caption.log");
    let log = fs::File::create(&log_path).map_err(|e| e.to_string())?;
    command
        .current_dir(&scratch.0)
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log);
    drop(socket);
    let mut child = ManagedChild::spawn(command)?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let base = format!("http://127.0.0.1:{port}");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(exit) = child.0.try_wait().map_err(|e| e.to_string())? {
            return Err(format!(
                "Carey caption server exited with {exit}. Log {}",
                log_path.display()
            ));
        }
        if client
            .get(format!("{base}/health"))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            break;
        }
        if tokio::time::Instant::now() > deadline {
            return Err(format!(
                "Carey caption server startup timed out. Log {}",
                log_path.display()
            ));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let endpoint = CaptionEndpoint {
        client: &client,
        base: &base,
    };
    let options = carey_caption::Options {
        dataset: dataset.to_owned(),
        lm: models[0].filename.clone(),
        dit: models[1].filename.clone(),
        vae: models[2].filename.clone(),
        overwrite,
        bpm_analysis: true,
        key_analysis: true,
        trigger: String::new(),
    };
    let display = |info: carey_caption::Progress<'_>| {
        println!("Labeling track {} of {}", info.index, info.total)
    };
    let result = if service == Service::Sa3 {
        carey_caption::label_sa3_dataset(
            &endpoint,
            &options,
            files,
            false,
            &scratch.0,
            display,
            || false,
        )
        .await
    } else {
        carey_caption::caption_dataset(
            &endpoint,
            &options,
            &scratch.0,
            display,
            || false,
            |error| {
                println!("Skipped a track after retrying.\n{error}");
                Ok(())
            },
        )
        .await
        .map(|(captioned, total)| println!("Captioned {captioned} of {total} dataset tracks."))
    };
    child.stop();
    result.map_err(|error| format!("{error}\nLog {}", log_path.display()))
}

/// SA3's public adapter uses these managed names to resolve client requests.
pub(crate) struct AdapterDirectory(pub PathBuf);

impl crate::sa3_adapter::AdapterRegistry for AdapterDirectory {
    fn entries(&self) -> Result<Vec<crate::sa3_adapter::AdapterLora>, String> {
        let mut adapters = Vec::new();
        for entry in fs::read_dir(&self.0).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            if !entry
                .file_type()
                .map_err(|error| error.to_string())?
                .is_file()
            {
                continue;
            }
            let path = entry.path();
            let Some(name) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| stem.strip_prefix("lora-"))
            else {
                continue;
            };
            if path.extension().is_none_or(|ext| ext != "gguf")
                || crate::training::sanitize_lora_name(name).as_deref() != Some(name)
                || ["default", "defaults", "none"].contains(&name)
            {
                continue;
            }
            let file = path.to_string_lossy().into_owned();
            adapters.push(crate::sa3_adapter::AdapterLora {
                name: name.to_owned(),
                source_path: file.clone(),
                native_path: Some(file),
                strength: 1.0,
                error: None,
            });
        }
        adapters.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(adapters)
    }

    fn paths(&self) -> (PathBuf, PathBuf) {
        (self.0.clone(), self.0.clone())
    }

    fn append_decoder<'a>(
        &'a self,
        _: &'a mut Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}
