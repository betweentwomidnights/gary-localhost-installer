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

fn adapter_dest(config: &Config, name: &str) -> Result<PathBuf, String> {
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
    let path = config.runtime_root.join("adapters").join(leaf);
    if path.exists() {
        return Err(format!(
            "A LoRA named {name} already exists. Choose another name."
        ));
    }
    Ok(path)
}

struct Scratch(PathBuf);
impl Scratch {
    fn new(config: &Config, kind: &str) -> Result<Self, String> {
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
) -> Result<PathBuf, String> {
    let dest = adapter_dest(config, name)?;
    let stage = Scratch::new(config, "adapter")?;
    if config.service == Service::Sa3 {
        let file = stage.0.join("adapter.gguf");
        fs::copy(source, &file).map_err(|e| e.to_string())?;
        crate::sa3_gguf::creative_adapter(&file)?;
        fs::rename(&file, &dest).map_err(|e| e.to_string())?;
    } else {
        let source = if source.join("checkpoint.current").is_file() {
            carey::resolve_checkpoint(source)?
        } else {
            source.to_owned()
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
        return publish(config, name, source, family, &[]);
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
    publish(config, name, &output, family, &[])
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
    pub progress: PathBuf,
    pub log: PathBuf,
    last_progress: Option<(String, u32, u32)>,
    _lease: Lease,
}

impl Job {
    pub fn cancel(&self) -> Result<(), String> {
        fs::write(self.directory.join("cancel.requested"), b"cancel\n").map_err(|e| e.to_string())
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
            )
            .map_err(|e| format!("{e}\nTraining output is in {}", self.output.display()))?;
            return Ok(Some(format!(
                "Training complete. LoRA added at {}",
                dest.display()
            )));
        }
        if let Ok(value) = sa3::read_progress(&self.progress) {
            let key = (value.phase.clone(), value.step, value.epoch);
            if self.last_progress.as_ref() != Some(&key) {
                println!(
                    "{} training · {} · step {}/{} · epoch {}",
                    self.config.service.id(),
                    value.phase,
                    value.step,
                    value.max_steps,
                    value.epoch
                );
                self.last_progress = Some(key);
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
    let mut captions = Vec::new();
    let mut command = tool(config, executable)?;
    match settings {
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
        }
        Settings::Carey(options) => {
            let staged = directory.join("dataset");
            let data =
                crate::carey_dataset::stage(&options.dataset, &staged, &options.trigger).await?;
            captions = data.captions;
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
        }
    }
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
    let fallback = component(Service::Carey, "carey-native::lm-1.7B::Q8_0")
        .and_then(|item| item.files.first());
    let lm = installed_carey_lms(&config.models_dir)
        .find(|file| file.filename.contains("1.7B"))
        .or_else(|| installed_carey_lms(&config.models_dir).next())
        .or(fallback)
        .ok_or("Carey's caption catalog has no LM")?;
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
    let log = fs::File::create(config.runtime_root.join("logs").join("caption.log"))
        .map_err(|e| e.to_string())?;
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
                "Carey caption server exited with {exit}. See logs/caption.log."
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
            return Err("Carey caption server startup timed out. See logs/caption.log.".into());
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
    if service == Service::Sa3 {
        carey_caption::label_sa3_dataset(
            &endpoint,
            &options,
            files,
            false,
            &scratch.0,
            display,
            || false,
        )
        .await?;
    } else {
        carey_caption::caption_dataset(&endpoint, &options, &scratch.0, display, || false).await?;
    }
    child.stop();
    Ok(())
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
