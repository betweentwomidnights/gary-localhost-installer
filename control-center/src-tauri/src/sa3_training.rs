//! Native training jobs share Gary's status contract and process ownership.
//! The native CLI owns progress, cancellation boundaries and checkpoint pairs.
use crate::{sa3_models, Sa3LoraTrainingState};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

pub static LAUNCH: Mutex<()> = Mutex::const_new(());
static NEXT_JOB: AtomicU64 = AtomicU64::new(0);

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Checkpoint {
    pub job_id: String,
    pub step: u32,
    pub epoch: Option<u32>,
    pub path: String,
    pub state_path: String,
}

fn runs(root: &Path, name: &str) -> Result<Vec<(PathBuf, Options)>, String> {
    let jobs = root.join("sa3/training/jobs");
    if !jobs.exists() {
        return Ok(Vec::new());
    }
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let jobs = jobs.canonicalize().map_err(|error| error.to_string())?;
    if !jobs.starts_with(&boundary) {
        return Err("Native job history escapes runtime storage".into());
    }
    let mut result = Vec::new();
    for job in std::fs::read_dir(&jobs).map_err(|error| error.to_string())? {
        let job = job.map_err(|error| error.to_string())?;
        if !job.file_type().map_err(|error| error.to_string())?.is_dir() {
            continue;
        }
        let path = job
            .path()
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if path.parent() != Some(jobs.as_path()) {
            return Err("Native job directory escapes history storage".into());
        }
        let options_path = path.join("native-options.json");
        if !options_path.is_file() {
            continue;
        }
        if options_path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(path.as_path())
        {
            return Err("Native job settings escape their job directory".into());
        }
        let options: Options = serde_json::from_slice(
            &std::fs::read(options_path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if options.name == name {
            let run = path.join("run");
            if run.is_dir() {
                let run = run.canonicalize().map_err(|error| error.to_string())?;
                if run.parent() != Some(path.as_path()) {
                    return Err("Native run escapes its job directory".into());
                }
                result.push((run, options));
            }
        }
    }
    Ok(result)
}

pub fn dataset_for_checkpoint(
    root: &Path,
    name: &str,
    source: &Path,
) -> Result<Option<PathBuf>, String> {
    let source = source.canonicalize().map_err(|error| error.to_string())?;
    Ok(runs(root, name)?
        .into_iter()
        .find(|(run, _)| source.parent() == Some(run.as_path()))
        .map(|(_, options)| options.dataset))
}

pub fn checkpoints(root: &Path, name: &str) -> Result<Vec<Checkpoint>, String> {
    let mut result = Vec::new();
    for (run, _) in runs(root, name)? {
        let mut epochs = std::collections::BTreeMap::new();
        if let Ok(metrics) = std::fs::read_to_string(run.join("metrics.jsonl")) {
            for line in metrics.lines() {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                    if let (Some(step), Some(epoch)) =
                        (value["update"].as_u64(), value["epoch"].as_u64())
                    {
                        epochs.insert(step as u32, epoch as u32);
                    }
                }
            }
        }
        for file in std::fs::read_dir(&run).map_err(|error| error.to_string())? {
            let file = file.map_err(|error| error.to_string())?;
            let filename = file.file_name();
            let filename = filename.to_string_lossy();
            let Some(step) = filename
                .strip_prefix("adapter-step-")
                .and_then(|name| name.strip_suffix(".gguf"))
                .and_then(|step| step.parse::<u32>().ok())
            else {
                continue;
            };
            let state = run.join(format!("trainer-state-step-{step}.gguf"));
            if !file.path().is_file() || !state.is_file() {
                continue;
            }
            let adapter = file
                .path()
                .canonicalize()
                .map_err(|error| error.to_string())?;
            let state = state.canonicalize().map_err(|error| error.to_string())?;
            if adapter.parent() != Some(run.as_path()) || state.parent() != Some(run.as_path()) {
                return Err("Native checkpoint pair escapes its training run".into());
            }
            result.push(Checkpoint {
                job_id: run
                    .parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into(),
                step,
                epoch: epochs.get(&step).copied(),
                path: adapter.to_string_lossy().into(),
                state_path: state.to_string_lossy().into(),
            });
        }
    }
    result.sort_by(|a, b| a.step.cmp(&b.step).then(a.job_id.cmp(&b.job_id)));
    Ok(result)
}

pub async fn select_checkpoint(root: &Path, name: &str, requested: &Path) -> Result<(), String> {
    let requested = requested
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let known = checkpoints(root, name)?
        .iter()
        .any(|checkpoint| Path::new(&checkpoint.path) == requested);
    if !known {
        return Err(
            "Select a paired checkpoint from this native LoRA's managed training history".into(),
        );
    }
    crate::sa3_loras::register_trained(root, name, &requested).await?;
    Ok(())
}

pub fn resume_options(root: &Path, name: &str, requested: &Path) -> Result<Options, String> {
    let requested = requested
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !checkpoints(root, name)?
        .iter()
        .any(|checkpoint| Path::new(&checkpoint.path) == requested)
    {
        return Err(
            "Select a paired checkpoint from this native LoRA's managed training history".into(),
        );
    }
    runs(root, name)?
        .into_iter()
        .find(|(run, _)| requested.parent() == Some(run.as_path()))
        .map(|(_, options)| options)
        .ok_or_else(|| "Original native training settings are unavailable".into())
}

#[derive(Debug, Deserialize)]
pub struct Progress {
    pub schema_version: u32,
    pub status: String,
    pub phase: String,
    pub message: String,
    pub error: String,
    pub step: u32,
    pub max_steps: u32,
    pub final_adapter: String,
    pub adapter_checkpoint: String,
    #[serde(default)]
    pub state_checkpoint: String,
}

pub fn read_progress(path: &Path) -> Result<Progress, String> {
    let data = std::fs::read(path).map_err(|error| error.to_string())?;
    let progress: Progress = serde_json::from_slice(&data).map_err(|error| error.to_string())?;
    if progress.schema_version != 1 {
        return Err("Unsupported native trainer progress schema".into());
    }
    Ok(progress)
}

pub(crate) fn save(path: &Path, value: &impl Serialize) -> Result<(), String> {
    save_with_policy(path, value, true)
}

pub(crate) fn save_with_policy(
    path: &Path,
    value: &impl Serialize,
    overwrite: bool,
) -> Result<(), String> {
    use std::io::Write;
    let stage = path.with_extension(format!(
        "json-{}-{}.tmp",
        std::process::id(),
        NEXT_JOB.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stage)
        .map_err(|error| error.to_string())?;
    let result = (|| {
        file.write_all(&serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        drop(file);
        if overwrite {
            std::fs::rename(&stage, path).map_err(|error| error.to_string())
        } else {
            match publish_new(&stage, path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
                Err(error) => Err(error.to_string()),
            }
        }
    })();
    if stage.exists() {
        let _ = std::fs::remove_file(stage);
    }
    result
}

#[cfg(windows)]
fn publish_new(stage: &Path, dest: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{MoveFileExW, MOVEFILE_WRITE_THROUGH};
    let stage: Vec<u16> = stage.as_os_str().encode_wide().chain(Some(0)).collect();
    let dest: Vec<u16> = dest.as_os_str().encode_wide().chain(Some(0)).collect();
    // No REPLACE_EXISTING: preserve a curated pool created during caption scanning.
    // A rename also works on custom FAT/exFAT storage without hard-link support.
    if unsafe { MoveFileExW(stage.as_ptr(), dest.as_ptr(), MOVEFILE_WRITE_THROUGH) } != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(windows))]
fn publish_new(stage: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::hard_link(stage, dest)
}

pub(crate) fn checked_folder(root: &Path, folders: &[&str]) -> Result<PathBuf, String> {
    std::fs::create_dir_all(root).map_err(|error| error.to_string())?;
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let mut path = boundary.clone();
    for folder in folders {
        path.push(folder);
        std::fs::create_dir_all(&path).map_err(|error| error.to_string())?;
        path = path.canonicalize().map_err(|error| error.to_string())?;
        if !path.starts_with(&boundary) {
            return Err(
                "Native training storage points outside the selected runtime folder".into(),
            );
        }
    }
    Ok(path)
}

fn validate(options: &Options) -> Result<(), String> {
    if crate::sanitize_lora_name(&options.name).as_deref() != Some(options.name.as_str())
        || !options.dataset.is_dir()
        || options.steps == 0
        || options.steps > i32::MAX as u32
        || !(1..=256).contains(&options.rank)
        || !(1..=64).contains(&options.batch_size)
        || options.checkpoint_every == 0
        || options.checkpoint_every > i32::MAX as u32
        || !options.duration.is_finite()
        || !(0.1..=380.0).contains(&options.duration)
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.0
        || options.learning_rate > 1.0
        || !options.target_latent_rms.is_finite()
        || (options.target_latent_rms != 0.0 && !(0.5..=1.3).contains(&options.target_latent_rms))
        || !["F16", "Q4_K_M"].contains(&options.encoding.as_str())
    {
        return Err("Invalid native SA3 training settings or dataset folder".into());
    }
    crate::resolve_sa3_lora_layer_scope(&options.layer_scope)?;
    Ok(())
}

pub async fn probe(
    executable: &Path,
    runtime_path: Option<&std::ffi::OsStr>,
) -> Result<(), String> {
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
        return Err("Installed native trainer lacks progress and resumable cancellation support. Prepare a compatible sa3.cpp release before native training.".into());
    }
    Ok(())
}

pub struct Job {
    child: tokio::process::Child,
    root: PathBuf,
    status_path: PathBuf,
    progress_path: PathBuf,
    output: PathBuf,
    state: Sa3LoraTrainingState,
}

pub async fn start(
    root: &Path,
    executable: &Path,
    backend: &str,
    runtime_path: Option<&std::ffi::OsStr>,
    mut options: Options,
) -> Result<Job, String> {
    validate(&options)?;
    probe(executable, runtime_path).await?;
    let models = sa3_models::checked_models_dir(root)?;
    for id in [
        "sa3-native::text".into(),
        "sa3-native::medium-decoder".into(),
        format!("sa3-native::medium-base-{}", options.encoding),
    ] {
        let component = sa3_models::component(&id).unwrap();
        if !sa3_models::present(component, &models) {
            return Err(format!(
                "Prepare {} before native training.",
                component.label
            ));
        }
    }
    if let Some(resume) = &options.resume {
        let path = resume.canonicalize().map_err(|error| error.to_string())?;
        let jobs = root
            .join("sa3/training/jobs")
            .canonicalize()
            .map_err(|error| error.to_string())?;
        if !path.starts_with(&jobs) || !path.is_file() {
            return Err("Resume checkpoint must belong to a managed native training run".into());
        }
        let original: Options = serde_json::from_slice(
            &std::fs::read(
                path.parent()
                    .and_then(Path::parent)
                    .ok_or("Invalid resume path")?
                    .join("native-options.json"),
            )
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if original.name != options.name {
            return Err("Resume must use the original adapter name".into());
        }
        if original.fixed_prompt.trim() != options.fixed_prompt.trim() {
            return Err("Resume must keep the original fixed prompt".into());
        }
        options.prompt_config = original.prompt_config;
    }
    let jobs = checked_folder(root, &["sa3", "training", "jobs"])?;
    let id = format!(
        "native-{}-{}-{}",
        options.name,
        crate::now_epoch_seconds() as u64,
        NEXT_JOB.fetch_add(1, Ordering::Relaxed)
    );
    let job = jobs.join(&id);
    std::fs::create_dir(&job).map_err(|error| error.to_string())?;
    let output = job.join("run");
    let progress_path = job.join("native-progress.json");
    let status_path = job.join("status.json");
    let cancel_path = job.join("cancel.requested");
    let log_path = job.join("training.log");
    let cache = checked_folder(root, &["sa3", "training", "native-latents"])?;
    // Decoder precision and RMS are part of the native cache key; separate RMS
    // settings keep both reusable caches rather than invalidating one another.
    let decoder_hash = &sa3_models::component("sa3-native::medium-decoder")
        .unwrap()
        .files
        .iter()
        .find(|file| file.filename.contains("same-l"))
        .unwrap()
        .sha256;
    let leaf = format!(
        "medium-F32-{}-rms{}",
        &decoder_hash[..12],
        options.target_latent_rms
    );
    let cache = checked_folder(&cache, &[&leaf])?;
    let mut command = tokio::process::Command::new(executable);
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
        .arg(&models)
        .arg("--out")
        .arg(&output)
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
        .arg(&progress_path)
        .arg("--cancel-file")
        .arg(&cancel_path);
    let text = sa3_models::component("sa3-native::text").unwrap();
    let decoder = sa3_models::component("sa3-native::medium-decoder").unwrap();
    let base =
        sa3_models::component(&format!("sa3-native::medium-base-{}", options.encoding)).unwrap();
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
    if !options.fixed_prompt.trim().is_empty() {
        let config = options
            .prompt_config
            .clone()
            .unwrap_or_else(|| job.join("prompt-config.json"));
        if options.prompt_config.is_none() {
            save(
                &config,
                &json!({"prompt_config":{"use_tags":false,"use_paths":false,"use_fixed":true,"fixed_text":options.fixed_prompt.trim(),"balance_fixed":100,"balance_tags":0,"balance_paths":0,"trigger":""}}),
            )?;
        }
        command.arg("--prompt-config").arg(&config);
        options.prompt_config = Some(config);
    }
    save(&job.join("native-options.json"), &options)?;
    if let Some(resume) = &options.resume {
        command.arg("--resume").arg(resume);
    }
    let log = std::fs::File::create(&log_path).map_err(|error| error.to_string())?;
    command
        .current_dir(executable.parent().ok_or("Invalid native trainer path")?)
        .stdout(log.try_clone().map_err(|error| error.to_string())?)
        .stderr(log)
        .kill_on_drop(true);
    if let Some(path) = runtime_path {
        command.env("PATH", path);
    }
    crate::workload_job::configure_tokio_command(&mut command);
    let mut child = command.spawn().map_err(|error| error.to_string())?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    let state = Sa3LoraTrainingState {
        job_id: Some(id),
        name: Some(options.name),
        status: "starting".into(),
        phase: "loading".into(),
        message: "Launching native SA3 training.".into(),
        error: None,
        pid: child.id(),
        owner_pid: Some(std::process::id()),
        launcher_pid: child.id(),
        child_pid: None,
        run_dir: Some(output.to_string_lossy().into()),
        log_path: Some(log_path.to_string_lossy().into()),
        cancel_path: Some(cancel_path.to_string_lossy().into()),
        final_checkpoint_path: None,
        current_step: Some(0),
        max_steps: Some(options.steps),
        log_tail: String::new(),
        runtime: Some("sa3.cpp".into()),
        resume_checkpoint_path: None,
    };
    if let Err(error) = save(&status_path, &state).and_then(|_| {
        save(
            &root.join("sa3/training/current_job.json"),
            &json!({"jobId":state.job_id,"statusPath":status_path}),
        )
    }) {
        let _ = child.kill().await;
        return Err(error);
    }
    Ok(Job {
        child,
        root: root.to_path_buf(),
        status_path,
        progress_path,
        output,
        state,
    })
}

impl Job {
    pub fn status_path(&self) -> &Path {
        &self.status_path
    }
    pub fn state(&self) -> Sa3LoraTrainingState {
        self.state.clone()
    }
    fn update(&mut self) -> Result<(), String> {
        if self.progress_path.exists() {
            let progress = read_progress(&self.progress_path)?;
            self.state.current_step = Some(progress.step);
            self.state.max_steps = Some(progress.max_steps);
            self.state.phase = progress.phase;
            self.state.message = progress.message;
            // The process must exit and registration must finish before Gary
            // advertises a completed run or admits another native workload.
            self.state.status = "running".into();
        }
        if self
            .status_path
            .parent()
            .unwrap()
            .join("cancel.requested")
            .exists()
        {
            self.state.message =
                "Cancellation requested; waiting for a sample boundary and checkpoint save.".into();
        }
        save(&self.status_path, &self.state)
    }
    pub async fn monitor(self) -> Result<Sa3LoraTrainingState, String> {
        self.monitor_updates(|_| {}).await
    }
    async fn monitor_updates(
        mut self,
        mut changed: impl FnMut(&Sa3LoraTrainingState),
    ) -> Result<Sa3LoraTrainingState, String> {
        loop {
            match self.child.try_wait() {
                Ok(Some(exit)) => {
                    let terminal = read_progress(&self.progress_path)?;
                    self.state.current_step = Some(terminal.step);
                    self.state.status = terminal.status.clone();
                    self.state.phase = terminal.status;
                    if !terminal.adapter_checkpoint.is_empty()
                        && !terminal.state_checkpoint.is_empty()
                    {
                        let checkpoint = PathBuf::from(&terminal.adapter_checkpoint)
                            .canonicalize()
                            .map_err(|error| error.to_string())?;
                        let optimizer = PathBuf::from(&terminal.state_checkpoint)
                            .canonicalize()
                            .map_err(|error| error.to_string())?;
                        let boundary = self
                            .output
                            .canonicalize()
                            .map_err(|error| error.to_string())?;
                        if checkpoint.parent() != Some(boundary.as_path())
                            || optimizer.parent() != Some(boundary.as_path())
                            || !checkpoint.is_file()
                            || !optimizer.is_file()
                        {
                            return Err(
                                "Native trainer returned an unexpected checkpoint pair".into()
                            );
                        }
                        self.state.resume_checkpoint_path =
                            Some(checkpoint.to_string_lossy().into());
                    }
                    if !exit.success() || self.state.status == "failed" {
                        self.state.status = "failed".into();
                        self.state.phase = "failed".into();
                        let error = if terminal.error.is_empty() {
                            format!("Native trainer exited with {exit}")
                        } else {
                            terminal.error
                        };
                        self.state.message = error.clone();
                        self.state.error = Some(error);
                    } else if !["completed", "cancelled"].contains(&self.state.status.as_str()) {
                        return Err(
                            "Native trainer exited without a terminal progress state".into()
                        );
                    } else {
                        if self.state.status == "completed" && terminal.final_adapter.is_empty() {
                            return Err(
                                "Native trainer reported completion without a final adapter".into(),
                            );
                        }
                        if !terminal.final_adapter.is_empty() {
                            let final_path = PathBuf::from(&terminal.final_adapter)
                                .canonicalize()
                                .map_err(|error| error.to_string())?;
                            if final_path.parent()
                                != Some(
                                    self.output
                                        .canonicalize()
                                        .map_err(|error| error.to_string())?
                                        .as_path(),
                                )
                            {
                                return Err(
                                    "Native trainer returned an unexpected adapter path".into()
                                );
                            }
                            let registered = crate::sa3_loras::register_trained(
                                &self.root,
                                self.state.name.as_deref().unwrap(),
                                &final_path,
                            )
                            .await?;
                            self.state.final_checkpoint_path =
                                Some(registered.to_string_lossy().into());
                        }
                        self.state.message = if self.state.status == "cancelled" {
                            "Native training cancelled at a sample boundary."
                        } else {
                            "Native training completed and adapter registered."
                        }
                        .into();
                    }
                    self.state.pid = None;
                    self.state.owner_pid = None;
                    self.state.launcher_pid = None;
                    save(&self.status_path, &self.state)?;
                    return Ok(self.state);
                }
                Ok(None) => {
                    self.update()?;
                    changed(&self.state);
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                Err(error) => return Err(error.to_string()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn checkpoint_history_keeps_branches_and_selects_only_paired_owned_adapters() {
        let root = std::env::temp_dir().join(format!(
            "gary-native-history-{}",
            NEXT_JOB.fetch_add(1, Ordering::Relaxed)
        ));
        let dataset = root.join("dataset");
        std::fs::create_dir_all(&dataset).unwrap();
        std::fs::write(dataset.join("a.txt"), "neurofunk, 172 bpm, F minor").unwrap();
        let options = Options {
            name: "history".into(),
            dataset: dataset.clone(),
            fixed_prompt: String::new(),
            steps: 4,
            rank: 16,
            batch_size: 1,
            checkpoint_every: 1,
            duration: 47.0,
            learning_rate: 1e-4,
            target_latent_rms: 0.0,
            layer_scope: "transformer-core".into(),
            encoding: "F16".into(),
            resume: None,
            prompt_config: None,
        };
        let mut header = vec![0u8; 24];
        header[..4].copy_from_slice(b"GGUF");
        header[4..8].copy_from_slice(&3u32.to_le_bytes());
        header[8..16].copy_from_slice(&1u64.to_le_bytes());
        for job in ["branch-a", "branch-b"] {
            let path = checked_folder(&root, &["sa3", "training", "jobs", job]).unwrap();
            save(&path.join("native-options.json"), &options).unwrap();
            let run = checked_folder(&root, &["sa3", "training", "jobs", job, "run"]).unwrap();
            std::fs::write(run.join("adapter-step-2.gguf"), &header).unwrap();
            std::fs::write(run.join("trainer-state-step-2.gguf"), &header).unwrap();
            std::fs::write(run.join("adapter-step-3.gguf"), &header).unwrap();
            std::fs::write(run.join("metrics.jsonl"), "{\"epoch\":1,\"update\":2}\n").unwrap();
        }
        let history = checkpoints(&root, "history").unwrap();
        assert_eq!(
            history.len(),
            2,
            "unpaired adapters are excluded; resumed branches are retained"
        );
        assert_eq!(history[0].epoch, Some(1));
        assert_ne!(history[0].path, history[1].path);
        let restored = resume_options(&root, "history", Path::new(&history[0].path)).unwrap();
        assert_eq!(restored.dataset, options.dataset);
        assert_eq!(restored.duration, options.duration);
        assert_eq!(restored.encoding, options.encoding);
        assert!(resume_options(&root, "another-name", Path::new(&history[0].path)).is_err());
        select_checkpoint(&root, "history", Path::new(&history[0].path))
            .await
            .unwrap();
        let catalog = crate::sa3_loras::state(&root).unwrap();
        assert_eq!(catalog.entries[0].source_path, history[0].path);
        assert_eq!(catalog.entries[0].training_checkpoints.len(), 2);
        assert_eq!(
            catalog.entries[0].prompts_path.as_deref(),
            Some(dataset.to_str().unwrap())
        );
        let pool: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("sa3/prompts/history.json")).unwrap())
                .unwrap();
        assert_eq!(pool["dice"]["instrumental"], json!(["neurofunk"]));
        let owned = root.join("sa3/training/jobs/branch-a/run/adapter-step-3.gguf");
        assert!(select_checkpoint(&root, "history", &owned).await.is_err());
        assert!(
            select_checkpoint(&root, "another-name", Path::new(&history[0].path))
                .await
                .is_err()
        );
        std::fs::write(root.join("sa3/prompts/history.json"), b"curated").unwrap();
        select_checkpoint(&root, "history", Path::new(&history[1].path))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(root.join("sa3/prompts/history.json")).unwrap(),
            b"curated"
        );
        assert!(Path::new(&history[0].path).is_file());
        crate::remove_managed_path(&root, &std::env::temp_dir().canonicalize().unwrap()).unwrap();
    }

    #[tokio::test]
    #[ignore = "registers and selects checkpoints from an existing isolated real native training run"]
    async fn real_native_checkpoint_history_and_prompts() {
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_TRAIN_HISTORY_ROOT")
                .expect("isolated training root required"),
        );
        let entries = crate::sa3_loras::read_catalog(&root).unwrap();
        let entry = entries
            .values()
            .find(|entry| entry.name == "koan-native-validation")
            .unwrap();
        let original_source = PathBuf::from(&entry.source_path);
        let original_hash = crate::native_runtime::sha256_file(&original_source)
            .await
            .unwrap();
        crate::sa3_loras::register_trained(&root, &entry.name, &original_source)
            .await
            .unwrap();
        let history = checkpoints(&root, &entry.name).unwrap();
        assert!(
            history.len() >= 3,
            "cancelled and resumed checkpoints retained: {history:?}"
        );
        select_checkpoint(&root, &entry.name, Path::new(&history[0].path))
            .await
            .unwrap();
        assert_eq!(
            crate::sa3_loras::read_catalog(&root).unwrap()[&entry.name].source_path,
            history[0].path
        );
        crate::sa3_loras::register_trained(&root, &entry.name, &original_source)
            .await
            .unwrap();
        let prompts: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("sa3/prompts/koan-native-validation.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(prompts["source"]["files"], 42);
        assert!(!prompts["dice"]["instrumental"]
            .as_array()
            .unwrap()
            .is_empty());
        assert_eq!(
            crate::native_runtime::sha256_file(&original_source)
                .await
                .unwrap(),
            original_hash
        );
        println!("PASS real native checkpoint branches, selection, source preservation and caption prompt registration");
    }

    #[test]
    fn progress_schema_and_settings_are_validated() {
        let root =
            std::env::temp_dir().join(format!("gary-sa3-native-options-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut options = Options {
            name: "fixture".into(),
            dataset: root.clone(),
            fixed_prompt: String::new(),
            steps: 3,
            rank: 16,
            batch_size: 1,
            checkpoint_every: 1,
            duration: 47.0,
            learning_rate: 1e-4,
            target_latent_rms: 0.0,
            layer_scope: "transformer-core".into(),
            encoding: "F16".into(),
            resume: None,
            prompt_config: None,
        };
        validate(&options).unwrap();
        options.duration = f64::NAN;
        assert!(validate(&options).is_err());
        let path = root.join("progress.json");
        save(&path,&json!({"schema_version":2,"status":"running","phase":"training","message":"","error":"","step":1,"max_steps":3,"final_adapter":"","adapter_checkpoint":""})).unwrap();
        assert!(read_progress(&path).unwrap_err().contains("schema"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    #[ignore = "runs CUDA training on the user-selected dataset in isolated managed storage"]
    async fn real_native_training_cancel_resume_and_registration() {
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_TRAIN_SMOKE_ROOT")
                .expect("new isolated root required"),
        );
        let models = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_TRAIN_MODELS")
                .expect("existing model folder required"),
        );
        let exe = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_TRAIN_BINARY").expect("compatible trainer required"),
        );
        let dataset = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_TRAIN_DATASET").expect("dataset required"),
        );
        let encoding =
            std::env::var("GARY4LOCAL_SA3_TRAIN_ENCODING").unwrap_or_else(|_| "F16".into());
        let duration = std::env::var("GARY4LOCAL_SA3_TRAIN_DURATION")
            .ok()
            .map(|value| value.parse::<f64>().unwrap())
            .unwrap_or(47.0);
        assert!(
            !root.exists(),
            "test must not overwrite an existing storage root"
        );
        let target = sa3_models::checked_models_dir(&root).unwrap();
        for id in [
            "sa3-native::text".into(),
            "sa3-native::medium-decoder".into(),
            format!("sa3-native::medium-base-{encoding}"),
        ] {
            for file in &sa3_models::component(&id).unwrap().files {
                std::fs::hard_link(models.join(&file.filename), target.join(&file.filename))
                    .unwrap();
            }
        }
        if let Some(source) =
            std::env::var_os("GARY4LOCAL_SA3_TRAIN_CACHE_SOURCE").map(PathBuf::from)
        {
            let decoder_hash = &sa3_models::component("sa3-native::medium-decoder")
                .unwrap()
                .files
                .iter()
                .find(|file| file.filename.contains("same-l"))
                .unwrap()
                .sha256;
            let leaf = format!("medium-F32-{}-rms0", &decoder_hash[..12]);
            let cache =
                checked_folder(&root, &["sa3", "training", "native-latents", &leaf]).unwrap();
            for file in std::fs::read_dir(source).unwrap() {
                let file = file.unwrap();
                assert!(file.file_type().unwrap().is_file());
                std::fs::copy(file.path(), cache.join(file.file_name())).unwrap();
            }
        }
        let mut originals = Vec::new();
        for file in std::fs::read_dir(&dataset).unwrap() {
            let path = file.unwrap().path();
            if path.is_file() {
                originals.push((
                    path.clone(),
                    crate::native_runtime::sha256_file(&path).await.unwrap(),
                ));
            }
        }
        let options = Options {
            name: "koan-native-validation".into(),
            dataset: dataset.clone(),
            fixed_prompt: std::env::var("GARY4LOCAL_SA3_TRAIN_FIXED_PROMPT").unwrap_or_default(),
            steps: 8,
            rank: 16,
            batch_size: 1,
            checkpoint_every: 1,
            duration,
            learning_rate: 1e-4,
            target_latent_rms: 0.0,
            layer_scope: "transformer-core".into(),
            encoding,
            resume: None,
            prompt_config: None,
        };
        let job = start(&root, &exe, "cuda", None, options.clone())
            .await
            .unwrap();
        let cancel = PathBuf::from(job.state.cancel_path.as_ref().unwrap());
        let mut requested = false;
        let first = job
            .monitor_updates(|state| {
                if state.current_step.unwrap_or(0) >= 1 && !requested {
                    std::fs::write(&cancel, "stop").unwrap();
                    requested = true;
                    println!(
                        "Requested native cancellation after step {}",
                        state.current_step.unwrap()
                    );
                }
            })
            .await
            .unwrap();
        assert!(requested);
        assert_eq!(first.status, "cancelled", "{:?}", first);
        assert!(Path::new(first.final_checkpoint_path.as_ref().unwrap()).is_file());
        let checkpoint = PathBuf::from(first.resume_checkpoint_path.as_ref().unwrap());
        let checkpoint_hash = crate::native_runtime::sha256_file(&checkpoint)
            .await
            .unwrap();
        let mut resumed = options;
        resumed.steps = first.current_step.unwrap() + 2;
        resumed.resume = Some(checkpoint.clone());
        let job = start(&root, &exe, "cuda", None, resumed.clone())
            .await
            .unwrap();
        let completed = job.monitor().await.unwrap();
        assert_eq!(completed.status, "completed", "{:?}", completed);
        assert_eq!(completed.current_step, Some(resumed.steps));
        assert_eq!(
            crate::native_runtime::sha256_file(&checkpoint)
                .await
                .unwrap(),
            checkpoint_hash,
            "resume must preserve its immutable source checkpoint"
        );
        let catalog = crate::sa3_loras::state(&root).unwrap();
        assert_eq!(catalog.entries.len(), 1);
        assert_eq!(
            catalog.entries[0].native_path,
            completed.final_checkpoint_path
        );
        for (path, hash) in originals {
            assert_eq!(
                crate::native_runtime::sha256_file(&path).await.unwrap(),
                hash,
                "dataset file changed: {}",
                path.display()
            );
        }
        assert!(
            !dataset.join("latents").exists(),
            "managed pre-encode must not write a cache into the dataset"
        );
        println!("PASS native GPU cancellation at step {:?}, resume to {:?}, immutable checkpoints, dataset preservation and native registration",first.current_step,completed.current_step);
    }
}
