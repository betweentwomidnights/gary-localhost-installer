//! Small line-oriented human console over the same service supervisors as the
//! machine CLI. A console owns only the tasks it starts, never stale disk PIDs.
use crate::service::Service;
use crate::{
    cli, command,
    headless::{self, Config},
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub const HELP: &str = "Gary — local GPU services\n\n\
  gary                         Open the interactive service console\n\
  gary --config FILE            Open a console with a specific config\n\
  gary start [service]              Run the service in the foreground; Ctrl+C stops it\n\
  gary status [service]             Show service status\n\
  gary stop [service]               Stop the selected service session\n\
  gary doctor                  Inspect setup without running GPU work\n\
  gary models                  Show the selected weights, size and storage folder\n\
  gary download-models         Download and verify the selected managed weights\n\
  gary install                 Explicitly install the pinned GPU runtime\n\n\
  Add --config FILE to any service command. Otherwise use the saved default.\n\
  GARY4LOCAL_HEADLESS_CONFIG overrides the default config path.\n\
  Add --json for the noninteractive agent protocol. Advanced commands:\n\
  init, validate, capabilities, schema, serve, version (see COMMAND --help).\n\n\
  Console: /setup /models /install /start /stop /status /doctor /logs /help /quit\n\
  Use /start all and /stop all inside the console.\n\
  Services: sa3, gary, melodyflow, stable-audio, carey, foundation, yuey. LoRA tools: /add lora and /train lora for SA3 and Carey.\n\
  Leaving a console stops only services started by that console.\n\
  No runtime/model downloads happen just by opening Gary.";

fn say(text: impl std::fmt::Display) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{text}")
        .and_then(|_| out.flush())
        .map_err(|e| e.to_string())
}
fn prompt(text: &str) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    write!(out, "{text}")
        .and_then(|_| out.flush())
        .map_err(|e| e.to_string())
}

pub fn default_config() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("GARY4LOCAL_HEADLESS_CONFIG") {
        let path = PathBuf::from(path);
        return if path.is_absolute() {
            Ok(path)
        } else {
            Err("GARY4LOCAL_HEADLESS_CONFIG must be an absolute path.".into())
        };
    }
    standard_config()
}

pub(crate) fn standard_config() -> Result<PathBuf, String> {
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base =
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Application Support"));
    #[cfg(not(any(windows, target_os = "macos")))]
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")));
    base.filter(|p| p.is_absolute())
        .map(|p| p.join("gary4local/headless/sa3.json"))
        .ok_or_else(|| "Cannot find a config directory. Supply --config FILE.".into())
}

fn session_args(args: &[OsString]) -> Option<Result<Option<PathBuf>, String>> {
    let mut args = args;
    if args.first().is_some_and(|arg| arg == "shell") {
        args = &args[1..];
    }
    if args.is_empty() {
        return Some(Ok(None));
    }
    if args.first().is_some_and(|arg| arg == "--config") {
        return Some(if args.len() == 2 {
            Ok(Some(PathBuf::from(&args[1])))
        } else {
            Err("Use gary --config FILE, or gary COMMAND --config FILE.".into())
        });
    }
    None
}

pub async fn entry(mut args: Vec<OsString>) -> Result<i32, String> {
    if args == [OsString::from("--help")]
        || args == [OsString::from("-h")]
        || args == [OsString::from("help")]
    {
        say(HELP)?;
        return Ok(0);
    }
    if args == [OsString::from("status")] {
        let (events, _) = mpsc::unbounded_channel();
        Console::new(default_config()?, events).status(None).await?;
        return Ok(0);
    }
    if let Some(config) = session_args(&args) {
        let config = config?;
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            say(HELP)?;
            if args.is_empty() {
                return Ok(0);
            }
            return Err("The interactive console needs a terminal. Use gary status, start, stop or --json from scripts.".into());
        }
        let config = config.map(Ok).unwrap_or_else(default_config)?;
        return shell(std::path::absolute(config).map_err(|e| e.to_string())?).await;
    }
    let position = args.iter().position(|arg| {
        matches!(
            arg.to_str(),
            Some(
                "start"
                    | "status"
                    | "stop"
                    | "doctor"
                    | "models"
                    | "download-models"
                    | "install"
                    | "init"
                    | "validate"
                    | "serve"
            )
        )
    });
    if let Some(index) = position {
        let mut service = match args.windows(2).find(|pair| pair[0] == "--service") {
            Some(pair) => Service::parse(&pair[1].to_string_lossy())?,
            None => Service::Sa3,
        };
        if args
            .get(index + 1)
            .is_some_and(|arg| !arg.to_string_lossy().starts_with('-'))
        {
            service = Service::parse(&args.remove(index + 1).to_string_lossy())?;
            args.extend(["--service".into(), service.id().into()]);
        }
        if args[index] == "start" {
            args[index] = "serve".into();
        }
        if !args.iter().any(|arg| arg == "--config")
            && !args.iter().any(|arg| arg == "--help" || arg == "-h")
        {
            let base = default_config()?;
            if service != Service::Sa3
                && args[index] == "init"
                && !args.iter().any(|arg| arg == "--runtime-root")
            {
                args.extend([
                    "--runtime-root".into(),
                    companion_root(&base, service)?.into_os_string(),
                ]);
            }
            args.extend([
                "--config".into(),
                profile_path(&base, service).into_os_string(),
            ]);
        }
    }
    Ok(command::run(args, true).await)
}

fn profile_path(base: &Path, service: Service) -> PathBuf {
    if service == Service::Sa3 {
        return base.to_owned();
    }
    let mut group = base.file_name().unwrap_or_default().to_os_string();
    group.push(".services");
    base.with_file_name(group)
        .join(format!("{}.json", service.id()))
}

fn companion_root(base: &Path, service: Service) -> Result<PathBuf, String> {
    let root = if base.exists() {
        Config::read(base)?.runtime_root
    } else {
        cli::init_root(base)
    };
    let mut group = root.into_os_string();
    group.push(".services");
    Ok(PathBuf::from(group).join(service.id()))
}

#[derive(Debug, PartialEq)]
enum Action {
    Setup,
    Models,
    Install,
    Start,
    StartAll,
    Stop,
    StopAll,
    Status,
    Doctor,
    Logs,
    Help,
    Quit,
    AddLora,
    TrainLora,
    Training,
    CancelTraining,
}
fn action(line: &str) -> Result<Option<(Action, Option<Service>)>, String> {
    let words: Vec<_> = line.split_whitespace().collect();
    if words.is_empty() {
        return Ok(None);
    }
    let name = words[0].trim_start_matches('/').to_ascii_lowercase();
    if name == "cancel" && words.len() == 2 && words[1].eq_ignore_ascii_case("training") {
        return Ok(Some((Action::CancelTraining, None)));
    }
    if matches!(name.as_str(), "add" | "train")
        && words
            .get(1)
            .is_some_and(|word| word.eq_ignore_ascii_case("lora"))
        && words.len() <= 3
    {
        let service = words.get(2).map(|id| Service::parse(id)).transpose()?;
        if service.is_some_and(|s| !matches!(s, Service::Sa3 | Service::Carey)) {
            return Err("LoRA tools are available for SA3 and Carey.".into());
        }
        return Ok(Some((
            if name == "add" {
                Action::AddLora
            } else {
                Action::TrainLora
            },
            service,
        )));
    }
    let service_action = matches!(
        name.as_str(),
        "start" | "stop" | "status" | "services" | "models" | "install" | "doctor" | "logs"
    );
    if words.len() > 1 && !(service_action && words.len() == 2) {
        return Err("Use a command on its own or with sa3, for example /start sa3. Type /help for commands.".into());
    }
    if words
        .get(1)
        .is_some_and(|id| id.eq_ignore_ascii_case("all"))
    {
        return match name.as_str() {
            "start" => Ok(Some((Action::StartAll, None))),
            "stop" => Ok(Some((Action::StopAll, None))),
            _ => Err("Use all with /start or /stop.".into()),
        };
    }
    let service = words.get(1).map(|id| Service::parse(id)).transpose()?;
    Ok(Some((
        match name.as_str() {
            "models" => Action::Models,
            "setup" => Action::Setup,
            "install" => Action::Install,
            "start" => Action::Start,
            "stop" => Action::Stop,
            "status" | "services" => Action::Status,
            "doctor" => Action::Doctor,
            "logs" => Action::Logs,
            "training" => Action::Training,
            "help" | "?" => Action::Help,
            "quit" | "exit" => Action::Quit,
            _ => {
                return Err(format!(
                    "Unknown command '{}'. Type /help for commands.",
                    words[0]
                ))
            }
        },
        service,
    )))
}
#[path = "console_input.rs"]
mod console_input;
use console_input::Input;
#[path = "console_training.rs"]
mod console_training;

async fn answer(input: &mut Input, question: &str) -> Result<String, String> {
    prompt(question)?;
    let answer = input
        .recv()
        .await
        .ok_or("Setup cancelled (end of input).")??;
    let answer = answer.trim().to_string();
    if matches!(answer.as_str(), "/cancel" | "/quit" | "/exit") {
        return Err("Setup cancelled; no config was saved.".into());
    }
    Ok(answer)
}
fn model_path(text: &str) -> PathBuf {
    let text = text.trim();
    let text = if text.len() >= 2
        && ((text.starts_with('"') && text.ends_with('"'))
            || (text.starts_with('\'') && text.ends_with('\'')))
    {
        &text[1..text.len() - 1]
    } else {
        text
    };
    if text == "~" || text.starts_with("~/") || text.starts_with("~\\") {
        if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
            return PathBuf::from(home).join(text.get(2..).unwrap_or(""));
        }
    }
    PathBuf::from(text)
}
async fn setup(path: &Path, input: &mut Input) -> Result<(), String> {
    if path.exists() {
        return Err(format!("Config already exists: {}\nSetup will not overwrite it. Use /doctor to inspect it or edit the file deliberately.",path.display()));
    }
    say("Set up SA3 — /cancel leaves setup without saving.\nPress Enter to let Gary manage your SA3 weights. You can download them after setup.")?;
    let models = loop {
        let text = answer(
            input,
            "Press Enter for managed storage, or enter a folder of existing SA3 models: ",
        )
        .await?;
        if text.is_empty() {
            break None;
        }
        let path = model_path(&text);
        if !text.is_empty() && path.is_dir() {
            break Some(path);
        }
        say("That folder does not exist. Enter a folder path, or /cancel.")?;
    };
    let mut root = cli::init_root(path);
    if models.is_none() {
        say(format!(
            "Gary will keep the runtime and weights together in {}",
            root.display()
        ))?;
        let location = answer(input, "Storage folder [Enter for default]: ").await?;
        if !location.is_empty() {
            root = model_path(&location);
            if !root.is_absolute() {
                return Err("Choose an absolute storage path, then run /setup again.".into());
            }
        }
    }
    let port = loop {
        let value = answer(input, "Client port [28300]: ").await?;
        let value = if value.is_empty() {
            "28300".to_string()
        } else {
            value
        };
        if value.parse::<u16>().is_ok_and(|p| p > 0) {
            break value;
        }
        say("Choose a port from 1 to 65535.")?;
    };
    let device = loop {
        let value = answer(input, "GPU device [Vulkan0]: ").await?;
        let value = if value.is_empty() {
            "Vulkan0".to_string()
        } else {
            value
        };
        if (value.starts_with("Vulkan") && value[6..].parse::<u32>().is_ok())
            || (value.starts_with("CUDA") && value[4..].parse::<u32>().is_ok())
        {
            break value;
        }
        say("Use an exact GPU name such as Vulkan0 or CUDA0. Linux published previews currently provide Vulkan.")?;
    };
    #[cfg(target_os = "linux")]
    let encoding = {
        let choices = ["f16", "q8_0", "q5_k_m", "q4_k_m"];
        choices[input.select("Generation precision", &choices).await?].to_owned()
    };
    #[cfg(not(target_os = "linux"))]
    let encoding = loop {
        let value = answer(
            input,
            "Generation precision: f16, q8_0, q5_k_m, q4_k_m [f16]: ",
        )
        .await?
        .to_ascii_lowercase();
        let value = if value.is_empty() {
            "f16".to_string()
        } else {
            value
        };
        if ["f16", "q8_0", "q5_k_m", "q4_k_m"].contains(&value.as_str()) {
            break value;
        }
        say("Choose one of the listed inference precisions.")?;
    };
    let mut args: Vec<OsString> = vec![
        "init".into(),
        "--config".into(),
        path.into(),
        "--runtime-root".into(),
        root.into_os_string(),
        "--port".into(),
        port.into(),
        "--device".into(),
        device.into(),
        "--encoding".into(),
        encoding.into(),
    ];
    if let Some(models) = models {
        args.extend(["--models-dir".into(), models.into_os_string()]);
    }
    let request = cli::Request::parse(&args).map_err(|e| e.message)?;
    say(format!(
        "Save config to {}? No runtime or model downloads will start.",
        path.display()
    ))?;
    if !matches!(
        answer(input, "Save [y/N]: ")
            .await?
            .to_ascii_lowercase()
            .as_str(),
        "y" | "yes"
    ) {
        return say("Setup cancelled; no config was saved.");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let result = cli::execute(&request, |_| Ok(()))
        .await
        .map_err(|e| format!("{}\n{}", e.message, e.hint))?;
    command::human_output("init", &result.data)?;
    say("Next: /models to prepare weights, /install for the runtime, then /start.")
}

fn service_label(service: Service) -> &'static str {
    if service == Service::Sa3 {
        "SA3"
    } else {
        service.id()
    }
}

fn start_blockers(report: &Value, service: Service) -> String {
    let id = service.id();
    let mut lines: Vec<String> = report["checks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|check| check["status"] == "error")
        .map(|check| match check["code"].as_str() {
            Some("installed_runtime") => format!("Runtime missing or invalid. Run /install {id}."),
            Some("models_directory" | "model_files") => {
                format!("Required weights are missing or incomplete. See /models {id}.")
            }
            _ => check["message"]
                .as_str()
                .unwrap_or("A startup check failed.")
                .to_owned(),
        })
        .collect();
    lines.push(format!("Run /doctor {id} for details."));
    lines.join("\n")
}

fn model_size(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0} MiB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{bytes} bytes")
    }
}

async fn choose_row(
    input: &mut Input,
    title: &str,
    rows: &[String],
    initial: usize,
) -> Result<usize, String> {
    if rows.len() == 1 {
        return Ok(0);
    }
    #[cfg(target_os = "linux")]
    {
        let labels: Vec<_> = rows.iter().map(String::as_str).collect();
        input.select_at(title, &labels, initial).await
    }
    #[cfg(not(target_os = "linux"))]
    {
        say(title)?;
        for (index, row) in rows.iter().enumerate() {
            say(format!("{}  {row}", index + 1))?;
        }
        let selected = answer(input, &format!("Select a row [{}]: ", initial + 1)).await?;
        if selected.is_empty() {
            return Ok(initial);
        }
        selected
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .filter(|&n| n < rows.len())
            .ok_or_else(|| "Choose a listed row.".into())
    }
}

fn checkpoint_label(service: Service, model: &str) -> String {
    if service != Service::Carey {
        return model.to_owned();
    }
    match model.strip_prefix("xl-") {
        Some(name) => format!("XL {name}"),
        None => format!("2B {model}"),
    }
}

struct Running {
    id: u64,
    config: Config,
    session: Option<String>,
    task: JoinHandle<()>,
}
impl Drop for Running {
    fn drop(&mut self) {
        self.task.abort();
    }
}
enum Event {
    Ready(Service, u64, Value),
    Finished(Service, u64, Result<Value, String>),
}

struct Console {
    path: PathBuf,
    owned: HashMap<Service, Running>,
    serial: u64,
    events: mpsc::UnboundedSender<Event>,
    training: Option<crate::training_job::Job>,
}
impl Console {
    fn new(path: PathBuf, events: mpsc::UnboundedSender<Event>) -> Self {
        Self {
            path,
            owned: HashMap::new(),
            serial: 0,
            events,
            training: None,
        }
    }
    fn config(&self, service: Service) -> Result<Config, String> {
        let path = profile_path(&self.path, service);
        let config = Config::read(&path)
            .map_err(|e| format!("{e}\nUse /install {} to prepare the service.", service.id()))?;
        if config.service != service {
            return Err(format!("{} belongs to a different service", path.display()));
        }
        Ok(config)
    }
    async fn status(&self, only: Option<Service>) -> Result<(), String> {
        say("SERVICE       STATE             ADDRESS")?;
        for service in Service::ALL {
            if only.is_some_and(|id| id != service) {
                continue;
            }
            let id = service.id();
            let path = profile_path(&self.path, service);
            if !self.owned.contains_key(&service) && !path.exists() {
                say(format!("{id:<14}not configured"))?;
                continue;
            }
            match self.state(service).await {
                Ok(s) => {
                    let state = if s["running"] == true {
                        "running"
                    } else if s["operation_active"] == true {
                        "busy / starting"
                    } else {
                        "stopped"
                    };
                    let url = if s["running"] == true {
                        s["public_url"].as_str().unwrap_or("")
                    } else {
                        ""
                    };
                    say(format!("{id:<14}{state:<18}{url}"))?;
                    if s["running"] == true && !self.owned.contains_key(&service) {
                        say("Running in another session. /quit will leave it running.")?;
                    }
                }
                Err(e) => say(format!("{id:<14}{e}"))?,
            }
        }
        if !self.path.exists() {
            say("Use /setup for SA3 or /install SERVICE to prepare a service.")?;
        }
        Ok(())
    }
    async fn state(&self, service: Service) -> Result<Value, String> {
        let config = if let Some(host) = self.owned.get(&service) {
            host.config.clone()
        } else {
            self.config(service)?
        };
        let mut state = headless::status(&config).await?;
        if self.owned.contains_key(&service) {
            state["operation_active"] = json!(true);
        }
        Ok(state)
    }
    async fn stop_owned(&mut self, service: Service) -> Result<(), String> {
        if let Some(mut host) = self.owned.remove(&service) {
            let result = if let Some(session) = &host.session {
                headless::stop(&host.config, Some(session))
                    .await
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            } else {
                Ok(())
            };
            host.task.abort();
            let _ = (&mut host.task).await;
            result?;
        }
        Ok(())
    }
    async fn choose(&self, action: &Action, input: &mut Input) -> Result<Service, String> {
        let mut choices = Vec::new();
        for service in Service::ALL {
            if matches!(action, Action::AddLora | Action::TrainLora)
                && !matches!(service, Service::Sa3 | Service::Carey)
            {
                continue;
            }
            if matches!(action, Action::Stop | Action::Start) {
                if !self.owned.contains_key(&service) && !profile_path(&self.path, service).exists()
                {
                    continue;
                }
                let active = self.state(service).await?["operation_active"] == true;
                if active != matches!(action, Action::Stop) {
                    continue;
                }
            }
            choices.push(service);
        }
        match choices.as_slice() {
            [] => {
                return Err(if matches!(action, Action::Stop) {
                    "No services are running."
                } else {
                    "No stopped service is configured. Use /install SERVICE."
                }
                .into())
            }
            [one] => return Ok(*one),
            _ => {}
        }
        let names: Vec<_> = choices.iter().map(|s| s.id()).collect();
        #[cfg(target_os = "linux")]
        {
            Ok(choices[input.select("Choose a service", &names).await?])
        }
        #[cfg(not(target_os = "linux"))]
        {
            let choice =
                answer(input, &format!("Choose a service ({}): ", names.join(", "))).await?;
            Service::parse(&choice)
        }
    }
    async fn ensure_profile(&self, service: Service) -> Result<(), String> {
        let path = profile_path(&self.path, service);
        if path.exists() {
            return Ok(());
        }
        let mut args = vec![
            "init".into(),
            "--service".into(),
            service.id().into(),
            "--config".into(),
            path.into_os_string(),
        ];
        if service != Service::Sa3 {
            // Keep companion services beside the selected SA3 storage, each with its own lease.
            args.extend([
                "--runtime-root".into(),
                companion_root(&self.path, service)?.into_os_string(),
            ]);
        }
        let request = cli::Request::parse(&args).map_err(|e| e.message)?;
        cli::execute(&request, |_| Ok(()))
            .await
            .map_err(|e| e.message)?;
        Ok(())
    }
    async fn pick_model(
        &self,
        config: &Config,
        input: &mut Input,
        start: bool,
    ) -> Result<Option<Config>, String> {
        let service = config.service;
        let label = service_label(service);
        let path = profile_path(&self.path, service);
        let original = std::fs::read(&path).map_err(|e| e.to_string())?;
        let current = Config::from_bytes(&original).map_err(|e| e.to_string())?;
        let variants = crate::model_catalog::variants(service);
        let mut checkpoints = Vec::new();
        for &(model, _) in &variants {
            if checkpoints.last() != Some(&model) {
                checkpoints.push(model);
            }
        }
        let choice_rows: Vec<_> = checkpoints
            .iter()
            .map(|&model| {
                let name = checkpoint_label(service, model);
                let selected = current
                    .selected_models
                    .get(model)
                    .map(String::as_str)
                    .or_else(|| (current.model == model).then_some(current.encoding.as_str()));
                match selected {
                    Some(tier) => format!("{name}  ·  {}", tier.to_ascii_uppercase()),
                    None => name,
                }
            })
            .collect();
        let checkpoint = checkpoints[choose_row(
            input,
            &format!("{label} model"),
            &choice_rows,
            checkpoints
                .iter()
                .position(|&m| m == current.model)
                .unwrap_or(0),
        )
        .await?];
        let recommended = current.recommended_tier(checkpoint);
        let mut options = Vec::new();
        let mut rows = Vec::new();
        let mut initial = 0;
        let mut installed_choice = None;
        let mut installed = Vec::new();
        for (_, tier) in variants
            .into_iter()
            .filter(|(model, _)| *model == checkpoint)
        {
            let mut candidate = current.clone();
            candidate.model = checkpoint.to_owned();
            candidate.encoding = tier.to_ascii_lowercase();
            candidate.selected_models.clear();
            let plan = match headless::model_plan(&candidate) {
                Ok(plan) => plan,
                Err(_) => continue,
            };
            let bytes = plan["missing_bytes"].as_u64().unwrap_or_default();
            if bytes == 0 {
                installed.push(options.len());
            }
            let selected = current
                .selected_models
                .get(checkpoint)
                .is_some_and(|saved| saved.eq_ignore_ascii_case(tier))
                || (current.model == checkpoint && current.encoding.eq_ignore_ascii_case(tier));
            let mut row = if bytes == 0 {
                format!("{tier:<8} Installed")
            } else {
                format!("{tier:<8} Download {}", model_size(bytes))
            };
            if selected {
                row.push_str(if bytes == 0 {
                    " · selected"
                } else {
                    " · configured"
                });
                if bytes == 0 {
                    installed_choice = Some(options.len());
                }
            }
            if Some(tier) == recommended {
                row.push_str(" · recommended");
                initial = options.len();
            }
            rows.push(row);
            options.push((candidate, bytes));
        }
        if rows.is_empty() {
            return Err("No published precisions match this model configuration.".into());
        }
        let index = choose_row(
            input,
            &format!("{} · precision", checkpoint_label(service, checkpoint)),
            &rows,
            installed_choice
                .or_else(|| match installed.as_slice() {
                    [only] => Some(*only),
                    _ => None,
                })
                .unwrap_or(initial),
        )
        .await?;
        let (candidate, bytes) = &options[index];
        if !start {
            let plan = headless::model_removal(&current, &candidate.model, &candidate.encoding)?;
            if !plan.is_empty() {
                let primary = if *bytes == 0 {
                    "Use this precision"
                } else {
                    "Download and use"
                };
                let choices = [
                    primary.to_owned(),
                    format!("Remove weights ({})", model_size(plan.bytes())),
                ];
                if choose_row(
                    input,
                    &format!(
                        "{label} {} {}",
                        checkpoint_label(service, checkpoint),
                        candidate.encoding.to_ascii_uppercase()
                    ),
                    &choices,
                    0,
                )
                .await?
                    == 1
                {
                    let confirm = answer(
                        input,
                        &format!(
                            "Remove {label} {} {} files totaling {}? [y/N] ",
                            checkpoint_label(service, checkpoint),
                            candidate.encoding.to_ascii_uppercase(),
                            model_size(plan.bytes())
                        ),
                    )
                    .await?;
                    if matches!(confirm.to_ascii_lowercase().as_str(), "y" | "yes") {
                        headless::remove_model(
                            &path,
                            &original,
                            &current,
                            &candidate.model,
                            &candidate.encoding,
                            &plan,
                        )?;
                        say("Model weights removed. Shared files are kept.")?;
                    } else {
                        say("Model removal cancelled.")?;
                    }
                    return Ok(None);
                }
            }
        }
        if *bytes > 0 {
            let action = if start {
                "Download and start"
            } else {
                "Download and use"
            };
            let question = format!(
                "{action} {label} {} ({}, {})? [y/N] ",
                checkpoint_label(service, checkpoint),
                candidate.encoding.to_ascii_uppercase(),
                model_size(*bytes)
            );
            let confirm = answer(input, &question).await?;
            if !matches!(confirm.to_ascii_lowercase().as_str(), "y" | "yes") {
                say("Model selection cancelled.")?;
                return Ok(None);
            }
        }
        let updated =
            headless::select_model(&path, &original, &current, candidate, *bytes > 0).await?;
        say(format!(
            "{label} will use {} at {}.",
            checkpoint_label(service, checkpoint),
            updated.encoding.to_ascii_uppercase()
        ))?;
        Ok(Some(updated))
    }

    async fn start_service(
        &mut self,
        service: Service,
        input: Option<&mut Input>,
    ) -> Result<(), String> {
        let id = service.id();
        let label = service_label(service);
        if self.state(service).await?["operation_active"] == true {
            return Err(format!("{label} is already active."));
        }
        let mut config = self.config(service)?;
        let report = headless::doctor(&config).await?;
        if report["can_attempt_serve"] != true {
            let models_only = report["checks"].as_array().is_some_and(|checks| {
                checks
                    .iter()
                    .filter(|check| check["status"] == "error")
                    .all(|check| {
                        matches!(
                            check["code"].as_str(),
                            Some("models_directory" | "model_files")
                        )
                    })
            });
            let Some(input) = input.filter(|_| models_only) else {
                return Err(start_blockers(&report, service));
            };
            if config.models_dir != config.managed_models_dir() {
                return Err(start_blockers(&report, service));
            }
            let Some(updated) = self.pick_model(&config, input, true).await? else {
                return Ok(());
            };
            config = updated;
            let report = headless::doctor(&config).await?;
            if report["can_attempt_serve"] != true {
                return Err(start_blockers(&report, service));
            }
        }
        self.serial += 1;
        let serial = self.serial;
        let tx = self.events.clone();
        let cfg = config.clone();
        let task = tokio::spawn(async move {
            let result = headless::serve(&cfg, headless::Sa3Startup::Cold, |value| {
                tx.send(Event::Ready(service, serial, value))
                    .map_err(|e| e.to_string())
            })
            .await;
            let _ = tx.send(Event::Finished(service, serial, result));
        });
        self.owned.insert(
            service,
            Running {
                id: serial,
                config,
                session: None,
                task,
            },
        );
        say(format!("Starting {label}... /stop {id} cancels startup."))
    }

    async fn stop_service(&mut self, service: Service) -> Result<bool, String> {
        if self.owned.contains_key(&service) {
            self.stop_owned(service).await?;
            return Ok(true);
        }
        let config = self.config(service)?;
        let state = headless::status(&config).await?;
        if state["operation_active"] != true {
            return Ok(false);
        }
        let session = state["state"]["session"]
            .as_str()
            .filter(|_| state["running"] == true)
            .ok_or("The service is busy in another session. Wait for startup or installation to finish.")?;
        headless::stop(&config, Some(session))
            .await
            .map_err(|e| e.to_string())?;
        Ok(true)
    }

    async fn handle_all(&mut self, action: Action) -> Result<(), String> {
        for service in Service::ALL {
            if !self.owned.contains_key(&service) && !profile_path(&self.path, service).exists() {
                continue;
            }
            let label = service_label(service);
            if action == Action::StartAll {
                if let Err(error) = self.start_service(service, None).await {
                    say(format!("{label} skipped: {error}"))?;
                }
            } else {
                match self.stop_service(service).await {
                    Ok(true) => say(format!("{label} stopped."))?,
                    Ok(false) => {}
                    Err(error) => say(format!("Could not stop {label}: {error}"))?,
                }
            }
        }
        self.status(None).await
    }

    async fn handle(
        &mut self,
        action: Action,
        explicit: Option<Service>,
        input: &mut Input,
    ) -> Result<bool, String> {
        match action {
            Action::Quit => return Ok(true),
            Action::Help => {
                say("/setup — Set up SA3\n/install [service] — Prepare a service and install its GPU runtime\n/models [service] — Inspect and download the selected weights\n/start [service|all] — Start one service or all configured services\n/stop [service|all] — Stop services or cancel startup\n/status — Show all services and addresses\n/doctor [service] — Inspect prerequisites\n/logs [service] — Show the native log path\n/add lora [sa3|carey] — Import an adapter\n/train lora [sa3|carey] — Prepare a dataset and train\n/training — Show training progress and log paths\n/cancel training — Request a checkpoint and stop training\n/quit — Stop services started here and leave")?;
                return Ok(false);
            }
            Action::StartAll | Action::StopAll => {
                self.handle_all(action).await?;
                return Ok(false);
            }
            Action::Training => {
                if let Some(job) = &self.training {
                    if let Some(progress) = job.progress_summary() {
                        say(progress)?;
                    }
                    say(format!(
                        "Progress {}\nLog {}",
                        job.progress.display(),
                        job.log.display()
                    ))?;
                } else {
                    say("No training job is running in this console.")?;
                }
                return Ok(false);
            }
            Action::CancelTraining => {
                if let Some(job) = &self.training {
                    job.cancel()?;
                    say("Cancellation requested.")?;
                } else {
                    say("No training job is running in this console.")?;
                }
                return Ok(false);
            }
            Action::Setup => {
                setup(&self.path, input).await?;
                return Ok(false);
            }
            Action::Status => {
                self.status(explicit).await?;
                return Ok(false);
            }
            _ => {}
        }
        let service = match explicit {
            Some(s) => s,
            None => self.choose(&action, input).await?,
        };
        let label = service_label(service);
        match action {
            Action::TrainLora => {
                self.train_lora(input, service).await?;
                return Ok(false);
            }
            Action::AddLora => {
                self.add_lora(input, service).await?;
                return Ok(false);
            }
            Action::Start => {
                self.start_service(service, Some(input)).await?;
                return Ok(false);
            }
            Action::Stop => {
                self.stop_service(service).await?;
                say(format!("{label} stopped."))?;
                self.status(None).await?;
                return Ok(false);
            }
            Action::Install | Action::Models => self.ensure_profile(service).await?,
            _ => {}
        }
        let config = if let Some(host) = self.owned.get(&service) {
            host.config.clone()
        } else {
            self.config(service)?
        };
        match action {
            Action::Logs => say(config.log_path().display())?,
            Action::Doctor => command::human_output("doctor", &headless::doctor(&config).await?)?,
            Action::Install => {
                say(format!("Installing {label} and checking the GPU..."))?;
                command::human_output("install", &headless::install(&config).await?)?;
            }
            Action::Models => {
                if config.models_dir != config.managed_models_dir() {
                    command::human_output("models", &headless::model_plan(&config)?)?;
                } else if self.state(service).await?["operation_active"] == true {
                    command::human_output("models", &headless::model_plan(&config)?)?;
                    for (model, encoding) in &config.selected_models {
                        say(format!(
                            "  {} · {}",
                            checkpoint_label(service, model),
                            encoding.to_ascii_uppercase()
                        ))?;
                    }
                    say(format!("Stop {label} to change or remove models."))?;
                } else {
                    self.pick_model(&config, input, false).await?;
                }
            }
            _ => unreachable!(),
        }
        Ok(false)
    }
}
async fn termination() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=term.recv()=>{} }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
async fn shell(path: PathBuf) -> Result<i32, String> {
    let (tx, mut ready) = mpsc::unbounded_channel();
    let mut console = Console::new(path, tx);
    let mut input = Input::new();
    let shutdown = termination();
    tokio::pin!(shutdown);
    // Register signal handlers before showing a prompt, including the first
    // prompt. Keep the same future alive through setup and startup.
    tokio::select! { biased; _=&mut shutdown=>return Ok(0), _=std::future::ready(())=>{} }
    say(format!("Gary {} — local GPU services\nConfig: {}\n\nType /help for commands. /quit or Ctrl+C stops services started here.\n",env!("CARGO_PKG_VERSION"),console.path.display()))?;
    if let Err(e) = console.status(None).await {
        say(format!("{e}\nType /help for commands."))?;
    }
    let mut training_tick = tokio::time::interval(std::time::Duration::from_secs(1));
    let mut show_prompt = true;
    loop {
        if show_prompt {
            prompt("\ngary> ")?;
        }
        show_prompt = true;
        tokio::select! {
            _=&mut shutdown=>break,
            line=input.recv()=>{
                let Some(line)=line else { break; };
                let action=line.and_then(|line|action(&line));
                match action {
                    Ok(None)=>{},
                    Err(e)=>say(e)?,
                    Ok(Some((action,explicit)))=>{
                        tokio::select! {
                            _=&mut shutdown=>break,
                            result=console.handle(action,explicit,&mut input)=>match result {
                                Ok(true)=>break,Ok(false)=>{},Err(e)=>say(e)?,
                            }
                        }
                    }
                }
            }
            _=training_tick.tick(), if console.training.is_some()=>{
                let job=console.training.as_mut().expect("active training job");
                match job.poll() {
                    Ok(None)=>{show_prompt=false;continue;},
                    Ok(Some(message))=>say(message)?,
                    Err(error)=>say(error)?,
                }
                console.training.take();
            }
            Some(event)=ready.recv()=>{
                match event {
                    Event::Ready(service,id,value)=>{
                        if let Some(host)=console.owned.get_mut(&service).filter(|h|h.id==id){
                            host.session=value["session"].as_str().map(str::to_owned);
                            let label = service_label(service);
                            say(format!("\n{label} ready at {}",value["public_url"].as_str().unwrap_or("")))?;
                            console.status(None).await?;
                        }
                    }
                    Event::Finished(service,id,result)=>{
                        if console.owned.get(&service).is_some_and(|h|h.id==id) {
                            console.owned.remove(&service);
                            match result { Ok(_)=>say(format!("{} stopped.",service.id()))?, Err(e)=>say(format!("{} stopped: {e}",service.id()))? }
                            console.status(None).await?;
                        }
                    }
                }
            }
        }
    }
    say("\nLeaving Gary; stopping any service started in this console...")?;
    // The captured owner config/nonce is used, even if the file was edited.
    // Never stop a service that was running before this console opened.
    let mut errors = Vec::new();
    if let Some(mut job) = console.training.take() {
        say("Cancelling the active training job and waiting for its checkpoint.")?;
        if let Err(error) = job.cancel_and_wait().await {
            errors.push(error);
        }
    }
    for service in Service::ALL {
        if let Err(e) = console.stop_owned(service).await {
            errors.push(e);
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    say("Goodbye.")?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slash_and_plain_commands_have_the_same_small_service_grammar() {
        assert_eq!(
            action("/start sa3"),
            Ok(Some((Action::Start, Some(Service::Sa3))))
        );
        assert_eq!(action("stop"), Ok(Some((Action::Stop, None))));
        assert_eq!(action("/start all"), Ok(Some((Action::StartAll, None))));
        assert_eq!(action("stop ALL"), Ok(Some((Action::StopAll, None))));
        assert!(action("/install all").is_err());
        assert!(action("/stop all sa3").is_err());
        assert_eq!(action("/services"), Ok(Some((Action::Status, None))));
        assert_eq!(action("  "), Ok(None));
        assert_eq!(
            action("/start yuey"),
            Ok(Some((Action::Start, Some(Service::Yuey))))
        );
        assert!(action("/stop sa3 --force").is_err());
        assert!(action("/train").is_err());
    }
    #[test]
    fn model_folders_can_contain_spaces_and_are_never_shell_evaluated() {
        assert_eq!(
            model_path("\"a folder with spaces\""),
            PathBuf::from("a folder with spaces")
        );
        assert_eq!(
            model_path("$(touch something)"),
            PathBuf::from("$(touch something)")
        );
    }
}
