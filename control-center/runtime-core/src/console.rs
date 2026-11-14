//! Small line-oriented human console over the same single-SA3 host as the
//! machine CLI. A console owns only the tasks it starts, never stale disk PIDs.
use crate::{
    cli, command,
    headless::{self, Config},
};
use serde_json::{json, Value};
use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub const HELP: &str = "Gary — local GPU services\n\n\
  gary                         Open the interactive service console\n\
  gary --config FILE            Open a console with a specific config\n\
  gary start [sa3]              Run SA3 in the foreground; Ctrl+C stops it\n\
  gary status [sa3]             Show service status\n\
  gary stop [sa3]               Stop the current configured SA3 session\n\
  gary doctor                  Inspect setup without running GPU work\n\
  gary install                 Explicitly install the pinned GPU runtime\n\n\
  Add --config FILE to any service command. Otherwise use the saved default.\n\
  GARY4LOCAL_HEADLESS_CONFIG overrides the default config path.\n\
  Add --json for the noninteractive agent protocol. Advanced commands:\n\
  init, validate, capabilities, schema, serve, version (see COMMAND --help).\n\n\
  Console: /setup /install /start /stop /status /doctor /logs /help /quit\n\
  SA3 only in this preview. Training is not implemented yet.\n\
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
                "start" | "status" | "stop" | "doctor" | "install" | "init" | "validate" | "serve"
            )
        )
    });
    if let Some(index) = position {
        if args
            .get(index + 1)
            .is_some_and(|arg| !arg.to_string_lossy().starts_with('-'))
        {
            if args[index + 1] != "sa3" {
                return Err(
                    "This preview manages SA3 only. Use gary start sa3, stop sa3 or status.".into(),
                );
            }
            args.remove(index + 1);
        }
        if args[index] == "start" {
            args[index] = "serve".into();
        }
        if !args.iter().any(|arg| arg == "--config")
            && !args.iter().any(|arg| arg == "--help" || arg == "-h")
        {
            args.extend([
                OsString::from("--config"),
                default_config()?.into_os_string(),
            ]);
        }
    }
    Ok(command::run(args, true).await)
}

#[derive(Debug, PartialEq)]
enum Action {
    Setup,
    Install,
    Start,
    Stop,
    Status,
    Doctor,
    Logs,
    Help,
    Quit,
}
fn action(line: &str) -> Result<Option<Action>, String> {
    let words: Vec<_> = line.split_whitespace().collect();
    if words.is_empty() {
        return Ok(None);
    }
    let name = words[0].trim_start_matches('/').to_ascii_lowercase();
    let service_action = matches!(
        name.as_str(),
        "start" | "stop" | "status" | "services" | "install" | "doctor" | "logs"
    );
    if words.len() > 1
        && !(service_action && words.len() == 2 && words[1].eq_ignore_ascii_case("sa3"))
    {
        return Err("Use a command on its own or with sa3, for example /start sa3. Type /help for commands.".into());
    }
    Ok(Some(match name.as_str() {
        "setup" => Action::Setup,
        "install" => Action::Install,
        "start" => Action::Start,
        "stop" => Action::Stop,
        "status" | "services" => Action::Status,
        "doctor" => Action::Doctor,
        "logs" => Action::Logs,
        "help" | "?" => Action::Help,
        "quit" | "exit" => Action::Quit,
        _ => {
            return Err(format!(
                "Unknown command '{}'. Type /help for commands.",
                words[0]
            ))
        }
    }))
}
type Input = mpsc::UnboundedReceiver<Result<String, String>>;
fn input() -> Input {
    let (tx, rx) = mpsc::unbounded_channel();
    // A plain OS thread is intentional: Tokio's blocking-stdin task would keep
    // runtime shutdown waiting for Enter after Ctrl+C/SIGTERM.
    std::thread::spawn(move || loop {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if tx.send(Ok(line)).is_err() {
                    break;
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e.to_string()));
                break;
            }
        }
    });
    rx
}
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
    say("Set up SA3 — /cancel leaves setup without saving.\nExisting medium weights are reused: F16 T5, F32 autoencoder, and your chosen DiT precision.")?;
    let models = loop {
        let text = answer(input, "Folder containing your existing SA3 models: ").await?;
        let path = model_path(&text);
        if !text.is_empty() && path.is_dir() {
            break path;
        }
        say("That folder does not exist. Enter a folder path, or /cancel.")?;
    };
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
        say("Choose one of the listed precisions matching your downloaded DiT weights.")?;
    };
    let args: Vec<OsString> = vec![
        "init".into(),
        "--config".into(),
        path.into(),
        "--models-dir".into(),
        models.into_os_string(),
        "--port".into(),
        port.into(),
        "--device".into(),
        device.into(),
        "--encoding".into(),
        encoding.into(),
    ];
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
    say("Next: /doctor, then /install when you want to install the runtime.")
}

struct Running {
    id: u64,
    config: Config,
    session: Option<String>,
    task: JoinHandle<Result<Value, String>>,
}
impl Drop for Running {
    fn drop(&mut self) {
        self.task.abort();
    }
}
struct Console {
    path: PathBuf,
    owned: Option<Running>,
    serial: u64,
    ready: mpsc::UnboundedSender<(u64, Value)>,
}
impl Console {
    fn config(&self) -> Result<Config, String> {
        Config::read_checked(&self.path).map_err(|e| {
            format!(
                "{}\nUse /setup for a new config, or /doctor after correcting it.\nConfig: {}",
                e.message,
                self.path.display()
            )
        })
    }
    async fn status(&self) -> Result<(), String> {
        if !self.path.exists() {
            return say("SERVICE  STATE             ADDRESS\nSA3      not configured\n\nType /setup to choose your model folder and GPU settings.");
        }
        let mut status = headless::status(&self.config()?).await?;
        if self
            .owned
            .as_ref()
            .is_some_and(|host| host.session.is_none())
        {
            status["operation_active"] = json!(true);
        }
        command::human_output("status", &status)?;
        if status["running"] == true && self.owned.is_none() {
            say("Running in another session. /quit will leave it running; /stop explicitly stops it.")?;
        }
        Ok(())
    }
    async fn stop_owned(&mut self) -> Result<(), String> {
        if let Some(host) = self.owned.as_mut() {
            if let Some(session) = &host.session {
                headless::stop(&host.config, Some(session))
                    .await
                    .map_err(|e| e.to_string())?;
            } else {
                host.task.abort();
            }
            // Abort is also the bounded fallback after a normal stop has
            // released ownership. Await it so child/lock destructors finish.
            host.task.abort();
            let _ = (&mut host.task).await;
            self.owned.take();
        }
        Ok(())
    }
    async fn handle(&mut self, action: Action, input: &mut Input) -> Result<bool, String> {
        match action {
            Action::Quit=>return Ok(true),
            Action::Help=>say("/setup        Choose existing models, GPU, precision and port\n/install      Install the pinned runtime (explicit download and GPU probe)\n/start [sa3]  Start SA3; you can keep using this console while it loads\n/stop [sa3]   Stop SA3, or cancel startup\n/status       Show services and addresses (/services also works)\n/doctor       Read-only preflight; no GPU work or downloads\n/logs         Show the native log path\n/quit         Stop services started here and leave (Ctrl+C also works)\n\nCommands work with or without '/'. Training is not implemented yet.")?,
            Action::Setup=>setup(&self.path,input).await?,
            Action::Status=>self.status().await?,
            Action::Logs=>say(self.config()?.runtime_root.join("logs/sa3-native.log").display())?,
            Action::Doctor=>command::human_output("doctor",&headless::doctor(&self.config()?).await?)?,
            Action::Install=>{
                if self.owned.is_some() { return Err("Stop SA3 before installing its runtime.".into()); }
                say("Installing verified SA3 runtime packages and checking the GPU...")?;
                command::human_output("install",&headless::install(&self.config()?).await?)?;
                say("Next: /doctor, then /start.")?;
            }
            Action::Start=>{
                if self.owned.is_some() { say("SA3 is already started or starting in this console. Use /status or /stop.")?; return Ok(false); }
                let config=self.config()?;
                let status=headless::status(&config).await?;
                if status["operation_active"]==true { self.status().await?; return Ok(false); }
                let preflight=headless::doctor(&config).await?;
                if preflight["can_attempt_serve"]!=true {
                    command::human_output("doctor",&preflight)?;
                    return Err("Resolve the preflight checks before /start. Use /install if the runtime is missing; model files must already exist.".into());
                }
                self.serial+=1;
                let id=self.serial;
                let ready=self.ready.clone();
                let owned_config=config.clone();
                let task=tokio::spawn(async move { headless::serve(&owned_config,move |value|ready.send((id,value)).map_err(|e|e.to_string())).await });
                self.owned=Some(Running {id,config,session:None,task});
                say("Starting SA3 and loading models... /status shows progress; /stop cancels startup.")?;
            }
            Action::Stop=>{
                if self.owned.is_some() { self.stop_owned().await?; }
                else {
                    let config=self.config()?;
                    let status=headless::status(&config).await?;
                    let session=status["state"]["session"].as_str();
                    headless::stop(&config,session).await.map_err(|e|e.to_string())?;
                }
                say("SA3 stopped.")?;
                self.status().await?;
            }
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
    let mut console = Console {
        path,
        owned: None,
        serial: 0,
        ready: tx,
    };
    let mut input = input();
    let shutdown = termination();
    tokio::pin!(shutdown);
    // Register signal handlers before showing a prompt, including the first
    // prompt. Keep the same future alive through setup and startup.
    tokio::select! { biased; _=&mut shutdown=>return Ok(0), _=std::future::ready(())=>{} }
    say(format!("Gary {} — local GPU services\nConfig: {}\n\nType /help for commands. /quit or Ctrl+C stops services started here.\n",env!("CARGO_PKG_VERSION"),console.path.display()))?;
    if let Err(e) = console.status().await {
        say(format!("{e}\nType /help for commands."))?;
    }
    loop {
        prompt("\ngary> ")?;
        tokio::select! {
            _=&mut shutdown=>break,
            line=input.recv()=>{
                let Some(line)=line else { break; };
                let action=line.and_then(|line|action(&line));
                match action {
                    Ok(None)=>{},
                    Err(e)=>say(e)?,
                    Ok(Some(action))=>{
                        tokio::select! {
                            _=&mut shutdown=>break,
                            result=console.handle(action,&mut input)=>match result {
                                Ok(true)=>break,Ok(false)=>{},Err(e)=>say(e)?,
                            }
                        }
                    }
                }
            }
            event=ready.recv(), if console.owned.is_some()=>{
                if let Some((id,value))=event {
                    if let Some(host)=console.owned.as_mut().filter(|host|host.id==id) {
                        host.session=value["session"].as_str().map(str::to_owned);
                        say(format!("\nSA3 ready at {}",value["public_url"].as_str().unwrap_or("")))?;
                        console.status().await?;
                    }
                }
            }
            result=async { (&mut console.owned.as_mut().expect("guarded host").task).await }, if console.owned.is_some()=>{
                console.owned.take();
                match result {
                    Ok(Ok(_))=>say("\nSA3 stopped.")?,
                    Ok(Err(e))=>say(format!("\nSA3 stopped with an error: {e}\nUse /doctor and /logs to inspect it."))?,
                    Err(e)=>say(format!("\nSA3 task ended: {e}"))?,
                }
                console.status().await?;
            }
        }
    }
    say("\nLeaving Gary; stopping any service started in this console...")?;
    // The captured owner config/nonce is used, even if the file was edited.
    // Never stop a service that was running before this console opened.
    console.stop_owned().await?;
    say("Goodbye.")?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slash_and_plain_commands_have_the_same_small_service_grammar() {
        assert_eq!(action("/start sa3"), Ok(Some(Action::Start)));
        assert_eq!(action("stop"), Ok(Some(Action::Stop)));
        assert_eq!(action("/services"), Ok(Some(Action::Status)));
        assert_eq!(action("  "), Ok(None));
        assert!(action("/start yuey").is_err());
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
