//! Noninteractive command discovery and a versioned stdout protocol.
use crate::headless::{self, Config};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
    pub hint: String,
    #[serde(skip)]
    pub exit_code: i32,
}
impl Error {
    fn new(
        code: &'static str,
        message: impl Into<String>,
        hint: impl Into<String>,
        exit_code: i32,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            hint: hint.into(),
            exit_code,
        }
    }
    fn usage(message: impl Into<String>) -> Self {
        Self::new(
            "invalid_arguments",
            message,
            "Run --help, or capabilities --json for command discovery.",
            2,
        )
    }
}
impl From<headless::ConfigError> for Error {
    fn from(error: headless::ConfigError) -> Self {
        Self::new(error.code, error.message, error.hint, 3)
    }
}

#[derive(Debug)]
pub struct Request {
    pub command: String,
    pub json: bool,
    pub help: bool,
    options: BTreeMap<String, OsString>,
}
const COMMANDS: &[&str] = &[
    "init",
    "validate",
    "doctor",
    "capabilities",
    "schema",
    "install",
    "serve",
    "status",
    "stop",
    "version",
    "help",
];
const INIT_OPTIONS: &[&str] = &[
    "config",
    "models-dir",
    "runtime-root",
    "device",
    "port",
    "encoding",
];
fn command_flags(command: &str) -> &'static [&'static str] {
    match command {
        "init" => INIT_OPTIONS,
        "validate" | "doctor" | "install" | "serve" | "status" => &["config"],
        "stop" => &["config", "session"],
        "schema" => &["target"],
        _ => &[],
    }
}
impl Request {
    pub fn parse(args: &[OsString]) -> Result<Self, Error> {
        let mut request = Self {
            command: String::new(),
            json: args.iter().any(|arg| arg == "--json"),
            help: false,
            options: BTreeMap::new(),
        };
        let mut index = 0;
        while index < args.len() {
            let arg = &args[index];
            match arg.to_str() {
                Some("--json") => {}
                Some("--help" | "-h") => request.help = true,
                Some("--version") if request.command.is_empty() => {
                    request.command = "version".into()
                }
                Some(flag) if flag.starts_with("--") => {
                    let name = &flag[2..];
                    if !INIT_OPTIONS.contains(&name) && !["session", "target"].contains(&name) {
                        return Err(Error::usage(format!("Unknown option: {flag}")));
                    }
                    index += 1;
                    let value = args
                        .get(index)
                        .filter(|value| !value.to_string_lossy().starts_with("--"))
                        .ok_or_else(|| Error::usage(format!("{flag} needs a value")))?;
                    if request.options.insert(name.into(), value.clone()).is_some() {
                        return Err(Error::usage(format!("Duplicate option: {flag}")));
                    }
                }
                Some(command) if request.command.is_empty() && COMMANDS.contains(&command) => {
                    request.command = command.into()
                }
                _ => {
                    return Err(Error::usage(format!(
                        "Unknown command or argument: {}",
                        arg.to_string_lossy()
                    )))
                }
            }
            index += 1;
        }
        if request.command.is_empty() {
            request.command = "help".into();
        }
        if request.help || request.command == "help" {
            return Ok(request);
        }
        let allowed = command_flags(&request.command);
        for option in request.options.keys() {
            if !allowed.contains(&option.as_str()) {
                return Err(Error::usage(format!(
                    "--{option} is not an option for {}",
                    request.command
                )));
            }
        }
        if allowed.contains(&"config") && !request.options.contains_key("config") {
            return Err(Error::usage("--config FILE is required"));
        }
        if request.command == "init" && !request.options.contains_key("models-dir") {
            return Err(Error::usage(
                "init requires --models-dir DIR pointing to existing weights",
            ));
        }
        Ok(request)
    }
    fn path(&self, name: &str) -> Option<PathBuf> {
        self.options.get(name).map(PathBuf::from)
    }
    fn text(&self, name: &str) -> Result<Option<&str>, Error> {
        self.options
            .get(name)
            .map(|value| {
                value
                    .to_str()
                    .ok_or_else(|| Error::usage(format!("--{name} must be UTF-8 text")))
            })
            .transpose()
    }
}

pub fn help() -> &'static str {
    "gary4local-server COMMAND [OPTIONS]\n\n\
     init --config FILE --models-dir DIR [--runtime-root DIR] [--device Vulkan0] [--port 28300] [--encoding f16]\n\
     validate --config FILE      Check configuration and resolve pinned packages; no downloads\n\
     doctor --config FILE        Read-only storage/runtime/model/port preflight; no GPU jobs\n\
     install --config FILE       Explicitly install and GPU-probe verified runtime packages\n\
     serve --config FILE         Foreground supervisor; load the model, then report readiness\n\
     status --config FILE        Inspect the matching owned session and health\n\
     stop --config FILE [--session ID]  Stop an owned session; ID fences against a newer restart\n\
     capabilities                Discover commands, supported features and platform packages\n\
     schema [--target config|response]  Print a JSON Schema\n\
     version                     Print the binary version\n\n\
     --json                      Versioned JSON results/errors; serve emits newline-delimited JSON\n\
     --help                      Global or per-command help (never reads config)\n\n\
     SA3 only. All service listeners bind to 127.0.0.1. Models are never downloaded.\n\
     Training commands are not implemented. init never overwrites a different config."
}

pub fn capabilities() -> Value {
    let mut manifest: crate::manifest::Manifest =
        serde_json::from_str(crate::manifest::BUNDLED_JSON).expect("bundled manifest");
    manifest
        .resolve_native_bundles()
        .expect("bundled native bundles");
    let packages = manifest
        .services
        .iter()
        .find(|service| service.id == "sa3")
        .and_then(|service| service.native.as_ref())
        .and_then(|native| native.platforms.get(crate::runtime::current_platform()));
    let mut backends = packages.map(|platform| platform.backends.iter().map(|(name, package)| {
        json!({"name":name,"install_supported":package.requires.is_empty(),"requires":package.requires})
    }).collect::<Vec<_>>()).unwrap_or_default();
    backends.sort_by_key(|backend| backend["name"].as_str().unwrap_or("").to_string());
    json!({"binary":"gary4local-server","version":env!("CARGO_PKG_VERSION"),"schema_version":SCHEMA_VERSION,
        "platform":crate::runtime::current_platform(),"services":["sa3"],"validated_models":["medium"],
        "commands":COMMANDS.iter().filter(|command| **command != "help").map(|command| json!({
            "name":command,"requires_config":matches!(*command,"init"|"validate"|"doctor"|"install"|"serve"|"status"|"stop"),
            "effects":match *command {"init"=>"writes_config","install"=>"installs_runtime_and_probes_gpu","serve"=>"loads_model_and_serves","stop"=>"stops_owned_session",_=>"read_only"},
            "json_stream":*command=="serve","supports_expected_session":*command=="stop",
            "options":command_flags(command).iter().map(|flag| json!({"flag":format!("--{flag}"),
                "required":*flag=="config" || *command=="init" && *flag=="models-dir",
                "value":match *flag {"config"=>"FILE","models-dir"|"runtime-root"=>"DIR","port"=>"PORT","session"=>"SESSION_ID",_=>"TEXT"}})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "features":{"training":false,"model_downloads":false,"named_lora_registry":false,"desktop_profile_migration":false,"loopback_only":true},
        "published_backends":backends,"local_archive_backends":["cuda","vulkan"],
        "exit_codes":{"0":"success","1":"operation failed","2":"invalid arguments","3":"config or init failed","4":"doctor found blocking issues"},
        "json_protocol":{"format":"one JSON result per finite command; serve emits ready event then terminal result","progress":"stderr","versioned":true},
        "next_steps":["init","validate","doctor","install","doctor","serve","status","stop"]})
}

pub fn config_schema() -> Value {
    json!({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:gary4local:headless-config:v1","title":"gary4local headless SA3 configuration",
        "type":"object","additionalProperties":false,
        "required":["models_dir","device","public_port","model","encoding"],
        "properties":{
            "runtime_root":{"type":"string","minLength":1,"description":"Absolute normalized dedicated runtime directory. Defaults to platform data storage."},
            "models_dir":{"type":"string","minLength":1,"description":"Absolute directory containing existing model files; no model downloads are performed."},
            "core":{"$ref":"#/$defs/package"},"backend":{"$ref":"#/$defs/package"},
            "manifest":{"type":["string","null"],"description":"Optional absolute path to a local package manifest; otherwise use bundled pins."},
            "device":{"type":"string","pattern":"^(CUDA|Vulkan)","description":"Exact GPU name returned by engine properties, such as Vulkan0 or CUDA0."},
            "public_port":{"type":"integer","minimum":1,"maximum":65535},
            "model":{"type":"string","description":"Native SA3 model family. medium is the validated headless path."},
            "encoding":{"type":"string","description":"Generation precision, such as f16 or q4_k_m."},
            "t5_encoding":{"type":"string","default":"f16"},"ae_encoding":{"type":"string","default":"f16","description":"Must match existing weights. Current medium preview weights use f32; init sets it explicitly."},
            "threads":{"type":"integer","minimum":1,"maximum":256,"default":4},"defaults":{"type":"object","default":{}}},
        "$defs":{"package":{"type":"object","additionalProperties":false,"required":["path","sha256"],"properties":{
            "path":{"type":"string","minLength":1,"description":"Absolute local ZIP path."},"sha256":{"type":"string","pattern":"^[A-Fa-f0-9]{64}$"}}}},
        "dependentRequired":{"core":["backend"],"backend":["core"]},
        "allOf":[{"not":{"required":["manifest","core"],"properties":{"manifest":{"type":"string"}}}}]})
}

pub fn response_schema() -> Value {
    json!({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:gary4local:cli-response:v1",
        "title":"gary4local CLI JSON result or event","type":"object",
        "required":["schema_version","kind","command","event","ok","exit_code","data","error"],
        "properties":{"schema_version":{"const":SCHEMA_VERSION},"kind":{"enum":["result","event"]},
            "command":{"type":"string"},"event":{"type":["string","null"]},"ok":{"type":"boolean"},
            "exit_code":{"type":"integer","minimum":0,"maximum":255},"data":{},
            "error":{"oneOf":[{"type":"null"},{"type":"object","required":["code","message","hint"],
                "properties":{"code":{"type":"string"},"message":{"type":"string"},"hint":{"type":"string"}}}]}},
        "description":"With --json each stdout line is one complete versioned message. Ignore unknown additive fields; diagnostics are on stderr."})
}

pub fn envelope(
    command: &str,
    kind: &str,
    event: Option<&str>,
    data: Value,
    error: Option<&Error>,
    exit_code: i32,
) -> Value {
    json!({"schema_version":SCHEMA_VERSION,"kind":kind,"command":command,"event":event,"ok":exit_code==0,
        "exit_code":exit_code,"data":data,"error":error})
}

pub struct Outcome {
    pub data: Value,
    pub error: Option<Error>,
    pub exit_code: i32,
}
impl Outcome {
    fn success(data: Value) -> Self {
        Self {
            data,
            error: None,
            exit_code: 0,
        }
    }
}

fn absolute(path: &Path) -> Result<PathBuf, Error> {
    let value = std::path::absolute(path).map_err(|error| {
        Error::new(
            "path_unreadable",
            error.to_string(),
            "Check the supplied path.",
            3,
        )
    })?;
    if value.to_str().is_none() {
        return Err(Error::new(
            "path_not_utf8",
            "JSON configuration requires UTF-8 paths",
            "Choose a config/runtime/model path representable in JSON.",
            3,
        ));
    }
    if value
        .components()
        .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(Error::new(
            "config_invalid",
            "Use normalized paths without '..'",
            "Supply an absolute normalized path.",
            3,
        ));
    }
    Ok(value)
}
// Compare real ancestors so ordinary Windows paths and canonical verbatim
// paths (or paths through an existing symlink) cannot bypass overlap checks.
// Keep the supplied root unchanged: ownership checks must still reject a root
// that is itself a symlink.
fn comparable(path: &Path) -> Result<PathBuf, Error> {
    let mut ancestor = path;
    let mut missing = Vec::new();
    loop {
        match ancestor.canonicalize() {
            Ok(mut resolved) => {
                for name in missing.iter().rev() {
                    resolved.push(name);
                }
                return Ok(resolved);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if let (Some(name), Some(parent)) = (ancestor.file_name(), ancestor.parent()) {
                    missing.push(name.to_os_string());
                    ancestor = parent;
                } else {
                    return Err(Error::new(
                        "path_unreadable",
                        error.to_string(),
                        "Check the path and its parent directory.",
                        3,
                    ));
                }
            }
            Err(error) => {
                return Err(Error::new(
                    "path_unreadable",
                    error.to_string(),
                    "Check path permissions and existing parent directories.",
                    3,
                ))
            }
        }
    }
}
fn init(request: &Request) -> Result<Value, Error> {
    use std::io::Write;
    let config_path = absolute(&request.path("config").unwrap())?;
    let models = request
        .path("models-dir")
        .unwrap()
        .canonicalize()
        .map_err(|error| {
            Error::new(
                "models_unreadable",
                error.to_string(),
                "--models-dir must name an existing directory containing your weights.",
                3,
            )
        })?;
    if !models.is_dir() {
        return Err(Error::new(
            "models_unreadable",
            "--models-dir is not a directory",
            "Choose the directory containing existing weights.",
            3,
        ));
    }
    if models.to_str().is_none() {
        return Err(Error::new(
            "path_not_utf8",
            "JSON configuration requires UTF-8 model paths",
            "Choose a model path representable in JSON.",
            3,
        ));
    }
    let root = match request.path("runtime-root") {
        Some(path) => absolute(&path)?,
        None => config_path.with_file_name(format!(
            "{}.data",
            config_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        )),
    };
    let compared_root = comparable(&root)?;
    if comparable(&config_path)?.starts_with(&compared_root)
        || models.starts_with(&compared_root)
        || compared_root.starts_with(&models)
    {
        return Err(Error::new("config_invalid", "Keep config, runtime root and existing model sources separate", "Store the config outside the runtime root; choose a root that does not overlap the models directory.", 3));
    }
    let port = request
        .text("port")?
        .unwrap_or("28300")
        .parse::<u16>()
        .map_err(|_| Error::usage("--port must be an integer from 1 to 65535"))?;
    let body = json!({"runtime_root":root,"models_dir":models,"device":request.text("device")?.unwrap_or("Vulkan0"),
        "public_port":port,"model":"medium","encoding":request.text("encoding")?.unwrap_or("f16").to_ascii_lowercase(),
        "t5_encoding":"f16","ae_encoding":"f32","threads":4,"defaults":{}});
    let raw = serde_json::to_vec_pretty(&body).expect("JSON config");
    let config = Config::from_bytes(&raw)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let created = match options.open(&config_path) {
        Ok(mut file) => {
            file.write_all(&raw).and_then(|_| file.write_all(b"\n")).and_then(|_| file.sync_all())
            .map_err(|error| Error::new("config_write_failed", error.to_string(), "Check available storage and permissions; inspect the new config before using it.", 3))?;
            true
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = std::fs::read(&config_path)
                .ok()
                .and_then(|raw| serde_json::from_slice::<Value>(&raw).ok());
            if existing.as_ref() != Some(&body) {
                return Err(Error::new("config_exists", "A different config already exists at that path", "Inspect it with validate, edit it deliberately, or choose another --config path. init never overwrites it.", 3));
            }
            false
        }
        Err(error) => {
            return Err(Error::new(
                "config_write_failed",
                error.to_string(),
                "Create the config's parent directory and check write permissions.",
                3,
            ))
        }
    };
    Ok(
        json!({"created":created,"config_path":config_path,"config":body,"resolved":config.summary(),"next_commands":["validate","doctor","install"]}),
    )
}

pub async fn execute(
    request: &Request,
    emit_event: impl Fn(Value) -> Result<(), String>,
) -> Result<Outcome, Error> {
    if request.help || request.command == "help" {
        return Ok(Outcome::success(json!({"help":help()})));
    }
    match request.command.as_str() {
        "version" => {
            return Ok(Outcome::success(
                json!({"version":env!("CARGO_PKG_VERSION"),"schema_version":SCHEMA_VERSION}),
            ))
        }
        "capabilities" => return Ok(Outcome::success(capabilities())),
        "schema" => {
            return Ok(Outcome::success(
                match request.text("target")?.unwrap_or("config") {
                    "config" => config_schema(),
                    "response" => response_schema(),
                    _ => return Err(Error::usage("schema --target must be config or response")),
                },
            ))
        }
        "init" => return init(request).map(Outcome::success),
        _ => {}
    }
    let path = request.path("config").unwrap();
    let config = Config::read_checked(&path)?;
    if request.command == "stop" {
        let expected = request.text("session")?;
        if expected == Some("") {
            return Err(Error::usage("--session cannot be empty"));
        }
        return headless::stop(&config, expected).await.map(Outcome::success).map_err(|error| {
            let code = if matches!(error, headless::StopError::SessionMismatch { .. }) { "session_mismatch" } else { "stop_failed" };
            Error::new(code, error.to_string(), "Inspect status. Never retry with a newer session ID unless stopping that session is intended.", 1)
        });
    }
    if request.command == "validate" {
        return Ok(Outcome::success(
            json!({"valid":true,"config_path":path.to_string_lossy(),"resolved":config.summary(),"checks":"configuration only; no GPU or model load"}),
        ));
    }
    let result = match request.command.as_str() {
        "doctor" => headless::doctor(&config).await,
        "install" => headless::install(&config).await,
        "serve" => headless::serve(&config, emit_event).await,
        "status" => headless::status(&config).await,
        _ => unreachable!("validated command"),
    }.map_err(|message| Error::new(match request.command.as_str() {
        "install"=>"install_failed","serve"=>"serve_failed","status"=>"status_failed","stop"=>"stop_failed",_=>"doctor_failed",
    }, message, "Run doctor --config FILE --json for preflight details; inspect the reported native log if startup failed. Do not blindly retry a mutation.", 1))?;
    if request.command == "doctor" && result["can_attempt_serve"] == false {
        return Ok(Outcome { data: result, error: Some(Error::new("not_ready", "Preflight found blocking issues", "Follow the checks and next_actions in data; install and serve remain explicit actions.", 4)), exit_code: 4 });
    }
    Ok(Outcome::success(result))
}
