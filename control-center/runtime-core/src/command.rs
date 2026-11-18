use crate::cli::{self, Request};
use serde_json::Value;
use std::io::Write;

fn output(value: &Value, pretty: bool) -> Result<(), String> {
    let mut stdout = std::io::stdout().lock();
    let text = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|error| error.to_string())?;
    writeln!(stdout, "{text}")
        .and_then(|_| stdout.flush())
        .map_err(|error| error.to_string())
}

pub fn human_output(command: &str, data: &Value) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    let service = data["service"].as_str().unwrap_or("sa3");
    let service = if service == "sa3" { "SA3" } else { service };
    let write = (|| -> std::io::Result<()> {
        match command {
            "models" | "download-models" => {
                writeln!(
                    out,
                    "{service} {} {} — {:.2} GiB total",
                    data["model"].as_str().unwrap_or(""),
                    data["encoding"].as_str().unwrap_or(""),
                    data["bytes"].as_u64().unwrap_or(0) as f64 / 1073741824.0
                )?;
                writeln!(
                    out,
                    "Storage: {}",
                    data["models_dir"].as_str().unwrap_or("")
                )?;
                let missing = data["missing_bytes"].as_u64().unwrap_or(0);
                if missing == 0 {
                    writeln!(out, "All files are present at the expected size.")?;
                } else {
                    writeln!(out, "{missing} bytes of missing or incomplete weights")?;
                }
                for file in data["files"].as_array().into_iter().flatten() {
                    writeln!(out, "  {}", file["filename"].as_str().unwrap_or(""))?;
                }
                if command == "download-models" {
                    writeln!(out, "{service} weights verified.")?;
                } else if data["managed"] == false {
                    writeln!(out, "Using external weights from your selected folder.")?;
                }
            }
            "install" => writeln!(
                out,
                "{service} runtime installed ({})",
                data["backend"].as_str().unwrap_or("GPU")
            )?,
            "stop" => writeln!(out, "{service} stopped.")?,
            "serve" => writeln!(
                out,
                "{service} ready at {}. Press Ctrl+C to stop.",
                data["public_url"].as_str().unwrap_or("")
            )?,
            "status" => {
                writeln!(out, "SERVICE  STATE             ADDRESS")?;
                let state = if data["running"] == true {
                    "running"
                } else if data["operation_active"] == true {
                    "busy / starting"
                } else {
                    "stopped"
                };
                writeln!(
                    out,
                    "{service:<13}{state:<18}{}",
                    if data["running"] == true {
                        data["public_url"].as_str().unwrap_or("")
                    } else {
                        ""
                    }
                )?;
                if data["running"] == true && data["health"].is_null() {
                    writeln!(out, "Health is unavailable; inspect the service log.")?;
                }
            }
            "init" => {
                writeln!(
                    out,
                    "{}: {}",
                    if data["created"] == true {
                        "Created config"
                    } else {
                        "Config already matches"
                    },
                    data["config_path"].as_str().unwrap_or("")
                )?;
                writeln!(
                    out,
                    "Runtime root: {}",
                    data["config"]["runtime_root"].as_str().unwrap_or("")
                )?;
                writeln!(
                    out,
                    "Use models to inspect weights and install to prepare the runtime."
                )?;
            }
            "validate" => {
                writeln!(
                    out,
                    "Configuration is valid for {}.",
                    data["resolved"]["platform"]
                        .as_str()
                        .unwrap_or("this platform")
                )?;
                writeln!(out, "{}", data["checks"].as_str().unwrap_or(""))?;
            }
            "doctor" => {
                writeln!(
                    out,
                    "{service} preflight: {}",
                    if data["can_attempt_serve"] == true {
                        "startup can be attempted"
                    } else {
                        "blocked"
                    }
                )?;
                for check in data["checks"].as_array().into_iter().flatten() {
                    let detail = check["message"]
                        .as_str()
                        .or_else(|| check["path"].as_str())
                        .or_else(|| check["directory"].as_str())
                        .unwrap_or("");
                    writeln!(
                        out,
                        "[{}] {}: {detail}",
                        check["status"].as_str().unwrap_or(""),
                        check["code"].as_str().unwrap_or("")
                    )?;
                    if let Some(hint) = check["hint"].as_str() {
                        writeln!(out, "  {hint}")?;
                    }
                }
                writeln!(out,"Scope: filesystem metadata and port availability. Install/serve perform GPU probing and model loading.")?;
            }
            _ => unreachable!("human command"),
        }
        out.flush()
    })();
    write.map_err(|error| error.to_string())
}

pub async fn run(args: Vec<std::ffi::OsString>, human: bool) -> i32 {
    let json = args.iter().any(|arg| arg == "--json");
    let request = match Request::parse(&args) {
        Ok(request) => request,
        Err(error) => {
            if json {
                let _ = output(
                    &cli::envelope(
                        "unknown",
                        "result",
                        None,
                        Value::Null,
                        Some(&error),
                        error.exit_code,
                    ),
                    false,
                );
            } else {
                eprintln!("{}\n{}", error.message, error.hint);
            }
            return error.exit_code;
        }
    };
    let result = cli::execute(&request, |ready| {
        if request.json {
            output(
                &cli::envelope("serve", "event", Some("ready"), ready, None, 0),
                false,
            )
        } else if human {
            human_output("serve", &ready)
        } else {
            output(&ready, false)
        }
    })
    .await;
    let exit_code = match result {
        Ok(outcome) => {
            let printed = if request.json {
                output(
                    &cli::envelope(
                        &request.command,
                        "result",
                        None,
                        outcome.data,
                        outcome.error.as_ref(),
                        outcome.exit_code,
                    ),
                    false,
                )
            } else if request.help || request.command == "help" {
                writeln!(std::io::stdout().lock(), "{}", cli::help())
                    .map_err(|error| error.to_string())
            } else if request.command == "version" {
                writeln!(std::io::stdout().lock(), "{}", env!("CARGO_PKG_VERSION"))
                    .map_err(|error| error.to_string())
            } else if request.command == "serve" {
                Ok(())
            } else if matches!(
                request.command.as_str(),
                "init" | "validate" | "doctor" | "models" | "download-models"
            ) || human
                && matches!(request.command.as_str(), "install" | "status" | "stop")
            {
                human_output(&request.command, &outcome.data)
            } else {
                output(
                    &outcome.data,
                    !matches!(request.command.as_str(), "install" | "status" | "stop"),
                )
            };
            if let Some(error) = outcome.error {
                if !request.json {
                    eprintln!("{}\n{}", error.message, error.hint);
                }
            }
            if let Err(error) = printed {
                eprintln!("Cannot write command output: {error}");
                1
            } else {
                outcome.exit_code
            }
        }
        Err(error) => {
            if request.json {
                let _ = output(
                    &cli::envelope(
                        &request.command,
                        "result",
                        None,
                        Value::Null,
                        Some(&error),
                        error.exit_code,
                    ),
                    false,
                );
            } else {
                eprintln!("{}\n{}", error.message, error.hint);
            }
            error.exit_code
        }
    };
    exit_code
}
