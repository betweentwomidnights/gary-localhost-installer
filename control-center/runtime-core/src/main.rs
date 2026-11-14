use gary4local_core::cli::{self, Request};
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

fn human_output(command: &str, data: &Value) -> Result<(), String> {
    let mut out = std::io::stdout().lock();
    let write = (|| -> std::io::Result<()> {
        match command {
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
                writeln!(out,"Next: validate, doctor, then an explicit install. Existing model files are reused.")?;
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
                    "SA3 preflight: {}",
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

#[tokio::main]
async fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
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
            std::process::exit(error.exit_code);
        }
    };
    let result = cli::execute(&request, |ready| {
        if request.json {
            output(
                &cli::envelope("serve", "event", Some("ready"), ready, None, 0),
                false,
            )
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
            } else if matches!(request.command.as_str(), "init" | "validate" | "doctor") {
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
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}
