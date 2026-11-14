#[tokio::main]
async fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let json = args.iter().any(|arg| arg == "--json");
    let code = match gary4local_core::console::entry(args).await {
        Ok(code) => code,
        Err(message) => {
            if json {
                use std::io::Write;
                let error = gary4local_core::cli::Error {
                    code: "invalid_arguments",
                    message,
                    hint: "Run gary --help, or use capabilities --json for the service protocol."
                        .into(),
                    exit_code: 2,
                };
                let response = gary4local_core::cli::envelope(
                    "unknown",
                    "result",
                    None,
                    serde_json::Value::Null,
                    Some(&error),
                    2,
                );
                let _ = writeln!(std::io::stdout().lock(), "{response}");
            } else {
                eprintln!("{message}");
            }
            2
        }
    };
    if code != 0 {
        std::process::exit(code);
    }
}
