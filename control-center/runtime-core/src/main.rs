use gary4local_core::headless::{self, Config};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && args[0] == "--version" {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if args.is_empty() || args[0] == "--help" {
        println!("gary4local-server <install|serve|status|stop> --config FILE\n\nExperimental SA3 supervisor. Uses verified local core/backend ZIPs.\nAll listeners bind to 127.0.0.1. No models are downloaded.");
        return Ok(());
    }
    if args.len() != 3 || args[1] != "--config" {
        return Err("Use gary4local-server <install|serve|status|stop> --config FILE".into());
    }
    let config = Config::read(std::path::Path::new(&args[2]))?;
    match args[0].to_str() {
        Some("install") => headless::install(&config).await,
        Some("serve") => headless::serve(&config).await,
        Some("status") => headless::status(&config).await,
        Some("stop") => headless::stop(&config).await,
        _ => Err("Unknown command; use --help".into()),
    }
}
