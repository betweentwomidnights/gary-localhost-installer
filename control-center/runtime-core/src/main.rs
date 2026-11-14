#[tokio::main]
async fn main() -> std::process::ExitCode {
    let code = gary4local_core::command::run(std::env::args_os().skip(1).collect(), false).await;
    std::process::ExitCode::from(u8::try_from(code).unwrap_or(1))
}
