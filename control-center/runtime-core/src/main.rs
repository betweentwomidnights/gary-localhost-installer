#[tokio::main]
async fn main() {
    let code = gary4local_core::command::run(std::env::args_os().skip(1).collect(), false).await;
    if code != 0 {
        std::process::exit(code);
    }
}
