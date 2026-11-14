use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};

struct Fixture {
    root: PathBuf,
    config: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "gary-console-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        Self {
            config: root.join("sa3.json"),
            root,
        }
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_gary"))
            .args(args)
            .env("GARY4LOCAL_HEADLESS_CONFIG", &self.config)
            .output()
            .unwrap()
    }
    fn init(&self) {
        let models = self.root.join("models with spaces");
        std::fs::create_dir(&models).unwrap();
        let result = self.run(&["init", "--models-dir", models.to_str().unwrap(), "--json"]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
fn text(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn opening_without_a_terminal_never_prompts_or_writes_configuration() {
    let fixture = Fixture::new();
    let result = fixture.run(&[]);
    assert!(result.status.success());
    assert!(text(&result).contains("interactive service console"));
    assert!(!fixture.config.exists());
    let result = fixture.run(&["shell"]);
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("needs a terminal"));
}
#[test]
fn short_commands_use_the_saved_config_and_readable_service_status() {
    let fixture = Fixture::new();
    fixture.init();
    let status = fixture.run(&["status", "sa3"]);
    assert!(status.status.success());
    assert!(text(&status).contains("SERVICE  STATE"));
    assert!(text(&status).contains("stopped"));
    assert!(!text(&status).starts_with('{'));
    let stop = fixture.run(&["stop"]);
    assert!(stop.status.success());
    assert!(text(&stop).contains("SA3 stopped"));
    assert!(!fixture.root.join("sa3.data").exists());
}
#[test]
fn start_alias_uses_the_foreground_host_and_never_installs_automatically() {
    let fixture = Fixture::new();
    fixture.init();
    let result = fixture.run(&["start", "sa3", "--json"]);
    assert_eq!(result.status.code(), Some(1));
    let result: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(result["command"], "serve");
    assert_eq!(result["error"]["code"], "serve_failed");
    assert!(!fixture.root.join("sa3.data/downloads").exists());
}
#[test]
fn agent_results_remain_versioned_and_unsupported_services_are_not_invented() {
    let fixture = Fixture::new();
    fixture.init();
    let output = fixture.run(&["status", "--json"]);
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["schema_version"], 1);
    assert_eq!(result["data"]["running"], false);
    let output = fixture.run(&["start", "yuey", "--json"]);
    assert_eq!(output.status.code(), Some(2));
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["error"]["code"], "invalid_arguments");
    assert!(!fixture.root.join("sa3.data").exists());
}
#[test]
fn explicit_config_overrides_the_environment_and_help_does_not_read_it() {
    let fixture = Fixture::new();
    fixture.init();
    let output = fixture.run(&[
        "status",
        "--config",
        fixture.root.join("missing.json").to_str().unwrap(),
        "--json",
    ]);
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["error"]["code"], "config_unreadable");
    assert!(fixture.run(&["--help"]).status.success());
    assert!(fixture.run(&["start", "--help"]).status.success());
}
