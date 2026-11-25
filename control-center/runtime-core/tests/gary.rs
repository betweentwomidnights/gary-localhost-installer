use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};

struct Fixture {
    root: PathBuf,
    config: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "gary-console-{}-{}-{serial}",
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
fn standard_profiles_follow_desktops_selected_library_without_creating_it() {
    let fixture = Fixture::new();
    let config_home = fixture.root.join("config");
    let data_home = fixture.root.join("local");
    let settings = data_home.join("com.betweentwomidnights.gary4local/storage.json");
    std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_gary"))
            .args(args)
            .env_remove("GARY4LOCAL_HEADLESS_CONFIG")
            .env_remove("GARY4LOCAL_RUNTIME_ROOT")
            .env_remove("GARY4LOCAL_HOME")
            .env("XDG_CONFIG_HOME", &config_home)
            .env("XDG_DATA_HOME", &data_home)
            .env("LOCALAPPDATA", &data_home)
            .output()
            .unwrap()
    };
    for service in ["sa3", "carey"] {
        let result = run(&["init", service, "--json"]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        let result: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(result["data"]["config"]["library_root"], "desktop");
    }
    for name in ["library-one", "moved-library"] {
        let library = fixture.root.join(name);
        std::fs::write(
            &settings,
            serde_json::json!({"runtimeRoot":library}).to_string(),
        )
        .unwrap();
        for service in ["sa3", "carey"] {
            let result = run(&["validate", service, "--json"]);
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stdout)
            );
            let result: Value = serde_json::from_slice(&result.stdout).unwrap();
            assert_eq!(
                result["data"]["resolved"]["library_root"],
                serde_json::json!(library)
            );
        }
        assert!(
            !library.exists(),
            "init/validate must not create desktop library storage"
        );
    }
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
    let output = fixture.run(&["start", "unknown-engine", "--json"]);
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

#[test]
fn named_and_flagged_service_commands_use_isolated_profiles() {
    let fixture = Fixture::new();
    for args in [vec!["init","carey"],vec!["init","--service","gary"]] {
        assert!(fixture.run(&args).status.success());
    }
    assert!(!fixture.config.exists());
    for service in ["carey","gary"] {
        let result=fixture.run(&["validate",service,"--json"]);
        assert!(result.status.success());
        let value:Value=serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(value["data"]["resolved"]["service"],service);
        assert!(value["data"]["config_path"].as_str().unwrap().ends_with(&format!("{service}.json")));
    }
    let list=fixture.run(&["status"]);
    assert!(list.status.success());
    for id in ["sa3","gary","melodyflow","stable-audio","carey","foundation","yuey"] {
        assert!(text(&list).lines().any(|row|row.split_whitespace().next()==Some(id)));
    }
}

#[test]
fn standard_profiles_share_desktops_models_and_merge_their_own_once() {
    let fixture = Fixture::new();
    let config_home = fixture.root.join("config");
    let data_home = fixture.root.join("local");
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_gary"))
            .args(args)
            .env_remove("GARY4LOCAL_HEADLESS_CONFIG")
            .env_remove("GARY4LOCAL_RUNTIME_ROOT")
            .env_remove("GARY4LOCAL_HOME")
            .env("XDG_CONFIG_HOME", &config_home)
            .env("XDG_DATA_HOME", &data_home)
            .env("LOCALAPPDATA", &data_home)
            .output()
            .unwrap()
    };
    // Pin desktop's storage: its default differs by platform and by legacy folders on this machine.
    let desktop_runtime = fixture.root.join("desktop-runtime");
    let settings = data_home.join("com.betweentwomidnights.gary4local/storage.json");
    std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
    std::fs::write(&settings, serde_json::json!({"runtimeRoot":desktop_runtime}).to_string()).unwrap();
    let desktop_models = desktop_runtime.join("models");
    for service in ["sa3", "carey", "yuey"] {
        let result = run(&["init", service, "--json"]);
        assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stdout));
        let result: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(result["data"]["config"]["models_root"], "desktop");
        let result: Value = serde_json::from_slice(&run(&["validate", service, "--json"]).stdout).unwrap();
        assert_eq!(result["data"]["resolved"]["models_dir"], serde_json::json!(desktop_models.join(service)));
    }
    assert!(!desktop_models.exists(), "init/validate must not create desktop model storage");

    // A profile from before sharing: no models_root, and weights in its own runtime root.
    let profile = config_home.join("gary4local/headless/sa3.json");
    let profile = if profile.exists() { profile } else { data_home.join("gary4local/headless/sa3.json") };
    let mut body: Value = serde_json::from_slice(&std::fs::read(&profile).unwrap()).unwrap();
    body.as_object_mut().unwrap().remove("models_root");
    std::fs::write(&profile, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
    let runtime = std::path::PathBuf::from(body["runtime_root"].as_str().unwrap());
    let own = runtime.join("models/sa3");
    std::fs::create_dir_all(&own).unwrap();
    std::fs::write(runtime.join(".gary4local-headless"), "gary4local-server v1\n").unwrap();
    std::fs::write(own.join("only-headless.gguf"), "headless weights").unwrap();
    std::fs::create_dir_all(desktop_models.join("sa3")).unwrap();
    std::fs::write(own.join("both.gguf"), "same bytes").unwrap();
    std::fs::write(desktop_models.join("sa3/both.gguf"), "same bytes").unwrap();

    // Read-only commands leave it alone.
    assert!(run(&["validate", "sa3", "--json"]).status.success());
    assert!(own.join("only-headless.gguf").exists());

    // The first command that changes storage merges it, even though this serve fails later.
    let result = run(&["serve", "sa3", "--json"]);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("now uses the models folder shared with desktop"), "{stderr}");
    assert!(stderr.contains("moved 1 file(s)") && stderr.contains("removed 1 duplicate(s)"), "{stderr}");
    assert_eq!(std::fs::read_to_string(desktop_models.join("sa3/only-headless.gguf")).unwrap(), "headless weights");
    assert!(!own.exists());
    let body: Value = serde_json::from_slice(&std::fs::read(&profile).unwrap()).unwrap();
    assert_eq!(body["models_root"], "desktop");

    // And only once.
    let result = run(&["serve", "sa3", "--json"]);
    assert!(!String::from_utf8_lossy(&result.stderr).contains("shared with desktop"));
}
