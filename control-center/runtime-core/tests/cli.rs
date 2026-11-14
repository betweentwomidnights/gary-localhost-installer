use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::{Command, Output};

fn command(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gary4local-server"))
        .args(args)
        .output()
        .unwrap()
}
fn parsed(output: &Output, code: i32) -> Value {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).lines().count(),
        1,
        "machine stdout must contain exactly one JSON result"
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["exit_code"], code);
    assert_eq!(value["ok"], code == 0);
    assert_eq!(value["kind"], "result");
    value
}
struct Fixture {
    root: PathBuf,
    config: PathBuf,
    models: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "gary-cli-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models = root.join("existing model files");
        std::fs::create_dir_all(&models).unwrap();
        let config = root.join("agent config.json");
        Self {
            root,
            config,
            models,
        }
    }
    fn run(&self, action: &str, extra: &[&str]) -> Output {
        let mut args = vec![action, "--config", self.config.to_str().unwrap(), "--json"];
        args.extend_from_slice(extra);
        command(&args)
    }
    fn init(&self) -> Value {
        parsed(
            &self.run("init", &["--models-dir", self.models.to_str().unwrap()]),
            0,
        )
    }
    fn body(&self) -> Value {
        serde_json::from_slice(&std::fs::read(&self.config).unwrap()).unwrap()
    }
    fn write(&self, body: &Value) {
        std::fs::write(&self.config, serde_json::to_vec(body).unwrap()).unwrap();
    }
    fn owned(&self) -> PathBuf {
        PathBuf::from(self.body()["runtime_root"].as_str().unwrap())
    }
    fn fake_install(&self) {
        let plan = parsed(&self.run("validate", &[]), 0);
        let resolved = &plan["data"]["resolved"];
        let runtime = PathBuf::from(resolved["runtime_directory"].as_str().unwrap());
        std::fs::create_dir_all(&runtime).unwrap();
        std::fs::write(
            self.owned().join(".gary4local-headless"),
            "gary4local-server v1\n",
        )
        .unwrap();
        std::fs::write(runtime.join("BUILD-INFO.json"), json!({"service":"sa3", "platform":resolved["platform"], "dirty":false,"tests_run":true,
            "commit":"fixture","ggml_commit":"fixture","version":resolved["expected_version"]}).to_string()).unwrap();
        std::fs::write(runtime.join("package-checksums.json"), json!({"core":resolved["packages"]["core"]["sha256"],"backend":resolved["packages"]["backend"]["sha256"]}).to_string()).unwrap();
        let name = if cfg!(windows) {
            "sa3-server.exe"
        } else {
            "sa3-server"
        };
        std::fs::write(runtime.join(name), b"never execute this inventory fixture").unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn discovery_is_config_free_and_machine_options_are_discoverable() {
    let a = parsed(&command(&["capabilities", "--json"]), 0);
    let b = parsed(&command(&["--json", "capabilities"]), 0);
    assert_eq!(a, b);
    assert_eq!(a["data"]["features"]["training"], false);
    assert_eq!(a["data"]["features"]["model_downloads"], false);
    let commands = a["data"]["commands"].as_array().unwrap();
    let stop = commands
        .iter()
        .find(|entry| entry["name"] == "stop")
        .unwrap();
    assert_eq!(stop["supports_expected_session"], true);
    assert!(stop["options"]
        .as_array()
        .unwrap()
        .iter()
        .any(|option| option["flag"] == "--session"));
    assert_eq!(
        parsed(&command(&["version", "--json"]), 0)["data"]["version"],
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn schemas_and_subcommand_help_do_not_need_a_config() {
    let config = parsed(&command(&["schema", "--json"]), 0);
    assert_eq!(config["data"]["additionalProperties"], false);
    let response = parsed(&command(&["schema", "--target", "response", "--json"]), 0);
    assert_eq!(response["data"]["properties"]["schema_version"]["const"], 1);
    for action in ["install", "serve", "status", "stop", "doctor", "init"] {
        assert_eq!(
            parsed(&command(&[action, "--help", "--json"]), 0)["data"]["help"].is_string(),
            true
        );
    }
    assert_eq!(
        parsed(&command(&["schema", "--target", "bogus", "--json"]), 2)["error"]["code"],
        "invalid_arguments"
    );
}

#[test]
fn init_is_idempotent_without_owning_storage_or_overwriting_different_config() {
    let fixture = Fixture::new();
    let initial = fixture.init();
    assert_eq!(initial["data"]["created"], true);
    assert_eq!(fixture.body()["ae_encoding"], "f32");
    assert!(!fixture.owned().exists());
    let bytes = std::fs::read(&fixture.config).unwrap();
    assert_eq!(fixture.init()["data"]["created"], false);
    let refused = fixture.run(
        "init",
        &[
            "--models-dir",
            fixture.models.to_str().unwrap(),
            "--encoding",
            "q4_k_m",
        ],
    );
    assert_eq!(parsed(&refused, 3)["error"]["code"], "config_exists");
    assert_eq!(std::fs::read(&fixture.config).unwrap(), bytes);
    assert!(!fixture.owned().exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&fixture.config)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}

#[test]
fn init_refuses_model_root_overlap_and_config_inside_owned_root() {
    let fixture = Fixture::new();
    let overlap = fixture.run(
        "init",
        &[
            "--models-dir",
            fixture.models.to_str().unwrap(),
            "--runtime-root",
            fixture.models.to_str().unwrap(),
        ],
    );
    assert_eq!(parsed(&overlap, 3)["error"]["code"], "config_invalid");
    assert!(!fixture.config.exists());
    let containing = fixture.run(
        "init",
        &[
            "--models-dir",
            fixture.models.to_str().unwrap(),
            "--runtime-root",
            fixture.root.to_str().unwrap(),
        ],
    );
    assert_eq!(parsed(&containing, 3)["error"]["code"], "config_invalid");
    assert!(!fixture.config.exists());
}

#[cfg(windows)]
#[test]
fn init_compares_ordinary_windows_paths_with_canonical_model_paths() {
    let fixture = Fixture::new();
    let ordinary = fixture.models.to_str().unwrap().trim_start_matches(r"\\?\");
    assert_ne!(ordinary, fixture.models.to_str().unwrap());
    let output = fixture.run(
        "init",
        &["--models-dir", ordinary, "--runtime-root", ordinary],
    );
    assert_eq!(parsed(&output, 3)["error"]["code"], "config_invalid");
    assert!(!fixture.config.exists());
}

#[test]
fn validate_and_missing_resource_doctor_never_create_runtime_storage() {
    let fixture = Fixture::new();
    fixture.init();
    assert_eq!(
        parsed(&fixture.run("validate", &[]), 0)["data"]["valid"],
        true
    );
    let report = parsed(&fixture.run("doctor", &[]), 4);
    assert_eq!(report["error"]["code"], "not_ready");
    assert_eq!(report["data"]["read_only"], true);
    assert!(!fixture.owned().exists());
    assert!(std::fs::read_dir(&fixture.models).unwrap().next().is_none());
    let checks = report["data"]["checks"].as_array().unwrap();
    assert!(checks
        .iter()
        .any(|check| check["code"] == "model_files" && check["status"] == "error"));
    assert!(checks
        .iter()
        .any(|check| check["code"] == "gpu_probe" && check["status"] == "not_checked"));
}

#[test]
fn doctor_inspects_without_executing_an_installed_binary_and_detects_occupied_ports() {
    let fixture = Fixture::new();
    fixture.init();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut body = fixture.body();
    body["model"] = json!("custom-layout");
    body["public_port"] = json!(listener.local_addr().unwrap().port());
    fixture.write(&body);
    fixture.fake_install();
    let report = parsed(&fixture.run("doctor", &[]), 4);
    assert!(report["data"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["code"] == "public_port" && check["status"] == "error"));
    assert!(!fixture.owned().join("logs").exists());
    drop(listener);
    let report = parsed(&fixture.run("doctor", &[]), 0);
    assert_eq!(report["data"]["can_attempt_serve"], true);
    assert!(report["data"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["code"] == "model_files" && check["status"] == "warning"));
}

#[test]
fn status_and_stop_are_idempotent_without_creating_a_missing_root() {
    let fixture = Fixture::new();
    fixture.init();
    assert_eq!(
        parsed(&fixture.run("status", &[]), 0)["data"]["running"],
        false
    );
    for _ in 0..2 {
        assert_eq!(
            parsed(&fixture.run("stop", &["--session", "old-session"]), 0)["data"]["stopped"],
            true
        );
    }
    assert!(!fixture.owned().exists());
    let output = command(&["status", "--config", fixture.config.to_str().unwrap()]);
    assert!(output.status.success());
    let legacy: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(legacy["running"], false);
    assert!(legacy.get("schema_version").is_none());
}

#[test]
fn ownership_and_config_failures_have_stable_machine_errors_without_mutation() {
    let fixture = Fixture::new();
    assert_eq!(
        parsed(&fixture.run("validate", &[]), 3)["error"]["code"],
        "config_unreadable"
    );
    fixture.init();
    let mut body = fixture.body();
    body["typo"] = json!(true);
    fixture.write(&body);
    assert_eq!(
        parsed(&fixture.run("validate", &[]), 3)["error"]["code"],
        "config_invalid_json"
    );
    body.as_object_mut().unwrap().remove("typo");
    fixture.write(&body);
    std::fs::create_dir(fixture.owned()).unwrap();
    let unrelated = fixture.owned().join("user data.txt");
    std::fs::write(&unrelated, b"preserve me").unwrap();
    assert_eq!(
        parsed(&fixture.run("stop", &[]), 1)["error"]["code"],
        "stop_failed"
    );
    assert_eq!(
        parsed(&fixture.run("doctor", &[]), 4)["data"]["can_attempt_serve"],
        false
    );
    assert_eq!(std::fs::read(unrelated).unwrap(), b"preserve me");
    assert!(!fixture.owned().join("stop.request").exists());
    assert!(!fixture.owned().join(".gary4local-headless").exists());
}

#[test]
fn invalid_arguments_and_pre_readiness_start_failures_are_json_results() {
    assert_eq!(
        parsed(&command(&["train", "--json"]), 2)["error"]["code"],
        "invalid_arguments"
    );
    assert_eq!(
        parsed(
            &command(&["status", "--config", "one", "--config", "two", "--json"]),
            2
        )["error"]["code"],
        "invalid_arguments"
    );
    let fixture = Fixture::new();
    fixture.init();
    let failure = parsed(&fixture.run("serve", &[]), 1);
    assert_eq!(failure["error"]["code"], "serve_failed");
    assert_eq!(failure["event"], Value::Null);
}

#[cfg(unix)]
#[test]
fn stop_rejects_a_root_symlink_without_writing_through_it() {
    let fixture = Fixture::new();
    fixture.init();
    let elsewhere = fixture.root.join("unrelated directory");
    std::fs::create_dir(&elsewhere).unwrap();
    std::fs::write(
        elsewhere.join(".gary4local-headless"),
        "gary4local-server v1\n",
    )
    .unwrap();
    std::os::unix::fs::symlink(&elsewhere, fixture.owned()).unwrap();
    assert_eq!(
        parsed(&fixture.run("stop", &[]), 1)["error"]["code"],
        "stop_failed"
    );
    assert!(!elsewhere.join("stop.request").exists());
}
