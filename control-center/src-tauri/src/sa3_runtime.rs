//! The selected SA3 runtime belongs to the active storage profile.
//! Once selected, interrupted cleanup must never silently restore Python.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Selection {
    pub schema_version: u32,
    pub encoding: String,
    pub verified_release: String,
    pub backend: String,
    pub activated_at: u64,
    #[serde(default)]
    pub cleanup_complete: bool,
    #[serde(default)]
    pub cleanup_errors: Vec<String>,
}

// Reject an older or mismatched binary before hashing large models or starting
// the private validation server. Live /health checks still verify the launched
// process later; this is the model-free compatibility gate.
fn check_server_info(info: &serde_json::Value) -> Result<(), String> {
    if info["schema_version"] != 1
        || info["service"] != "sa3"
        || info["version"]
            .as_str()
            .map_or(true, |version| version.trim().is_empty())
    {
        return Err("Installed SA3 server has an unsupported control contract. Prepare a compatible sa3.cpp release before migration.".into());
    }
    for capability in [
        "fixed_prefix",
        "request_splice",
        "conditioning_duration",
        "model_lifecycle",
    ] {
        if info["capabilities"][capability] != true {
            return Err(format!("Installed SA3 server lacks {capability}. Prepare a compatible sa3.cpp release before migration."));
        }
    }
    Ok(())
}

pub async fn probe_server(
    executable: &Path,
    runtime_path: Option<&std::ffi::OsStr>,
) -> Result<(), String> {
    let mut command = tokio::process::Command::new(executable);
    // Older servers ignore unknown flags. Their invalid-port guard ensures a
    // missing --control-info cannot accidentally start the public service.
    command
        .args(["--control-info", "--port", "0"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    if let Some(parent) = executable.parent() {
        command.current_dir(parent);
    }
    if let Some(path) = runtime_path {
        command.env("PATH", path);
    }
    crate::workload_job::configure_tokio_command(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| format!("Cannot check native SA3 server: {error}"))?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    let result = tokio::time::timeout(std::time::Duration::from_secs(15), child.wait_with_output())
        .await.map_err(|_| "Native SA3 server capability check timed out. Prepare a compatible sa3.cpp release before migration.")?
        .map_err(|error| format!("Cannot read native SA3 server capabilities: {error}"))?;
    if !result.status.success() {
        return Err("Installed SA3 server does not support offline capability checks. Prepare a compatible sa3.cpp release before migration.".into());
    }
    let info = serde_json::from_slice(&result.stdout)
        .map_err(|error| format!("Invalid native SA3 server capabilities: {error}"))?;
    check_server_info(&info)
}

pub fn selection_path(root: &Path) -> PathBuf {
    root.join("sa3/native-runtime.json")
}

impl Selection {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.verified_release.is_empty()
            || !["cuda", "vulkan", "metal", "cpu", "auto"].contains(&self.backend.as_str())
        {
            return Err("Invalid or unsupported native SA3 runtime selection".into());
        }
        crate::sa3_models::preparation_ids(&self.encoding, None)?;
        Ok(())
    }
}

pub fn read(root: &Path) -> Result<Option<Selection>, String> {
    let path = selection_path(root);
    match std::fs::symlink_metadata(&path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("Cannot inspect native SA3 selection: {error}")),
    }
    let boundary = root.canonicalize().map_err(|error| error.to_string())?;
    let dir = root
        .join("sa3")
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !dir.starts_with(&boundary)
        || path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err("Native SA3 selection points outside the active storage profile".into());
    }
    let selection: Selection =
        serde_json::from_slice(&std::fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("Cannot read native SA3 selection: {error}"))?;
    selection.validate()?;
    Ok(Some(selection))
}

pub fn save(root: &Path, selection: &Selection) -> Result<(), String> {
    selection.validate()?;
    let dir = crate::sa3_training::checked_folder(root, &["sa3"])?;
    let path = dir.join("native-runtime.json");
    if std::fs::symlink_metadata(&path).is_ok()
        && path
            .canonicalize()
            .map_err(|error| error.to_string())?
            .parent()
            != Some(dir.as_path())
    {
        return Err("Native SA3 selection points outside the active storage profile".into());
    }
    crate::sa3_training::save(&path, selection)
}

pub fn launch_args(args: &[String], encoding: &str) -> Result<Vec<String>, String> {
    crate::sa3_models::preparation_ids(encoding, None)?;
    let mut args = args.to_vec();
    let indices: Vec<usize> = args
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (value == "--encoding").then_some(index))
        .collect();
    match indices.as_slice() {
        [] => {
            args.extend(["--encoding".into(), encoding.to_ascii_lowercase()]);
        }
        [index] if index + 1 < args.len() && !args[index + 1].starts_with("--") => {
            args[index + 1] = encoding.to_ascii_lowercase();
        }
        _ => return Err("Native SA3 launch has an invalid encoding argument".into()),
    }
    Ok(args)
}

pub fn missing_models(root: &Path, encoding: &str) -> Option<String> {
    match crate::sa3_models::preparation_ids(encoding, None) {
        Err(error) => Some(error),
        Ok(ids) => {
            let dir = crate::sa3_models::models_dir(root);
            let missing: Vec<&str> = ids
                .iter()
                .filter_map(|id| crate::sa3_models::component(id))
                .filter(|component| !crate::sa3_models::present(component, &dir))
                .map(|component| component.label.as_str())
                .collect();
            (!missing.is_empty())
                .then(|| format!("needs prepared native models: {}", missing.join(", ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn selection() -> Selection {
        Selection {
            schema_version: 1,
            encoding: "Q4_K_M".into(),
            verified_release: "0.1.2".into(),
            backend: "cuda".into(),
            activated_at: 1,
            cleanup_complete: false,
            cleanup_errors: vec!["locked legacy environment".into()],
        }
    }

    #[test]
    fn profile_choice_survives_incomplete_cleanup_and_rejects_invalid_models() {
        let root = std::env::temp_dir().join(format!(
            "gary-sa3-selection-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(read(&root).unwrap().is_none());
        assert!(!root.exists());
        save(&root, &selection()).unwrap();
        let loaded = read(&root).unwrap().unwrap();
        assert!(!loaded.cleanup_complete);
        assert_eq!(loaded.encoding, "Q4_K_M");
        assert_eq!(loaded.cleanup_errors.len(), 1);
        assert!(read(&root.join("different-profile")).unwrap().is_none());
        let mut invalid = selection();
        invalid.encoding = "../model".into();
        assert!(save(&root, &invalid).is_err());
        std::fs::write(selection_path(&root), "{}").unwrap();
        assert!(read(&root).is_err());
        crate::remove_managed_path(&root, &std::env::temp_dir().canonicalize().unwrap()).unwrap();
    }

    #[test]
    fn server_contract_requires_sa3_schema_and_each_typed_capability() {
        let good = serde_json::json!({"schema_version":1,"service":"sa3","version":"0.1.2",
            "capabilities":{"fixed_prefix":true,"request_splice":true,
                "conditioning_duration":true,"model_lifecycle":true,"future_feature":true}});
        check_server_info(&good).unwrap();
        for (key, value) in [
            ("schema_version", serde_json::json!(2)),
            ("service", serde_json::json!("sat")),
            ("version", serde_json::json!(" ")),
        ] {
            let mut invalid = good.clone();
            invalid[key] = value;
            assert!(check_server_info(&invalid).is_err());
        }
        for capability in [
            "fixed_prefix",
            "request_splice",
            "conditioning_duration",
            "model_lifecycle",
        ] {
            for value in [
                serde_json::Value::Null,
                serde_json::json!(false),
                serde_json::json!("true"),
            ] {
                let mut invalid = good.clone();
                invalid["capabilities"][capability] = value;
                assert!(check_server_info(&invalid)
                    .unwrap_err()
                    .contains(capability));
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires locally built and published native server executables"]
    async fn real_server_offline_probe_accepts_compatible_and_rejects_old_runtime() {
        let executable = PathBuf::from(std::env::var_os("GARY4LOCAL_SA3_SMOKE_BINARY").unwrap());
        probe_server(&executable, None).await.unwrap();
        let old = PathBuf::from(std::env::var_os("GARY4LOCAL_SA3_OLD_SMOKE_BINARY").unwrap());
        let started = std::time::Instant::now();
        let error = probe_server(&old, None).await.unwrap_err();
        assert!(error.contains("does not support offline"), "{error}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[test]
    fn selected_encoding_replaces_only_the_inference_dit_argument() {
        let args: Vec<String> = [
            "--model",
            "medium",
            "--encoding",
            "f16",
            "--t5-encoding",
            "f16",
            "--ae-encoding",
            "f32",
        ]
        .iter()
        .map(|arg| arg.to_string())
        .collect();
        let changed = launch_args(&args, "Q4_K_M").unwrap();
        assert_eq!(changed[3], "q4_k_m");
        assert_eq!(&changed[4..], &args[4..]);
        assert!(launch_args(&["--encoding".into()], "F16").is_err());
    }
}
