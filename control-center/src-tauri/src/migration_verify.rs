//! The shared parts of proving a native runtime before a service switches to
//! it. What to prove is the service's; how to run its server privately for
//! that proof is the same for every service.
use crate::native_runtime::NativeLaunch;
use std::path::Path;
use std::time::{Duration, Instant};

/// The service's own launch, started on a private port so no client can reach
/// it and nothing else on the machine changes. It has answered /health with
/// every required capability by the time `start` returns, and it is killed
/// when this is dropped, so an early return from a failed check cannot leave
/// it running.
pub struct PrivateServer {
    child: tokio::process::Child,
    pub client: reqwest::Client,
    /// `http://127.0.0.1:<port>`.
    pub base: String,
}

impl PrivateServer {
    /// `log_dir` receives the server's output as `server-<nonce>.log`, kept
    /// for diagnosis whatever the outcome.
    pub async fn start(
        launch: &NativeLaunch,
        log_dir: &Path,
        label: &str,
        required_capabilities: &[&str],
    ) -> Result<Self, String> {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(log_dir.join(format!("server-{nonce}.log")))
            .map_err(|error| error.to_string())?;
        let reservation = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|error| error.to_string())?;
        let port = reservation
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        // The service's own launch; --port overrides whatever port its
        // environment sets.
        let mut command = launch.tokio_command();
        crate::hide_console_window(&mut command);
        command
            .arg("--port")
            .arg(port.to_string())
            .stdout(log.try_clone().map_err(|error| error.to_string())?)
            .stderr(log)
            .kill_on_drop(true);
        crate::workload_job::configure_tokio_command(&mut command);
        drop(reservation);
        let mut child = command.spawn().map_err(|error| error.to_string())?;
        if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
            let _ = child.kill().await;
            return Err(error);
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .pool_max_idle_per_host(0)
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|error| error.to_string())?;
        let base = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + Duration::from_secs(30);
        let health: serde_json::Value = loop {
            if let Some(exit) = child.try_wait().map_err(|error| error.to_string())? {
                return Err(format!(
                    "Native {label} migration check exited with {exit}; inspect migration-checks logs"
                ));
            }
            if let Ok(response) = client.get(format!("{base}/health")).send().await {
                if response.status().is_success() {
                    break response.json().await.map_err(|error| error.to_string())?;
                }
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "Native {label} migration check did not start; inspect migration-checks logs"
                ));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        for capability in required_capabilities {
            if health["capabilities"][capability] != true {
                return Err(format!(
                    "Installed {label} runtime lacks {capability}; prepare a compatible release before migration"
                ));
            }
        }
        Ok(Self { child, client, base })
    }

    /// Stop the server once the checks are done, and wait for it to exit.
    pub async fn stop(mut self) -> Result<(), String> {
        self.child.kill().await.map_err(|error| error.to_string())?;
        self.child.wait().await.map_err(|error| error.to_string())?;
        Ok(())
    }
}

/// Submit a job to the private server and wait for its audio: the
/// session/poll_status contract every GGML server shares. Returns the WAV.
pub async fn run_audio_job(
    server: &PrivateServer,
    path: &str,
    body: &serde_json::Value,
    what: &str,
) -> Result<Vec<u8>, String> {
    use base64::Engine;
    let submitted: serde_json::Value = server
        .client
        .post(format!("{}{path}", server.base))
        .json(body)
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("{what} was refused: {error}"))?
        .json()
        .await
        .map_err(|error| error.to_string())?;
    let id = submitted["session_id"]
        .as_str()
        .filter(|id| {
            !id.is_empty()
                && id
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '-')
        })
        .ok_or_else(|| format!("{what} returned an invalid job"))?
        .to_string();
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let status: serde_json::Value = server
            .client
            .get(format!("{}/poll_status/{id}", server.base))
            .send()
            .await
            .map_err(|error| error.to_string())?
            .error_for_status()
            .map_err(|error| error.to_string())?
            .json()
            .await
            .map_err(|error| error.to_string())?;
        match status["status"].as_str() {
            Some("completed") => {
                return base64::engine::general_purpose::STANDARD
                    .decode(
                        status["audio_data"]
                            .as_str()
                            .ok_or_else(|| format!("{what} returned no audio"))?,
                    )
                    .map_err(|error| error.to_string());
            }
            Some("failed") => return Err(format!("{what} failed: {}", status["error"])),
            _ => {}
        }
        if Instant::now() >= deadline {
            return Err(format!("{what} timed out"));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Generate on a private server that answers synchronously with
/// `{success, audio_base64, metadata}`, as jerry's contract does. Returns the
/// WAV and the metadata. The wait is per request, so a slow backend has as
/// long as a polled job would.
pub async fn run_sync_audio(
    server: &PrivateServer,
    path: &str,
    body: &serde_json::Value,
    what: &str,
) -> Result<(Vec<u8>, serde_json::Value), String> {
    use base64::Engine;
    let response = server
        .client
        .post(format!("{}{path}", server.base))
        .timeout(Duration::from_secs(300))
        .json(body)
        .send()
        .await
        .map_err(|error| format!("{what} failed: {error}"))?;
    let status = response.status();
    let reply: serde_json::Value = response.json().await.map_err(|error| error.to_string())?;
    if !status.is_success() || reply["success"] != true {
        return Err(format!("{what} failed: {}", reply["error"].as_str().unwrap_or(status.as_str())));
    }
    let audio = base64::engine::general_purpose::STANDARD
        .decode(reply["audio_base64"].as_str().ok_or_else(|| format!("{what} returned no audio"))?)
        .map_err(|error| error.to_string())?;
    Ok((audio, reply["metadata"].clone()))
}

/// Check a proof's WAV is 44.1 kHz stereo and exactly `samples` long.
pub fn check_wav(wav: &[u8], samples: u32, what: &str) -> Result<(), String> {
    let reader = hound::WavReader::new(std::io::Cursor::new(wav)).map_err(|error| error.to_string())?;
    let spec = reader.spec();
    if spec.sample_rate != 44100 || spec.channels != 2 || reader.duration() != samples {
        return Err(format!(
            "{what} check returned {} Hz, {} channels, {} samples; expected 44100 Hz stereo, {samples}",
            spec.sample_rate,
            spec.channels,
            reader.duration()
        ));
    }
    Ok(())
}

/// A metadata fingerprint of what a runtime proof covered: each path's size
/// and modification time, plus `identity` (release, backend, tier...). It
/// hashes no file contents, so recomputing it is cheap; a missing file counts
/// as a difference.
pub fn fingerprint(paths: &[std::path::PathBuf], identity: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    let stamps: Vec<(String, Option<(u64, u128)>)> = paths
        .iter()
        .map(|path| {
            let metadata = std::fs::metadata(path).ok().and_then(|metadata| {
                let nanos = metadata
                    .modified()
                    .ok()?
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()?
                    .as_nanos();
                Some((metadata.len(), nanos))
            });
            (path.to_string_lossy().into_owned(), metadata)
        })
        .collect();
    let bytes = serde_json::to_vec(&(identity, stamps)).unwrap_or_default();
    format!("{:x}", Sha256::digest(bytes))
}
