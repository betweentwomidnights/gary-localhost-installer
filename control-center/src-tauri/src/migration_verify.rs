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
