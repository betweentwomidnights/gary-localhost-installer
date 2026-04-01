use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A single model that can be downloaded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,           // e.g. "thepatch/bleeps-large-6"
    pub display_name: String, // e.g. "bleeps-large-6"
    pub service: String,      // e.g. "gary"
    pub size_category: Option<String>, // "small", "medium", "large"
    pub group: Option<String>,         // checkpoint group name
    pub epoch: Option<u32>,            // checkpoint epoch number
    pub status: ModelStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Available,
    Downloading,
    Downloaded,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub progress: f64,   // 0.0 - 1.0
    pub status: ModelStatus,
    pub message: String,
    pub error: Option<String>,
}

/// A checkpoint file from a finetune repo (fetched dynamically)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    pub repo: String,
    pub filename: String,
    pub cached: bool,
}

pub struct ModelManager {
    /// Tracks download progress for active downloads
    downloads: HashMap<String, DownloadProgress>,
    /// Cache of known model statuses
    model_cache: HashMap<String, ModelStatus>,
    /// Dynamically fetched finetune checkpoints (repo → list of checkpoint entries)
    finetune_checkpoints: HashMap<String, Vec<CheckpointEntry>>,
    /// Root of the backend-installer repo (for finding service checkpoint dirs)
    repo_root: PathBuf,
}

impl ModelManager {
    pub fn new(repo_root: PathBuf) -> Self {
        Self {
            downloads: HashMap::new(),
            model_cache: HashMap::new(),
            finetune_checkpoints: HashMap::new(),
            repo_root,
        }
    }

    /// Get the HuggingFace cache directory — uses the standard default location.
    /// This matches what huggingface_hub uses when no HF_HOME is set:
    /// Windows: C:\Users\<user>\.cache\huggingface
    /// Linux/Mac: ~/.cache/huggingface
    pub fn hf_cache_dir() -> PathBuf {
        if let Ok(hf_home) = std::env::var("HF_HOME") {
            PathBuf::from(hf_home)
        } else {
            PathBuf::from(std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("HOME"))
                .unwrap_or_default())
                .join(".cache")
                .join("huggingface")
        }
    }

    /// Get the full list of Gary models with their download status
    pub fn get_gary_models(&self) -> Vec<ModelEntry> {
        let models_by_size: Vec<(&str, &[&str])> = vec![
            ("small", &[
                "thepatch/vanya_ai_dnb_0.1",
                "thepatch/gary_orchestra_2",
                "thepatch/keygen-gary-v2-small-8",
                "thepatch/keygen-gary-v2-small-12",
            ][..]),
            ("medium", &[
                "thepatch/bleeps-medium",
                "thepatch/keygen-gary-medium-12",
            ][..]),
            ("large", &[
                "thepatch/hoenn_lofi",
                "thepatch/bleeps-large-6",
                "thepatch/bleeps-large-8",
                "thepatch/bleeps-large-10",
                "thepatch/bleeps-large-14",
                "thepatch/bleeps-large-20",
                "thepatch/keygen-gary-v2-large-12",
                "thepatch/keygen-gary-v2-large-16",
            ][..]),
        ];

        let mut entries = Vec::new();

        for (size, model_list) in models_by_size {
            for model_path in model_list {
                let name = model_path.split('/').last().unwrap_or(model_path);

                // Parse checkpoint info
                let parts: Vec<&str> = name.rsplitn(2, '-').collect();
                let (group, epoch) = if parts.len() == 2 && parts[0].parse::<u32>().is_ok() {
                    (Some(parts[1].to_string()), Some(parts[0].parse::<u32>().unwrap()))
                } else {
                    (None, None)
                };

                let status = self.get_model_status(model_path);

                entries.push(ModelEntry {
                    id: model_path.to_string(),
                    display_name: name.to_string(),
                    service: "gary".to_string(),
                    size_category: Some(size.to_string()),
                    group,
                    epoch,
                    status,
                });
            }
        }

        entries
    }

    /// Get Jerry (Stable Audio) base model + any fetched finetune checkpoints
    pub fn get_jerry_models(&self) -> Vec<ModelEntry> {
        let mut entries = Vec::new();

        // Base model
        let base_id = "stabilityai/stable-audio-open-small";
        entries.push(ModelEntry {
            id: base_id.to_string(),
            display_name: "stable-audio-open-small (base)".to_string(),
            service: "stable-audio".to_string(),
            size_category: Some("base".to_string()),
            group: None,
            epoch: None,
            status: self.get_model_status(base_id),
        });

        // Any dynamically fetched finetune checkpoints
        for (repo, checkpoints) in &self.finetune_checkpoints {
            for ckpt in checkpoints {
                // Composite ID: "repo::filename" so we can identify it uniquely
                let composite_id = format!("{}::{}", repo, ckpt.filename);
                let status = self.get_model_status(&composite_id);
                entries.push(ModelEntry {
                    id: composite_id,
                    display_name: ckpt.filename.clone(),
                    service: "stable-audio".to_string(),
                    size_category: Some("finetune".to_string()),
                    group: Some(repo.clone()),
                    epoch: None,
                    status,
                });
            }
        }

        entries
    }

    /// Get Carey (ACE-Step) models — just the base model components.
    /// Carey stores models in services/carey/checkpoints/ (not HF cache).
    pub fn get_carey_models(&self) -> Vec<ModelEntry> {
        let checkpoint_dir = self.repo_root
            .join("services").join("carey").join("checkpoints");

        // The three components needed for the base model
        let components: Vec<(&str, &str, &[&str])> = vec![
            ("carey::acestep-v15-base", "acestep-v15-base (DiT)", &[
                "acestep-v15-base/config.json",
                "acestep-v15-base/model.safetensors",
            ]),
            ("carey::vae", "VAE (encoder/decoder)", &[
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ]),
            ("carey::Qwen3-Embedding-0.6B", "Qwen3 Embedding (text)", &[
                "Qwen3-Embedding-0.6B/config.json",
                "Qwen3-Embedding-0.6B/tokenizer.json",
            ]),
        ];

        components.iter().map(|(id, name, check_files)| {
            let status = self.get_carey_component_status(id, &checkpoint_dir, check_files);
            ModelEntry {
                id: id.to_string(),
                display_name: name.to_string(),
                service: "carey".to_string(),
                size_category: Some("base".to_string()),
                group: None,
                epoch: None,
                status,
            }
        }).collect()
    }

    /// Check if a Carey model component is downloaded by looking in the checkpoints dir
    fn get_carey_component_status(&self, id: &str, checkpoint_dir: &std::path::Path, check_files: &[&str]) -> ModelStatus {
        // Check active downloads first
        if let Some(dl) = self.downloads.get(id) {
            return dl.status.clone();
        }
        if let Some(status) = self.model_cache.get(id) {
            return status.clone();
        }
        // Check if the required files exist on disk
        let all_present = check_files.iter().all(|f| checkpoint_dir.join(f).exists());
        if all_present {
            ModelStatus::Downloaded
        } else {
            ModelStatus::Available
        }
    }

    /// Get Foundation-1 model — single model with 2 files.
    /// Foundation stores models in %APPDATA%/Gary4JUCE/models/foundation-1/
    pub fn get_foundation_models(&self) -> Vec<ModelEntry> {
        let models_dir = Self::foundation_models_dir();
        let model_dir = models_dir.join("foundation-1");

        let id = "foundation::foundation-1";
        let status = self.get_foundation_status(id, &model_dir);

        vec![ModelEntry {
            id: id.to_string(),
            display_name: "Foundation-1 (RoyalCities)".to_string(),
            service: "foundation".to_string(),
            size_category: Some("large".to_string()),
            group: None,
            epoch: None,
            status,
        }]
    }

    /// Get the Foundation models directory (%APPDATA%/Gary4JUCE/models)
    fn foundation_models_dir() -> std::path::PathBuf {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
            format!("{}\\AppData\\Roaming", home)
        });
        std::path::PathBuf::from(appdata).join("Gary4JUCE").join("models")
    }

    /// Check if Foundation-1 model files are present
    fn get_foundation_status(&self, id: &str, model_dir: &std::path::Path) -> ModelStatus {
        if let Some(dl) = self.downloads.get(id) {
            return dl.status.clone();
        }
        if let Some(status) = self.model_cache.get(id) {
            return status.clone();
        }

        let ckpt = model_dir.join("Foundation_1.safetensors");
        let config = model_dir.join("model_config.json");
        if ckpt.exists() && config.exists() {
            ModelStatus::Downloaded
        } else {
            ModelStatus::Available
        }
    }

    /// Store fetched checkpoint list for a finetune repo
    pub fn set_finetune_checkpoints(&mut self, repo: &str, filenames: Vec<String>) {
        let entries: Vec<CheckpointEntry> = filenames.into_iter().map(|f| {
            let cached = Self::is_finetune_file_cached(repo, &f);
            CheckpointEntry {
                repo: repo.to_string(),
                filename: f,
                cached,
            }
        }).collect();
        self.finetune_checkpoints.insert(repo.to_string(), entries);
    }

    /// Get stored finetune checkpoints
    pub fn get_finetune_checkpoints(&self) -> &HashMap<String, Vec<CheckpointEntry>> {
        &self.finetune_checkpoints
    }

    /// Check if a specific file from a finetune repo is cached
    fn is_finetune_file_cached(repo: &str, filename: &str) -> bool {
        let cache_dir = Self::hf_cache_dir();
        let hub_dir = cache_dir.join("hub");
        let folder_name = format!("models--{}", repo.replace('/', "--"));
        let snapshots_dir = hub_dir.join(&folder_name).join("snapshots");

        if !snapshots_dir.exists() {
            return false;
        }

        // Check all snapshot directories for this file
        if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
            for entry in entries.flatten() {
                if entry.path().join(filename).exists() {
                    return true;
                }
            }
        }
        false
    }

    /// Check if a model is downloaded by looking in the HF cache.
    /// Supports composite IDs like "repo::filename" for finetune checkpoints.
    fn get_model_status(&self, model_id: &str) -> ModelStatus {
        // Check active downloads first
        if let Some(dl) = self.downloads.get(model_id) {
            return dl.status.clone();
        }

        // Check our cache
        if let Some(status) = self.model_cache.get(model_id) {
            return status.clone();
        }

        // Check the HF cache on disk
        if model_id.contains("::") {
            // Composite ID: "repo::filename"
            let parts: Vec<&str> = model_id.splitn(2, "::").collect();
            if parts.len() == 2 && Self::is_finetune_file_cached(parts[0], parts[1]) {
                return ModelStatus::Downloaded;
            }
        } else if Self::is_model_cached(model_id) {
            return ModelStatus::Downloaded;
        }

        ModelStatus::Available
    }

    /// Check if a model exists in the HuggingFace cache
    fn is_model_cached(model_id: &str) -> bool {
        let cache_dir = Self::hf_cache_dir();
        let hub_dir = cache_dir.join("hub");

        // HF cache uses format: models--org--name
        let folder_name = format!("models--{}", model_id.replace('/', "--"));
        let model_dir = hub_dir.join(&folder_name);

        if !model_dir.exists() {
            return false;
        }

        // Check for snapshot directories (indicates completed download)
        let snapshots_dir = model_dir.join("snapshots");
        if snapshots_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                return entries.count() > 0;
            }
        }

        false
    }

    /// Mark a download as started
    pub fn set_download_started(&mut self, model_id: &str) {
        self.downloads.insert(model_id.to_string(), DownloadProgress {
            model_id: model_id.to_string(),
            progress: 0.0,
            status: ModelStatus::Downloading,
            message: "Starting download...".to_string(),
            error: None,
        });
    }

    /// Update download progress
    pub fn set_download_progress(&mut self, model_id: &str, progress: f64, message: &str) {
        if let Some(dl) = self.downloads.get_mut(model_id) {
            dl.progress = progress;
            dl.message = message.to_string();
        }
    }

    /// Mark download as complete
    pub fn set_download_done(&mut self, model_id: &str, error: Option<String>) {
        if let Some(err) = &error {
            self.downloads.insert(model_id.to_string(), DownloadProgress {
                model_id: model_id.to_string(),
                progress: 0.0,
                status: ModelStatus::Failed,
                message: err.clone(),
                error: Some(err.clone()),
            });
        } else {
            self.downloads.remove(model_id);
            self.model_cache.insert(model_id.to_string(), ModelStatus::Downloaded);
        }
    }

    /// Get all current download progress
    pub fn get_download_progress(&self) -> Vec<DownloadProgress> {
        self.downloads.values().cloned().collect()
    }

    /// Refresh cached statuses from disk
    pub fn refresh_statuses(&mut self) {
        self.model_cache.clear();
    }
}

/// Run a model download using the service's Python venv.
///
/// Supports two ID formats:
/// - "org/repo" — downloads the entire repo (all files)
/// - "org/repo::filename.ckpt" — downloads a single file from the repo
///
/// Uses streaming requests with byte-level progress reporting via JSON on stdout.
pub async fn download_model(
    model_id: String,
    python_exe: PathBuf,
    manager: Arc<Mutex<ModelManager>>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    let cache_dir = ModelManager::hf_cache_dir();

    std::fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Cannot create cache dir: {}", e))?;

    // Parse composite ID: "repo::filename" means download a single file
    let (repo_id, single_file) = if model_id.contains("::") {
        let parts: Vec<&str> = model_id.splitn(2, "::").collect();
        (parts[0].to_string(), Some(parts[1].to_string()))
    } else {
        (model_id.clone(), None)
    };

    let script = if let Some(ref filename) = single_file {
        // Single-file download (for finetune checkpoints)
        format!(
            r#"
import sys, os, json

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import hf_hub_download, constants
import requests, hashlib

repo_id = "{repo_id}"
filename = "{filename}"

def report(pct, msg):
    print(json.dumps({{"p": round(pct, 4), "m": msg}}), flush=True)

def fmt_size(b):
    if b < 1024: return str(b) + "B"
    if b < 1024**2: return f"{{b/1024:.0f}}KB"
    if b < 1024**3: return f"{{b/1024**2:.1f}}MB"
    return f"{{b/1024**3:.2f}}GB"

try:
    short = filename if len(filename) < 50 else "..." + filename[-47:]
    report(0.0, f"Downloading {{short}}...")

    # Get the download URL
    url = f"https://huggingface.co/{{repo_id}}/resolve/main/{{filename}}"
    headers = {{}}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {{token}}"

    resp = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    hub_dir = constants.HF_HUB_CACHE
    folder = os.path.join(hub_dir, "models--" + repo_id.replace("/", "--"))
    blobs_dir = os.path.join(folder, "blobs")
    os.makedirs(blobs_dir, exist_ok=True)

    tmp_path = os.path.join(blobs_dir, filename.replace("/", "_") + ".downloading")
    received = 0
    last_pct = -1

    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            received += len(chunk)
            if total > 0:
                pct = int(received * 100 / total)
                if pct != last_pct:
                    report(received / total, f"{{short}} {{fmt_size(received)}}/{{fmt_size(total)}}")
                    last_pct = pct

    # Hash and place in blobs
    sha = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)
            if not data: break
            sha.update(data)
    blob_path = os.path.join(blobs_dir, sha.hexdigest())
    os.replace(tmp_path, blob_path)

    # Let hf_hub_download finalize cache structure (refs, snapshots)
    hf_hub_download(repo_id, filename=filename)

    report(1.0, "done")
    print(json.dumps({{"status": "done"}}), flush=True)

except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}), flush=True)
    sys.exit(1)
"#,
            repo_id = repo_id,
            filename = filename,
        )
    } else {
        // Full-repo download (for Gary models, base models)
        format!(
            r#"
import sys, os, json

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import HfApi, hf_hub_download, constants

model_id = "{repo_id}"

def report(pct, msg):
    print(json.dumps({{"p": round(pct, 4), "m": msg}}), flush=True)

def fmt_size(b):
    if b < 1024: return str(b) + "B"
    if b < 1024**2: return f"{{b/1024:.0f}}KB"
    if b < 1024**3: return f"{{b/1024**2:.1f}}MB"
    return f"{{b/1024**3:.2f}}GB"

try:
    report(0.0, "Fetching repo info...")
    api = HfApi()
    repo_info = api.repo_info(model_id, files_metadata=True)
    siblings = [s for s in repo_info.siblings if not s.rfilename.startswith(".")]
    file_entries = [(s.rfilename, s.size or (s.lfs.size if s.lfs else 0) or 0) for s in siblings]
    total_bytes = sum(sz for _, sz in file_entries)
    completed_bytes = 0

    if total_bytes == 0:
        report(1.0, "No files")
        print(json.dumps({{"status": "done"}}), flush=True)
        sys.exit(0)

    report(0.0, f"Downloading {{len(file_entries)}} files ({{fmt_size(total_bytes)}})")

    for i, (filename, file_size) in enumerate(file_entries):
        short = filename if len(filename) < 40 else "..." + filename[-37:]

        if file_size < 5 * 1024 * 1024:
            report(completed_bytes / total_bytes, f"{{short}} ({{fmt_size(file_size)}})")
            hf_hub_download(model_id, filename=filename)
            completed_bytes += file_size
            report(completed_bytes / total_bytes, f"{{short}} done")
            continue

        import requests, hashlib

        hub_dir = constants.HF_HUB_CACHE
        folder = os.path.join(hub_dir, "models--" + model_id.replace("/", "--"))
        blobs_dir = os.path.join(folder, "blobs")
        refs_dir = os.path.join(folder, "refs")
        os.makedirs(blobs_dir, exist_ok=True)
        os.makedirs(refs_dir, exist_ok=True)

        url = f"https://huggingface.co/{{model_id}}/resolve/main/{{filename}}"
        headers = {{}}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {{token}}"

        resp = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        resp.raise_for_status()

        tmp_path = os.path.join(blobs_dir, f"{{filename}}.downloading")
        received = 0
        last_pct_reported = -1

        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                received += len(chunk)
                current_total = completed_bytes + received
                pct = current_total / total_bytes
                pct_int = int(pct * 100)
                if pct_int != last_pct_reported:
                    report(pct, f"{{short}} {{fmt_size(current_total)}}/{{fmt_size(total_bytes)}}")
                    last_pct_reported = pct_int

        sha = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            while True:
                data = f.read(1024 * 1024)
                if not data: break
                sha.update(data)
        blob_path = os.path.join(blobs_dir, sha.hexdigest())
        os.replace(tmp_path, blob_path)

        hf_hub_download(model_id, filename=filename)

        completed_bytes += file_size
        report(completed_bytes / total_bytes, f"{{short}} done ({{i+1}}/{{len(file_entries)}})")

    report(1.0, "done")
    print(json.dumps({{"status": "done"}}), flush=True)

except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}), flush=True)
    sys.exit(1)
"#,
            repo_id = repo_id,
        )
    };

    let mut child = tokio::process::Command::new(python_exe.to_str().unwrap())
        .args(["-u", "-c", &script])
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start download: {}", e))?;

    // Read stdout for our JSON progress lines
    let stdout = child.stdout.take();
    let model_id_clone = model_id.clone();
    let mgr_out = manager.clone();
    let handle_out = handle.clone();

    let stdout_task = tauri::async_runtime::spawn(async move {
        if let Some(stdout) = stdout {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                // Parse our JSON progress: {"p": 0.45, "m": "Downloading ..."}
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(pct) = msg.get("p").and_then(|v| v.as_f64()) {
                        let label = msg.get("m")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let mut mgr = mgr_out.lock().await;
                        mgr.set_download_progress(&model_id_clone, pct, &label);
                        drop(mgr);
                        emit_model_status(&mgr_out, &handle_out).await;
                    }
                }
            }
        }
    });

    // Drain stderr so the child process doesn't block
    let stderr = child.stderr.take();
    let stderr_task = tauri::async_runtime::spawn(async move {
        if let Some(stderr) = stderr {
            use tokio::io::{AsyncReadExt, BufReader};
            let mut reader = BufReader::new(stderr);
            let mut buf = [0u8; 4096];
            while let Ok(n) = reader.read(&mut buf).await {
                if n == 0 { break; }
            }
        }
    });

    let _ = stdout_task.await;
    let _ = stderr_task.await;

    let exit_status = child.wait().await
        .map_err(|e| format!("Download process error: {}", e))?;

    if exit_status.success() {
        let mut mgr = manager.lock().await;
        mgr.set_download_done(&model_id, None);
        drop(mgr);
        emit_model_status(&manager, &handle).await;
        Ok(())
    } else {
        let msg = format!("Download failed for {}", model_id);
        let mut mgr = manager.lock().await;
        mgr.set_download_done(&model_id, Some(msg.clone()));
        drop(mgr);
        emit_model_status(&manager, &handle).await;
        Err(msg)
    }
}

/// Fetch checkpoint filenames from a HuggingFace repo using the service's API.
/// Falls back to listing .ckpt files via huggingface_hub if service is not running.
pub async fn fetch_checkpoints(
    repo: String,
    service_port: u16,
) -> Result<Vec<String>, String> {
    // Try the running service first (it has huggingface_hub available)
    let url = format!("http://127.0.0.1:{}/models/checkpoints", service_port);
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&serde_json::json!({ "finetune_repo": repo }))
        .send()
        .await;

    match resp {
        Ok(r) if r.status().is_success() => {
            let body: serde_json::Value = r.json().await
                .map_err(|e| format!("Bad response: {}", e))?;
            if let Some(arr) = body.get("checkpoints").and_then(|v| v.as_array()) {
                let names: Vec<String> = arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                return Ok(names);
            }
            Err(body.get("error").and_then(|v| v.as_str()).unwrap_or("Unknown error").to_string())
        }
        _ => {
            Err("Jerry service is not running. Start it first to fetch checkpoints.".to_string())
        }
    }
}

/// Download a Carey (ACE-Step) model component to the checkpoints directory.
///
/// Carey models are stored in services/carey/checkpoints/ using snapshot_download.
/// The model_id format is "carey::component_name" (e.g. "carey::acestep-v15-base").
pub async fn download_carey_model(
    model_id: String,
    python_exe: PathBuf,
    checkpoint_dir: PathBuf,
    manager: Arc<Mutex<ModelManager>>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    std::fs::create_dir_all(&checkpoint_dir)
        .map_err(|e| format!("Cannot create checkpoint dir: {}", e))?;

    // Extract component name from "carey::component_name"
    let component = model_id.strip_prefix("carey::").unwrap_or(&model_id);

    // Map component to HF repo and allow_patterns
    let (repo_id, allow_pattern) = match component {
        "acestep-v15-base" => ("ACE-Step/acestep-v15-base", ""),
        "vae" => ("ACE-Step/Ace-Step1.5", "vae/**"),
        "Qwen3-Embedding-0.6B" => ("ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/**"),
        other => return Err(format!("Unknown Carey component: {}", other)),
    };

    let ckpt_str = checkpoint_dir.to_string_lossy().to_string();

    // For separate repos (acestep-v15-base), download to checkpoints/component_name/
    // For unified repo components (vae, Qwen3), download matching files to checkpoints/
    //
    // Uses streaming requests with byte-level progress (same approach as Gary),
    // placing files directly in the target directory instead of HF cache.
    let prefix_filter = if allow_pattern.is_empty() {
        // Separate repo — download all files
        String::new()
    } else {
        // Unified repo — only files under this component's subfolder
        component.to_string()
    };

    let script = format!(
        r#"
import sys, os, json, requests

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import HfApi

repo_id = "{repo_id}"
ckpt_dir = r"{ckpt_dir}"
component = "{component}"
prefix_filter = "{prefix_filter}"

def report(pct, msg):
    print(json.dumps({{"p": round(pct, 4), "m": msg}}), flush=True)

def fmt_size(b):
    if b < 1024: return str(b) + "B"
    if b < 1024**2: return f"{{b/1024:.0f}}KB"
    if b < 1024**3: return f"{{b/1024**2:.1f}}MB"
    return f"{{b/1024**3:.2f}}GB"

try:
    report(0.0, "Fetching file list...")
    api = HfApi()
    repo_info = api.repo_info(repo_id, files_metadata=True)
    siblings = [s for s in repo_info.siblings if not s.rfilename.startswith(".")]

    # Filter to only the files we need
    if prefix_filter:
        siblings = [s for s in siblings if s.rfilename.startswith(prefix_filter + "/")]

    file_entries = [(s.rfilename, s.size or (s.lfs.size if s.lfs else 0) or 0) for s in siblings]
    total_bytes = sum(sz for _, sz in file_entries)
    completed_bytes = 0

    if not file_entries:
        report(1.0, "No files to download")
        print(json.dumps({{"status": "done"}}), flush=True)
        sys.exit(0)

    report(0.0, f"Downloading {{len(file_entries)}} files ({{fmt_size(total_bytes)}})")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {{}}
    if token:
        headers["Authorization"] = f"Bearer {{token}}"

    for i, (filename, file_size) in enumerate(file_entries):
        # Determine output path
        if prefix_filter:
            # Unified repo: file is like "vae/config.json" -> checkpoints/vae/config.json
            out_path = os.path.join(ckpt_dir, filename)
        else:
            # Separate repo: file is like "config.json" -> checkpoints/component/config.json
            out_path = os.path.join(ckpt_dir, component, filename)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        short = filename if len(filename) < 40 else "..." + filename[-37:]

        # Small files: direct download
        if file_size < 5 * 1024 * 1024:
            report(completed_bytes / max(total_bytes, 1), f"{{short}} ({{fmt_size(file_size)}})")
            url = f"https://huggingface.co/{{repo_id}}/resolve/main/{{filename}}"
            resp = requests.get(url, headers=headers, allow_redirects=True)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            completed_bytes += file_size
            report(completed_bytes / max(total_bytes, 1), f"{{short}} done")
            continue

        # Large files: stream with progress
        url = f"https://huggingface.co/{{repo_id}}/resolve/main/{{filename}}"
        resp = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        resp.raise_for_status()

        received = 0
        last_pct = -1
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                received += len(chunk)
                current_total = completed_bytes + received
                pct = int(current_total * 100 / max(total_bytes, 1))
                if pct != last_pct:
                    report(current_total / max(total_bytes, 1), f"{{short}} {{fmt_size(current_total)}}/{{fmt_size(total_bytes)}}")
                    last_pct = pct

        completed_bytes += file_size
        report(completed_bytes / max(total_bytes, 1), f"{{short}} done ({{i+1}}/{{len(file_entries)}})")

    report(1.0, "done")
    print(json.dumps({{"status": "done"}}), flush=True)

except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}), flush=True)
    sys.exit(1)
"#,
        repo_id = repo_id,
        ckpt_dir = ckpt_str,
        component = component,
        prefix_filter = prefix_filter,
    );

    let mut child = tokio::process::Command::new(python_exe.to_str().unwrap())
        .args(["-u", "-c", &script])
        .env("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start download: {}", e))?;

    // Read stdout for JSON progress
    let stdout = child.stdout.take();
    let model_id_clone = model_id.clone();
    let mgr_out = manager.clone();
    let handle_out = handle.clone();

    let stdout_task = tauri::async_runtime::spawn(async move {
        if let Some(stdout) = stdout {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(pct) = msg.get("p").and_then(|v| v.as_f64()) {
                        let label = msg.get("m")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let mut mgr = mgr_out.lock().await;
                        mgr.set_download_progress(&model_id_clone, pct, &label);
                        drop(mgr);
                        emit_model_status(&mgr_out, &handle_out).await;
                    }
                }
            }
        }
    });

    // Drain stderr
    let stderr = child.stderr.take();
    let stderr_task = tauri::async_runtime::spawn(async move {
        if let Some(stderr) = stderr {
            use tokio::io::{AsyncReadExt, BufReader};
            let mut reader = BufReader::new(stderr);
            let mut buf = [0u8; 4096];
            while let Ok(n) = reader.read(&mut buf).await {
                if n == 0 { break; }
            }
        }
    });

    let _ = stdout_task.await;
    let _ = stderr_task.await;

    let exit_status = child.wait().await
        .map_err(|e| format!("Download process error: {}", e))?;

    if exit_status.success() {
        let mut mgr = manager.lock().await;
        mgr.set_download_done(&model_id, None);
        drop(mgr);
        emit_model_status(&manager, &handle).await;
        Ok(())
    } else {
        let msg = format!("Download failed for {}", model_id);
        let mut mgr = manager.lock().await;
        mgr.set_download_done(&model_id, Some(msg.clone()));
        drop(mgr);
        emit_model_status(&manager, &handle).await;
        Err(msg)
    }
}

/// Download Foundation-1 model files to %APPDATA%/Gary4JUCE/models/foundation-1/
///
/// Uses streaming requests with byte-level progress (same approach as Gary/Carey).
/// Downloads Foundation_1.safetensors and model_config.json from RoyalCities/Foundation-1.
pub async fn download_foundation_model(
    model_id: String,
    python_exe: PathBuf,
    model_dir: PathBuf,
    manager: Arc<Mutex<ModelManager>>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    std::fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Cannot create model dir: {}", e))?;

    let model_dir_str = model_dir.to_string_lossy().to_string();

    let script = format!(
        r#"
import sys, os, json, requests

repo_id = "RoyalCities/Foundation-1"
model_dir = r"{model_dir}"

files = [
    ("model_config.json", 0),
    ("Foundation_1.safetensors", 0),
]

def report(pct, msg):
    print(json.dumps({{"p": round(pct, 4), "m": msg}}), flush=True)

def fmt_size(b):
    if b < 1024: return str(b) + "B"
    if b < 1024**2: return f"{{b/1024:.0f}}KB"
    if b < 1024**3: return f"{{b/1024**2:.1f}}MB"
    return f"{{b/1024**3:.2f}}GB"

try:
    report(0.0, "Fetching file info...")

    from huggingface_hub import HfApi
    api = HfApi()
    info = api.repo_info(repo_id, files_metadata=True)

    # Build file list with real sizes
    size_map = {{}}
    for s in info.siblings:
        sz = s.size or (s.lfs.size if s.lfs else 0) or 0
        size_map[s.rfilename] = sz

    file_entries = [(f, size_map.get(f, 0)) for f, _ in files]
    total_bytes = sum(sz for _, sz in file_entries)
    completed_bytes = 0

    report(0.0, f"Downloading {{len(file_entries)}} files ({{fmt_size(total_bytes)}})")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {{}}
    if token:
        headers["Authorization"] = f"Bearer {{token}}"

    for i, (filename, file_size) in enumerate(file_entries):
        out_path = os.path.join(model_dir, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Skip if already exists
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            completed_bytes += file_size
            report(completed_bytes / max(total_bytes, 1), f"{{filename}} (cached)")
            continue

        # Small files: direct download
        if file_size < 5 * 1024 * 1024:
            report(completed_bytes / max(total_bytes, 1), f"{{filename}} ({{fmt_size(file_size)}})")
            url = f"https://huggingface.co/{{repo_id}}/resolve/main/{{filename}}"
            resp = requests.get(url, headers=headers, allow_redirects=True)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            completed_bytes += file_size
            report(completed_bytes / max(total_bytes, 1), f"{{filename}} done")
            continue

        # Large files: stream with progress
        url = f"https://huggingface.co/{{repo_id}}/resolve/main/{{filename}}"
        resp = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        resp.raise_for_status()

        received = 0
        last_pct = -1
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                received += len(chunk)
                current_total = completed_bytes + received
                pct = int(current_total * 100 / max(total_bytes, 1))
                if pct != last_pct:
                    report(current_total / max(total_bytes, 1), f"{{filename}} {{fmt_size(current_total)}}/{{fmt_size(total_bytes)}}")
                    last_pct = pct

        completed_bytes += file_size
        report(completed_bytes / max(total_bytes, 1), f"{{filename}} done ({{i+1}}/{{len(file_entries)}})")

    report(1.0, "done")
    print(json.dumps({{"status": "done"}}), flush=True)

except Exception as e:
    print(json.dumps({{"status": "error", "error": str(e)}}), flush=True)
    sys.exit(1)
"#,
        model_dir = model_dir_str,
    );

    let mut child = tokio::process::Command::new(python_exe.to_str().unwrap())
        .args(["-u", "-c", &script])
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start download: {}", e))?;

    // Read stdout for JSON progress
    let stdout = child.stdout.take();
    let model_id_clone = model_id.clone();
    let mgr_out = manager.clone();
    let handle_out = handle.clone();

    let stdout_task = tauri::async_runtime::spawn(async move {
        if let Some(stdout) = stdout {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(pct) = msg.get("p").and_then(|v| v.as_f64()) {
                        let label = msg.get("m")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let mut mgr = mgr_out.lock().await;
                        mgr.set_download_progress(&model_id_clone, pct, &label);
                        drop(mgr);
                        emit_model_status(&mgr_out, &handle_out).await;
                    }
                }
            }
        }
    });

    // Drain stderr
    let stderr = child.stderr.take();
    let stderr_task = tauri::async_runtime::spawn(async move {
        if let Some(stderr) = stderr {
            use tokio::io::{AsyncReadExt, BufReader};
            let mut reader = BufReader::new(stderr);
            let mut buf = [0u8; 4096];
            while let Ok(n) = reader.read(&mut buf).await {
                if n == 0 { break; }
            }
        }
    });

    let _ = stdout_task.await;
    let _ = stderr_task.await;

    let exit_status = child.wait().await
        .map_err(|e| format!("Download process error: {}", e))?;

    if exit_status.success() {
        let mut mgr = manager.lock().await;
        mgr.set_download_done(&model_id, None);
        drop(mgr);
        emit_model_status(&manager, &handle).await;
        Ok(())
    } else {
        let msg = format!("Download failed for {}", model_id);
        let mut mgr = manager.lock().await;
        mgr.set_download_done(&model_id, Some(msg.clone()));
        drop(mgr);
        emit_model_status(&manager, &handle).await;
        Err(msg)
    }
}

/// Emit model status update to the frontend
pub async fn emit_model_status(manager: &Arc<Mutex<ModelManager>>, handle: &tauri::AppHandle) {
    let mgr = manager.lock().await;
    let mut models = mgr.get_gary_models();
    models.extend(mgr.get_jerry_models());
    models.extend(mgr.get_carey_models());
    models.extend(mgr.get_foundation_models());
    let progress = mgr.get_download_progress();
    drop(mgr);

    use tauri::Emitter;
    let _ = handle.emit("models-updated", &models);
    let _ = handle.emit("download-progress", &progress);
}
