//! Model downloads for native services, straight from Hugging Face with no
//! Python environment involved. Files land as plain files in one folder,
//! which is how the GGML servers find their weights, and each one is kept
//! only if its SHA-256 matches what the repo publishes.

use crate::model_manager::{emit_model_status, emit_model_status_from, ModelManager};
use crate::native_runtime::{download_verified, is_sha256, parse_sha256sums};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Download `files` from `repo` into `dest_dir` and report progress under
/// `model_id`, as the Python downloaders do.
pub async fn download_hf_files(
    model_id: String,
    repo: String,
    files: Vec<String>,
    dest_dir: PathBuf,
    manager: Arc<Mutex<ModelManager>>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    let result = download(&model_id, &repo, &files, &dest_dir, &manager, &handle).await;
    let error = result
        .as_ref()
        .err()
        .map(|error| format!("Download failed for {model_id}: {error}"));
    {
        let mut mgr = manager.lock().await;
        mgr.set_download_done(&model_id, error.clone());
    }
    emit_model_status(&manager, &handle).await;
    match error {
        Some(error) => {
            log::error!("{error}");
            Err(error)
        }
        None => Ok(()),
    }
}

/// Per-file size, and the LFS SHA-256 when the file is stored in LFS.
struct RemoteFile {
    size: Option<u64>,
    lfs_sha256: Option<String>,
}

async fn repo_tree(
    client: &reqwest::Client,
    repo: &str,
    headers: &reqwest::header::HeaderMap,
) -> HashMap<String, RemoteFile> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main");
    let Ok(response) = client.get(&url).headers(headers.clone()).send().await else {
        return HashMap::new();
    };
    let Ok(entries) = response.json::<Vec<serde_json::Value>>().await else {
        return HashMap::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let path = entry.get("path")?.as_str()?.to_string();
            let size = entry.get("size").and_then(|value| value.as_u64());
            let lfs_sha256 = entry
                .pointer("/lfs/oid")
                .and_then(|value| value.as_str())
                .filter(|oid| is_sha256(oid))
                .map(str::to_ascii_lowercase);
            Some((path, RemoteFile { size, lfs_sha256 }))
        })
        .collect()
}

async fn repo_sha256sums(
    client: &reqwest::Client,
    repo: &str,
    headers: &reqwest::header::HeaderMap,
) -> HashMap<String, String> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/SHA256SUMS");
    match client.get(&url).headers(headers.clone()).send().await {
        Ok(response) if response.status().is_success() => response
            .text()
            .await
            .map(|raw| parse_sha256sums(&raw))
            .unwrap_or_default(),
        _ => HashMap::new(),
    }
}

fn friendly_status(error: String) -> String {
    if error.contains("HTTP 401") || error.contains("HTTP 403") {
        format!("{error}. Hugging Face refused the request; check the token saved in gary4local.")
    } else {
        error
    }
}

async fn download(
    model_id: &str,
    repo: &str,
    files: &[String],
    dest_dir: &std::path::Path,
    manager: &Arc<Mutex<ModelManager>>,
    handle: &tauri::AppHandle,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(|error| format!("cannot create an HTTP client: {error}"))?;
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(token) = crate::read_hf_token() {
        if let Ok(value) = format!("Bearer {token}").parse() {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
    }

    // SHA256SUMS covers every file, LFS or not; the tree API supplies sizes
    // for the progress bar and an LFS hash for repos that publish no sums.
    let sums = repo_sha256sums(&client, repo, &headers).await;
    let tree = repo_tree(&client, repo, &headers).await;
    let mut plan = Vec::new();
    for file in files {
        let remote = tree.get(file);
        let sha256 = sums
            .get(file)
            .cloned()
            .or_else(|| remote.and_then(|remote| remote.lfs_sha256.clone()))
            .ok_or_else(|| {
                format!("{repo} publishes no checksum for {file}, so it will not be downloaded")
            })?;
        plan.push((file.clone(), sha256, remote.and_then(|remote| remote.size)));
    }
    let total: u64 = plan.iter().filter_map(|(_, _, size)| *size).sum();

    let mut done: u64 = 0;
    for (index, (file, sha256, size)) in plan.iter().enumerate() {
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        let label = format!("{} ({}/{})", file, index + 1, plan.len());
        let manager_progress = manager.clone();
        let handle_progress = handle.clone();
        let model = model_id.to_string();
        let mut last = Instant::now() - Duration::from_secs(1);
        let mut on_progress = move |received: u64, file_total: Option<u64>| {
            if last.elapsed() < Duration::from_millis(250) {
                return;
            }
            last = Instant::now();
            let overall = if total > 0 {
                (done + received) as f64 / total as f64
            } else {
                file_total
                    .filter(|total| *total > 0)
                    .map(|total| received as f64 / total as f64)
                    .unwrap_or(0.0)
            };
            let message = format!(
                "{label} {:.1}/{:.1} GB",
                received as f64 / 1e9,
                file_total.unwrap_or(0) as f64 / 1e9
            );
            // Never wait on the lock from inside the transfer loop.
            if let Ok(mut mgr) = manager_progress.try_lock() {
                mgr.set_download_progress(&model, overall.min(1.0), &message);
                emit_model_status_from(&mgr, &handle_progress);
            }
        };
        download_verified(
            &client,
            &url,
            headers.clone(),
            sha256,
            &dest_dir.join(file),
            &mut on_progress,
        )
        .await
        .map_err(friendly_status)?;
        done += size.unwrap_or(0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Talks to Hugging Face, so it only runs when asked:
    /// `cargo test -- --ignored downloads_a_published_yuey_file`
    #[test]
    #[ignore]
    fn downloads_a_published_yuey_file_and_keeps_it_only_when_verified() {
        tauri::async_runtime::block_on(async {
            let repo = crate::model_manager::YUEY_REPO;
            let file = "yue2-qwen.tiktoken";
            let client = reqwest::Client::new();
            let headers = reqwest::header::HeaderMap::new();

            let sums = repo_sha256sums(&client, repo, &headers).await;
            let tree = repo_tree(&client, repo, &headers).await;
            let sha256 = sums
                .get(file)
                .expect("SHA256SUMS lists the tokenizer")
                .clone();
            let size = tree
                .get(file)
                .and_then(|remote| remote.size)
                .expect("the tree API reports its size");

            let dir =
                std::env::temp_dir().join(format!("gary4local-hf-test-{}", std::process::id()));
            let dest = dir.join(file);
            let mut calls = 0u32;
            let mut on_progress = |_: u64, _: Option<u64>| calls += 1;
            download_verified(
                &client,
                &format!("https://huggingface.co/{repo}/resolve/main/{file}"),
                headers.clone(),
                &sha256,
                &dest,
                &mut on_progress,
            )
            .await
            .expect("the download verifies");
            assert_eq!(std::fs::metadata(&dest).unwrap().len(), size);
            assert!(calls > 0);

            // A second run finds the verified file and makes no request.
            let mut second_calls = 0u32;
            let mut quiet = |_: u64, _: Option<u64>| second_calls += 1;
            download_verified(
                &client,
                "https://invalid.example/never-fetched",
                headers,
                &sha256,
                &dest,
                &mut quiet,
            )
            .await
            .expect("a verified file is kept");
            assert_eq!(second_calls, 0);
            let _ = std::fs::remove_dir_all(&dir);
        });
    }
}
