//! Shared verified/resumable runtime downloads.
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

pub fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit())
}

/// `sha256sum` output: `<hash>  <name>` per line, name optionally `*`-prefixed.
pub fn parse_sha256sums(raw: &str) -> HashMap<String, String> {
    raw.lines()
        .filter_map(|line| {
            let (hash, name) = line.trim().split_once(char::is_whitespace)?;
            let name = name.trim().trim_start_matches('*');
            (is_sha256(hash) && !name.is_empty())
                .then(|| (name.to_string(), hash.to_ascii_lowercase()))
        })
        .collect()
}

pub async fn sha256_file(path: &Path) -> Result<String, String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await
    .map_err(|error| format!("checksum task failed: {error}"))?
}

pub fn partial_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default();
    dest.with_file_name(format!("{name}.partial"))
}

/// Download `url` to `dest`, resuming a `.partial` left by an interrupted
/// attempt, and keep it only if its SHA-256 is the expected one. A file
/// already at `dest` with the right hash is kept without a request.
pub async fn download_verified(
    client: &reqwest::Client,
    url: &str,
    headers: reqwest::header::HeaderMap,
    expected_sha256: &str,
    dest: &Path,
    on_progress: &mut (dyn FnMut(u64, Option<u64>) + Send),
) -> Result<(), String> {
    let expected = expected_sha256.to_ascii_lowercase();
    let name = dest
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| url.to_string());

    if dest.is_file() && sha256_file(dest).await? == expected {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }

    let partial = partial_path(dest);
    let mut resume_from = tokio::fs::metadata(&partial)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let mut request = client.get(url).headers(headers);
    if resume_from > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
    }
    let mut response = request
        .send()
        .await
        .map_err(|error| format!("downloading {name} failed: {error}"))?;
    let status = response.status();

    // 416 means the partial file already holds every byte; it is checked below.
    if status != reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        if !status.is_success() {
            return Err(format!("downloading {name} failed: HTTP {status}"));
        }
        let resumed = resume_from > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
        if !resumed {
            resume_from = 0;
        }
        let total = response.content_length().map(|length| length + resume_from);
        let mut options = tokio::fs::OpenOptions::new();
        options.create(true).write(true);
        if resumed {
            options.append(true);
        } else {
            options.truncate(true);
        }
        let file = options
            .open(&partial)
            .await
            .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;
        // Network chunks can be tiny. Batch them to avoid a blocking file-I/O
        // task (and Windows file-system overhead) for every received chunk.
        let mut file = tokio::io::BufWriter::with_capacity(1 << 20, file);

        let mut received = resume_from;
        on_progress(received, total);
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("downloading {name} failed: {error}"))?
        {
            file.write_all(&chunk)
                .await
                .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;
            received += chunk.len() as u64;
            on_progress(received, total);
        }
        file.flush()
            .await
            .map_err(|error| format!("cannot write {}: {error}", partial.display()))?;
    }

    let actual = sha256_file(&partial).await?;
    if actual != expected {
        let _ = tokio::fs::remove_file(&partial).await;
        return Err(format!(
            "{name} did not match its published checksum (expected {expected}, got {actual}); the download was discarded"
        ));
    }
    if dest.exists() {
        let _ = tokio::fs::remove_file(dest).await;
    }
    tokio::fs::rename(&partial, dest)
        .await
        .map_err(|error| format!("cannot move {name} into place: {error}"))
}
