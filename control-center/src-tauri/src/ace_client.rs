//! A client for ace-server's job contract: POST a request (JSON, or multipart
//! form data with source audio), poll GET /job?id=N until it is done, then
//! fetch the result, which for /synth is multipart/mixed with one audio part
//! and one latent part per track.
use std::time::{Duration, Instant};

/// What a job is submitted with.
pub enum Body<'a> {
    Json(&'a serde_json::Value),
    /// The request JSON plus a source audio file, as the `request` and
    /// `audio` form parts.
    WithAudio(&'a serde_json::Value, &'a [u8]),
}

/// Submit to `path` (e.g. "/synth") and wait for the job to finish; returns
/// the result body and its content type.
pub async fn run(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    body: Body<'_>,
    timeout: Duration,
    what: &str,
) -> Result<(Vec<u8>, String), String> {
    let request = client.post(format!("{base}{path}"));
    let request = match body {
        Body::Json(json) => request.json(json),
        Body::WithAudio(json, audio) => {
            let (content_type, form) = form_data(json, audio);
            request.header(reqwest::header::CONTENT_TYPE, content_type).body(form)
        }
    };
    let response = request.send().await.map_err(|error| format!("{what} failed: {error}"))?;
    let status = response.status();
    let reply: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("{what} returned an unreadable reply: {error}"))?;
    if !status.is_success() {
        return Err(format!("{what} was refused: {}", reply["error"].as_str().unwrap_or(status.as_str())));
    }
    let id = reply["id"]
        .as_str()
        // Job ids are hex, e.g. "baa8fd4e81540fb3"; they go into a query string.
        .filter(|id| !id.is_empty() && id.len() <= 64 && id.bytes().all(|byte| byte.is_ascii_alphanumeric()))
        .ok_or_else(|| format!("{what} returned an invalid job"))?
        .to_string();
    let deadline = Instant::now() + timeout;
    loop {
        let status: serde_json::Value = client
            .get(format!("{base}/job?id={id}"))
            .send()
            .await
            .map_err(|error| error.to_string())?
            .error_for_status()
            .map_err(|error| error.to_string())?
            .json()
            .await
            .map_err(|error| error.to_string())?;
        match status["status"].as_str() {
            Some("done") => break,
            Some("running") => {}
            other => return Err(format!("{what} {}; inspect the server log", other.unwrap_or("failed"))),
        }
        if Instant::now() >= deadline {
            let _ = client.post(format!("{base}/job?id={id}&cancel=1")).send().await;
            return Err(format!("{what} timed out"));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let response = client
        .get(format!("{base}/job?id={id}&result=1"))
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| format!("{what} result: {error}"))?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = response.bytes().await.map_err(|error| error.to_string())?;
    Ok((bytes.to_vec(), content_type))
}

/// Render a /synth request and return its first track's audio.
pub async fn synth_audio(
    client: &reqwest::Client,
    base: &str,
    body: Body<'_>,
    timeout: Duration,
    what: &str,
) -> Result<Vec<u8>, String> {
    let (result, content_type) = run(client, base, "/synth", body, timeout, what).await?;
    parts(&result, &content_type)?
        .into_iter()
        .find(|(headers, _)| headers.to_ascii_lowercase().contains("content-type: audio/"))
        .map(|(_, body)| body.to_vec())
        .ok_or_else(|| format!("{what} returned no audio"))
}

/// The parts of a multipart body as (headers, body), in order.
pub fn parts<'a>(body: &'a [u8], content_type: &str) -> Result<Vec<(String, &'a [u8])>, String> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|param| param.strip_prefix("boundary="))
        .map(|boundary| boundary.trim_matches('"'))
        .filter(|boundary| !boundary.is_empty())
        .ok_or("The result is not multipart")?;
    // A part ends only at CRLF followed by the boundary, so audio bytes that
    // happen to contain the boundary text cannot end one early.
    let delimiter = format!("\r\n--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let mut rest = body
        .strip_prefix(&delimiter[2..])
        .ok_or("The multipart result does not start with its boundary")?;
    loop {
        if rest.starts_with(b"--") {
            return Ok(parts);
        }
        let rest_of_line = rest.strip_prefix(b"\r\n").ok_or("The multipart result is malformed")?;
        let headers_end = find(rest_of_line, b"\r\n\r\n").ok_or("A multipart part has no body")?;
        let headers = String::from_utf8_lossy(&rest_of_line[..headers_end]).into_owned();
        let content = &rest_of_line[headers_end + 4..];
        let end = find(content, &delimiter).ok_or("The multipart result is truncated")?;
        parts.push((headers, &content[..end]));
        rest = &content[end + delimiter.len()..];
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

/// A multipart/form-data body with the `request` JSON and `audio` parts.
fn form_data(json: &serde_json::Value, audio: &[u8]) -> (String, Vec<u8>) {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let boundary = format!("gary4local-{nonce:x}");
    let mut body = Vec::with_capacity(audio.len() + 1024);
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"request\"; filename=\"request.json\"\r\n\
             Content-Type: application/json\r\n\r\n{json}\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"source.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(audio);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_results_split_into_their_audio_and_latent_parts() {
        // The audio carries the boundary text without a CRLF before it.
        let body = b"--ace-batch-boundary\r\nContent-Type: audio/wav\r\n\r\nRIFF--ace-batch-boundary\r\n--x\r\n\
--ace-batch-boundary\r\nContent-Type: application/octet-stream\r\n\
Content-Disposition: form-data; name=\"latent\"\r\n\r\n\x00\x01\r\n--ace-batch-boundary--\r\n";
        let parts = parts(body, "multipart/mixed; boundary=ace-batch-boundary").unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].1, b"RIFF--ace-batch-boundary\r\n--x");
        assert!(parts[1].0.contains("name=\"latent\""));
        assert_eq!(parts[1].1, b"\x00\x01");
        assert!(super::parts(body, "audio/wav").is_err());
        assert!(super::parts(b"--ace-batch-boundary\r\nContent-Type: audio/wav\r\n\r\nRIFF", "multipart/mixed; boundary=ace-batch-boundary").is_err());
    }

    #[test]
    fn form_data_carries_the_request_and_the_audio() {
        let json = serde_json::json!({"task_type": "repaint"});
        let (content_type, body) = form_data(&json, b"RIFFdata");
        let boundary = content_type.strip_prefix("multipart/form-data; boundary=").unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with(&format!("--{boundary}\r\n")));
        assert!(text.contains("name=\"request\""));
        assert!(text.contains("{\"task_type\":\"repaint\"}"));
        assert!(text.contains("name=\"audio\"; filename=\"source.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFdata\r\n"));
        assert!(text.ends_with(&format!("--{boundary}--\r\n")));
    }
}
