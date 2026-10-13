//! CPU-only sidecar metadata suggestions through the shared native bundle.
//! Compressed formats are decoded by FFmpeg using argument arrays, never a shell.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ANALYSIS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Suggestion {
    pub bpm: Option<i64>,
    pub keyscale: String,
    pub suggestion: String,
    pub bpm_confidence: Option<f64>,
    pub key_confidence: Option<f64>,
}

#[derive(Deserialize)]
struct Output {
    ok: bool,
    schema_version: Option<u32>,
    bpm: Option<i64>,
    #[serde(default)]
    keyscale: String,
    #[serde(default)]
    suggestion: String,
    bpm_confidence: Option<f64>,
    key_confidence: Option<f64>,
    error: Option<String>,
}

pub fn workload_id() -> String {
    format!(
        "metadata-analysis-{}-{}",
        std::process::id(),
        NEXT_ANALYSIS.fetch_add(1, Ordering::Relaxed)
    )
}

async fn run(
    mut command: tokio::process::Command,
    label: &str,
) -> Result<std::process::Output, String> {
    command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    crate::workload_job::configure_tokio_command(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| format!("Cannot start {label}: {error}"))?;
    if let Err(error) = crate::workload_job::enroll_tokio_child(&child) {
        let _ = child.kill().await;
        return Err(error);
    }
    child
        .wait_with_output()
        .await
        .map_err(|error| format!("Cannot read {label} output: {error}"))
}

pub async fn probe(tool: &Path) -> Result<(), String> {
    if !tool.is_file() {
        return Err("Prepare a compatible SA3 C++ runtime with the native audio analysis tool before migration.".into());
    }
    let mut command = tokio::process::Command::new(tool);
    command.arg("--control-info");
    let output = run(command, "native audio analysis").await?;
    let info: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Invalid native audio analysis capabilities: {error}"))?;
    if !output.status.success()
        || info["schema_version"] != 1
        || info["cpu_only"] != true
        || info["wav_input"] != true
    {
        return Err(
            "Installed SA3 audio analysis tool is incompatible; prepare a compatible runtime."
                .into(),
        );
    }
    Ok(())
}

fn parse(output: &std::process::Output, native: bool) -> Result<Suggestion, String> {
    let parsed: Output = serde_json::from_slice(&output.stdout).map_err(|error| {
        let detail = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(1000)
            .collect::<String>();
        format!("Cannot read audio analysis result: {error}. {detail}")
    })?;
    if !output.status.success() || !parsed.ok {
        return Err(parsed
            .error
            .unwrap_or_else(|| "BPM/key analysis failed.".into()));
    }
    if native && parsed.schema_version != Some(1) {
        return Err("Native audio analysis returned an unsupported result schema.".into());
    }
    if parsed.bpm.is_some_and(|bpm| !(1..=400).contains(&bpm))
        || parsed
            .bpm_confidence
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || parsed
            .key_confidence
            .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err("Audio analysis returned invalid metadata.".into());
    }
    Ok(Suggestion {
        bpm: parsed.bpm,
        keyscale: parsed.keyscale,
        suggestion: parsed.suggestion,
        bpm_confidence: parsed.bpm_confidence,
        key_confidence: parsed.key_confidence,
    })
}

struct DecodedAudio {
    path: PathBuf,
    boundary: PathBuf,
}
impl Drop for DecodedAudio {
    fn drop(&mut self) {
        let _ = crate::remove_managed_path(&self.path, &self.boundary);
    }
}

pub async fn analyze(root: &Path, tool: &Path, audio: &Path) -> Result<Suggestion, String> {
    probe(tool).await?;
    let mut decoded = None;
    if !audio
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
    {
        let boundary = crate::sa3_training::checked_folder(root, &["sa3", "analysis"])?;
        let path = boundary.join(format!("{}.wav", workload_id()));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| format!("Cannot reserve decoded analysis audio: {error}"))?;
        decoded = Some(DecodedAudio {
            path: path.clone(),
            boundary,
        });
        let mut command = tokio::process::Command::new("ffmpeg");
        command
            .args(["-nostdin", "-v", "error", "-y", "-i"])
            .arg(audio)
            .args(["-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "wav"])
            .arg(&path);
        let output = run(command, "FFmpeg (required for compressed audio analysis)").await?;
        if !output.status.success() {
            return Err(format!(
                "Audio decoding failed: {}",
                String::from_utf8_lossy(&output.stderr)
                    .chars()
                    .take(1000)
                    .collect::<String>()
            ));
        }
    }
    let input = decoded
        .as_ref()
        .map_or(audio, |decoded| decoded.path.as_path());
    let mut command = tokio::process::Command::new(tool);
    command.arg("--in").arg(input);
    parse(&run(command, "native audio analysis").await?, true)
}

/// Compatibility before the profile's native activation. Cleanup cannot use
/// this branch, and a selected native profile never falls back to Python.
pub async fn analyze_legacy(root: &Path, audio: &Path) -> Result<Suggestion, String> {
    let python = root.join("services/sa3/env/Scripts/python.exe");
    if !python.is_file() {
        return Err("Prepare SA3's C++ runtime before using BPM/key suggestions.".into());
    }
    let script = root.join("services/sa3/analyze_audio.py");
    let mut command = tokio::process::Command::new(python);
    command
        .arg(script)
        .arg(audio)
        .current_dir(root.join("services/sa3"));
    parse(&run(command, "legacy audio analysis").await?, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn missing_native_analysis_tool_reports_required_preparation() {
        let root = std::env::temp_dir().join(workload_id());
        let error = analyze(&root, &root.join("absent.exe"), &root.join("audio.wav"))
            .await
            .unwrap_err();
        assert!(error.contains("compatible SA3 C++ runtime"));
        assert!(!root.exists(), "preflight must not create files");
    }

    #[tokio::test]
    #[ignore = "uses a built native analysis tool and installed FFmpeg"]
    async fn real_native_analysis_supports_formats_and_literal_paths_without_python() {
        let tool = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_ANALYSIS_TOOL").expect("native tool required"),
        );
        let root = PathBuf::from(
            std::env::var_os("GARY4LOCAL_SA3_ANALYSIS_ROOT").expect("isolated root required"),
        );
        assert!(!root.exists(), "fresh isolated root required");
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("triad ♭ %TEMP% & quote's.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&source, spec).unwrap();
        for sample in 0..48000 * 8 {
            let time = sample as f64 / 48000.0;
            let value = [261.625565, 329.627557, 391.995436]
                .iter()
                .map(|hz| (std::f64::consts::TAU * hz * time).sin())
                .sum::<f64>()
                / 4.0;
            for _ in 0..2 {
                writer.write_sample((value * 8388607.0) as i32).unwrap();
            }
        }
        writer.finalize().unwrap();
        let source_hash = crate::native_runtime::sha256_file(&source).await.unwrap();
        let result = analyze(&root, &tool, &source).await.unwrap();
        assert_eq!(result.keyscale, "C major");
        for extension in ["flac", "mp3", "ogg", "opus", "m4a", "aiff", "aif"] {
            let path = root.join(format!("compressed ♭ %TEMP% & quote's.{extension}"));
            let mut command = tokio::process::Command::new("ffmpeg");
            command
                .args(["-nostdin", "-v", "error", "-i"])
                .arg(&source)
                .arg(&path);
            let output = run(command, "FFmpeg test fixture encoder").await.unwrap();
            assert!(
                output.status.success(),
                "{extension}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let hash = crate::native_runtime::sha256_file(&path).await.unwrap();
            let result = analyze(&root, &tool, &path).await.unwrap();
            assert_eq!(result.keyscale, "C major", "{extension}");
            assert_eq!(
                crate::native_runtime::sha256_file(&path).await.unwrap(),
                hash
            );
            assert!(std::fs::read_dir(root.join("sa3/analysis"))
                .unwrap()
                .next()
                .is_none());
        }
        let invalid = root.join("broken.flac");
        std::fs::write(&invalid, b"invalid audio").unwrap();
        assert!(analyze(&root, &tool, &invalid)
            .await
            .unwrap_err()
            .contains("decoding failed"));
        assert!(std::fs::read_dir(root.join("sa3/analysis"))
            .unwrap()
            .next()
            .is_none());
        assert_eq!(
            crate::native_runtime::sha256_file(&source).await.unwrap(),
            source_hash
        );
        assert!(!root.join("services/sa3/env").exists());
        println!("Verified native sidecar analysis without Python, including all eight dataset extensions: {}", root.display());
    }
}
