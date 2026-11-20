//! Native Carey dataset captioning through ace-server's /caption. The server
//! suggests a caption, tempo, key and language from each song and returns its
//! local BPM/key analysis beside them; what Carey writes to a sidecar follows
//! the Python job's policy, ported here: a filename BPM is ground truth, local
//! analysis corrects the LM where it is confident, LM lyrics are never written
//! (they are invented, not transcribed), and unusable captions are retried on
//! shorter excerpts before failing.

use crate::ace_client::{self, Body, JobError};
use crate::dataset::{self, CareyAceSidecarFields};
pub struct CaptionEndpoint<'a> {
    pub client: &'a reqwest::Client,
    pub base: &'a str,
}
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

/// /caption takes up to ten minutes of WAV.
const MAX_CAPTION_SECONDS: f64 = 600.0;
/// The Python job's fallback excerpt for a caption that fails quality checks.
const EXCERPT_SECONDS: f64 = 120.0;
const SHORT_EXCERPT_SECONDS: f64 = 30.0;
/// The Python job's thresholds; the native estimators score confidence the
/// same way (best over runner-up for tempo, best minus runner-up for key).
const BPM_TOLERANCE: f64 = 5.0;
const BPM_MIN_CONFIDENCE: f64 = 1.2;
const KEY_MIN_CONFIDENCE: f64 = 0.15;

pub struct Options {
    pub dataset: PathBuf,
    /// Exact GGUF names in the server's registry.
    pub lm: String,
    pub dit: String,
    pub vae: String,
    pub overwrite: bool,
    pub bpm_analysis: bool,
    pub key_analysis: bool,
    pub trigger: String,
}

// ---------------------------------------------------------------------------
// Policy, ported from services/carey (bpm_analysis.py, key_analysis.py and
// train_lora_job.py) so a native sidecar matches a Python one.

#[derive(Debug, Clone, PartialEq)]
pub struct BpmDecision {
    pub bpm: Option<u32>,
    pub source: &'static str,
    pub lm: Option<f64>,
    pub local: Option<f64>,
    pub filename: Option<u32>,
}

/// A BPM written in a file name, like `loop_bpm120.wav` or `bpm-92-take.wav`.
pub fn filename_bpm(name: &str) -> Option<u32> {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = PATTERN
        .get_or_init(|| Regex::new(r"(?i)(?:^|[_-])bpm[_-]?(\d{2,3})(?:[_-]|\.|$)").unwrap());
    let bpm: u32 = pattern.captures(name)?.get(1)?.as_str().parse().ok()?;
    (1..=400).contains(&bpm).then_some(bpm)
}

fn bpm_value(value: &serde_json::Value) -> Option<f64> {
    let bpm = value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse().ok())?;
    (1.0..=400.0).contains(&bpm).then_some(bpm)
}

fn round_bpm(value: Option<f64>) -> Option<u32> {
    value
        .map(|bpm| bpm.round() as u32)
        .filter(|bpm| (1..=400).contains(bpm))
}

/// `bpm` and its halvings and doublings within 40-240.
fn tempo_family(bpm: f64) -> Vec<f64> {
    let mut family = vec![bpm];
    let mut current = bpm;
    while current / 2.0 >= 40.0 {
        current /= 2.0;
        family.push(current);
    }
    current = bpm;
    while current * 2.0 <= 240.0 {
        current *= 2.0;
        family.push(current);
    }
    family
}

pub fn choose_bpm(
    filename: Option<u32>,
    lm: Option<f64>,
    local: Option<(f64, f64)>,
) -> BpmDecision {
    let local_bpm = local.map(|(bpm, _)| bpm);
    let decision = |bpm, source| BpmDecision {
        bpm,
        source,
        lm,
        local: local_bpm,
        filename,
    };
    if let Some(bpm) = filename {
        return decision(Some(bpm), "filename");
    }
    let Some((bpm, confidence)) = local else {
        return decision(round_bpm(lm), if lm.is_some() { "lm" } else { "missing" });
    };
    let Some(lm_bpm) = lm else {
        return decision(round_bpm(Some(bpm)), "local");
    };
    if confidence < BPM_MIN_CONFIDENCE {
        return decision(round_bpm(lm), "lm_local_low_confidence");
    }
    if tempo_family(bpm)
        .iter()
        .any(|candidate| (lm_bpm - candidate).abs() <= BPM_TOLERANCE)
    {
        return decision(round_bpm(lm), "lm_agrees_with_local");
    }
    decision(round_bpm(Some(bpm)), "local_overrode_lm")
}

#[derive(Debug, Clone, PartialEq)]
pub struct KeyDecision {
    pub keyscale: String,
    pub source: &'static str,
    pub lm: String,
    pub local: String,
}

const NOTES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// "F minor" from "f min", "Bb major", "F♯_minor"...; empty when unreadable.
pub fn normalize_keyscale(value: &str) -> String {
    let text = value.replace('_', " ").replace('-', " ");
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() < 2 || text.trim().eq_ignore_ascii_case("n/a") {
        return String::new();
    }
    let note = parts[0].to_uppercase().replace('♭', "B").replace('♯', "#");
    let note = match note.as_str() {
        "DB" => "C#",
        "EB" => "D#",
        "GB" => "F#",
        "AB" => "G#",
        "BB" => "A#",
        other => other,
    };
    if !NOTES.contains(&note) {
        return String::new();
    }
    let mode = match parts[1].to_lowercase().as_str() {
        "maj" | "major" => "major",
        "min" | "minor" => "minor",
        _ => return String::new(),
    };
    format!("{note} {mode}")
}

pub fn choose_key(lm: &str, local: Option<(&str, f64)>) -> KeyDecision {
    let lm = normalize_keyscale(lm);
    let Some((local_key, confidence)) = local else {
        let source = if lm.is_empty() { "missing" } else { "lm" };
        return KeyDecision {
            keyscale: lm.clone(),
            source,
            lm,
            local: String::new(),
        };
    };
    let local = local_key.to_string();
    if !lm.is_empty() && lm == normalize_keyscale(&local) {
        return KeyDecision {
            keyscale: lm.clone(),
            source: "lm_agrees_with_local",
            lm,
            local,
        };
    }
    if confidence >= KEY_MIN_CONFIDENCE {
        let source = if lm.is_empty() {
            "local"
        } else {
            "local_overrode_lm"
        };
        return KeyDecision {
            keyscale: local.clone(),
            source,
            lm,
            local,
        };
    }
    let source = if lm.is_empty() {
        "missing"
    } else {
        "lm_local_low_confidence"
    };
    KeyDecision {
        keyscale: lm.clone(),
        source,
        lm,
        local,
    }
}

/// Why an LM caption or genre is unusable, as the Python job judges it.
pub fn quality_error(text: &str, field: &str) -> Option<String> {
    static FIELDS: OnceLock<Regex> = OnceLock::new();
    static WORDS: OnceLock<Regex> = OnceLock::new();
    if text.trim().is_empty() {
        return Some(format!("{field} is empty"));
    }
    if text.contains("<|audio_code_") {
        return Some(format!("{field} contains raw ACE audio-code tokens"));
    }
    if text.contains('\u{fffd}') || text.contains("ï¿½") {
        return Some(format!("{field} contains invalid replacement characters"));
    }
    let compact: String = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let distinct: std::collections::HashSet<char> = compact.chars().collect();
    if compact.chars().count() >= 24 && distinct.len() <= 3 {
        return Some(format!("{field} is dominated by repeated characters"));
    }
    // Twenty-one or more of one punctuation mark in a row.
    let mut run = (' ', 0usize);
    for character in compact.chars() {
        run = if character == run.0 {
            (character, run.1 + 1)
        } else {
            (character, 1)
        };
        if "!?._=-".contains(character) && run.1 >= 21 {
            return Some(format!("{field} is dominated by repeated punctuation"));
        }
    }
    if field == "caption" {
        let length = text.trim().chars().count();
        let letters = text
            .chars()
            .filter(|character| character.is_alphabetic())
            .count();
        if length >= 80 && (letters as f64) / (text.chars().count().max(1) as f64) < 0.15 {
            return Some(format!("{field} has too little word content"));
        }
        let words = WORDS.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z'-]{2,}").unwrap());
        if length >= 40 && words.find_iter(text).count() < 3 {
            return Some(format!("{field} has too little descriptive text"));
        }
        let fields = FIELDS.get_or_init(|| {
            Regex::new(
                r"(?i)(?:^|[^a-z])(?:bpm|duration|genres?|keyscale|language|timesignature)\s*:",
            )
            .unwrap()
        });
        if fields.is_match(text) {
            return Some(format!(
                "{field} appears to contain embedded metadata fields"
            ));
        }
    }
    None
}

fn lyrics_are_structural_only(lyrics: &str) -> bool {
    lyrics
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .all(|line| {
            line.len() > 2
                && line.starts_with('[')
                && line.ends_with(']')
                && !line[1..line.len() - 1].contains(']')
        })
}

/// Whether the LM heard an instrumental: it says so, or it wrote no lyrics, or
/// only section markers. ace-server leaves `is_instrumental` null for now.
pub fn is_instrumental(result: &serde_json::Value) -> bool {
    match &result["is_instrumental"] {
        serde_json::Value::Bool(explicit) => return *explicit,
        serde_json::Value::String(text) if !text.trim().is_empty() => {
            return ["1", "true", "yes", "y", "on"].contains(&text.trim().to_lowercase().as_str())
        }
        _ => {}
    }
    let lyrics = result["lyrics"].as_str().unwrap_or_default().trim();
    lyrics.is_empty()
        || lyrics.to_lowercase().contains("[instrumental]")
        || lyrics_are_structural_only(lyrics)
}

/// The tempo and key for `audio` from one /caption result: a filename BPM
/// first, then the LM corrected by local analysis where it is confident.
fn tempo_and_key(
    audio: &Path,
    result: &serde_json::Value,
    options: &Options,
) -> (BpmDecision, KeyDecision) {
    let local = &result["local_analysis"];
    let local_bpm = options
        .bpm_analysis
        .then(|| Some((bpm_value(&local["bpm"])?, local["bpm_confidence"].as_f64()?)))
        .flatten();
    let local_key = options
        .key_analysis
        .then(|| {
            let key = local["keyscale"]
                .as_str()
                .filter(|key| !key.trim().is_empty())?;
            Some((key, local["key_confidence"].as_f64()?))
        })
        .flatten();
    let name = audio
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let bpm = choose_bpm(filename_bpm(&name), bpm_value(&result["bpm"]), local_bpm);
    let key = choose_key(result["keyscale"].as_str().unwrap_or_default(), local_key);
    (bpm, key)
}

/// The Carey sidecar fields for `audio` from one /caption result.
pub fn sidecar_fields(
    audio: &Path,
    result: &serde_json::Value,
    options: &Options,
) -> CareyAceSidecarFields {
    let (bpm, key) = tempo_and_key(audio, result, options);
    let instrumental = is_instrumental(result);
    let text = |value: Option<f64>| value.map(|number| number.to_string()).unwrap_or_default();
    CareyAceSidecarFields {
        caption: result["caption"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string(),
        genre: result["genre"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string(),
        bpm: bpm.bpm.map(|bpm| bpm.to_string()).unwrap_or_default(),
        bpm_source: bpm.source.into(),
        lm_bpm: text(bpm.lm),
        local_bpm: text(bpm.local),
        filename_bpm: bpm.filename.map(|bpm| bpm.to_string()).unwrap_or_default(),
        keyscale: key.keyscale,
        key_source: key.source.into(),
        lm_keyscale: key.lm,
        local_keyscale: key.local,
        // The LM's time signature, as the editor's optional field; staging
        // reads "4" or "4/4", and an empty one trains as 4/4.
        timesignature: result["time_signature"]
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string(),
        language: if instrumental {
            "unknown".into()
        } else {
            result["language"]
                .as_str()
                .unwrap_or_default()
                .trim()
                .to_string()
        },
        is_instrumental: instrumental,
        custom_tag: options.trigger.trim().to_string(),
        // Bring your own lyrics: the LM's are invented. Instrumental sidecars
        // get the canonical marker when rendered.
        lyrics: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Running it

/// `audio` as WAV bytes /caption accepts. A WAV that fits the requested window
/// goes through directly. FFmpeg decodes other inputs and makes excerpts.
async fn caption_audio(
    audio: &Path,
    scratch: &Path,
    excerpt: Option<(f64, Option<f64>)>,
) -> Result<Vec<u8>, String> {
    let is_wav = audio
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
    let duration = if is_wav {
        hound::WavReader::open(audio)
            .ok()
            .map(|reader| reader.duration() as f64 / reader.spec().sample_rate as f64)
    } else {
        None
    };
    let window = excerpt.or_else(|| {
        duration
            .filter(|seconds| *seconds > MAX_CAPTION_SECONDS)
            .map(|_| (EXCERPT_SECONDS, None))
    });
    if duration.is_some_and(|seconds| window.is_none_or(|(limit, _)| seconds <= limit)) {
        return std::fs::read(audio).map_err(|error| error.to_string());
    }
    let decoded = scratch.join("caption.wav");
    let _ = std::fs::remove_file(&decoded);
    let mut command = tokio::process::Command::new("ffmpeg");
    #[cfg(windows)]
    command.creation_flags(0x08000000);
    command.args(["-nostdin", "-v", "error", "-y"]);
    if let Some((seconds, offset)) = window {
        // Keep the whole excerpt within a known duration.
        let start = duration.map_or(0.0, |length| {
            let last_start = (length - seconds).max(0.0);
            offset.unwrap_or(last_start / 2.0).min(last_start)
        });
        command
            .arg("-ss")
            .arg(start.to_string())
            .arg("-t")
            .arg(seconds.to_string());
    }
    command
        .arg("-i")
        .arg(audio)
        .args(["-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "wav"])
        .arg(&decoded)
        .kill_on_drop(true);
    let output = command
        .output()
        .await
        .map_err(|error| format!("FFmpeg is needed to caption {}: {error}", audio.display()))?;
    if !output.status.success() {
        return Err(format!(
            "Could not decode {}: {}",
            audio.display(),
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(500)
                .collect::<String>()
        ));
    }
    let bytes = std::fs::read(&decoded).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_file(&decoded);
    // A decoded song past the limit is cut to the excerpt instead.
    let seconds = hound::WavReader::new(std::io::Cursor::new(&bytes))
        .map(|reader| reader.duration() as f64 / reader.spec().sample_rate as f64)
        .unwrap_or(0.0);
    if window.is_none() && seconds > MAX_CAPTION_SECONDS {
        return Box::pin(caption_audio(audio, scratch, Some((EXCERPT_SECONDS, None)))).await;
    }
    Ok(bytes)
}

async fn caption_once(
    server: &CaptionEndpoint<'_>,
    options: &Options,
    audio: &[u8],
    what: &str,
) -> Result<serde_json::Value, JobError> {
    let request =
        serde_json::json!({"lm_model": options.lm, "synth_model": options.dit, "vae": options.vae});
    let (body, _) = ace_client::run_job(
        server.client,
        server.base,
        "/caption",
        Body::WithAudio(&request, audio),
        Duration::from_secs(900),
        what,
    )
    .await?;
    serde_json::from_slice(&body)
        .map_err(|error| JobError::Other(format!("{what} returned unreadable JSON: {error}")))
}

/// Retry model failures and unusable results on shorter excerpts.
/// Connection errors and rejected requests still stop the run immediately.
async fn caption_checked(
    server: &CaptionEndpoint<'_>,
    options: &Options,
    audio: &Path,
    scratch: &Path,
    what: &str,
    check: impl Fn(&serde_json::Value) -> Option<String>,
    cancelled: impl Fn() -> bool,
) -> Result<serde_json::Value, JobError> {
    let mut failures = Vec::new();
    let mut failed_job = false;
    for (description, window) in [
        ("", None),
        (" (120-second excerpt)", Some((EXCERPT_SECONDS, None))),
        (" (30-second early excerpt)", Some((SHORT_EXCERPT_SECONDS, Some(20.0)))),
        (" (30-second middle excerpt)", Some((SHORT_EXCERPT_SECONDS, None))),
    ] {
        if cancelled() {
            return Err(JobError::Other("Captioning cancelled.".into()));
        }
        // If the model could not finish at all, try a much shorter section first.
        if failed_job && window == Some((EXCERPT_SECONDS, None)) {
            continue;
        }
        let attempt = format!("{what}{description}");
        let audio = caption_audio(audio, scratch, window)
            .await
            .map_err(|error| format!("{attempt}: {error}"))?;
        match caption_once(server, options, &audio, &attempt).await {
            Ok(result) => match check(&result) {
                Some(reason) => failures.push(format!("{attempt}: {reason}")),
                None => return Ok(result),
            },
            Err(JobError::Failed(message)) => {
                failed_job = true;
                failures.push(message);
            }
            Err(error) => return Err(error),
        }
    }
    Err(JobError::Failed(failures.join("\n")))
}

/// What a caption run reports as it goes.
pub struct Progress<'a> {
    pub index: usize,
    pub total: usize,
    pub name: &'a str,
}

/// Caption every song in `options.dataset` that has no sidecar (or every
/// song, with `overwrite`), writing Carey sidecars beside them. `cancelled`
/// is checked between songs and retries. `on_unusable` can stop the batch
/// or continue after a song exhausts its retries without writing a sidecar.
/// Returns how many were captioned and how many songs the dataset holds.
pub async fn caption_dataset(
    server: &CaptionEndpoint<'_>,
    options: &Options,
    scratch: &Path,
    mut progress: impl FnMut(Progress<'_>),
    cancelled: impl Fn() -> bool,
    mut on_unusable: impl FnMut(String) -> Result<(), String>,
) -> Result<(usize, usize), String> {
    let entries = dataset::read_carey_ace_dataset_sidecars(&options.dataset.to_string_lossy())?;
    let pending: Vec<_> = entries
        .iter()
        .filter(|entry| options.overwrite || !entry.exists)
        .collect();
    let mut captioned = 0;
    for (index, entry) in pending.iter().enumerate() {
        if cancelled() {
            return Err("Captioning cancelled.".into());
        }
        progress(Progress {
            index: index + 1,
            total: pending.len(),
            name: &entry.relative_path,
        });
        let audio = PathBuf::from(&entry.audio_path);
        let what = format!("Captioning {}", entry.relative_path);
        let check = |result: &serde_json::Value| {
            quality_error(result["caption"].as_str().unwrap_or_default(), "caption").or_else(|| {
                let genre = result["genre"].as_str().unwrap_or_default();
                (!genre.trim().is_empty())
                    .then(|| quality_error(genre, "genre"))
                    .flatten()
            })
        };
        let result = match caption_checked(server, options, &audio, scratch, &what, check, &cancelled).await {
            Ok(result) => result,
            Err(JobError::Failed(error)) => {
                on_unusable(error)?;
                continue;
            }
            Err(error) => return Err(error.message()),
        };
        let fields = sidecar_fields(&audio, &result, options);
        write_sidecar(
            Path::new(&entry.sidecar_path),
            &dataset::render_carey_ace_sidecar(&fields),
            options.overwrite,
        )
        .map_err(|error| format!("Cannot write {}: {error}", entry.sidecar_path))?;
        captioned += 1;
    }
    Ok((captioned, entries.len()))
}

// ---------------------------------------------------------------------------
// SA3 auto-label: the same captioner, keeping only the genre, tempo and key
// for SA3's one-line sidecars. Ported from services/carey/sa3_autolabel.py.

/// The LM's genre for an SA3 sidecar, or why it is unusable.
pub fn sa3_genre(result: &serde_json::Value) -> Result<String, String> {
    let genre = result["genre"].as_str().unwrap_or_default().trim();
    if genre.is_empty()
        || ["n/a", "na", "none", "null", "unknown"].contains(&genre.to_lowercase().as_str())
    {
        return Err("no genre".into());
    }
    quality_error(genre, "genre").map_or_else(|| Ok(genre.to_string()), Err)
}

/// An SA3 sidecar: "genre, 145 bpm, C minor", or with `labeled` the style of
/// SA3's own training data.
pub fn sa3_sidecar(labeled: bool, genre: &str, bpm: Option<u32>, keyscale: &str) -> String {
    if labeled {
        let mut parts = vec![
            "TrackType: Music".to_string(),
            "VocalType: Instrumental".to_string(),
        ];
        parts.extend((!genre.is_empty()).then(|| format!("Genre: {genre}")));
        parts.extend(bpm.map(|bpm| format!("BPM: {bpm}")));
        parts.extend((!keyscale.is_empty()).then(|| format!("Key: {keyscale}")));
        return parts.join(", ");
    }
    let bpm = bpm.map(|bpm| format!("{bpm} bpm")).unwrap_or_default();
    [genre, &bpm, keyscale]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Label every song in `files`, overwriting the `.txt` sidecar beside each.
/// `cancelled` is checked between songs.
pub async fn label_sa3_dataset(
    server: &CaptionEndpoint<'_>,
    options: &Options,
    files: &[PathBuf],
    labeled: bool,
    scratch: &Path,
    mut progress: impl FnMut(Progress<'_>),
    cancelled: impl Fn() -> bool,
) -> Result<(), String> {
    for (index, audio) in files.iter().enumerate() {
        if cancelled() {
            return Err("Auto-label cancelled.".into());
        }
        let name = audio
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        progress(Progress {
            index: index + 1,
            total: files.len(),
            name: &name,
        });
        let what = format!("Labeling {name}");
        let result = caption_checked(
            server, options, audio, scratch, &what, |result| sa3_genre(result).err(), &cancelled,
        ).await.map_err(JobError::message)?;
        let genre = sa3_genre(&result)?;
        let (bpm, key) = tempo_and_key(audio, &result, options);
        let sidecar = audio.with_extension("txt");
        write_sidecar(
            &sidecar,
            &(sa3_sidecar(labeled, &genre, bpm.bpm, &key.keyscale) + "\n"),
            options.overwrite,
        )
        .map_err(|error| format!("Cannot write {}: {error}", sidecar.display()))?;
    }
    Ok(())
}

fn write_sidecar(path: &Path, text: &str, overwrite: bool) -> std::io::Result<()> {
    use std::io::Write;
    if overwrite {
        crate::dataset::check_writable_sidecar(path)?;
        return std::fs::write(path, text);
    }
    match std::fs::File::create_new(path) {
        Ok(mut file) => file.write_all(text.as_bytes()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        extract::Query,
        routing::{get, post},
        Json, Router,
    };
    use serde_json::{json, Value};
    use std::{
        collections::HashMap,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    #[tokio::test]
    async fn retries_when_the_caption_job_fails_without_a_result() {
        let requests = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/caption",
                post({
                    let requests = requests.clone();
                    move || async move {
                        let id = requests.fetch_add(1, Ordering::Relaxed) + 1;
                        Json(json!({"id": id.to_string()}))
                    }
                }),
            )
            .route(
                "/job",
                get(|Query(query): Query<HashMap<String, String>>| async move {
                    if !matches!(query["id"].as_str(), "3" | "7") {
                        Json(json!({"status":"failed","error":"Token limit reached"}))
                    } else if query.contains_key("result") {
                        Json(json!({"caption":"Warm synth chords"}))
                    } else {
                        Json(json!({"status":"done"}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let root =
            std::env::temp_dir().join(format!("gary-caption-retry-{}", crate::headless::serial()));
        std::fs::create_dir(&root).unwrap();
        let audio = root.join("track.wav");
        let mut writer = hound::WavWriter::create(
            &audio,
            hound::WavSpec {
                channels: 1,
                sample_rate: 8000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.write_sample(0_i16).unwrap();
        writer.finalize().unwrap();
        let options = Options {
            dataset: root.clone(),
            lm: "lm".into(),
            dit: "dit".into(),
            vae: "vae".into(),
            overwrite: false,
            bpm_analysis: false,
            key_analysis: false,
            trigger: String::new(),
        };
        let endpoint = CaptionEndpoint {
            client: &reqwest::Client::new(),
            base: &base,
        };
        let result: Value = caption_checked(
            &endpoint,
            &options,
            &audio,
            &root,
            "Captioning track.wav",
            |result| quality_error(result["caption"].as_str().unwrap_or_default(), "caption"),
            || false,
        )
        .await
        .unwrap();
        assert_eq!(result["caption"], "Warm synth chords");
        assert_eq!(requests.load(Ordering::Relaxed), 3);

        std::fs::copy(&audio, root.join("a.wav")).unwrap();
        std::fs::write(root.join("a.txt"), "Existing label").unwrap();
        let options = Options {
            overwrite: true,
            ..options
        };
        let mut skipped = Vec::new();
        let outcome = caption_dataset(
            &endpoint,
            &options,
            &root,
            |_| {},
            || false,
            |error| {
                skipped.push(error);
                Ok(())
            },
        )
        .await
        .unwrap();
        assert_eq!(outcome, (1, 2));
        assert_eq!(skipped.len(), 1);
        assert!(skipped[0].contains("Token limit reached"));
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "Existing label"
        );
        assert!(std::fs::read_to_string(root.join("track.txt"))
            .unwrap()
            .contains("Warm synth chords"));
        task.abort();
        std::fs::remove_dir_all(root).unwrap();
    }
}
