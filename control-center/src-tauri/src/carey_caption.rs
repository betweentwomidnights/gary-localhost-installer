//! Native Carey dataset captioning through ace-server's /caption. The server
//! suggests a caption, tempo, key and language from each song and returns its
//! local BPM/key analysis beside them; what Carey writes to a sidecar follows
//! the Python job's policy, ported here: a filename BPM is ground truth, local
//! analysis corrects the LM where it is confident, LM lyrics are never written
//! (they are invented, not transcribed), and unusable captions are retried on
//! a two-minute excerpt before failing.

use crate::ace_client::{self, Body};
use crate::migration_verify::PrivateServer;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

/// /caption takes up to ten minutes of WAV.
const MAX_CAPTION_SECONDS: f64 = 600.0;
/// The Python job's fallback excerpt for a caption that fails quality checks.
const EXCERPT_SECONDS: f64 = 120.0;
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
    let pattern = PATTERN.get_or_init(|| Regex::new(r"(?i)(?:^|[_-])bpm[_-]?(\d{2,3})(?:[_-]|\.|$)").unwrap());
    let bpm: u32 = pattern.captures(name)?.get(1)?.as_str().parse().ok()?;
    (1..=400).contains(&bpm).then_some(bpm)
}

fn bpm_value(value: &serde_json::Value) -> Option<f64> {
    let bpm = value.as_f64().or_else(|| value.as_str()?.trim().parse().ok())?;
    (1.0..=400.0).contains(&bpm).then_some(bpm)
}

fn round_bpm(value: Option<f64>) -> Option<u32> {
    value.map(|bpm| bpm.round() as u32).filter(|bpm| (1..=400).contains(bpm))
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

pub fn choose_bpm(filename: Option<u32>, lm: Option<f64>, local: Option<(f64, f64)>) -> BpmDecision {
    let local_bpm = local.map(|(bpm, _)| bpm);
    let decision = |bpm, source| BpmDecision { bpm, source, lm, local: local_bpm, filename };
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
    if tempo_family(bpm).iter().any(|candidate| (lm_bpm - candidate).abs() <= BPM_TOLERANCE) {
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

const NOTES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

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
        return KeyDecision { keyscale: lm.clone(), source, lm, local: String::new() };
    };
    let local = local_key.to_string();
    if !lm.is_empty() && lm == normalize_keyscale(&local) {
        return KeyDecision { keyscale: lm.clone(), source: "lm_agrees_with_local", lm, local };
    }
    if confidence >= KEY_MIN_CONFIDENCE {
        let source = if lm.is_empty() { "local" } else { "local_overrode_lm" };
        return KeyDecision { keyscale: local.clone(), source, lm, local };
    }
    let source = if lm.is_empty() { "missing" } else { "lm_local_low_confidence" };
    KeyDecision { keyscale: lm.clone(), source, lm, local }
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
    let compact: String = text.chars().filter(|character| !character.is_whitespace()).collect();
    let distinct: std::collections::HashSet<char> = compact.chars().collect();
    if compact.chars().count() >= 24 && distinct.len() <= 3 {
        return Some(format!("{field} is dominated by repeated characters"));
    }
    // Twenty-one or more of one punctuation mark in a row.
    let mut run = (' ', 0usize);
    for character in compact.chars() {
        run = if character == run.0 { (character, run.1 + 1) } else { (character, 1) };
        if "!?._=-".contains(character) && run.1 >= 21 {
            return Some(format!("{field} is dominated by repeated punctuation"));
        }
    }
    if field == "caption" {
        let length = text.trim().chars().count();
        let letters = text.chars().filter(|character| character.is_alphabetic()).count();
        if length >= 80 && (letters as f64) / (text.chars().count().max(1) as f64) < 0.15 {
            return Some(format!("{field} has too little word content"));
        }
        let words = WORDS.get_or_init(|| Regex::new(r"[A-Za-z][A-Za-z'-]{2,}").unwrap());
        if length >= 40 && words.find_iter(text).count() < 3 {
            return Some(format!("{field} has too little descriptive text"));
        }
        let fields = FIELDS.get_or_init(|| {
            Regex::new(r"(?i)(?:^|[^a-z])(?:bpm|duration|genres?|keyscale|language|timesignature)\s*:").unwrap()
        });
        if fields.is_match(text) {
            return Some(format!("{field} appears to contain embedded metadata fields"));
        }
    }
    None
}

fn lyrics_are_structural_only(lyrics: &str) -> bool {
    lyrics
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .all(|line| line.len() > 2 && line.starts_with('[') && line.ends_with(']') && !line[1..line.len() - 1].contains(']'))
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
    lyrics.is_empty() || lyrics.to_lowercase().contains("[instrumental]") || lyrics_are_structural_only(lyrics)
}

/// The tempo and key for `audio` from one /caption result: a filename BPM
/// first, then the LM corrected by local analysis where it is confident.
fn tempo_and_key(audio: &Path, result: &serde_json::Value, options: &Options) -> (BpmDecision, KeyDecision) {
    let local = &result["local_analysis"];
    let local_bpm = options
        .bpm_analysis
        .then(|| Some((bpm_value(&local["bpm"])?, local["bpm_confidence"].as_f64()?)))
        .flatten();
    let local_key = options
        .key_analysis
        .then(|| {
            let key = local["keyscale"].as_str().filter(|key| !key.trim().is_empty())?;
            Some((key, local["key_confidence"].as_f64()?))
        })
        .flatten();
    let name = audio.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let bpm = choose_bpm(filename_bpm(&name), bpm_value(&result["bpm"]), local_bpm);
    let key = choose_key(result["keyscale"].as_str().unwrap_or_default(), local_key);
    (bpm, key)
}

/// The Carey sidecar fields for `audio` from one /caption result.
pub fn sidecar_fields(
    audio: &Path,
    result: &serde_json::Value,
    options: &Options,
) -> crate::CareyAceSidecarFields {
    let (bpm, key) = tempo_and_key(audio, result, options);
    let instrumental = is_instrumental(result);
    let text = |value: Option<f64>| value.map(|number| number.to_string()).unwrap_or_default();
    crate::CareyAceSidecarFields {
        caption: result["caption"].as_str().unwrap_or_default().trim().to_string(),
        genre: result["genre"].as_str().unwrap_or_default().trim().to_string(),
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
        timesignature: result["time_signature"].as_str().unwrap_or_default().trim().to_string(),
        language: if instrumental {
            "unknown".into()
        } else {
            result["language"].as_str().unwrap_or_default().trim().to_string()
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

/// The downloaded build of the `size` LM ("0.6B", "1.7B", "4B") to caption
/// with, best precision first.
pub fn lm_file(root: &Path, size: &str) -> Result<String, String> {
    let models = crate::carey_models::models();
    crate::carey_models::TIERS
        .iter()
        .filter_map(|tier| models.component(&format!("carey-native::lm-{size}::{tier}")))
        .find(|component| models.missing(root, std::slice::from_ref(&component.id)).is_none())
        .map(|component| component.files[0].filename.clone())
        .ok_or_else(|| format!("Download the {size} language model from Carey's models to caption with it."))
}

/// The shared VAE every Carey tier downloads.
pub fn vae_file() -> Result<String, String> {
    crate::carey_models::models()
        .component(crate::carey_models::SHARED)
        .map(|shared| shared.files[1].filename.clone())
        .ok_or_else(|| "Carey's catalog has no shared models".into())
}

/// A downloaded DiT whose audio tokenizer feeds the LM, for callers that do
/// not choose one: the 2B models before XL, best precision first. Every
/// Carey tier downloads 2B base, so one is there once Carey is migrated.
pub fn caption_dit(root: &Path) -> Result<String, String> {
    use crate::carey_models::{DEFAULT_DIT, SIBLING_DITS, TIERS, XL_DITS};
    let models = crate::carey_models::models();
    std::iter::once(DEFAULT_DIT)
        .chain(SIBLING_DITS.iter().copied())
        .chain(XL_DITS.iter().copied())
        .flat_map(|variant| TIERS.iter().map(move |tier| crate::carey_models::dit_component(variant, tier)))
        .filter_map(|id| models.component(&id))
        .find(|component| models.missing(root, std::slice::from_ref(&component.id)).is_none())
        .map(|component| component.files[0].filename.clone())
        .ok_or_else(|| "Download one of Carey's models to caption with.".into())
}

/// `audio` as WAV bytes /caption accepts: the file itself when it is a WAV of
/// at most ten minutes, otherwise FFmpeg's decode, cut to a centered
/// `excerpt` when one is asked for or the song is too long.
async fn caption_audio(audio: &Path, scratch: &Path, excerpt: Option<f64>) -> Result<Vec<u8>, String> {
    let is_wav = audio.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("wav"));
    let duration = if is_wav {
        hound::WavReader::open(audio)
            .ok()
            .map(|reader| reader.duration() as f64 / reader.spec().sample_rate as f64)
    } else {
        None
    };
    let window = excerpt.or_else(|| duration.filter(|seconds| *seconds > MAX_CAPTION_SECONDS).map(|_| EXCERPT_SECONDS));
    if is_wav && duration.is_some() && window.is_none() {
        return std::fs::read(audio).map_err(|error| error.to_string());
    }
    let decoded = scratch.join("caption.wav");
    let _ = std::fs::remove_file(&decoded);
    let mut command = tokio::process::Command::new("ffmpeg");
    crate::hide_console_window(&mut command);
    command.args(["-nostdin", "-v", "error", "-y"]);
    if let Some(window) = window {
        // Centered when the length is known; FFmpeg reads a shorter file to its end.
        let start = duration.map_or(0.0, |seconds| ((seconds - window) / 2.0).max(0.0));
        command.arg("-ss").arg(start.to_string()).arg("-t").arg(window.to_string());
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
            String::from_utf8_lossy(&output.stderr).chars().take(500).collect::<String>()
        ));
    }
    let bytes = std::fs::read(&decoded).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_file(&decoded);
    // A decoded song past the limit is cut to the excerpt instead.
    let seconds = hound::WavReader::new(std::io::Cursor::new(&bytes))
        .map(|reader| reader.duration() as f64 / reader.spec().sample_rate as f64)
        .unwrap_or(0.0);
    if window.is_none() && seconds > MAX_CAPTION_SECONDS {
        return Box::pin(caption_audio(audio, scratch, Some(EXCERPT_SECONDS))).await;
    }
    Ok(bytes)
}

async fn caption_once(server: &PrivateServer, options: &Options, audio: &[u8], what: &str) -> Result<serde_json::Value, String> {
    let request = serde_json::json!({"lm_model": options.lm, "synth_model": options.dit, "vae": options.vae});
    let (body, _) = ace_client::run(
        &server.client,
        &server.base,
        "/caption",
        Body::WithAudio(&request, audio),
        Duration::from_secs(900),
        what,
    )
    .await?;
    serde_json::from_slice(&body).map_err(|error| format!("{what} returned unreadable JSON: {error}"))
}

/// /caption's result for `audio`, asked again on a centered two-minute
/// excerpt when `check` finds the whole song's unusable. The inner `Err` is
/// why the excerpt's is unusable too.
async fn caption_checked(
    server: &PrivateServer,
    options: &Options,
    audio: &Path,
    scratch: &Path,
    what: &str,
    check: impl Fn(&serde_json::Value) -> Option<String>,
) -> Result<Result<serde_json::Value, String>, String> {
    let result = caption_once(server, options, &caption_audio(audio, scratch, None).await?, what).await?;
    if check(&result).is_none() {
        return Ok(Ok(result));
    }
    let result = caption_once(server, options, &caption_audio(audio, scratch, Some(EXCERPT_SECONDS)).await?, what).await?;
    Ok(check(&result).map_or(Ok(result), Err))
}

/// What a caption run reports as it goes.
pub struct Progress<'a> {
    pub index: usize,
    pub total: usize,
    pub name: &'a str,
}

/// Caption every song in `options.dataset` that has no sidecar (or every
/// song, with `overwrite`), writing Carey sidecars beside them. `cancelled`
/// is checked between songs. Returns how many were captioned and how many
/// songs the dataset holds.
pub async fn caption_dataset(
    server: &PrivateServer,
    options: &Options,
    scratch: &Path,
    mut progress: impl FnMut(Progress<'_>),
    cancelled: impl Fn() -> bool,
) -> Result<(usize, usize), String> {
    let entries = crate::read_carey_ace_dataset_sidecars(&options.dataset.to_string_lossy())?;
    let pending: Vec<_> = entries.iter().filter(|entry| options.overwrite || !entry.exists).collect();
    for (index, entry) in pending.iter().enumerate() {
        if cancelled() {
            return Err("Captioning cancelled.".into());
        }
        progress(Progress { index: index + 1, total: pending.len(), name: &entry.relative_path });
        let audio = PathBuf::from(&entry.audio_path);
        let what = format!("Captioning {}", entry.relative_path);
        let check = |result: &serde_json::Value| {
            quality_error(result["caption"].as_str().unwrap_or_default(), "caption").or_else(|| {
                let genre = result["genre"].as_str().unwrap_or_default();
                (!genre.trim().is_empty()).then(|| quality_error(genre, "genre")).flatten()
            })
        };
        let result = caption_checked(server, options, &audio, scratch, &what, check).await?.map_err(|reason| {
            format!(
                "The LM wrote an unusable caption for {}: {reason}. Try another LM size, or write that sidecar in the dataset editor.",
                entry.relative_path
            )
        })?;
        let fields = sidecar_fields(&audio, &result, options);
        std::fs::write(&entry.sidecar_path, crate::render_carey_ace_sidecar(&fields))
            .map_err(|error| format!("Cannot write {}: {error}", entry.sidecar_path))?;
    }
    Ok((pending.len(), entries.len()))
}

// ---------------------------------------------------------------------------
// SA3 auto-label: the same captioner, keeping only the genre, tempo and key
// for SA3's one-line sidecars. Ported from services/carey/sa3_autolabel.py.

/// The LM's genre for an SA3 sidecar, or why it is unusable.
pub fn sa3_genre(result: &serde_json::Value) -> Result<String, String> {
    let genre = result["genre"].as_str().unwrap_or_default().trim();
    if genre.is_empty() || ["n/a", "na", "none", "null", "unknown"].contains(&genre.to_lowercase().as_str()) {
        return Err("no genre".into());
    }
    quality_error(genre, "genre").map_or_else(|| Ok(genre.to_string()), Err)
}

/// An SA3 sidecar: "genre, 145 bpm, C minor", or with `labeled` the style of
/// SA3's own training data.
pub fn sa3_sidecar(labeled: bool, genre: &str, bpm: Option<u32>, keyscale: &str) -> String {
    if labeled {
        let mut parts = vec!["TrackType: Music".to_string(), "VocalType: Instrumental".to_string()];
        parts.extend((!genre.is_empty()).then(|| format!("Genre: {genre}")));
        parts.extend(bpm.map(|bpm| format!("BPM: {bpm}")));
        parts.extend((!keyscale.is_empty()).then(|| format!("Key: {keyscale}")));
        return parts.join(", ");
    }
    let bpm = bpm.map(|bpm| format!("{bpm} bpm")).unwrap_or_default();
    [genre, &bpm, keyscale].into_iter().filter(|part| !part.is_empty()).collect::<Vec<_>>().join(", ")
}

/// Label every song in `files`, overwriting the `.txt` sidecar beside each.
/// `cancelled` is checked between songs.
pub async fn label_sa3_dataset(
    server: &PrivateServer,
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
        let name = audio.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        progress(Progress { index: index + 1, total: files.len(), name: &name });
        let what = format!("Labeling {name}");
        let result = caption_checked(server, options, audio, scratch, &what, |result| sa3_genre(result).err())
            .await?
            .map_err(|reason| {
                format!("The LM gave {name} an unusable genre ({reason}). Try another LM size, or write that sidecar by hand.")
            })?;
        let genre = sa3_genre(&result)?;
        let (bpm, key) = tempo_and_key(audio, &result, options);
        let sidecar = audio.with_extension("txt");
        std::fs::write(&sidecar, sa3_sidecar(labeled, &genre, bpm.bpm, &key.keyscale) + "\n")
            .map_err(|error| format!("Cannot write {}: {error}", sidecar.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bpm_follows_the_python_precedence() {
        assert_eq!(filename_bpm("break_bpm92_take2.wav"), Some(92));
        assert_eq!(filename_bpm("bpm-174.flac"), Some(174));
        assert_eq!(filename_bpm("subpm120.wav"), None);
        assert_eq!(choose_bpm(Some(92), Some(140.0), Some((90.0, 3.0))).source, "filename");
        assert_eq!(choose_bpm(None, Some(120.0), None).bpm, Some(120));
        assert_eq!(choose_bpm(None, None, None).source, "missing");
        assert_eq!(choose_bpm(None, None, Some((97.6, 2.0))).bpm, Some(98));
        // Half-time agrees: the LM's value stands.
        let half = choose_bpm(None, Some(87.0), Some((174.0, 2.0)));
        assert_eq!((half.bpm, half.source), (Some(87), "lm_agrees_with_local"));
        let low = choose_bpm(None, Some(120.0), Some((97.0, 1.1)));
        assert_eq!((low.bpm, low.source), (Some(120), "lm_local_low_confidence"));
        let overrode = choose_bpm(None, Some(120.0), Some((97.0, 1.5)));
        assert_eq!((overrode.bpm, overrode.source), (Some(97), "local_overrode_lm"));
    }

    #[test]
    fn keys_normalize_and_follow_the_python_precedence() {
        assert_eq!(normalize_keyscale("bb maj"), "A# major");
        assert_eq!(normalize_keyscale("F♯_minor"), "F# minor");
        assert_eq!(normalize_keyscale("N/A"), "");
        assert_eq!(normalize_keyscale("C dorian"), "");
        assert_eq!(choose_key("F minor", Some(("F minor", 0.01))).source, "lm_agrees_with_local");
        let overrode = choose_key("C major", Some(("A minor", 0.2)));
        assert_eq!((overrode.keyscale.as_str(), overrode.source), ("A minor", "local_overrode_lm"));
        let low = choose_key("C major", Some(("A minor", 0.1)));
        assert_eq!((low.keyscale.as_str(), low.source), ("C major", "lm_local_low_confidence"));
        assert_eq!(choose_key("", None).source, "missing");
    }

    #[test]
    fn unusable_captions_are_caught() {
        assert!(quality_error("A warm lo-fi beat with dusty drums and a mellow Rhodes.", "caption").is_none());
        assert!(quality_error("  ", "caption").unwrap().contains("empty"));
        assert!(quality_error("<|audio_code_12|><|audio_code_4|>", "caption").is_some());
        assert!(quality_error(&"a".repeat(30), "caption").unwrap().contains("repeated characters"));
        assert!(quality_error(&format!("great track{}", "!".repeat(25)), "caption").unwrap().contains("punctuation"));
        assert!(quality_error("A song. bpm: 120, keyscale: C major", "caption").unwrap().contains("embedded"));
        assert!(quality_error("12 34 56 78 90 12 34 56 78 90 12 34 56 78 90 12 34 56 78 90", "caption").is_some());
    }

    #[test]
    fn a_caption_result_becomes_a_carey_sidecar() {
        let options = Options {
            dataset: PathBuf::new(),
            lm: String::new(),
            dit: String::new(),
            vae: String::new(),
            overwrite: false,
            bpm_analysis: true,
            key_analysis: true,
            trigger: " my-tag ".into(),
        };
        let result = json!({
            "caption": " Dusty boom bap with a mellow Rhodes. ", "genre": "", "is_instrumental": null,
            "lyrics": "[verse]\n[chorus]", "bpm": 120, "keyscale": "C major", "language": "en", "time_signature": "3",
            "local_analysis": {"bpm": 92, "bpm_confidence": 1.8, "keyscale": "F minor", "key_confidence": 0.3},
        });
        let fields = sidecar_fields(Path::new("song.wav"), &result, &options);
        assert_eq!(fields.caption, "Dusty boom bap with a mellow Rhodes.");
        assert_eq!((fields.bpm.as_str(), fields.bpm_source.as_str()), ("92", "local_overrode_lm"));
        assert_eq!((fields.keyscale.as_str(), fields.key_source.as_str()), ("F minor", "local_overrode_lm"));
        assert!(fields.is_instrumental, "section markers alone are an instrumental");
        assert_eq!(fields.language, "unknown");
        assert_eq!(fields.custom_tag, "my-tag");
        assert_eq!(fields.timesignature, "3");
        assert!(fields.lyrics.is_empty());
        let staged = crate::carey_training::metadata(&fields, "").unwrap();
        assert!(staged.contains("bpm: 92\nkey: F minor\nsignature: 3\n") && staged.ends_with("lyrics: [Instrumental]\n"), "{staged}");
        let vocal = json!({"caption": "A sad piano ballad with soft vocals.", "lyrics": "I walked alone", "bpm": 70, "language": "en"});
        let without_analysis = Options { bpm_analysis: false, key_analysis: false, ..options };
        let fields = sidecar_fields(Path::new("ballad_bpm68.wav"), &vocal, &without_analysis);
        assert_eq!((fields.bpm.as_str(), fields.bpm_source.as_str()), ("68", "filename"));
        assert!(!fields.is_instrumental && fields.lyrics.is_empty(), "vocal lyrics are the user's to write");
    }

    #[test]
    fn a_caption_result_becomes_an_sa3_sidecar_as_the_python_helper_wrote_it() {
        assert_eq!(sa3_genre(&json!({"genre": " lo-fi hip hop "})).unwrap(), "lo-fi hip hop");
        for unusable in [json!({}), json!({"genre": "N/A"}), json!({"genre": "unknown"}), json!({"genre": "<|audio_code_12|>"})] {
            assert!(sa3_genre(&unusable).is_err(), "{unusable}");
        }
        assert_eq!(sa3_sidecar(false, "techno", Some(145), "C minor"), "techno, 145 bpm, C minor");
        assert_eq!(sa3_sidecar(false, "techno", None, ""), "techno");
        assert_eq!(
            sa3_sidecar(true, "techno", Some(145), "C minor"),
            "TrackType: Music, VocalType: Instrumental, Genre: techno, BPM: 145, Key: C minor"
        );
        assert_eq!(sa3_sidecar(true, "techno", None, ""), "TrackType: Music, VocalType: Instrumental, Genre: techno");
    }

    /// GARY4LOCAL_CAREY_CAPTION_SMOKE_ROOT (a new folder), _MODELS (a folder
    /// with the 2B base, the shared GGUFs and an LM: _LM names it, the 0.6B
    /// Q8_0 by default) and _SERVER (an ace-server.exe; _BACKEND, cuda by
    /// default, names its build). Renders two songs, captions them as Carey
    /// would, stages the sidecars it wrote for training, then auto-labels the
    /// same songs as SA3 would.
    #[tokio::test]
    #[ignore = "captions real renders with a real ace-server and LM"]
    async fn real_native_captioning_writes_trainable_sidecars() {
        let var = |name: &str| std::env::var(format!("GARY4LOCAL_CAREY_CAPTION_SMOKE_{name}")).ok();
        let root = PathBuf::from(var("ROOT").expect("a new isolated root is required"));
        assert!(!root.exists(), "the smoke root must not exist yet");
        let models = PathBuf::from(var("MODELS").expect("a model folder is required"));
        let server_exe = PathBuf::from(var("SERVER").expect("an ace-server is required"));
        std::fs::create_dir_all(root.join("dataset/nested")).unwrap();
        let installed = crate::native_runtime::NativeInstall {
            dir: server_exe.parent().unwrap().into(),
            backend: var("BACKEND").unwrap_or_else(|| "cuda".into()),
            runtimes: Vec::new(),
            version: None,
            fallback_reason: None,
        };
        let launch = crate::native_runtime::NativeLaunch::new(
            &root,
            "ace-server.exe",
            &installed,
            Vec::new(),
            Vec::new(),
            &[("ACE_MODELS_DIR".into(), models.to_string_lossy().into())],
        );
        let server = PrivateServer::start(&launch, &root, "Carey captioner", &[]).await.unwrap();
        let dit = "acestep-v15-base-BF16.gguf";
        for (path, caption) in [
            (root.join("dataset/groove_bpm96.wav"), "funky drum groove with slap bass and clavinet"),
            (root.join("dataset/nested/pad.wav"), "slow ambient synth pad with soft piano"),
        ] {
            let request = json!({"caption": caption, "lyrics": "[Instrumental]", "duration": 20, "bpm": 96,
                "seed": 7, "inference_steps": 30, "synth_model": dit, "output_format": "wav16"});
            let audio = ace_client::synth_audio(&server.client, &server.base, Body::Json(&request),
                Duration::from_secs(600), "render").await.unwrap();
            std::fs::write(path, audio).unwrap();
        }
        let options = Options {
            dataset: root.join("dataset"),
            lm: var("LM").unwrap_or_else(|| "acestep-5Hz-lm-0.6B-Q8_0.gguf".into()),
            dit: dit.into(),
            vae: "vae-BF16.gguf".into(),
            overwrite: false,
            bpm_analysis: true,
            key_analysis: true,
            trigger: "smoke".into(),
        };
        let (captioned, total) = caption_dataset(&server, &options, &root, |progress| {
            println!("[{}/{}] {}", progress.index, progress.total, progress.name)
        }, || false)
        .await
        .unwrap();
        assert_eq!((captioned, total), (2, 2));
        for sidecar in ["dataset/groove_bpm96.txt", "dataset/nested/pad.txt"] {
            let raw = std::fs::read_to_string(root.join(sidecar)).unwrap();
            println!("--- {sidecar}\n{raw}");
            let fields = crate::parse_carey_ace_sidecar_content(&raw);
            assert!(quality_error(&fields.caption, "caption").is_none());
            assert_eq!(fields.custom_tag, "smoke");
            if fields.is_instrumental {
                crate::carey_training::metadata(&fields, "").unwrap();
            }
        }
        let groove = crate::parse_carey_ace_sidecar_content(&std::fs::read_to_string(root.join("dataset/groove_bpm96.txt")).unwrap());
        assert_eq!((groove.bpm.as_str(), groove.bpm_source.as_str()), ("96", "filename"));

        // SA3's auto-label overwrites the same sidecars with one line each.
        let files = [root.join("dataset/groove_bpm96.wav"), root.join("dataset/nested/pad.wav")];
        label_sa3_dataset(&server, &Options { overwrite: true, trigger: String::new(), ..options }, &files, false, &root, |_| {}, || false)
            .await
            .unwrap();
        server.stop().await.unwrap();
        for sidecar in ["dataset/groove_bpm96.txt", "dataset/nested/pad.txt"] {
            let raw = std::fs::read_to_string(root.join(sidecar)).unwrap();
            println!("--- SA3 {sidecar}: {raw}");
            assert_eq!(raw.lines().count(), 1, "{raw}");
        }
        let groove = std::fs::read_to_string(root.join("dataset/groove_bpm96.txt")).unwrap();
        assert!(groove.contains(", 96 bpm"), "the filename BPM wins: {groove}");
    }
}
