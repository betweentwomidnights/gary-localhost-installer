//! Gary's legacy HTTP contract translated to sa3.cpp's unified generation API.
//! Audio algorithms stay in the native pipeline. This module owns upload files,
//! client defaults, exact requested lengths and client-facing job metadata.

use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Map, Value};
use std::io::Cursor;

const SAMPLE_RATE: f64 = 44100.0;
const MAX_DURATION: f64 = 380.0;

pub const REQUIRED_CAPABILITIES: &[&str] = &[
    "fixed_prefix",
    "request_splice",
    "conditioning_duration",
    "model_lifecycle",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Generate,
    Loop,
    Transform,
    Continue,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Generate => "generate",
            Self::Loop => "loop",
            Self::Transform => "transform",
            Self::Continue => "continue",
        }
    }
}

pub struct InputAudio {
    pub bytes: Vec<u8>,
    pub seconds: f64,
    pub sample_rate: u32,
    pub channels: u16,
}

pub struct Translation {
    pub native: Value,
    pub metadata: Value,
    pub submitted: Value,
    pub audio: Option<InputAudio>,
    pub mode: Mode,
}

fn number(data: &Map<String, Value>, key: &str, default: f64) -> Result<f64, String> {
    match data.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_f64()
            .or_else(|| value.as_str()?.parse().ok())
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("{key} must be a finite number")),
    }
}

fn text<'a>(data: &'a Map<String, Value>, key: &str, default: &'a str) -> Result<&'a str, String> {
    match data.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::String(value)) => Ok(value),
        _ => Err(format!("{key} must be text")),
    }
}

fn boolean(data: &Map<String, Value>, key: &str, default: bool) -> Result<bool, String> {
    match data.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::Number(value)) if value.as_f64() == Some(0.0) => Ok(false),
        Some(Value::Number(value)) if value.as_f64() == Some(1.0) => Ok(true),
        Some(Value::String(value)) => match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" | "" => Ok(false),
            _ => Err(format!("{key} must be a boolean")),
        },
        _ => Err(format!("{key} must be a boolean")),
    }
}

fn optional_number(data: &Map<String, Value>, key: &str, default: Value) -> Result<Value, String> {
    match data.get(key) {
        None => Ok(default),
        Some(Value::Null | Value::Bool(false)) => Ok(Value::Null),
        Some(Value::String(value))
            if ["", "off", "none", "disable", "disabled"]
                .contains(&value.to_ascii_lowercase().as_str()) =>
        {
            Ok(Value::Null)
        }
        _ => Ok(json!(number(data, key, 0.0)?)),
    }
}

fn audio(data: &Map<String, Value>) -> Result<InputAudio, String> {
    let encoded = text(data, "audio_data", "")?;
    let encoded = if encoded.starts_with("data:") {
        encoded.split_once(',').ok_or("invalid audio data URL")?.1
    } else {
        encoded
    };
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|error| format!("could not decode audio_data: {error}"))?;
    let mut reader = hound::WavReader::new(Cursor::new(&bytes))
        .map_err(|error| format!("audio_data must contain a WAV: {error}"))?;
    let spec = reader.spec();
    if !(1..=2).contains(&spec.channels) || spec.sample_rate == 0 {
        return Err("audio_data must be mono or stereo with a positive sample rate".into());
    }
    // Validate the complete sample payload before writing it or queuing a job.
    // The native reader handles PCM 16/24/32 and float32 without host resampling.
    let count = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16 | 24 | 32) => reader
            .samples::<i32>()
            .try_fold(0u64, |n, sample| sample.map(|_| n + 1)),
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().try_fold(0u64, |n, sample| {
            sample.and_then(|value| {
                if value.is_finite() {
                    Ok(n + 1)
                } else {
                    Err(hound::Error::FormatError("non-finite WAV sample"))
                }
            })
        }),
        _ => return Err("audio_data must use PCM 16/24/32 or float32 WAV samples".into()),
    }
    .map_err(|error| format!("invalid WAV sample payload: {error}"))?;
    let seconds = count as f64 / spec.channels as f64 / spec.sample_rate as f64;
    if seconds <= 0.0 || seconds > MAX_DURATION {
        return Err(format!(
            "input duration must be in (0, {MAX_DURATION}] seconds"
        ));
    }
    Ok(InputAudio {
        bytes,
        seconds,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

fn tail_pad(data: &Map<String, Value>) -> Result<f64, String> {
    for key in ["tail_pad_seconds", "tail_pad", "continuation_tail_pad"] {
        if data
            .get(key)
            .is_some_and(|value| !value.is_null() && value != "")
        {
            return Ok(number(data, key, 6.0)?.clamp(0.0, 60.0));
        }
    }
    Ok(6.0)
}

/// Translate only known client fields. Native filesystem paths, canvas/frame
/// counts and private controls are chosen by the host, never forwarded blindly.
/// A list of `{name, strength}` LoRAs by catalog name. Published adapters only
/// support full-range application, so intervals and layer filters are refused
/// rather than silently dropped.
fn catalog_loras(value: &Value, field: &str) -> Result<Vec<Value>, String> {
    let mut loras = Vec::new();
    for entry in value.as_array().ok_or(format!("{field} must be a list"))? {
        let entry = entry
            .as_object()
            .ok_or(format!("each {field} entry must be an object"))?;
        let name = text(entry, "name", "")?.trim();
        if name.is_empty() || name.contains(['/', '\\', ':']) || name == "." || name == ".." {
            return Err("each LoRA needs a catalog name".into());
        }
        if number(entry, "interval_min", 0.0)? != 0.0
            || number(entry, "interval_max", 1.0)? != 1.0
            || !text(entry, "layer_filter", "")?.is_empty()
        {
            return Err(format!(
                "native LoRA '{name}' does not yet support intervals or layer filters"
            ));
        }
        loras.push(json!({"name":name, "strength":number(entry, "strength", 1.0)?}));
    }
    Ok(loras)
}

const MAX_LANES: usize = 8;
const MAX_ENVELOPE_POINTS: usize = 256;

/// `[[seconds, amount], ...]`, also accepting sa3.cpp's `{t, amount}` points.
fn envelope(value: &Value, field: &str) -> Result<Value, String> {
    let points = value
        .as_array()
        .ok_or(format!("{field} must be a list of [seconds, amount]"))?;
    if points.len() > MAX_ENVELOPE_POINTS {
        return Err(format!("{field}: at most {MAX_ENVELOPE_POINTS} points"));
    }
    let mut out = Vec::with_capacity(points.len());
    for point in points {
        let (t, amount) = match point {
            Value::Array(pair) if pair.len() == 2 => (pair[0].as_f64(), pair[1].as_f64()),
            Value::Object(map) => (
                map.get("t").and_then(Value::as_f64),
                map.get("amount").and_then(Value::as_f64),
            ),
            _ => (None, None),
        };
        match (t, amount) {
            (Some(t), Some(amount)) if t.is_finite() && amount.is_finite() => {
                out.push(json!([t, amount]))
            }
            _ => return Err(format!("{field}: each point is [seconds, amount]")),
        }
    }
    Ok(json!(out))
}

/// LoRA automation, documented in sa3.cpp's docs/SERVER.md. Envelope times are
/// seconds on the output timeline; for a continuation, 0 is the start of the
/// source, which the adapter sends whole. Lane LoRAs use catalog names like
/// `loras`, and submit() maps them to paths.
fn automation(data: &Map<String, Value>, native: &mut Value) -> Result<(), String> {
    if let Some(value) = data.get("lora_lanes").filter(|value| !value.is_null()) {
        let entries = value.as_array().ok_or("lora_lanes must be a list")?;
        if entries.len() > MAX_LANES {
            return Err(format!("lora_lanes: at most {MAX_LANES} lanes"));
        }
        let mut lanes = Vec::with_capacity(entries.len());
        for entry in entries {
            let entry = entry
                .as_object()
                .ok_or("each lora_lanes entry must be an object")?;
            let loras = match entry.get("loras").filter(|value| !value.is_null()) {
                Some(value) => catalog_loras(value, "lora_lanes loras")?,
                None => Vec::new(),
            };
            let points = envelope(
                entry.get("envelope").unwrap_or(&Value::Null),
                "lora_lanes envelope",
            )?;
            if points.as_array().unwrap().is_empty() {
                return Err("lora_lanes: each lane needs an envelope".into());
            }
            let mut lane = json!({"loras":loras, "envelope":points});
            let prompt = text(entry, "prompt", "")?.trim();
            if !prompt.is_empty() {
                lane["prompt"] = json!(prompt);
            }
            lanes.push(lane);
        }
        if !lanes.is_empty() {
            native["lora_lanes"] = json!(lanes);
        }
    }
    if let Some(value) = data.get("lora_envelope").filter(|value| !value.is_null()) {
        native["lora_envelope"] = envelope(value, "lora_envelope")?;
    }
    if data
        .get("lora_envelope_mode")
        .is_some_and(|value| !value.is_null())
    {
        let mode = text(data, "lora_envelope_mode", "")?.to_ascii_lowercase();
        if !["velocity", "layer"].contains(&mode.as_str()) {
            return Err("lora_envelope_mode must be velocity or layer".into());
        }
        native["lora_envelope_mode"] = json!(mode);
    }
    for key in [
        "lane_isolation",
        "lane_isolation_until",
        "lane_average_until",
    ] {
        if data.get(key).is_some_and(|value| !value.is_null()) {
            let value = number(data, key, 0.0)?;
            if value < 0.0 {
                return Err(format!("{key} must not be negative"));
            }
            native[key] = json!(value);
        }
    }
    Ok(())
}

fn uses_automation(native: &Value) -> bool {
    native.get("lora_lanes").is_some() || native.get("lora_envelope").is_some()
}

pub fn translate(mode: Mode, body: Value) -> Result<Translation, String> {
    let data = body.as_object().ok_or("JSON object required")?;
    let prompt = text(data, "prompt", "")?.trim();
    if prompt.is_empty() {
        return Err("prompt is required".into());
    }
    let negative = text(data, "negative_prompt", "low quality")?.trim();
    let steps = number(data, "steps", 8.0)?;
    if steps.fract() != 0.0 || !(1.0..=200.0).contains(&steps) {
        return Err("steps must be an integer in [1, 200]".into());
    }
    let cfg = number(data, "cfg_scale", 1.0)?;
    if !(0.0..=25.0).contains(&cfg) {
        return Err("cfg_scale must be in [0, 25]".into());
    }
    let shift = text(data, "shift", "default")?.to_ascii_lowercase();
    let native_shift = match shift.as_str() {
        "" | "default" | "logsnr" => "LogSNR",
        "none" => "None",
        "flux" => "Flux",
        "full" => "Full",
        _ => return Err("shift must be default, none, logsnr, flux or full".into()),
    };
    let sampler = text(data, "sampler_type", "pingpong")?;
    if sampler != "pingpong" {
        return Err(format!(
            "native SA3 does not yet support sampler_type '{sampler}'"
        ));
    }
    // Resolve negative/omitted seeds upstream and use the returned seed in both
    // submission and polling metadata; never manufacture a second random seed.
    let seed = match data.get("seed") {
        None | Some(Value::Null) => json!(-1),
        Some(value) if value.as_i64().is_some() || value.as_u64().is_some() => value.clone(),
        Some(Value::String(value)) => json!(value
            .parse::<i64>()
            .map_err(|_| "seed must be an integer")?),
        _ => return Err("seed must be an integer".into()),
    };
    let mut native = json!({"prompt":prompt, "negative_prompt":negative, "steps":steps as u32,
        "cfg_scale":cfg, "dist_shift":native_shift, "seed":seed, "keep_models":false,
        "duration_padding_sec":6});
    for (key, default) in [
        ("latent_rescale", json!(1)),
        ("latent_shift", json!(0)),
        ("latent_target_std", Value::Null),
        ("latent_adapt_min", json!(0.9)),
        ("latent_adapt_max", json!(1)),
        ("peak_normalize_db", json!(2)),
        ("limiter_ceiling_db", json!(-0.3)),
        ("limiter_knee", json!(0.8)),
    ] {
        let value = optional_number(data, key, default.clone())?;
        native[key] = if value.is_null()
            && ![
                "latent_target_std",
                "peak_normalize_db",
                "limiter_ceiling_db",
            ]
            .contains(&key)
        {
            default
        } else {
            value
        };
    }
    let mut loras = Vec::new();
    if let Some(value) = data.get("loras").filter(|value| !value.is_null()) {
        loras = catalog_loras(value, "loras")?;
    } else if let Some(name) = data
        .get("lora")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty() && !name.trim().eq_ignore_ascii_case("none"))
    {
        if name.contains(['/', '\\', ':']) {
            return Err("LoRA must be a catalog name".into());
        }
        loras.push(json!({"name":name.trim().to_ascii_lowercase(), "strength":number(data, "lora_strength", 1.0)?}));
    }
    native["loras"] = json!(loras);
    automation(data, &mut native)?;
    let input = if matches!(mode, Mode::Transform | Mode::Continue) {
        Some(audio(data)?)
    } else {
        None
    };
    let mut details;
    let canvas;
    let target;
    match mode {
        Mode::Generate => {
            let duration = number(data, "duration", 30.0)?;
            if duration <= 0.0 || duration > MAX_DURATION {
                return Err(format!("duration must be in (0, {MAX_DURATION}] seconds"));
            }
            let pad = tail_pad(data)?;
            canvas = duration + pad;
            target = (duration * SAMPLE_RATE).round_ties_even() as u64;
            details = json!({"duration":duration,"tail_pad":pad,"gen_duration":canvas,"target_samples":target});
        }
        Mode::Loop => {
            let bpm = if data.contains_key("bpm") {
                number(data, "bpm", 0.0)?
            } else {
                extract_bpm(prompt).unwrap_or(0.0)
            };
            if bpm <= 0.0 {
                return Err("BPM required in prompt or bpm field".into());
            }
            let bars = number(data, "bars", 8.0)?;
            if ![4.0, 8.0, 16.0, 32.0].contains(&bars) {
                return Err("bars must be 4, 8, 16 or 32".into());
            }
            let duration = 240.0 / bpm * bars;
            canvas = duration + 2.0;
            if canvas > MAX_DURATION {
                return Err("loop exceeds the maximum duration with padding".into());
            }
            target = (duration * SAMPLE_RATE).round_ties_even() as u64;
            details = json!({"bpm":bpm,"bars":bars as u32,"seconds_per_bar":240.0/bpm,
                "loop_duration":duration,"gen_duration":canvas,"target_samples":target});
        }
        Mode::Transform => {
            let source = input.as_ref().unwrap();
            let strength = number(data, "strength", 0.9)?.clamp(0.01, 1.0);
            canvas = source.seconds + 0.5;
            target = (source.seconds * SAMPLE_RATE).round_ties_even() as u64;
            native["init_noise_level"] = json!(strength);
            // The pipeline already adds the reference's hidden 6 s canvas
            // for transform; this field applies only to text generation.
            details = json!({"strength":strength,"input_duration":source.seconds,
                "input_sr":source.sample_rate,"input_channels":source.channels,"target_samples":target});
        }
        Mode::Continue => {
            let source = input.as_ref().unwrap();
            let extension = number(data, "continuation_seconds", 8.0)?;
            let duration = source.seconds + extension;
            if extension <= 0.0 || duration > MAX_DURATION {
                return Err(format!(
                    "source + continuation must be in (0, {MAX_DURATION}] seconds"
                ));
            }
            let continuation_mode =
                text(data, "continuation_mode", "inpaint")?.to_ascii_lowercase();
            if !["inpaint", "latent_prefix"].contains(&continuation_mode.as_str()) {
                return Err("continuation_mode must be inpaint or latent_prefix".into());
            }
            let pad = tail_pad(data)?;
            canvas = duration + pad;
            target = (duration * SAMPLE_RATE).round_ties_even() as u64;
            let overlap = number(data, "mask_overlap", 0.2)?;
            let xfade = number(data, "splice_xfade", 0.03)?;
            if overlap < 0.0 || !(0.0..=1.0).contains(&xfade) {
                return Err(
                    "mask_overlap must be non-negative; splice_xfade must be in [0, 1]".into(),
                );
            }
            let applied_overlap = overlap.min((source.seconds - 0.05).max(0.0));
            let mask_start = source.seconds - applied_overlap;
            let splice = boolean(data, "splice_source", true)?;
            native["inpaint_start"] = json!(if splice { source.seconds } else { mask_start });
            native["inpaint_end"] = json!(canvas);
            native["fixed_prefix"] = json!(continuation_mode == "latent_prefix");
            native["mask_overlap"] = json!(overlap);
            native["splice_source"] = json!(splice);
            native["splice_xfade"] = json!(xfade);
            native["splice_gain_match"] = json!(boolean(data, "splice_gain_match", true)?);
            details = json!({"mode":continuation_mode,"source_duration":source.seconds,
                "continuation_seconds":extension,"total_duration":duration,"tail_mode":"regen_past",
                "tail_pad":pad,"gen_duration":canvas,"mask_overlap":applied_overlap,"requested_mask_overlap":overlap,
                "mask_start_seconds":mask_start,"mask_end_seconds":canvas,"sampler_type":sampler,
                "input_sr":source.sample_rate,"input_channels":source.channels,"target_samples":target,
                "splice_source":splice,"splice_xfade":xfade,"splice_gain_match":native["splice_gain_match"]});
        }
    }
    if target == 0 {
        return Err("requested output is shorter than one sample".into());
    }
    native["duration"] = json!(canvas);
    native["conditioning_seconds_total"] = json!(canvas);
    if mode == Mode::Continue {
        native["inpaint_padding_sec"] = json!(6);
    }
    native["target_samples"] = json!(target);
    let mut metadata = json!({"mode":mode.name(),"prompt":prompt,"negative_prompt":negative,
        "steps":steps as u32,"cfg_scale":cfg,"shift":shift,"sampler_type":sampler,"duration":canvas,"sample_rate":44100});
    metadata[mode.name()] = details.clone();
    details["prompt"] = json!(prompt);
    Ok(Translation {
        native,
        metadata,
        submitted: details,
        audio: input,
        mode,
    })
}

fn extract_bpm(prompt: &str) -> Option<f64> {
    let lower = prompt.to_ascii_lowercase();
    for (end, _) in lower.match_indices("bpm") {
        let prefix = lower[..end].trim_end();
        let start = prefix
            .rfind(|ch: char| !ch.is_ascii_digit() && ch != '.')
            .map_or(0, |at| at + 1);
        if let Ok(value) = prefix[start..].parse::<f64>() {
            return Some(value);
        }
    }
    None
}

use axum::{
    extract::{DefaultBodyLimit, Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

/// The transport reads prepared adapters through the host's registry. Desktop
/// retains its existing validation/training catalog; headless can omit it.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterLora {
    pub name: String,
    pub source_path: String,
    pub native_path: Option<String>,
    pub strength: f64,
    pub error: Option<String>,
}
pub trait AdapterRegistry: Send + Sync {
    fn entries(&self) -> Result<Vec<AdapterLora>, String>;
    fn paths(&self) -> (PathBuf, PathBuf);
    fn append_decoder<'a>(
        &'a self,
        request: &'a mut Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>>;
}
#[derive(Default)]
pub struct AdapterOptions {
    pub registry: Option<Arc<dyn AdapterRegistry>>,
    pub default_lora: Option<String>,
    pub decoder_correction: bool,
}

#[derive(Clone)]
pub struct AdapterState {
    upstream: String,
    uploads: PathBuf,
    client: reqwest::Client,
    jobs: Arc<Mutex<HashMap<String, ClientJob>>>,
    defaults: Map<String, Value>,
    registry: Option<Arc<dyn AdapterRegistry>>,
    default_lora: Option<String>,
    decoder_correction: bool,
}

struct AvailableLoras {
    entries: Vec<AdapterLora>,
    unavailable: Vec<Value>,
    model_loaded: Value,
    model: Value,
}

fn compatible_loras(entries: Vec<AdapterLora>, upstream: &Value) -> AvailableLoras {
    let native = upstream["loras"].as_array();
    let mut available = Vec::new();
    let mut unavailable = Vec::new();
    for entry in entries {
        let matched = entry.native_path.as_ref().is_some_and(|path| {
            let local = std::path::Path::new(path).canonicalize().ok();
            local.is_some()
                && native.is_some_and(|items| {
                    items.iter().any(|item| {
                        item["target"] == "dit"
                            && item["path"]
                                .as_str()
                                .and_then(|path| std::path::Path::new(path).canonicalize().ok())
                                == local
                    })
                })
        });
        if matched && entry.error.is_none() {
            available.push(entry);
        } else {
            unavailable.push(json!({"name":entry.name, "error":entry.error.unwrap_or_else(||
                if entry.native_path.is_none() { "Prepare this adapter for C++ first.".into() }
                else { "Adapter is incompatible with the selected model, or model components are missing.".into() })}));
        }
    }
    AvailableLoras {
        entries: available,
        unavailable,
        model_loaded: upstream["model_loaded"].clone(),
        model: upstream["model"].clone(),
    }
}

struct ClientJob {
    metadata: Value,
    mode: Mode,
    created: Instant,
}
type Reply = (StatusCode, Json<Value>);
static NEXT_UPLOAD: AtomicU64 = AtomicU64::new(0);

fn failure(status: StatusCode, error: impl ToString) -> Reply {
    (
        status,
        Json(json!({"success":false,"error":error.to_string()})),
    )
}

impl AdapterState {
    pub fn with_registry(
        mut self,
        registry: Arc<dyn AdapterRegistry>,
        decoder_correction: bool,
    ) -> Self {
        self.registry = Some(registry);
        self.decoder_correction = decoder_correction;
        self
    }

    pub fn new(port: u16, uploads: PathBuf) -> Result<Self, String> {
        Ok(Self {
            upstream: format!("http://127.0.0.1:{port}"),
            uploads,
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(30))
                // Native model loads can exceed cpp-httplib's short keep-alive
                // timeout. Fresh loopback connections avoid stale pooled sockets.
                .pool_max_idle_per_host(0)
                .build()
                .map_err(|error| error.to_string())?,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            defaults: Map::new(),
            registry: None,
            default_lora: None,
            decoder_correction: false,
        })
    }

    async fn available_loras(&self) -> Result<AvailableLoras, Reply> {
        let entries = if let Some(registry) = self.registry.clone() {
            match tokio::task::spawn_blocking(move || registry.entries()).await {
                Ok(Ok(entries)) => entries,
                Ok(Err(error)) => return Err(failure(StatusCode::SERVICE_UNAVAILABLE, error)),
                Err(error) => return Err(failure(StatusCode::INTERNAL_SERVER_ERROR, error)),
            }
        } else {
            Vec::new()
        };
        let upstream = match self.request(reqwest::Method::GET, "/loras", None).await {
            Ok((status, body)) if status.is_success() && body["success"] != false => body,
            Ok((status, body)) => return Err((status, Json(body))),
            Err(error) => return Err(failure(StatusCode::BAD_GATEWAY, error)),
        };
        Ok(compatible_loras(entries, &upstream))
    }

    /// Maps catalog names to native paths in `loras` and in every automation
    /// lane's `loras`, loading the catalog once.
    async fn map_requested_loras(&self, request: &mut Value) -> Result<(), Reply> {
        let named = |list: &Value| list.as_array().is_some_and(|list| !list.is_empty());
        let lanes_named = request["lora_lanes"]
            .as_array()
            .is_some_and(|lanes| lanes.iter().any(|lane| named(&lane["loras"])));
        if !named(&request["loras"]) && !lanes_named {
            return Ok(());
        }
        let catalog = self.available_loras().await?;
        self.map_lora_list(&catalog, &mut request["loras"])?;
        if let Some(lanes) = request["lora_lanes"].as_array_mut() {
            for lane in lanes {
                self.map_lora_list(&catalog, &mut lane["loras"])?;
            }
        }
        Ok(())
    }

    fn map_lora_list(&self, catalog: &AvailableLoras, list: &mut Value) -> Result<(), Reply> {
        let Some(loras) = list.as_array_mut() else {
            return Ok(());
        };
        for item in loras.iter_mut() {
            let name = item["name"].as_str().unwrap().trim().to_ascii_lowercase();
            let selected = if name == "default" {
                self.default_lora
                    .as_deref()
                    .or_else(|| catalog.entries.first().map(|entry| entry.name.as_str()))
                    .or_else(|| {
                        catalog
                            .unavailable
                            .first()
                            .and_then(|entry| entry["name"].as_str())
                    })
            } else {
                Some(name.as_str())
            };
            let Some(selected) = selected else {
                continue;
            };
            let Some(entry) = catalog.entries.iter().find(|entry| entry.name == selected) else {
                let detail = catalog
                    .unavailable
                    .iter()
                    .find(|entry| entry["name"] == selected)
                    .and_then(|entry| entry["error"].as_str())
                    .unwrap_or("No compatible registered adapter with this name.");
                return Err(failure(
                    StatusCode::BAD_REQUEST,
                    format!("LoRA '{selected}': {detail}"),
                ));
            };
            // Only the resolved catalog path crosses the private API boundary.
            item["path"] = json!(entry.native_path);
            item.as_object_mut().unwrap().remove("name");
        }
        // Legacy 'default' means no adapter when the registry is empty.
        loras.retain(|entry| entry.get("path").is_some());
        Ok(())
    }

    pub async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<(StatusCode, Value), String> {
        let mut request = self
            .client
            .request(method, format!("{}{path}", self.upstream));
        if matches!(path, "/load" | "/reload") {
            request = request.timeout(Duration::from_secs(180));
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request
            .send()
            .await
            .map_err(|error| format!("SA3 native service unavailable: {error}"))?;
        let status = response.status();
        let body = response
            .json::<Value>()
            .await
            .map_err(|error| format!("invalid native SA3 response: {error}"))?;
        Ok((status, body))
    }

    async fn lifecycle(&self, method: reqwest::Method, path: &str) -> Reply {
        match self.request(reqwest::Method::GET, "/health", None).await {
            Ok((status, body)) if status.is_success() => {
                if body.pointer("/capabilities/model_lifecycle") != Some(&json!(true)) {
                    return failure(StatusCode::SERVICE_UNAVAILABLE,
                        "SA3 native runtime lacks model_lifecycle support; install a compatible release before migration");
                }
            }
            Ok((status, body)) => return (status, Json(body)),
            Err(error) => return failure(StatusCode::BAD_GATEWAY, error),
        }
        match self.request(method, path, None).await {
            Ok((status, body)) => (status, Json(body)),
            Err(error) => failure(StatusCode::BAD_GATEWAY, error),
        }
    }

    async fn submit(&self, mode: Mode, mut body: Value) -> Reply {
        if let Some(data) = body.as_object_mut() {
            for (key, value) in &self.defaults {
                if key == "tail_pad_seconds"
                    && ["tail_pad_seconds", "tail_pad", "continuation_tail_pad"]
                        .iter()
                        .any(|alias| data.contains_key(*alias))
                {
                    continue;
                }
                data.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        let translation = tokio::task::spawn_blocking(move || translate(mode, body)).await;
        let mut translated = match translation {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => return failure(StatusCode::BAD_REQUEST, error),
            Err(error) => return failure(StatusCode::INTERNAL_SERVER_ERROR, error),
        };
        // Old native releases silently ignore unknown fields. Verify support
        // before dispatch so a pilot never silently changes the client mode.
        let health = match self.request(reqwest::Method::GET, "/health", None).await {
            Ok((status, body)) if status.is_success() => body,
            Ok((status, body)) => return (status, Json(body)),
            Err(error) => return failure(StatusCode::BAD_GATEWAY, error),
        };
        let mut required = vec!["conditioning_duration"];
        if uses_automation(&translated.native) {
            required.push("lora_lanes");
        }
        if mode == Mode::Continue {
            required.push("request_splice");
            if translated.native["fixed_prefix"] == true {
                required.push("fixed_prefix");
            }
        }
        for capability in required {
            if health["capabilities"][capability] != true {
                return failure(StatusCode::SERVICE_UNAVAILABLE,
                    format!("The installed sa3.cpp release lacks {capability}; prepare a compatible runtime before native migration."));
            }
        }
        if let Err(reply) = self.map_requested_loras(&mut translated.native).await {
            return reply;
        }
        if self.decoder_correction {
            let Some(registry) = &self.registry else {
                return failure(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Decoder correction registry is missing",
                );
            };
            if let Err(error) = registry.append_decoder(&mut translated.native).await {
                return failure(StatusCode::SERVICE_UNAVAILABLE, error);
            }
        }
        let upload = if let Some(audio) = translated.audio.take() {
            if let Err(error) = tokio::fs::create_dir_all(&self.uploads).await {
                return failure(StatusCode::INTERNAL_SERVER_ERROR, error);
            }
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = self.uploads.join(format!(
                "input-{stamp}-{}.wav",
                NEXT_UPLOAD.fetch_add(1, Ordering::Relaxed)
            ));
            let mut options = tokio::fs::OpenOptions::new();
            options.write(true).create_new(true);
            use tokio::io::AsyncWriteExt;
            match options.open(&path).await {
                Ok(mut file) => {
                    if let Err(error) = file.write_all(&audio.bytes).await {
                        drop(file);
                        let _ = tokio::fs::remove_file(&path).await;
                        return failure(StatusCode::INTERNAL_SERVER_ERROR, error);
                    }
                }
                Err(error) => return failure(StatusCode::INTERNAL_SERVER_ERROR, error),
            }
            translated.native["init_path"] = json!(path);
            Some(path)
        } else {
            None
        };
        let result = self
            .request(reqwest::Method::POST, "/generate", Some(&translated.native))
            .await;
        // Native request parsing reads init_path into GenParams synchronously
        // before returning. An HTTP reply means the source can be removed even
        // if the queued job later fails. On an ambiguous transport failure keep
        // the upload for maintenance after the native process has stopped.
        if result.is_ok() {
            if let Some(path) = &upload {
                let _ = tokio::fs::remove_file(path).await;
            }
        }
        let (status, response) = match result {
            Ok(value) => value,
            Err(error) => return failure(StatusCode::BAD_GATEWAY, error),
        };
        if !status.is_success() || response["success"] == false {
            return (status, Json(response));
        }
        let Some(id) = response["session_id"]
            .as_str()
            .filter(|id| valid_session_id(id))
        else {
            return failure(
                StatusCode::BAD_GATEWAY,
                "native SA3 returned an invalid session identifier",
            );
        };
        translated.metadata["seed"] = response["seed"].clone();
        let mut submitted = translated.submitted;
        submitted["success"] = json!(true);
        submitted["session_id"] = json!(id);
        submitted["seed"] = response["seed"].clone();
        let mut jobs = self.jobs.lock().await;
        jobs.retain(|_, job| job.created.elapsed() < Duration::from_secs(3600));
        jobs.insert(
            id.to_string(),
            ClientJob {
                metadata: translated.metadata,
                mode: translated.mode,
                created: Instant::now(),
            },
        );
        (status, Json(submitted))
    }
}

fn valid_session_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == b'-' || ch == b'_')
}

pub fn client_defaults(env: &[(&str, String)]) -> Map<String, Value> {
    let mapping = [
        ("SA3_PEAK_NORMALIZE_DB", "peak_normalize_db"),
        ("SA3_LIMITER_CEILING_DB", "limiter_ceiling_db"),
        ("SA3_LATENT_RESCALE", "latent_rescale"),
        ("SA3_LATENT_SHIFT", "latent_shift"),
        ("SA3_LATENT_TARGET_STD", "latent_target_std"),
        ("SA3_TAIL_PAD_SECONDS", "tail_pad_seconds"),
        ("SA3_CONTINUE_SPLICE_SOURCE", "splice_source"),
        ("SA3_CONTINUE_SPLICE_XFADE", "splice_xfade"),
        ("SA3_CONTINUE_SPLICE_GAIN_MATCH", "splice_gain_match"),
        ("SA3_CONTINUE_MASK_OVERLAP", "mask_overlap"),
    ];
    mapping
        .iter()
        .filter_map(|(name, key)| {
            let value = env
                .iter()
                .rev()
                .find(|(candidate, _)| candidate == name)?
                .1
                .trim();
            (!value.is_empty()).then(|| (key.to_string(), json!(value)))
        })
        .collect()
}

pub fn router(state: AdapterState) -> Router {
    Router::new()
        .route("/generate", post(generate))
        .route("/generate/loop", post(generate_loop))
        .route("/transform", post(transform))
        .route("/continue", post(continue_audio))
        .route("/poll_status/{id}", get(poll))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/load", post(load))
        .route("/reload", post(reload))
        .route("/unload", post(unload))
        .route("/loras", get(loras))
        .route("/prompts", get(prompts))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state)
}

/// Owned by the native service process entry. Reserve the public port before
/// spawning the child, so a conflict cannot leave a native process orphaned.
pub struct AdapterListener {
    listener: std::net::TcpListener,
    state: AdapterState,
}

pub struct AdapterHandle {
    task: tokio::task::JoinHandle<()>,
}

impl Drop for AdapterHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl AdapterListener {
    pub fn bind(
        public_port: u16,
        native_port: u16,
        uploads: PathBuf,
        defaults: Map<String, Value>,
        options: AdapterOptions,
    ) -> Result<Self, String> {
        if public_port == native_port {
            return Err("SA3 adapter and native server need different ports".into());
        }
        let listener = crate::runtime::bind_loopback(public_port).map_err(|error| {
            format!("Cannot bind SA3 client adapter on port {public_port}: {error}")
        })?;
        Self::from_listener(listener, native_port, uploads, defaults, options)
    }

    pub fn from_listener(
        listener: std::net::TcpListener,
        native_port: u16,
        uploads: PathBuf,
        defaults: Map<String, Value>,
        options: AdapterOptions,
    ) -> Result<Self, String> {
        if listener.local_addr().map_err(|e| e.to_string())?.port() == native_port {
            return Err("SA3 adapter and native server need different ports".into());
        }
        listener
            .set_nonblocking(true)
            .map_err(|error| error.to_string())?;
        let mut state = AdapterState::new(native_port, uploads)?;
        state.defaults = defaults;
        state.registry = options.registry;
        state.decoder_correction = options.decoder_correction;
        state.default_lora = options
            .default_lora
            .filter(|name| !name.trim().is_empty())
            .map(|name| name.trim().to_ascii_lowercase());
        Ok(Self { listener, state })
    }

    pub fn local_port(&self) -> Result<u16, String> {
        self.listener
            .local_addr()
            .map(|a| a.port())
            .map_err(|e| e.to_string())
    }
    pub async fn serve(self) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::from_std(self.listener)?;
        axum::serve(listener, router(self.state)).await
    }
    pub fn spawn(self) -> AdapterHandle {
        let task = tokio::spawn(async move {
            if let Err(error) = self.serve().await {
                log::error!("SA3 client adapter stopped: {error}");
            }
        });
        AdapterHandle { task }
    }
}

async fn generate(State(state): State<AdapterState>, Json(body): Json<Value>) -> Reply {
    state.submit(Mode::Generate, body).await
}
async fn generate_loop(State(state): State<AdapterState>, Json(body): Json<Value>) -> Reply {
    state.submit(Mode::Loop, body).await
}
async fn transform(State(state): State<AdapterState>, Json(body): Json<Value>) -> Reply {
    state.submit(Mode::Transform, body).await
}
async fn continue_audio(State(state): State<AdapterState>, Json(body): Json<Value>) -> Reply {
    state.submit(Mode::Continue, body).await
}

async fn health(State(state): State<AdapterState>) -> Reply {
    match state.request(reqwest::Method::GET, "/health", None).await {
        Ok((status, mut body)) => {
            body["decoder_lora_enabled"] = json!(state.decoder_correction);
            // capabilities.lora_lanes is sa3.cpp's. gary4local 0.7.0 passed it
            // through but dropped the lanes, so clients look for this too.
            body["lora_lanes_forwarded"] = json!(true);
            body["model_loaded"] = body["loaded"].clone();
            body["model_loading"] = body.get("loading").cloned().unwrap_or(json!(false));
            body["model_error"] = body.get("error").cloned().unwrap_or(Value::Null);
            (status, Json(body))
        }
        Err(error) => failure(StatusCode::BAD_GATEWAY, error),
    }
}

async fn ready(State(state): State<AdapterState>) -> Reply {
    state.lifecycle(reqwest::Method::GET, "/ready").await
}
async fn load(State(state): State<AdapterState>) -> Reply {
    state.lifecycle(reqwest::Method::POST, "/load").await
}
async fn reload(State(state): State<AdapterState>) -> Reply {
    state.lifecycle(reqwest::Method::POST, "/reload").await
}
async fn unload(State(state): State<AdapterState>) -> Reply {
    state.lifecycle(reqwest::Method::POST, "/unload").await
}

async fn loras(State(state): State<AdapterState>) -> Reply {
    let catalog = match state.available_loras().await {
        Ok(catalog) => catalog,
        Err(reply) => return reply,
    };
    let entries: Vec<_> = catalog.entries.iter().enumerate().map(|(index, entry)|
        json!({"index":index, "name":entry.name, "path":entry.source_path, "native_path":entry.native_path, "strength":entry.strength})).collect();
    (
        StatusCode::OK,
        Json(
            json!({"success":true, "loras":entries, "unavailable_loras":catalog.unavailable,
        "default_lora":state.default_lora, "model_loaded":catalog.model_loaded, "model":catalog.model,
        "lora_dir":state.registry.as_ref().map(|registry| registry.paths().0),
        "registry_path":state.registry.as_ref().map(|registry| registry.paths().1)}),
        ),
    )
}

async fn prompts(
    State(state): State<AdapterState>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Reply {
    // Keep repeated lora query parameters and encoding intact, while refusing
    // arbitrary filesystem paths before the native prompt parser opens files.
    let path = format!(
        "/prompts{}",
        query.map(|query| format!("?{query}")).unwrap_or_default()
    );
    let url = reqwest::Url::parse(&format!("{}{path}", state.upstream)).unwrap();
    for (key, value) in url.query_pairs().filter(|(key, _)| key == "lora") {
        let _ = key;
        for name in value
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            if sanitize_lora_name(&name.to_ascii_lowercase()).as_deref()
                != Some(name.to_ascii_lowercase().as_str())
            {
                return failure(
                    StatusCode::BAD_REQUEST,
                    "Prompt LoRA must be a catalog name",
                );
            }
        }
    }
    match state.request(reqwest::Method::GET, &path, None).await {
        Ok((status, body)) => (status, Json(body)),
        Err(error) => failure(StatusCode::BAD_GATEWAY, error),
    }
}

async fn poll(
    State(state): State<AdapterState>,
    Path(id): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Reply {
    if !valid_session_id(&id) {
        return failure(StatusCode::BAD_REQUEST, "invalid session identifier");
    }
    let consume = query
        .get("consume")
        .is_some_and(|value| value == "1" || value == "true");
    let path = format!(
        "/poll_status/{id}{}",
        if consume { "?consume=1" } else { "" }
    );
    let (status, mut body) = match state.request(reqwest::Method::GET, &path, None).await {
        Ok(value) => value,
        Err(error) => return failure(StatusCode::BAD_GATEWAY, error),
    };
    let mut jobs = state.jobs.lock().await;
    if let Some(job) = jobs.get(&id) {
        let active = !["completed", "failed"].contains(&body["status"].as_str().unwrap_or(""));
        body["transform_in_progress"] = json!(active && job.mode == Mode::Transform);
        if body["status"] == "completed" {
            let mut metadata = job.metadata.clone();
            if let Some(native) = body["meta"].as_object() {
                for (key, value) in native {
                    metadata[key] = value.clone();
                }
            }
            if job.mode == Mode::Continue {
                if let Some(splice) = metadata["splice"].as_object().cloned() {
                    for (key, value) in splice {
                        metadata["continue"][&key] = value;
                    }
                }
                metadata["continue"]["prefix_latent_tokens"] =
                    metadata["prefix_latent_tokens"].clone();
                metadata["continue"]["latent_sample_size"] = metadata["latent_sample_size"].clone();
            }
            body["meta"] = metadata;
        }
    }
    if (consume && body["status"] == "completed") || status == StatusCode::NOT_FOUND {
        jobs.remove(&id);
    }
    (status, Json(body))
}
#[cfg(test)]
mod tests {
    use super::*;

    fn source(channels: u16) -> String {
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(
                &mut buffer,
                hound::WavSpec {
                    channels,
                    sample_rate: 48000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for frame in 0..12000 {
                for _ in 0..channels {
                    writer.write_sample((frame % 300) as i16).unwrap();
                }
            }
            writer.finalize().unwrap();
        }
        STANDARD.encode(buffer.into_inner())
    }

    #[test]
    fn continuation_keeps_canvas_crop_and_overlap_separate() {
        let result = translate(Mode::Continue, json!({"prompt":"tone", "audio_data":source(1),
            "continuation_seconds":0.25,"continuation_tail_pad":2,"continuation_mode":"latent_prefix",
            "mask_overlap":0.1,"shift":"full","seed":4294967295u64})).unwrap();
        assert_eq!(result.native["inpaint_start"], 0.25);
        assert_eq!(result.native["inpaint_end"], 2.5);
        assert_eq!(result.native["conditioning_seconds_total"], 2.5);
        assert_eq!(result.native["inpaint_padding_sec"], 6);
        assert_eq!(result.native["target_samples"], 22050);
        assert_eq!(result.native["fixed_prefix"], true);
        assert_eq!(result.native["dist_shift"], "Full");
        assert_eq!(result.native["seed"], 4294967295u64);
        assert!(
            (result.metadata["continue"]["mask_start_seconds"]
                .as_f64()
                .unwrap()
                - 0.15)
                .abs()
                < 1e-9
        );
        assert_eq!(result.metadata["continue"]["input_channels"], 1);
        let no_splice = translate(
            Mode::Continue,
            json!({"prompt":"tone","audio_data":source(2),
            "mask_overlap":0.1,"splice_source":false}),
        )
        .unwrap();
        assert!((no_splice.native["inpaint_start"].as_f64().unwrap() - 0.15).abs() < 1e-9);
        assert_eq!(no_splice.native["fixed_prefix"], false);
    }

    #[test]
    fn transform_uses_source_rate_and_exact_length_without_an_inpaint_mask() {
        let result = translate(
            Mode::Transform,
            json!({"prompt":"tone","audio_data":source(2),"strength":1.4}),
        )
        .unwrap();
        assert_eq!(result.native["duration"], 0.75);
        assert_eq!(result.native["target_samples"], 11025);
        assert_eq!(result.native["init_noise_level"], 1.0);
        assert!(result.native.get("inpaint_start").is_none());
        assert_eq!(result.audio.unwrap().sample_rate, 48000);
    }

    #[test]
    fn generation_and_loops_keep_padding_separate_from_returned_audio() {
        let generated = translate(
            Mode::Generate,
            json!({"prompt":"tone","duration":3,"tail_pad":2}),
        )
        .unwrap();
        assert_eq!(generated.native["duration"], 5.0);
        assert_eq!(generated.native["conditioning_seconds_total"], 5.0);
        assert_eq!(generated.native["duration_padding_sec"], 6);
        assert_eq!(generated.native["target_samples"], 132300);
        let looped = translate(Mode::Loop, json!({"prompt":"120 BPM drums","bars":4})).unwrap();
        assert_eq!(looped.native["duration"], 10.0);
        assert_eq!(looped.native["target_samples"], 352800);
        assert_eq!(looped.metadata["loop"]["loop_duration"], 8.0);
    }

    #[test]
    fn invalid_audio_and_unsupported_controls_fail_before_queuing() {
        for body in [
            json!([]),
            json!({"prompt":""}),
            json!({"prompt":"tone","steps":1.2}),
            json!({"prompt":"tone","duration":"nan"}),
            json!({"prompt":"tone","shift":"bogus"}),
            json!({"prompt":"tone","sampler_type":"euler"}),
            json!({"prompt":"tone","loras":[{"name":"lora","interval_max":0.5}]}),
            json!({"prompt":"tone","loras":[{"name":"../lora"}]}),
            json!({"prompt":"tone","lora_lanes":{"loras":[]}}),
            json!({"prompt":"tone","lora_lanes":[{"loras":[{"name":"kev"}]}]}),
            json!({"prompt":"tone","lora_lanes":[{"loras":[{"name":"kev"}],"envelope":[[0]]}]}),
            json!({"prompt":"tone","lora_lanes":[{"loras":[{"name":"C:/kev.safetensors"}],"envelope":[[0,1]]}]}),
            json!({"prompt":"tone","lora_lanes":vec![json!({"envelope":[[0,1]]}); 9]}),
            json!({"prompt":"tone","lora_envelope":(0..257).map(|t| json!([t,1])).collect::<Vec<_>>()}),
            json!({"prompt":"tone","lora_envelope_mode":"sideways"}),
            json!({"prompt":"tone","lane_isolation":-1}),
        ] {
            assert!(translate(Mode::Generate, body.clone()).is_err(), "{body}");
        }
        assert!(translate(
            Mode::Transform,
            json!({"prompt":"tone","audio_data":STANDARD.encode(b"not a wave")})
        )
        .is_err());
        let mut truncated = STANDARD.decode(source(1)).unwrap();
        truncated.truncate(truncated.len() - 10);
        assert!(translate(
            Mode::Transform,
            json!({"prompt":"tone","audio_data":STANDARD.encode(truncated)})
        )
        .is_err());
    }

    #[test]
    fn client_optional_loudness_fields_preserve_off_and_defaults() {
        let result = translate(
            Mode::Generate,
            json!({"prompt":"tone","latent_rescale":"off",
            "peak_normalize_db":"off","limiter_ceiling_db":null,"latent_target_std":"0.7"}),
        )
        .unwrap();
        assert_eq!(result.native["latent_rescale"], 1);
        assert_eq!(result.native["latent_target_std"], 0.7);
        assert!(result.native["peak_normalize_db"].is_null());
        assert!(result.native["limiter_ceiling_db"].is_null());
    }

    #[test]
    fn automation_lanes_and_blend_reach_the_native_request() {
        let result = translate(
            Mode::Generate,
            json!({"prompt":"tone","loras":[{"name":"kev","strength":0.5}],
                "lora_lanes":[
                    {"loras":[{"name":"Keygen","strength":1.2}],"envelope":[[0,0],{"t":20,"amount":1}],
                     "prompt":"  demoscene chiptune  "},
                    {"envelope":[[0,1],[20,0]]}],
                "lane_isolation":1,"lane_isolation_until":0,"lane_average_until":0.9}),
        )
        .unwrap();
        let lanes = result.native["lora_lanes"].as_array().unwrap();
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes[0]["loras"], json!([{"name":"Keygen","strength":1.2}]));
        assert_eq!(lanes[0]["envelope"], json!([[0.0, 0.0], [20.0, 1.0]]));
        assert_eq!(lanes[0]["prompt"], "demoscene chiptune");
        assert_eq!(lanes[1]["loras"], json!([]));
        assert!(lanes[1].get("prompt").is_none());
        assert_eq!(result.native["lane_isolation"], 1.0);
        assert_eq!(result.native["lane_isolation_until"], 0.0);
        assert_eq!(result.native["lane_average_until"], 0.9);
        assert!(uses_automation(&result.native));

        let single = translate(
            Mode::Generate,
            json!({"prompt":"tone","loras":[{"name":"kev"}],"lora_envelope":[[0,0.2],[10,1.3]],
                "lora_envelope_mode":"Layer"}),
        )
        .unwrap();
        assert_eq!(
            single.native["lora_envelope"],
            json!([[0.0, 0.2], [10.0, 1.3]])
        );
        assert_eq!(single.native["lora_envelope_mode"], "layer");
        assert!(uses_automation(&single.native));

        let plain = translate(Mode::Generate, json!({"prompt":"tone","lora_lanes":null})).unwrap();
        for key in [
            "lora_lanes",
            "lora_envelope",
            "lora_envelope_mode",
            "lane_isolation",
        ] {
            assert!(plain.native.get(key).is_none(), "{key}");
        }
        assert!(!uses_automation(&plain.native));
    }

    #[test]
    fn lane_lora_names_map_to_native_paths() {
        let state = AdapterState::new(1, std::env::temp_dir()).unwrap();
        let kev: AdapterLora = serde_json::from_value(json!({
            "name":"kev","sourcePath":"kev.safetensors","nativePath":"C:/loras/kev.gguf","strength":1.0}))
        .unwrap();
        let catalog = AvailableLoras {
            entries: vec![kev],
            unavailable: vec![json!({"name":"old","error":"Prepare this adapter for C++ first."})],
            model_loaded: Value::Null,
            model: Value::Null,
        };
        let mut list = json!([{"name":"kev","strength":0.7}]);
        assert!(state.map_lora_list(&catalog, &mut list).is_ok());
        assert_eq!(list, json!([{"path":"C:/loras/kev.gguf","strength":0.7}]));
        let mut list = json!([{"name":"old"}]);
        let Err((status, body)) = state.map_lora_list(&catalog, &mut list) else {
            panic!("an unprepared lane LoRA was accepted");
        };
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.0["error"]
            .as_str()
            .unwrap()
            .contains("Prepare this adapter"));
    }

    #[test]
    fn automation_is_refused_by_a_runtime_without_lanes() {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let native = Router::new().route("/health", get(|| async { Json(json!({"loaded":false,
                "capabilities":{"conditioning_duration":true}})) }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let task = tokio::spawn(async move { axum::serve(listener, native).await.unwrap(); });
            let state = AdapterState::new(port, std::env::temp_dir()).unwrap();
            let (status, body) = state
                .submit(Mode::Generate, json!({"prompt":"tone","lora_lanes":[{"envelope":[[0,1]],"prompt":"bells"}]}))
                .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert!(body.0["error"].as_str().unwrap().contains("lora_lanes"));
            assert!(state.jobs.lock().await.is_empty());
            task.abort();
        });
    }

    #[test]
    fn transport_translates_jobs_uploads_and_consumed_metadata() {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let seen = Arc::new(Mutex::new(Vec::<Value>::new()));
            let recorded = seen.clone();
            let native = Router::new().route("/health", get(|| async { Json(json!({"loaded":false,
                "capabilities":{"conditioning_duration":true,"request_splice":true,"fixed_prefix":true}})) }))
                .route("/generate", post(move |Json(body): Json<Value>| {
                    let recorded = recorded.clone(); async move {
                        if let Some(path) = body["init_path"].as_str() { assert!(std::path::Path::new(path).is_file()); }
                        recorded.lock().await.push(body);
                        Json(json!({"success":true,"session_id":"native-job","seed":77}))
                    }
                }))
                .route("/poll_status/{id}", get(|| async { Json(json!({"success":true,"status":"completed",
                    "audio_data":"returned-audio","meta":{"seed":77,"prefix_latent_tokens":2,"latent_sample_size":12,
                    "splice":{"splice_applied":true,"mask_start_seconds":0.15,"splice_gain":1}}})) }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let native_port = listener.local_addr().unwrap().port();
            let native_task = tokio::spawn(async move { axum::serve(listener, native).await.unwrap(); });
            let root = std::env::temp_dir().join(format!("gary-sa3-adapter-{}", NEXT_UPLOAD.fetch_add(1,Ordering::Relaxed)));
            let mut state = AdapterState::new(native_port, root.clone()).unwrap();
            state.defaults = client_defaults(&[("SA3_TAIL_PAD_SECONDS", "6".into()),
                ("SA3_PEAK_NORMALIZE_DB", "off".into()),("SA3_CONTINUE_SPLICE_XFADE", "0.06".into())]);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let app = router(state.clone());
            let adapter_task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
            let client = reqwest::Client::new();
            let submitted: Value = client.post(format!("http://127.0.0.1:{port}/continue"))
                .json(&json!({"prompt":"tone","audio_data":source(2),"continuation_mode":"latent_prefix",
                    "continuation_seconds":0.25,"continuation_tail_pad":0,"mask_overlap":0.1}))
                .send().await.unwrap().error_for_status().unwrap().json().await.unwrap();
            assert_eq!(submitted["seed"],77);
            assert_eq!(submitted["target_samples"],22050);
            assert_eq!(state.jobs.lock().await.len(),1);
            assert_eq!(std::fs::read_dir(&root).unwrap().count(),0);
            assert_eq!(seen.lock().await[0]["fixed_prefix"],true);
            assert_eq!(seen.lock().await[0]["duration"],0.5);
            assert_eq!(seen.lock().await[0]["splice_xfade"],0.06);
            assert!(seen.lock().await[0]["peak_normalize_db"].is_null());
            assert!(seen.lock().await[0].get("audio_data").is_none());
            let polled: Value = client.get(format!("http://127.0.0.1:{port}/poll_status/native-job?consume=1"))
                .send().await.unwrap().json().await.unwrap();
            assert_eq!(polled["meta"]["mode"],"continue");
            assert_eq!(polled["meta"]["continue"]["splice_applied"],true);
            assert_eq!(polled["meta"]["continue"]["prefix_latent_tokens"],2);
            assert_eq!(polled["audio_data"],"returned-audio");
            assert!(state.jobs.lock().await.is_empty());
            adapter_task.abort(); native_task.abort();
            std::fs::remove_dir_all(root).unwrap();
        });
    }

    #[test]
    fn an_older_native_release_is_refused_before_uploading_or_submitting() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let native =
                    Router::new().route("/health", get(|| async { Json(json!({"loaded":false})) }));
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let port = listener.local_addr().unwrap().port();
                let task = tokio::spawn(async move {
                    axum::serve(listener, native).await.unwrap();
                });
                let root = std::env::temp_dir().join(format!(
                    "gary-sa3-refusal-{}",
                    NEXT_UPLOAD.fetch_add(1, Ordering::Relaxed)
                ));
                let state = AdapterState::new(port, root.clone()).unwrap();
                let (status, body) = state
                    .submit(
                        Mode::Continue,
                        json!({"prompt":"tone","audio_data":source(1)}),
                    )
                    .await;
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
                assert!(body.0["error"]
                    .as_str()
                    .unwrap()
                    .contains("conditioning_duration"));
                assert!(!root.exists());
                assert!(state.jobs.lock().await.is_empty());
                let (status, body) = state.lifecycle(reqwest::Method::POST, "/unload").await;
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
                assert!(body.0["error"]
                    .as_str()
                    .unwrap()
                    .contains("model_lifecycle"));
                task.abort();
            });
    }

    #[test]
    fn lifecycle_routes_preserve_native_readiness_failures_and_busy_responses() {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let native = Router::new()
                .route("/health", get(|| async { Json(json!({"loaded":false,"loading":false,
                    "error":"missing GGUF","last_load_seconds":0.25,"capabilities":{"model_lifecycle":true}})) }))
                .route("/ready", get(|| async { (StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"ready":false,"loading":false,"error":"missing GGUF"}))) }))
                .route("/load", post(|| async { (StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"success":false,"loaded":false,"error":"missing GGUF"}))) }))
                .route("/reload", post(|| async { (StatusCode::CONFLICT,
                    Json(json!({"success":false,"error":"generation in progress"}))) }))
                .route("/unload", post(|| async { Json(json!({"success":true,"status":"unloaded","loaded":false})) }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let native_port = listener.local_addr().unwrap().port();
            let native_task = tokio::spawn(async move { axum::serve(listener,native).await.unwrap(); });
            let state = AdapterState::new(native_port, PathBuf::from("unused-upload-root")).unwrap();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let task = tokio::spawn(async move { axum::serve(listener,router(state)).await.unwrap(); });
            let client = reqwest::Client::new();
            let body:Value = client.get(format!("http://127.0.0.1:{port}/health")).send().await.unwrap().json().await.unwrap();
            assert_eq!(body["model_loaded"],false);
            assert_eq!(body["model_loading"],false);
            assert_eq!(body["model_error"],"missing GGUF");
            assert_eq!(body["last_load_seconds"],0.25);
            for (method,path,expected) in [(reqwest::Method::GET,"ready",503),
                (reqwest::Method::POST,"load",503),(reqwest::Method::POST,"reload",409),
                (reqwest::Method::POST,"unload",200)] {
                let response = client.request(method,format!("http://127.0.0.1:{port}/{path}")).send().await.unwrap();
                assert_eq!(response.status().as_u16(),expected);
                let body:Value = response.json().await.unwrap();
                if path == "unload" { assert_eq!(body["success"],true); }
                else { assert!(body["error"].as_str().unwrap().contains(if path == "reload" { "in progress" } else { "missing GGUF" })); }
            }
            task.abort(); native_task.abort();
        });
    }
}

fn sanitize_lora_name(raw: &str) -> Option<String> {
    let normalized = raw.trim().to_lowercase();
    if normalized.is_empty() {
        return None;
    }
    if normalized
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
    {
        Some(normalized)
    } else {
        None
    }
}
