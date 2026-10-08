//! Gary's legacy HTTP contract translated to sa3.cpp's unified generation API.
//! Audio algorithms stay in the native pipeline. This module owns upload files,
//! client defaults, exact requested lengths and client-facing job metadata.

use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Map, Value};
use std::io::Cursor;

const SAMPLE_RATE: f64 = 44100.0;
const MAX_DURATION: f64 = 380.0;

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
    // Published adapters currently support full-range application. Reject
    // unsupported intervals/filters instead of changing their meaning silently.
    let mut loras = Vec::new();
    if let Some(value) = data.get("loras").filter(|value| !value.is_null()) {
        for entry in value.as_array().ok_or("loras must be a list")? {
            let entry = entry
                .as_object()
                .ok_or("each loras entry must be an object")?;
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
    } else if let Some(name) = data
        .get("lora")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
    {
        if name.contains(['/', '\\', ':']) {
            return Err("LoRA must be a catalog name".into());
        }
        loras.push(json!({"name":name, "strength":number(data, "lora_strength", 1.0)?}));
    }
    native["loras"] = json!(loras);
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

#[derive(Clone)]
pub struct AdapterState {
    upstream: String,
    uploads: PathBuf,
    client: reqwest::Client,
    jobs: Arc<Mutex<HashMap<String, ClientJob>>>,
    defaults: Map<String, Value>,
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
    pub fn new(port: u16, uploads: PathBuf) -> Result<Self, String> {
        Ok(Self {
            upstream: format!("http://127.0.0.1:{port}"),
            uploads,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                // Native model loads can exceed cpp-httplib's short keep-alive
                // timeout. Fresh loopback connections avoid stale pooled sockets.
                .pool_max_idle_per_host(0)
                .build()
                .map_err(|error| error.to_string())?,
            jobs: Arc::new(Mutex::new(HashMap::new())),
            defaults: Map::new(),
        })
    }

    async fn request(
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
    task: tauri::async_runtime::JoinHandle<()>,
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
    ) -> Result<Self, String> {
        if public_port == native_port {
            return Err("SA3 adapter and native server need different ports".into());
        }
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, public_port))
            .map_err(|error| {
                format!("Cannot bind SA3 client adapter on port {public_port}: {error}")
            })?;
        listener
            .set_nonblocking(true)
            .map_err(|error| error.to_string())?;
        let mut state = AdapterState::new(native_port, uploads)?;
        state.defaults = defaults;
        Ok(Self { listener, state })
    }

    pub fn spawn(self) -> AdapterHandle {
        let task = tauri::async_runtime::spawn(async move {
            let result = match tokio::net::TcpListener::from_std(self.listener) {
                Ok(listener) => axum::serve(listener, router(self.state)).await,
                Err(error) => Err(error),
            };
            if let Err(error) = result {
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

    #[test]
    #[ignore = "requires a compatible real native server and existing GGUF models"]
    fn real_native_adapter_smoke() {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let binary = std::env::var_os("GARY4LOCAL_SA3_SMOKE_BINARY").expect("set GARY4LOCAL_SA3_SMOKE_BINARY");
            let models = std::env::var_os("GARY4LOCAL_SA3_SMOKE_MODELS").expect("set GARY4LOCAL_SA3_SMOKE_MODELS");
            let root = PathBuf::from(std::env::var_os("GARY4LOCAL_SA3_ADAPTER_SMOKE_ROOT").expect("set an isolated artifact root"));
            assert!(root.is_absolute());
            std::fs::create_dir_all(&root).unwrap();
            let reservation = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let native_port = reservation.local_addr().unwrap().port(); drop(reservation);
            let log = std::fs::File::create(root.join("native-server.log")).unwrap();
            let mut command = tokio::process::Command::new(binary);
            command.args(["--model","small-music","--encoding","q4_k_m","--t5-encoding","f16",
                "--ae-encoding","f16","--threads","8","--models-dir"])
                .arg(models).arg("--port").arg(native_port.to_string()).env("SA3_DEVICE","cpu")
                .current_dir(&root).stdout(log.try_clone().unwrap()).stderr(log).kill_on_drop(true);
            crate::hide_console_window(&mut command);
            let mut native = command.spawn().unwrap();
            let state = AdapterState::new(native_port, root.join("uploads")).unwrap();
            let deadline = Instant::now() + Duration::from_secs(20);
            loop {
                assert!(native.try_wait().unwrap().is_none(), "server exited; inspect native-server.log");
                if state.request(reqwest::Method::GET,"/health",None).await.is_ok() { break; }
                assert!(Instant::now() < deadline,"server startup timeout");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let task = tokio::spawn(async move { axum::serve(listener,router(state)).await.unwrap(); });
            let client = reqwest::Client::new();
            let base = format!("http://127.0.0.1:{port}");
            assert_eq!(client.get(format!("{base}/ready")).send().await.unwrap().status(),StatusCode::SERVICE_UNAVAILABLE);
            let loaded:Value = client.post(format!("{base}/load")).send().await.unwrap().error_for_status().unwrap().json().await.unwrap();
            assert_eq!(loaded["status"],"loaded");
            let health:Value = client.get(format!("{base}/health")).send().await.unwrap().json().await.unwrap();
            assert_eq!(health["model_loaded"],true,"unexpected health: {health}"); assert_eq!(health["model_loading"],false); assert!(health["model_error"].is_null());
            assert_eq!(client.get(format!("{base}/ready")).send().await.unwrap().status(),StatusCode::OK);
            for (route, mut body, expected_samples) in [
                ("generate",json!({"duration":0.25,"tail_pad":0}),11025),
                ("generate/loop",json!({"bpm":960,"bars":4}),44100),
                ("transform",json!({"audio_data":source(1),"strength":0.5}),11025),
                ("continue",json!({"audio_data":source(2),"continuation_seconds":0.25,"tail_pad":0,
                    "continuation_mode":"latent_prefix","mask_overlap":0.1,"splice_gain_match":false}),22050),
            ] {
                body["prompt"] = json!("a soft tone"); body["steps"] = json!(1); body["seed"] = json!(4294967295u64);
                body["peak_normalize_db"] = Value::Null; body["limiter_ceiling_db"] = Value::Null;
                let response = client.post(format!("http://127.0.0.1:{port}/{route}")).json(&body).send().await.unwrap();
                let status = response.status(); let submitted:Value = response.json().await.unwrap();
                assert!(status.is_success(),"{route}: {submitted}");
                assert_eq!(submitted["seed"],4294967295u64);
                let url = format!("http://127.0.0.1:{port}/poll_status/{}",submitted["session_id"].as_str().unwrap());
                let deadline = Instant::now() + Duration::from_secs(180);
                let completed = loop {
                    let polled:Value = client.get(&url).send().await.unwrap().json().await.unwrap();
                    assert_ne!(polled["status"],"failed","{route}: {polled}");
                    if polled["status"] == "completed" { break polled; }
                    assert!(Instant::now() < deadline,"{route} timed out");
                    tokio::time::sleep(Duration::from_millis(200)).await;
                };
                let wav = STANDARD.decode(completed["audio_data"].as_str().unwrap()).unwrap();
                let reader = hound::WavReader::new(Cursor::new(&wav)).unwrap();
                assert_eq!(reader.spec().sample_rate,44100); assert_eq!(reader.spec().channels,2);
                assert_eq!(reader.duration(),expected_samples);
                let expected_conditioning: f64 = match route { "generate" => 0.25, "generate/loop" => 3.0, "transform" => 0.75, _ => 0.5 };
                assert_eq!(completed["meta"]["conditioning_seconds_total"],expected_conditioning);
                assert_eq!(completed["meta"]["conditioning_latent_frames"], (expected_conditioning * 44100.0 / 4096.0).ceil() as u32);
                if route == "continue" {
                    assert_eq!(completed["meta"]["continue"]["prefix_latent_tokens"],2);
                    assert_eq!(completed["meta"]["continue"]["splice_applied"],true);
                }
                let consumed:Value = client.get(format!("{url}?consume=1")).send().await.unwrap().json().await.unwrap();
                assert_eq!(consumed,completed);
                assert_eq!(client.get(&url).send().await.unwrap().status(),StatusCode::NOT_FOUND);
                std::fs::write(root.join(format!("{}.wav",route.replace('/' , "-"))),wav).unwrap();
                std::fs::write(root.join(format!("{}.json",route.replace('/' , "-"))),serde_json::to_vec_pretty(&completed["meta"]).unwrap()).unwrap();
                println!("PASS {route}: {expected_samples} samples, recalled seed and matching normal/consume metadata");
            }
            assert_eq!(client.post(format!("{base}/reload")).send().await.unwrap().status(),StatusCode::OK);
            assert_eq!(client.post(format!("{base}/unload")).send().await.unwrap().status(),StatusCode::OK);
            assert_eq!(client.get(format!("{base}/ready")).send().await.unwrap().status(),StatusCode::SERVICE_UNAVAILABLE);
            println!("PASS native lifecycle through public adapter: load, readiness, reload, unload");
            native.kill().await.unwrap(); native.wait().await.unwrap(); task.abort();
            assert_eq!(std::fs::read_dir(root.join("uploads")).unwrap().count(),0);
        });
    }
}
