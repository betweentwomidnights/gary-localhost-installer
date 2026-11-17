//! Desktop registry and runtime bridge for the shared SA3 client adapter.
pub use gary4local_core::sa3_adapter::client_defaults;
use gary4local_core::sa3_adapter::{AdapterLora, AdapterOptions, AdapterRegistry};
use serde_json::{Map, Value};
use std::path::PathBuf;
use std::sync::Arc;

struct DesktopRegistry(PathBuf);
impl AdapterRegistry for DesktopRegistry {
    fn entries(&self) -> Result<Vec<AdapterLora>, String> {
        Ok(crate::sa3_loras::state(&self.0)?
            .entries
            .into_iter()
            .map(|e| AdapterLora {
                name: e.name,
                source_path: e.source_path,
                native_path: e.native_path,
                strength: e.strength,
                error: e.error,
            })
            .collect())
    }
    fn paths(&self) -> (PathBuf, PathBuf) {
        (
            crate::sa3_loras::adapters_dir(&self.0),
            crate::sa3_loras::catalog_path(&self.0),
        )
    }
    fn append_decoder<'a>(
        &'a self,
        request: &'a mut Value,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
        Box::pin(crate::sa3_decoder::append_request(&self.0, true, request))
    }
}
pub struct AdapterListener(gary4local_core::sa3_adapter::AdapterListener);
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
        registry_root: PathBuf,
        default_lora: Option<String>,
    ) -> Result<Self, String> {
        // Keep desktop catalog/prompt preparation and decoder defaults intact.
        crate::sa3_prompts::ensure_defaults(&registry_root)?;
        let options = AdapterOptions {
            registry: Some(Arc::new(DesktopRegistry(registry_root))),
            default_lora,
            decoder_correction: crate::sa3_use_decoder_lora_enabled(),
        };
        gary4local_core::sa3_adapter::AdapterListener::bind(
            public_port,
            native_port,
            uploads,
            defaults,
            options,
        )
        .map(Self)
    }
    pub fn spawn(self) -> AdapterHandle {
        let task = tauri::async_runtime::spawn(async move {
            if let Err(error) = self.0.serve().await {
                log::error!("SA3 client adapter stopped: {error}");
            }
        });
        AdapterHandle { task }
    }
}
impl crate::migration::ClientAdapter for AdapterListener {
    fn start(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        Box::new((*self).spawn())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use base64::{engine::general_purpose::STANDARD, Engine};
    use gary4local_core::sa3_adapter::{router, AdapterState};
    use serde_json::json;
    use std::io::Cursor;
    use std::time::{Duration, Instant};
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
            let variant = std::env::var("GARY4LOCAL_SA3_SMOKE_VARIANT").unwrap_or_else(|_| "small-music".into());
            let encoding = std::env::var("GARY4LOCAL_SA3_SMOKE_ENCODING").unwrap_or_else(|_| "q4_k_m".into());
            let ae_encoding = std::env::var("GARY4LOCAL_SA3_SMOKE_AE_ENCODING").unwrap_or_else(|_| "f16".into());
            let device = std::env::var("GARY4LOCAL_SA3_SMOKE_DEVICE").unwrap_or_else(|_| "cpu".into());
            let lora_name = std::env::var("GARY4LOCAL_SA3_SMOKE_LORA_NAME").unwrap_or_else(|_| "koan".into());
            let mut command = tokio::process::Command::new(binary);
            command.args(["--model",variant.as_str(),"--encoding",encoding.as_str(),"--t5-encoding","f16",
                "--ae-encoding",ae_encoding.as_str(),"--threads","8","--models-dir"])
                .arg(models).arg("--port").arg(native_port.to_string()).env("SA3_DEVICE",&device)
                .current_dir(&root).stdout(log.try_clone().unwrap()).stderr(log).kill_on_drop(true);
            let registry = std::env::var_os("GARY4LOCAL_SA3_LORA_REGISTRY").map(PathBuf::from);
            let real_prompt_pool = registry.as_ref().is_some_and(|registry| registry.join(format!("sa3/prompts/{lora_name}.json")).is_file());
            if let Some(registry) = &registry {
                command.env("SA3_ADAPTERS_DIR",crate::sa3_loras::adapters_dir(registry));
                let prompts = if real_prompt_pool { registry.join("sa3/prompts") } else { root.join("prompts") };
                if !real_prompt_pool {
                std::fs::create_dir_all(&prompts).unwrap();
                std::fs::write(prompts.join("defaults.json"),json!({"version":1,"dice":{"generic":["default tone"]}}).to_string()).unwrap();
                std::fs::write(prompts.join(format!("{lora_name}.json")),json!({"dice":{"generic":["koan tone"]}}).to_string()).unwrap();
                }
                command.env("SA3_PROMPTS_DIR",prompts);
            }
            crate::hide_console_window(&mut command);
            let mut native = command.spawn().unwrap();
            let mut state = AdapterState::new(native_port, root.join("uploads")).unwrap();
            if let Some(registry) = registry.clone() {
                state = state.with_registry(Arc::new(DesktopRegistry(registry)), std::env::var("GARY4LOCAL_SA3_SMOKE_DECODER").is_ok_and(|value| value == "1"));
            }
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
            if registry.is_some() {
                let menu:Value = client.get(format!("{base}/loras")).send().await.unwrap().json().await.unwrap();
                assert_eq!(menu["loras"][0]["name"],lora_name,"converted Koan missing from small model menu: {menu}");
                assert_eq!(menu["loras"].as_array().unwrap().len(),1,"failed legacy conversion must remain unavailable");
                let prompts:Value = client.get(format!("{base}/prompts?lora={lora_name}&lora=missing")).send().await.unwrap().json().await.unwrap();
                if real_prompt_pool { assert!(!prompts["prompts"]["dice"]["instrumental"].as_array().unwrap().is_empty(),"trained caption pool missing: {prompts}"); }
                else { assert_eq!(prompts["prompts"]["dice"]["generic"][0],"koan tone"); }
                assert_eq!(prompts["missing_loras"],json!(["missing"]));
                assert_eq!(client.get(format!("{base}/prompts?lora=..%2Fsecret")).send().await.unwrap().status(),StatusCode::BAD_REQUEST);
                let error = client.post(format!("{base}/transform")).json(&json!({"prompt":"tone","audio_data":source(1),"lora":"legacy"})).send().await.unwrap();
                assert_eq!(error.status(),StatusCode::BAD_REQUEST);
                assert!(!root.join("uploads").exists(),"failed LoRA request must not stage audio");
                println!("PASS registered LoRA menu, native prompt pools, and rejection before audio staging");
            }
            assert_eq!(client.get(format!("{base}/ready")).send().await.unwrap().status(),StatusCode::SERVICE_UNAVAILABLE);
            let loaded:Value = client.post(format!("{base}/load")).send().await.unwrap().error_for_status().unwrap().json().await.unwrap();
            assert_eq!(loaded["status"],"loaded");
            let health:Value = client.get(format!("{base}/health")).send().await.unwrap().json().await.unwrap();
            assert_eq!(health["decoder_lora_enabled"], std::env::var("GARY4LOCAL_SA3_SMOKE_DECODER").is_ok_and(|value| value == "1"));
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
                if registry.is_some() { body["lora"] = json!(if std::env::var("GARY4LOCAL_SA3_SMOKE_DECODER").is_ok_and(|value| value == "1") && matches!(route, "generate" | "continue") { "none" } else { "default" }); body["lora_strength"] = json!(0.8); }
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
            if let Ok(duration) = std::env::var("GARY4LOCAL_SA3_SMOKE_LISTEN_DURATION") {
                let duration: f64 = duration.parse().unwrap();
                let response = client.post(format!("{base}/generate")).json(&json!({"prompt":"glitch hop, neurofunk, electronica, intricate bass and syncopated percussion","duration":duration,"tail_pad":0,"steps":20,"seed":42,"lora":lora_name,"lora_strength":0.8})).send().await.unwrap().error_for_status().unwrap();
                let submitted: Value = response.json().await.unwrap();
                let url = format!("{base}/poll_status/{}", submitted["session_id"].as_str().unwrap());
                let deadline = Instant::now() + Duration::from_secs(300);
                let completed = loop {
                    let polled: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
                    assert_ne!(polled["status"], "failed", "listening sample: {polled}");
                    if polled["status"] == "completed" { break polled; }
                    assert!(Instant::now() < deadline, "listening sample timed out");
                    tokio::time::sleep(Duration::from_millis(200)).await;
                };
                let wav = STANDARD.decode(completed["audio_data"].as_str().unwrap()).unwrap();
                let reader = hound::WavReader::new(Cursor::new(&wav)).unwrap();
                assert_eq!(reader.duration(), (duration * 44100.0).round() as u32);
                assert!(reader.into_samples::<i16>().any(|sample| sample.unwrap() != 0), "listening sample is silent");
                std::fs::write(root.join("trained-adapter-listening.wav"), wav).unwrap();
                std::fs::write(root.join("trained-adapter-listening.json"), serde_json::to_vec_pretty(&completed["meta"]).unwrap()).unwrap();
                println!("PASS trained adapter listening sample: {duration} seconds, 20 steps");
            }
            assert_eq!(client.post(format!("{base}/reload")).send().await.unwrap().status(),StatusCode::OK);
            assert_eq!(client.post(format!("{base}/unload")).send().await.unwrap().status(),StatusCode::OK);
            assert_eq!(client.get(format!("{base}/ready")).send().await.unwrap().status(),StatusCode::SERVICE_UNAVAILABLE);
            println!("PASS native lifecycle through public adapter: load, readiness, reload, unload");
            if registry.is_some() {
                let selected = client.post(format!("http://127.0.0.1:{native_port}/models/select"))
                    .json(&json!({"variant":if variant == "medium" { "small-music" } else { "medium" },"encoding":"q4_k_m"})).send().await.unwrap();
                assert_eq!(selected.status(),StatusCode::OK,"medium test model components must be available");
                let menu:Value = client.get(format!("{base}/loras")).send().await.unwrap().json().await.unwrap();
                assert!(menu["loras"].as_array().unwrap().is_empty(),"trained adapter must be excluded from incompatible model: {menu}");
                let rejected = client.post(format!("{base}/transform")).json(&json!({"prompt":"tone","audio_data":source(1),"lora":lora_name})).send().await.unwrap();
                assert_eq!(rejected.status(),StatusCode::BAD_REQUEST);
                println!("PASS incompatible small adapter omitted and rejected for medium model");
            }
            native.kill().await.unwrap(); native.wait().await.unwrap(); task.abort();
            assert_eq!(std::fs::read_dir(root.join("uploads")).unwrap().count(),0);
        });
    }
}
