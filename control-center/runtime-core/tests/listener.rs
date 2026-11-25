//! The LAN listener refuses requests without the pairing token and forwards
//! the rest, streaming, to the loopback service named by the path prefix.
use axum::extract::Request;
use axum::Router;
use gary4local_core::listener::{self, Settings, Shared};
use gary4local_core::service::Service;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Echoes what the service received, so tests can see what was forwarded.
async fn echo(request: Request) -> axum::Json<Value> {
    let (parts, body) = request.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let header = |name: &str| {
        parts
            .headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    axum::Json(json!({
        "method": parts.method.as_str(),
        "path": parts.uri.path(),
        "query": parts.uri.query(),
        "authorization": header("authorization"),
        "host": header("host"),
        "forwarded_for": header("x-forwarded-for"),
        "custom": header("x-gary-test"),
        "body_len": bytes.len(),
        "body_sum": bytes.iter().map(|b| *b as u64).sum::<u64>(),
    }))
}

async fn spawn_echo() -> u16 {
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = socket.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(socket, Router::new().fallback(echo))
            .await
            .unwrap();
    });
    port
}

/// A port nothing listens on, held bound so the OS can't hand it to another
/// test's connection while this one is still dialing it.
fn closed_port() -> (u16, tokio::net::TcpSocket) {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    (socket.local_addr().unwrap().port(), socket)
}

struct Harness {
    url: String,
    dir: PathBuf,
    settings: PathBuf,
    shared: Arc<Shared>,
    token: String,
    client: reqwest::Client,
}
impl Drop for Harness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

async fn harness(ports: HashMap<Service, u16>) -> Harness {
    // The counter matters: tests start together, and Windows can give two of
    // them the same timestamp, which would make them share a token file.
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "gary-listener-test-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let settings_path = listener::settings_path(&dir.join("sa3.json"));
    let mut settings = Settings {
        enabled: true,
        port: 0,
        bind: "127.0.0.1".parse().unwrap(),
        ..Settings::default()
    };
    let token = settings.rotate().unwrap();
    settings.save(&settings_path).unwrap();
    let socket = listener::bind(&settings).await.unwrap();
    let url = format!("http://{}", socket.local_addr().unwrap());
    let shared = Shared::new(settings_path.clone(), ports);
    tokio::spawn(listener::serve(socket, shared.clone()));
    Harness {
        url,
        dir,
        settings: settings_path,
        shared,
        token,
        client: reqwest::Client::builder().no_proxy().build().unwrap(),
    }
}

impl Harness {
    async fn get(&self, path: &str, token: Option<&str>) -> (u16, Value) {
        let mut request = self.client.get(format!("{}{path}", self.url));
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap_or(Value::Null))
    }
}

#[tokio::test]
async fn requests_without_the_right_token_never_reach_a_service() {
    let echo = spawn_echo().await;
    let h = harness(HashMap::from([(Service::Sa3, echo)])).await;

    let (status, body) = h.get("/sa3/health", None).await;
    assert_eq!(status, 401);
    assert_eq!(body["error"], "pairing_token_required");

    let (status, body) = h.get("/sa3/health", Some("gary-wrong-token")).await;
    assert_eq!(status, 401);
    assert_eq!(body["error"], "pairing_token_rejected");

    // Authentication comes before routing, so unknown paths reveal nothing.
    let (status, _) = h.get("/ssh/anything", None).await;
    assert_eq!(status, 401);

    let raw = h
        .client
        .get(format!("{}/sa3/health", h.url))
        .header("authorization", format!("Basic {}", h.token))
        .send()
        .await
        .unwrap();
    assert_eq!(raw.status().as_u16(), 401);
    assert!(raw.headers().contains_key("www-authenticate"));

    let stats = h.shared.stats();
    assert_eq!(stats.forwarded, 0);
    assert_eq!(stats.rejected, 4);
    assert_eq!(stats.last_rejected_from, Some("127.0.0.1".parse().unwrap()));
}

#[tokio::test]
async fn forwards_to_the_prefixed_service_without_the_token() {
    let sa3 = spawn_echo().await;
    let carey = spawn_echo().await;
    let h = harness(HashMap::from([
        (Service::Sa3, sa3),
        (Service::Carey, carey),
    ]))
    .await;

    let (status, body) = h
        .get("/sa3/poll_status/abc?consume=1", Some(&h.token))
        .await;
    assert_eq!(status, 200);
    assert_eq!(body["path"], "/poll_status/abc");
    assert_eq!(body["query"], "consume=1");
    assert_eq!(
        body["authorization"],
        Value::Null,
        "the token stays on the listener"
    );
    assert_eq!(body["host"], format!("127.0.0.1:{sa3}"));
    assert_eq!(body["forwarded_for"], "127.0.0.1");

    // Tokens are case-insensitive, and gary4juce's tab names route too.
    let (status, body) = h.get("/CAREY", Some(&h.token.to_ascii_uppercase())).await;
    assert_eq!(status, 200);
    assert_eq!(body["path"], "/");
    assert_eq!(body["host"], format!("127.0.0.1:{carey}"));

    let posted: Value = h
        .client
        .post(format!("{}/sa3/generate", h.url))
        .bearer_auth(&h.token)
        .header("x-gary-test", "kept")
        .json(&json!({"prompt": "dub techno"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(posted["method"], "POST");
    assert_eq!(posted["path"], "/generate");
    assert_eq!(posted["custom"], "kept");
    assert_eq!(
        posted["body_len"],
        json!({"prompt": "dub techno"}).to_string().len()
    );
    assert_eq!(h.shared.stats().forwarded, 3);
}

#[tokio::test]
async fn every_request_on_a_kept_alive_connection_is_checked() {
    let echo = spawn_echo().await;
    let h = harness(HashMap::from([(Service::Sa3, echo)])).await;
    // One client, one pooled connection: the second request must still fail.
    assert_eq!(h.get("/sa3/health", Some(&h.token)).await.0, 200);
    assert_eq!(h.get("/sa3/health", Some("gary-nope")).await.0, 401);
    assert_eq!(h.get("/sa3/health", None).await.0, 401);
    assert_eq!(h.get("/sa3/health", Some(&h.token)).await.0, 200);
}

#[tokio::test]
async fn rotating_or_revoking_the_token_takes_effect_immediately() {
    let echo = spawn_echo().await;
    let h = harness(HashMap::from([(Service::Sa3, echo)])).await;
    assert_eq!(h.get("/sa3/health", Some(&h.token)).await.0, 200);

    // Another Gary process rotates the token on disk.
    let mut settings = Settings::load(&h.settings).unwrap();
    let rotated = settings.rotate().unwrap();
    settings.save(&h.settings).unwrap();
    assert_eq!(h.get("/sa3/health", Some(&h.token)).await.0, 401);
    assert_eq!(h.get("/sa3/health", Some(&rotated)).await.0, 200);

    settings.revoke();
    settings.save(&h.settings).unwrap();
    assert_eq!(h.get("/sa3/health", Some(&rotated)).await.0, 401);
    assert_eq!(h.get("/sa3/health", Some("")).await.0, 401);
}

#[tokio::test]
async fn unknown_and_offline_services_get_clear_errors() {
    let (port, _held) = closed_port();
    let h = harness(HashMap::from([(Service::Sa3, port)])).await;
    let (status, body) = h.get("/sa3/health", Some(&h.token)).await;
    assert_eq!(status, 502);
    assert_eq!(body["error"], "service_offline");

    let (status, body) = h.get("/ssh/health", Some(&h.token)).await;
    assert_eq!(status, 404);
    assert_eq!(body["error"], "unknown_service");
    let (status, _) = h.get("/", Some(&h.token)).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn large_uploads_arrive_intact() {
    let echo = spawn_echo().await;
    let h = harness(HashMap::from([(Service::Carey, echo)])).await;
    // Bigger than axum's 2 MB default extractor limit, which must not apply.
    let upload: Vec<u8> = (0..24 * 1024 * 1024u32).map(|i| (i % 251) as u8).collect();
    let expected: u64 = upload.iter().map(|b| *b as u64).sum();
    let body: Value = h
        .client
        .post(format!("{}/carey/upload", h.url))
        .bearer_auth(&h.token)
        .body(upload.clone())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["body_len"], upload.len());
    assert_eq!(body["body_sum"], expected);
}

#[tokio::test]
async fn responses_stream_through_without_buffering() {
    // A service that sends one chunk, then waits until the client has seen
    // it before finishing. A buffering proxy would deadlock here.
    let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = socket.local_addr().unwrap().port();
    let (release, released) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let (mut stream, _) = socket.accept().await.unwrap();
        let mut head = Vec::new();
        let mut byte = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).await.unwrap();
            head.push(byte[0]);
        }
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/plain\r\n\r\n5\r\nfirst\r\n")
            .await
            .unwrap();
        stream.flush().await.unwrap();
        released.await.unwrap();
        stream.write_all(b"6\r\nsecond\r\n0\r\n\r\n").await.unwrap();
    });
    let h = harness(HashMap::from([(Service::Yuey, port)])).await;
    let mut response = h
        .client
        .get(format!("{}/yuey/stream", h.url))
        .bearer_auth(&h.token)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let first = tokio::time::timeout(std::time::Duration::from_secs(5), response.chunk())
        .await
        .expect("first chunk arrives before the service finishes")
        .unwrap()
        .unwrap();
    assert_eq!(&first[..], b"first");
    release.send(()).unwrap();
    let mut rest = Vec::new();
    while let Some(chunk) = response.chunk().await.unwrap() {
        rest.extend_from_slice(&chunk);
    }
    assert_eq!(rest, b"second");
}

#[test]
fn service_ports_fall_back_to_the_desktop_ports() {
    let dir = std::env::temp_dir().join(format!("gary-listener-ports-{}", std::process::id()));
    let ports = listener::service_ports(&dir.join("sa3.json"));
    assert_eq!(ports[&Service::Gary], 8000);
    assert_eq!(ports[&Service::Melodyflow], 8002);
    assert_eq!(ports[&Service::Carey], 8003);
    assert_eq!(ports[&Service::StableAudio], 8005);
    assert_eq!(ports[&Service::Sa3], 8006);
    assert_eq!(ports[&Service::Yuey], 8007);
    assert_eq!(ports[&Service::Foundation], 8015);
}
