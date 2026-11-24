//! Opt-in LAN listener for headless Gary.
//!
//! One network-facing port forwards authenticated HTTP requests to the
//! services, which stay bound to loopback. The first path segment picks the
//! service (`/sa3/health` reaches `127.0.0.1:8006/health`) and every request
//! must carry `Authorization: Bearer <pairing token>`. Only the token's SHA-256
//! is stored; it is re-read for every request, so `listen rotate` or
//! `listen revoke` takes effect immediately in every running Gary.
//!
//! There is no TLS. The token and audio cross the LAN in cleartext.
use crate::service::Service;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode, Uri, Version};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const DEFAULT_PORT: u16 = 8020;
pub const SETTINGS_FILE: &str = "listener.json";

/// Path prefixes clients may use. gary4juce's tab names come first; the
/// gary4local service ids are accepted too.
pub const ROUTES: [(&str, Service); 10] = [
    ("gary", Service::Gary),
    ("terry", Service::Melodyflow),
    ("melodyflow", Service::Melodyflow),
    ("jerry", Service::StableAudio),
    ("stable-audio", Service::StableAudio),
    ("carey", Service::Carey),
    ("foundation", Service::Foundation),
    ("foundation-1", Service::Foundation),
    ("sa3", Service::Sa3),
    ("yuey", Service::Yuey),
];

fn default_port() -> u16 {
    DEFAULT_PORT
}
fn default_bind() -> IpAddr {
    IpAddr::V4(Ipv4Addr::UNSPECIFIED)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// Start the listener with the interactive console. Off until turned on.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Address to bind. 0.0.0.0 means every IPv4 interface.
    #[serde(default = "default_bind")]
    pub bind: IpAddr,
    /// Hex SHA-256 of the pairing token. The token itself is never stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_created_unix: Option<u64>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: false,
            port: DEFAULT_PORT,
            bind: default_bind(),
            token_sha256: None,
            token_created_unix: None,
        }
    }
}

/// `listener.json` lives beside the console's config file.
pub fn settings_path(config: &Path) -> PathBuf {
    config.with_file_name(SETTINGS_FILE)
}

impl Settings {
    pub fn load(path: &Path) -> Result<Self, String> {
        match std::fs::read(path) {
            Ok(raw) => serde_json::from_slice(&raw)
                .map_err(|e| format!("Cannot read {}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("Cannot read {}: {e}", path.display())),
        }
    }
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        crate::headless::write_json(path, self)
    }
    pub fn paired(&self) -> bool {
        self.token_sha256.is_some()
    }
    /// Replace any token with a new one. The returned token is the only copy.
    pub fn rotate(&mut self) -> Result<String, String> {
        let token = generate_token()?;
        self.token_sha256 = Some(hex(&digest(&token)));
        self.token_created_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());
        Ok(token)
    }
    pub fn revoke(&mut self) {
        self.token_sha256 = None;
        self.token_created_unix = None;
    }
}

// Crockford base32: no I, L, O or U, so a token read aloud or retyped from a
// terminal survives. 20 symbols carry 100 random bits.
const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const TOKEN_SYMBOLS: usize = 20;

pub fn generate_token() -> Result<String, String> {
    let mut bytes = [0u8; TOKEN_SYMBOLS];
    getrandom::fill(&mut bytes).map_err(|e| format!("Cannot generate a pairing token: {e}"))?;
    let symbols: Vec<char> = bytes
        .iter()
        .map(|b| ALPHABET[(b & 31) as usize].to_ascii_lowercase() as char)
        .collect();
    let groups: Vec<String> = symbols.chunks(5).map(|c| c.iter().collect()).collect();
    Ok(format!("gary-{}", groups.join("-")))
}

/// Case, spaces and dashes don't matter, so `GARY-ABCDE...` and a token
/// retyped in lowercase hash the same.
pub fn digest(token: &str) -> [u8; 32] {
    let normal: String = token
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-')
        .map(|c| c.to_ascii_uppercase())
        .collect();
    Sha256::digest(normal.as_bytes()).into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut difference = 0u8;
    for (x, y) in a.iter().zip(b) {
        difference |= x ^ y;
    }
    std::hint::black_box(difference) == 0
}

/// Compare digests, not strings, and in constant time, so neither the token's
/// length nor a matching prefix shows up in response timing.
pub fn token_matches(stored_sha256: &str, presented: &str) -> bool {
    let Some(stored) = unhex(stored_sha256) else {
        return false;
    };
    // Hash even an empty or oversized token so every rejection costs the same.
    let presented = if presented.len() > 256 { "" } else { presented };
    constant_time_eq(&stored, &digest(presented)) && !presented.is_empty()
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?.trim();
    let (scheme, token) = value.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

/// Split `/sa3/poll_status/x?y=1` into `("sa3", "/poll_status/x?y=1")`.
pub fn split_route(path_and_query: &str) -> Option<(&str, String)> {
    let trimmed = path_and_query.strip_prefix('/')?;
    let end = trimmed.find(['/', '?']).unwrap_or(trimmed.len());
    let (name, rest) = trimmed.split_at(end);
    if name.is_empty() {
        return None;
    }
    Some((
        name,
        if rest.starts_with('/') {
            rest.to_owned()
        } else {
            format!("/{rest}")
        },
    ))
}

pub fn route(name: &str) -> Option<Service> {
    ROUTES
        .iter()
        .find(|(prefix, _)| prefix.eq_ignore_ascii_case(name))
        .map(|(_, service)| *service)
}

/// Each service's client port, from its headless profile where one exists.
pub fn service_ports(config: &Path) -> HashMap<Service, u16> {
    Service::ALL
        .into_iter()
        .map(|service| {
            let port =
                crate::headless::Config::read(&crate::console::profile_path(config, service))
                    .ok()
                    .filter(|c| c.service == service)
                    .map(|c| c.public_port)
                    .unwrap_or_else(|| service.default_port());
            (service, port)
        })
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub forwarded: u64,
    pub rejected: u64,
    pub last_rejected_from: Option<IpAddr>,
}

pub struct Shared {
    settings: PathBuf,
    ports: HashMap<Service, u16>,
    upstream: IpAddr,
    stats: Mutex<Stats>,
}

impl Shared {
    pub fn new(settings: PathBuf, ports: HashMap<Service, u16>) -> Arc<Self> {
        Arc::new(Self {
            settings,
            ports,
            upstream: IpAddr::V4(Ipv4Addr::LOCALHOST),
            stats: Mutex::new(Stats::default()),
        })
    }
    pub fn stats(&self) -> Stats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

fn problem(status: StatusCode, code: &str, message: &str) -> Response {
    let mut response = (status, Json(json!({"error": code, "message": message}))).into_response();
    if status == StatusCode::UNAUTHORIZED {
        response.headers_mut().insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer realm=\"gary4local\""),
        );
    }
    response
}

const HOP_BY_HOP: [HeaderName; 8] = [
    header::CONNECTION,
    HeaderName::from_static("keep-alive"),
    header::PROXY_AUTHENTICATE,
    header::PROXY_AUTHORIZATION,
    header::TE,
    header::TRAILER,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
];

fn strip_hop_by_hop(headers: &mut HeaderMap) {
    let listed: Vec<HeaderName> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|name| HeaderName::from_bytes(name.trim().as_bytes()).ok())
        .collect();
    for name in HOP_BY_HOP.iter().chain(&listed) {
        headers.remove(name);
    }
}

async fn forward(
    State(shared): State<Arc<Shared>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    // Authenticate before routing, so a client without the token learns
    // nothing about which services exist or run.
    let stored = Settings::load(&shared.settings)
        .ok()
        .and_then(|s| s.token_sha256);
    let presented = bearer(request.headers());
    let authorized = match (&stored, presented) {
        (Some(stored), Some(token)) => token_matches(stored, token),
        _ => false,
    };
    if !authorized {
        if let Ok(mut stats) = shared.stats.lock() {
            stats.rejected += 1;
            stats.last_rejected_from = Some(peer.ip());
        }
        return if presented.is_none() {
            problem(
                StatusCode::UNAUTHORIZED,
                "pairing_token_required",
                "Send the pairing token from gary4local as Authorization: Bearer <token>.",
            )
        } else {
            problem(
                StatusCode::UNAUTHORIZED,
                "pairing_token_rejected",
                "The pairing token doesn't match. Copy the current token from gary4local.",
            )
        };
    }
    let (mut parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or("/");
    let Some((name, rest)) = split_route(path_and_query) else {
        return problem(
            StatusCode::NOT_FOUND,
            "unknown_service",
            "Start the path with a service: /gary, /terry, /jerry, /carey, /foundation, /sa3 or /yuey.",
        );
    };
    let Some(service) = route(name) else {
        return problem(
            StatusCode::NOT_FOUND,
            "unknown_service",
            &format!("No service is called {name}."),
        );
    };
    let Some(&port) = shared.ports.get(&service) else {
        return problem(
            StatusCode::NOT_FOUND,
            "unknown_service",
            "No port for that service.",
        );
    };
    let Ok(uri) = rest.parse::<Uri>() else {
        return problem(
            StatusCode::BAD_REQUEST,
            "bad_path",
            "The request path isn't valid.",
        );
    };
    strip_hop_by_hop(&mut parts.headers);
    parts.headers.remove(header::AUTHORIZATION);
    parts.headers.insert(
        header::HOST,
        HeaderValue::from_str(&format!("127.0.0.1:{port}")).expect("ascii host"),
    );
    if let Ok(value) = HeaderValue::from_str(&peer.ip().to_string()) {
        parts.headers.insert("x-forwarded-for", value);
    }
    parts.uri = uri;
    parts.version = Version::HTTP_11;

    let offline = || {
        problem(
            StatusCode::BAD_GATEWAY,
            "service_offline",
            &format!("{} isn't running on this gary4local machine.", service.id()),
        )
    };
    let Ok(stream) = tokio::net::TcpStream::connect((shared.upstream, port)).await else {
        return offline();
    };
    let _ = stream.set_nodelay(true);
    let Ok((mut sender, connection)) =
        hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(stream)).await
    else {
        return offline();
    };
    // The connection task drives both bodies; it ends when the response body
    // has been read or the client goes away.
    tokio::spawn(async move {
        let _ = connection.await;
    });
    let response = match sender
        .send_request(axum::http::Request::from_parts(parts, body))
        .await
    {
        Ok(response) => response,
        Err(error) => {
            return problem(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                &format!("{} closed the connection: {error}", service.id()),
            )
        }
    };
    if let Ok(mut stats) = shared.stats.lock() {
        stats.forwarded += 1;
    }
    let (mut parts, incoming) = response.into_parts();
    strip_hop_by_hop(&mut parts.headers);
    Response::from_parts(parts, Body::new(incoming))
}

pub fn router(shared: Arc<Shared>) -> Router {
    Router::new().fallback(forward).with_state(shared)
}

pub async fn bind(settings: &Settings) -> Result<tokio::net::TcpListener, String> {
    tokio::net::TcpListener::bind((settings.bind, settings.port))
        .await
        .map_err(|e| {
            format!(
                "Cannot open the LAN listener on {}:{}: {e}",
                settings.bind, settings.port
            )
        })
}

pub async fn serve(listener: tokio::net::TcpListener, shared: Arc<Shared>) -> std::io::Result<()> {
    axum::serve(
        listener,
        router(shared).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
}

/// Best guess at the address other machines use. Connecting a UDP socket
/// sends nothing; it only asks the OS which interface routes outward.
pub fn lan_address(bind: IpAddr) -> Option<IpAddr> {
    if !bind.is_unspecified() {
        return Some(bind);
    }
    let socket = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((Ipv4Addr::new(192, 0, 2, 1), 9)).ok()?;
    let ip = socket.local_addr().ok()?.ip();
    (!ip.is_unspecified() && !ip.is_loopback()).then_some(ip)
}

pub fn address_line(settings: &Settings) -> String {
    match lan_address(settings.bind) {
        Some(ip) => format!("http://{ip}:{}", settings.port),
        None => format!("port {} on this machine's LAN address", settings.port),
    }
}

pub const NO_TLS_NOTICE: &str =
    "No TLS: the token and audio cross your network unencrypted. Use it only on networks you trust.";

/// Lines shown when a token is created. This is the only time it's visible.
pub fn token_notice(token: &str) -> String {
    format!(
        "Pairing token (shown once): {token}\n\
         In gary4juce: gary settings > gary4local host, then paste it into the token field.\n\
         Lost it? Use listen rotate for a new one; the old token stops working."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_random_typable_and_match_only_themselves() {
        let token = generate_token().unwrap();
        let other = generate_token().unwrap();
        assert_ne!(token, other);
        assert_eq!(token.len(), "gary-".len() + 20 + 3);
        assert!(token
            .chars()
            .all(|c| c == '-' || c.is_ascii_digit() || c.is_ascii_lowercase()));
        assert!(!token[5..].contains(['i', 'l', 'o', 'u']));

        let stored = hex(&digest(&token));
        assert!(token_matches(&stored, &token));
        assert!(token_matches(&stored, &token.to_ascii_uppercase()));
        assert!(token_matches(&stored, &token.replace('-', " ")));
        assert!(!token_matches(&stored, &other));
        assert!(!token_matches(&stored, &token[..token.len() - 1]));
        assert!(!token_matches(&stored, ""));
        assert!(!token_matches(&stored, &"a".repeat(10_000)));
        assert!(!token_matches("not hex", &token));
        assert!(!token_matches(&stored[..63], &token));
        // The empty string's digest must never authorize an empty token.
        assert!(!token_matches(&hex(&digest("")), ""));
    }

    #[test]
    fn constant_time_comparison_checks_every_byte() {
        let a = [7u8; 32];
        let mut b = a;
        assert!(constant_time_eq(&a, &b));
        b[31] ^= 1;
        assert!(!constant_time_eq(&a, &b));
        b = a;
        b[0] ^= 0x80;
        assert!(!constant_time_eq(&a, &b));
    }

    #[test]
    fn bearer_scheme_is_required() {
        let mut headers = HeaderMap::new();
        assert_eq!(bearer(&headers), None);
        headers.insert(header::AUTHORIZATION, "Bearer gary-abc".parse().unwrap());
        assert_eq!(bearer(&headers), Some("gary-abc"));
        headers.insert(header::AUTHORIZATION, "bearer  gary-abc ".parse().unwrap());
        assert_eq!(bearer(&headers), Some("gary-abc"));
        headers.insert(header::AUTHORIZATION, "Basic Z2FyeQ==".parse().unwrap());
        assert_eq!(bearer(&headers), None);
        headers.insert(header::AUTHORIZATION, "gary-abc".parse().unwrap());
        assert_eq!(bearer(&headers), None);
    }

    #[test]
    fn routes_strip_only_the_service_prefix() {
        assert_eq!(split_route("/sa3/health"), Some(("sa3", "/health".into())));
        assert_eq!(
            split_route("/sa3/poll_status/a?consume=1"),
            Some(("sa3", "/poll_status/a?consume=1".into()))
        );
        assert_eq!(split_route("/sa3"), Some(("sa3", "/".into())));
        assert_eq!(split_route("/sa3?x=1"), Some(("sa3", "/?x=1".into())));
        assert_eq!(split_route("/"), None);
        assert_eq!(split_route("//health"), None);
        assert_eq!(route("SA3"), Some(Service::Sa3));
        assert_eq!(route("terry"), Some(Service::Melodyflow));
        assert_eq!(route("jerry"), Some(Service::StableAudio));
        assert_eq!(route("stable-audio"), Some(Service::StableAudio));
        assert_eq!(route("ssh"), None);
        for service in Service::ALL {
            assert!(ROUTES.iter().any(|(_, s)| *s == service));
        }
    }

    #[test]
    fn settings_default_off_and_never_store_the_token() {
        let dir = std::env::temp_dir().join(format!("gary-listener-{}", crate::headless::serial()));
        let path = settings_path(&dir.join("sa3.json"));
        let mut settings = Settings::load(&path).unwrap();
        assert_eq!(settings, Settings::default());
        assert!(!settings.enabled);
        assert_eq!(settings.port, 8020);
        let token = settings.rotate().unwrap();
        settings.save(&path).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.to_ascii_lowercase().contains(&token[5..]));
        let loaded = Settings::load(&path).unwrap();
        assert!(token_matches(
            loaded.token_sha256.as_deref().unwrap(),
            &token
        ));
        let mut revoked = loaded;
        revoked.revoke();
        assert!(!revoked.paired());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
