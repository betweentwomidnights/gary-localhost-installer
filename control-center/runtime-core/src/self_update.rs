//! `gary update`: the headless console's own signed update feed.
//!
//! A managed install, made by install-gary.sh, looks like this:
//!
//! ```text
//! <app>/versions/0.3.0/gary      one extracted, verified archive per version
//! <app>/versions/0.3.1/gary
//! <app>/current  -> versions/0.3.1
//! <app>/previous -> versions/0.3.0
//! ```
//!
//! The launcher in ~/.local/bin runs `<app>/current/gary`. An update checks the
//! archive against the feed's SHA-256 and the gary4local updater key, extracts
//! it into a new version folder, runs it once, then replaces `current` with a
//! rename, so a new `gary` sees the old version or the new one, never a mix.
//! Config, runtime roots, models and the LoRA library live outside `<app>`, and
//! nothing here writes to them.
use crate::cli;
use crate::headless::{self, Config};
use crate::service::Service;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{IsTerminal, Read, Write};
use std::path::{Component, Path, PathBuf};

pub const FEED_BASE: &str =
    "https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local";
/// Headless archives are experimental, so like the other Linux builds they
/// follow `preview`.
pub const CHANNEL: &str = "preview";
/// The key that signs every gary4local release, the same one desktop's
/// tauri-plugin-updater trusts. A test keeps the two copies equal.
pub const UPDATER_PUBKEY: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEQwQzIzMzA0NTlCQTQ1RkYKUldUL1JicFpCRFBDMEdLaDlQdkJYMmpLZ09FMHNwMjVVZlYrYVNCL0Z5Rmg5Y2JqOXp4am5xWlMK";
/// A headless archive is two binaries and some text; anything near this is
/// not one, and the cap keeps a bad URL from filling the disk.
const MAX_ARCHIVE_BYTES: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u64, pub u64, pub u64);
impl Version {
    /// Plain MAJOR.MINOR.PATCH, which is all the feed validator lets through.
    pub fn parse(text: &str) -> Result<Self, String> {
        let number = |part: &str| {
            (!part.is_empty()
                && part.len() <= 9
                && part.bytes().all(|b| b.is_ascii_digit())
                && !(part.len() > 1 && part.starts_with('0')))
            .then(|| part.parse().ok())
            .flatten()
        };
        match text.split('.').collect::<Vec<_>>().as_slice() {
            [a, b, c] => match (number(a), number(b), number(c)) {
                (Some(a), Some(b), Some(c)) => Ok(Self(a, b, c)),
                _ => Err(format!("'{text}' isn't a MAJOR.MINOR.PATCH version.")),
            },
            _ => Err(format!("'{text}' isn't a MAJOR.MINOR.PATCH version.")),
        }
    }
    pub fn current() -> Self {
        Self::parse(env!("CARGO_PKG_VERSION")).expect("runtime-core has a plain version")
    }
}
impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.0, self.1, self.2)
    }
}

/// One headless architecture: its feed folder and key, the archive name CI
/// gives it, and the `platform` its BUILD-INFO.json records.
#[derive(Debug, PartialEq, Eq)]
pub struct Platform {
    pub key: &'static str,
    pub folder: &'static str,
    pub archive: &'static str,
    pub build: &'static str,
}
pub const LINUX_X64: Platform = Platform {
    key: "linux-x86_64-headless",
    folder: "headless-linux-x86_64",
    archive: "gary4local-server-linux-x64-experimental.tar.gz",
    build: "linux-x64",
};
pub const LINUX_ARM64: Platform = Platform {
    key: "linux-aarch64-headless",
    folder: "headless-linux-aarch64",
    archive: "gary4local-server-linux-arm64-experimental.tar.gz",
    build: "linux-arm64",
};
/// Headless archives exist for Linux x64 and arm64 only.
pub fn platform() -> Option<&'static Platform> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some(&LINUX_X64)
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some(&LINUX_ARM64)
    } else {
        None
    }
}

/// Where to look. The environment overrides exist for testing a feed before
/// it's published, as desktop's GARY4LOCAL_NATIVE_UPDATER_* do.
#[derive(Clone, Debug)]
pub struct Source {
    pub feed: String,
    pub pubkey: String,
}
impl Source {
    pub fn for_platform(platform: &Platform) -> Self {
        let env = |name| std::env::var(name).ok().filter(|v: &String| !v.trim().is_empty());
        Self {
            feed: env("GARY4LOCAL_HEADLESS_UPDATE_FEED")
                .unwrap_or_else(|| format!("{FEED_BASE}/{}", platform.folder)),
            pubkey: env("GARY4LOCAL_HEADLESS_UPDATER_PUBKEY")
                .unwrap_or_else(|| UPDATER_PUBKEY.into()),
        }
    }
    /// A feed served from this machine may link plain-HTTP archives on this
    /// machine; the signature check still applies.
    fn loopback(&self) -> bool {
        is_loopback_http(&self.feed)
    }
}
fn is_loopback_http(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else {
        return false;
    };
    let authority = rest.split('/').next().unwrap_or_default();
    let host = if authority.starts_with('[') {
        authority.split(']').next().map(|h| format!("{h}]"))
    } else {
        authority.split(':').next().map(str::to_owned)
    };
    matches!(host.as_deref(), Some("127.0.0.1" | "localhost" | "[::1]"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub version: Version,
    pub notes: Vec<String>,
    pub published_at: String,
    pub url: String,
    pub sha256: String,
    pub signature: String,
}

#[derive(Deserialize)]
struct Phase1 {
    channel: String,
    latest_version: String,
    download_url: String,
    sha256: String,
    published_at: String,
    #[serde(default)]
    notes: Vec<String>,
}
#[derive(Deserialize)]
struct Native {
    version: String,
    pub_date: String,
    platforms: BTreeMap<String, NativePlatform>,
}
#[derive(Deserialize)]
struct NativePlatform {
    signature: String,
    url: String,
}

const MID_PUBLISH: &str = "The update feed's two files disagree, which usually means a release is being published right now. Try again in a few minutes.";

/// Read the two files the generator writes, `<channel>.json` and
/// `native-<channel>.json`, and accept them only if they describe the same
/// archive for this platform. These are the validator's checks, made again on
/// the reading side.
pub fn parse_feed(
    phase1: &[u8],
    native: &[u8],
    platform: &Platform,
    allow_loopback_http: bool,
) -> Result<Release, String> {
    let phase: Phase1 = serde_json::from_slice(phase1)
        .map_err(|e| format!("The update feed's {CHANNEL}.json is malformed: {e}"))?;
    let native: Native = serde_json::from_slice(native)
        .map_err(|e| format!("The update feed's native-{CHANNEL}.json is malformed: {e}"))?;
    if phase.channel != CHANNEL {
        return Err(format!(
            "The update feed is for the {} channel, not {CHANNEL}.",
            phase.channel
        ));
    }
    if phase.latest_version != native.version || phase.published_at != native.pub_date {
        return Err(MID_PUBLISH.into());
    }
    let version = Version::parse(&native.version)?;
    let entry = match (native.platforms.len(), native.platforms.get(platform.key)) {
        (1, Some(entry)) => entry,
        _ => {
            return Err(format!(
                "The update feed isn't for {}; it lists {}.",
                platform.key,
                native.platforms.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        }
    };
    if entry.url != phase.download_url {
        return Err(MID_PUBLISH.into());
    }
    let secure = entry.url.starts_with("https://")
        || (allow_loopback_http && is_loopback_http(&entry.url));
    if !secure {
        return Err("The update feed's download link isn't HTTPS.".into());
    }
    if !entry.url.ends_with(&format!("/{}", platform.archive)) {
        return Err(format!(
            "The update feed links {}, not a {} archive.",
            entry.url, platform.archive
        ));
    }
    if phase.sha256.len() != 64
        || !phase
            .sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("The update feed's SHA-256 isn't 64 lowercase hex characters.".into());
    }
    if entry.signature.trim().is_empty() {
        return Err("The update feed has no signature for the archive.".into());
    }
    Ok(Release {
        version,
        notes: phase.notes,
        published_at: phase.published_at,
        url: entry.url.clone(),
        sha256: phase.sha256,
        signature: entry.signature.clone(),
    })
}

/// Tauri stores minisign keys and signatures as base64 of their text form.
fn minisign_text(base64_text: &str, what: &str) -> Result<String, String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_text.trim())
        .map_err(|_| format!("The {what} isn't valid base64."))?;
    String::from_utf8(bytes).map_err(|_| format!("The {what} isn't valid text."))
}

const BAD_SIGNATURE: &str =
    "The download isn't signed by the gary4local updater key. Nothing was installed.";

/// Check a downloaded archive against the release's SHA-256 and its minisign
/// signature, in one read. tauri signer signs prehashed, so legacy signatures
/// are refused.
pub fn verify_archive(path: &Path, release: &Release, pubkey: &str) -> Result<(), String> {
    let key = minisign_verify::PublicKey::decode(&minisign_text(pubkey, "updater public key")?)
        .map_err(|e| format!("The updater public key is invalid: {e}"))?;
    let signature =
        minisign_verify::Signature::decode(&minisign_text(&release.signature, "feed signature")?)
            .map_err(|e| format!("The feed's signature is malformed: {e}"))?;
    let mut verifier = key
        .verify_stream(&signature)
        .map_err(|_| BAD_SIGNATURE.to_string())?;
    let mut sha = Sha256::new();
    let mut file =
        File::open(path).map_err(|e| format!("Can't read {}: {e}", path.display()))?;
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("Can't read {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        verifier.update(&buffer[..read]);
        sha.update(&buffer[..read]);
    }
    if format!("{:x}", sha.finalize()) != release.sha256 {
        return Err(
            "The download doesn't match the feed's SHA-256. Nothing was installed.".into(),
        );
    }
    verifier.finalize().map_err(|_| BAD_SIGNATURE.to_string())
}

/// Unpack into a new folder. Only plain files and folders with relative,
/// non-escaping paths are accepted; the CI archive has nothing else.
pub fn extract(archive: &Path, dest: &Path) -> Result<(), String> {
    let error = |e: std::io::Error| format!("Can't unpack the archive: {e}");
    fs::create_dir(dest).map_err(error)?;
    let file = File::open(archive).map_err(error)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(std::io::BufReader::new(file)));
    for entry in tar.entries().map_err(error)? {
        let mut entry = entry.map_err(error)?;
        let path = entry.path().map_err(error)?.into_owned();
        let mut relative = PathBuf::new();
        for component in path.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(part) => relative.push(part),
                _ => {
                    return Err(format!(
                        "The archive has an unsafe path, {}. Nothing was installed.",
                        path.display()
                    ))
                }
            }
        }
        if relative.as_os_str().is_empty() {
            continue;
        }
        let target = dest.join(&relative);
        match entry.header().entry_type() {
            tar::EntryType::Directory => fs::create_dir_all(&target).map_err(error)?,
            tar::EntryType::Regular => {
                if target.exists() {
                    return Err(format!("The archive lists {} twice.", path.display()));
                }
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent).map_err(error)?;
                }
                entry.set_preserve_permissions(false);
                entry.unpack(&target).map_err(error)?;
            }
            other => {
                return Err(format!(
                    "The archive has an unexpected {other:?} entry, {}. Nothing was installed.",
                    path.display()
                ))
            }
        }
    }
    Ok(())
}

/// The unpacked archive must be the release the feed promised, for this
/// machine, and its `gary` must run.
fn check_package(dir: &Path, release: &Release, platform: &Platform) -> Result<(), String> {
    for name in ["gary", "gary4local-server", "install-gary.sh", "BUILD-INFO.json"] {
        if !dir.join(name).is_file() {
            return Err(format!("The archive has no {name}. Nothing was installed."));
        }
    }
    let info: Value = fs::read(dir.join("BUILD-INFO.json"))
        .ok()
        .and_then(|raw| serde_json::from_slice(&raw).ok())
        .ok_or("The archive's BUILD-INFO.json is unreadable. Nothing was installed.")?;
    let version = release.version.to_string();
    if info["version"] != version.as_str() || info["platform"] != platform.build {
        return Err(format!(
            "The archive says it's {} for {}, but the feed promised {version} for {}. Nothing was installed.",
            info["version"], info["platform"], platform.build
        ));
    }
    runs_as(&dir.join("gary"), release.version)
}

/// Run `gary --version` from a version folder before pointing anything at it.
fn runs_as(gary: &Path, version: Version) -> Result<(), String> {
    let output = std::process::Command::new(gary)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("{} doesn't run on this machine: {e}", gary.display()))?;
    let reported = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || reported != version.to_string() {
        return Err(format!(
            "{} reported '{reported}' instead of {version}.",
            gary.display()
        ));
    }
    Ok(())
}

/// The managed install folder.
#[derive(Clone, Debug)]
pub struct AppDir {
    root: PathBuf,
}
impl AppDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// The managed install an executable runs from: `<app>/versions/<v>/gary`
    /// with a `current` link beside `versions`. Launchers resolve to the real
    /// file, so this is the version folder even when run through `current`.
    pub fn of_executable(exe: &Path) -> Option<Self> {
        let exe = fs::canonicalize(exe).ok()?;
        let versions = exe.parent()?.parent()?;
        if versions.file_name()? != "versions" {
            return None;
        }
        let app = Self::new(versions.parent()?);
        app.linked("current").is_some().then_some(app)
    }
    /// The version a link names. Links are always `versions/<version>`.
    pub fn linked(&self, name: &str) -> Option<Version> {
        let target = fs::read_link(self.root.join(name)).ok()?;
        let mut parts = target.components();
        match (parts.next(), parts.next(), parts.next()) {
            (Some(Component::Normal(dir)), Some(Component::Normal(version)), None)
                if dir == "versions" =>
            {
                Version::parse(version.to_str()?).ok()
            }
            _ => None,
        }
    }
    pub fn version_dir(&self, version: Version) -> PathBuf {
        self.root.join("versions").join(version.to_string())
    }
    /// One update or rollback at a time; the OS drops the lock with the file.
    fn lock(&self) -> Result<File, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.root.join("update.lock"))
            .map_err(|e| format!("Can't open the update lock: {e}"))?;
        file.try_lock()
            .map_err(|_| "Another gary update or rollback is running.".to_string())?;
        Ok(file)
    }
}

/// Removes a half-made version folder unless the install got far enough to keep it.
struct Scratch(Option<PathBuf>);
impl Drop for Scratch {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

/// Install a downloaded archive as a new version and make it current.
/// `idle` is checked after taking the update lock, so nothing can start
/// between the check and the switch through another `gary update`. Returns
/// the version that was current before, which stays installed for rollback.
pub fn install(
    app: &AppDir,
    platform: &Platform,
    release: &Release,
    archive: &Path,
    pubkey: &str,
    idle: impl Fn() -> Result<(), String>,
) -> Result<Option<Version>, String> {
    let _lock = app.lock()?;
    idle()?;
    let current = app.linked("current");
    if current == Some(release.version) {
        return Err(format!("Gary {} is already installed.", release.version));
    }
    verify_archive(archive, release, pubkey)?;
    let versions = app.root.join("versions");
    let staging = versions.join(format!(".staging-{}", headless::serial()));
    let mut scratch = Scratch(Some(staging.clone()));
    extract(archive, &staging)?;
    check_package(&staging, release, platform)?;
    let dest = app.version_dir(release.version);
    if dest.exists() {
        // Left by a rolled-back or interrupted update. The fresh, verified copy wins.
        let discard = versions.join(format!(".discard-{}", headless::serial()));
        fs::rename(&dest, &discard).map_err(|e| format!("Can't replace {}: {e}", dest.display()))?;
        let _ = fs::remove_dir_all(&discard);
    }
    fs::rename(&staging, &dest).map_err(|e| format!("Can't install {}: {e}", dest.display()))?;
    scratch.0 = None;
    if let Some(current) = current {
        point(app, "previous", current)?;
    }
    point(app, "current", release.version)?;
    prune(app);
    Ok(current)
}

/// Make `previous` current again, and current the new `previous`.
pub fn rollback(app: &AppDir, idle: impl Fn() -> Result<(), String>) -> Result<(Version, Version), String> {
    let _lock = app.lock()?;
    idle()?;
    let current = app
        .linked("current")
        .ok_or("This install has no current version to roll back from.")?;
    let previous = app
        .linked("previous")
        .filter(|previous| *previous != current)
        .ok_or("There's no previous version to roll back to.")?;
    runs_as(&app.version_dir(previous).join("gary"), previous)?;
    point(app, "previous", current)?;
    point(app, "current", previous)?;
    Ok((current, previous))
}

/// Replace a link by renaming a new one over it, which POSIX makes atomic.
#[cfg(unix)]
fn point(app: &AppDir, name: &str, version: Version) -> Result<(), String> {
    let link = app.root.join(name);
    let temporary = app.root.join(format!(".{name}-{}", headless::serial()));
    let target = Path::new("versions").join(version.to_string());
    std::os::unix::fs::symlink(&target, &temporary)
        .map_err(|e| format!("Can't point {name} at {version}: {e}"))?;
    if let Err(e) = fs::rename(&temporary, &link) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("Can't point {name} at {version}: {e}"));
    }
    // Persist the rename before reporting success.
    if let Ok(dir) = File::open(&app.root) {
        let _ = dir.sync_all();
    }
    Ok(())
}
#[cfg(not(unix))]
fn point(_: &AppDir, _: &str, _: Version) -> Result<(), String> {
    Err("gary update installs the Linux headless archives.".into())
}

/// Keep the current and previous versions; remove older ones and leftovers.
fn prune(app: &AppDir) {
    let keep: Vec<String> = ["current", "previous"]
        .iter()
        .filter_map(|link| app.linked(link))
        .map(|v| v.to_string())
        .collect();
    let Ok(entries) = fs::read_dir(app.root.join("versions")) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !keep.contains(&name) {
            let _ = fs::remove_dir_all(entry.path());
        }
    }
    let _ = fs::remove_dir_all(app.root.join("downloads"));
}

/// Services whose runtime root is held by a serving session, a model or
/// runtime install, or a training job: all take the same `manager.lock`.
pub fn running_work() -> Result<Vec<&'static str>, String> {
    let base = crate::console::default_config()?;
    let mut busy = Vec::new();
    for service in Service::ALL {
        let path = crate::console::profile_path(&base, service);
        if !path.is_file() {
            continue;
        }
        // A profile that doesn't parse has no runtime root anything could run from.
        let Ok(config) = Config::read(&path) else {
            continue;
        };
        if headless::active(&config)? {
            busy.push(service.id());
        }
    }
    Ok(busy)
}
pub fn ensure_idle() -> Result<(), String> {
    let busy = running_work()?;
    if busy.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Gary can't change versions while {} {} running a service, an install or training. Stop {}, and let training finish or cancel it, then try again. Updates and rollbacks never interrupt running work.",
        busy.join(", "),
        if busy.len() == 1 { "is" } else { "are" },
        if busy.len() == 1 { "it" } else { "them" }
    ))
}

fn client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(concat!("gary4local-headless/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())
}

/// Fetch and check the feed for this platform.
pub async fn fetch(source: &Source, platform: &Platform) -> Result<Release, String> {
    let client = client()?;
    let base = source.feed.trim_end_matches('/');
    let mut files = Vec::new();
    for name in [format!("{CHANNEL}.json"), format!("native-{CHANNEL}.json")] {
        let url = format!("{base}/{name}");
        let response = client
            .get(&url)
            .timeout(std::time::Duration::from_secs(20))
            .send()
            .await
            .map_err(|e| format!("Can't reach the update feed: {e}"))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(format!(
                "No headless update feed is published at {base} yet."
            ));
        }
        let response = response
            .error_for_status()
            .map_err(|e| format!("The update feed returned an error: {e}"))?;
        files.push(
            response
                .bytes()
                .await
                .map_err(|e| format!("Can't read the update feed: {e}"))?,
        );
    }
    parse_feed(&files[0], &files[1], platform, source.loopback())
}

/// Download the release's archive to `dest`. It's verified when installed.
pub async fn download(release: &Release, dest: &Path, progress: impl Fn(String)) -> Result<(), String> {
    let client = client()?;
    let mut response = client
        .get(&release.url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("Can't download {}: {e}", release.url))?;
    if response.content_length().is_some_and(|n| n > MAX_ARCHIVE_BYTES) {
        return Err("The download is far larger than a headless archive. Nothing was installed.".into());
    }
    if let Some(size) = response.content_length() {
        progress(format!(
            "Downloading Gary {} ({:.1} MiB)...",
            release.version,
            size as f64 / f64::from(1 << 20)
        ));
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = File::create(dest).map_err(|e| format!("Can't write {}: {e}", dest.display()))?;
    let mut written = 0u64;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("The download stopped: {e}"))?
    {
        written += chunk.len() as u64;
        if written > MAX_ARCHIVE_BYTES {
            return Err("The download is far larger than a headless archive. Nothing was installed.".into());
        }
        file.write_all(&chunk)
            .map_err(|e| format!("Can't write {}: {e}", dest.display()))?;
    }
    file.sync_all().map_err(|e| e.to_string())
}

/// Download, verify and install, then remove the download.
pub async fn download_and_install(
    app: &AppDir,
    platform: &'static Platform,
    release: &Release,
    source: &Source,
    progress: impl Fn(String),
) -> Result<Option<Version>, String> {
    let archive = app
        .root
        .join("downloads")
        .join(format!("{}-{}", headless::serial(), platform.archive));
    let result = async {
        download(release, &archive, &progress).await?;
        progress("Checking the signature and installing...".into());
        let (app, release, archive, pubkey) = (
            app.clone(),
            release.clone(),
            archive.clone(),
            source.pubkey.clone(),
        );
        tokio::task::spawn_blocking(move || {
            install(&app, platform, &release, &archive, &pubkey, ensure_idle)
        })
        .await
        .map_err(|e| e.to_string())?
    }
    .await;
    let _ = fs::remove_file(&archive);
    result
}

/// Version, date and notes, as both the command and the console show them.
pub fn summary(release: &Release, current: Version) -> String {
    let date = release.published_at.get(..10).unwrap_or(&release.published_at);
    let mut text = format!(
        "Gary {} is available (you have {current}), published {date}.",
        release.version
    );
    for note in &release.notes {
        text.push_str("\n  - ");
        text.push_str(note);
    }
    text
}

pub fn unmanaged_message() -> String {
    let location = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.display().to_string()))
        .unwrap_or_else(|| "this folder".into());
    format!(
        "This gary runs from {location}, which isn't a managed install, so it can't update itself. Extract the new archive and run its install-gary.sh --replace once; after that, gary update works."
    )
}

/// The console's one-line notice: a newer version, if the feed answers quickly.
pub fn notice_enabled() -> bool {
    platform().is_some()
        && std::env::var("GARY4LOCAL_HEADLESS_UPDATE_CHECK").map_or(true, |v| v.trim() != "0")
}
pub async fn newer_version() -> Option<Version> {
    let platform = platform()?;
    let source = Source::for_platform(platform);
    let release = tokio::time::timeout(std::time::Duration::from_secs(8), fetch(&source, platform))
        .await
        .ok()?
        .ok()?;
    (release.version > Version::current()).then_some(release.version)
}

pub const HELP: &str = "gary update — check for a new Gary and install it after you confirm\n\n\
  gary update              Show the new version and its notes, then ask before installing\n\
  gary update --check      Only report; never installs (add --json for scripts)\n\
  gary update --yes        Install without asking\n\
  gary update --rollback   Switch back to the version before the last update\n\n\
  Updates install into a new version folder and switch over in one step. They\n\
  refuse while a service, an install or training is running, and never change\n\
  your config, runtimes, models or LoRA library.";

struct Out {
    json: bool,
}
impl Out {
    fn say(&self, text: impl fmt::Display) {
        if !self.json {
            println!("{text}");
        }
    }
}
fn failure(code: &'static str, message: impl Into<String>, hint: impl Into<String>) -> cli::Error {
    cli::Error {
        code,
        message: message.into(),
        hint: hint.into(),
        exit_code: 1,
    }
}

/// `gary update [--check | --yes | --rollback] [--json]`
pub async fn run(args: &[OsString]) -> i32 {
    let (mut check, mut yes, mut rollback, mut json) = (false, false, false, false);
    for arg in args {
        match arg.to_str() {
            Some("--check") => check = true,
            Some("--yes" | "-y") => yes = true,
            Some("--rollback") => rollback = true,
            Some("--json") => json = true,
            Some("--help" | "-h") => {
                println!("{HELP}");
                return 0;
            }
            _ => {
                let error = cli::Error {
                    code: "invalid_arguments",
                    message: format!("Unknown update option: {}", arg.to_string_lossy()),
                    hint: "Run gary update --help.".into(),
                    exit_code: 2,
                };
                return report(json, Err(error));
            }
        }
    }
    if check && (yes || rollback) {
        let error = cli::Error {
            code: "invalid_arguments",
            message: "--check only reports; it can't be combined with --yes or --rollback.".into(),
            hint: "Run gary update --help.".into(),
            exit_code: 2,
        };
        return report(json, Err(error));
    }
    let out = Out { json };
    let result = if rollback {
        roll_back(&out, yes)
    } else {
        update(&out, check, yes).await
    };
    report(json, result)
}

fn report(json: bool, result: Result<Value, cli::Error>) -> i32 {
    let (data, error, code) = match result {
        Ok(data) => (data, None, 0),
        Err(error) => {
            let code = error.exit_code;
            (Value::Null, Some(error), code)
        }
    };
    if json {
        println!(
            "{}",
            cli::envelope("update", "result", None, data, error.as_ref(), code)
        );
    } else if let Some(error) = error {
        eprintln!("{}\n{}", error.message, error.hint);
    }
    code
}

fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}
fn confirm(question: &str) -> bool {
    print!("{question} [y/N] ");
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).is_ok()
        && matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

async fn update(out: &Out, check: bool, yes: bool) -> Result<Value, cli::Error> {
    let platform = platform().ok_or_else(|| {
        failure(
            "unsupported_platform",
            "gary update installs the Linux x64 and arm64 headless archives.",
            "On other systems, build or download Gary by hand.",
        )
    })?;
    let source = Source::for_platform(platform);
    let release = fetch(&source, platform)
        .await
        .map_err(|m| failure("update_check_failed", m, "Check the network, then try again."))?;
    let current = Version::current();
    let available = release.version > current;
    let mut data = json!({"current":current.to_string(),"latest":release.version.to_string(),
        "update_available":available,"published_at":release.published_at,"notes":release.notes,
        "url":release.url,"installed":false});
    if !available {
        out.say(format!("Gary {current} is up to date."));
        return Ok(data);
    }
    out.say(summary(&release, current));
    if check {
        out.say("Run gary update to install it.");
        return Ok(data);
    }
    let exe = std::env::current_exe().map_err(|e| failure("unmanaged_install", e.to_string(), ""))?;
    let app = AppDir::of_executable(&exe).ok_or_else(|| {
        failure(
            "unmanaged_install",
            unmanaged_message(),
            "See the updating section of the headless README.",
        )
    })?;
    ensure_idle().map_err(|m| failure("busy", m, "Use gary status to see what's running."))?;
    if !yes {
        if out.json || !interactive() {
            return Err(failure(
                "confirmation_required",
                "Nothing was installed.",
                "Run gary update --yes to install it.",
            ));
        }
        if !confirm(&format!("Install Gary {}?", release.version)) {
            out.say("Nothing was installed.");
            return Ok(data);
        }
    }
    let previous = download_and_install(&app, platform, &release, &source, |line| out.say(line))
        .await
        .map_err(|m| failure("update_failed", m, "Your current version is unchanged."))?;
    data["installed"] = json!(true);
    data["previous"] = json!(previous.map(|v| v.to_string()));
    out.say(format!("Installed Gary {}. Run gary again to use it.", release.version));
    if let Some(previous) = previous {
        out.say(format!(
            "Gary {previous} is kept; gary update --rollback switches back to it."
        ));
    }
    Ok(data)
}

fn roll_back(out: &Out, yes: bool) -> Result<Value, cli::Error> {
    let exe = std::env::current_exe().map_err(|e| failure("unmanaged_install", e.to_string(), ""))?;
    let app = AppDir::of_executable(&exe).ok_or_else(|| {
        failure(
            "unmanaged_install",
            unmanaged_message(),
            "See the updating section of the headless README.",
        )
    })?;
    let (current, previous) = match (app.linked("current"), app.linked("previous")) {
        (Some(current), Some(previous)) if current != previous => (current, previous),
        _ => {
            return Err(failure(
                "no_previous_version",
                "There's no previous version to roll back to.",
                "",
            ))
        }
    };
    ensure_idle().map_err(|m| failure("busy", m, "Use gary status to see what's running."))?;
    if !yes {
        if out.json || !interactive() {
            return Err(failure(
                "confirmation_required",
                "Nothing was changed.",
                "Run gary update --rollback --yes to switch back.",
            ));
        }
        if !confirm(&format!("Switch back from Gary {current} to {previous}?")) {
            out.say("Nothing was changed.");
            return Ok(json!({"rolled_back":false}));
        }
    }
    let (from, to) = rollback(&app, ensure_idle)
        .map_err(|m| failure("rollback_failed", m, "Your current version is unchanged."))?;
    out.say(format!(
        "Switched back to Gary {to}. Gary {from} is kept as the previous version; gary update reinstalls the newest."
    ));
    Ok(json!({"rolled_back":true,"from":from.to_string(),"to":to.to_string()}))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIG: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IHNpZ25hdHVyZSBmcm9tIHRhdXJpIHNlY3JldCBrZXkKUlVUL1JicFpCRFBDMEphd1VsdzhWbHJsRHFwQ05LS1J2c2ZHZDNYNGE4MUpiUHo3MGVUcmRrWlRETjd5UDlFVG54ZHlZeENJN2RtaHdsNktKenpIZ0hld3dWREdCTStFVEE0PQp0cnVzdGVkIGNvbW1lbnQ6IHRpbWVzdGFtcDoxNzk1NTMxNTM4CWZpbGU6Z2FyeTRsb2NhbF8wLjcuNl9hcm02NC5kZWIKaTJFTWJlbVZXdTNweVhSa1hqdXJVd2JLcHNtM2JPRVV5SGw1L0h0cFBsbnlJckl2alNaMk5TRmdMZ2hBWEJXWFRVOVEwTlFWdWtlaG9PL0grQWJSQWc9PQo=";
    const URL: &str = "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.7.7/gary4local-server-linux-arm64-experimental.tar.gz";
    const SHA: &str = "53dfb1e08a967c6f2340f5319c04a06e81e9500b1b6a7fb8e1cdc882a5215cc3";

    /// The generator's output, in the generator's own formatting.
    fn feed(version: &str, key: &str, url: &str) -> (String, String) {
        let phase = format!(
            "{{\n    \"channel\":  \"preview\",\n    \"latest_version\":  \"{version}\",\n    \"download_url\":  \"{url}\",\n    \"sha256\":  \"{SHA}\",\n    \"published_at\":  \"2026-11-24T14:57:45.0000000+00:00\",\n    \"notes\":  [\n                  \"Headless updates itself.\",\n                  \"Compatible with gary4juce v5.0.7.\"\n              ]\n}}"
        );
        let native = format!(
            "{{\n    \"version\":  \"{version}\",\n    \"notes\":  \"Headless updates itself.\\nCompatible with gary4juce v5.0.7.\",\n    \"pub_date\":  \"2026-11-24T14:57:45.0000000+00:00\",\n    \"platforms\":  {{\n                      \"{key}\":  {{\n                                                \"signature\":  \"{SIG}\",\n                                                \"url\":  \"{url}\"\n                                            }}\n                  }}\n}}"
        );
        (phase, native)
    }

    #[test]
    fn versions_are_plain_and_ordered_numerically() {
        assert_eq!(Version::parse("0.3.10"), Ok(Version(0, 3, 10)));
        assert!(Version::parse("0.3.10").unwrap() > Version::parse("0.3.9").unwrap());
        for bad in ["0.3", "0.3.1.2", "0.3.1-beta", "v0.3.1", "0.03.1", "", "0..1"] {
            assert!(Version::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(Version::current().to_string(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn a_generated_feed_parses_into_one_release() {
        let (phase, native) = feed("0.3.1", LINUX_ARM64.key, URL);
        let release = parse_feed(phase.as_bytes(), native.as_bytes(), &LINUX_ARM64, false).unwrap();
        assert_eq!(release.version, Version(0, 3, 1));
        assert_eq!(release.url, URL);
        assert_eq!(release.sha256, SHA);
        assert_eq!(release.notes.len(), 2);
        assert!(summary(&release, Version(0, 3, 0))
            .starts_with("Gary 0.3.1 is available (you have 0.3.0), published 2026-11-24."));
    }

    #[test]
    fn feeds_that_disagree_or_target_something_else_are_refused() {
        let parse = |phase: &str, native: &str| {
            parse_feed(phase.as_bytes(), native.as_bytes(), &LINUX_ARM64, false).unwrap_err()
        };
        let (phase, native) = feed("0.3.1", LINUX_ARM64.key, URL);
        let (_, newer) = feed("0.3.2", LINUX_ARM64.key, URL);
        assert!(parse(&phase, &newer).contains("disagree"));
        let (_, x64) = feed("0.3.1", LINUX_X64.key, URL);
        assert!(parse(&phase, &x64).contains("isn't for linux-aarch64-headless"));
        let (_, deb) = feed("0.3.1", "linux-aarch64-deb", URL);
        assert!(parse(&phase, &deb).contains("isn't for"));
        let both = native.replace(
            "\"platforms\":  {",
            &format!("\"platforms\":  {{ \"{}\": {{\"signature\":\"{SIG}\",\"url\":\"{URL}\"}},", LINUX_X64.key),
        );
        assert!(parse(&phase, &both).contains("isn't for"));
        let http = URL.replace("https://", "http://");
        let (p, n) = feed("0.3.1", LINUX_ARM64.key, &http);
        assert!(parse(&p, &n).contains("isn't HTTPS"));
        let x64_archive = URL.replace("arm64", "x64");
        let (p, n) = feed("0.3.1", LINUX_ARM64.key, &x64_archive);
        assert!(parse(&p, &n).contains("not a gary4local-server-linux-arm64"));
        let (p, n) = feed("0.3.1-beta", LINUX_ARM64.key, URL);
        assert!(parse(&p, &n).contains("MAJOR.MINOR.PATCH"));
        assert!(parse(&phase.replace(SHA, &SHA.to_uppercase()), &native).contains("SHA-256"));
        assert!(parse(&phase.replace("preview", "stable"), &native).contains("stable channel"));
        assert!(parse(&phase, &native.replace("14:57:45", "14:57:46")).contains("disagree"));
        assert!(parse(&phase, &native.replace(SIG, " ")).contains("no signature"));
        assert!(parse("not json", &native).contains("malformed"));
    }

    #[test]
    fn loopback_feeds_may_link_plain_http_archives_on_this_machine_only() {
        let local = "http://127.0.0.1:9/v/gary4local-server-linux-arm64-experimental.tar.gz";
        let (p, n) = feed("0.3.1", LINUX_ARM64.key, local);
        assert!(parse_feed(p.as_bytes(), n.as_bytes(), &LINUX_ARM64, true).is_ok());
        assert!(parse_feed(p.as_bytes(), n.as_bytes(), &LINUX_ARM64, false).is_err());
        let remote = "http://example.com/gary4local-server-linux-arm64-experimental.tar.gz";
        let (p, n) = feed("0.3.1", LINUX_ARM64.key, remote);
        assert!(parse_feed(p.as_bytes(), n.as_bytes(), &LINUX_ARM64, true).is_err());
        assert!(is_loopback_http("http://localhost:8080/x"));
        assert!(is_loopback_http("http://[::1]:8080/x"));
        assert!(!is_loopback_http("http://127.0.0.1.example.com/x"));
        assert!(!is_loopback_http("https://127.0.0.1/x"));
    }

    #[test]
    fn the_built_in_key_is_desktops_updater_key() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src-tauri");
        let config: Value = serde_json::from_str(
            &fs::read_to_string(root.join("tauri.updater.conf.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config["plugins"]["updater"]["pubkey"], UPDATER_PUBKEY);
        let update_rs = fs::read_to_string(root.join("src/update.rs")).unwrap();
        assert!(update_rs.contains(UPDATER_PUBKEY));
        minisign_verify::PublicKey::decode(&minisign_text(UPDATER_PUBKEY, "key").unwrap()).unwrap();
        // A real published signature decodes, and is from that key.
        let signature =
            minisign_verify::Signature::decode(&minisign_text(SIG, "signature").unwrap()).unwrap();
        let key =
            minisign_verify::PublicKey::decode(&minisign_text(UPDATER_PUBKEY, "key").unwrap()).unwrap();
        assert!(key.verify_stream(&signature).is_ok());
    }

    #[test]
    fn platforms_name_the_archives_ci_builds() {
        let workflow = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/headless.yml"),
        )
        .unwrap();
        for platform in [&LINUX_X64, &LINUX_ARM64] {
            assert!(workflow.contains(&format!("platform: {}", platform.build)));
            assert_eq!(
                platform.archive,
                format!("gary4local-server-{}-experimental.tar.gz", platform.build)
            );
        }
        assert!(workflow.contains("gary4local-server-$PLATFORM-experimental.tar.gz"));
    }

    pub(super) mod fixture {
        use super::super::*;
        use std::io::Cursor;

        pub struct Dir(pub PathBuf);
        impl Dir {
            pub fn new(name: &str) -> Self {
                let path = std::env::temp_dir().join(format!(
                    "gary-update-{name}-{}-{}",
                    std::process::id(),
                    headless::serial()
                ));
                fs::create_dir_all(&path).unwrap();
                Self(path)
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }

        pub struct Signer {
            pk: minisign::PublicKey,
            sk: minisign::SecretKey,
        }
        impl Signer {
            pub fn new() -> Self {
                let pair = minisign::KeyPair::generate_unencrypted_keypair().unwrap();
                Self { pk: pair.pk, sk: pair.sk }
            }
            /// Base64 of the key's text form, as tauri stores it.
            pub fn pubkey(&self) -> String {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .encode(self.pk.to_box().unwrap().to_string())
            }
            /// What `tauri signer sign` writes to a .sig file.
            pub fn sign(&self, data: &[u8]) -> String {
                use base64::Engine;
                let signature =
                    minisign::sign(Some(&self.pk), &self.sk, Cursor::new(data), None, None).unwrap();
                base64::engine::general_purpose::STANDARD.encode(signature.to_string())
            }
        }

        /// An archive shaped like headless.yml's: `./`-prefixed entries.
        pub fn archive(version: &str, build: &str, reports: &str) -> Vec<u8> {
            let info = format!(
                "{{\"service\":\"gary4local-server\",\"version\":\"{version}\",\"commit\":\"abc\",\"platform\":\"{build}\",\"experimental\":true}}\n"
            );
            let gary = format!("#!/bin/sh\nprintf '%s\\n' '{reports}'\n");
            let files: [(&str, &[u8], u32); 5] = [
                ("./gary", gary.as_bytes(), 0o755),
                ("./gary4local-server", b"#!/bin/sh\n", 0o755),
                ("./install-gary.sh", b"#!/bin/sh\n", 0o644),
                ("./README.md", b"readme\n", 0o644),
                ("./BUILD-INFO.json", info.as_bytes(), 0o644),
            ];
            raw_archive(&files, &[])
        }
        pub fn raw_archive(files: &[(&str, &[u8], u32)], links: &[(&str, &str)]) -> Vec<u8> {
            let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
                Vec::new(),
                flate2::Compression::fast(),
            ));
            let mut root = tar::Header::new_gnu();
            root.set_entry_type(tar::EntryType::Directory);
            root.as_old_mut().name[..2].copy_from_slice(b"./");
            root.set_mode(0o755);
            root.set_size(0);
            root.set_cksum();
            builder.append(&root, std::io::empty()).unwrap();
            for (path, data, mode) in files {
                let mut header = tar::Header::new_gnu();
                // Write the name directly so tests can build unsafe paths.
                header.as_old_mut().name[..path.len()].copy_from_slice(path.as_bytes());
                header.set_entry_type(tar::EntryType::Regular);
                header.set_mode(*mode);
                header.set_size(data.len() as u64);
                header.set_cksum();
                builder.append(&header, *data).unwrap();
            }
            for (path, target) in links {
                let mut header = tar::Header::new_gnu();
                header.set_entry_type(tar::EntryType::Symlink);
                header.set_path(path).unwrap();
                header.set_link_name(target).unwrap();
                header.set_size(0);
                header.set_cksum();
                builder.append(&header, std::io::empty()).unwrap();
            }
            builder.into_inner().unwrap().finish().unwrap()
        }

        pub fn release(version: Version, data: &[u8], signer: &Signer) -> Release {
            Release {
                version,
                notes: vec![],
                published_at: "2026-11-24T00:00:00Z".into(),
                url: format!("https://example.com/{}", LINUX_ARM64.archive),
                sha256: format!("{:x}", Sha256::digest(data)),
                signature: signer.sign(data),
            }
        }
    }
    use fixture::*;

    #[test]
    fn only_the_signed_bytes_with_the_feeds_hash_verify() {
        let dir = Dir::new("verify");
        let signer = Signer::new();
        let data = archive("0.3.1", "linux-arm64", "0.3.1");
        let path = dir.0.join("archive.tar.gz");
        fs::write(&path, &data).unwrap();
        let release = release(Version(0, 3, 1), &data, &signer);
        verify_archive(&path, &release, &signer.pubkey()).unwrap();

        // A byte changed after signing fails the hash.
        let mut tampered = data.clone();
        tampered[20] ^= 1;
        fs::write(&path, &tampered).unwrap();
        assert!(verify_archive(&path, &release, &signer.pubkey())
            .unwrap_err()
            .contains("SHA-256"));
        // A feed that matches tampered bytes still needs the key's signature.
        let rehashed = Release {
            sha256: format!("{:x}", Sha256::digest(&tampered)),
            ..release.clone()
        };
        assert_eq!(
            verify_archive(&path, &rehashed, &signer.pubkey()).unwrap_err(),
            BAD_SIGNATURE
        );
        // So does one signed by any other key, including the real key's absence.
        fs::write(&path, &data).unwrap();
        let other = Signer::new();
        let forged = Release {
            signature: other.sign(&data),
            ..release.clone()
        };
        assert_eq!(
            verify_archive(&path, &forged, &signer.pubkey()).unwrap_err(),
            BAD_SIGNATURE
        );
        assert_eq!(
            verify_archive(&path, &release, UPDATER_PUBKEY).unwrap_err(),
            BAD_SIGNATURE
        );
        assert!(verify_archive(&path, &Release { signature: "%%".into(), ..release }, UPDATER_PUBKEY)
            .unwrap_err()
            .contains("base64"));
    }

    #[test]
    fn extraction_refuses_paths_and_entries_that_could_escape() {
        let dir = Dir::new("extract");
        let path = dir.0.join("a.tar.gz");
        let good = archive("0.3.1", "linux-arm64", "0.3.1");
        fs::write(&path, good).unwrap();
        extract(&path, &dir.0.join("good")).unwrap();
        assert!(dir.0.join("good/gary").is_file());
        assert!(dir.0.join("good/BUILD-INFO.json").is_file());

        let cases: [(&str, Vec<u8>, &str); 4] = [
            ("parent", raw_archive(&[("../escape", b"x", 0o644)], &[]), "unsafe path"),
            ("absolute", raw_archive(&[("/tmp/escape", b"x", 0o644)], &[]), "unsafe path"),
            ("symlink", raw_archive(&[], &[("gary", "/bin/sh")]), "unexpected"),
            (
                "twice",
                raw_archive(&[("./gary", b"a", 0o755), ("gary", b"b", 0o755)], &[]),
                "twice",
            ),
        ];
        for (name, data, expected) in cases {
            fs::write(&path, data).unwrap();
            let error = extract(&path, &dir.0.join(name)).unwrap_err();
            assert!(error.contains(expected), "{name}: {error}");
        }
        assert!(!dir.0.join("escape").exists());
    }
}

#[cfg(all(test, unix))]
mod unix_tests {
    use super::tests::fixture::*;
    use super::*;

    /// A managed install of 0.3.0, as install-gary.sh leaves it.
    fn installed(dir: &Dir) -> AppDir {
        let app = AppDir::new(dir.0.join("app"));
        let data = archive("0.3.0", "linux-arm64", "0.3.0");
        let path = dir.0.join("first.tar.gz");
        fs::write(&path, data).unwrap();
        fs::create_dir_all(app.root().join("versions")).unwrap();
        extract(&path, &app.version_dir(Version(0, 3, 0))).unwrap();
        point(&app, "current", Version(0, 3, 0)).unwrap();
        app
    }
    fn stage(dir: &Dir, version: &str, build: &str, reports: &str, signer: &Signer) -> (PathBuf, Release) {
        let data = archive(version, build, reports);
        let path = dir.0.join(format!("{version}-{build}-{reports}.tar.gz"));
        fs::write(&path, &data).unwrap();
        (path, release(Version::parse(version).unwrap(), &data, signer))
    }
    fn idle() -> Result<(), String> {
        Ok(())
    }
    fn versions(app: &AppDir) -> Vec<String> {
        let mut names: Vec<_> = fs::read_dir(app.root().join("versions"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn an_update_switches_current_keeps_previous_and_prunes_older() {
        let dir = Dir::new("switch");
        let signer = Signer::new();
        let app = installed(&dir);
        let (path, release) = stage(&dir, "0.3.1", "linux-arm64", "0.3.1", &signer);
        let previous = install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle).unwrap();
        assert_eq!(previous, Some(Version(0, 3, 0)));
        assert_eq!(app.linked("current"), Some(Version(0, 3, 1)));
        assert_eq!(app.linked("previous"), Some(Version(0, 3, 0)));
        // The link is relative, so the whole folder can move.
        assert_eq!(
            fs::read_link(app.root().join("current")).unwrap(),
            Path::new("versions/0.3.1")
        );
        let ran = std::process::Command::new(app.root().join("current/gary")).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&ran.stdout).trim(), "0.3.1");
        // The running executable finds the app from its real path.
        let found = AppDir::of_executable(&app.root().join("current/gary")).unwrap();
        assert_eq!(fs::canonicalize(found.root()).unwrap(), fs::canonicalize(app.root()).unwrap());

        let (path, release) = stage(&dir, "0.3.2", "linux-arm64", "0.3.2", &signer);
        install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle).unwrap();
        assert_eq!(versions(&app), ["0.3.1", "0.3.2"]);
        assert!(install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle)
            .unwrap_err()
            .contains("already installed"));
        // No temporary links or folders are left behind.
        let leftovers: Vec<_> = fs::read_dir(app.root())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn a_refused_update_leaves_the_install_exactly_as_it_was() {
        let dir = Dir::new("refuse");
        let signer = Signer::new();
        let app = installed(&dir);
        let unchanged = |app: &AppDir| {
            assert_eq!(app.linked("current"), Some(Version(0, 3, 0)));
            assert_eq!(app.linked("previous"), None);
            assert_eq!(versions(app), ["0.3.0"]);
        };
        let (path, release) = stage(&dir, "0.3.1", "linux-arm64", "0.3.1", &signer);
        let busy = || Err::<(), _>("sa3 is running".to_string());
        assert_eq!(
            install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), busy).unwrap_err(),
            "sa3 is running"
        );
        unchanged(&app);
        assert_eq!(
            install(&app, &LINUX_ARM64, &release, &path, &Signer::new().pubkey(), idle).unwrap_err(),
            BAD_SIGNATURE
        );
        unchanged(&app);
        // Signed and hashed, but not what the feed promised.
        let (path, release) = stage(&dir, "0.3.1", "linux-x64", "0.3.1", &signer);
        assert!(install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle)
            .unwrap_err()
            .contains("for \"linux-x64\""));
        unchanged(&app);
        let (path, mut release) = stage(&dir, "0.3.1", "linux-arm64", "0.3.1", &signer);
        release.version = Version(0, 3, 2);
        assert!(install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle)
            .unwrap_err()
            .contains("promised 0.3.2"));
        unchanged(&app);
        // An archive whose gary doesn't run as the version it claims.
        let (path, release) = stage(&dir, "0.3.1", "linux-arm64", "0.2.9", &signer);
        assert!(install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle)
            .unwrap_err()
            .contains("reported '0.2.9'"));
        unchanged(&app);
        // One update at a time.
        let _held = app.lock().unwrap();
        let (path, release) = stage(&dir, "0.3.1", "linux-arm64", "0.3.1", &signer);
        assert!(install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle)
            .unwrap_err()
            .contains("Another gary update"));
        unchanged(&app);
    }

    #[test]
    fn rollback_swaps_current_and_previous_and_update_reinstalls_cleanly() {
        let dir = Dir::new("rollback");
        let signer = Signer::new();
        let app = installed(&dir);
        assert!(rollback(&app, idle).unwrap_err().contains("no previous version"));
        let (path, release) = stage(&dir, "0.3.1", "linux-arm64", "0.3.1", &signer);
        install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle).unwrap();
        assert_eq!(
            rollback(&app, || Err("busy".into())).unwrap_err(),
            "busy"
        );
        assert_eq!(app.linked("current"), Some(Version(0, 3, 1)));
        assert_eq!(rollback(&app, idle).unwrap(), (Version(0, 3, 1), Version(0, 3, 0)));
        assert_eq!(app.linked("current"), Some(Version(0, 3, 0)));
        assert_eq!(app.linked("previous"), Some(Version(0, 3, 1)));
        let ran = std::process::Command::new(app.root().join("current/gary")).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&ran.stdout).trim(), "0.3.0");
        // Updating again replaces the rolled-back copy with a fresh verified one.
        fs::write(app.version_dir(Version(0, 3, 1)).join("gary"), "#!/bin/sh\nexit 1\n").unwrap();
        install(&app, &LINUX_ARM64, &release, &path, &signer.pubkey(), idle).unwrap();
        assert_eq!(app.linked("current"), Some(Version(0, 3, 1)));
        assert_eq!(app.linked("previous"), Some(Version(0, 3, 0)));
        let ran = std::process::Command::new(app.root().join("current/gary")).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&ran.stdout).trim(), "0.3.1");
        // A previous version that no longer runs isn't switched to.
        fs::write(app.version_dir(Version(0, 3, 0)).join("gary"), "#!/bin/sh\nexit 1\n").unwrap();
        assert!(rollback(&app, idle).is_err());
        assert_eq!(app.linked("current"), Some(Version(0, 3, 1)));
    }

    #[test]
    fn only_a_versions_folder_with_a_current_link_is_managed() {
        let dir = Dir::new("managed");
        let app = installed(&dir);
        assert!(AppDir::of_executable(&app.version_dir(Version(0, 3, 0)).join("gary")).is_some());
        let loose = dir.0.join("extracted");
        fs::create_dir_all(&loose).unwrap();
        fs::write(loose.join("gary"), "").unwrap();
        assert!(AppDir::of_executable(&loose.join("gary")).is_none());
        fs::remove_file(app.root().join("current")).unwrap();
        assert!(AppDir::of_executable(&app.version_dir(Version(0, 3, 0)).join("gary")).is_none());
    }

    #[tokio::test]
    async fn the_feed_and_archive_download_end_to_end_from_a_local_server() {
        use axum::{routing::get, Router};
        let dir = Dir::new("e2e");
        let signer = Signer::new();
        let app = installed(&dir);
        let data = archive("0.3.1", "linux-arm64", "0.3.1");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let archive_url = format!("{base}/download/{}", LINUX_ARM64.archive);
        let release = release(Version(0, 3, 1), &data, &signer);
        let phase = json!({"channel":"preview","latest_version":"0.3.1","download_url":archive_url,
            "sha256":release.sha256,"published_at":"2026-11-24T00:00:00Z","notes":["Test."]})
        .to_string();
        let native = json!({"version":"0.3.1","notes":"Test.","pub_date":"2026-11-24T00:00:00Z",
            "platforms":{LINUX_ARM64.key:{"signature":release.signature,"url":archive_url}}})
        .to_string();
        let router = Router::new()
            .route("/feed/preview.json", get(move || std::future::ready(phase.clone())))
            .route("/feed/native-preview.json", get(move || std::future::ready(native.clone())))
            .route(
                &format!("/download/{}", LINUX_ARM64.archive),
                get(move || std::future::ready(data.clone())),
            );
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let source = Source {
            feed: format!("{base}/feed/"),
            pubkey: signer.pubkey(),
        };
        let fetched = fetch(&source, &LINUX_ARM64).await.unwrap();
        assert_eq!(fetched.version, Version(0, 3, 1));
        let missing = Source {
            feed: format!("{base}/nothing"),
            ..source.clone()
        };
        assert!(fetch(&missing, &LINUX_ARM64).await.unwrap_err().contains("No headless update feed"));
        let archive = dir.0.join("download.tar.gz");
        download(&fetched, &archive, |_| {}).await.unwrap();
        install(&app, &LINUX_ARM64, &fetched, &archive, &source.pubkey, idle).unwrap();
        assert_eq!(app.linked("current"), Some(Version(0, 3, 1)));
    }
}
