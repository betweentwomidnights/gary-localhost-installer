# macOS desktop

gary4local ships for Apple Silicon on macOS 13.3 or newer, starting with v0.7.6.
Validate each release on the M4 before sharing it. Intel Macs and macOS headless
mode are not yet validated.

## Build locally

Install Xcode Command Line Tools (`xcode-select --install`), Rust and Node.js 24,
then build from `main`:

```sh
git clone https://github.com/betweentwomidnights/gary-localhost-installer.git
cd gary-localhost-installer/control-center
npm ci
VITE_ENABLE_APP_UPDATER=0 npm run tauri -- build --target aarch64-apple-darwin
```

Open `src-tauri/target/aarch64-apple-darwin/release/bundle/macos/gary4local.app`.
The local build uses an ad-hoc signature and needs no signing credentials.
Runtime storage defaults to
`~/Library/Application Support/com.betweentwomidnights.gary4local/runtime`.

## Signed builds in GitHub

The **macOS desktop preview** workflow builds the app and disk image on GitHub's
Apple Silicon runner. PR builds use ad-hoc signing. Releases and manual runs
with **sign** enabled use Developer ID signing and notarization.
Download `gary4local-desktop-macos-arm64` from the completed run. CI verifies
the app signature, Gatekeeper assessment and stapled notarization tickets for
both the app and disk image.

The repository uses the same six Actions secrets as the upstream runtime repos:

- `MACOS_CERT_P12`: base64 Developer ID Application certificate and private key
- `MACOS_CERT_PASSWORD`: certificate export password
- `APPLE_TEAM_ID`: certificate's developer team
- `APPLE_NOTARY_KEY_P8`: base64 App Store Connect API private key
- `APPLE_NOTARY_KEY_ID` and `APPLE_NOTARY_ISSUER_ID`: API key identifiers

CI imports these into a temporary keychain and removes them after the build.
It checks notarization authentication before compiling. An HTTP 401 here means
the notary key, key ID and issuer must be checked together in Actions secrets.
The M4 does not need the certificate, private keys or manual codesigning.
The [Tauri signing guide](https://v2.tauri.app/distribute/sign/macos/) describes
the Developer ID and API key setup.

Publishing a matching `v<app-version>` GitHub release automatically builds,
signs, notarizes and attaches the Mac assets. Manual runs only retain Actions
artifacts.

The Mac app updates itself from the `preview` feed in
`docs/updates/gary4local/macos-aarch64/`. CI also packs the stapled app as
`gary4local_<version>_aarch64.app.tar.gz`, the archive `tauri-plugin-updater`
installs. The updater key never goes to GitHub, so the release machine signs that
archive after download, as it does the Linux `.deb`s; see
[releasing](releasing.md). Apple's signing credentials are separate from the
updater key, and no Mac needs the updater key.

The update replaces `gary4local.app` where it's installed and restarts. It asks
for an administrator password only if the app's folder isn't writable.

## M4 check before sharing

**Set up all runtimes** installs the published Metal packages for AudioCraft,
Stable Audio, Carey, and Yuey. Download models separately from each service's
Models panel.

Check launch from Applications, the icon, model downloads, Metal runtime
installation, generation, SA3/Carey training when their packages are available,
and restart/storage persistence. CI packaging checks cannot establish GPU or GUI
behavior. Runtime package availability is a separate gate from app signing.
