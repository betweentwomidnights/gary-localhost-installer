# macOS desktop preview

The first preview targets Apple Silicon on macOS 13.3 or newer. Validate it on
the M4 before sharing a build. Intel Macs and macOS headless mode are not yet
validated.

## Build locally

Install Xcode Command Line Tools (`xcode-select --install`), Rust and Node.js 24,
then build from the source branch:

```sh
git clone https://github.com/betweentwomidnights/gary-localhost-installer.git
cd gary-localhost-installer
git switch feature/macos-native-desktop
cd control-center
npm ci
VITE_ENABLE_APP_UPDATER=0 npm run tauri -- build --target aarch64-apple-darwin
```

Open `src-tauri/target/aarch64-apple-darwin/release/bundle/macos/gary4local.app`.
The local build uses an ad-hoc signature and needs no signing credentials.
Runtime storage defaults to
`~/Library/Application Support/com.betweentwomidnights.gary4local/runtime`.

## Signed builds in GitHub

The **macOS desktop preview** workflow builds the app and disk image on GitHub's
Apple Silicon runner. PR builds use ad-hoc signing. Pushes to the preview branch
and manual runs with **sign** enabled use Developer ID signing and notarization.
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
The M4 does not need the certificate, private keys or manual codesigning.
The [Tauri signing guide](https://v2.tauri.app/distribute/sign/macos/) describes
the Developer ID and API key setup.

After this workflow is merged, publishing a matching `v<app-version>` GitHub
release automatically builds, signs, notarizes and attaches the Mac assets.
Manual preview runs only retain Actions artifacts. Mac in-app updates remain
disabled until their feeds and updater signatures are prepared; Apple's signing
credentials are separate from the existing updater key.

## M4 check before sharing

Check launch from Applications, the icon, model downloads, Metal runtime
installation, generation, SA3/Carey training when their packages are available,
and restart/storage persistence. CI packaging checks cannot establish GPU or GUI
behavior. Runtime package availability is a separate gate from app signing.
