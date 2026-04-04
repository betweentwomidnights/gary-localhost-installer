# gary4local Update Feeds

`gary4local` now supports two updater layers:

- Phase 1 manifest: release notes, version prompt, browser-download fallback
- Phase 2 native updater feed: signed in-app install on Windows

Production builds should use the baked-in stable endpoints:

- Phase 1: `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json`
- Phase 2: `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/native-stable.json`

The public updater key is also baked into the app for production builds. Users should not need PowerShell env vars to receive updates.

## Preview Overrides

Preview testing can still override the baked-in defaults at runtime:

- `GARY4LOCAL_UPDATE_MANIFEST_URL`
- `GARY4LOCAL_NATIVE_UPDATER_ENDPOINT`
- `GARY4LOCAL_NATIVE_UPDATER_PUBKEY`

PowerShell example:

```powershell
$env:GARY4LOCAL_UPDATE_MANIFEST_URL="https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/preview.json"
$env:GARY4LOCAL_NATIVE_UPDATER_ENDPOINT="https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/native-preview.json"
$env:GARY4LOCAL_NATIVE_UPDATER_PUBKEY = Get-Content C:\path\to\gary4local-updater.key.pub -Raw
& "$env:LOCALAPPDATA\gary4local\gary4local.exe"
```

## Signed Build

Signed updater artifacts are generated with the updater overlay config:

```powershell
cd control-center
$env:TAURI_SIGNING_PRIVATE_KEY="C:\path\to\gary4local-updater.key"
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD="your-passphrase"
npm.cmd run tauri build -- --config src-tauri/tauri.updater.conf.json
```

Notes:

- the updater overlay builds `nsis` only
- this is the artifact used by the in-app updater
- `.sig` files are produced beside the installer

## Feed Generator

Use the combined helper to generate both updater feeds from the same installer and signature:

```powershell
cd <repo-root>
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
  -Version "0.1.3" `
  -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.1.3/gary4local_0.1.3_x64-setup.exe" `
  -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe" `
  -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe.sig" `
  -Channel "stable" `
  -Notes @("Real release note one.", "Real release note two.")
```

That writes:

- `docs/updates/gary4local/stable.json`
- `docs/updates/gary4local/native-stable.json`

For preview releases, use `-Channel "preview"` instead. The same helper will write:

- `docs/updates/gary4local/preview.json`
- `docs/updates/gary4local/native-preview.json`

## Release Guide

The maintainer-facing release checklist lives in [PHASE2_RELEASE.md](../releasing/PHASE2_RELEASE.md).
