# Phase 2 Release Guide

This is the maintainer checklist for shipping a `gary4local` release with:

- Phase 1 manifest notes and fallback browser download
- Phase 2 signed in-app install

Normal users should not need any updater env vars. Production builds use the baked-in stable feeds:

- `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json`
- `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/native-stable.json`

## One-Time Setup

1. Keep the updater private key outside the repo.
2. Keep the updater public key committed in:
   - `control-center/src-tauri/src/update.rs`
   - `control-center/src-tauri/tauri.updater.conf.json`
3. Make sure GitHub Pages is publishing from `main /docs`.

## Per Release

1. Bump the app version in:
   - `control-center/package.json`
   - `control-center/package-lock.json`
   - `control-center/src-tauri/Cargo.toml`
   - `control-center/src-tauri/Cargo.lock`
   - `control-center/src-tauri/tauri.conf.json`
2. Build the signed NSIS updater artifact:

```powershell
cd C:\path\to\backend-installer\control-center
$env:TAURI_SIGNING_PRIVATE_KEY="C:\path\to\gary4local-updater.key"
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD="your real passphrase"
npm.cmd run tauri build -- --config src-tauri/tauri.updater.conf.json
```

3. Create the GitHub release tag, for example `v0.1.3`.
4. Upload both files from `control-center\src-tauri\target\release\bundle\nsis\`:
   - `gary4local_<version>_x64-setup.exe`
   - `gary4local_<version>_x64-setup.exe.sig`
5. Generate both updater feeds from the exact built installer and signature:

```powershell
cd C:\path\to\backend-installer
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
  -Version "0.1.3" `
  -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.1.3/gary4local_0.1.3_x64-setup.exe" `
  -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe" `
  -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe.sig" `
  -Channel "stable" `
  -Notes @("Release note one.", "Release note two.")
```

6. Review the generated files:
   - `docs/updates/gary4local/stable.json`
   - `docs/updates/gary4local/native-stable.json`
7. Commit those feed changes to `main` and push.
8. Wait for GitHub Pages to publish the updated JSON.
9. Sanity-check the live URLs:
   - `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json`
   - `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/native-stable.json`
10. Launch the currently installed app and verify it offers `install update`.

## Preview Testing

Use the same helper for preview releases, but switch the channel:

```powershell
cd C:\path\to\backend-installer
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
  -Version "0.1.2-preview.1" `
  -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.1.2-preview.1/gary4local_0.1.2-preview.1_x64-setup.exe" `
  -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.2-preview.1_x64-setup.exe" `
  -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.2-preview.1_x64-setup.exe.sig" `
  -Channel "preview" `
  -Notes @("Phase 2 updater preview")
```

That writes:

- `docs/updates/gary4local/preview.json`
- `docs/updates/gary4local/native-preview.json`

Preview apps can point at those feeds with runtime env overrides.

## Source Builds

Source builders can still disable the entire updater UI and backend check path:

```powershell
$env:VITE_ENABLE_APP_UPDATER='0'
npm.cmd run tauri build
Remove-Item Env:VITE_ENABLE_APP_UPDATER
```

That remains the recommended opt-out for forks and local-only builds that should not advertise public releases.
