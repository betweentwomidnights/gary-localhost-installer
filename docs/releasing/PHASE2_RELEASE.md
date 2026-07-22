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

1. Bump the app version in all five spots. they must match or the build fails:
   - `control-center/package.json`
   - `control-center/package-lock.json` (two `gary4local` entries; leave dependency versions alone)
   - `control-center/src-tauri/Cargo.toml`
   - `control-center/src-tauri/Cargo.lock` (the `gary4local` package block only)
   - `control-center/src-tauri/tauri.conf.json`

   `cargo check` from `control-center/src-tauri` is the quickest way to confirm
   `Cargo.lock` still agrees with `Cargo.toml`.

2. Write the release notes in the repo, in two places:
   - add a `## vX.Y.Z` section at the top of `CHANGELOG.md`, ending with a
     `compatible with gary4juce vA.B.C.` line.
   - replace the headline `## vX.Y.Z` section near the top of `README.md` with
     the new one. the README carries only the current release; everything older
     lives in the changelog. this step is easy to forget — v0.1.19 shipped with
     the README still showing v0.1.18.

   both should follow `kevs_docs_style.md` (lowercase headings, contractions,
   no marketing voice).

3. Build the signed NSIS updater artifact:

```powershell
cd C:\path\to\backend-installer\control-center
$env:TAURI_SIGNING_PRIVATE_KEY="C:\path\to\gary4local-updater.key"
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD="your real passphrase"
npm.cmd run tauri build -- --config src-tauri/tauri.updater.conf.json
```

   On the release machine there's a gitignored `LOCAL_SIGNING_NOTES.md` at the
   repo root with PowerShell helpers that wrap this step, so the key paths and
   passphrase never have to be typed by hand. It's intentionally not in the
   repo — if it's missing, this command is still the source of truth and the
   notes can be rebuilt around it.

4. Create the GitHub release tag, for example `v0.1.3`.
5. Upload both files from `control-center\src-tauri\target\release\bundle\nsis\`:
   - `gary4local_<version>_x64-setup.exe`
   - `gary4local_<version>_x64-setup.exe.sig`
6. Keep the gary4juce compatibility line appropriate for where it appears:
   - In the GitHub release notes, link to the current recommended gary4juce
     release tag.
   - In the updater feed notes, use a short plain-text line such as
     `Compatible with gary4juce v4.0.2.` Do not include a URL. The current
     update prompt does not provide clickable links, and the raw URL wastes
     limited UI space.
7. Generate both updater feeds from the exact built installer and signature:

```powershell
cd C:\path\to\backend-installer
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
  -Version "0.1.3" `
  -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.1.3/gary4local_0.1.3_x64-setup.exe" `
  -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe" `
  -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe.sig" `
  -Channel "stable" `
  -NotesText "Release note one.||Release note two."
```

8. Review the generated files:
   - `docs/updates/gary4local/stable.json`
   - `docs/updates/gary4local/native-stable.json`
9. Commit those feed changes to `main` and push.
10. Wait for GitHub Pages to publish the updated JSON.
11. Sanity-check the live URLs:
   - `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json`
   - `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/native-stable.json`
12. Launch the currently installed app and verify it offers `install update`.

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
  -NotesText "Phase 2 updater preview"
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
