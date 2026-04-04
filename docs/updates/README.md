# gary4local Update Manifest

Phase 1 uses a tiny static manifest so `gary4local` can check for new releases without depending on the DGX inference stack.

Default manifest URL:

`https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json`

Preview manifest URL:

`https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/preview.json`

That gives us a clean first deployment path:

1. commit the manifest under `docs/updates/gary4local/stable.json`
2. enable GitHub Pages for the repo's `docs/` directory
3. point the desktop app at the stable URL above

If you want to move it onto your own domain later, the safest shape is still a static endpoint. Put Cloudflare or your domain in front of the same JSON file instead of tying update checks to the DGX docker network.

Recommended release flow:

1. build the NSIS installer
2. upload the installer to the GitHub release
3. compute `SHA256SUMS.txt`
4. update `docs/updates/gary4local/stable.json`
5. merge so the static manifest goes live

## Preview Testing

Use `preview.json` to debug the updater UI before shipping a real stable release.

Suggested flow:

1. keep `stable.json` pointed at the real public release
2. edit `preview.json` with a version higher than your local build, such as `0.1.2-preview.1`
3. launch `gary4local` with `GARY4LOCAL_UPDATE_MANIFEST_URL` set to the preview manifest URL
4. exercise the modal, skip/resume behavior, notes, and link actions
5. once the UX feels right, ship the real stable release and update `stable.json`

PowerShell example for local testing:

```powershell
$env:GARY4LOCAL_UPDATE_MANIFEST_URL="https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/preview.json"
npm.cmd run tauri dev
```

The current implementation is a Phase 1 updater:

- it checks a lightweight manifest
- it shows update UI inside the app
- it opens a download link in the browser

If you want the fuller "download, install, relaunch" experience later, that should be a Phase 2 migration onto Tauri's updater plugin with signed update artifacts.

## Phase 2 Preview

The Phase 2 spike keeps the current Phase 1 manifest as the user-facing source of truth, but can upgrade the modal from `download update` to `install update` when a signed Tauri updater endpoint is also configured.

Runtime environment variables:

- `GARY4LOCAL_NATIVE_UPDATER_ENDPOINT`
- `GARY4LOCAL_NATIVE_UPDATER_PUBKEY`

Suggested preview setup:

1. keep using `preview.json` for the visible app notes and version prompt
2. publish a separate Tauri updater JSON for the same version
3. point `GARY4LOCAL_NATIVE_UPDATER_ENDPOINT` at that signed Tauri updater JSON
4. point `GARY4LOCAL_NATIVE_UPDATER_PUBKEY` at the public updater key content

Build notes:

- normal builds do not need updater artifacts
- Phase 2 test builds should use the config overlay at `control-center/src-tauri/tauri.updater.conf.json`
- Tauri updater artifacts also require `TAURI_SIGNING_PRIVATE_KEY` during build

PowerShell example:

```powershell
cd control-center
$env:TAURI_SIGNING_PRIVATE_KEY="C:\path\to\gary4local-updater.key"
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD=""
npm.cmd run tauri build -- --config src-tauri/tauri.updater.conf.json
```

The Tauri updater JSON format is different from the Phase 1 manifest. It needs `version`, optional `notes`, optional `pub_date`, and a platform entry containing `url` and inline `signature`.
