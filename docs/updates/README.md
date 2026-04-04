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
