# auto-updater

gary4local has an optional updater inside the UI. this doc is mostly here for
transparency and for future-us or anyone else who ends up maintaining release
builds. you don't need to enable the updater to build or use the project.

## what users see

when the updater is included, gary4local:

- checks a static HTTPS manifest on startup
- shows release notes inside the app
- offers `not now` and `skip this version`
- falls back to `download update` when only the phase 1 manifest is available
- offers `install update` when a signed native updater feed is also available

on Windows, `install update` closes gary4local and hands off to the signed
installer. on Linux, it installs the signed `.deb` with `dpkg -i` through
`pkexec`, which asks for your password once, then gary4local restarts.

each build follows one feed, fixed when it's built. Windows release builds
follow `stable`; Linux builds follow `preview` for their architecture, and the
AMD tester build, `gary4local-rocm`, follows its own preview feed:

- `docs/updates/gary4local/stable.json` and `native-stable.json` (Windows)
- `docs/updates/gary4local/linux-x86_64/preview.json` and `native-preview.json`
- `docs/updates/gary4local/linux-aarch64/preview.json` and `native-preview.json`
- `docs/updates/gary4local-rocm/preview.json` and `native-preview.json`

## building without the updater

`VITE_ENABLE_APP_UPDATER` is a build-time flag, not a runtime toggle. set it to
`0` before building and the update UI and backend manifest checks are left out.

```powershell
cd control-center
npm ci
$env:VITE_ENABLE_APP_UPDATER='0'
npm run tauri build
Remove-Item Env:VITE_ENABLE_APP_UPDATER
```

the same flag also works with `npm run tauri dev` for local development.

this is useful for forks, local-only builds, and source builds that shouldn't
advertise public GitHub releases.

## preview testing

for local preview testing, you can override the stable defaults at runtime:

```powershell
$env:GARY4LOCAL_UPDATE_MANIFEST_URL="https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/preview.json"
$env:GARY4LOCAL_NATIVE_UPDATER_ENDPOINT="https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/native-preview.json"
$env:GARY4LOCAL_NATIVE_UPDATER_PUBKEY = Get-Content C:\path\to\gary4local-updater.key.pub -Raw
& "$env:LOCALAPPDATA\gary4local\gary4local.exe"
```

`GARY4LOCAL_UPDATE_MANIFEST_URL`, `GARY4LOCAL_NATIVE_UPDATER_ENDPOINT`, and
`GARY4LOCAL_NATIVE_UPDATER_PUBKEY` are runtime-only overrides and aren't baked
into production builds.

maintainer release instructions live in [releasing gary4local](releasing.md).
