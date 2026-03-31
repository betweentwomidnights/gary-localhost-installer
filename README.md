# gary4local

Local Windows control center and bundled backend services for `gary4juce`.

This branch is the `v2-refactor` Tauri/Rust implementation. The old PyInstaller/Inno Setup flow is intentionally not part of this branch anymore; that legacy path stays preserved on the old branch history.

## What Lives Here

- `control-center/`
  Tauri + Svelte desktop app that manages the local services, model downloads, installer flow, tray menu, and production runtime sync into `%APPDATA%\Gary4JUCE`.
- `services/`
  The Python backends and model-specific code for Gary, Terry, Jerry, Carey, and Foundation.
- `keygen_music_for_installer.wav`
  Source loop used to generate the tiny installer music asset.

## Services

- `gary` / MusicGen: `http://localhost:8000`
- `terry` / MelodyFlow: `http://localhost:8002`
- `carey` / ACE-Step: `http://localhost:8003`
- `jerry` / Stable Audio: `http://localhost:8005`
- `foundation-1`: `http://localhost:8015`

## Repo Layout Notes

- Development runs directly from the repo.
- Production syncs the bundled service source into `%APPDATA%\Gary4JUCE\services`.
- Mutable runtime data such as logs, virtual environments, caches, and models live under `%APPDATA%\Gary4JUCE`, not inside the installed app folder.

## Development

Prerequisites:

- Windows 10 or 11
- Node.js 20+
- Rust toolchain for Tauri builds
- WebView2
- `ffmpeg` if you want to regenerate the installer audio asset

Run the app in development:

```powershell
cd control-center
npm install
npm run tauri dev
```

## Production Build

Build the installer:

```powershell
cd control-center
npm run tauri build
```

Artifacts land in:

- `control-center/src-tauri/target/release/bundle/nsis/`
- `control-center/src-tauri/target/release/bundle/msi/`

The current preferred Windows artifact is the NSIS setup executable.

## Unsigned Builds

The installers are currently unsigned. If you want to verify a build, generate it locally and compare hashes:

```powershell
certutil -hashfile .\control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.0_x64-setup.exe SHA256
```

## Related Repos

- Plugin frontend: <https://github.com/betweentwomidnights/gary4juce>
- Combined backend alternative: <https://github.com/betweentwomidnights/gary-backend-combined>
