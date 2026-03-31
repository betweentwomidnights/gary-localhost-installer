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

## Gary Localhost Optimizations

The local `gary` service applies several MusicGen inference optimizations that are specific to the localhost deployment:

- `musicgen_fast.py` converts remaining float32 parameters and buffers to fp16 to avoid extra dtype conversion overhead during generation.
- Self-attention layers are patched to use a pre-allocated static KV cache instead of repeatedly growing tensors with `torch.cat`.
- If available in the local Gary environment, Flash Attention 2 is patched in directly for MusicGen self-attention.
- The service performs a small first-load kernel warmup pass per model/device so later generations start faster.

For localhost we intentionally do **not** enable `torch.compile` by default. Gary unloads models after generation so we can support many finetunes on smaller GPUs without keeping large model instances resident, and that model lifecycle usually makes compile overhead a poor tradeoff.

## Carey Localhost Notes

The local `carey` service includes a small-GPU decode strategy that differs from upstream `ace-lego` behavior:

- During decode, localhost Carey can temporarily offload the DiT model so the VAE decode step has more VRAM available.
- The decode path falls back through progressively safer modes, including tiled decode, CPU-offloaded decode, and full CPU decode, instead of hard-failing immediately on lower-memory GPUs.

This helps ACE-Step remain usable on consumer cards where generation may fit in VRAM but decode is the step that would otherwise tip the process into an out-of-memory failure.

## Terry Localhost Optimizations

The local `terry` service now supports an optional Flash Attention 2 path for MelodyFlow on CUDA:

- Terry runs MelodyFlow on `torch 2.7.1` with the same Windows FA2 wheel family used by the other CUDA services.
- `melodyflow_fast.py` patches the DiT self-attention blocks at runtime to call FA2 directly when the wheel is installed and the tensors are in a supported CUDA dtype.
- Cross-attention stays on the existing AudioCraft attention path because those blocks can carry masks, so the FA2 patch stays focused on the large self-attention passes over audio latents.
- The optimization is disabled by default and can be enabled with `MELODYFLOW_USE_FLASH_ATTN=1` or from the `gary4local` Terry panel when that feature is included in the build.

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
npm ci
npm run tauri build
```

If you want a build that hides the experimental Terry Flash Attention toggle entirely, set the feature flag before building:

```powershell
cd control-center
npm ci
$env:VITE_ENABLE_MELODYFLOW_FA2_TOGGLE='0'
npm run tauri build
Remove-Item Env:VITE_ENABLE_MELODYFLOW_FA2_TOGGLE
```

Notes:

- `VITE_ENABLE_MELODYFLOW_FA2_TOGGLE` is a build-time flag, not a runtime toggle.
- When this flag is set to `0`, the Terry Flash Attention setting is removed from the UI and the packaged app forces MelodyFlow to stay on the standard attention path.
- Leaving the flag unset keeps the Terry Flash Attention panel available, but the optimization itself still defaults to off unless the user enables it.

Artifacts land in:

- `control-center/src-tauri/target/release/bundle/nsis/`
- `control-center/src-tauri/target/release/bundle/msi/`

The current preferred Windows artifact is the NSIS setup executable.

## Unsigned Builds

The installers are currently unsigned. The intended verification flow is:

1. Build the installer locally from this branch with one of the commands above.
2. Compare the generated hash against the release artifact hash.

Example:

```powershell
certutil -hashfile .\control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.0_x64-setup.exe SHA256
```

## Related Repos

- Plugin frontend: <https://github.com/betweentwomidnights/gary4juce>
- Combined backend alternative: <https://github.com/betweentwomidnights/gary-backend-combined>
