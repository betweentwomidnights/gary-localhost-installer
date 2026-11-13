# Experimental Linux desktop preview

This is an installable Tauri desktop application, with the bundled Svelte UI.
Native CI builds both `amd64` (Intel/AMD Linux PCs) and `arm64` (DGX Spark).
These first desktop previews target Ubuntu 24.04; they are not yet a general
Linux release. GPU generation is required; CPU generation is not a target.

The separate `gary4local-preview` package, executable and application identity
allow review without replacing a production installation. The app updater is
disabled at build time in both Rust and the frontend. Native engines still
download explicitly from the pinned manifest; models and GPU drivers are not
included. Linux previews currently offer Vulkan, including on NVIDIA.

## Install and launch

Verify the downloaded artifact ZIP against GitHub's artifact digest, then verify
the inner package using its supplied `SHA256SUMS-linux-*`. `BUILD-INFO.json`
records the source, architecture, toolchain, package hash and ELF requirements.

From the directory containing the correct package for your machine:

```sh
sha256sum -c SHA256SUMS-linux-arm64 # linux-x64 on a PC
sudo apt install ./gary4local-preview_*.deb
gary4local-preview-launch
```

The application-menu entry also uses this launcher. By default, all preview
runtime data, configuration and caches stay in
`$HOME/gary4local-packaging/desktop-gui`. On the Spark this is
`/home/kev/gary4local-packaging/desktop-gui`. Override `GARY4LOCAL_PREVIEW_HOME`
with an absolute path to keep another dedicated test directory. An explicit
`GARY4LOCAL_RUNTIME_ROOT` is respected. Run as your normal desktop user.

If the DEB is only extracted for inspection, set `GARY4LOCAL_PREVIEW_BINARY` to
the absolute extracted binary path before invoking the extracted launcher.
System UI dependencies still need to be present. Installing the actual package
and launching from its menu entry is the intended manual acceptance test.

## Manual acceptance

1. Launch in a real desktop session. Confirm the UI, services and settings render.
2. Check storage paths point inside the dedicated preview root. Exercise the
   folder dialog; cancel a change unless you intend to move test data.
3. Install one Vulkan runtime and select existing compatible weights where the
   UI supports them, or explicitly download the desired model. Do not copy the
   whole production model store. Existing servers may occupy default ports;
   inspect the port shown in the UI before starting a test service.
4. Generate from the normal client, check GPU/backend logs and listen to/export
   the result. Record the selected model, device and driver.
5. Stop, restart, and confirm settings persist. Test close/quit and tray behavior.
   Some desktops do not display tray icons; do not hide the window until a way
   to restore it is confirmed.
6. Quit before reinstalling/upgrading the preview. Confirm test data survives.
   `sudo apt remove gary4local-preview` removes the package; user data remains.

To clean up, quit the preview, confirm its test processes stopped, inspect its
dedicated directory, preserve wanted logs/audio, and remove only that directory.
No systemd service is installed or enabled by the desktop preview.

CI checks package architecture, executable modes, menu entry, pinned manifest,
absence of model/engine payloads, launcher isolation, Linux storage behavior,
and actual package installation/startup under Xvfb. Virtual-display startup
does not establish GPU inference, tray availability, or real desktop behavior.

## WSL validation

WSL2 can test Linux x64 packaging and NVIDIA CUDA execution using the Windows
driver bridge. WSLg can exercise desktop startup and dialogs. Its Vulkan devices
must be inspected: a software Vulkan device is not a GPU-generation result.
Both desktop and headless runtime checks reject known software Vulkan drivers,
even when ggml labels them as a GPU.
WSL does not replace native Linux driver/desktop tests or AMD hardware tests.
Do not install NVIDIA Linux display drivers into WSL.
