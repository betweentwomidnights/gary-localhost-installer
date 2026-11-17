# Linux desktop

This is an installable Tauri desktop application, with the bundled Svelte UI.
Native CI builds both `amd64` (Intel/AMD Linux PCs) and `arm64` (DGX Spark).
These first Linux builds target Ubuntu 24.04; they are not yet a general
Linux release. GPU generation is required; CPU generation is not a target.

It is the same `gary4local` app as on Windows: package, executable, application
identity (`com.betweentwomidnights.gary4local`) and data folders. Preview versus
stable is a matter of which updater feed a build follows, never a separate app.
The package replaces an installed `gary4local-preview` from earlier builds.

CI builds follow the preview channel (`GARY4LOCAL_UPDATE_CHANNEL=preview`) for
their architecture: `updates/gary4local/linux-x86_64/native-preview.json` or
`linux-aarch64/native-preview.json`. Accepting an update downloads the signed
`.deb` and installs it with `dpkg -i` through `pkexec`, which asks for your
password once, then restarts gary4local. Feed publishing is described in
`docs/updates/README.md` on main. Native engines still download explicitly from the pinned manifest; models and
GPU drivers are not included. Linux currently offers Vulkan, including on NVIDIA.

## Install and launch

Verify the downloaded artifact ZIP against GitHub's artifact digest, then verify
the inner package using its supplied `SHA256SUMS-linux-*`. `BUILD-INFO.json`
records the source, architecture, toolchain, package hash and ELF requirements.

From the directory containing the correct package for your machine:

```sh
sha256sum -c SHA256SUMS-linux-arm64 # linux-x64 on a PC
sh INSTALL.sh ./gary4local_*.deb
gary4local-launch
```

The application-menu entry also uses this launcher. gary4local keeps its
configuration in `~/.config`, its settings and storage choice in
`~/.local/share/com.betweentwomidnights.gary4local`, and its runtime, engines
and models in that folder's `runtime` unless you choose other storage in the app.

For debugging or a throwaway test, set `GARY4LOCAL_HOME` to an absolute path:
configuration, data and caches then all stay inside it, with the runtime in its
`data` folder. Set it in `~/.config/environment.d/` to cover menu launches too,
and unset it to return to the normal install. Data from an earlier
`gary4local-preview` build stays where it was, by default in
`$HOME/gary4local-packaging/desktop-gui`; point `GARY4LOCAL_HOME` there to keep
using it. An explicit `GARY4LOCAL_RUNTIME_ROOT` is respected. Run as your normal
desktop user.

`INSTALL.sh` requests sudo for APT and temporarily stages the public DEB where
APT's `_apt` user can read it. It cleans up that copy after installation and
does not loosen your home-directory permissions. A direct `sudo apt install`
from a private home can warn that it is running unsandboxed; that notice does
not by itself mean installation failed.

The Linux package uses Gary PNG icons generated from the repository's
`icon.png`; `.icns` is a macOS format. Its desktop-entry filename matches the
window's application class so GNOME can associate the correct dock icon.

On NVIDIA/X11, startup applies the verified WebKit DMA-BUF renderer workaround
before GTK initializes. This affects UI presentation, not CUDA/Vulkan generation.
AMD/Intel and Wayland sessions retain their defaults. An explicit
`WEBKIT_DISABLE_DMABUF_RENDERER` value is respected.

If the DEB is only extracted for inspection, set `GARY4LOCAL_BINARY` to
the absolute extracted binary path before invoking the extracted launcher.
System UI dependencies still need to be present. Installing the actual package
and launching from its menu entry is the intended manual acceptance test.

## Manual acceptance

1. Launch in a real desktop session. Confirm the UI, services and settings render.
2. Check storage paths point at the standard folders above, or inside
   `GARY4LOCAL_HOME` when it is set. Exercise the folder dialog; cancel a change
   unless you intend to move test data.
3. Install one Vulkan runtime and select existing compatible weights where the
   UI supports them, or explicitly download the desired model. Do not copy the
   whole production model store. Existing servers may occupy default ports;
   inspect the port shown in the UI before starting a test service.
4. Generate from the normal client, check GPU/backend logs and listen to/export
   the result. Record the selected model, device and driver.
5. Stop, restart, and confirm settings persist. Test close/quit and tray behavior.
   Some desktops do not display tray icons; do not hide the window until a way
   to restore it is confirmed.
6. Quit before reinstalling/upgrading. Confirm data survives.
   `sudo apt remove gary4local` removes the package; user data remains.

To clean up a sandboxed test, quit gary4local, confirm its test processes stopped,
inspect the `GARY4LOCAL_HOME` folder, preserve wanted logs/audio, and remove only
that folder.
No systemd service is installed or enabled by the desktop package.

CI checks package architecture, executable modes, menu entry, pinned manifest,
absence of model/engine payloads, that it replaces `gary4local-preview`, the
launcher's default and `GARY4LOCAL_HOME` behavior, Linux storage behavior,
and actual package installation/startup plus a rendered-window capture under
Xvfb. A uniform blank capture fails the check. Virtual-display startup
does not establish GPU inference, tray availability, or real desktop behavior.

## WSL validation

WSL2 can test Linux x64 packaging and NVIDIA CUDA execution using the Windows
driver bridge. WSLg can exercise desktop startup and dialogs. Its Vulkan devices
must be inspected: a software Vulkan device is not a GPU-generation result.
Both desktop and headless runtime checks reject known software Vulkan drivers,
even when ggml labels them as a GPU.
WSL does not replace native Linux driver/desktop tests or AMD hardware tests.
Do not install NVIDIA Linux display drivers into WSL.
