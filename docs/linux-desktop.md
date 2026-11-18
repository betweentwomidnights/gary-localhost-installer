# Linux desktop

gary4local runs on Linux as a `.deb` for `amd64` (Intel/AMD PCs) and `arm64`
(DGX Spark), built for Ubuntu 24.04. it's a preview for now: the arm64 build
has been tested end to end on a DGX Spark with gary4juce; the amd64 build is
built and checked by CI but hasn't been hand-tested on a desktop yet. GPU
generation is required; CPU generation isn't a target.

it's the same `gary4local` app as on Windows: package, executable, application
identity (`com.betweentwomidnights.gary4local`) and data folders. preview versus
stable is a matter of which update feed a build follows, never a separate app.
the package replaces an installed `gary4local-preview` from earlier builds.

Linux builds follow the preview channel for their architecture:
`updates/gary4local/linux-x86_64/native-preview.json` or
`linux-aarch64/native-preview.json`. accepting an update downloads the signed
`.deb` and installs it with `dpkg -i` through `pkexec`, which asks for your
password once, then restarts gary4local. how those feeds get published is in
[releasing gary4local](releasing.md).

native engines download from the pinned manifest when you set up a runtime;
models and GPU drivers aren't included. Linux uses Vulkan, on NVIDIA as well as
AMD and Intel. there's no Linux CUDA build.

## install and launch

download the `.deb` for your machine and the matching `SHA256SUMS-linux-x64`
or `SHA256SUMS-linux-arm64` from the
[latest release](https://github.com/betweentwomidnights/gary-localhost-installer/releases/latest),
then from that folder:

```sh
sha256sum -c --ignore-missing SHA256SUMS-linux-arm64   # linux-x64 on a PC
sudo apt install ./gary4local_*_arm64.deb              # _amd64.deb on a PC
gary4local-launch
```

a direct `sudo apt install` from a private home folder can warn that the
download is performed unsandboxed as root. that notice doesn't mean the install
failed.

a CI build artifact (the `gary4local-desktop-linux-*` zip from the
`Linux desktop` workflow) holds the same `.deb` plus `BUILD-INFO.json`, which
records the source, architecture, toolchain, package hash and ELF requirements,
and an `INSTALL.sh`. check the zip against GitHub's artifact digest, then from the unzipped folder:

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

`INSTALL.sh` requests sudo for APT and temporarily stages the `.deb` where
APT's `_apt` user can read it, so you don't get the unsandboxed warning. it
cleans up that copy after installation and doesn't loosen your home-directory
permissions.

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

## manual acceptance

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
