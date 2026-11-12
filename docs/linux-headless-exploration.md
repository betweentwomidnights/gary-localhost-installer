# Linux and headless exploration

Tracks [issue #13](https://github.com/betweentwomidnights/gary-localhost-installer/issues/13).
Branch: `feature/linux-headless`. This is the first portability pass, not a
Linux release or a working headless server.

Follow-up: [sa3.cpp packaging PR #95](https://github.com/betweentwomidnights/sa3.cpp/pull/95)
adds upstream Linux core/CUDA/Vulkan packages. On Spark, both ARM64 package sets
passed 52 CTests, extracted-package controls and real GB10 GPU probes, then
generated two-second audio through the native CLI using existing weights.
The work is contained in `/home/kev/gary4local-packaging`, with a storage ledger
and cleanup inventory; its initial footprint is about 1.3 GiB. These are engine
package checks. This application's Linux installation, compatibility adapters,
and headless supervisor still need integration work, and no Linux runtime URLs
have been pinned here. CPU generation is not a target; CPU libraries/tests
support the CUDA/Vulkan runtime paths.

## Spark baseline, November 12, 2026

SSH works with the existing NVIDIA Sync/OpenSSH configuration:

```powershell
ssh -o BatchMode=yes spark-c7ca
```

The host reports Ubuntu 24.04.5, `aarch64`, NVIDIA GB10, driver 580.178.04,
and an installed CUDA toolkit 13.0.88. Its package key is `linux-arm64`;
an ordinary Linux PC needs `linux-x64`. Both need independent runtime builds.
The PCI GPU is `000f:01:00.0`, vendor `0x10de`, display class `0x030000`.
Several other NVIDIA PCI devices are bridges, so counting vendor IDs alone
would overcount GPUs.

The machine already runs the remote backend. Development files and compilation
live separately under `~/gary4local-linux-exploration`. The compile/test
container has no GPU access, published ports, production mounts, or Docker
socket mount. No existing services or model stores are changed. CUDA/Vulkan
inference has **not** been validated by this compiler check.

The host has Node 18 and no Rust toolchain on its normal PATH. GTK/WebKit
runtime libraries exist, but their development pkg-config files were not
available. An ARM64 Rust container supplies the compiler and desktop headers;
it does not require a display or an install of development packages on the host.

## First portability fixes

- Recognize `linux-arm64` and Intel `macos-x64`, alongside the existing targets.
  Recognition does not imply that matching runtime packages have been published.
- Guard the Windows process API import and `taskkill` call. Stop a direct Unix
  child with Rust's process API. Full descendant cleanup is still outstanding.
- Make the service-manager process fixtures work on Unix as well as Windows.
- Detect Linux display-class PCI GPUs through sysfs, without X11, Wayland,
  `nvidia-smi`, or `vulkaninfo`. This identifies a backend candidate; the existing
  `--props` check must still prove that a downloaded backend can initialize.
- Preserve ordinary Unix permission bits when extracting runtime ZIPs, so
  packaged servers remain executable. Do not apply setuid/setgid/sticky bits.
- Open Linux browser links with `xdg-open`.

The untouched baseline failed its ARM64 library check on the unconditional
Windows import and `creation_flags` call in `service_manager.rs`. With these
fixes, `cargo check --locked --all-targets` passes on ARM64 Linux.

Validation on this branch:

| Check | Spark ARM64 Linux | Windows x64 |
| --- | --- | --- |
| `cargo check --locked --all-targets` | Pass | Pass |
| Native runtime unit tests | 23 passed, 1 ignored | 22 passed, 1 ignored |
| Service manager unit tests | 28 passed, 1 ignored | 28 passed, 1 ignored |
| `cargo build --locked --bin gary4local` | Pass; ARM64 ELF debug executable | Not repeated for this exploration |

The ignored tests require downloaded models/a supported GPU. The Unix-only ZIP
permission test accounts for the extra Linux test. The debug executable is at
`~/gary4local-linux-exploration/baseline/control-center/src-tauri/target/debug/gary4local`
on Spark (the target directory is reused from the baseline for build caching).
It is a desktop executable, not the proposed daemon. No desktop session,
frontend release bundle, or GPU inference was tested.

## Reproduce the compiler check

On a Linux machine with Docker, from this repository's root:

```sh
node control-center/src-tauri/scripts/stage_runtime_resources.mjs
docker build -t gary4local-linux-check -f smoke-tests/linux/Dockerfile smoke-tests/linux
docker run --rm --cpus=4 --memory=8g \
  -v "$PWD:/workspace:ro" \
  -v gary4local-linux-target:/target \
  -v gary4local-linux-cargo:/usr/local/cargo \
  -e CARGO_TARGET_DIR=/target \
  -e CARGO_BUILD_JOBS=4 \
  gary4local-linux-check
```

The staging command populates resources before the read-only container check;
alternatively, use a writable disposable source checkout so `build.rs` can
create its placeholder resource directories. Use
a current Node version compatible with `control-center/package.json` for the
frontend build; the Spark's system Node is not the frontend build environment.

To run the runtime or service-manager tests, use the same container options
with `cargo test --locked --lib native_runtime::tests::` or
`cargo test --locked --lib service_manager::tests::` appended after the image.
These tests do not need a display or GPU. Desktop startup, bundling, and GPU
generation are separate checks.

## Remaining Linux and macOS work

| Area | Current gap | Next step |
| --- | --- | --- |
| Runtime distribution | The manifest pins Windows x64 packages only. | Build/hash/pin Linux x64 and ARM64 packages in each engine repo; then macOS Metal packages. Keep the existing checksum and staging/probe checks. |
| Executable names | Manifest entries and native trainer/helper paths use `.exe`. | Resolve executable/tool names per platform across installation, launch, training, migration, and probes. Do not simply rename Linux binaries to `.exe`. |
| Shared libraries | Shared CUDA packs currently contain Windows DLLs and use `PATH`. | Define Linux `.so` dependencies, `$ORIGIN` RPATH, and CUDA toolkit/driver compatibility; audit macOS dylibs/install names separately. A Windows CUDA pack is not reusable on Spark. |
| Storage | APPDATA/LOCALAPPDATA/USERPROFILE and storage beside the executable are Windows defaults. | Use writable XDG directories on Linux and Application Support on macOS; preserve explicit runtime-root overrides. Packaged `/usr/bin` or AppImage mounts must not become data stores. |
| Process ownership | The non-Windows workload enrollment is a no-op; stop kills the direct child only. | Own Unix process groups, implement bounded graceful shutdown and forced descendant cleanup, and test both normal and abnormal exits. Use systemd cgroup ownership for the daemon. |
| Desktop shell | Compilation is now possible; tray/WebKit startup and packaging are untested. | Build the frontend and a Linux desktop bundle, then test on a Linux desktop. Validate macOS separately. |
| Updates | Published update feeds/artifacts target Windows. | Disable the updater in exploratory desktop builds; publish platform-specific updater artifacts only after packaging is validated. |
| Legacy paths | Python environment discovery and build commands assume Windows. | Keep legacy Windows migration intact; new Linux installs should stay on the native path. Audit fallback/training code rather than accidentally invoking Windows setup commands. |

## Headless design

Use one shared Rust service-management core with two entry points: the Tauri
desktop application and a `gary4local-server` CLI/daemon. The daemon must build
without GTK, WebKit, tray, or a display server. Hiding a Tauri window would
still initialize the desktop runtime and is not sufficient.

The core should own the manifest, runtime installation, model selection,
profile/storage resolution, process lifetime, health polling, and current
generation settings. Extract startup from `lib.rs`'s Tauri `setup` closure.
Replace core `AppHandle`/Tauri event dependencies with progress/status callbacks
and Tokio tasks. `native_runtime.rs` already has an event-sink installation
path, which is a useful starting point. Desktop commands and the daemon should
call the same code so they cannot diverge on settings or model compatibility.

Keep the Rust HTTP compatibility adapters in this core. For example,
`sa3_adapter.rs` owns the public Python-era routes while the native server
listens on a private port. Running only the C++ binaries would skip behavior
that existing clients rely on.

Start with explicit commands for inspecting status, installing a runtime,
selecting/downloading models, and serving a named set of services under an
explicit runtime root. Do not start all models automatically. The initial
daemon can use configuration plus stdout/journal logs; a browser management UI
can come later. A per-user systemd unit should define restart policy and stop
all owned child processes. Installation/model downloads must remain explicit.

### First remote connection

Keep all generation and adapter listeners on loopback initially. SSH forwarding
over the tailnet lets an existing localhost client exercise the Spark without
changing the servers' authentication or exposing service management. For
example, once an isolated test Gary service actually runs on Spark port 28000:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:28000 spark-c7ca
```

The Windows client then uses its existing `127.0.0.1:8000` endpoint. The local
port must be free; stop the local Gary service before forwarding. Reserve and
check a separate test port range on Spark, including private adapter/upstream
ports; the example does not claim that port 28000 is permanently available.

For direct tailnet/LAN access later, add the `gary4juce` backend URL setting and
an explicit listen address plus authentication/pairing policy. Bind private
native servers and administrative endpoints to loopback. Remote generation
and model/storage management need separate exposure decisions. A tailnet name
by itself does not authenticate an HTTP request, and `0.0.0.0` would expose
more interfaces than the tailnet. Review file paths/URLs and polling/download
routes too: a client cannot read a server-local output file directly.

## GPU validation sequence

1. Build one native engine for Spark ARM64 against its installed toolkit,
   recording its source/submodule commits, compiler flags, and CUDA architecture.
   Determine the GB10 target from the actual compiler/device; do not copy an
   x64 Windows architecture list blindly.
2. Stage separate CUDA and Vulkan packages. Run each server's `--version` and
   `--props` in isolation and capture the initialized backend/device. A detected
   PCI GPU or a Vulkan loader library alone does not prove backend support.
3. Run short fixed-seed generation through both the native API and the existing
   compatibility adapter on unused ports, with a dedicated model/output root.
   Reuse weights read-only where the engine supports it. Measure memory, time,
   output validity, and continuation/LoRA behavior. Coordinate GPU-heavy tests
   with the existing remote backend instead of contending with production.
4. Exercise install/update rollback, stop/restart, occupied ports, missing
   models, unavailable backends, and descendant cleanup through the shared
   supervisor. Repeat on Linux x64; Spark does not cover that architecture.
5. Add a GUI-free build/run test for the daemon, then desktop Linux packaging,
   then macOS Metal using the same manifest/platform contract.
