# Linux and headless exploration

Tracks [issue #13](https://github.com/betweentwomidnights/gary-localhost-installer/issues/13).
Branch: `feature/linux-headless`. This contains the first portability pass and
an experimental headless SA3 host; it is not a Linux desktop release.

Merged: [sa3.cpp packaging PR #95](https://github.com/betweentwomidnights/sa3.cpp/pull/95)
adds upstream Linux core/CUDA/Vulkan packages. On Spark, both ARM64 package sets
passed 52 CTests, extracted-package controls and real GB10 GPU probes, then
generated two-second audio through the native CLI using existing weights.
The work is contained in `/home/kev/gary4local-packaging`, with a storage ledger
and cleanup inventory; its initial footprint is about 1.3 GiB. These are engine
package checks. This application's Linux installation, compatibility adapters,
and headless supervisor still need integration work, and no Linux runtime URLs
have been pinned here. CPU generation is not a target; CPU libraries/tests
support the CUDA/Vulkan runtime paths.

### Sibling engine packages

The same upstream contract is now proposed in three draft PRs. Each preserves
the pinned ggml revision and packages its own tools, libraries, licenses and
data; ACE-Step retains `GGML_MAX_NAME=128` throughout. Never substitute another
engine's backend libraries.

| Engine | Packaging PR | No-tag Linux release CI | Native Spark CTests |
| --- | --- | --- | --- |
| Yuey | [#47](https://github.com/betweentwomidnights/yuey.cpp/pull/47) | [43304164829](https://github.com/betweentwomidnights/yuey.cpp/actions/runs/43304164829) | 25/25 |
| Audiocraft | [#15](https://github.com/betweentwomidnights/audiocraft.cpp/pull/15) | [43304681141](https://github.com/betweentwomidnights/audiocraft.cpp/actions/runs/43304681141) | 24/24 |
| ACE-Step | [#20](https://github.com/betweentwomidnights/acestep.cpp/pull/20) | [43304164014](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/43304164014) | 22/22 |

All six native x64/ARM64 Vulkan CI packaging jobs passed. The ARM64 CI artifacts
were downloaded, checked against GitHub's artifact SHA256 and the inner archive
checksums, and exercised on GB10. Separately built CUDA packages target this
Spark with `--cuda-arch native`; they are GPU smoke builds, not portable CUDA
releases. Source commits are Yuey `b7dcf3a`, Audiocraft `cd85b85`, and ACE-Step
`bd6a3067`; all retain ggml `e3db5210fe3ecdaa4d65b4c93d515f2cf15524b7`.

Each backend produced finite, nonzero audio: a short Yuey render from a supplied
ABC score, one-second MusicGen generation, a two-step MelodyFlow edit, and a
two-step ACE-Step DiT-only synth. Logs confirm CUDA0/Vulkan0 computation. These
checks do not cover Yuey's symbolic planner, ACE-Step LM generation, training,
adapters, perceptual quality, full-duration workloads or numerical parity.
Hosted x64 builds have packaging evidence, not hardware inference evidence.
Four extracted CI servers also returned healthy loopback HTTP responses;
all test processes were stopped and their private ports verified closed.

The dedicated Spark workspace retains `ENGINE-VALIDATION.json`, per-case
`validation/*/RESULT.json`, `BUILD-INFO.json`, logs and short WAVs. Nine temporary
Audiocraft/ACE-Step model copies were SHA256-verified, then removed as requested:
6,172,945,024 bytes freed, with sources/hashes retained in
`test-models/MODEL-COPIES.json` and cleanup evidence in
`validation/MODEL-CLEANUP.json`. Original Windows weights and existing Spark
model stores were not cleanup targets. The workspace measured 3.73 GiB after
cleanup, versus 9.48 GiB before. All work remains under the dedicated root;
the older 6.2 GiB application exploration checkout is separate.

The PR-triggered regression/release checks run separately from the successful
Linux-only dispatches above. All three sibling packaging PRs are now merged
(Yuey `4a5d8b0`, Audiocraft `c45f62b`, ACE-Step `5162473`). ACE-Step's cancelled
macOS checks never acquired hosted runners; its release packages and CUDA
compile passed. No tag or release has been published, and Linux publishing
remains disabled. The desktop runtime manifest still requires Linux assets.

### First headless SA3 pilot

`control-center/runtime-core` now builds `gary4local-server` independently of
Tauri. The desktop app uses its shared native launch types, ZIP extractor,
process-lifetime controls and SA3 compatibility adapter. Desktop registry/prompt
and decoder-correction behavior is supplied through an interface to the
existing implementations. The complete desktop service manager has not yet
been extracted; other services and desktop settings remain follow-up work.

The CLI has `install`, `serve`, `status` and `stop` commands with an explicit
JSON config. For now, installation uses existing local core/backend archives
and their verified SHA256 values, separate from production desktop profiles.
It checks clean/tested package metadata, platform and a real selected GPU,
then loads the specified model before announcing readiness. CPU fallback is
refused. Both listeners are loopback-only; the client port stays fixed while
each session gets a fresh private native port. No systemd unit was enabled.

Spark validation used the plain `rust:1.95-bookworm` image without desktop
development packages. Its ARM64 binary links libc/libm/libgcc, with no GTK,
WebKit or Tauri dependencies. Fourteen shared-core tests passed on Linux;
fifteen passed on Windows, including its existing Job Object tests. Windows
desktop checks also passed: 22 native-runtime tests and 28 service-manager
tests (one model-dependent test ignored in each suite), plus all-target checks.
The SA3 capability/profile tests passed with their model-dependent case ignored.

Both the local CUDA-smoke package and the ARM64 CI Vulkan package passed the
full headless exercise using existing medium F16 DiT/T5 and F32 autoencoder
weights. Through the public compatibility adapter, each returned exactly
88,200 nonzero stereo PCM16 frames at 44.1 kHz (two seconds). Native logs confirm
CUDA0/Vulkan0 on GB10. This establishes execution, not quality or backend parity.

The exercise also passed bad-checksum refusal, idempotent install, missing model
component rejection before readiness, occupied-port refusal, concurrent server
and install refusal, normal stop, restart, SIGTERM, native crash, supervisor
SIGKILL, and recovery. All owned test processes stopped and ports closed; the
existing weights' sizes/mtimes were unchanged. Linux process groups receive
bounded TERM/KILL shutdown and a parent-death signal for the direct native
child. The example systemd unit adds cgroup ownership for persistent operation.

Early tests exposed an incorrect autoencoder selection and a mismatch between
the private C++ listener's socket-reuse policy and a fixed-port reservation.
Startup now verifies model initialization, and the private port is selected
per session. Crash checks allow a bounded interval for the kernel to close
sockets. The earlier failed-case logs remain available for diagnosis.

Reproduction commands/configuration are in
[the headless README](../control-center/runtime-core/README.md).
`smoke-tests/linux/headless_sa3.py` runs the contained Spark tests. Evidence is
under `/home/kev/gary4local-packaging/headless/sa3-{cuda,vulkan}-lease/evidence`.
This host currently has no headless LoRA registry, decoder correction, training,
model downloading, desktop-profile migration, or manifest-based downloads.
Windows/macOS headless GPU execution and Linux desktop GUI startup remain
separate validation work. The new CI workflow tests GUI-free x64/ARM64 Linux
builds and desktop/shared-core regressions without publishing releases.

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
| Runtime distribution | The desktop manifest pins Windows x64 packages only; upstream Linux builders and the local-archive headless path are validated. | Publish versioned experimental assets with checksums, then pin them by platform; audit macOS Metal packages. |
| Executable names | Shared launch/installation resolves the native server name; other trainer/helper paths still use `.exe`. | Finish platform-specific tool names across training, migration, and probes. |
| Shared libraries | Shared CUDA packs currently contain Windows DLLs and use `PATH`. | Define Linux `.so` dependencies, `$ORIGIN` RPATH, and CUDA toolkit/driver compatibility; audit macOS dylibs/install names separately. A Windows CUDA pack is not reusable on Spark. |
| Storage | Headless uses an explicit or platform data root; desktop profiles still have Windows defaults. | Finish XDG/Application Support handling for desktop profiles and preserve explicit runtime-root overrides. |
| Process ownership | Shared Unix process groups and headless normal/crash cleanup are tested; persistent service installation is still explicit. | Validate the example systemd cgroup unit and more complex Python/training descendants separately. |
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
