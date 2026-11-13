# Experimental headless SA3 host

`gary4local-server` builds without Tauri, GTK, WebKit, tray libraries, Node or a
display. The desktop application now uses this crate's native launch types,
ZIP extraction, process-lifetime controls and SA3 HTTP adapter. Desktop LoRA
catalog validation, prompt preparation and decoder correction connect through
the adapter's registry interface and retain their existing implementations.

This first host manages one explicitly configured SA3 service. Its local JSON
configuration is separate from desktop profiles and migration records. It
installs pinned core/backend ZIPs with SHA256 verification, checks a real GPU,
loads the selected model, and exposes the existing Gary client contract through
the shared Rust adapter. It does not download models.

## Build and run

Use Rust 1.89 or newer (1.95 is tested), including a native C linker:

```sh
cargo build --locked --release --manifest-path control-center/runtime-core/Cargo.toml
```

Build from the repository layout: the core embeds the canonical
`services/manifests/services.json` at compile time.

Copy `headless.example.json` to a file of your own. Set absolute runtime/model
paths and an unused client port. Omitting `core` and `backend` selects the
packages pinned in that manifest for the current architecture and requested GPU.
Linux previews provide Vulkan on x64 and ARM64. `install` explicitly downloads
the two HTTPS archives into the owned `downloads/` cache and verifies their
SHA256 values before extraction. Repeated installs reuse verified cached bytes.
`serve`, `status`, and `stop` do not download anything.

For development, `headless.local.example.json` shows the local archive form.
Set both paths and full verified SHA256 values; use a matching core and backend
from the same engine/build. Alternatively, `manifest` can name an absolute local
manifest file instead of the bundled pins; it cannot accompany local archives.
Model component
encodings must match weights already present: Spark's tested medium model uses
F16 DiT/T5 and F32 autoencoder. The sample paths are placeholders, not downloads.

```sh
gary4local-server install --config /absolute/path/headless.json
gary4local-server serve --config /absolute/path/headless.json
```

`serve` stays in the foreground. From another shell:

```sh
gary4local-server status --config /absolute/path/headless.json
gary4local-server stop --config /absolute/path/headless.json
```

Both the client adapter and private native server bind to **127.0.0.1**. Use SSH
forwarding for a remote client. The example client port is 28300; a fresh private
native port is selected for each session and recorded in `state.json`. Only
forward the client port. The adapter supports its
existing generation, loop, transform, continuation, polling, health and model
lifecycle routes. Initial headless operation has no prepared LoRA registry or
decoder-correction selection; named adapters are rejected. Desktop behavior
continues to use its validated registry. Training and migration are not exposed
by this first CLI.

`install` is explicit and model-free. It requires clean/tested native build
metadata for the current architecture and a real CUDA/Vulkan device from
`--props`; CPU fallback is refused. It stages and probes before placing an
immutable package-hash directory in `runtimes/`. A failed candidate leaves older
directories intact. The installed identity includes both archive hashes, so a
new preview of the same engine version has a separate directory. Headless
packages currently require system GPU libraries rather than shared runtime
packs. CUDA smoke packages built for GB10 need
compatible system CUDA user libraries and are not portable CUDA releases.

`serve` initializes the configured model before reporting that it is running.
Missing components, occupied ports and incompatible controls fail startup and
release the owned child. It records native/supervisor PIDs, backend and a session
identifier in `state.json`; `status` checks the owner lock rather than trusting
stale PIDs. `stop` submits a session-specific request without signalling an
arbitrary PID from disk. Ctrl-C and SIGTERM also stop the owned service. A stale
stop request cannot stop a new session. Logs are in `logs/sa3-native.log`.

## Storage and process ownership

Set `runtime_root` to an empty, dedicated directory for experiments. The host
refuses nonempty roots without its ownership marker and refuses symlinks within
owned data. Package extraction, logs, uploads, temp files and GPU caches remain
there. Existing model files are read by path. Each root has an OS-held owner
lock, preventing concurrent servers and installs. Keep config files outside the
owned root or create the root with `install` before placing files inside it.

If omitted, the root uses XDG data storage on Linux, Application Support on
macOS, or LOCALAPPDATA on Windows. Explicit roots avoid mixing experiments with
an existing installation. A complete desktop-profile migration is not performed.

Windows keeps the existing kill-on-close Job Object. Unix uses a separate process
group with bounded TERM/KILL shutdown, including descendants that ignore TERM.
Linux also kills the direct native child if its supervisor dies. For persistent
service operation, the supplied **example** per-user systemd unit uses
`KillMode=control-group` to cover the entire process tree. It is not installed or
enabled automatically; set its binary/config paths before using it.

For cleanup, stop the matching host, preserve config/logs/audio you want, inspect
disk usage, and remove its explicitly named runtime root. Build caches are
separate from service data. The host does not remove model sources or desktop
data.

## Validation

The existing SA3 translation/transport tests moved with the adapter. Boundary
tests cover root ownership, concurrent locking and corrupt archives. Unix tests
cover group shutdown. Windows tests retain Job Object enrollment and cleanup.
`.github/workflows/headless.yml` builds/tests the core on native x64/ARM64 Linux
inside the plain Rust image, and runs Windows desktop/core regression checks.

`smoke-tests/linux/headless_sa3.py` exercises installation, model-component
failures, GPU generation through the public adapter, concurrent/occupied-port
refusal, normal stop, restart, SIGTERM, native crash, supervisor crash and
recovery. It is Spark-specific, reads existing weights and keeps its evidence
under `/home/kev/gary4local-packaging/headless`. See the installer exploration
document for actual results; the script's presence alone is not a test result.
