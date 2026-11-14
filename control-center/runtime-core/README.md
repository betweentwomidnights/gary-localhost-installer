# Experimental headless SA3 host (CLI 0.3)

## Human terminal console

The new `gary` command opens a small interactive service console in a terminal.
It shows the configured SA3 service, its state and address. Use `/setup` once to
choose existing weights, GPU, precision and port, then `/doctor`, `/install` and
`/start`. Commands also work without the slash. `/stop` stops SA3 or cancels
startup; `/status` refreshes the table; `/logs` shows the native log path.
`/quit`, Ctrl+C, SIGTERM and end of input stop services started by this console.
A service already running in another session is left running on exit; an
explicit `/stop` can stop that configured session. No daemon is registered.

Linux preview archives include both binaries and `install-gary.sh`:

```sh
sh /absolute/path/to/extracted-preview/install-gary.sh
gary
```

The installer creates a small launcher in `~/.local/bin` without replacing a
different existing command or changing shell profiles. Keep the extracted
preview in place. If needed, add that bin directory to your shell's PATH as
printed by the installer. `--config /absolute/path/sa3.json` on the installer
pins a contained experiment config for every launch. `--bin-dir DIR` selects a
different command directory. On Windows, build `gary.exe` with Cargo and place
it in a directory already on PATH; no Windows command installer is supplied yet.

Outside the console, `gary start sa3`, `gary status`, `gary stop` and
`gary doctor` reuse the saved config. `start` is a foreground host; Ctrl+C stops
it. `--config FILE` overrides the default. `GARY4LOCAL_HEADLESS_CONFIG` selects
another absolute default file. Otherwise config uses XDG config storage on Linux,
Application Support on macOS or LOCALAPPDATA on Windows, below
`gary4local/headless/sa3.json`. Setup writes only after the user confirms Save,
never overwrites an existing config and never downloads weights or runtimes.
Opening Gary does not create storage. A nonterminal invocation never prompts.

This is a line-oriented console, not a full-screen terminal UI. It currently
manages SA3 only; other services and guided training are future additions.
Scripts can use `gary COMMAND --json` or the original `gary4local-server`.
`gary start --json` aliases `serve`, including its protocol command name.

## Shared core

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
`services/manifests/services.json` and `control-center/src-tauri/sa3-models.json`
at compile time. Neither introduces desktop runtime dependencies.

Create a config pointing to weights you already have (create its parent folder
first):

```sh
gary4local-server init --config /absolute/path/sa3.json --models-dir /absolute/path/models --port 28300
gary4local-server validate --config /absolute/path/sa3.json
gary4local-server doctor --config /absolute/path/sa3.json
```

`init` defaults to Vulkan0, medium F16 DiT/T5 and F32 autoencoder, with a dedicated
`sa3.data` root beside `sa3.json`. It writes only the config, reuses an identical
config, and refuses to overwrite a different one. Use `--runtime-root`, `--device`
or `--encoding` to change those choices. Existing example JSON files also work;
their machine-specific paths must be replaced. Configs reject unknown fields.

`validate` resolves config and pinned package choices without inspecting the GPU
or loading models. `doctor` checks ownership, installed runtime metadata,
model presence/size and current client-port availability without creating data,
downloading, executing binaries or probing the GPU. Its exit code is 4 when
blocked; this is expected before the first install. A successful preflight means
startup can be attempted, not that GPU execution or model contents were verified.

Omitting `core` and `backend` selects the
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
gary4local-server stop --config /absolute/path/headless.json --session SESSION_FROM_READY_OR_STATUS
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

## Agent workflow

Use `capabilities --json` for command/options, effects, available platform
packages and feature discovery. `schema --target config` prints the config JSON
Schema; `schema --target response` prints the machine-response schema. Append
`--json` to wrap either schema in a versioned result.

With `--json`, finite commands emit exactly one compact JSON result on stdout,
including failures. Every response contains `schema_version`, `kind`, `command`,
`event`, `ok`, `exit_code`, `data` and `error`. Version 1 allows additive fields;
errors contain a stable `code`, a diagnostic `message` and a suggested `hint`.
Progress and native logs stay on stderr or in the reported log file. Exit codes
are 0 success, 1 operation failure, 2 invalid arguments, 3 config/init failure,
and 4 doctor blockers.

`serve --json` remains foreground and emits newline-delimited JSON: a `ready`
event only after model loading, followed by a terminal result when it exits.
Capture the ready event's `data.session`, `data.public_url` and `data.log_path`.
From a second process, use `status --json` and `stop --session ID --json` with
the same config. A mismatched live session returns `session_mismatch` without
stopping it. Repeating stop after the owner exits succeeds. Agents must not
silently replace their captured session ID with that of a newer owner.

For compatibility, without `--json` the original install/status/stop commands
retain their unversioned JSON and serve prints only its legacy ready object.
New init/validate/doctor commands have readable summaries. See
[`docs/headless-agent-workflow.md`](../../docs/headless-agent-workflow.md) for a
complete shell recipe and retry rules. Training is not implemented in this CLI.
The same guide is bundled as `AGENT-WORKFLOW.md` in CLI archives.

## Storage and process ownership

Set `runtime_root` to an empty, dedicated directory for experiments. The host
refuses nonempty roots without its ownership marker and refuses symlinks within
owned data. Package extraction, logs, uploads, temp files and GPU caches remain
there. Existing model files are read by path. Each root has an OS-held owner
lock, preventing concurrent servers and installs. The separate readable
`owner.json` nonce lets status/stop verify session identity even on Windows,
where the locked file cannot be read through another handle. It is never a
substitute for the held OS lock. Keep config files outside the
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

`smoke-tests/linux/headless_agent_cli.py` checks the versioned agent protocol,
explicit manifest installation, real readiness, fenced stop, generation and
shutdown on the Spark, keeping a fresh owned root and evidence in that same
workspace. It reads existing weights without copying or downloading them.
