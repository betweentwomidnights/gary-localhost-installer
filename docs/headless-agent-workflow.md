# Headless CLI for agents

The CLI supervises the seven native desktop services. Each foreground host owns one service. SA3 uses the shared HTTP adapter. Other services expose the native HTTP contract on the configured client port. Linux packages cover x64 and ARM64 Vulkan.
CUDA can use verified local archives with compatible system libraries; portable
published Linux CUDA packages are still separate work. CPU generation is refused.
There is no GUI dependency, interactive prompt or automatic model download.

That statement describes the explicit agent commands. The `gary` entry point
also offers a terminal-only human console with `/setup`, `/start`, `/stop` and
status display. It never opens prompts for redirected input/output or `--json`.
Use `gary COMMAND --json` for automation; its optional default config comes from
`GARY4LOCAL_HEADLESS_CONFIG` or the platform config directory. The original
`gary4local-server COMMAND --config FILE --json` remains unchanged. Training is available in the interactive console. See the runtime-core README for setup.

For another service, run `gary init carey --json`, then `gary install carey --json`,
`gary download-models carey --json` and `gary start carey --json`. Use
`gary status carey --json` to inspect the session, and
`gary stop carey --session SESSION --json` to stop it. Named commands use
per-service configs. An explicit `--config FILE` names the actual service
config. A ready event on the six direct native services means the HTTP server
answered its health check. For scripted SA3 starts, the ready event also confirms
model initialization. Interactive `/start sa3` starts with the model unloaded.

## Discover before acting

```sh
gary4local-server capabilities --json
gary4local-server schema --target config --json
gary4local-server schema --target response --json
```

Read `data.commands`, option requirements, effects and `data.features` from
capabilities. The interactive console offers LoRA training and import for SA3
and Carey. Agents can inspect the exposed adapters through the service APIs.
Standard SA3/Carey profiles share desktop's LoRA library. For an explicit custom
config, pass `init --library-root desktop` or an absolute persistent library
path. Omitting it for a custom config keeps that library isolated. A per-service
cross-process lock prevents desktop and headless from concurrently using it.

Command discovery describes this binary and its platform's published packages. `--help` also works without a config, globally or per command.

## Prepare and run

For managed storage, omit `--models-dir` at init, inspect `models --config FILE --json`, and explicitly run `download-models --config FILE --json` before serve. Downloads verify the desktop catalog’s pinned hashes.

This POSIX shell example uses existing weights and a dedicated working directory.
On Windows the same commands accept Windows paths; use PowerShell's own process
management. Set the model path. SA3 defaults to the desktop client port 8006.
Use `--port PORT` to choose a different loopback port when needed.

```sh
mkdir -p "$HOME/gary-headless"
cd "$HOME/gary-headless"
gary4local-server init --config "$PWD/sa3.json" \
  --models-dir /absolute/path/to/existing/models --json
gary4local-server validate --config "$PWD/sa3.json" --json
gary4local-server doctor --config "$PWD/sa3.json" --json
# First doctor normally exits 4 because no runtime is installed.
# Resolve ownership/model/port blockers before continuing.
gary4local-server install --config "$PWD/sa3.json" --json
gary4local-server doctor --config "$PWD/sa3.json" --json
gary4local-server serve --config "$PWD/sa3.json" --json \
  >serve.ndjson 2>serve.stderr &
supervisor_pid=$!
```

`init` writes only the config. Its default runtime root is `sa3.data` beside
`sa3.json`. Configs and existing weights stay outside the owned runtime root.
No directory is adopted or populated until an explicit install/serve. Model
weights must already match the selected component encodings. The generated
medium config uses F16 DiT/T5 and F32 AE; edit deliberately if your weights differ.

`install` explicitly downloads only pinned runtime archives over HTTPS, verifies
their SHA256 checksums, validates build metadata and probes the selected GPU.
Local archive pairs need both absolute paths and complete SHA256 values.
Reinstalls reuse verified cached archives. A failed candidate preserves older
installed directories. `serve` never downloads: it loads the selected model and
holds ownership until shutdown. See [native runtime packages](native-runtime-packages.md)
for the package contract.

Read stdout incrementally as NDJSON. Wait for `kind: "event", event: "ready"`,
check `schema_version == 1` and `ok == true`, then capture `data.session` and
`data.public_url`. A PID or a listening native port alone is not readiness.
Before ready, an error result means startup failed. After ready, the terminal
result and process exit describe shutdown. Keep stderr separately so progress
cannot corrupt JSON parsing. If the supervisor is forcibly killed, the stream
may end without a terminal result; inspect status rather than assuming success.

From another process:

```sh
gary4local-server status --config /absolute/path/to/sa3.json --json
gary4local-server stop --config /absolute/path/to/sa3.json \
  --session CAPTURED_READY_SESSION --json
```

Use the shared public HTTP adapter for generation and polling. Both its client
listener and the private native server bind to 127.0.0.1. Forward only the client
port over SSH for remote access. No LAN listener or authentication service is
introduced. `status.data.state.session` identifies the current owned host and
`status.data.health` contains its health response. Do not treat stale PID data
as authority. Stop never signals an arbitrary PID from a state file.

## Protocol and retries

A finite `--json` command writes one result line even for errors:

```json
{"schema_version":1,"kind":"result","command":"stop","event":null,"ok":false,"exit_code":1,"data":null,"error":{"code":"session_mismatch","message":"Refusing to stop a different session","hint":"Inspect status before taking further action."}}
```

The example illustrates the fields; diagnostic messages can change. Branch on
`error.code` and exit status, not message text. Accept unknown additive fields;
reject unsupported schema versions. `data` depends on the command. The response
schema validates the envelope, not every command payload. Parse errors use
`command: "unknown"`. Without `--json`, legacy command output remains available.

| Exit | Meaning | Recommended response |
| --- | --- | --- |
| 0 | Success | Inspect data; a doctor warning is not GPU validation. |
| 1 | Operation failed | Inspect error/hint, status and reported logs before retrying. |
| 2 | Invalid arguments | Consult capabilities/help and correct the invocation. |
| 3 | Config/init failed | Correct the config/path/package choice deliberately. |
| 4 | Doctor found blockers | Follow checks and next_actions; no mutation occurred. |

- Discovery, validate, doctor and status can be repeated. Doctor inspects
  metadata and briefly tests the client port, without executing the native binary
  or hashing model contents. Port availability is a snapshot, not a reservation.
- An identical init is idempotent; a different existing config returns
  `config_exists`. Init never overwrites it or creates its parent directories.
- Install is explicit and root-exclusive. Do not automatically install merely
  because doctor proposes it. A serve/start failure does not trigger downloads.
- Stop with a captured session is safe to repeat after that session exits.
  A different active session returns `session_mismatch`; keep it running until
  explicitly asked to stop that new session. Concurrent install/start ownership
  can instead return `stop_failed`; wait and inspect status.
- A stale stop request is ignored by a new owner. Concurrent installs/servers
  on one root are refused. To run independent experiments, use distinct config,
  runtime root and client port, and account for shared GPU memory.

## Storage and training

`runtime_root` contains runtime packages, owner/state records, logs, temporary
uploads and GPU caches. Models use managed storage by default; `--models-dir`
selects an existing external folder. `library_root` controls LoRA catalogs and
training history separately. For a container, mount persistent storage for the
runtime/model data and library you want to retain.

Each runtime root has an OS-held owner lock. Status and stop use a session nonce
rather than trusting a saved PID. Keep the config outside its runtime root, and
use an empty directory for a new root. The supplied systemd unit is an example;
set its paths and enable it yourself if needed.

The JSON command API covers service and model management. Training, labeling,
LoRA import and checkpoint selection currently use the interactive `gary`
console. Dedicated agent training commands are follow-up work; consult
capabilities rather than assuming they exist.

## building from source

Use Rust 1.89 or newer (1.95 is tested) and a native C linker:

```sh
cargo build --locked --release --manifest-path control-center/runtime-core/Cargo.toml --bins
```

Build from the repository layout: the crate embeds the runtime manifest and
model catalogs at compile time. The headless binaries build without desktop
libraries or a display.
