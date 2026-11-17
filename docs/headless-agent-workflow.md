# Headless CLI for agents

CLI 0.3 is an experimental single-SA3 supervisor using the desktop's shared
native runtime and HTTP adapter. Linux packages cover x64 and ARM64 Vulkan.
CUDA can use verified local archives with compatible system libraries; portable
published Linux CUDA packages are still separate work. CPU generation is refused.
There is no GUI dependency, interactive prompt or automatic model download.

That statement describes the explicit agent commands. The `gary` entry point
also offers a terminal-only human console with `/setup`, `/start`, `/stop` and
status display. It never opens prompts for redirected input/output or `--json`.
Use `gary COMMAND --json` for automation; its optional default config comes from
`GARY4LOCAL_HEADLESS_CONFIG` or the platform config directory. The original
`gary4local-server COMMAND --config FILE --json` remains unchanged. Interactive
training is still unsupported. See the runtime-core README for console setup.

## Discover before acting

```sh
gary4local-server capabilities --json
gary4local-server schema --target config --json
gary4local-server schema --target response --json
```

Read `data.commands`, option requirements, effects and `data.features` from
capabilities. Unsupported features currently include training, model downloads,
named LoRA registries and migration from desktop profiles. Do not invent commands
for them. Command discovery describes this binary, including its platform's
published packages. `--help` also works without a config, globally or per command.

## Prepare and run

This POSIX shell example uses existing weights and a dedicated working directory.
On Windows the same commands accept Windows paths; use PowerShell's own process
management. Set the model path and choose a free loopback port deliberately.

```sh
mkdir -p "$HOME/gary-headless"
cd "$HOME/gary-headless"
gary4local-server init --config "$PWD/sa3.json" \
  --models-dir /absolute/path/to/existing/models --port 28300 --json
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
holds ownership until shutdown. See the runtime-core README for package details.

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

## Storage and future training

Each dedicated root contains runtime archives, immutable package-hash runtime
directories, owner/state records, logs, temporary uploads and GPU caches. Models
are read from the external directory. Stop the matching session and preserve
wanted evidence before deleting an explicitly named experiment root. The CLI
does not remove models, register services or migrate desktop data. The supplied
systemd unit is an example and is never enabled automatically.

Training is the next layer after this foundation is validated. It should use
the same training implementation as desktop, with explicit dataset/output paths,
owned job IDs, discoverable schemas, progress, cancellation and checkpoint
reporting. This release makes no training or resume guarantee; discover
capabilities again when a future binary adds those operations.
