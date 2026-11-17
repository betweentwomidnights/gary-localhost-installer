# gary4local-core

Runtime building blocks shared by the gary4local desktop app on Windows and
Linux, built without Tauri, GTK, WebKit or tray libraries:

- `manifest`: the service manifest schema and resolution of shared native
  bundles (`services/manifests/services.json`).
- `download`: resumable downloads verified against pinned SHA-256 values, and
  `SHA256SUMS` parsing.
- `runtime`: native package installation, ZIP extraction, platform keys and
  launch/process-lifetime primitives.
- `sa3_adapter`: the SA3 HTTP adapter that serves the Gary client contract in
  front of `sa3-server`.
- `workload_job`: native workload jobs.

The desktop app (`control-center/src-tauri`) depends on this crate by path.
`.github/workflows/shared-core.yml` tests it on Linux x64/ARM64 without desktop
dependencies, and runs the Windows and Linux desktop regressions against it.

A GUI-free host built on this crate is in progress separately.
