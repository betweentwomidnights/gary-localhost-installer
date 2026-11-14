These smoke tests render the production Svelte components with Tauri's IPC/event
mocks. Catalogs come from the shipped model manifests. No runtime, model download,
service process, or user settings are touched.

From `control-center` with Node 24:

```sh
npm ci
npx playwright install --with-deps chromium webkit
npm run test:precision
npm run test:ui
```

The suite checks precision browsing versus activation, download targeting,
progress/retry, readiness gates, independent SA3 training precision, control
contrast, and storage layout at 800px. Chromium and WebKit run in headless mode.
Screenshots from failures are saved under the repository's ignored
`tmp/ui-test-results` directory.

Start tests also cover a fresh native install with one precision (no Use click),
multiple precisions requiring an explicit first choice, visible blockers,
pending/failed activation, and unchanged legacy Python starts. Backend tests in
`native_start` check verification-before-launch and generation readiness against
the shipped catalogs without loading models or running a GPU job.

These checks cover UI behavior; real GPU generation and desktop integration
still need the packaged application on the target machine.
