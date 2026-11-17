# update feeds

these are the files installed copies of gary4local read to find out about a new
version. GitHub Pages serves them from `main`, so whatever is committed here on
`main` is live.

```text
gary4local/                 Windows: stable.json, native-stable.json, preview.json, native-preview.json
gary4local/linux-x86_64/    Linux amd64: preview.json, native-preview.json
gary4local/linux-aarch64/   Linux arm64: preview.json, native-preview.json
gary4local-rocm/            the AMD tester build: preview.json, native-preview.json
```

`<channel>.json` drives the browser-download fallback and `native-<channel>.json`
the signed in-app install. both are generated from the signed installer by
`control-center/src-tauri/scripts/generate_update_feeds.ps1` and checked by
`validate_update_feeds.ps1`. don't edit them by hand.

how to publish them is in [releasing gary4local](../releasing.md). what users see,
and how to point a test build at a different feed, is in
[the auto-updater doc](../auto-updater.md).
