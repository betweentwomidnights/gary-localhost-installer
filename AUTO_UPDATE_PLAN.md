# gary4local Auto-Update Plan

## Goal

Give `gary4local` a simple Windows desktop updater flow so the app can say "vX.X is live, wanna update?" and guide the user through the new installer.

## Recommendation

Do not use the model-serving docker-compose network as the source of truth for app updates.

The updater should be independent from Gary, Terry, Carey, Jerry, and Foundation health. If the backend cluster is sick, the user should still be able to learn about a fixed release.

The best first shape is:

1. publish the NSIS installer and checksum
2. publish a tiny version manifest JSON at a stable URL
3. let `gary4local` check that manifest on startup
4. if a newer version exists, show a dialog with update notes and install options

## Why This Is Easier Than It Looks

Unlike the plugin, `gary4local` is a normal desktop app.

That means it can eventually support a more direct update path:

- check for a new version
- download installer
- prompt for restart/install

So `gary4local` is actually the easier of the two update stories.

## Phase 1

- Add a startup update check with a short timeout.
- Fetch a small JSON manifest over HTTPS.
- Compare installed version to `latestVersion`.
- If newer, show:
  - current version
  - latest version
  - short release notes
  - `download update` button
  - `skip this version` button

## Phase 2

- Evaluate using Tauri's updater flow or a small custom downloader.
- Download the NSIS installer automatically.
- Verify checksum before launch.
- Prompt user to close/restart into installer.

## Manifest Shape

```json
{
  "channel": "stable",
  "latest_version": "0.1.1",
  "download_url": "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.1.1/gary4local_0.1.1_x64-setup.exe",
  "sha256": "4885d54acccd4116ab2058ad1228e5049373acde43ad3ea65cbb84a49c131b83",
  "published_at": "2026-04-02T00:00:00Z",
  "notes": [
    "Carey cover mode now uses turbo locally.",
    "Carey progress reporting is smoother.",
    "Carey log viewer scrolling is fixed."
  ]
}
```

## Hosting Options

Best first choice:

- GitHub Pages, Cloudflare Pages, or a tiny static endpoint on your site

Also fine:

- GitHub Releases API directly

Less ideal:

- a dedicated container in the inference docker-compose stack

That container is not impossible, but it is one more thing to deploy and keep alive for something a static JSON file already solves.

## gary4local Integration Notes

- Check once on launch and optionally from a manual "check for updates" action.
- Fail quietly if offline.
- Store:
  - last check time
  - skipped version
  - optional auto-check toggle
- Keep the release prompt separate from service health so update UX does not depend on local Python env state.

## Release Workflow

1. Build NSIS installer.
2. Generate `SHA256SUMS.txt`.
3. Publish GitHub release.
4. Update the version manifest JSON with the new version, URL, and checksum.
5. `gary4local` sees the update next launch and offers it.

## First Implementation Target

Build phase 1 only:

- lightweight manifest
- startup check
- release popup
- open-download-link flow

That gets the user-friendly update message in place before we take on fully managed installer handoff.
