# Phase 2 Release Guide

This is the maintainer checklist for shipping a `gary4local` release with:

- Phase 1 manifest notes and fallback browser download
- Phase 2 signed in-app install

Normal users should not need any updater env vars. Production builds use the baked-in stable feeds:

- `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/stable.json`
- `https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local/native-stable.json`

## One-Time Setup

1. Keep the updater private key outside the repo.
2. Keep the updater public key committed in:
   - `control-center/src-tauri/src/update.rs`
   - `control-center/src-tauri/tauri.updater.conf.json`
3. Make sure GitHub Pages is publishing from `main /docs`.

## Per Release

1. Bump the app version in all five spots. they must match or the build fails:
   - `control-center/package.json`
   - `control-center/package-lock.json` (two `gary4local` entries; leave dependency versions alone)
   - `control-center/src-tauri/Cargo.toml`
   - `control-center/src-tauri/Cargo.lock` (the `gary4local` package block only)
   - `control-center/src-tauri/tauri.conf.json`

   `cargo check` from `control-center/src-tauri` is the quickest way to confirm
   `Cargo.lock` still agrees with `Cargo.toml`.

2. Write the release notes in the repo, in two places:
   - add a `## vX.Y.Z` section at the top of `CHANGELOG.md`, ending with a
     `compatible with gary4juce vA.B.C.` line.
   - replace the headline `## vX.Y.Z` section near the top of `README.md` with
     the new one. the README carries only the current release; everything older
     lives in the changelog. this step is easy to forget — v0.1.19 shipped with
     the README still showing v0.1.18.

   both should match the voice already in `CHANGELOG.md` — lowercase headings,
   contractions, no marketing language.

   the changelog, the github release notes, and the updater feed notes are three
   different registers, not one text pasted three times. see **the three tiers of
   release notes** in `kevs_docs_style.md` for which gets prose, which gets
   bullets, and what always earns a line in all three however short.

3. Build the signed NSIS updater artifact:

```powershell
cd C:\path\to\backend-installer\control-center
$env:TAURI_SIGNING_PRIVATE_KEY="C:\path\to\gary4local-updater.key"
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD="your real passphrase"
npm.cmd run tauri build -- --config src-tauri/tauri.updater.conf.json
```

   On the release machine there's a gitignored `LOCAL_SIGNING_NOTES.md` at the
   repo root with PowerShell helpers that wrap this step, so the key paths and
   passphrase never have to be typed by hand. It's intentionally not in the
   repo — if it's missing, this command is still the source of truth and the
   notes can be rebuilt around it.

4. Create the GitHub release tag, for example `v0.1.3`.
5. Upload both files from `control-center\src-tauri\target\release\bundle\nsis\`:
   - `gary4local_<version>_x64-setup.exe`
   - `gary4local_<version>_x64-setup.exe.sig`
6. Keep the gary4juce compatibility line appropriate for where it appears:
   - In the GitHub release notes, link to the current recommended gary4juce
     release tag.
   - In the updater feed notes, use a short plain-text line such as
     `Compatible with gary4juce v4.0.2.` Do not include a URL. The current
     update prompt does not provide clickable links, and the raw URL wastes
     limited UI space.
7. Generate both updater feeds from the exact built installer and signature. The
   output directory is derived from the installer filename — `gary4local_…`
   writes `docs/updates/gary4local/`, `gary4local-rocm_…` writes
   `docs/updates/gary4local-rocm/` — so leave `-OutputDir` off and check the
   `Product:` line the script prints. Passing an `-OutputDir` that disagrees
   with the installer is an error rather than a silent overwrite:

```powershell
cd C:\path\to\backend-installer
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
  -Version "0.1.3" `
  -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.1.3/gary4local_0.1.3_x64-setup.exe" `
  -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe" `
  -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe.sig" `
  -Channel "stable" `
  -NotesText "Release note one.||Release note two."
```

8. Review the generated files:
   - `docs/updates/gary4local/stable.json`
   - `docs/updates/gary4local/native-stable.json`
   - the generator now runs `validate_update_feeds.ps1` automatically. don't
     publish unless it prints `PASS`. this checks the raw timestamp strings,
     the intended channel/version, agreement between both feeds, and the hash
     and signature against the exact local installer and `.sig` file.
   - if you edit or copy a feed afterwards, run the validator again:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\validate_update_feeds.ps1 `
  -FeedDirectory "docs/updates/gary4local" -Channel stable -ExpectedVersion "0.1.3" `
  -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.3_x64-setup.exe"
if ($LASTEXITCODE -ne 0) { throw "Feed validation failed; don't publish." }
```

9. Commit those feed changes to `main` and push.
10. Wait for GitHub Pages to publish the updated JSON.
11. validate both live feeds at the URLs the app uses, without a cache-busting
    query. checking only the browser-download feed won't catch a broken native
    feed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\validate_update_feeds.ps1 `
  -BaseUrl "https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local" `
  -Channel stable -ExpectedVersion "0.1.3"
if ($LASTEXITCODE -ne 0) { throw "Live feed validation failed; the updater isn't verified." }
```

12. launch an older installed app configured for that channel and verify it
    offers `install update`. `download update` only proves the fallback path
    works. the validator checks feed metadata; it doesn't replace signature
    verification during an actual install. if the UI can't be tested, record
    that limitation rather than claiming the in-app update was verified.

## timestamps and the download-only fallback

leave `-PublishedAt` off to use the generator's current UTC time. if you want
the GitHub release's exact publication time, obtain it as a string with `--jq`:

```powershell
$releaseTag = "v0.1.3"
$publishedAt = gh release view $releaseTag --json publishedAt --jq '.publishedAt'
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($publishedAt) -or $publishedAt -eq 'null') {
    throw "Couldn't read the published release timestamp."
}
```

then pass `-PublishedAt $publishedAt` to the generator. don't construct the feed
JSON by hand or round-trip timestamps through `ConvertFrom-Json` and string
interpolation: PowerShell can turn them into locale-formatted dates. the
generator normalizes valid input to UTC RFC 3339 and refuses invalid input.
`2026-10-06T07:13:17Z` is valid; `10/06/2026 07:13:17` isn't valid in the feed.

we hit this on the `0.4.0-rocm.2` preview. the app offered `download update`
because Tauri rejected `pub_date`. the log showed:

```text
failed to deserialize update response: invalid value for `pub_date`: the 'year' component could not be parsed
Native updater check unavailable: Native updater check failed: invalid value for `pub_date`
```

for this error, regenerate the feeds using the already signed installer and
signature, validate them, and publish the correction on `main`. no rebuild,
replacement installer, or new release tag is needed. wait for Pages to deploy,
validate both live feeds, then check updates again. Pages/CDN caching can keep
the old feed visible for several minutes. for other download-only failures,
inspect the app log's `Native updater check unavailable` message before guessing
at the cause.

after changing either feed script, run the regression check in Windows
PowerShell before publishing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File smoke-tests\test_update_feed_dates.ps1
if ($LASTEXITCODE -ne 0) { throw "Update feed regression check failed." }
```

## Preview Testing

Use the same helper for preview releases, but switch the channel:

```powershell
cd C:\path\to\backend-installer
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
  -Version "0.1.2-preview.1" `
  -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.1.2-preview.1/gary4local_0.1.2-preview.1_x64-setup.exe" `
  -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.2-preview.1_x64-setup.exe" `
  -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.1.2-preview.1_x64-setup.exe.sig" `
  -Channel "preview" `
  -NotesText "Phase 2 updater preview"
```

That writes:

- `docs/updates/gary4local/preview.json`
- `docs/updates/gary4local/native-preview.json`

Preview apps can point at those feeds with runtime env overrides.
use the same local and live validation commands from steps 8 and 11 with
`-Channel preview` and the preview's exact version. confirm the tested app is
configured for that feed; a normal stable-channel app won't read it.

## ROCm Preview Releases

The AMD tester build is a different product (`gary4local-rocm`) living on
`feature/rocm-custom-runtime-storage`, and it is deliberately lighter than the
flow above. The differences are easy to get wrong, so they are spelled out here.

**It is not the same as a stable release:**

- No `CHANGELOG.md` entry and no `README.md` headline. Those track mainline
  gary4local releases only. A ROCm release commit touches the five version
  files and nothing else.
- The version is `0.2.1-rocm.N` — bump `N` for each tester build.
- The GitHub release is marked **prerelease**.

**The version bump lives on the ROCm branch. The feed lives on `main`.**

This trips people up every time. GitHub Pages serves `docs/updates/` from
`main`, so the feed the tester's *check updates* reads is `main`'s copy. The
ROCm branch has its own stale copies of those JSON files; ignore them, they are
not served. Both the `Release gary4local-rocm …` commit (ROCm branch) and the
`Update ROCm preview feed for …` commit (`main`) are required.

Order:

Before the version bump, merge the current `main` into the ROCm branch so the
shared services inherit the same fixes. retain the AMD product identity,
dependency pins and runtime profiles when resolving conflicts. check the final
`services/manifests/services.json` for exactly one entry per service ID, and
compare Yuey's entire `native` block with main. a clean automatic merge can
still leave two Yuey entries: during rc.4, that would have kept the AMD package
on v0.2.1 despite the new v0.2.2 entry. verify the staged manifest matches the
source after the signed build too.

1. Bump the five version files on `feature/rocm-custom-runtime-storage`, commit,
   and **push the branch**.
2. Build the signed installer.
3. Create the GitHub release. Push the branch *first* — `gh release create
   --target <branch>` resolves the branch name on the server, so with an
   unpushed branch the tag silently lands on whatever the remote tip still is.
   Prefer `--target <full-sha>` and verify afterwards:

   ```powershell
   git ls-remote --tags origin v0.2.1-rocm.N   # must equal the release commit
   ```

4. Generate the feeds. Run this from `main`, and note that `-OutputDir` is
   derived from the installer filename, so a `gary4local-rocm_…` installer can
   only ever write `docs/updates/gary4local-rocm/`:

   ```powershell
   git checkout main
   powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
     -Version "0.2.1-rocm.N" `
     -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.2.1-rocm.N/gary4local-rocm_0.2.1-rocm.N_x64-setup.exe" `
     -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local-rocm_0.2.1-rocm.N_x64-setup.exe" `
     -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local-rocm_0.2.1-rocm.N_x64-setup.exe.sig" `
     -Channel "preview"
   ```

   The script prints `Product: gary4local-rocm -> docs\updates\gary4local-rocm`.
   If it says `gary4local`, stop: the wrong installer was passed.
   it must also print `PASS` from the automatic feed validator before you
   commit the files. timestamp handling is the same as for regular releases.

5. Commit the two changed feed files to `main` and push. The release assets must
   already be uploaded at this point, or the tester's updater will offer the new
   version and 404 on the download.
6. validate both live feeds after Pages rebuilds. substitute the exact release
   version for `0.2.1-rocm.N`; the validator must print `PASS`:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\validate_update_feeds.ps1 `
     -BaseUrl "https://betweentwomidnights.github.io/gary-localhost-installer/updates/gary4local-rocm" `
     -Channel preview -ExpectedVersion "0.2.1-rocm.N"
   if ($LASTEXITCODE -ne 0) { throw "ROCm live feed validation failed." }
   ```

   then check updates in an older installed ROCm app and confirm it offers
   `install update`, as in step 12 above.

   If the feed still shows the old version well after the push, check whether
   the Pages build actually succeeded — during a GitHub incident it can fail
   with a bare "Page build failed." and leave the site serving the previous
   deploy:

   ```powershell
   gh api repos/betweentwomidnights/gary-localhost-installer/pages/builds `
     --jq '.[0:3][] | .status + "  " + .created_at + "  " + (.error.message // "")'
   ```

   Pages here is the legacy builder on `main:/docs`, so a rebuild can be forced
   without an empty commit:

   ```powershell
   gh api -X POST repos/betweentwomidnights/gary-localhost-installer/pages/builds
   ```

   The release assets are served by a different system and work immediately, so
   a direct download link is always available while Pages catches up.

## Source Builds

Source builders can still disable the entire updater UI and backend check path:

```powershell
$env:VITE_ENABLE_APP_UPDATER='0'
npm.cmd run tauri build
Remove-Item Env:VITE_ENABLE_APP_UPDATER
```

That remains the recommended opt-out for forks and local-only builds that should not advertise public releases.
