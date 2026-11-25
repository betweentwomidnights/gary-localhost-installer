# releasing gary4local

this is the maintainer checklist for shipping a gary4local release: the version
bump, the notes, the signed Windows installer, the signed Linux packages, the
GitHub release, and the update feeds that make installed apps offer it.

there's one app, `gary4local`, on every platform. the AMD tester's build,
`gary4local-rocm`, is a separate product on its own branch, and it has its own
section at the end.

the release machine also has a gitignored `LOCAL_SIGNING_NOTES.md` at the repo
root. it names the key's location and the PowerShell helpers that wrap the
signing steps below, so nobody types a key path or passphrase by hand. if it's
missing, the commands in this file are still the source of truth and the notes
can be rebuilt around them.

## where updates come from

GitHub Pages serves `main:/docs`, so a feed is only live once it's committed to
`main`. each feed folder holds two files per channel:

- `<channel>.json` is the phase 1 manifest: version, notes, and a browser
  download link. it's the fallback, and it's what makes the app say
  `download update`.
- `native-<channel>.json` is the signed feed for the in-app install. it's what
  makes the app say `install update`.

| feed folder | channels | who reads it |
|---|---|---|
| `docs/updates/gary4local/` | `stable`, `preview` | Windows builds |
| `docs/updates/gary4local/linux-x86_64/` | `preview` | Linux amd64 builds |
| `docs/updates/gary4local/linux-aarch64/` | `preview` | Linux arm64 builds (DGX Spark) |
| `docs/updates/gary4local/macos-aarch64/` | `preview` | Apple Silicon Mac builds |
| `docs/updates/gary4local/headless-linux-x86_64/` | `preview` | `gary update` in the x64 headless archive |
| `docs/updates/gary4local/headless-linux-aarch64/` | `preview` | `gary update` in the arm64 headless archive (DGX Spark) |
| `docs/updates/gary4local-rocm/` | `preview` | the `gary4local-rocm` tester build |

a build's channel is fixed when it's built: `stable` by default, `preview` when
`GARY4LOCAL_UPDATE_CHANNEL=preview` is set, which the Linux and macOS CI builds
do. each platform and architecture gets its own folder because a feed carries
one version and one download, so Linux and macOS can move on their own schedule
without offering Windows a version it can't install.

## one-time setup

1. keep the updater private key outside the repo, on the release machine only.
   it never goes to GitHub, which is why CI-built Linux packages get signed
   after download rather than in CI.
2. keep the updater public key committed in
   `control-center/src-tauri/src/update.rs` and
   `control-center/src-tauri/tauri.updater.conf.json`.
3. make sure GitHub Pages is publishing from `main /docs`.

## per release

macOS Developer ID signing and notarization run in GitHub CI using the upstream
Apple signing secrets. See [macOS desktop](macos-desktop.md) for the local M4
validation and CI artifact steps. This does not use the updater key.

### 1. bump the version

bump it in all five spots. they have to match or the build fails:

- `control-center/package.json`
- `control-center/package-lock.json` (the two `gary4local` entries; leave
  dependency versions alone)
- `control-center/src-tauri/Cargo.toml`
- `control-center/src-tauri/Cargo.lock` (the `gary4local` package block only)
- `control-center/src-tauri/tauri.conf.json`

`cargo check` from `control-center/src-tauri` is the quickest way to confirm
`Cargo.lock` still agrees with `Cargo.toml`.

### 2. write the notes

in the repo, in two places:

- add a `## vX.Y.Z - date` section at the top of `CHANGELOG.md`, ending with a
  `compatible with gary4juce vA.B.C.` line.
- replace the headline `## update` section near the top of `README.md`. the
  README carries only the current release; everything older lives in the
  changelog. this step is easy to forget. v0.1.19 shipped with the README still
  showing v0.1.18.

the changelog, the GitHub release notes, and the updater feed notes are three
different registers, not one text pasted three times. see **the three tiers of
release notes** in `kevs_docs_style.md` for which gets prose, which gets
bullets, and what always earns a line in all three however short. the gary4juce
compatibility line goes in all three: a link in the release notes, plain text
in the feed (the update popup can't open links, so a URL just wastes space).

commit and push the bump and the notes before building. the installers and the
release tag should all come from the same pushed commit.

### 3. build the signed Windows installer

```powershell
cd control-center
$env:TAURI_SIGNING_PRIVATE_KEY="C:\path\to\gary4local-updater.key"
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD="your real passphrase"
npm.cmd run tauri build -- --config src-tauri/tauri.updater.conf.json
```

that writes `gary4local_<version>_x64-setup.exe` and its `.sig` to
`control-center\src-tauri\target\release\bundle\nsis\`. on the release machine,
`Build-Gary4LocalSigned` does all of this (see `LOCAL_SIGNING_NOTES.md`).

### 4. build and sign the Linux packages

the `.deb`s are built by CI, not on the release machine. run the
`Linux desktop` workflow on `main` once the release commit is pushed:

```powershell
gh workflow run linux-desktop.yml --repo betweentwomidnights/gary-localhost-installer --ref main
```

when it finishes, check the run's `headSha` is the release commit, then
download its two package artifacts, `gary4local-desktop-linux-x64` and
`gary4local-desktop-linux-arm64`. the `-validation-` artifacts are CI evidence,
not release assets. each package artifact holds the `.deb`, a
`SHA256SUMS-linux-x64` or `SHA256SUMS-linux-arm64`, `BUILD-INFO.json`, and the
startup check's log and screenshot. before signing anything:

- the artifact zip matches the digest GitHub lists for it.
- the `.deb` matches its `SHA256SUMS-linux-*` line.
- `BUILD-INFO.json` says the release version, the release commit,
  `updater_enabled: true` and `update_channel: preview`.
- `STARTUP.json` says `installed_package_started: true`. CI starts the
  installed package on a virtual display with no GPU, so this proves it
  installs and draws, not that it generates.

then sign each `.deb` with the same updater key as the Windows installer. run
the signer from `control-center`, and pass the key file with `-f`:

```powershell
cd control-center
npx tauri signer sign -f C:\path\to\gary4local-updater.key -p "your real passphrase" path\to\gary4local_<version>_amd64.deb
npx tauri signer sign -f C:\path\to\gary4local-updater.key -p "your real passphrase" path\to\gary4local_<version>_arm64.deb
```

one trap: `tauri signer sign` reads `TAURI_SIGNING_PRIVATE_KEY` as the key's
*text*, while `tauri build` accepts it as a path. if your shell has it set to
the key's path from the Windows build, the signer fails. clear it for the
signer, or use the helper in `LOCAL_SIGNING_NOTES.md`, which does that for
you. each `.deb` should end up with a non-empty `.deb.sig` beside it.

### headless archives

also dispatch `headless.yml` on the exact release commit pushed to `main`.
its two Linux artifacts are `gary4local-server-linux-x64-experimental` and
`gary4local-server-linux-arm64-experimental`. each contains the corresponding
`.tar.gz` and `SHA256SUMS-linux-*`. check the artifact ZIP digest, the tarball
checksum, and the `BUILD-INFO.json` inside the tarball before uploading.
`commit` must be the release SHA and `platform` must match the architecture.

the headless version comes from `runtime-core/Cargo.toml`, independent of the
desktop app's version, and it's what `gary update` compares. when a release
changes anything in the headless archive, bump it in step 1 too, as plain
`MAJOR.MINOR.PATCH`: `runtime-core/Cargo.toml`, its `Cargo.lock`, and the
`gary4local-core` block in `src-tauri/Cargo.lock`. an unbumped archive would
never reach anyone, so the feed generator refuses a different archive under the
version its feed already carries. if the headless archive didn't change, skip
its feeds for that release.

then sign both tarballs with the updater key, exactly like the `.deb`s in
step 4 (same `tauri signer sign` command, same `TAURI_SIGNING_PRIVATE_KEY`
trap). each should get a non-empty `.tar.gz.sig` beside it.

upload both tarballs and their `.sig`s on the main release. preserve their
names, and upload copies of their checksum files named
`SHA256SUMS-headless-linux-x64` and `SHA256SUMS-headless-linux-arm64`, so they
don't collide with desktop's checksum assets. the checksum contents remain
unchanged. keep the downloaded ZIPs and provenance in the untracked per-release
folder.

release notes should separate the desktop `.deb`s from the headless archives.
for a new install, show extraction, `install-gary.sh`, PATH setup and `gary`.
for an existing one, say `gary update`. while people may still have an archive
from v0.7.6 or earlier, which can't update itself, also show the one-time
`install-gary.sh --replace` from the runtime-core README's updating section.

### 5. create the GitHub release

target the release commit by its full SHA, not a branch name: `gh release create
--target <branch>` resolves the branch on the server, so an unpushed or moved
branch quietly tags the wrong commit. create it as a draft, upload, compare
digests, then publish:

- `gary4local_<version>_x64-setup.exe` and `.sig`
- `gary4local_<version>_amd64.deb` and `.sig`
- `gary4local_<version>_arm64.deb` and `.sig`
- `SHA256SUMS-linux-x64` and `SHA256SUMS-linux-arm64`, straight from CI
- both headless `.tar.gz` archives, their `.sig`s, and their renamed `SHA256SUMS-headless-linux-*` copies

after uploading, each asset's `digest` from
`gh release view <tag> --json assets` should equal `sha256:` plus the local
file's hash. then publish it with `--latest`, and check the tag landed where you
meant it to:

```powershell
git ls-remote --tags origin v<version>   # must equal the release commit
```

### 5b. sign the Mac update archive

publishing also starts the `macOS desktop preview` workflow on the tag. it
refuses a tag that doesn't match `package.json`, then builds, signs and
notarizes the Mac app and attaches five assets to the release about 15 minutes
later: `gary4local_<version>_aarch64.dmg`,
`gary4local_<version>_macos-arm64.app.zip`,
`gary4local_<version>_aarch64.app.tar.gz`, `SHA256SUMS-macos-arm64` and
`BUILD-INFO-macos-arm64.json`. check that run went green before announcing the
Mac build.

the `.app.tar.gz` is the in-app update: the stapled app, packed the way
`tauri-plugin-updater` unpacks it. CI never has the updater key, so like the
`.deb`s it's signed here after download:

- the downloaded files match their `digest` in `gh release view <tag> --json assets`
  and their `SHA256SUMS-macos-arm64` lines.
- `BUILD-INFO-macos-arm64.json` says the release version and commit,
  `signed` and `notarized` true, `updater_enabled: true` and
  `update_channel: preview`.

then sign the archive from `control-center`, the same way as a `.deb`, and
upload the `.sig` to the release:

```powershell
npx tauri signer sign -f C:\path\to\gary4local-updater.key -p "your real passphrase" path\to\gary4local_<version>_aarch64.app.tar.gz
gh release upload v<version> path\to\gary4local_<version>_aarch64.app.tar.gz.sig
```

the same `TAURI_SIGNING_PRIVATE_KEY` trap applies as for the `.deb`s.

it helps to keep a per-release folder, `artifacts\release-v<version>\`
(untracked), with the three sets of notes and a publish script that checks all
of the above before it creates anything and has a dry-run mode.
`artifacts\release-v0.7.4\publish.ps1` on the release machine is a working
example covering Windows, Linux and ROCm in one go.

### 6. generate the feeds

run the generator once per installer, from the exact signed file, after the
release assets are uploaded. the output folder and the platform key come from
the installer's name, so leave `-OutputDir` off and check the `Product:` line it
prints:

| installer | writes | platform key |
|---|---|---|
| `gary4local_<v>_x64-setup.exe` | `docs/updates/gary4local/` | `windows-x86_64` |
| `gary4local_<v>_amd64.deb` | `docs/updates/gary4local/linux-x86_64/` | `linux-x86_64-deb` |
| `gary4local_<v>_arm64.deb` | `docs/updates/gary4local/linux-aarch64/` | `linux-aarch64-deb` |
| `gary4local_<v>_aarch64.app.tar.gz` | `docs/updates/gary4local/macos-aarch64/` | `darwin-aarch64` |
| `gary4local-rocm_<v>_x64-setup.exe` | `docs/updates/gary4local-rocm/` | `windows-x86_64` |
| `gary4local-server-linux-x64-experimental.tar.gz` | `docs/updates/gary4local/headless-linux-x86_64/` | `linux-x86_64-headless` |
| `gary4local-server-linux-arm64-experimental.tar.gz` | `docs/updates/gary4local/headless-linux-aarch64/` | `linux-aarch64-headless` |

the headless archives carry no version in their names, so `-Version` is the
headless version from `runtime-core/Cargo.toml`, not the desktop one, while the
download URL uses the release tag. `gary update` checks `BUILD-INFO.json` inside
the archive against that version and platform before installing.

`linux-<arch>-deb` is the first key `tauri-plugin-updater` looks up for a
`.deb` install. on a Mac it tries `darwin-aarch64-app`, then `darwin-aarch64`;
the feed uses the second, which matches either way. the Mac feeds point both
the browser fallback and the in-app install at the `.app.tar.gz`. passing an
`-OutputDir` that disagrees with the installer is an
error rather than a silent overwrite.

Windows gets both channels; Linux and macOS get `preview`:

```powershell
$publishedAt = gh release view v0.7.4 --json publishedAt --jq '.publishedAt'
foreach ($channel in 'stable', 'preview') {
    powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
      -Version "0.7.4" `
      -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.7.4/gary4local_0.7.4_x64-setup.exe" `
      -InstallerPath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.7.4_x64-setup.exe" `
      -SignaturePath "control-center\src-tauri\target\release\bundle\nsis\gary4local_0.7.4_x64-setup.exe.sig" `
      -Channel $channel -PublishedAt $publishedAt `
      -NotesText "Release note one.||Release note two.||Compatible with gary4juce v5.0.5."
}
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
  -Version "0.7.4" `
  -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.7.4/gary4local_0.7.4_arm64.deb" `
  -InstallerPath "path\to\gary4local_0.7.4_arm64.deb" `
  -SignaturePath "path\to\gary4local_0.7.4_arm64.deb.sig" `
  -Channel "preview" -PublishedAt $publishedAt `
  -NotesText "Linux note one.||Linux note two."
# and the same again for the amd64 .deb
```

the Linux notes don't have to match the Windows ones. a Linux user doesn't care
that Windows runtime pins didn't change.

### 7. review, commit and push the feeds

the generator runs `validate_update_feeds.ps1` itself. don't publish unless every
run printed `PASS`: it checks the raw timestamp strings, the intended channel and
version, agreement between the two files, and the hash and signature against the
exact local installer and `.sig`. if you edit or copy a feed afterwards, run the
validator again:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\validate_update_feeds.ps1 `
  -FeedDirectory "docs/updates/gary4local/linux-aarch64" -Channel preview -ExpectedVersion "0.7.4"
if ($LASTEXITCODE -ne 0) { throw "Feed validation failed; don't publish." }
```

commit the feed files to `main` and push. the release assets have to be
uploaded before this, or installed apps will offer the update and 404 on the
download.

### 8. check the live feeds

wait for the `pages-build-deployment` run for that commit, then validate every
feed folder you changed at the URLs the app uses, without a cache-busting query.
checking only the browser-download file won't catch a broken native feed:

```powershell
$base = "https://betweentwomidnights.github.io/gary-localhost-installer/updates"
foreach ($feed in @("gary4local stable", "gary4local preview", "gary4local/linux-x86_64 preview", "gary4local/linux-aarch64 preview")) {
    $folder, $channel = $feed -split ' '
    powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\validate_update_feeds.ps1 `
      -BaseUrl "$base/$folder" -Channel $channel -ExpectedVersion "0.7.4"
    if ($LASTEXITCODE -ne 0) { throw "Live feed validation failed: $feed" }
}
```

the headless feeds carry the headless version, so validate them separately:

```powershell
foreach ($folder in "gary4local/headless-linux-x86_64", "gary4local/headless-linux-aarch64") {
    powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\validate_update_feeds.ps1 `
      -BaseUrl "$base/$folder" -Channel preview -ExpectedVersion "0.4.0"
    if ($LASTEXITCODE -ne 0) { throw "Live feed validation failed: $folder" }
}
```

### 9. update an installed app

launch an older installed app on that channel and check it offers
`install update`, then let it install. `download update` only proves the
fallback works, and the validator checks feed metadata; it doesn't replace the
signature check during a real install. on Linux, installing asks for the
password once (`pkexec dpkg -i`), then gary4local restarts. if you can't test
the UI, write that down instead of claiming the in-app update was verified.

for headless, run `gary update --check` on an installed older version (the
DGX Spark is the usual one), then `gary update` and say yes. `gary --version`
should print the new version afterwards, and `gary update --rollback` should
switch back and forth. the check alone only proves the feed parses; the
signature is verified during the install.

## publishing the first headless feed

the archives released with v0.7.6 and earlier have no `gary update` and run
from wherever they were extracted, so the first release with the updater
starts the managed installs. once, for that release:

1. bump `runtime-core` to `0.4.0` in step 1. `0.3.0` was already shipped
   without an updater, and the feed should never offer a version number two
   different archives have used.
2. build, sign and upload both headless archives and their `.sig`s as above.
3. generate both headless feeds. there are no existing files in
   `docs/updates/gary4local/headless-linux-*/` yet, so the generator creates the
   folders. each run must print `Product: gary4local (linux-<arch>-headless)` and
   `PASS`:

   ```powershell
   $tag = "v0.7.7"
   $publishedAt = gh release view $tag --json publishedAt --jq '.publishedAt'
   foreach ($arch in "x64", "arm64") {
       $name = "gary4local-server-linux-$arch-experimental.tar.gz"
       powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
         -Version "0.4.0" `
         -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/$tag/$name" `
         -InstallerPath "path\to\$name" -SignaturePath "path\to\$name.sig" `
         -Channel preview -PublishedAt $publishedAt `
         -NotesText "Headless note one.||Headless note two."
   }
   ```

4. commit and push the two feed folders, then validate them live as in step 8.
5. install the new archive with `install-gary.sh --replace` on the Spark, then
   check that `gary update --check` says it's up to date. nobody can test an
   actual headless update until the next headless release, so step 9 starts
   then. write that down rather than claiming it was verified.
6. in the release notes, give existing headless users the one-time
   `--replace` steps. after this release, `gary update` handles it.

## timestamps and the download-only fallback

leave `-PublishedAt` off to use the generator's current UTC time, or read the
GitHub release's exact publication time as a string with `--jq`:

```powershell
$releaseTag = "v0.7.4"
$publishedAt = gh release view $releaseTag --json publishedAt --jq '.publishedAt'
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($publishedAt) -or $publishedAt -eq 'null') {
    throw "Couldn't read the published release timestamp."
}
```

don't construct the feed JSON by hand or round-trip timestamps through
`ConvertFrom-Json` and string interpolation: PowerShell can turn them into
locale-formatted dates. the generator normalizes valid input to UTC RFC 3339
and refuses invalid input. `2026-10-06T07:13:17Z` is valid;
`10/06/2026 07:13:17` isn't.

we hit this on the `0.4.0-rocm.2` preview. the app offered `download update`
because Tauri rejected `pub_date`. the log showed:

```text
failed to deserialize update response: invalid value for `pub_date`: the 'year' component could not be parsed
Native updater check unavailable: Native updater check failed: invalid value for `pub_date`
```

for this error, regenerate the feeds from the already signed installer and
signature, validate them, and publish the correction on `main`. no rebuild,
replacement installer, or new release tag is needed. for other download-only
failures, read the app log's `Native updater check unavailable` message before
guessing at the cause.

after changing either feed script, run the regression check in Windows
PowerShell before publishing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File smoke-tests\test_update_feed_dates.ps1
if ($LASTEXITCODE -ne 0) { throw "Update feed regression check failed." }
```

## when Pages doesn't update

Pages/CDN caching can keep the old feed visible for several minutes. if it still
shows the old version well after the push, check whether the Pages build
actually succeeded. during a GitHub incident it can fail with a bare
"Page build failed." and leave the site serving the previous deploy:

```powershell
gh api repos/betweentwomidnights/gary-localhost-installer/pages/builds `
  --jq '.[0:3][] | .status + "  " + .created_at + "  " + (.error.message // "")'
```

Pages here is the legacy builder on `main:/docs`, so a rebuild can be forced
without an empty commit:

```powershell
gh api -X POST repos/betweentwomidnights/gary-localhost-installer/pages/builds
```

release assets are served by a different system and work immediately, so a
direct download link is always available while Pages catches up.

## ROCm preview releases

`gary4local-rocm` is the AMD tester build: a different product (its own name,
identifier, data folder and feed) on `feature/rocm-custom-runtime-storage`. it
exists because of its Windows ROCm Python profiles; native services already run
on AMD through Vulkan in the regular app. it's Windows only.

its version follows the mainline release it's built on: `X.Y.Z-rocm.N`, with
`N` going up for tester-only rebuilds of the same mainline version. mainline
can ship before the tester signs off; the ROCm prerelease sits on top of it.

### bring main down

merge `main` into the ROCm branch. never merge the ROCm branch into `main`: it
carries ROCm-only manifests and compatibility patches. single fixes can still be
cherry-picked down.

when resolving conflicts, keep the ROCm identity and take the substance of
main's change:

- `package.json`, `Cargo.toml` and `tauri.conf.json` keep the `gary4local-rocm`
  name, the `com.betweentwomidnights.gary4local.rocm` identifier, and the ROCm
  version.
- `storage.rs` keeps `LOCAL_DATA_DIR_NAME` (`...gary4local.rocm`) for the
  Windows data folder.
- `update.rs` keeps main's feed code but points `UPDATE_FEEDS_BASE` at
  `updates/gary4local-rocm` and defaults the channel to `preview`, since that's
  the only feed ROCm has ever published.
- `AppUpdateModal.svelte` says `gary4local-rocm`.
- `services/manifests/services.json` keeps the ROCm `pythonVersion`,
  `acceleratorProfile`, build steps and env. check it has exactly one entry
  per service ID, and compare yuey's whole `native` block with main's. a clean
  automatic merge once left two yuey entries, which would have kept the AMD
  package on an old version. check the staged manifest matches the source after
  the signed build too.
- where the ROCm side deliberately avoids something (it skips
  `torchaudio.load/save`, for example), taking main's version wholesale quietly
  reverts a ROCm fix.

run the Rust tests, `svelte-check` and the build on the merged branch before
committing.

### release it

1. bump the five version files on the ROCm branch, and add a
   `## vX.Y.Z-rocm.N` entry to *its* `CHANGELOG.md` and `README.md`. main's
   changelog and README never mention ROCm releases. commit and **push the
   branch**.
2. build the signed installer from a separate checkout of the ROCm branch (a
   worktree works), not from the main checkout. the output name must start
   `gary4local-rocm_`; if it says `gary4local_`, you built main. the 0.4.0-rocm.1
   release first came back as a second regular build this way.
3. create the release by full SHA, as a **prerelease**, not latest:
   `--prerelease --latest=false`.
4. generate its feed from the main checkout (Pages only serves `main`). the ROCm
   branch has its own stale copies of the feed files; ignore them, they aren't
   served. the generator must print `Product: gary4local-rocm` and `PASS`:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File control-center\src-tauri\scripts\generate_update_feeds.ps1 `
     -Version "0.7.4-rocm.1" `
     -ArtifactUrl "https://github.com/betweentwomidnights/gary-localhost-installer/releases/download/v0.7.4-rocm.1/gary4local-rocm_0.7.4-rocm.1_x64-setup.exe" `
     -InstallerPath "path\to\gary4local-rocm_0.7.4-rocm.1_x64-setup.exe" `
     -SignaturePath "path\to\gary4local-rocm_0.7.4-rocm.1_x64-setup.exe.sig" `
     -Channel "preview"
   ```

5. commit the two feed files to `main`, push, and validate the live feed at
   `.../updates/gary4local-rocm` with `-Channel preview`, as in step 8. the
   tester's app should then offer `install update`.

the ROCm feed notes are short and casual. they're for one tester, not the usual
release-note voice.

## source builds

source builders can leave out the whole updater, UI and backend check:

```powershell
$env:VITE_ENABLE_APP_UPDATER='0'
npm.cmd run tauri build
Remove-Item Env:VITE_ENABLE_APP_UPDATER
```

that's the recommended opt-out for forks and local-only builds that shouldn't
advertise public releases.
