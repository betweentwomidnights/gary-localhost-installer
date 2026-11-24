param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$ArtifactUrl,

    [Parameter(Mandatory = $true)]
    [string]$InstallerPath,

    [Parameter(Mandatory = $true)]
    [string]$SignaturePath,

    [ValidateSet("stable", "preview")]
    [string]$Channel = "stable",

    # Leave empty to derive from the installer filename. See below - hardcoding a
    # default here is what let a rocm build overwrite the cuda feed.
    [string]$OutputDir = "",

    [string]$PublishedAt = (Get-Date).ToUniversalTime().ToString("o"),

    [string]$NotesText = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Notes = @()
)

$ErrorActionPreference = 'Stop'

function Ensure-ParentDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

$resolvedInstallerPath = (Resolve-Path -LiteralPath $InstallerPath).Path

# Which product this installer belongs to, taken from the filename: everything
# before the first underscore, so `gary4local_0.2.1_x64-setup.exe` gives
# `gary4local` and `gary4local-rocm_0.2.1-rocm.18_x64-setup.exe` gives
# `gary4local-rocm`. The feed directory follows from the artifact rather than
# from a per-branch default, because those defaults drifted apart between main
# and the rocm branch and a rocm release silently overwrote the cuda feed.
$installerLeaf = Split-Path -Leaf $resolvedInstallerPath
$product = ($installerLeaf -split '_', 2)[0]
# The headless archive carries no version or underscore in its name; it is the
# same gary4local product, fed from its own per-architecture folder.
$headlessArchive = [regex]::Match($installerLeaf, '^gary4local-server-linux-(x64|arm64)-experimental\.tar\.gz$')
if ($headlessArchive.Success) { $product = 'gary4local' }
if ([string]::IsNullOrWhiteSpace($product)) {
    throw "Could not read a product name from installer filename '$installerLeaf'."
}

# Which platform it installs on, also from the filename. Windows keeps the
# product folder it always had. Each Linux architecture gets a folder of its own
# beside it: a feed carries one version, and the phase-1 feed one download URL,
# so Linux test builds can move without offering Windows a version it can't
# install. The native key is the one tauri-plugin-updater looks up first for a
# .deb install ({os}-{arch}-deb). Same app, separate channel files.
# The headless console reads its own folders; only `gary update` reads them.
if ($headlessArchive.Success) {
    $headlessArch = @{ 'x64' = 'x86_64'; 'arm64' = 'aarch64' }[$headlessArchive.Groups[1].Value]
    $platformKey = "linux-$headlessArch-headless"
    $platformDir = "headless-linux-$headlessArch"
}
elseif ($installerLeaf -like '*_x64-setup.exe') {
    $platformKey = 'windows-x86_64'
    $platformDir = ''
}
elseif ($installerLeaf -like '*_amd64.deb') {
    $platformKey = 'linux-x86_64-deb'
    $platformDir = 'linux-x86_64'
}
elseif ($installerLeaf -like '*_arm64.deb') {
    $platformKey = 'linux-aarch64-deb'
    $platformDir = 'linux-aarch64'
}
elseif ($installerLeaf -like '*_aarch64.app.tar.gz') {
    # tauri-plugin-updater tries darwin-aarch64-app first, then darwin-aarch64.
    $platformKey = 'darwin-aarch64'
    $platformDir = 'macos-aarch64'
}
else {
    throw "Unrecognized installer '$installerLeaf': expected a Windows *_x64-setup.exe, a Linux *_amd64.deb / *_arm64.deb, a macOS *_aarch64.app.tar.gz, or a headless gary4local-server-linux-<x64|arm64>-experimental.tar.gz."
}

$defaultOutputDir = Join-Path "docs\updates" $product
if ($platformDir) { $defaultOutputDir = Join-Path $defaultOutputDir $platformDir }
$effectiveOutputDir = $OutputDir

$effectiveNotes = @(
    $Notes | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

# Recover from `powershell -File ... -Notes @("a","b","c")`, which can
# misbind extra note strings into the following positional parameters.
if (
    [string]::IsNullOrWhiteSpace($NotesText) -and
    -not [string]::IsNullOrWhiteSpace($OutputDir) -and
    $OutputDir -ne $defaultOutputDir -and
    -not [System.IO.Path]::IsPathRooted($OutputDir) -and
    $OutputDir -notmatch '[\\/]'
) {
    $effectiveNotes += $OutputDir
    $effectiveOutputDir = $defaultOutputDir
}

if ([string]::IsNullOrWhiteSpace($effectiveOutputDir)) {
    $effectiveOutputDir = $defaultOutputDir
}
elseif (
    ($platformDir -and ((Split-Path -Leaf $effectiveOutputDir) -ne $platformDir -or
        (Split-Path -Leaf (Split-Path -Parent $effectiveOutputDir)) -ne $product)) -or
    (-not $platformDir -and (Split-Path -Leaf $effectiveOutputDir) -ne $product)
) {
    throw ("Output directory '$effectiveOutputDir' does not match installer " +
        "'$installerLeaf'. Pass -OutputDir '$defaultOutputDir', or omit -OutputDir and let " +
        "it follow the installer.")
}

Write-Host "Product: $product ($platformKey) -> $effectiveOutputDir"

$resolvedSignaturePath = (Resolve-Path -LiteralPath $SignaturePath).Path
$resolvedOutputDir = if ([System.IO.Path]::IsPathRooted($effectiveOutputDir)) {
    $effectiveOutputDir
} else {
    Join-Path (Get-Location) $effectiveOutputDir
}

$phase1OutputPath = Join-Path $resolvedOutputDir "$Channel.json"
$nativeOutputPath = Join-Path $resolvedOutputDir "native-$Channel.json"

$effectivePublishedAt = $PublishedAt
$parsedPublishedAt = [System.DateTimeOffset]::MinValue

if (-not [string]::IsNullOrWhiteSpace($NotesText)) {
    $effectiveNotes += @(
        $NotesText -split '\|\|' |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

if (-not [System.DateTimeOffset]::TryParse($effectivePublishedAt, [ref]$parsedPublishedAt)) {
    throw "PublishedAt must be a timestamp. Omit it to use the current UTC time, or pass an RFC 3339 string such as 2026-10-06T07:13:17Z."
}
else {
    # PowerShell can coerce JSON timestamps to DateTime values, then bind them
    # to this string parameter using the local date format. Tauri requires
    # RFC 3339 even when TryParse accepts that local format.
    $effectivePublishedAt = $parsedPublishedAt.ToUniversalTime().ToString("o", [System.Globalization.CultureInfo]::InvariantCulture)
}

$signature = (Get-Content -Raw -LiteralPath $resolvedSignaturePath).Trim()
if ([string]::IsNullOrWhiteSpace($signature)) {
    throw "Signature file '$resolvedSignaturePath' was empty."
}

$sha256 = (Get-FileHash -LiteralPath $resolvedInstallerPath -Algorithm SHA256).Hash.ToLowerInvariant()

# The headless version is runtime-core's own, bumped by hand. `gary update`
# offers only a newer version, so a different archive published under the
# version already in the feed would never reach anyone who has that version.
# Regenerating from the same archive (to fix notes or a date) stays allowed.
if ($headlessArchive.Success -and $Version -cnotmatch '^\d+\.\d+\.\d+$') {
    throw "A headless feed version must be MAJOR.MINOR.PATCH; found '$Version'."
}
if ($headlessArchive.Success -and (Test-Path -LiteralPath $phase1OutputPath)) {
    $existing = Get-Content -LiteralPath $phase1OutputPath -Raw | ConvertFrom-Json
    if ($existing.latest_version -ceq $Version -and $existing.sha256 -cne $sha256) {
        throw ("The $platformDir feed already publishes $Version for a different archive. " +
            "Bump the version in control-center/runtime-core/Cargo.toml and rebuild.")
    }
}

$phase1Payload = [ordered]@{
    channel = $Channel
    latest_version = $Version
    download_url = $ArtifactUrl
    sha256 = $sha256
    published_at = $effectivePublishedAt
    notes = @($effectiveNotes)
}

$platforms = [ordered]@{}
$platforms[$platformKey] = [ordered]@{
    signature = $signature
    url = $ArtifactUrl
}

$nativePayload = [ordered]@{
    version = $Version
    notes = if ($effectiveNotes.Count -gt 0) { ($effectiveNotes -join "`n") } else { "" }
    pub_date = $effectivePublishedAt
    platforms = $platforms
}

Ensure-ParentDirectory -Path $phase1OutputPath
Ensure-ParentDirectory -Path $nativeOutputPath

Set-Content -LiteralPath $phase1OutputPath -Value ($phase1Payload | ConvertTo-Json -Depth 6)
Set-Content -LiteralPath $nativeOutputPath -Value ($nativePayload | ConvertTo-Json -Depth 6)

& (Join-Path $PSScriptRoot 'validate_update_feeds.ps1') -FeedDirectory $resolvedOutputDir `
    -Channel $Channel -ExpectedVersion $Version -InstallerPath $resolvedInstallerPath -SignaturePath $resolvedSignaturePath

Write-Host "Generated phase-1 manifest:" $phase1OutputPath
Write-Host "Generated native updater feed:" $nativeOutputPath
Write-Host "Installer SHA256:" $sha256
