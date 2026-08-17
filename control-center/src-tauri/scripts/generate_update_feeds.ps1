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
if ([string]::IsNullOrWhiteSpace($product)) {
    throw "Could not read a product name from installer filename '$installerLeaf'."
}

$defaultOutputDir = Join-Path "docs\updates" $product
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
elseif ((Split-Path -Leaf $effectiveOutputDir) -ne $product) {
    throw ("Output directory '$effectiveOutputDir' does not match installer product " +
        "'$product'. Pass -OutputDir '$defaultOutputDir', or omit -OutputDir and let " +
        "it follow the installer.")
}

Write-Host "Product: $product -> $effectiveOutputDir"

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
    if (-not [string]::IsNullOrWhiteSpace($effectivePublishedAt)) {
        $effectiveNotes += $effectivePublishedAt
    }
    $effectivePublishedAt = (Get-Date).ToUniversalTime().ToString("o")
}

$signature = (Get-Content -Raw -LiteralPath $resolvedSignaturePath).Trim()
if ([string]::IsNullOrWhiteSpace($signature)) {
    throw "Signature file '$resolvedSignaturePath' was empty."
}

$sha256 = (Get-FileHash -LiteralPath $resolvedInstallerPath -Algorithm SHA256).Hash.ToLowerInvariant()

$phase1Payload = [ordered]@{
    channel = $Channel
    latest_version = $Version
    download_url = $ArtifactUrl
    sha256 = $sha256
    published_at = $effectivePublishedAt
    notes = @($effectiveNotes)
}

$platforms = [ordered]@{}
$platforms["windows-x86_64"] = [ordered]@{
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

Write-Host "Generated phase-1 manifest:" $phase1OutputPath
Write-Host "Generated native updater feed:" $nativeOutputPath
Write-Host "Installer SHA256:" $sha256
