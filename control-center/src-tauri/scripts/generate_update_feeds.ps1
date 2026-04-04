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

    [string]$OutputDir = "docs\updates\gary4local",

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

$defaultOutputDir = "docs\updates\gary4local"
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

$resolvedInstallerPath = (Resolve-Path -LiteralPath $InstallerPath).Path
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
