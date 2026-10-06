[CmdletBinding(DefaultParameterSetName = 'Local')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Local')]
    [string]$FeedDirectory,

    [Parameter(Mandatory = $true, ParameterSetName = 'Live')]
    [string]$BaseUrl,

    [Parameter(Mandatory = $true)]
    [ValidateSet('stable', 'preview')]
    [string]$Channel,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedVersion,

    [Parameter(ParameterSetName = 'Local')]
    [string]$InstallerPath = '',

    [Parameter(ParameterSetName = 'Local')]
    [string]$SignaturePath = ''
)

$ErrorActionPreference = 'Stop'
$isLive = $PSCmdlet.ParameterSetName -eq 'Live'

function Read-Feed {
    param([string]$Name)
    if ($isLive) {
        # Use the same public URL as the app, without a cache-busting query.
        return (Invoke-WebRequest -UseBasicParsing -Uri ($BaseUrl.TrimEnd('/') + '/' + $Name)).Content
    }
    return Get-Content -LiteralPath (Join-Path $FeedDirectory $Name) -Raw
}

function Read-PublishedDate {
    param([string]$Json, [string]$Field)
    # Inspect the JSON string before ConvertFrom-Json can coerce it to a date
    # and hide an invalid serialized format.
    $matches = [regex]::Matches($Json, ('"' + $Field + '"\s*:\s*"([^"]+)"'))
    if ($matches.Count -ne 1) { throw "Expected exactly one $Field timestamp." }
    $value = $matches[0].Groups[1].Value
    if ($value -cnotmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?(?:Z|[+-]\d{2}:\d{2})$') {
        throw "$Field must be a UTC RFC 3339 string, for example 2026-10-06T07:13:17Z. Found '$value'."
    }
    $date = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse($value, [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::None, [ref]$date) -or $date.Offset -ne [TimeSpan]::Zero) {
        throw "$Field is not a valid UTC timestamp: '$value'."
    }
    return $date
}

$phaseJson = Read-Feed "$Channel.json"
$nativeJson = Read-Feed "native-$Channel.json"
$phaseDate = Read-PublishedDate $phaseJson 'published_at'
$nativeDate = Read-PublishedDate $nativeJson 'pub_date'
$phase = $phaseJson | ConvertFrom-Json
$native = $nativeJson | ConvertFrom-Json
$platform = $native.platforms.'windows-x86_64'

if ($phase.channel -cne $Channel -or $phase.latest_version -cne $ExpectedVersion -or
    $native.version -cne $ExpectedVersion) { throw 'Feed channel/version does not match the intended release.' }
if ($phaseDate -ne $nativeDate) { throw 'The two feeds disagree on the publication time.' }
if (-not $platform -or [string]::IsNullOrWhiteSpace($phase.download_url) -or
    $phase.download_url -cne $platform.url) { throw 'The two feeds must reference the same Windows installer URL.' }
$url = [Uri]$phase.download_url
if (-not $url.IsAbsoluteUri -or $url.Scheme -ne 'https') { throw 'Installer URL must use HTTPS.' }
if ($phase.sha256 -cnotmatch '^[0-9a-f]{64}$') { throw 'Installer SHA-256 must be 64 lowercase hexadecimal characters.' }
if ([string]::IsNullOrWhiteSpace($platform.signature)) { throw 'Native feed is missing its updater signature.' }
if ([Convert]::FromBase64String($platform.signature).Length -eq 0) { throw 'Updater signature is empty.' }

if (-not [string]::IsNullOrWhiteSpace($SignaturePath) -and [string]::IsNullOrWhiteSpace($InstallerPath)) {
    throw 'Pass InstallerPath when checking a local signature.'
}
if (-not [string]::IsNullOrWhiteSpace($InstallerPath)) {
    if ([string]::IsNullOrWhiteSpace($SignaturePath)) { $SignaturePath = "$InstallerPath.sig" }
    $hash = (Get-FileHash -LiteralPath $InstallerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($phase.sha256 -cne $hash) { throw 'Feed SHA-256 differs from the built installer.' }
    $signature = (Get-Content -LiteralPath $SignaturePath -Raw).Trim()
    if ($platform.signature -cne $signature) { throw 'Feed signature differs from the built .sig file.' }
}

Write-Output "PASS: $Channel $ExpectedVersion feed dates, versions, URLs and signature metadata agree."
