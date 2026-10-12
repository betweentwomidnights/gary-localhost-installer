# Run only after the caller has verified the pinned download's SHA-256.
# Without -Install this verifies the publisher and version without executing it.
param(
    [Parameter(Mandatory = $true)][string]$Installer,
    [Parameter(Mandatory = $true)][string]$ExpectedVersion,
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
try {
    $signature = Get-AuthenticodeSignature -LiteralPath $Installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch '(^|,\s*)CN=Microsoft Corporation(,|$)') {
        throw 'Visual C++ prerequisite installer does not have a valid Microsoft Corporation signature.'
    }
    $version = (Get-Item -LiteralPath $Installer).VersionInfo.ProductVersion
    if ($version -ne $ExpectedVersion) { throw "Unexpected Visual C++ prerequisite installer version: $version" }
    if (-not $Install) {
        @{ verified = $true; version = $version; publisher = 'Microsoft Corporation' } | ConvertTo-Json -Compress
        exit 0
    }
    # Windows presents its own administrator-consent prompt. Cancelling it must
    # fail the runtime install; never fall back to CPU or mark the service ready.
    $process = Start-Process -FilePath $Installer -ArgumentList '/install', '/passive', '/norestart' -Verb RunAs -WindowStyle Hidden -Wait -PassThru
    exit $process.ExitCode
} catch {
    Write-Error $_
    exit 1
}
