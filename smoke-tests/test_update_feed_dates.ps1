$ErrorActionPreference = 'Stop'
$generator = Join-Path $PSScriptRoot '..\control-center\src-tauri\scripts\generate_update_feeds.ps1'
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testRoot = Join-Path $tempRoot ('gary-update-feed-test-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null

try {
    $installer = Join-Path $testRoot 'gary4local-rocm_0.4.0-rocm.2_x64-setup.exe'
    [IO.File]::WriteAllText($installer, 'test installer')
    [IO.File]::WriteAllText("$installer.sig", 'test signature')
    $output = Join-Path $testRoot 'gary4local-rocm'
    $expected = [DateTimeOffset]::Parse('2026-10-06T07:13:17Z')

    # The second input reproduces a timestamp coerced to a local-format string
    # before binding to PublishedAt. Both must produce the same UTC instant.
    foreach ($inputDate in @('2026-10-06T07:13:17Z', '10/06/2026 07:13:17 +00:00')) {
        & $generator -Version '0.4.0-rocm.2' -ArtifactUrl 'https://example.com/setup.exe' `
            -InstallerPath $installer -SignaturePath "$installer.sig" -Channel preview `
            -OutputDir $output -PublishedAt $inputDate -NotesText 'Test update.'

        foreach ($feed in @('preview.json', 'native-preview.json')) {
            # Read raw JSON to prevent PowerShell converting the date again.
            $json = Get-Content -LiteralPath (Join-Path $output $feed) -Raw
            $match = [regex]::Match($json, '"(?:published_at|pub_date)"\s*:\s*"([^"]+)"')
            if (-not $match.Success) { throw "Missing date in $feed" }
            $date = [DateTimeOffset]::ParseExact($match.Groups[1].Value, 'o', [Globalization.CultureInfo]::InvariantCulture)
            if ($date -ne $expected -or $date.Offset -ne [TimeSpan]::Zero) {
                throw "Wrong published instant in $feed"
            }
        }
    }
    Write-Output 'PASS: ISO and local-format inputs produce UTC RFC 3339 dates in both update feeds.'
}
finally {
    $resolvedTestRoot = [IO.Path]::GetFullPath($testRoot)
    if ($resolvedTestRoot.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedTestRoot) -match '^gary-update-feed-test-[0-9a-f]{32}$') {
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
