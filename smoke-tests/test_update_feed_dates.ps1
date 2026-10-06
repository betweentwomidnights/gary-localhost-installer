$ErrorActionPreference = 'Stop'
$generator = Join-Path $PSScriptRoot '..\control-center\src-tauri\scripts\generate_update_feeds.ps1'
$validator = Join-Path $PSScriptRoot '..\control-center\src-tauri\scripts\validate_update_feeds.ps1'
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testRoot = Join-Path $tempRoot ('gary-update-feed-test-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null

try {
    $installer = Join-Path $testRoot 'gary4local-rocm_0.4.0-rocm.2_x64-setup.exe'
    [IO.File]::WriteAllText($installer, 'test installer')
    $signature = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('test signature'))
    [IO.File]::WriteAllText("$installer.sig", $signature)
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

    function Assert-Rejected {
        param([scriptblock]$Action, [string]$ExpectedMessage)
        try { $null = & $Action }
        catch {
            if (-not $_.Exception.Message.Contains($ExpectedMessage)) { throw }
            Write-Output "PASS: rejected $ExpectedMessage"
            return
        }
        throw "Validation unexpectedly accepted: $ExpectedMessage"
    }

    $nativePath = Join-Path $output 'native-preview.json'
    $goodNative = Get-Content -LiteralPath $nativePath -Raw
    $badNative = [regex]::Replace($goodNative, '"pub_date"\s*:\s*"[^"]+"', '"pub_date": "10/06/2026 07:13:17"')
    [IO.File]::WriteAllText($nativePath, $badNative)
    Assert-Rejected { & $validator -FeedDirectory $output -Channel preview -ExpectedVersion '0.4.0-rocm.2' } `
        'pub_date must be a UTC RFC 3339 string'
    [IO.File]::WriteAllText($nativePath, $goodNative)

    Assert-Rejected { & $validator -FeedDirectory $output -Channel preview -ExpectedVersion '0.4.0-rocm.3' } `
        'Feed channel/version does not match'
    [IO.File]::WriteAllText($installer, 'changed installer')
    Assert-Rejected { & $validator -FeedDirectory $output -Channel preview -ExpectedVersion '0.4.0-rocm.2' -InstallerPath $installer } `
        'Feed SHA-256 differs'
    [IO.File]::WriteAllText($installer, 'test installer')
    [IO.File]::WriteAllText("$installer.sig", [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('different signature')))
    Assert-Rejected { & $validator -FeedDirectory $output -Channel preview -ExpectedVersion '0.4.0-rocm.2' -InstallerPath $installer } `
        'Feed signature differs'

    Assert-Rejected { & $generator -Version '0.4.0-rocm.2' -ArtifactUrl 'https://example.com/setup.exe' `
        -InstallerPath $installer -SignaturePath "$installer.sig" -Channel preview -OutputDir $output -PublishedAt 'not a date' } `
        'PublishedAt must be a timestamp'
    if ((Get-Content -LiteralPath $nativePath -Raw) -cne $goodNative) { throw 'Invalid date input changed the existing feed.' }
}
finally {
    $resolvedTestRoot = [IO.Path]::GetFullPath($testRoot)
    if ($resolvedTestRoot.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedTestRoot) -match '^gary-update-feed-test-[0-9a-f]{32}$') {
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
}
