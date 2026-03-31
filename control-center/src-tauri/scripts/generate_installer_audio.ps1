$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$srcTauriDir = Split-Path -Parent $scriptDir
$controlCenterDir = Split-Path -Parent $srcTauriDir
$repoRoot = Split-Path -Parent $controlCenterDir

$sourcePath = Join-Path $repoRoot "keygen_music_for_installer.wav"
$outputPath = Join-Path $srcTauriDir "windows\\installer-audio.wav"
$sampleRate = 16000

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Source audio not found: $sourcePath"
}

$ffmpeg = Get-Command ffmpeg -ErrorAction Stop

& $ffmpeg.Source -y -i $sourcePath -ac 1 -ar $sampleRate -c:a adpcm_ima_wav $outputPath

if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed with exit code $LASTEXITCODE"
}

Get-Item -LiteralPath $outputPath | Select-Object FullName, Length, LastWriteTime
