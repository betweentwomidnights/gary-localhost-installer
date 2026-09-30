# ABOUTME: Packages the CUDA runtime every native service's CUDA backend needs
# ABOUTME: (cudart and cuBLAS, with NVIDIA's EULA) as one zip plus SHA256SUMS.
#
# gary4local installs this once, under native-runtimes/, and puts it on PATH for
# any native service whose CUDA package `requires` it. It lives here rather than
# in a service's release because it's the same DLLs for every service, and a
# CUDA runtime update should be one deliberate release, not a side effect of
# whichever service shipped last. See docs/native-runtime-packages.md.
#
# The zip is named for the exact toolkit it came from (cudart-12.8.1-...), and
# its release tag is too (runtime-cudart-12.8.1), so a published pack never
# changes. The manifest names it by the compatibility class services ask for
# (cudart-12.8).
#
# Usage:
#   control-center\src-tauri\scripts\package_cuda_runtime.ps1 [-OutDir dist\cuda-runtime]
param(
    [string]$OutDir = "dist\cuda-runtime"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $root

function Fail([string]$message) {
    Write-Error "package-cuda-runtime: $message"
    exit 1
}

# The output folder is emptied below; keep that inside this checkout.
$outPath = [System.IO.Path]::GetFullPath($(if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $root $OutDir }))
if (-not $outPath.StartsWith($root.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    Fail "-OutDir must be inside $root"
}

if (-not $env:CUDA_PATH) { Fail "CUDA_PATH is not set; install the CUDA Toolkit" }
$versionJson = Join-Path $env:CUDA_PATH "version.json"
if (-not (Test-Path $versionJson)) { Fail "$versionJson is missing; a full CUDA Toolkit install writes it" }
$cudaVersion = (Get-Content $versionJson -Raw | ConvertFrom-Json).cuda.version
if ($cudaVersion -notmatch '^(\d+)\.(\d+)\.(\d+)$') { Fail "cannot read a x.y.z CUDA version from $versionJson" }
$major = $Matches[1]
$runtimeName = "cudart-$($Matches[1]).$($Matches[2])"
$zipName = "cudart-$cudaVersion-windows-x64.zip"

Write-Host "cuda      $cudaVersion ($env:CUDA_PATH)"
Write-Host "runtime   $runtimeName"

$stage = Join-Path $outPath "stage"
if (Test-Path $outPath) { Remove-Item -Recurse -Force $outPath }
New-Item -ItemType Directory -Force $stage | Out-Null

# What ggml-cuda.dll imports from the toolkit. Nothing else is redistributed.
$bin = Join-Path $env:CUDA_PATH "bin"
$files = @("cudart64_$major.dll", "cublas64_$major.dll", "cublasLt64_$major.dll")
foreach ($file in $files) {
    $source = Join-Path $bin $file
    if (-not (Test-Path $source)) { Fail "$file is not in $bin" }
    Copy-Item $source $stage
}
$eula = Join-Path $env:CUDA_PATH "EULA.txt"
if (-not (Test-Path $eula)) { Fail "EULA.txt is not in $env:CUDA_PATH; it has to ship with the DLLs" }
Copy-Item $eula (Join-Path $stage "NVIDIA-CUDA-EULA.txt")

$info = [ordered]@{
    runtime   = $runtimeName
    cuda      = $cudaVersion
    platform  = "windows-x64"
    files     = $files
    built_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
[System.IO.File]::WriteAllText(
    (Join-Path $stage "RUNTIME-INFO.json"),
    ($info | ConvertTo-Json) + "`n",
    (New-Object System.Text.UTF8Encoding($false)))

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = Join-Path $outPath $zipName
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $stage, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $false)
Remove-Item -Recurse -Force $stage

$hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
# LF endings and no BOM, so `sha256sum -c SHA256SUMS` works as-is.
[System.IO.File]::WriteAllText(
    (Join-Path $outPath "SHA256SUMS"),
    "$hash  $zipName`n",
    (New-Object System.Text.UTF8Encoding($false)))

$megabytes = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host ("{0,-36} {1,8} MB  {2}" -f $zipName, $megabytes, $hash)
Write-Host "runtime -> $outPath"
