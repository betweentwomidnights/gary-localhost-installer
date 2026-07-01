param(
    [string]$WorkDir = $(Join-Path $env:LOCALAPPDATA "gary4local-rocm-preflight"),
    [switch]$Fresh,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$RocmVersion = "7.2.1"
$PythonVersion = "3.12"
$ExpectedAmdDriverPackage = "26.2.2"
$RocmBaseUrl = "https://repo.radeon.com/rocm/windows/rocm-rel-$RocmVersion"

$RocmSdkUrls = @(
    "$RocmBaseUrl/rocm_sdk_core-$RocmVersion-py3-none-win_amd64.whl",
    "$RocmBaseUrl/rocm_sdk_devel-$RocmVersion-py3-none-win_amd64.whl",
    "$RocmBaseUrl/rocm_sdk_libraries_custom-$RocmVersion-py3-none-win_amd64.whl",
    "$RocmBaseUrl/rocm-$RocmVersion.tar.gz"
)

$TorchUrls = @(
    "$RocmBaseUrl/torch-2.9.1%2Brocm$RocmVersion-cp312-cp312-win_amd64.whl",
    "$RocmBaseUrl/torchaudio-2.9.1%2Brocm$RocmVersion-cp312-cp312-win_amd64.whl",
    "$RocmBaseUrl/torchvision-0.24.1%2Brocm$RocmVersion-cp312-cp312-win_amd64.whl"
)

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ==="
}

function Invoke-Logged {
    param(
        [string]$Program,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host ('$ ' + $Program + ' ' + ($Arguments -join ' '))
    & $Program @Arguments
}

function Remove-WorkDirSafely {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolved = [System.IO.Path]::GetFullPath($Path)
    $localAppData = [System.IO.Path]::GetFullPath($env:LOCALAPPDATA)
    $temp = [System.IO.Path]::GetFullPath($env:TEMP)

    if (-not ($resolved.StartsWith($localAppData, [System.StringComparison]::OrdinalIgnoreCase) -or
              $resolved.StartsWith($temp, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Refusing to remove '$resolved'. Use a WorkDir under LOCALAPPDATA or TEMP."
    }

    Remove-Item -LiteralPath $resolved -Recurse -Force
}

Write-Host "gary4local-rocm Windows PyTorch/ROCm preflight"
Write-Host "ROCm target: $RocmVersion"
Write-Host "Python target: $PythonVersion"
Write-Host "AMD driver package expected by AMD docs for this ROCm release: $ExpectedAmdDriverPackage"
Write-Host "Work dir: $WorkDir"

Write-Section "windows"
Get-CimInstance Win32_OperatingSystem |
    Select-Object Caption, Version, BuildNumber, OSArchitecture |
    Format-List

Write-Section "amd graphics"
$amdGpus = Get-CimInstance Win32_VideoController |
    Where-Object { $_.Name -match "AMD|Radeon" }

if (-not $amdGpus) {
    Write-Warning "No AMD/Radeon graphics adapter was reported by Win32_VideoController."
} else {
    $amdGpus |
        Select-Object Name, DriverVersion, DriverDate, AdapterRAM, PNPDeviceID |
        Format-List
}

if ($SkipInstall) {
    Write-Warning "SkipInstall was set, so Python/ROCm/PyTorch checks were not run."
    exit 0
}

Write-Section "uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found on PATH. Install gary4local once or install uv from https://docs.astral.sh/uv/ first."
}
Invoke-Logged "uv" @("--version")

if ($Fresh) {
    Write-Section "fresh work dir"
    Remove-WorkDirSafely -Path $WorkDir
}

Write-Section "python env"
Invoke-Logged "uv" @("venv", "--python", $PythonVersion, "--seed", $WorkDir)

$Python = Join-Path $WorkDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python was not created at $Python"
}
Invoke-Logged $Python @("--version")

Write-Section "install rocm sdk wheels"
$RocmInstallArgs = @("pip", "install") + $RocmSdkUrls + @("--python", $Python)
Invoke-Logged "uv" $RocmInstallArgs

Write-Section "install pytorch rocm wheels"
$TorchInstallArgs = @("pip", "install") + $TorchUrls + @("--python", $Python)
Invoke-Logged "uv" $TorchInstallArgs

Write-Section "torch hip diagnostic"
$Diagnostic = @'
import json
import os
import platform
import sys

result = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
}

try:
    import torch

    torch_info = {
        "version": torch.__version__,
        "version_cuda": torch.version.cuda,
        "version_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    }

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            device = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": getattr(props, "total_memory", None),
                "major": getattr(props, "major", None),
                "minor": getattr(props, "minor", None),
                "gcn_arch_name": getattr(props, "gcnArchName", None),
            }
            devices.append(device)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            torch_info["mem_get_info"] = {
                "free": free_mem,
                "total": total_mem,
            }
        except Exception as exc:
            torch_info["mem_get_info_error"] = repr(exc)

    result["torch"] = torch_info
    result["devices"] = devices
except Exception as exc:
    result["torch_import_error"] = repr(exc)

print(json.dumps(result, indent=2))

torch_info = result.get("torch") or {}
if not torch_info.get("version_hip"):
    raise SystemExit("torch imported, but torch.version.hip was empty")
if not torch_info.get("cuda_available"):
    raise SystemExit("torch imported with HIP, but torch.cuda.is_available() was false")
'@

& $Python -c $Diagnostic
