param(
    [string]$MakeNsis = "$env:LOCALAPPDATA\tauri\NSIS\makensis.exe"
)

# Compile the real uninstall hook into a small fixture, then exercise it
# without installing or uninstalling gary4local or touching real user data.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$hookPath = Join-Path $repoRoot 'control-center\src-tauri\windows\nsis-hooks.nsh'
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('gary-uninstall-test-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $fixtureRoot | Out-Null
$fixtureScript = @'
!include MUI2.nsh
!include nsDialogs.nsh
!include LogicLib.nsh
!include FileFunc.nsh
!include "__HOOK_PATH__"
Name "Runtime cleanup fixture"
OutFile "cleanup-fixture.exe"
RequestExecutionLevel user
InstallDir "$TEMP\gary-uninstall-fixture"
Var DeleteAppDataCheckboxState
Var UpdateMode
!undef MUI_PAGE_CUSTOMFUNCTION_SHOW
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
Section
SetOutPath "$INSTDIR"
WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd
Function un.onInit
${GetParameters} $R0
${GetOptions} $R0 "/DELETE=" $DeleteAppDataCheckboxState
${GetOptions} $R0 "/UPDATE=" $UpdateMode
FunctionEnd
Section "Uninstall"
!insertmacro NSIS_HOOK_POSTUNINSTALL
SectionEnd
'@
$fixturePath = Join-Path $fixtureRoot 'cleanup-fixture.nsi'
Set-Content -LiteralPath $fixturePath -Value $fixtureScript.Replace('__HOOK_PATH__', $hookPath)
& $MakeNsis /V2 $fixturePath
if ($LASTEXITCODE -ne 0) { throw 'NSIS fixture compilation failed' }

$cases = @(
    @{ Name = 'checked'; Delete = '1'; Update = '0'; Keep = $false },
    @{ Name = 'unchecked'; Delete = '0'; Update = '0'; Keep = $true },
    @{ Name = 'update'; Delete = '1'; Update = '1'; Keep = $true }
)
foreach ($case in $cases) {
    # Include spaces to check install-folder argument handling.
    $caseDir = Join-Path $fixtureRoot ($case.Name + ' install')
    $dataDir = Join-Path $caseDir 'gary4local-data'
    $customDir = Join-Path $caseDir 'custom-storage'
    New-Item -ItemType Directory -Force -Path $dataDir, $customDir | Out-Null
    Set-Content -LiteralPath (Join-Path $dataDir 'model-fixture.txt') -Value 'default models'
    Set-Content -LiteralPath (Join-Path $customDir 'model-fixture.txt') -Value 'custom models'
    $installer = Start-Process -FilePath (Join-Path $fixtureRoot 'cleanup-fixture.exe') -ArgumentList @('/S', ('/D=' + $caseDir)) -WindowStyle Hidden -Wait -PassThru
    if ($installer.ExitCode -ne 0) { throw "Fixture install failed: $($case.Name)" }
    $uninstaller = Start-Process -FilePath (Join-Path $caseDir 'uninstall.exe') -ArgumentList @('/S', ('/DELETE=' + $case.Delete), ('/UPDATE=' + $case.Update), ('_?=' + $caseDir)) -WindowStyle Hidden -Wait -PassThru
    if ($uninstaller.ExitCode -ne 0) { throw "Fixture uninstall failed: $($case.Name)" }
    if ((Test-Path -LiteralPath $dataDir) -ne $case.Keep) { throw "Default-data preservation mismatch: $($case.Name)" }
    if (!(Test-Path -LiteralPath (Join-Path $customDir 'model-fixture.txt'))) { throw "Custom storage was removed: $($case.Name)" }
    Write-Output "$($case.Name): PASS"
}
Write-Output "Fixture artifacts: $fixtureRoot"
