; Gary4JUCE Local Backend Installer - Improved Version
; Save this as: gary4juce-setup.iss
[Setup]
AppName=gary4juce
AppVersion=1.0.0
AppPublisher=the collabage patch, inc.
DefaultDirName={autopf}\Gary4JUCE
DefaultGroupName=Gary4JUCE Local Backend
OutputDir=output
OutputBaseFilename=Gary4JUCE-LocalBackend-Setup
SetupIconFile=icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
; CORRECTED: Need admin to install to Program Files (but apps run as user)
PrivilegesRequired=admin
UninstallDisplayIcon={app}\gary4juce-control-center.exe
; Add version info
VersionInfoVersion=1.0.0.0
VersionInfoCompany=the collabage patch, inc.
VersionInfoDescription=Gary4JUCE Local Backend Services
VersionInfoProductName=Gary4JUCE Local Backend

[Files]
Source: "dist\gary4juce-installer.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\gary4juce-control-center.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "icon.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.txt"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; Create the main app directory
Name: "{app}"; Permissions: users-modify

[Icons]
; Start Menu shortcuts
Name: "{group}\Gary4JUCE Control Center"; Filename: "{app}\gary4juce-control-center.exe"; Comment: "Start and manage Gary4JUCE backend services"
Name: "{group}\Reinstall Backend Services"; Filename: "{app}\gary4juce-installer.exe"; Parameters: "--clean"; Comment: "Reinstall backend services from scratch"
Name: "{group}\{cm:UninstallProgram,Gary4JUCE Local Backend}"; Filename: "{uninstallexe}"
; Desktop shortcut (optional, user can uncheck during install)
Name: "{autodesktop}\Gary4JUCE Control Center"; Filename: "{app}\gary4juce-control-center.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
; Run installer to set up backend services (required)
; The installer will handle Python installation automatically if needed
Filename: "{app}\gary4juce-installer.exe"; Description: "Set up backend services (Python will be installed automatically if needed)"; Flags: runascurrentuser waituntilterminated
; Optionally launch control center after installation
Filename: "{app}\gary4juce-control-center.exe"; Description: "Launch Gary4JUCE Control Center"; Flags: postinstall nowait skipifsilent

[UninstallRun]
; FIXED: Only run the installer with --uninstall flag
; The installer handles stopping services internally
Filename: "{app}\gary4juce-installer.exe"; Parameters: "--uninstall"; Flags: runascurrentuser waituntilterminated

[UninstallDelete]
; Clean up any remaining files
Type: filesandordirs; Name: "{app}"

; REMOVED: All Python checking code - let gary4juce-installer.exe handle it