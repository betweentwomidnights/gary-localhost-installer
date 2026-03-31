; Installer copy for the default current-user distribution model.
; This file is included before Tauri defines PRODUCTNAME, so use explicit
; strings here unless we move to a fully custom NSIS template.

!define INSTALLER_AUDIO_SOURCE "${__FILEDIR__}\installer-audio.wav"

!define SND_ASYNC 0x0001
!define SND_NODEFAULT 0x0002
!define SND_LOOP 0x0008
!define SND_FILENAME 0x00020000

Var InstallerMusicMuted
Var InstallerMusicCheckbox

!define MUI_CUSTOMFUNCTION_GUIINIT InstallerGuiInit
!define MUI_PAGE_CUSTOMFUNCTION_SHOW WelcomeMusicShow

!define MUI_WELCOMEPAGE_TEXT "this installer uses current-user mode, so it does not require administrator access.$\r$\n$\r$\napplication files are installed under your windows user profile, usually:$\r$\n$\r$\n    $LOCALAPPDATA\gary4local$\r$\n$\r$\nruntime data, service environments, logs, and local models are stored in:$\r$\n$\r$\n    $APPDATA\Gary4JUCE$\r$\n$\r$\nthis means gary4local will not install to C:\Program Files in this mode."

!define MUI_DIRECTORYPAGE_TEXT_TOP "choose where to place the installed application files.$\r$\n$\r$\ninstalled app files live in the folder you choose here.$\r$\n$\r$\nruntime data, service environments, logs, and local models stay in:$\r$\n$\r$\n    $APPDATA\Gary4JUCE"

!define MUI_FINISHPAGE_TEXT "setup has finished installing gary4local.$\r$\n$\r$\ninstalled app files:$\r$\n$INSTDIR$\r$\n$\r$\nruntime data:$\r$\n$APPDATA\Gary4JUCE$\r$\n$\r$\nto uninstall later, use windows installed apps, run uninstall.exe from the install folder, or run this setup again to enter maintenance mode."

Function InstallerGuiInit
  InitPluginsDir
  File "/oname=$PLUGINSDIR\installer-audio.wav" "${INSTALLER_AUDIO_SOURCE}"
  StrCmp $InstallerMusicMuted 1 +2 0
    Call StartInstallerMusic
FunctionEnd

Function .onGUIEnd
  Call StopInstallerMusic
FunctionEnd

Function StartInstallerMusic
  StrCmp $InstallerMusicMuted 1 done
  IfFileExists "$PLUGINSDIR\installer-audio.wav" 0 done
  System::Call 'winmm.dll::PlaySound(t "$PLUGINSDIR\installer-audio.wav", p 0, i ${SND_ASYNC}|${SND_LOOP}|${SND_FILENAME}|${SND_NODEFAULT}) i .r0'
done:
FunctionEnd

Function StopInstallerMusic
  System::Call 'winmm.dll::PlaySound(p 0, p 0, i 0) i .r0'
FunctionEnd

Function WelcomeMusicShow
  ${NSD_CreateCheckbox} 120u 179u 96u 10u "mute music"
  Pop $InstallerMusicCheckbox
  ${NSD_OnClick} $InstallerMusicCheckbox WelcomeMusicToggle

  StrCmp $InstallerMusicMuted 1 checked unchecked
checked:
  SendMessage $InstallerMusicCheckbox ${BM_SETCHECK} ${BST_CHECKED} 0
  Return
unchecked:
  SendMessage $InstallerMusicCheckbox ${BM_SETCHECK} ${BST_UNCHECKED} 0
FunctionEnd

Function WelcomeMusicToggle
  ${NSD_GetState} $InstallerMusicCheckbox $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $InstallerMusicMuted 1
    Call StopInstallerMusic
  ${Else}
    StrCpy $InstallerMusicMuted 0
    Call StartInstallerMusic
  ${EndIf}
FunctionEnd
