#ifndef MyAppVersion
  #error MyAppVersion must be supplied by scripts/build_installer.ps1
#endif
#ifndef MyAppNumericVersion
  #error MyAppNumericVersion must be supplied by scripts/build_installer.ps1
#endif

[Setup]
AppId={{EF13FFAD-8A48-48C4-950A-FD6ECA91AB5B}
AppName=MASHr FaceSwap
AppVersion={#MyAppVersion}
AppPublisher=MASHr Systems
AppPublisherURL=https://face.mashr.ai/
AppSupportURL=https://github.com/MASHR-SYSTEM/faceswap/issues
DefaultDirName={localappdata}\Programs\MASHr FaceSwap
DefaultGroupName=MASHr FaceSwap
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=MASHr-FaceSwap-{#MyAppVersion}-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=MASHr FaceSwap
VersionInfoVersion={#MyAppNumericVersion}
VersionInfoCompany=MASHr Systems
VersionInfoDescription=MASHr FaceSwap installer

[Files]
Source: "..\dist\FaceSwap\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\MASHr FaceSwap"; Filename: "{app}\FaceSwap.exe"
Name: "{autodesktop}\MASHr FaceSwap"; Filename: "{app}\FaceSwap.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\FaceSwap.exe"; Description: "Launch MASHr FaceSwap"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; User models, settings and logs under LocalAppData\MASHr\FaceSwap are intentionally retained.
