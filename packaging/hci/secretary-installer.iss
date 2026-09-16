#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{735DA49C-B833-4DBB-9C90-198C79B6C83C}
AppName=Relic Secretary
AppVersion={#MyAppVersion}
AppPublisher=Relic
DefaultDirName={localappdata}\Programs\Relic Secretary
DefaultGroupName=Relic Secretary
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=release
OutputBaseFilename=Relic-Secretary-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\Secretary.exe

[Files]
Source: "build\dist\Secretary\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Icons]
Name: "{autoprograms}\Relic P2 Transparent"; Filename: "{app}\Secretary.exe"; Parameters: "--p2"
Name: "{autoprograms}\Relic P3 Secretary"; Filename: "{app}\Secretary.exe"; Parameters: "--p3"
Name: "{autodesktop}\Relic P2 Transparent"; Filename: "{app}\Secretary.exe"; Parameters: "--p2"; Tasks: desktopicon
Name: "{autodesktop}\Relic P3 Secretary"; Filename: "{app}\Secretary.exe"; Parameters: "--p3"; Tasks: desktopicon

[Run]
Filename: "{app}\Secretary.exe"; Parameters: "--p3"; Description: "Launch Relic P3 Secretary"; Flags: nowait postinstall skipifsilent
