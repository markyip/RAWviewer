@echo off

:: RAWviewer Uninstaller v1.3

setlocal EnableExtensions

:: Handles removal from the install folder or Windows Settings (UninstallString).



set "BASE_DIR=%~dp0"

if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"



if /I not "%~1"=="__CLEANUP__" goto :STAGE1

goto :CLEANUP



:STAGE1

set "TEMP_SCRIPT=%TEMP%\uninstall_rawviewer_%RANDOM%.bat"

copy /y "%~f0" "%TEMP_SCRIPT%" >nul

if errorlevel 1 (

    echo Failed to prepare uninstaller.

    pause

    exit /b 1

)

start "" /min "%ComSpec%" /c call "%TEMP_SCRIPT%" __CLEANUP__ "%BASE_DIR%"

exit /b 0



:CLEANUP

set "TARGET_DIR=%~2"

if not defined TARGET_DIR set "TARGET_DIR=%BASE_DIR%"



timeout /t 2 /nobreak >nul



taskkill /F /T /IM RAWviewer.exe >nul 2>&1

taskkill /F /T /IM RAWviewer_Setup.exe >nul 2>&1

taskkill /F /T /IM RAWviewer_Uninstaller.exe >nul 2>&1



set "TARGET_ESC=%TARGET_DIR:\=\\%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$td = $env:TARGET_ESC; Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'pixi.exe'\" | Where-Object { $_.CommandLine -match 'src[/\\]main\.py' -or $_.CommandLine -match 'src[/\\]bootstrap\.py' -or ($td -and $_.CommandLine -like ('*' + $td + '*')) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1



reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RAWviewer" /f >nul 2>&1

:: Remove file associations (Open With); keep HKCU\Software\RAWviewer QSettings unless FULL uninstall.
reg delete "HKCU\Software\Classes\Applications\RAWviewer.exe" /f >nul 2>&1
reg delete "HKCU\Software\Classes\RAWviewer.Image" /f >nul 2>&1
reg delete "HKCU\Software\RAWviewer\Capabilities" /f >nul 2>&1
reg delete "HKCU\Software\RegisteredApplications" /v RAWviewer /f >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$progId = 'RAWviewer.Image';" ^
  "$exts = @('.cr2','.cr3','.nef','.arw','.dng','.orf','.rw2','.pef','.srw','.x3f','.raf','.3fr','.fff','.iiq','.cap','.erf','.mef','.mos','.nrw','.rwl','.srf','.jpeg','.jpg','.png','.webp','.heif','.heic','.tif','.tiff');" ^
  "foreach ($ext in $exts) {" ^
  "  $owp = \"HKCU:\\Software\\Classes\\$ext\\OpenWithProgids\";" ^
  "  if (Test-Path -LiteralPath $owp) { Remove-ItemProperty -LiteralPath $owp -Name $progId -ErrorAction SilentlyContinue };" ^
  "  $extKey = \"HKCU:\\Software\\Classes\\$ext\";" ^
  "  if (Test-Path -LiteralPath $extKey) {" ^
  "    $def = (Get-ItemProperty -LiteralPath $extKey -Name '(default)' -ErrorAction SilentlyContinue).'(default)';" ^
  "    if ($def -eq $progId) { Remove-ItemProperty -LiteralPath $extKey -Name '(default)' -ErrorAction SilentlyContinue }" ^
  "  };" ^
  "  $owl = \"HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\$ext\\OpenWithList\";" ^
  "  if (Test-Path -LiteralPath $owl) {" ^
  "    $props = Get-ItemProperty -LiteralPath $owl;" ^
  "    foreach ($name in @($props.PSObject.Properties.Name)) {" ^
  "      if ($name -in @('PSPath','PSParentPath','PSChildName','PSDrive','PSProvider')) { continue };" ^
  "      if ($props.$name -eq 'RAWviewer.exe') { Remove-ItemProperty -LiteralPath $owl -Name $name -ErrorAction SilentlyContinue }" ^
  "    }" ^
  "  }" ^
  "}" >nul 2>&1

if exist "%USERPROFILE%\.rawviewer_cache" rd /s /q "%USERPROFILE%\.rawviewer_cache" >nul 2>&1

if /I "%RAWVIEWER_UNINSTALL_FULL%"=="1" (
    reg delete "HKCU\Software\RAWviewer" /f >nul 2>&1
)



set "DESKTOP=%USERPROFILE%\Desktop"

if exist "%DESKTOP%\RAWviewer.lnk" del /f /q "%DESKTOP%\RAWviewer.lnk" >nul 2>&1



set "START_MENU_PROG=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

if exist "%START_MENU_PROG%\RAWviewer.lnk" del /f /q "%START_MENU_PROG%\RAWviewer.lnk" >nul 2>&1



set "RETRY=0"

:RETRY_LOOP

if exist "%TARGET_DIR%" (

    rd /s /q "%TARGET_DIR%" >nul 2>&1

    if exist "%TARGET_DIR%" (

        set /a RETRY+=1

        if %RETRY% LSS 10 (

            timeout /t 2 /nobreak >nul

            goto :RETRY_LOOP

        )

    )

)



set "LOGS_DIR=%LOCALAPPDATA%\RAWviewer\logs"
if exist "%LOGS_DIR%" rd /s /q "%LOGS_DIR%" >nul 2>&1
if exist "%LOCALAPPDATA%\RAWviewer\map_tiles" rd /s /q "%LOCALAPPDATA%\RAWviewer\map_tiles" >nul 2>&1
if exist "%LOCALAPPDATA%\RAWviewer" (
    rd /s /q "%LOCALAPPDATA%\RAWviewer" >nul 2>&1
)

set "UNINSTALL_MSG=RAWviewer has been removed."

if exist "%TARGET_DIR%" (

    set "UNINSTALL_MSG=RAWviewer was removed from Windows, but some files could not be deleted. Close RAWviewer if it is still running, then delete the install folder manually."

)
if /I "%RAWVIEWER_UNINSTALL_FULL%"=="1" (
    set "UNINSTALL_MSG=%UNINSTALL_MSG% User cache and preferences were also removed."
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; [System.Windows.Forms.MessageBox]::Show($env:UNINSTALL_MSG,'RAWviewer Uninstall','OK','Information') | Out-Null"



(goto) 2>nul & del "%~f0"

