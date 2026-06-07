@echo off
:: RAWviewer Uninstaller v1.2
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

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | Where-Object { $_.CommandLine -match 'src[/\\]main\.py' -or $_.CommandLine -match 'src[/\\]bootstrap\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RAWviewer" /f >nul 2>&1

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

set "UNINSTALL_MSG=RAWviewer has been removed."
if exist "%TARGET_DIR%" (
    set "UNINSTALL_MSG=RAWviewer was removed from Windows, but some files could not be deleted. Close RAWviewer if it is still running, then delete the install folder manually."
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; [System.Windows.Forms.MessageBox]::Show($env:UNINSTALL_MSG,'RAWviewer Uninstall','OK','Information') | Out-Null"

(goto) 2>nul & del "%~f0"
