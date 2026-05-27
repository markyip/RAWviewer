@echo off
:: RAWviewer Uninstaller v1.0
setlocal
:: RAWviewer Silent Uninstaller
:: This script handles the automated removal of RAWviewer triggered by Windows Settings.

:: Standardize Path: Get current directory WITHOUT trailing backslash
set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"

:: 1. If running as CLEANUP (from TEMP), perform the wipe
if "%~1"=="__CLEANUP__" goto :CLEANUP

:: 2. Migration to TEMP
:: Since we can't delete the folder while this script is running from it,
:: we copy ourselves to TEMP and launch from there.
set "TEMP_SCRIPT=%TEMP%\uninstall_rawviewer_%RANDOM%.bat"
copy /y "%~f0" "%TEMP_SCRIPT%" >nul

:: Launch the second stage HIDDEN via PowerShell
set "EXE_CMD=powershell.exe -WindowStyle Hidden -Command \"& '%TEMP_SCRIPT%' __CLEANUP__ '%BASE_DIR%'\""
start "" %EXE_CMD%
exit /b

:CLEANUP
set "TARGET_DIR=%~2"

:: Wait a bit for the original launcher to exit
timeout /t 2 /nobreak >nul

:: 3. Kill Processes
:: Force-kill any lingering processes
taskkill /F /T /IM RAWviewer.exe >nul 2>&1
taskkill /F /T /IM RAWviewer_Setup.exe >nul 2>&1
taskkill /F /T /IM RAWviewer_Uninstaller.exe >nul 2>&1

:: Fallback: PowerShell commandline match for python windowless processes
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | Where-Object { $_.CommandLine -match 'src[/\\]main\.py' -or $_.CommandLine -match 'src[/\\]bootstrap\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1

:: 4. Remove Registry Keys
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RAWviewer" /f >nul 2>&1

:: 5. Remove Shortcuts
set "DESKTOP=%USERPROFILE%\Desktop"
if exist "%DESKTOP%\RAWviewer.lnk" del /f /q "%DESKTOP%\RAWviewer.lnk" >nul 2>&1

set "START_MENU_PROG=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
if exist "%START_MENU_PROG%\RAWviewer.lnk" del /f /q "%START_MENU_PROG%\RAWviewer.lnk" >nul 2>&1

:: 6. Wipe App Directory (with retry loop)
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

:: 7. Vanish (Self-delete the temp script)
(goto) 2>nul & del "%~f0"
