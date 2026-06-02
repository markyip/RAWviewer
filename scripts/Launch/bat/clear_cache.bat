@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Full cache + session wipe for RAWviewer (dev and installed builds).
REM Removes: ~/.rawviewer_cache, logs, dev logs, ALL QSettings (HKCU\Software\RAWviewer).
REM Does NOT remove: %%LOCALAPPDATA%%\RAWviewer app install (RAWviewer.exe, bundled models).
cd /d "%~dp0..\..\.."

echo.
echo ========================================
echo  RAWviewer - clear ALL cache and state
echo ========================================
echo.

call :KillApp
timeout /t 2 /nobreak >nul

set "CLEARED=0"
set "FAILED=0"

call :RemoveTree "%USERPROFILE%\.rawviewer_cache" "image/EXIF/semantic/thumbnail cache"
call :RemoveTree "%LOCALAPPDATA%\RAWviewer\logs" "runtime logs (%%LOCALAPPDATA%%\RAWviewer\logs)"
call :RemoveTree "%APPDATA%\RAWviewer\logs" "roaming logs (%%APPDATA%%\RAWviewer\logs)"
call :RemoveTree "src\logs" "repository dev logs (src\logs)"

REM Optional cache subfolders under the install root (if present).
call :RemoveTree "%LOCALAPPDATA%\RAWviewer\cache" "install cache folder"
call :RemoveTree "%LOCALAPPDATA%\RAWviewer\CrashDumps" "crash dumps"

echo Clearing QSettings / session registry (window, sort, last folder, rotations, semantic flags)...
reg query "HKCU\Software\RAWviewer" >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    reg delete "HKCU\Software\RAWviewer" /f >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "CLEARED=1"
        echo   Removed HKCU\Software\RAWviewer
    ) else (
        echo   WARNING: Could not remove registry key HKCU\Software\RAWviewer
        set "FAILED=1"
    )
) else (
    echo   Registry already clean
)

echo.
echo Verifying cold-start state...
if exist "%USERPROFILE%\.rawviewer_cache" (
    echo   WARNING: %%USERPROFILE%%\.rawviewer_cache still exists
    set "FAILED=1"
) else (
    echo   OK: %%USERPROFILE%%\.rawviewer_cache removed
)
reg query "HKCU\Software\RAWviewer" /v last_session_folder >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo   WARNING: last_session_folder still in HKCU\Software\RAWviewer
    set "FAILED=1"
) else (
    echo   OK: no last_session_folder in registry
)

if "!FAILED!"=="0" (
    echo.>.rawviewer_cold_start
    echo   Next run_debug.bat will skip session restore ^(one launch^).
)

echo.
if "!FAILED!"=="1" (
    echo Finished with warnings. Close RAWviewer and any Python dev instance, then run again.
) else if "!CLEARED!"=="1" (
    echo Cache and session state cleared.
    echo For a true UI cold start: run scripts\Launch\bat\run_debug.bat next.
) else (
    echo Nothing found to clear ^(already clean^).
    echo.>.rawviewer_cold_start
)
echo.
echo Not removed: %%LOCALAPPDATA%%\RAWviewer application files ^(exe, installer models^).
echo Manual override: set RAWVIEWER_DISABLE_SESSION_RESTORE=1 before run_debug.bat
echo.
pause
exit /b 0

:KillApp
echo Closing RAWviewer if running...
taskkill /IM RAWviewer.exe /F >nul 2>&1
REM Dev runs via python main.py — stop common launcher patterns.
for /f "tokens=2" %%P in ('wmic process where "name='python.exe' and CommandLine like '%%main.py%%'" get ProcessId 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /PID %%P /F >nul 2>&1
)
for /f "tokens=2" %%P in ('wmic process where "name='python.exe' and CommandLine like '%%RAWviewer%%'" get ProcessId 2^>nul ^| findstr /r "[0-9]"') do (
    taskkill /PID %%P /F >nul 2>&1
)
exit /b 0

:RemoveTree
set "TARGET=%~1"
set "LABEL=%~2"
if not exist "%TARGET%" exit /b 0
echo Removing %LABEL%:
echo   %TARGET%
set "TRIES=0"
:RemoveTreeRetry
set /a TRIES+=1
rmdir /S /Q "%TARGET%" 2>nul
if exist "%TARGET%" (
    if !TRIES! LSS 5 (
        timeout /t 1 /nobreak >nul
        goto RemoveTreeRetry
    )
    echo   WARNING: Could not fully remove - file may be locked.
    set "FAILED=1"
) else (
    set "CLEARED=1"
)
exit /b 0
