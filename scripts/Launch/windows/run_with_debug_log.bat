@echo off
setlocal EnableExtensions
REM Troubleshoot launcher for the *installed* Windows app.
REM Copied next to RAWviewer.exe by Setup (same as clear_cache.bat).
REM
REM What it does:
REM   - Enables full file logging (DEBUG) under %%LOCALAPPDATA%%\RAWviewer\logs
REM   - Turns on gallery/INDEX verbose lines useful for freeze diagnosis
REM   - Launches RAWviewer with --file-log
REM   - Opens the logs folder so you can grab the newest rawviewer_*.log
REM
REM Usage (after install):
REM   1. Double-click run_with_debug_log.bat in the install folder
REM   2. Reproduce the issue (session restore -> gallery)
REM   3. Quit RAWviewer
REM   4. Send the newest rawviewer_YYYYMMDD_HHMMSS.log from the logs folder
REM      (also rawviewer_latest.log points at the same session)

cd /d "%~dp0"

set "RAWVIEWER_FILE_LOG=1"
set "RAWVIEWER_REDIRECT_STDIO=1"
set "RAWVIEWER_VERBOSE_INFO_LOGS=1"
set "RAWVIEWER_FOCUS_GALLERY_SWITCH=1"
set "RAWVIEWER_FATAL_DUMP=1"
set "RAWVIEWER_DEBUG=1"

set "LOG_DIR=%LOCALAPPDATA%\RAWviewer\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

echo.
echo ========================================
echo  RAWviewer - debug log session
echo ========================================
echo.
echo Logs folder:
echo   %LOG_DIR%
echo.
echo Reproduce the bug, then quit the app.
echo Send the newest rawviewer_*.log (or rawviewer_latest.log).
echo.
echo Opening logs folder...
start "" explorer "%LOG_DIR%"

if not exist "%~dp0RAWviewer.exe" (
    echo ERROR: RAWviewer.exe not found next to this script.
    echo Run this from the install folder ^(usually %%LOCALAPPDATA%%\RAWviewer^).
    pause
    exit /b 1
)

echo Starting RAWviewer with file logging...
start "" "%~dp0RAWviewer.exe" --file-log %*
echo.
echo When finished, press any key to open the logs folder again.
pause >nul
start "" explorer "%LOG_DIR%"
endlocal
