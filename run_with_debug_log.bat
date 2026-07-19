@echo off
setlocal EnableExtensions
REM Prefer the installed app (troubleshoot built builds). Fall back to CUDA dev launch.

set "INSTALLED=%LOCALAPPDATA%\RAWviewer\RAWviewer.exe"
set "INSTALLED_BAT=%LOCALAPPDATA%\RAWviewer\run_with_debug_log.bat"

if exist "%INSTALLED_BAT%" (
    call "%INSTALLED_BAT%" %*
    exit /b %ERRORLEVEL%
)

if exist "%INSTALLED%" (
    set "RAWVIEWER_FILE_LOG=1"
    set "RAWVIEWER_REDIRECT_STDIO=1"
    set "RAWVIEWER_VERBOSE_INFO_LOGS=1"
    set "RAWVIEWER_FOCUS_GALLERY_SWITCH=1"
    set "RAWVIEWER_FATAL_DUMP=1"
    set "RAWVIEWER_DEBUG=1"
    set "LOG_DIR=%LOCALAPPDATA%\RAWviewer\logs"
    if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
    echo Starting installed RAWviewer with debug logging...
    echo Logs: %LOG_DIR%
    start "" explorer "%LOG_DIR%"
    start "" "%INSTALLED%" --file-log %*
    exit /b 0
)

echo No installed RAWviewer found. Launching CUDA dev with file logging...
set "RAWVIEWER_FILE_LOG=1"
set "RAWVIEWER_REDIRECT_STDIO=1"
set "RAWVIEWER_VERBOSE_INFO_LOGS=1"
set "RAWVIEWER_FOCUS_GALLERY_SWITCH=1"
set "RAWVIEWER_FATAL_DUMP=1"
set "RAWVIEWER_DEBUG=1"
call "%~dp0scripts\Launch\windows\launch_dev_cuda.bat" --file-log %*
endlocal
