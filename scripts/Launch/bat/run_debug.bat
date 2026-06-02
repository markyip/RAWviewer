@echo off
REM Run from repo root (scripts\Launch\bat -> ..\..\..)
cd /d "%~dp0..\..\.."

echo Running RAWviewer in debug mode...
echo To wipe caches/logs/settings: scripts\Launch\bat\clear_cache.bat
echo GPU single-image view: enabled (RAWVIEWER_GPU_VIEW=1)
echo All debug logs will be displayed in this console window.
echo.
echo Press Ctrl+C to stop the application.
echo.

set RAWVIEWER_USE_PROCESS_POOL=1
set RAWVIEWER_VERBOSE_INFO_LOGS=0
set RAWVIEWER_VERBOSE_CONSOLE=0
set RAWVIEWER_FOCUS_GALLERY_SWITCH=1
set RAWVIEWER_FILE_LOG=1
set RAWVIEWER_FATAL_DUMP=1
set RAWVIEWER_PROGRESSIVE_RAW_LOAD=1
set RAWVIEWER_NAV_PRELOAD_DISPLAY=1
set RAWVIEWER_AUTO_METADATA_INDEX=1
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0
set RAWVIEWER_INDEX_PAUSE_IN_GALLERY=1
REM GPU-accelerated single-image view (QGraphicsView + OpenGL). Set =0 to test legacy scroll area.
set RAWVIEWER_GPU_VIEW=1

where pixi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using Pixi environment...
    call pixi run python -u src/main.py %*
) else (
    REM Activate virtual environment if it exists
    if exist "rawviewer_env\Scripts\activate.bat" (
        echo Using virtual environment rawviewer_env...
        call rawviewer_env\Scripts\activate.bat
    ) else (
        echo Using system Python...
    )
    call python -u src/main.py %*
)
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %EXIT_CODE% EQU 0 (
    echo Application exited normally (code: %EXIT_CODE%)
) else (
    echo Application exited with error code: %EXIT_CODE%
    echo.
    echo Check logs in src\logs\ and %%LOCALAPPDATA%%\RAWviewer\logs.
)
echo ========================================
echo.

pause
