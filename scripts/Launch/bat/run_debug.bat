@echo off
REM Run from repo root (scripts\Launch\bat -> ..\..\..)
cd /d "%~dp0..\..\.."

if exist "%~dp0..\..\..\.rawviewer_cold_start" (
    set RAWVIEWER_DISABLE_SESSION_RESTORE=1
    del /f /q "%~dp0..\..\..\.rawviewer_cold_start" >nul 2>&1
    echo [run_debug] Cold start: session restore disabled for this launch ^(after clear_cache.bat^).
)

echo Running RAWviewer in debug mode...
echo To wipe caches/logs/settings: scripts\Launch\bat\clear_cache.bat
echo GPU single-image view: enabled (RAWVIEWER_GPU_VIEW=1)
echo GPU demosaic: enabled (RAWVIEWER_PREFER_GPU_DECODE=1)
echo CUDA-GL display: enabled (RAWVIEWER_GPU_CUDA_GL=1)
echo Semantic ONNX backend: CUDA preferred with DirectML fallback (Model variant B)
echo All debug logs will be displayed in this console window.
echo.
echo Press Ctrl+C to stop the application.
echo.

set RAWVIEWER_USE_PROCESS_POOL=0
set RAWVIEWER_VERBOSE_INFO_LOGS=1
set RAWVIEWER_VERBOSE_CONSOLE=0
set RAWVIEWER_FOCUS_GALLERY_SWITCH=1
set RAWVIEWER_FILE_LOG=1
set RAWVIEWER_FATAL_DUMP=1
set RAWVIEWER_PROGRESSIVE_RAW_LOAD=1
set RAWVIEWER_NAV_PRELOAD_DISPLAY=1
set RAWVIEWER_AUTO_METADATA_INDEX=1
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=1
set RAWVIEWER_MOBILECLIP_VARIANT=b
set RAWVIEWER_INDEX_PAUSE_IN_GALLERY=1
set RAWVIEWER_ORT_PROVIDERS=CUDAExecutionProvider,DmlExecutionProvider,CPUExecutionProvider
set RAWVIEWER_GPU_VIEW=1
set RAWVIEWER_PREFER_GPU_DECODE=1
set RAWVIEWER_GPU_CUDA_GL=1

where pixi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Using Pixi environment...
    if exist "%~dp0..\..\..\.pixi\envs\default\python.exe" (
        "%~dp0..\..\..\.pixi\envs\default\python.exe" -u "%~dp0..\..\..\src\main.py" %*
    ) else (
        call pixi run python -u "%~dp0..\..\..\src\main.py" %*
    )
) else (
    if exist "%~dp0..\..\..\rawviewer_env\Scripts\activate.bat" (
        echo Using virtual environment rawviewer_env...
        call "%~dp0..\..\..\rawviewer_env\Scripts\activate.bat"
    ) else (
        echo Using system Python...
    )
    call python -u "%~dp0..\..\..\src\main.py" %*
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
