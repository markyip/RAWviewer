@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Shared Windows dev launcher.
REM Usage:
REM   launch_dev.bat lite|standard|full|directml|cuda [args...]
REM Repo root: scripts\Launch\windows -> ..\..\..
cd /d "%~dp0..\..\.."

set "MODE=%~1"
if "%MODE%"=="" set "MODE=full"

if /I "%MODE%"=="standard" set "MODE=lite"
if /I "%MODE%"=="plus" set "MODE=full"
if /I "%MODE%"=="dml" set "MODE=directml"

if /I not "%MODE%"=="lite" if /I not "%MODE%"=="full" if /I not "%MODE%"=="directml" if /I not "%MODE%"=="cuda" (
    echo Usage: %~nx0 lite^|standard^|full^|directml^|cuda [app args...]
    echo.
    echo   lite / standard  Standard edition ^(no semantic/face AI, CPU demosaic^)
    echo   full             Plus defaults ^(semantic on; GPU demosaic on if CuPy present^)
    echo   directml         Plus DirectML-like ^(semantic on, CPU demosaic^)
    echo   cuda             Plus CUDA-like ^(semantic on, CuPy GPU demosaic^)
    exit /b 2
)

REM Rebuild argv without the mode token (shift does not update %%*).
set "APP_ARGS="
set "_seen_mode=0"
for %%A in (%*) do (
    if "!_seen_mode!"=="0" (
        set "_seen_mode=1"
    ) else (
        set "APP_ARGS=!APP_ARGS! %%A"
    )
)

if exist "%~dp0..\..\..\.rawviewer_cold_start" (
    set RAWVIEWER_DISABLE_SESSION_RESTORE=1
    del /f /q "%~dp0..\..\..\.rawviewer_cold_start" >nul 2>&1
    echo [launch_dev] Cold start: session restore disabled for this launch ^(after clear_cache.bat^).
)

set RAWVIEWER_VERBOSE_INFO_LOGS=1
set RAWVIEWER_VERBOSE_CONSOLE=0
set RAWVIEWER_FILE_LOG=1
set RAWVIEWER_FATAL_DUMP=1
set RAWVIEWER_PROGRESSIVE_RAW_LOAD=1
set RAWVIEWER_NAV_PRELOAD_DISPLAY=1
set RAWVIEWER_AUTO_METADATA_INDEX=1
set RAWVIEWER_INDEX_PAUSE_IN_GALLERY=1
set RAWVIEWER_FOCUS_GALLERY_SWITCH=1
set RAWVIEWER_GPU_VIEW=1
set RAWVIEWER_PERF_V2=1
set RAWVIEWER_MOBILECLIP_VARIANT=b
set RAWVIEWER_USE_PROCESS_POOL=

if /I "%MODE%"=="lite" goto :mode_lite
if /I "%MODE%"=="directml" goto :mode_directml
if /I "%MODE%"=="cuda" goto :mode_cuda
goto :mode_full

:mode_lite
set RAWVIEWER_BUILD_PROFILE=lite
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0
set RAWVIEWER_ENABLE_FACE_SCAN=0
set RAWVIEWER_PREFER_GPU_DECODE=0
set RAWVIEWER_GPU_CUDA_GL=0
set RAWVIEWER_ORT_PROVIDERS=
echo [launch_dev] Profile: Standard ^(lite^) — no AI search; CPU Fast RAW
goto :run

:mode_directml
set RAWVIEWER_BUILD_PROFILE=full
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=1
set RAWVIEWER_ENABLE_FACE_SCAN=1
set RAWVIEWER_PREFER_GPU_DECODE=0
set RAWVIEWER_GPU_CUDA_GL=0
set RAWVIEWER_ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider
echo [launch_dev] Profile: Plus DirectML-like — AI search on DirectML; CPU demosaic
goto :run

:mode_cuda
set RAWVIEWER_BUILD_PROFILE=full
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=1
set RAWVIEWER_ENABLE_FACE_SCAN=1
set RAWVIEWER_PREFER_GPU_DECODE=1
set RAWVIEWER_GPU_CUDA_GL=1
set RAWVIEWER_ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider
echo [launch_dev] Profile: Plus CUDA-like — AI search on DirectML; CuPy GPU demosaic
goto :run

:mode_full
set RAWVIEWER_BUILD_PROFILE=full
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=1
set RAWVIEWER_ENABLE_FACE_SCAN=1
set RAWVIEWER_PREFER_GPU_DECODE=1
set RAWVIEWER_GPU_CUDA_GL=1
set RAWVIEWER_ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider
echo [launch_dev] Profile: Plus ^(full^) — semantic on; prefer GPU demosaic when available
goto :run

:run
echo To wipe caches: scripts\Launch\windows\clear_cache.bat
echo.
echo Press Ctrl+C to stop.
echo.

where pixi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    if exist "%~dp0..\..\..\.pixi\envs\default\python.exe" (
        echo Using Pixi environment...
        "%~dp0..\..\..\.pixi\envs\default\python.exe" -u "%~dp0..\..\..\src\main.py" !APP_ARGS!
    ) else (
        echo Using pixi run...
        call pixi run python -u "%~dp0..\..\..\src\main.py" !APP_ARGS!
    )
) else (
    if exist "%~dp0..\..\..\rawviewer_env\Scripts\activate.bat" (
        echo Using virtual environment rawviewer_env...
        call "%~dp0..\..\..\rawviewer_env\Scripts\activate.bat"
    ) else (
        echo Using system Python...
    )
    call python -u "%~dp0..\..\..\src\main.py" !APP_ARGS!
)

set EXIT_CODE=%ERRORLEVEL%
echo.
echo ========================================
if %EXIT_CODE% EQU 0 (
    echo Application exited normally ^(code: %EXIT_CODE%^)
) else (
    echo Application exited with error code: %EXIT_CODE%
    echo Check logs in src\logs\ and %%LOCALAPPDATA%%\RAWviewer\logs.
)
echo ========================================
echo.
pause
exit /b %EXIT_CODE%
