@echo off
REM Launch RAWviewer from source in lite profile (no semantic/face AI).
cd /d "%~dp0..\..\.."

set RAWVIEWER_BUILD_PROFILE=lite
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0
set RAWVIEWER_ENABLE_FACE_SCAN=0
set RAWVIEWER_AUTO_METADATA_INDEX=1
set RAWVIEWER_TEST_SEMANTIC=0
set RAWVIEWER_ENABLE_LOCATION_MAP=0
if not defined RAWVIEWER_PREVIEW_CACHE_ADAPTIVE set RAWVIEWER_PREVIEW_CACHE_ADAPTIVE=1
if not defined RAWVIEWER_MEMORY_PREVIEW_MAX set RAWVIEWER_MEMORY_PREVIEW_MAX=2304
set RAWVIEWER_GPU_VIEW=1
set RAWVIEWER_DEBUG=1
set PYTHONPATH=%CD%\src;%PYTHONPATH%

where pixi >nul 2>nul
if %ERRORLEVEL% EQU 0 if exist "pixi.toml" if exist ".pixi\envs\default\python.exe" (
    echo Using pixi env ^(.pixi\envs\default^)...
    call pixi run python src\main.py %*
    exit /b %ERRORLEVEL%
)

if exist "rawviewer_env\Scripts\activate.bat" (
    call rawviewer_env\Scripts\activate.bat
)

echo Launching RAWviewer LITE from %CD% (metadata/GPS search only, semantic off)...
python src\main.py %*
