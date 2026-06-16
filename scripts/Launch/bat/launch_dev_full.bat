@echo off
REM Launch RAWviewer from source in full profile (semantic search + face scan).
cd /d "%~dp0..\..\.."

set RAWVIEWER_BUILD_PROFILE=full
set RAWVIEWER_ENABLE_SEMANTIC_SEARCH=1
set RAWVIEWER_ENABLE_FACE_SCAN=1
set RAWVIEWER_AUTO_METADATA_INDEX=1
set RAWVIEWER_TEST_SEMANTIC=1
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

echo Launching RAWviewer FULL from %CD% (semantic + face enabled)...
python src\main.py %*
