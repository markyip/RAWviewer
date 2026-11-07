@echo off
REM Run from repo root (scripts\Launch\windows -> ..\..\..)
cd /d "%~dp0..\..\.."

set "SETUP_EXE=RAWviewer_Setup.exe"

echo RAWviewer Windows Build Script (Unified Installer)
echo ==================================================
echo Output:  dist\%SETUP_EXE%
echo.
echo Dependency manager: uv when available, otherwise pip.
echo   Force pip:    set RAWVIEWER_USE_UV=0
echo   Require uv:   set RAWVIEWER_USE_UV=1
echo.

set "RAWVIEWER_VENV_USE_UV=1"
if /I "%RAWVIEWER_USE_UV%"=="0" set "RAWVIEWER_VENV_USE_UV=0"

if not exist "rawviewer_env" (
    if "%RAWVIEWER_VENV_USE_UV%"=="0" (
        echo Creating virtual environment with python -m venv...
        python -m venv rawviewer_env
    ) else (
        where uv >nul 2>nul
        if errorlevel 1 (
            echo Creating virtual environment with python -m venv...
            python -m venv rawviewer_env
        ) else (
            echo Creating virtual environment with uv...
            uv venv rawviewer_env --seed
            if errorlevel 1 (
                echo [WARNING] uv venv failed; falling back to python -m venv.
                python -m venv rawviewer_env
            )
        )
    )
    echo Virtual environment created.
)

echo Activating virtual environment...
call rawviewer_env\Scripts\activate.bat

echo Checking MobileCLIP2 ONNX models...
python scripts/models/download_mobileclip_onnx.py

echo Checking for running RAWviewer instances...
taskkill /F /IM RAWviewer.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_CUDA.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_DirectML.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_Lite.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul

echo Cleaning previous builds...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist *.spec del /q *.spec 2>nul

REM Ensure git submodules for LibRaw (ARW6 / new Sony) before packaging.
if exist ".gitmodules" (
    echo Ensuring third_party LibRaw submodules...
    git submodule update --init --recursive third_party/LibRaw third_party/LibRaw-cmake
    if errorlevel 1 (
        echo [ERROR] git submodule update failed. Clone with submodules or run:
        echo   git submodule update --init --recursive
        pause
        exit /b 1
    )
)

echo Building Windows unified installer and launcher stub...
echo   (build.py will swap an ARW6-capable LibRaw into rawpy — see scripts\libraw\)
python build.py --profile full
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed! Check the error messages above.
    pause
    exit /b %errorlevel%
)

echo.
echo Build completed successfully!
echo.
echo Upload to GitHub Releases:
echo   dist\%SETUP_EXE%
echo.
echo End users run the Setup exe once, then launch RAWviewer.exe from:
echo   - Desktop shortcut
echo   - %%LOCALAPPDATA%%\RAWviewer\RAWviewer.exe

pause
