@echo off
REM Run from repo root (scripts\Launch\bat -> ..\..\..)
cd /d "%~dp0..\..\.."

set "SETUP_EXE=RAWviewer_Setup.exe"

echo RAWviewer Windows Build Script (Unified Installer)
echo ==================================================
echo Output:  dist\%SETUP_EXE%
echo.

if not exist "rawviewer_env" (
    echo Creating virtual environment...
    python -m venv rawviewer_env
    echo Virtual environment created.
)

echo Activating virtual environment...
call rawviewer_env\Scripts\activate.bat

echo Checking MobileCLIP2 ONNX models...
python scripts/download_mobileclip_onnx.py

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

echo Building Windows unified installer and launcher stub...
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
