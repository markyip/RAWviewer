@echo off
REM Build RAWviewer Lite for Windows (no semantic/face AI; EXIF+GPS search, no map overlay).
cd /d "%~dp0..\..\.."

set "SETUP_EXE=RAWviewer_Setup_Lite.exe"

echo RAWviewer Windows Lite Build Script
echo ===================================
echo Profile: lite — viewing + metadata/GPS gallery search + map (no MobileCLIP / face scan).
echo Output:  dist\%SETUP_EXE%
echo.

if not exist "rawviewer_env" (
    echo Creating virtual environment...
    python -m venv rawviewer_env
    echo Virtual environment created.
)

echo Activating virtual environment...
call rawviewer_env\Scripts\activate.bat

echo Skipping MobileCLIP ONNX models (not used in lite profile)...

echo Checking for running RAWviewer instances...
taskkill /F /IM RAWviewer.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_CUDA.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_DirectML.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_Lite.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul

echo Cleaning previous builds...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist *.spec del /q *.spec 2>nul

echo Building Windows lite installer and launcher stub...
set RAWVIEWER_BUILD_PROFILE=lite
python build.py --profile lite
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Lite build failed! Check the error messages above.
    pause
    exit /b %errorlevel%
)

echo.
echo Lite build completed successfully!
echo.
echo Upload to GitHub Releases:
echo   dist\%SETUP_EXE%
echo.
echo End users run the Setup exe once, then launch RAWviewer.exe from:
echo   - Desktop shortcut
echo   - %%LOCALAPPDATA%%\RAWviewer\RAWviewer.exe
echo.
echo Dev run from source: scripts\Launch\bat\launch_dev_lite.bat

pause
