@echo off
echo SkySpotter Windows Build Script
echo ===============================
echo.

REM Check if virtual environment exists
if not exist "rawviewer_env" (
    echo Creating virtual environment...
    python -m venv rawviewer_env
    echo Virtual environment created.
)

REM Activate virtual environment
echo Activating virtual environment...
call rawviewer_env\Scripts\activate.bat

REM Download MobileCLIP2 ONNX models for bundling if missing
echo Checking MobileCLIP2 ONNX models...
python scripts/download_mobileclip_onnx.py

REM Install/upgrade dependencies
echo Installing dependencies...
pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread pyexiv2 Pillow psutil numpy qtawesome pyqtgraph sentence-transformers reverse-geocoder pycountry pywin32
pip install tokenizers==0.22.2

REM Try to close any running SkySpotter.exe instances
echo Checking for running SkySpotter instances...
taskkill /F /IM SkySpotter.exe /T >nul 2>&1
if %errorlevel% == 0 (
    echo Closed running RAWviewer.exe instances
    timeout /t 1 /nobreak >nul
)

REM Clean previous builds (build.py will handle this more gracefully, but we try here too)
echo Cleaning previous builds...
if exist build rmdir /s /q build 2>nul
if exist dist (
    REM Try to delete exe first
    if exist dist\SkySpotter.exe del /f /q dist\SkySpotter.exe 2>nul
    REM Then try to remove directory
    rmdir /s /q dist 2>nul
)
if exist *.spec del /q *.spec 2>nul

REM Build the application
echo Building SkySpotter...
python build.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed! Check the error messages above.
    pause
    exit /b %errorlevel%
)

echo.
echo Build completed successfully!
echo.
echo You can find the application at:
echo   - dist\SkySpotter\
echo.
echo To run the app:
echo   1. Double-click SkySpotter.exe in dist\SkySpotter\
echo   2. Run: dist\SkySpotter\SkySpotter.exe

pause 