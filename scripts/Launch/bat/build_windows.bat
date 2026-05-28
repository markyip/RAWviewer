@echo off
REM Run from repo root (scripts\Launch\bat -> ..\..\..)
cd /d "%~dp0..\..\.."

echo RAWviewer Windows Build Script
echo ===============================
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

echo Installing dependencies...
pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread pyexiv2 Pillow psutil numpy qtawesome pyqtgraph onnxruntime-directml reverse-geocoder pycountry pywin32 opencv-python-headless huggingface-hub requests

echo Checking for running RAWviewer instances...
taskkill /F /IM RAWviewer.exe /T >nul 2>&1
if %errorlevel% == 0 (
    echo Closed running RAWviewer.exe instances
    timeout /t 1 /nobreak >nul
)

echo Cleaning previous builds...
if exist build rmdir /s /q build 2>nul
if exist dist (
    if exist dist\RAWviewer.exe del /f /q dist\RAWviewer.exe 2>nul
    rmdir /s /q dist 2>nul
)
if exist *.spec del /q *.spec 2>nul

echo Building RAWviewer...
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
echo You can find the executable at:
echo   - dist\RAWviewer.exe
echo.
echo To run the app:
echo   1. Double-click RAWviewer.exe in File Explorer
echo   2. Run: dist\RAWviewer.exe
echo   3. Run: .\dist\RAWviewer.exe

pause
