@echo off
echo RAWviewer Windows Build Script
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

REM Install/upgrade dependencies
echo Installing dependencies...
pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow psutil numpy qtawesome pyqtgraph

REM Try to close any running RAWviewer.exe instances
echo Checking for running RAWviewer instances...
taskkill /F /IM RAWviewer.exe /T >nul 2>&1
if %errorlevel% == 0 (
    echo Closed running RAWviewer.exe instances
    timeout /t 1 /nobreak >nul
)

REM Clean previous builds (build.py will handle this more gracefully, but we try here too)
echo Cleaning previous builds...
if exist build rmdir /s /q build 2>nul
if exist dist (
    REM Try to delete exe first
    if exist dist\RAWviewer.exe del /f /q dist\RAWviewer.exe 2>nul
    REM Then try to remove directory
    rmdir /s /q dist 2>nul
)
if exist *.spec del /q *.spec 2>nul

REM Build the application
echo Building RAWviewer...
python build.py

echo.
echo Build completed!
echo.
echo You can find the executable at:
echo   - dist\RAWviewer.exe
echo.
echo To run the app:
echo   1. Double-click RAWviewer.exe in File Explorer
echo   2. Run: dist\RAWviewer.exe
echo   3. Run: .\dist\RAWviewer.exe

pause 