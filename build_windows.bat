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
pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del *.spec

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