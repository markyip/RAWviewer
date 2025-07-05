@echo off
echo RAW Image Viewer Build Script
echo ==============================
echo.
echo This will build a standalone Windows executable for RAW Image Viewer
echo.
pause
echo.
echo Starting build process...
echo.
REM Build the RAWviewer executable with icon

REM Check if .venv exists
if not exist .venv\Scripts\activate.bat (
    echo [ERROR] .venv not found! Please create the virtual environment first.
    exit /b 1
)

REM Check if venv is already activated (by checking VIRTUAL_ENV)
if "%VIRTUAL_ENV%"=="" (
    echo [INFO] Activating .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] .venv already activated.
)

REM Run PyInstaller with icon
pyinstaller --onefile --windowed --icon "D:/Development/RAWviewer/appicon.ico" src/main.py --name RAWviewer

echo.
echo Build process completed.
echo.
pause