@echo off
REM Run from repo root (scripts\Launch\bat -> ..\..\..)
cd /d "%~dp0..\..\.."

set "ACCEL=%~1"
if "%ACCEL%"=="" set "ACCEL=cuda"
if /I not "%ACCEL%"=="cuda" if /I not "%ACCEL%"=="directml" (
    echo [ERROR] Invalid backend "%ACCEL%". Use: cuda or directml
    exit /b 1
)

echo RAWviewer Windows Build Script
echo ===============================
echo Backend: %ACCEL%
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

echo Building RAWviewer (GPU single-image viewport enabled by default)...
python build.py --windows-accel %ACCEL%
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
