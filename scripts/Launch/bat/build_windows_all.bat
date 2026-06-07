@echo off
REM Build both Windows installer variants (DirectML + CUDA) in one run.
cd /d "%~dp0..\..\.."

echo RAWviewer Windows Build Script (DirectML + CUDA)
echo ================================================
echo Outputs:
echo   dist\RAWviewer_Setup_DirectML.exe
echo   dist\RAWviewer_Setup_CUDA.exe
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
if %errorlevel% neq 0 (
    echo [ERROR] MobileCLIP model check failed.
    pause
    exit /b %errorlevel%
)

echo Checking for running RAWviewer instances...
taskkill /F /IM RAWviewer.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_CUDA.exe /T >nul 2>&1
taskkill /F /IM RAWviewer_Setup_DirectML.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul

echo Cleaning previous builds...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist *.spec del /q *.spec 2>nul

echo.
echo [1/2] Building DirectML installer...
python build.py --windows-accel directml
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] DirectML build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo [2/2] Building CUDA installer (keeping DirectML output)...
python build.py --windows-accel cuda --keep-dist
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] CUDA build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo Build completed successfully!
echo.
echo Upload both files to GitHub Releases:
echo   dist\RAWviewer_Setup_DirectML.exe
echo   dist\RAWviewer_Setup_CUDA.exe

pause
