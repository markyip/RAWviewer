@echo off
title RAWviewer - GPU Accelerated Dev Launcher
echo ========================================================
echo   Starting RAWviewer in GPU-Accelerated Development Mode
echo   Using Conda Environment: Base
echo ========================================================
echo.

set PYTHON_EXE=C:\ProgramData\miniconda3\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Could not find python.exe at %PYTHON_EXE%
    echo Please verify that Miniconda is installed under C:\ProgramData\miniconda3
    pause
    exit /b 1
)

echo [INFO] Launching app...
"%PYTHON_EXE%" src\main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application crashed or exited with error code %ERRORLEVEL%.
    pause
)
