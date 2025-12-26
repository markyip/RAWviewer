@echo off
echo Running RAWviewer in debug mode...
echo All debug logs will be displayed in this console window.
echo.
echo Press Ctrl+C to stop the application.
echo.

REM Activate virtual environment if it exists
if exist "rawviewer_env\Scripts\activate.bat" (
    call rawviewer_env\Scripts\activate.bat
)

REM Run the application with Python to see console output
REM Use errorlevel to check if the application crashed
REM Force unbuffered output with -u flag
echo.
echo ========================================
echo Starting Python application...
echo ========================================
echo.
python -u src/main.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ========================================
if %EXIT_CODE% EQU 0 (
    echo Application exited normally (code: %EXIT_CODE%)
) else (
    echo Application exited with error code: %EXIT_CODE%
    echo.
    echo Check the log file in src\logs\ for detailed error information.
)
echo ========================================
echo.

REM Always pause to keep window open so user can see any error messages
pause



