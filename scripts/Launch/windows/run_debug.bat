@echo off
REM Verbose debug launcher. Optional first arg selects edition-like mode.
REM   run_debug.bat              -> full (Plus, GPU demosaic on)
REM   run_debug.bat lite         -> Standard
REM   run_debug.bat directml     -> Plus DirectML-like
REM   run_debug.bat cuda         -> Plus CUDA-like
REM   run_debug.bat menu         -> interactive picker
cd /d "%~dp0"

set "MODE=%~1"
if /I "%MODE%"=="menu" goto :menu
if "%MODE%"=="" (
    call "%~dp0launch_dev.bat" full
    exit /b %ERRORLEVEL%
)
if /I "%MODE%"=="lite" goto :dispatch
if /I "%MODE%"=="standard" goto :dispatch
if /I "%MODE%"=="full" goto :dispatch
if /I "%MODE%"=="directml" goto :dispatch
if /I "%MODE%"=="dml" goto :dispatch
if /I "%MODE%"=="cuda" goto :dispatch
if /I "%MODE%"=="plus" goto :dispatch

call "%~dp0launch_dev.bat" full %*
exit /b %ERRORLEVEL%

:menu
echo.
echo RAWviewer debug launch
echo ======================
echo   1^) Standard ^(lite^) — no AI search, CPU demosaic
echo   2^) Plus DirectML-like — AI search, CPU demosaic
echo   3^) Plus CUDA-like — AI search, CuPy GPU demosaic
echo   4^) Plus full ^(default debug^)
echo.
set /p CHOICE=Select [1-4, default 4]: 
if "%CHOICE%"=="" set CHOICE=4
if "%CHOICE%"=="1" set MODE=lite
if "%CHOICE%"=="2" set MODE=directml
if "%CHOICE%"=="3" set MODE=cuda
if "%CHOICE%"=="4" set MODE=full
if not defined MODE set MODE=full
echo.
call "%~dp0launch_dev.bat" %MODE%
exit /b %ERRORLEVEL%

:dispatch
call "%~dp0launch_dev.bat" %*
exit /b %ERRORLEVEL%
