@echo off
REM Smoke-test full vs lite profiles and report dist artifacts (Windows).
cd /d "%~dp0..\..\.."

echo RAWviewer build profile smoke test
echo ==================================
echo.

if exist "rawviewer_env\Scripts\activate.bat" (
    call rawviewer_env\Scripts\activate.bat
)

set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo [1/4] py_compile changed Python modules...
python -m py_compile build.py src\build_profile.py src\rawviewer_profile.py src\bootstrap.py
if %errorlevel% neq 0 (
    echo [FAIL] py_compile
    exit /b 1
)
echo [OK] py_compile

echo.
echo [2/4] Profile resolution (full / lite)...
python -c "import os,sys; sys.path.insert(0,'src'); os.environ['RAWVIEWER_BUILD_PROFILE']='full'; import importlib; import rawviewer_profile as rp; importlib.reload(rp); assert rp.resolved_profile()=='full'; os.environ['RAWVIEWER_BUILD_PROFILE']='lite'; importlib.reload(rp); assert rp.resolved_profile()=='lite'; print('[OK] rawviewer_profile.resolved_profile()')"
if %errorlevel% neq 0 exit /b 1

echo.
echo [3/4] Dist artifacts (optional — run build scripts first)...
set FOUND=0
for %%F in (
    RAWviewer_Setup.exe
    RAWviewer_Setup_CUDA.exe
    RAWviewer_Setup_DirectML.exe
    RAWviewer_Setup_Lite.exe
) do (
    if exist "dist\%%F" (
        echo   [found] dist\%%F
        set FOUND=1
    )
)
if "%FOUND%"=="0" echo   (none — build with scripts\Launch\windows\build_windows.bat)

echo.
echo [4/4] Dev launch commands:
echo   scripts\Launch\windows\launch_dev_lite.bat
echo   scripts\Launch\windows\launch_dev_full.bat
echo   scripts\Launch\windows\launch_dev_directml.bat
echo   scripts\Launch\windows\launch_dev_cuda.bat
echo   scripts\Launch\windows\run_debug.bat menu
echo.
echo Smoke test passed.
