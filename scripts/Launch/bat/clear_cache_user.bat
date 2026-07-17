@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM End-user cache wipe (copied next to RAWviewer.exe on Windows install).
REM Removes: %%USERPROFILE%%\.rawviewer_cache, logs, HKCU\Software\RAWviewer.
REM Does NOT remove: the app install, photos, or XMP sidecars.

echo.
echo ========================================
echo  RAWviewer - clear cache and session
echo ========================================
echo.
echo This removes local photo cache and app preferences so upgrades
echo can use the newer, faster search/index defaults.
echo Your photos and XMP sidecars are NOT deleted.
echo.

echo Closing RAWviewer if running...
taskkill /F /IM RAWviewer.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul

set "CLEARED=0"
set "FAILED=0"

call :RemoveTree "%USERPROFILE%\.rawviewer_cache" "image/EXIF/semantic/thumbnail cache"
call :RemoveTree "%LOCALAPPDATA%\RAWviewer\logs" "runtime logs"
call :RemoveTree "%APPDATA%\RAWviewer\logs" "roaming logs"
call :RemoveTree "%LOCALAPPDATA%\RAWviewer\cache" "install cache folder"
call :RemoveTree "%LOCALAPPDATA%\RAWviewer\CrashDumps" "crash dumps"

echo Clearing QSettings / session registry...
reg query "HKCU\Software\RAWviewer" >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    reg delete "HKCU\Software\RAWviewer" /f >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "CLEARED=1"
        echo   Removed HKCU\Software\RAWviewer
    ) else (
        echo   WARNING: Could not remove registry key HKCU\Software\RAWviewer
        set "FAILED=1"
    )
) else (
    echo   Registry already clean
)

echo.
if !FAILED! NEQ 0 (
    echo Finished with warnings — close RAWviewer fully and run again if needed.
    pause
    exit /b 1
)
if "!CLEARED!"=="0" (
    echo Nothing to clear — cache was already clean.
) else (
    echo Done. Reopen RAWviewer and your folder; the first open may rebuild cache.
)
pause
exit /b 0

:RemoveTree
set "TARGET=%~1"
set "LABEL=%~2"
if not exist "%TARGET%" exit /b 0
echo Removing %LABEL%:
echo   %TARGET%
rmdir /s /q "%TARGET%" >nul 2>&1
if exist "%TARGET%" (
    echo   WARNING: Could not fully remove — file may be locked.
    set "FAILED=1"
) else (
    set "CLEARED=1"
)
exit /b 0
