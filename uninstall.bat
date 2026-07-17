@echo off
:: RAWviewer Uninstaller v1.4
:: Handles removal from the install folder or Windows Settings (UninstallString).
:: Removes: install dir, pixi env, AI models (MobileCLIP / denoise), Open With
:: associations, Desktop/Start Menu shortcuts, %%USERPROFILE%%\.rawviewer_cache,
:: %%LOCALAPPDATA%%\RAWviewer (logs, map tiles, runtime models), and
:: %%APPDATA%%\RAWviewer. Set RAWVIEWER_UNINSTALL_FULL=1 to also clear
:: HKCU\Software\RAWviewer QSettings (window layout, sort, last folder).

setlocal EnableExtensions EnableDelayedExpansion

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"

if /I not "%~1"=="__CLEANUP__" goto :STAGE1
goto :CLEANUP

:STAGE1
set "TEMP_SCRIPT=%TEMP%\uninstall_rawviewer_%RANDOM%.bat"
copy /y "%~f0" "%TEMP_SCRIPT%" >nul
if errorlevel 1 (
    echo Failed to prepare uninstaller.
    pause
    exit /b 1
)
start "" /min "%ComSpec%" /c call "%TEMP_SCRIPT%" __CLEANUP__ "%BASE_DIR%"
exit /b 0

:CLEANUP
set "TARGET_DIR=%~2"
if not defined TARGET_DIR set "TARGET_DIR=%BASE_DIR%"

timeout /t 2 /nobreak >nul

:: Stop app / setup / helper processes
taskkill /F /T /IM RAWviewer.exe >nul 2>&1
taskkill /F /T /IM RAWviewer_Setup.exe >nul 2>&1
taskkill /F /T /IM RAWviewer_Uninstaller.exe >nul 2>&1
taskkill /F /T /IM WindowsShareHelper.exe >nul 2>&1

:: Dev / pixi runs launched from the install folder
set "TARGET_ESC=%TARGET_DIR:\=\\%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$td = $env:TARGET_ESC;" ^
  "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe' OR Name = 'pixi.exe'\" |" ^
  " Where-Object { $_.CommandLine -match 'src[/\\]main\.py' -or $_.CommandLine -match 'src[/\\]bootstrap\.py' -or ($td -and $_.CommandLine -like ('*' + $td + '*')) } |" ^
  " ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

:: Apps & Features / Settings → Apps entry
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RAWviewer" /f >nul 2>&1

:: Open With / Default Programs registration (must track bootstrap._register_file_associations)
reg delete "HKCU\Software\Classes\Applications\RAWviewer.exe" /f >nul 2>&1
reg delete "HKCU\Software\Classes\RAWviewer.Image" /f >nul 2>&1
reg delete "HKCU\Software\RAWviewer\Capabilities" /f >nul 2>&1
reg delete "HKCU\Software\RegisteredApplications" /v RAWviewer /f >nul 2>&1

:: Per-extension OpenWithProgids / OpenWithList / accidental ProgId default
:: Keep in sync with src/raw_file_extensions.get_supported_extensions()
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$progId = 'RAWviewer.Image';" ^
  "$exts = @(" ^
  "  '.cr2','.cr3','.nef','.arw','.dng','.orf','.rw2','.pef','.srw','.x3f','.raf'," ^
  "  '.3fr','.fff','.iiq','.cap','.erf','.mef','.mos','.nrw','.rwl','.srf'," ^
  "  '.jpeg','.jpg','.png','.gif','.webp','.heif','.heic','.avif','.tif','.tiff'" ^
  ");" ^
  "foreach ($ext in $exts) {" ^
  "  $owp = \"HKCU:\\Software\\Classes\\$ext\\OpenWithProgids\";" ^
  "  if (Test-Path -LiteralPath $owp) { Remove-ItemProperty -LiteralPath $owp -Name $progId -ErrorAction SilentlyContinue };" ^
  "  $extKey = \"HKCU:\\Software\\Classes\\$ext\";" ^
  "  if (Test-Path -LiteralPath $extKey) {" ^
  "    $def = (Get-ItemProperty -LiteralPath $extKey -Name '(default)' -ErrorAction SilentlyContinue).'(default)';" ^
  "    if ($def -eq $progId) { Remove-ItemProperty -LiteralPath $extKey -Name '(default)' -ErrorAction SilentlyContinue }" ^
  "  };" ^
  "  $owl = \"HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\$ext\\OpenWithList\";" ^
  "  if (Test-Path -LiteralPath $owl) {" ^
  "    $props = Get-ItemProperty -LiteralPath $owl;" ^
  "    foreach ($name in @($props.PSObject.Properties.Name)) {" ^
  "      if ($name -in @('PSPath','PSParentPath','PSChildName','PSDrive','PSProvider')) { continue };" ^
  "      if ($props.$name -eq 'RAWviewer.exe') { Remove-ItemProperty -LiteralPath $owl -Name $name -ErrorAction SilentlyContinue }" ^
  "    }" ^
  "  };" ^
  "  $owp2 = \"HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\$ext\\OpenWithProgids\";" ^
  "  if (Test-Path -LiteralPath $owp2) { Remove-ItemProperty -LiteralPath $owp2 -Name $progId -ErrorAction SilentlyContinue }" ^
  "}" >nul 2>&1

:: Refresh shell icons / associations
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Add-Type -Namespace Win32 -Name Native -MemberDefinition '[DllImport(\"shell32.dll\")] public static extern void SHChangeNotify(int e, int f, IntPtr i, IntPtr d);';" ^
  "[Win32.Native]::SHChangeNotify(0x8000000, 0, [IntPtr]::Zero, [IntPtr]::Zero)" >nul 2>&1

:: Photo / EXIF / semantic / YuNet / CoreML caches
if exist "%USERPROFILE%\.rawviewer_cache" rd /s /q "%USERPROFILE%\.rawviewer_cache" >nul 2>&1

:: Optional Hugging Face hub leftovers from model downloads (best-effort)
if exist "%USERPROFILE%\.cache\huggingface\hub\models--plhery--mobileclip2-onnx" (
    rd /s /q "%USERPROFILE%\.cache\huggingface\hub\models--plhery--mobileclip2-onnx" >nul 2>&1
)
if exist "%USERPROFILE%\.cache\huggingface\hub\models--apple--coreml-mobileclip" (
    rd /s /q "%USERPROFILE%\.cache\huggingface\hub\models--apple--coreml-mobileclip" >nul 2>&1
)

if /I "%RAWVIEWER_UNINSTALL_FULL%"=="1" (
    reg delete "HKCU\Software\RAWviewer" /f >nul 2>&1
)

:: Shortcuts
set "DESKTOP=%USERPROFILE%\Desktop"
if exist "%DESKTOP%\RAWviewer.lnk" del /f /q "%DESKTOP%\RAWviewer.lnk" >nul 2>&1
if defined PUBLIC if exist "%PUBLIC%\Desktop\RAWviewer.lnk" del /f /q "%PUBLIC%\Desktop\RAWviewer.lnk" >nul 2>&1

set "START_MENU_PROG=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
if exist "%START_MENU_PROG%\RAWviewer.lnk" del /f /q "%START_MENU_PROG%\RAWviewer.lnk" >nul 2>&1

:: Install folder (pixi env, bundled models, src, uninstall.bat itself after copy-out)
set "RETRY=0"
:RETRY_LOOP
if exist "%TARGET_DIR%" (
    rd /s /q "%TARGET_DIR%" >nul 2>&1
    if exist "%TARGET_DIR%" (
        set /a RETRY+=1
        if !RETRY! LSS 10 (
            timeout /t 2 /nobreak >nul
            goto :RETRY_LOOP
        )
    )
)

:: Runtime data outside a custom install dir (and residual after install wipe):
:: logs, map_tiles, cache, CrashDumps, models\1xDeNoise_*.safetensors, etc.
if exist "%LOCALAPPDATA%\RAWviewer\logs" rd /s /q "%LOCALAPPDATA%\RAWviewer\logs" >nul 2>&1
if exist "%LOCALAPPDATA%\RAWviewer\map_tiles" rd /s /q "%LOCALAPPDATA%\RAWviewer\map_tiles" >nul 2>&1
if exist "%LOCALAPPDATA%\RAWviewer\cache" rd /s /q "%LOCALAPPDATA%\RAWviewer\cache" >nul 2>&1
if exist "%LOCALAPPDATA%\RAWviewer\CrashDumps" rd /s /q "%LOCALAPPDATA%\RAWviewer\CrashDumps" >nul 2>&1
if exist "%LOCALAPPDATA%\RAWviewer\models" rd /s /q "%LOCALAPPDATA%\RAWviewer\models" >nul 2>&1
if exist "%LOCALAPPDATA%\RAWviewer" (
    rd /s /q "%LOCALAPPDATA%\RAWviewer" >nul 2>&1
)
if exist "%APPDATA%\RAWviewer" (
    rd /s /q "%APPDATA%\RAWviewer" >nul 2>&1
)

set "UNINSTALL_MSG=RAWviewer has been removed."
if exist "%TARGET_DIR%" (
    set "UNINSTALL_MSG=RAWviewer was removed from Windows, but some files could not be deleted. Close RAWviewer if it is still running, then delete the install folder manually."
)
if /I "%RAWVIEWER_UNINSTALL_FULL%"=="1" (
    set "UNINSTALL_MSG=!UNINSTALL_MSG! Preferences (QSettings) were also removed."
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Add-Type -AssemblyName System.Windows.Forms;" ^
  "$msg = [Environment]::GetEnvironmentVariable('UNINSTALL_MSG');" ^
  "if (-not $msg) { $msg = 'RAWviewer has been removed.' };" ^
  "[System.Windows.Forms.MessageBox]::Show($msg, 'RAWviewer Uninstall', 'OK', 'Information') | Out-Null"

(goto) 2>nul & del "%~f0"
