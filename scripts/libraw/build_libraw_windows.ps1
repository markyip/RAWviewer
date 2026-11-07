# Build LibRaw from third_party/ and install raw_r.dll into the active rawpy.
#
# Why: stock rawpy 0.27 ships LibRaw 0.22.1, which cannot demosaic some newer
# Sony ARWs (ILCE-7M5 / ARW6). LibRaw master in third_party/ includes the
# decoder; we swap only the DLL so the Cython extension stays the PyPI wheel.
#
# Requires: Visual Studio 2017+ with the C++ toolset (MSVC + nmake) and cmake.
# VS is discovered via vswhere (any edition/version, including VS 2022 and VS
# 2026); set RAWVIEWER_VSDEVCMD to override a non-standard install.
# zlib + libjpeg (vcpkg x64-windows-static) is bootstrapped under
# .cache/libraw_vcpkg when CMAKE_PREFIX_PATH is unset.
#
# Usage (from repo root, after rawpy is installed into the target interpreter):
#   powershell -ExecutionPolicy Bypass -File scripts\libraw\build_libraw_windows.ps1
#   powershell ... -File scripts\libraw\build_libraw_windows.ps1 -PythonExe path\to\python.exe
#
# Env:
#   RAWVIEWER_LIBRAW_SKIP=1   - no-op
#   RAWVIEWER_LIBRAW_FORCE=1  - rebuild even if cache matches
#   RAWVIEWER_VSDEVCMD        - override path to VsDevCmd.bat
#   CMAKE_PREFIX_PATH         - override vcpkg prefix

[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$SkipVcpkgBootstrap
)

$ErrorActionPreference = "Stop"

if ($env:RAWVIEWER_LIBRAW_SKIP -eq "1") {
    Write-Host "[libraw] RAWVIEWER_LIBRAW_SKIP=1 - leaving stock LibRaw in place."
    exit 0
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$LibRaw = Join-Path $RepoRoot "third_party\LibRaw"
$LibRawCmake = Join-Path $RepoRoot "third_party\LibRaw-cmake"
$CacheDir = Join-Path $RepoRoot ".cache\libraw_master\windows"
$CachedDll = Join-Path $CacheDir "raw_r.dll"
$VcpkgDir = Join-Path $RepoRoot ".cache\libraw_vcpkg"
$PinFile = Join-Path $PSScriptRoot "PIN.txt"

function Get-PinnedLibRaw {
    if (-not (Test-Path $PinFile)) { return "" }
    $line = Get-Content $PinFile | Where-Object { $_ -match '^LibRaw=' } | Select-Object -First 1
    if ($line -match '^LibRaw=([0-9a-fA-F]+)') { return $Matches[1] }
    return ""
}

function Initialize-VS {
    $vsDevCmd = $null
    $vsInstall = $null

    if ($env:RAWVIEWER_VSDEVCMD) {
        if (-not (Test-Path $env:RAWVIEWER_VSDEVCMD)) {
            throw "RAWVIEWER_VSDEVCMD does not exist: $($env:RAWVIEWER_VSDEVCMD)"
        }
        $vsDevCmd = (Resolve-Path $env:RAWVIEWER_VSDEVCMD).Path
    }

    if (-not $vsDevCmd) {
        $installs = @()
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path $vswhere)) {
            $vswhereCmd = Get-Command vswhere -ErrorAction SilentlyContinue
            if ($vswhereCmd) { $vswhere = $vswhereCmd.Source }
        }
        if ($vswhere -and (Test-Path $vswhere)) {
            # Prefer installs that actually include the MSVC C++ toolset.
            $installs = @(
                & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null |
                    Where-Object { $_ }
            )
            if ($installs.Count -eq 0) {
                # Still inspect installs without C++ so the error below is actionable.
                $installs = @(
                    & $vswhere -products * -property installationPath 2>$null |
                        Where-Object { $_ }
                )
            }
        }

        foreach ($install in $installs) {
            foreach ($bat in @("VsDevCmd.bat", "LaunchDevCmd.bat")) {
                $candidate = Join-Path $install "Common7\Tools\$bat"
                if (Test-Path $candidate) {
                    $vsInstall = $install
                    $vsDevCmd = $candidate
                    break
                }
            }
            if ($vsDevCmd) { break }
        }
    }

    if (-not $vsDevCmd) {
        # Last resort for machines without vswhere / non-standard layouts.
        $roots = @(
            (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\2022"),
            (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022"),
            (Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\18"),
            (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\18")
        )
        foreach ($root in $roots) {
            foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
                $candidate = Join-Path $root "$edition\Common7\Tools\VsDevCmd.bat"
                if (Test-Path $candidate) {
                    $vsDevCmd = $candidate
                    $vsInstall = Join-Path $root $edition
                    break
                }
            }
            if ($vsDevCmd) { break }
        }
    }

    if (-not $vsDevCmd) {
        throw ("Visual Studio with C++ tools not found (need MSVC + nmake).`n" +
            "  Install the 'Desktop development with C++' workload in Visual Studio 2017+`n" +
            "  (VS 2022 and 2026 are both supported), or set RAWVIEWER_VSDEVCMD to VsDevCmd.bat.")
    }

    Write-Host "[libraw] VS env: $vsDevCmd"
    & "${env:COMSPEC}" /s /c "`"$vsDevCmd`" -arch=x64 -no_logo && set" | ForEach-Object {
        $name, $value = $_ -split "=", 2
        if ($name) {
            try { Set-Item -Path "env:$name" -Value $value } catch {}
        }
    }

    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        $where = if ($vsInstall) { " ($vsInstall)" } else { "" }
        throw ("cl.exe not on PATH after VsDevCmd$where - install the C++ toolset.`n" +
            "  Visual Studio Installer -> Modify -> 'Desktop development with C++'.")
    }
    if (-not (Get-Command nmake -ErrorAction SilentlyContinue)) {
        throw "nmake.exe not on PATH after VsDevCmd."
    }

    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        $cmakeDirs = @()
        if ($vsInstall) {
            $cmakeDirs += Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
        }
        $cmakeDirs += @(
            (Join-Path ${env:ProgramFiles} "CMake\bin"),
            (Join-Path ${env:ProgramFiles(x86)} "CMake\bin"),
            (Join-Path $env:LOCALAPPDATA "Programs\CMake\bin")
        )
        foreach ($dir in $cmakeDirs) {
            $exe = Join-Path $dir "cmake.exe"
            if (Test-Path $exe) {
                $env:PATH = "$dir;$env:PATH"
                Write-Host "[libraw] cmake: $exe"
                break
            }
        }
    }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        throw ("cmake.exe not on PATH.`n" +
            "  Install CMake (winget install Kitware.CMake) or add the 'C++ CMake tools for Windows'`n" +
            "  component in the Visual Studio Installer, then reopen the shell.")
    }
}

function Resolve-Python {
    if ($PythonExe -and (Test-Path $PythonExe)) { return (Resolve-Path $PythonExe).Path }
    $venvPy = Join-Path $RepoRoot "rawviewer_env\Scripts\python.exe"
    if (Test-Path $venvPy) { return (Resolve-Path $venvPy).Path }
    $pixiPy = Join-Path $RepoRoot ".pixi\envs\default\python.exe"
    if (Test-Path $pixiPy) { return (Resolve-Path $pixiPy).Path }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "No Python interpreter found (pass -PythonExe or create rawviewer_env / pixi env)."
}

function Ensure-Submodules {
    if (-not (Test-Path (Join-Path $LibRaw "src\decoders\sony_arw6.cpp"))) {
        Write-Host "[libraw] Initializing third_party submodules..."
        Push-Location $RepoRoot
        try {
            git submodule update --init --recursive third_party/LibRaw third_party/LibRaw-cmake
        } finally {
            Pop-Location
        }
    }
    if (-not (Test-Path (Join-Path $LibRaw "src\decoders\sony_arw6.cpp"))) {
        throw "third_party/LibRaw missing sony_arw6.cpp - run: git submodule update --init --recursive"
    }
    if (-not (Test-Path (Join-Path $LibRawCmake "CMakeLists.txt"))) {
        throw "third_party/LibRaw-cmake missing CMakeLists.txt"
    }
}

function Ensure-VcpkgPrefix {
    if ($env:CMAKE_PREFIX_PATH -and (Test-Path $env:CMAKE_PREFIX_PATH)) {
        Write-Host "[libraw] CMAKE_PREFIX_PATH=$($env:CMAKE_PREFIX_PATH)"
        return
    }
    $prefix = Join-Path $VcpkgDir "installed\x64-windows-static"
    if (-not (Test-Path (Join-Path $prefix "lib"))) {
        if ($SkipVcpkgBootstrap) {
            throw "CMAKE_PREFIX_PATH unset and $prefix missing (SkipVcpkgBootstrap)."
        }
        Write-Host "[libraw] Bootstrapping vcpkg zlib + libjpeg-turbo (x64-windows-static)..."
        if (-not (Test-Path (Join-Path $VcpkgDir "vcpkg.exe"))) {
            if (Test-Path $VcpkgDir) {
                Remove-Item -Recurse -Force $VcpkgDir
            }
            git clone https://github.com/microsoft/vcpkg --depth 1 $VcpkgDir
            & (Join-Path $VcpkgDir "bootstrap-vcpkg.bat") -disableMetrics
            if ($LASTEXITCODE -ne 0) { throw "vcpkg bootstrap failed" }
        }
        & (Join-Path $VcpkgDir "vcpkg.exe") install zlib "libjpeg-turbo[jpeg8]" --triplet=x64-windows-static --recurse
        if ($LASTEXITCODE -ne 0) { throw "vcpkg install failed" }
    }
    $env:CMAKE_PREFIX_PATH = $prefix
    Write-Host "[libraw] CMAKE_PREFIX_PATH=$($env:CMAKE_PREFIX_PATH)"
}

function Build-LibRawDll {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    # Keep the cmake tree under .cache so the LibRaw-cmake submodule stays clean.
    $buildDir = Join-Path $CacheDir "cmake-build"
    if (Test-Path $buildDir) {
        Remove-Item -Recurse -Force $buildDir
    }
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

    $librawPosix = ($LibRaw -replace "\\", "/")
    $cmakeSrc = ($LibRawCmake -replace "\\", "/")
    $cfg = @(
        "cmake", $cmakeSrc
        "-G", "NMake Makefiles"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_PREFIX_PATH=$($env:CMAKE_PREFIX_PATH)"
        "-DLIBRAW_PATH=$librawPosix"
        "-DENABLE_X3FTOOLS=ON"
        "-DENABLE_6BY9RPI=ON"
        "-DENABLE_EXAMPLES=OFF"
        "-DENABLE_OPENMP=ON"
        "-DENABLE_RAWSPEED=OFF"
        "-DENABLE_LCMS=OFF"
        "-DENABLE_JASPER=OFF"
        "-DCMAKE_INSTALL_PREFIX=install"
    )
    Push-Location $buildDir
    try {
        Write-Host "[libraw] cmake configure..."
        & $cfg[0] $cfg[1..($cfg.Length - 1)]
        if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
        Write-Host "[libraw] cmake --build --target install..."
        cmake --build . --target install
        if ($LASTEXITCODE -ne 0) { throw "cmake build/install failed" }
    } finally {
        Pop-Location
    }

    $built = Join-Path $buildDir "install\bin\raw_r.dll"
    if (-not (Test-Path $built)) {
        throw "raw_r.dll not produced at $built"
    }
    Copy-Item -Force $built $CachedDll
    $pin = Get-PinnedLibRaw
    if ($pin) {
        Set-Content -Path (Join-Path $CacheDir "SOURCE_COMMIT.txt") -Value $pin
    }
    Write-Host "[libraw] Cached $CachedDll"
}

function Install-IntoRawpy([string]$Py) {
    $rawpyDir = & $Py -c "import rawpy, os; print(os.path.dirname(rawpy.__file__))"
    if (-not $rawpyDir -or -not (Test-Path $rawpyDir)) {
        throw "rawpy not importable with $Py - pip install rawpy first."
    }
    $dest = Join-Path $rawpyDir.Trim() "raw_r.dll"
    $bak = "$dest.stock"
    if ((Test-Path $dest) -and -not (Test-Path $bak)) {
        Copy-Item -Force $dest $bak
        Write-Host "[libraw] Backed up stock DLL -> $bak"
    }
    Copy-Item -Force $CachedDll $dest
    # OpenMP runtime often needed beside the DLL (same as rawpy wheels).
    $vcomp = Join-Path $rawpyDir.Trim() "vcomp140.dll"
    if (-not (Test-Path $vcomp) -and $env:VCToolsRedistDir) {
        $hits = Get-ChildItem -Path $env:VCToolsRedistDir -Recurse -Filter "vcomp140.dll" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\" } | Select-Object -First 1
        if ($hits) {
            Copy-Item -Force $hits.FullName $vcomp
            Write-Host "[libraw] Bundled $($hits.Name)"
        }
    }
    Write-Host "[libraw] Installed raw_r.dll -> $dest"
}

function Test-Rawpy([string]$Py) {
    & $Py -c @"
import rawpy, os
print('rawpy', rawpy.__version__, 'libraw', rawpy.libraw_version)
print('dll', os.path.join(os.path.dirname(rawpy.__file__), 'raw_r.dll'))
"@
    if ($LASTEXITCODE -ne 0) { throw "rawpy import failed after DLL swap" }
}

# --- main ---
Ensure-Submodules
$py = Resolve-Python
Write-Host "[libraw] Python: $py"

$pin = Get-PinnedLibRaw
$cacheCommit = ""
if (Test-Path (Join-Path $CacheDir "SOURCE_COMMIT.txt")) {
    $cacheCommit = (Get-Content (Join-Path $CacheDir "SOURCE_COMMIT.txt") -Raw).Trim()
}
$needBuild = ($env:RAWVIEWER_LIBRAW_FORCE -eq "1") -or
    (-not (Test-Path $CachedDll)) -or
    ($pin -and $cacheCommit -and ($pin -ne $cacheCommit))

if ($needBuild) {
    Initialize-VS
    Ensure-VcpkgPrefix
    Build-LibRawDll
} else {
    Write-Host "[libraw] Reusing cached DLL: $CachedDll"
}

Install-IntoRawpy $py
Test-Rawpy $py
Write-Host "[libraw] OK - packaging LibRaw (ARW6-capable) is in place."
