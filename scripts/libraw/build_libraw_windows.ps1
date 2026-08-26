# Build LibRaw from third_party/ and install raw_r.dll into the active rawpy.
#
# Why: stock rawpy 0.27 ships LibRaw 0.22.1, which cannot demosaic some newer
# Sony ARWs (ILCE-7M5 / ARW6). LibRaw master in third_party/ includes the
# decoder; we swap only the DLL so the Cython extension stays the PyPI wheel.
#
# Requires: VS 2017+ (nmake), cmake, zlib + libjpeg (vcpkg x64-windows-static
# is bootstrapped under .cache/libraw_vcpkg when CMAKE_PREFIX_PATH is unset).
#
# Usage (from repo root, after rawpy is installed into the target interpreter):
#   powershell -ExecutionPolicy Bypass -File scripts\libraw\build_libraw_windows.ps1
#   powershell ... -File scripts\libraw\build_libraw_windows.ps1 -PythonExe path\to\python.exe
#
# Env:
#   RAWVIEWER_LIBRAW_SKIP=1   — no-op
#   RAWVIEWER_LIBRAW_FORCE=1  — rebuild even if cache matches
#   CMAKE_PREFIX_PATH         — override vcpkg prefix

[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [switch]$SkipVcpkgBootstrap
)

$ErrorActionPreference = "Stop"

if ($env:RAWVIEWER_LIBRAW_SKIP -eq "1") {
    Write-Host "[libraw] RAWVIEWER_LIBRAW_SKIP=1 — leaving stock LibRaw in place."
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
    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    )
    $vs = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $vs) {
        throw "Visual Studio VsDevCmd.bat not found (need MSVC + nmake)."
    }
    Write-Host "[libraw] VS env: $vs"
    & "${env:COMSPEC}" /s /c "`"$vs`" -arch=x64 -no_logo && set" | ForEach-Object {
        $name, $value = $_ -split "=", 2
        if ($name) {
            try { Set-Item -Path "env:$name" -Value $value } catch {}
        }
    }
    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        throw "cl.exe not on PATH after VsDevCmd — install MSVC C++ tools."
    }
    if (-not (Get-Command nmake -ErrorAction SilentlyContinue)) {
        throw "nmake.exe not on PATH after VsDevCmd."
    }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        throw "cmake.exe not on PATH."
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
        throw "third_party/LibRaw missing sony_arw6.cpp — run: git submodule update --init --recursive"
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
        throw "rawpy not importable with $Py — pip install rawpy first."
    }
    $dest = Join-Path $rawpyDir.Trim() "raw_r.dll"
    $bak = "$dest.stock"
    if ((Test-Path $dest) -and -not (Test-Path $bak)) {
        Copy-Item -Force $dest $bak
        Write-Host "[libraw] Backed up stock DLL → $bak"
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
    Write-Host "[libraw] Installed raw_r.dll → $dest"
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
Write-Host "[libraw] OK — packaging LibRaw (ARW6-capable) is in place."
