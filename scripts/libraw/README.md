# LibRaw packaging (ARW6 / new Sony bodies)

Stock **rawpy 0.27** wheels ship **LibRaw 0.22.1**. That build cannot open some
newer Sony ARWs (e.g. **ILCE-7M5 / A7 V** CRAW HQ — `LibRawFileUnsupportedError`).

LibRaw **master** (post–PR [#824](https://github.com/LibRaw/LibRaw/pull/824)
ARW6 decoder) can. Until a tagged LibRaw + rawpy wheel includes that decoder,
RAWviewer packages **swap** a self-built `raw_r` / `libraw_r` into the installed
rawpy package (same pattern as the macOS OpenMP swap).

## Sources in this repo

| Path | Role |
|------|------|
| [`third_party/LibRaw`](../../third_party/LibRaw) | LibRaw sources (git submodule) |
| [`third_party/LibRaw-cmake`](../../third_party/LibRaw-cmake) | CMake wrapper for Windows builds |

Pinned commits are recorded in `.gitmodules` / the submodule gitlinks. See
[`PIN.txt`](PIN.txt) for the human-readable pin and why.

```bash
git submodule update --init --recursive third_party/LibRaw third_party/LibRaw-cmake
```

## Scripts

| Script | Platform | What it does |
|--------|----------|----------------|
| [`build_libraw_windows.ps1`](build_libraw_windows.ps1) | Windows | CMake-build LibRaw → replace `rawpy/raw_r.dll` |
| [`build_libraw_openmp.sh`](build_libraw_openmp.sh) | macOS arm64 | Build OpenMP LibRaw **from the submodule** → replace `libraw_r.25.dylib` |
| [`check_libraw_parallelism.py`](check_libraw_parallelism.py) | macOS | Smoke-check OpenMP |

Official packaging hooks:

- Windows: `scripts/Launch/windows/build_windows.bat` calls the PowerShell script
  after the venv has `rawpy` installed.
- macOS: `scripts/Launch/macos/build_macos.sh` → `ensure_openmp_libraw` (unchanged
  entry point; the shell script now compiles submodule sources).

## Skip / force

| Env | Effect |
|-----|--------|
| `RAWVIEWER_LIBRAW_SKIP=1` | Do not rebuild/swap (use stock rawpy LibRaw) |
| `RAWVIEWER_LIBRAW_FORCE=1` | Rebuild even when a cached DLL/dylib is present |
| `RAWVIEWER_LIBRAW_OPENMP_STANDALONE=1` | macOS: ship `libomp` beside rawpy (release builds) |
| `RAWVIEWER_LIBRAW_OPENMP_SKIP_COMPILE=1` | macOS: reuse `.cache/libraw_openmp` dylib |
| `CMAKE_PREFIX_PATH` | Windows: zlib/jpeg (vcpkg `x64-windows-static` recommended) |

## Verify

```bash
python -c "import rawpy; print(rawpy.libraw_version)"
# Optional: point at an A7 V ARW
python -c "import rawpy; rawpy.imread(r'path\to\file.ARW').close(); print('OK')"
```

When LibRaw still cannot demosaic a file, Adjust falls back to the embedded
JPEG as an 8-bit edit base (same path as Nikon HE/HE*). Persisted demosaic
thumb/grid/preview caches invalidate automatically when the native LibRaw
binary beside rawpy changes (`libraw_runtime_stamp` in `image_cache.py`).
