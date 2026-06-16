# Launch scripts

Scripts for local development and packaging. All paths assume the **repository root** as the working directory (each script `cd`s there automatically).

**Version:** release **v2.3.2** (`build.py` `VERSION`, `QApplication` version, `pixi.toml` workspace version).

## Windows (`.bat`)

| Script | Purpose |
|--------|---------|
| [`bat/run_debug.bat`](bat/run_debug.bat) | Run `src/main.py` with debug logging and `rawviewer_env` if present (full profile) |
| [`bat/clear_cache.bat`](bat/clear_cache.bat) | Wipe memory/disk image caches, semantic index DB, logs, and QSettings (full fresh start) |
| [`bat/build_windows_full.bat`](bat/build_windows_full.bat) | Build **full** Windows installer (`RAWviewer_Setup_CUDA.exe` or `_DirectML.exe`) |
| [`bat/build_windows_lite.bat`](bat/build_windows_lite.bat) | Build **lite** installer (`RAWviewer_Setup_Lite.exe`) — no semantic/face AI; single path (no CUDA/DirectML split) |
| [`bat/build_windows.bat`](bat/build_windows.bat) | Same as full build with selectable backend (`cuda` default, or `directml`) |
| [`bat/build_windows_cuda.bat`](bat/build_windows_cuda.bat) | Full build with CUDA backend (`onnxruntime-gpu`) |
| [`bat/build_windows_directml.bat`](bat/build_windows_directml.bat) | Full build with DirectML backend (`onnxruntime-directml`) |
| [`bat/build_windows_all.bat`](bat/build_windows_all.bat) | Build **both** DirectML and CUDA full installers into `dist/` |
| [`bat/launch_dev_full.bat`](bat/launch_dev_full.bat) | Run from source with `RAWVIEWER_BUILD_PROFILE=full` |
| [`bat/launch_dev_lite.bat`](bat/launch_dev_lite.bat) | Run from source with `RAWVIEWER_BUILD_PROFILE=lite` (semantic/face off) |
| [`bat/test_builds.bat`](bat/test_builds.bat) | Smoke test profile resolution + optional `dist/` artifact check |

**Build outputs (Windows):**

| Profile | Installer artifact |
|---------|-------------------|
| Full (CUDA) | `dist/RAWviewer_Setup_CUDA.exe` |
| Full (DirectML) | `dist/RAWviewer_Setup_DirectML.exe` |
| Lite | `dist/RAWviewer_Setup_Lite.exe` |

From repo root:

```batch
scripts\Launch\bat\run_debug.bat
scripts\Launch\bat\clear_cache.bat
scripts\Launch\bat\build_windows_full.bat
scripts\Launch\bat\build_windows_full.bat directml
scripts\Launch\bat\build_windows_lite.bat
scripts\Launch\bat\build_windows_all.bat
scripts\Launch\bat\launch_dev_full.bat
scripts\Launch\bat\launch_dev_lite.bat
scripts\Launch\bat\test_builds.bat
clear_cache.bat
```

### Windows — bottom bar (v2.3)

| Platform behavior | Status in `main` (v2.3) |
|-------------------|-------------------------|
| **Open with another app** (Lightroom, Photoshop, …) | Implemented and visible from the bottom external-app button. Single file uses native Open With (`OpenAs_RunDLLW` / `SHOpenWithDialog` + `OAIF_EXEC`); multi-file selection can launch a chosen editor executable. |
| **System share** (Mail, Teams, …) | Legacy helper code remains (`_share_windows_ui_chain`) but the product path now focuses on opening originals in editing apps. |

Optional WinRT helper (dev):

```batch
cd src\windows_share_helper
dotnet build -c Release
```

Output: `src/windows_share_helper/bin/Release/net8.0-windows10.0.19041.0/WindowsShareHelper.exe` (git-ignored; rebuild locally).

## macOS (shell)

Official macOS release only; there is no Linux build or installer.

| Script | Purpose |
|--------|---------|
| [`shell/launch_dev.sh`](shell/launch_dev.sh) | Run `src/main.py` with verbose dev env (full profile defaults) |
| [`shell/launch_dev_full.sh`](shell/launch_dev_full.sh) | Same as full dev launch (`RAWVIEWER_BUILD_PROFILE=full`) |
| [`shell/launch_dev_lite.sh`](shell/launch_dev_lite.sh) | Run from source with lite profile (semantic/face off) |
| [`shell/clear_cache.sh`](shell/clear_cache.sh) | Wipe image/EXIF/semantic caches, logs, and QSettings (full fresh start) |
| [`shell/build_macos_full.sh`](shell/build_macos_full.sh) | macOS **full** build → `dist/RAWviewer.app` + release zip |
| [`shell/build_macos_lite.sh`](shell/build_macos_lite.sh) | macOS **lite** build → `dist/RAWviewer_Lite.app` + release zip |
| [`shell/build_macos.sh`](shell/build_macos.sh) | Underlying build script; accepts `full` or `lite` as first argument |
| [`shell/test_builds.sh`](shell/test_builds.sh) | Smoke test profile resolution + optional `dist/` artifact check |

**Build outputs (macOS):**

| Profile | App bundle | Release zip |
|---------|------------|-------------|
| Full | `dist/RAWviewer.app` | `dist/RAWviewer-v{VERSION}-macOS.zip` |
| Lite | `dist/RAWviewer_Lite.app` | `dist/RAWviewer-v{VERSION}-macOS-Lite.zip` |

```bash
chmod +x scripts/Launch/shell/*.sh
./scripts/Launch/shell/launch_dev_full.sh
./scripts/Launch/shell/launch_dev_lite.sh
./scripts/Launch/shell/clear_cache.sh
./scripts/Launch/shell/build_macos_full.sh
./scripts/Launch/shell/build_macos_lite.sh
./scripts/Launch/shell/test_builds.sh
./clear_cache.sh
```

**Recommended for day-to-day dev:** `pixi run start` or `./scripts/Launch/shell/launch_dev_full.sh` (macOS) / `scripts/Launch/bat/launch_dev_full.bat` (Windows).

### macOS — build process (`build_macos.sh [full|lite]`)

1. **Requires macOS** (`darwin`), `python3` on PATH.
2. Creates or reuses **`rawviewer_env/`** at repo root.
3. Optional **Homebrew** deps for `pyexiv2`: `inih`, `gettext` (`brew install inih gettext` if the wheel build fails).
4. Installs PyQt6, rawpy, PyInstaller, **scipy**, **pyobjc** (Cocoa / CoreML / Quartz / Vision), and other runtime deps; **pyexiv2** is **required** (`brew install inih gettext` if the wheel build fails).
5. Uninstalls heavy unused ML stacks (`torch`, `sentence-transformers`, …) to keep the app bundle smaller.
6. Cleans `build/`, `dist/`, `*.spec`, then runs **`python build.py --profile full|lite`** (version **2.3.2**, updates `Info.plist`; MobileCLIP models are **not** bundled — users download in-app on full builds).
7. Packages release zip with the app bundle, **`install_macos_app.sh`**, **`remove_macos_quarantine.sh`**, and **`Start Here.txt`**.

**End-user install:** extract the zip, then in Terminal:

```bash
cd /path/to/RAWviewer-v2.3.2-macOS
bash install_macos_app.sh
```

(Tip: type `cd ` and drag the folder onto Terminal.)

The script clears macOS download quarantine, copies RAWviewer to Applications, and opens it. Double-clicking the unsigned app from a download is often blocked before quarantine is cleared — **Terminal + `bash install_macos_app.sh`** is the supported path.

**Output:** `dist/RAWviewer.app` or `dist/RAWviewer_Lite.app`, plus matching release zip under `dist/`.

**Pixi alternative:** `pixi install && pixi run python build.py --profile full` (or `--profile lite`).

### macOS — dev run & preflight (`launch_dev.sh`)

`launch_dev.sh` / `launch_dev_*.sh` prefer `.pixi/envs/default` when `pixi` and `pixi.toml` are present (CoreML for full semantic preflight); otherwise they activate `rawviewer_env/`.

Before `src/main.py`, the script can run (skippable) checks:

| Check | Env | Default |
|-------|-----|---------|
| `pyexiv2` import | `RAWVIEWER_TEST_PYEXIV2` | `1` |
| MobileCLIP / semantic backend | `RAWVIEWER_TEST_SEMANTIC` | `1` (needs `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=1`) |

Skip checks when iterating on unrelated features:

```bash
RAWVIEWER_TEST_PYEXIV2=0 RAWVIEWER_TEST_SEMANTIC=0 ./scripts/Launch/shell/launch_dev.sh
```

**Default dev env (v2.3):**

| Variable | Default | Notes |
|----------|---------|--------|
| `RAWVIEWER_GPU_VIEW` | `1` (release default) | OpenGL single-image viewport; set `0` for legacy scroll view |
| `RAWVIEWER_SHARE_MENU` | `1` | Qt menu of `NSSharingService` targets (reliable under Qt6) |
| `RAWVIEWER_ENABLE_SEMANTIC_SEARCH` | `1` | Semantic search on |
| `RAWVIEWER_DEBUG` | `1` | Verbose logging |

Share opt-in (see [`docs/macos-sharing-v21-v22.md`](../../docs/macos-sharing-v21-v22.md)):

| Variable | Purpose |
|----------|---------|
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | Try `NSSharingServicePicker` first (~900ms menu fallback) |
| `RAWVIEWER_SHARE_SHOW_AIRDROP=1` | Show AirDrop in the share menu |
| `RAWVIEWER_SHARE_DEBUG=1` | Share lines in status bar + `[SHARE]` logs |
| `RAWVIEWER_FILE_LOG=1` | Persistent file logging (share diagnostics) |

Pass env vars as arguments: `./scripts/Launch/shell/launch_dev.sh RAWVIEWER_GPU_VIEW=0`

### macOS — release smoke test (manual)

After `build_macos.sh` or `pixi run python build.py`:

1. **Gatekeeper:** `xattr -cr dist/RAWviewer.app` then `open dist/RAWviewer.app`.
2. **About / version:** Help or logs should report app version **2.3.2**.
3. **Single-image view:** Open a JPEG/RAW folder → one file → bottom **share** icon visible.
4. **Share:** Click share → Qt menu lists Mail / Messages / etc.; pick Mail and confirm attachment path (not an empty spinner).
5. **Semantic (if models bundled):** Search field accepts a text query; index progress in status area.
6. **GPU view:** Pan/zoom; toggle `RAWVIEWER_GPU_VIEW=0` if comparing share behavior.
7. **Frozen vs dev:** Re-test share on the `.app` build; sandbox entitlements differ from `python src/main.py` (see sharing doc).

Logs: dev console `[SHARE]`; packaged app under `~/Library/Logs/` or paths noted in app logging when `RAWVIEWER_FILE_LOG=1`.

## Virtual environments

| Path | Used by |
|------|---------|
| `rawviewer_env/` | `run_debug.bat`, `build_windows.bat`, `build_macos.sh`, `build.py` |
| `.pixi/envs/default/` | `pixi install` / `pixi run start` (see root `pixi.toml`) |
| `.venv/` | Not referenced by these scripts; optional IDE/local use |

**`clear_cache.bat`** / **`clear_cache.sh`** close RAWviewer (and dev `python … main.py` instances), then delete `~/.rawviewer_cache`, log folders, and session state (Windows: `HKCU\Software\RAWviewer`; macOS: `~/Library/Preferences/com.RAWviewer.RAWviewer.plist`). They do **not** remove the installed app (Windows: `%LOCALAPPDATA%\RAWviewer`; macOS: `/Applications/RAWviewer.app` or repo `models/`).

## Known issues (platform)

- **Windows Open with:** The bottom external-app button is shown on `win32`. Single-file Open With uses the native picker; multi-file Open With uses a chosen editor executable because the Windows native picker is file-oriented.
- **macOS native share popover:** Often spins empty under the Qt6 host; **product default is the Qt share menu**, not the popover.
