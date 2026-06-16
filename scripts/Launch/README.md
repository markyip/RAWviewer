# Launch scripts

Scripts for local development and packaging. All paths assume the **repository root** as the working directory (each script `cd`s there automatically).

**Version:** release **v2.3.2** (`build.py` `VERSION`, `QApplication` version, `pixi.toml` workspace version).

## Windows (`.bat`)

| Script | Purpose |
|--------|---------|
| [`bat/run_debug.bat`](bat/run_debug.bat) | Run `src/main.py` with debug logging and `rawviewer_env` if present |
| [`bat/clear_cache.bat`](bat/clear_cache.bat) | Wipe memory/disk image caches, semantic index DB, logs, and QSettings (full fresh start) |
| [`bat/build_windows.bat`](bat/build_windows.bat) | Build Windows app with selectable backend (`cuda` default, or `directml`) |
| [`bat/build_windows_cuda.bat`](bat/build_windows_cuda.bat) | Build Windows app with CUDA backend (`onnxruntime-gpu`) |
| [`bat/build_windows_directml.bat`](bat/build_windows_directml.bat) | Build Windows app with DirectML backend (`onnxruntime-directml`) |
| [`bat/build_windows_all.bat`](bat/build_windows_all.bat) | Build **both** DirectML and CUDA installers into `dist/` |

From repo root:

```batch
scripts\Launch\bat\run_debug.bat
scripts\Launch\bat\clear_cache.bat
scripts\Launch\bat\build_windows.bat
scripts\Launch\bat\build_windows.bat cuda
scripts\Launch\bat\build_windows.bat directml
scripts\Launch\bat\build_windows_cuda.bat
scripts\Launch\bat\build_windows_directml.bat
scripts\Launch\bat\build_windows_all.bat
clear_cache.bat
```

### Windows — bottom bar (v2.3)

| Platform behavior | Status in `main` (v2.3) |
|-------------------|-------------------------|
| **Open with another app** (Lightroom, Photoshop, …) | Implemented (`OpenAs_RunDLLW` / `SHOpenWithDialog` + `OAIF_EXEC`). **UI:** bottom share button is currently **hidden on Windows**; wiring exists via `_dispatch_share_bottom` but is not connected to a visible control — see [Known issues](#known-issues-platform). |
| **System share** (Mail, Teams, …) | `_share_windows_ui_chain` (helper → WinRT → shell verb → clipboard). Dev-only unless the bottom button is re-enabled. |

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
| [`shell/launch_dev.sh`](shell/launch_dev.sh) | Run `src/main.py` with verbose dev env (GPU view, share menu, semantic preflight) |
| [`shell/clear_cache.sh`](shell/clear_cache.sh) | Wipe image/EXIF/semantic caches, logs, and QSettings (full fresh start) |
| [`shell/build_macos.sh`](shell/build_macos.sh) | macOS build via `rawviewer_env` + `build.py` → `dist/RAWviewer.app` |
| [`shell/build_macos_lite.sh`](shell/build_macos_lite.sh) | **Lite** macOS build: no semantic/face AI; EXIF+GPS search; higher prefetch → `dist/RAWviewer-v*-macOS-lite.zip` |
| [`shell/launch_dev_lite.sh`](shell/launch_dev_lite.sh) | Run from source in **lite** profile (metadata/GPS search only; skips MobileCLIP preflight) |

```bash
chmod +x scripts/Launch/shell/*.sh
./scripts/Launch/shell/launch_dev.sh
./scripts/Launch/shell/launch_dev_lite.sh
./scripts/Launch/shell/clear_cache.sh
./scripts/Launch/shell/build_macos.sh
./scripts/Launch/shell/build_macos_lite.sh
./clear_cache.sh
```

**Recommended for day-to-day dev:** `pixi run start` or `./scripts/Launch/shell/launch_dev.sh` (macOS) / `scripts/Launch/bat/run_debug.bat` (Windows).

### macOS — build process (`build_macos.sh`)

1. **Requires macOS** (`darwin`), `python3` on PATH.
2. Creates or reuses **`rawviewer_env/`** at repo root.
3. Optional **Homebrew** deps for `pyexiv2`: `inih`, `gettext` (`brew install inih gettext` if the wheel build fails).
4. Installs PyQt6, rawpy, PyInstaller, **scipy**, **pyobjc** (Cocoa / CoreML / Quartz / Vision), and other runtime deps; **pyexiv2** is **required** (`brew install inih gettext` if the wheel build fails).
5. Uninstalls heavy unused ML stacks (`torch`, `sentence-transformers`, …) to keep the app bundle smaller.
6. Cleans `build/`, `dist/`, `*.spec`, then runs **`python build.py`** (version **2.3.2**, updates `Info.plist`; MobileCLIP models are **not** bundled — users download in-app).
7. Packages **`dist/RAWviewer-v2.3.2-macOS.zip`** with `RAWviewer.app`, **`install_macos_app.sh`**, **`remove_macos_quarantine.sh`**, and **`Start Here.txt`**.

### macOS — lite profile (`build_macos_lite.sh` / `launch_dev_lite.sh`)

Build-time profile (`build.py --profile lite`) disables **MobileCLIP semantic search** and **face detection**, keeps **GPS city/country** (`reverse-geocoder` + **scipy**), and raises prefetch defaults (nav radius, idle display prefetch, filmstrip radius, preload threads). Same codebase as full; behavior is controlled by baked `src/build_profile.py` + runtime hook.

```bash
./scripts/Launch/shell/launch_dev_lite.sh          # debug from source
./scripts/Launch/shell/build_macos_lite.sh         # dist/RAWviewer-v2.3.2-macOS-lite.zip (~73 MB vs ~82 MB full)
python build.py --profile lite                     # equivalent PyInstaller step
```

Override any lite prefetch default via env (see `src/rawviewer_profile.py`).

**RAM-adaptive navigation prefetch (lite and full):** when `RAWVIEWER_NAV_PRELOAD_ADAPTIVE=1` (default) and `RAWVIEWER_NAV_PRELOAD_RADIUS` is unset, single-view neighbor prefetch radius scales with **available RAM** (~5–16 lite, ~4–10 full). Caps: `RAWVIEWER_NAV_PRELOAD_RADIUS_MIN` / `_MAX`. Gallery idle thumbnails use a larger batch in lite and `PRELOAD_NEXT` priority when semantic/face indexing is off.

**End-user install:** extract the zip, then in Terminal:

```bash
cd /path/to/RAWviewer-v2.3.2-macOS
bash install_macos_app.sh
```

(Tip: type `cd ` and drag the folder onto Terminal.)

The script clears macOS download quarantine, copies RAWviewer to Applications, and opens it. Double-clicking the unsigned app from a download is often blocked before quarantine is cleared — **Terminal + `bash install_macos_app.sh`** is the supported path.

**Output:** `dist/RAWviewer.app` and `dist/RAWviewer-v2.3.2-macOS.zip`.

**Pixi alternative:** `pixi install && pixi run python build.py` (then test with `bash install_macos_app.sh` from a folder containing the app).

### macOS — dev run & preflight (`launch_dev.sh`)

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

- **Windows Open with:** Native picker code is in `main.py`, but the bottom-bar control is **not shown** on `win32` in current `main` (regression after macOS share work). Re-enable requires showing the button and `clicked` → `_on_share_bottom_button_clicked` (see git `8b3f54a`). Until then, use Explorer **Open with** on the file.
- **macOS native share popover:** Often spins empty under the Qt6 host; **product default is the Qt share menu**, not the popover.
