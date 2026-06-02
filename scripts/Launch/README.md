# Launch scripts

Scripts for local development and packaging. All paths assume the **repository root** as the working directory (each script `cd`s there automatically).

## Windows (`.bat`)

| Script | Purpose |
|--------|---------|
| [`bat/run_debug.bat`](bat/run_debug.bat) | Run `src/main.py` with debug logging and `rawviewer_env` if present |
| [`bat/clear_cache.bat`](bat/clear_cache.bat) | Wipe memory/disk image caches, semantic index DB, logs, and QSettings (full fresh start) |
| [`bat/build_windows.bat`](bat/build_windows.bat) | Create `rawviewer_env`, install deps, run `build.py` |

From repo root:

```batch
scripts\Launch\bat\run_debug.bat
scripts\Launch\bat\clear_cache.bat
scripts\Launch\bat\build_windows.bat
clear_cache.bat
```

**Windows capture-time benchmark (POC):** `python scripts\compare_shell_capture_times.py "<folder>"` — see [`docs/POC_SHELL_CAPTURE_TIMES.md`](../../docs/POC_SHELL_CAPTURE_TIMES.md).

## macOS (shell)

Official macOS release only; there is no Linux build or installer.

| Script | Purpose |
|--------|---------|
| [`shell/launch_dev.sh`](shell/launch_dev.sh) | Run `src/main.py` with verbose dev env vars (macOS dev) |
| [`shell/build_macos.sh`](shell/build_macos.sh) | macOS build via `rawviewer_env` + `build.py` |

```bash
chmod +x scripts/Launch/shell/*.sh
./scripts/Launch/shell/launch_dev.sh
./scripts/Launch/shell/build_macos.sh
```

## Virtual environments

| Path | Used by |
|------|---------|
| `rawviewer_env/` | `run_debug.bat`, `build_windows.bat`, `build_macos.sh`, `build.py` |
| `.pixi/envs/default/` | `pixi install` / `pixi run start` (see root `pixi.toml`) |
| `.venv/` | Not referenced by these scripts; optional IDE/local use |

**Recommended for day-to-day dev:** `pixi run start` or `scripts/Launch/bat/run_debug.bat` (Windows) / `scripts/Launch/shell/launch_dev.sh` (macOS).

Root-level `run_debug.bat`, `build_windows.bat`, `clear_cache.bat`, `launch_dev.sh`, and `build_macos.sh` are thin wrappers that forward here.

**`clear_cache.bat`** closes RAWviewer (and dev `python … main.py` instances), then deletes `~\.rawviewer_cache`, log folders, and the `HKCU\Software\RAWviewer` registry key. It does **not** remove the installed app under `%LOCALAPPDATA%\RAWviewer` (exe, bundled models).

## Windows share helper (optional, dev)

Sources live under `src/windows_share_helper/`. Build the Release helper when testing WinRT share fallbacks (macOS uses the native share sheet; the shipping Windows bottom-bar button opens **Open with another app** instead):

```batch
cd src\windows_share_helper
dotnet build -c Release
```

Output: `src/windows_share_helper/bin/Release/net8.0-windows10.0.19041.0/WindowsShareHelper.exe` (ignored by git; rebuild locally as needed).
