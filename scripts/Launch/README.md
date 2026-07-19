# Launch scripts

Local development, packaging, and (macOS) release-zip helpers.  
Working directory for every script is the **repository root** (each script `cd`s there).

**Layout**

```
scripts/Launch/
  README.md
  windows/          Windows .bat (dev + build + cache)
  macos/            macOS .sh (dev + build)
  macos/release/    Assets copied into the macOS release zip
```

---

## Windows (`windows/`)

| Script | Purpose |
|--------|---------|
| [`windows/launch_dev.bat`](windows/launch_dev.bat) | Shared launcher: `lite` \| `full` \| `directml` \| `cuda` |
| [`windows/launch_dev_lite.bat`](windows/launch_dev_lite.bat) | Standard — no semantic/face; CPU demosaic |
| [`windows/launch_dev_full.bat`](windows/launch_dev_full.bat) | Plus — semantic on; prefer GPU demosaic |
| [`windows/launch_dev_directml.bat`](windows/launch_dev_directml.bat) | Plus DirectML-like — semantic on; CPU demosaic |
| [`windows/launch_dev_cuda.bat`](windows/launch_dev_cuda.bat) | Plus CUDA-like — semantic on; CuPy GPU demosaic |
| [`windows/run_debug.bat`](windows/run_debug.bat) | Same as launch_dev; supports `menu` / mode args |
| [`windows/clear_cache.bat`](windows/clear_cache.bat) | Dev: wipe caches, logs, QSettings |
| [`windows/clear_cache_user.bat`](windows/clear_cache_user.bat) | Copied next to `RAWviewer.exe` by the installer |
| [`windows/build_windows.bat`](windows/build_windows.bat) | Build unified `dist/RAWviewer_Setup.exe` |
| [`windows/test_builds.bat`](windows/test_builds.bat) | Smoke-test profile resolution |

```batch
scripts\Launch\windows\run_debug.bat menu
scripts\Launch\windows\launch_dev_lite.bat
scripts\Launch\windows\launch_dev_directml.bat
scripts\Launch\windows\launch_dev_cuda.bat
scripts\Launch\windows\clear_cache.bat
scripts\Launch\windows\build_windows.bat
```

### Plus DirectML vs Plus CUDA

| | DirectML | CUDA |
|--|:--:|:--:|
| Semantic / face | ONNX **DirectML** | Same |
| RAW demosaic | CPU Fast RAW | **CuPy** GPU |
| Install deps | No CuPy | `cupy-cuda12x` |

Installer artifact: `dist/RAWviewer_Setup.exe` (wizard picks Standard / Plus DirectML / Plus CUDA).

---

## macOS (`macos/`)

| Script | Purpose |
|--------|---------|
| [`macos/launch_dev.sh`](macos/launch_dev.sh) | Dev launch (Plus defaults) |
| [`macos/launch_dev_full.sh`](macos/launch_dev_full.sh) | Plus profile |
| [`macos/launch_dev_lite.sh`](macos/launch_dev_lite.sh) | Standard profile |
| [`macos/clear_cache.sh`](macos/clear_cache.sh) | Dev: wipe caches / prefs |
| [`macos/build_macos.sh`](macos/build_macos.sh) | Build `full` or `lite` |
| [`macos/build_macos_full.sh`](macos/build_macos_full.sh) | → `dist/RAWviewer.app` + zip |
| [`macos/build_macos_lite.sh`](macos/build_macos_lite.sh) | → `dist/RAWviewer_Lite.app` + zip |
| [`macos/build_macos_all.sh`](macos/build_macos_all.sh) | Build both profiles |
| [`macos/test_builds.sh`](macos/test_builds.sh) | Smoke-test profiles |

```bash
chmod +x scripts/Launch/macos/*.sh scripts/Launch/macos/release/*
./scripts/Launch/macos/launch_dev_full.sh
./scripts/Launch/macos/launch_dev_lite.sh
./scripts/Launch/macos/clear_cache.sh
./scripts/Launch/macos/build_macos_full.sh
```

### Release zip assets (`macos/release/`)

Copied into `dist/RAWviewer-v*-macOS*.zip` by `build_macos.sh`:

| File | Role |
|------|------|
| `install_macos_app.sh` / `Install RAWviewer.command` | Install to Applications + clear quarantine |
| `uninstall_macos_app.sh` / `Uninstall RAWviewer.command` | Remove app + user cache |
| `clear_macos_cache.sh` / `Clear Cache.command` | User cache wipe (keep app) |
| `remove_macos_quarantine.sh` / `Remove Quarantine.command` | Quarantine-only helper |
| `macos_release_readme.txt` | Becomes `Start Here.txt` in the zip |

---

## Notes

- Env-var reference for memory / GPU / session: [`docs/DEVELOPING.md`](../../docs/DEVELOPING.md).
- Day-to-day: `pixi run start`, or `launch_dev_full` on your platform.
- **clear_cache** does not uninstall the app; use `uninstall.bat` (Windows install folder) or `uninstall_macos_app.sh` (release zip).
