# RAWviewer v2.3.1

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

![Version](https://img.shields.io/badge/version-2.3.1-blue)
![Downloads](https://img.shields.io/github/downloads/markyip/RAWviewer/total) 
![License](https://img.shields.io/badge/license-MIT-green)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/markyip)

**RAWviewer** is a fast photo viewer for **Windows and macOS**. Open a folder of RAW or JPEG files, check sharpness, search your shots, and cull rejects — **100% on your computer, no cloud upload.**

Official releases: [GitHub Releases](https://github.com/markyip/RAWviewer/releases/latest)

---

## Download & install

### Windows

1. Go to **[Releases](https://github.com/markyip/RAWviewer/releases/latest)** and download:
   - **`RAWviewer_Setup_DirectML.exe`** — **recommended for most PCs** (AMD / Intel / NVIDIA; no CUDA setup)
   - **`RAWviewer_Setup_CUDA.exe`** — optional if you have an **NVIDIA** GPU with CUDA and want maximum AI search speed
2. Run the **Setup** installer and pick an install folder (default: `%LOCALAPPDATA%\RAWviewer`).
3. Stay online while setup runs. The installer downloads:
   - the **Pixi** runtime manager
   - **Python and all app dependencies** (`pixi install`)
   - **MobileCLIP ONNX models** for offline AI search (no Hugging Face account needed)
   
   Setup usually takes several minutes. If AI models fail to download, installation still completes and you can browse photos; open **Search** in the gallery later to download the models.
4. Open **RAWviewer** from the **Desktop shortcut**, **Start Menu**, or **`RAWviewer.exe`** in the install folder (default: `%LOCALAPPDATA%\RAWviewer\RAWviewer.exe`).  
   **`RAWviewer_Setup_*.exe`** is only for install/repair — it does not open the photo viewer.

**Uninstall:** Windows **Settings → Apps → Installed apps → RAWviewer → Uninstall**, or run `uninstall.bat` in the install folder. Your photo cache and app preferences in `%USERPROFILE%\.rawviewer_cache` and Windows settings are kept unless you delete them manually.

If Windows shows **“Protected your PC”**: click **More info** → **Run anyway**.

### macOS (13 Ventura or newer)

1. Download **`RAWviewer-v2.3.0-macOS.zip`** from **[Releases](https://github.com/markyip/RAWviewer/releases/latest)**.
2. Double-click the zip to extract the folder. Open **`Start Here.txt`** — it tells you which file to click.
3. **Recommended:** double-click **`Install RAWviewer.command`** → **Install** → **Open**.  
   **Or:** double-click **`Remove Quarantine.command`**, then open **`RAWviewer.app`** in the same folder.

If macOS blocks a script the first time: **right-click it → Open → Open** (once only).

> **Mac too old?** Prebuilt apps need **macOS 13+**. Monterey (12) and older are not supported. Details are in [Advanced → macOS version support](#macos-version-support).

---

## Using RAWviewer

1. **Open a folder** (File menu, drag-and-drop, or double-click a photo with RAWviewer set as default).
2. **Gallery view** — scroll the grid; click a photo for full-screen view.
3. **Check sharpness** — press **`Space`** for 100% zoom; **`←` / `→`** for prev/next.
4. **Search** — click the search icon in gallery view; type plain words (e.g. `sunset`, `airplane`) or filters like `camera:sony` `iso<800`.
5. **Reject a shot** — **`↓`** moves it to a **Discard** subfolder; **Delete** removes it (with confirmation).
6. **`Esc`** returns from single view to the gallery.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| **Space** | Fit ↔ 100% zoom |
| **← / →** | Previous / next image |
| **↓** | Move to Discard folder |
| **Delete** | Delete (with confirm) |
| **Esc** | Back to gallery |
| **H** | Show / hide histogram |
| **F** | Show focus / subject box (when supported) |

### Mouse & trackpad

- **Double-click** — zoom in on a point, or back to fit
- **Drag** — pan when zoomed in
- **Pinch / Ctrl+scroll** — zoom
- **Scroll wheel** — next/previous (single view) or scroll gallery grid
- **Film strip** — move the pointer to the bottom in single view for thumbnails

### Search tips

- Type everyday words: `dog on beach`, `crowd`, `portrait`.
- Add filters on the same line: `jet takeoff camera:canon iso<800`.
- Words like **`people`** or **`face`** filter by detected faces, not AI meaning — if nothing matches, try `crowd` or `spectators` instead.
- Clear the search box to show the whole folder again.

Full search syntax, focus-overlay brands, and power-user options are in **[Advanced reference](#advanced-reference)** below.

---

## What you get

- Fast **RAW + JPEG** viewing (Canon, Nikon, Sony, DNG, and many more)
- **Offline AI search** — describe photos in plain language; nothing leaves your PC
- **Metadata filters** — camera, lens, ISO, date, GPS, file type
- **Focus overlay** (`F`) on many Canon / Nikon / Sony / Olympus / Panasonic files
- **Share** (macOS single view) — Mail, Messages, etc.
- Remembers your last folder and position

---

## Supported formats

**RAW:** CR2, CR3, NEF, ARW, DNG, ORF, RW2, RAF, and other LibRaw types  
**Standard:** JPEG, TIFF, HEIF

---

## Something not working?

### Windows

| Problem | What to do |
|---------|------------|
| SmartScreen warning | More info → Run anyway |
| Slow search | Prefer **DirectML** on most PCs; use **CUDA** only with NVIDIA + CUDA |
| Installer stuck on “Downloading models” | Needs internet once; check firewall, VPN, or proxy; click **Retry** if setup failed |
| Opened Setup again instead of the app | Launch **`RAWviewer.exe`** or the Desktop shortcut — not **`RAWviewer_Setup_*.exe`** |
| AI search missing after install | Open gallery **Search** → accept the download prompt (browsing still works) |
| Crash | Enable file logging with `RAWVIEWER_FILE_LOG=1`, then check the install folder |

### macOS

| Problem | What to do |
|---------|------------|
| “App is damaged” / won’t open | Run **`Install RAWviewer.command`** or **`Remove Quarantine.command`** from the zip |
| Still blocked | Terminal: `xattr -cr /Applications/RAWviewer.app` |
| Can’t read Desktop/Documents | System Settings → Privacy → **Full Disk Access** → add RAWviewer |
| Search says models missing | Re-download the release zip; rebuild instructions for developers are below |

More detail: [`scripts/Launch/README.md`](scripts/Launch/README.md)

---

## Advanced reference

*Everything below is optional — for search power users, developers, and troubleshooting.*

### Gallery search syntax

Separate words with spaces. Use `key:value` filters:

| Kind | Example |
|------|---------|
| Free text + filter | `jet takeoff camera:sony iso<800` |
| Camera / lens | `camera:canon` · `lens:70-200` |
| ISO / year | `iso<=800` · `year>=2024` |
| Place | `city:tokyo` · `country:jp` |
| File name | `filename:_dsc` |
| Format | `format:raw` · `format:jpeg` · `format:cr3` |
| Date | `date:2024-05` |
| GPS / faces | `has:gps` · `has:face` · `no:face` |

**Face vs semantic search:** `face`, `people`, `person`, etc. use stored face counts (`has:face`), not the neural search.

**Indexing:** Semantic search and face counts run in the background on large folders (metadata + AI first, faces after). The gallery becomes searchable before face tagging finishes.

### MobileCLIP models (AI search)

| Platform | Default | Change it |
|----------|---------|-----------|
| **Windows** | MobileCLIP2-**B** | Set `RAWVIEWER_MOBILECLIP_VARIANT` to `s0`, `s2`, `b`, or `l14` |
| **macOS** | Bundled Core ML (**S0** or **S2** in the app) | Replace models before building; see `models/mobileclip2_coreml/` |

Windows variants download from Hugging Face (`plhery/mobileclip2-onnx`) into separate cache folders (e.g. `~/.rawviewer_cache/mobileclip_onnx_s0`).

### Focus overlay (`F`) by brand

| Brand | Support |
|-------|---------|
| Canon CR2/CR3, Nikon NEF, Sony ARW, Olympus ORF, Panasonic RW2 | Yes (maker AF) |
| JPEG / TIFF / HEIF | Sometimes (EXIF SubjectArea) |
| Fujifilm RAF, Hasselblad 3FR, Pentax PEF, Samsung SRW, Sigma X3F | No |
| Typical Adobe DNG | Usually no |

Requires **pyexiv2** for maker-note AF on RAW.

### Environment variables

<details>
<summary><strong>Click to expand — dev / tuning flags</strong></summary>

| Variable | Effect |
|----------|--------|
| `RAWVIEWER_MOBILECLIP_VARIANT` | Windows ONNX model: `b` (default), `s0`, `s2`, `l14` |
| `RAWVIEWER_GPU_VIEW=1` | GPU single-image viewport (OpenGL zoom/pan; on by default in release builds) |
| `RAWVIEWER_GPU_VIEW=0` | Force legacy scroll-area single-image view |
| `RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1` | Same color pipeline for fit vs 100% zoom on RAW (default on) |
| `RAWVIEWER_EXIF_BACKEND=auto` | `auto`, `pyexiv2`, or `exifread` |
| `RAWVIEWER_SHARE_MENU=1` | macOS: Qt share menu (recommended) |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | macOS: try native share sheet first |
| `RAWVIEWER_INDEX_DEFER_FACE_SCAN=1` | Defer face scan until after semantic index (default) |

Full list and dev defaults: [`scripts/Launch/README.md`](scripts/Launch/README.md), [`docs/macos-sharing-v21-v22.md`](docs/macos-sharing-v21-v22.md).

</details>

### macOS version support

| Your Mac | Official `.zip` | Build from source |
|----------|-----------------|-------------------|
| macOS 13 Ventura (Intel) | ✅ | `build_macos.sh` or Pixi |
| macOS 13 Ventura (Apple Silicon) | ✅ | Use **`build_macos.sh`** (Pixi needs 14+) |
| macOS 14 Sonoma+ | ✅ | Pixi or `build_macos.sh` |
| macOS 12 Monterey or older | ❌ | ❌ |

### Development & building from source

Scripts: [`scripts/Launch/`](scripts/Launch/README.md)

| Platform | Run | Build |
|----------|-----|-------|
| Windows | `scripts\Launch\bat\run_debug.bat` | `scripts\Launch\bat\build_windows_all.bat` (both), or `build_windows_directml.bat` / `build_windows_cuda.bat` |
| macOS | `./scripts/Launch/shell/launch_dev.sh` | `./scripts/Launch/shell/build_macos.sh` |

**Pixi (optional):** `pixi install` → `pixi run start`  
**macOS release zip:** `./scripts/Launch/shell/build_macos.sh` → `dist/RAWviewer-v2.3.1-macOS.zip` (prebuilt macOS downloads may still be **2.3.0** until a new macOS release is published)

Dependencies are in `pixi.toml`. Build scripts use a local `rawviewer_env/` venv when packaging.

<details>
<summary><strong>Building from source (commands)</strong></summary>

**Windows**
```batch
scripts\Launch\bat\build_windows.bat
```

**macOS**
```bash
./scripts/Launch/shell/build_macos.sh
# or: pixi install && pixi run python build.py
```

</details>

### Architecture (brief)

- **ImageLoadManager** — threaded load queue  
- **UnifiedImageProcessor** — RAW/JPEG/TIFF via one path  
- **Cache** — memory-first; optional disk cache via env  
- **Semantic index** — SQLite + local embeddings (Core ML on macOS, ONNX on Windows)

### Upcoming (development branch)

Not in a release yet — tracked on a separate development branch while we assess whether the integration effort is worth it.

**GPU-accelerated RAW decoding** — Early GPU decode works, but **correct color rendering** (matching the current LibRaw / embedded-JPEG pipeline) is still unresolved. We will only ship it if color accuracy and maintenance cost are acceptable.

This is separate from the GPU **viewport** (OpenGL zoom/pan on decoded pixels, on by default in release builds; set `RAWVIEWER_GPU_VIEW=0` to disable); the upcoming work targets **RAW decode** itself.

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Pull requests welcome on [GitHub](https://github.com/markyip/RAWviewer).

## Support

1. Check [Troubleshooting](#something-not-working) above  
2. Search [existing issues](https://github.com/markyip/RAWviewer/issues)  
3. Open a new issue with OS version, steps, and logs if possible  

## ☕ Buy Me a Coffee

If RAWviewer helps your workflow, you can [buy me a coffee](https://www.buymeacoffee.com/markyip) ☕

---

**Enjoy your photos.** 📸
