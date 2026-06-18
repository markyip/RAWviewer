# RAWviewer v2.4

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.4-blue" alt="Version">
  <img src="https://img.shields.io/github/downloads/markyip/RAWviewer/total" alt="Downloads">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <a href="https://www.buymeacoffee.com/markyip">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee" alt="Buy Me a Coffee">
  </a>
</p>

**RAWviewer** is a fast photo viewer for **Windows and macOS**. Open a folder of RAW or JPEG files, check sharpness, search your shots, and cull rejects — **100% on your computer, no cloud upload.**

**What’s new in v2.4:** **Lite** edition and a **single Windows installer**; **drag photos out** to other apps; **composition guides** (**G**); plus steadier large folders and correct portrait thumbnails.

Official releases: [GitHub Releases](https://github.com/markyip/RAWviewer/releases/latest)

**Contents:** [Choose your version](#choose-your-version) · [RAWviewer Lite](#rawviewer-lite) · [Download & install](#download--install) · [Using RAWviewer](#using-rawviewer) · [Troubleshooting](#something-not-working) · [Advanced reference](#advanced-reference) · [For developers](#for-developers)

---

## Choose your version

| | **Lite** | **Full** |
|---|----------|----------|
| **Best for** | Fast browsing and culling; smaller download | AI-powered search and face filters |
| **Gallery search** | Metadata only (camera, ISO, date, GPS, filename, …) | Plain-language AI search *plus* metadata filters |
| **Face filters** (`has:face`, `people`, …) | No | Yes |
| **Install size / time** | Smaller; no AI model download | Larger; downloads AI models on first setup (Full only) |

**Windows installer:** `RAWviewer_Setup.exe` (Unified Installer)  
**macOS zips:** `RAWviewer-v2.4-macOS.zip` · `RAWviewer-v2.4-macOS-Lite.zip`

Not sure? Start with **Lite** if you mainly browse and cull by eye. Choose **Full** if you want to search with words like `sunset on beach` or filter by detected faces.

---

## RAWviewer Lite

**Lite** is the same RAWviewer viewing experience as **Full**, without the offline AI search stack. You get the gallery, film strip, zoom, histogram, focus overlay, composition guides, drag-out sharing, and metadata-based gallery search — but not plain-language semantic search or face filters.

| | **Why choose Lite** |
|---|---------------------|
| **Less storage** | Smaller download and install; no large AI model files on disk |
| **Faster feel** | No background AI indexing; Lite is tuned to prefetch nearby photos so scrolling and **← / →** navigation stay responsive |
| **Same culling tools** | Discard folder, bookmarks, multi-select, drag to other apps |
| **Search** | By metadata only (`camera:`, `iso:`, `city:`, date, filename, …) — not by describing a scene in words |

**Trade-off:** Lite does **not** run semantic search. Typing `sunset` or `dog on beach` won’t match photos by meaning. Use filters like `camera:canon year>=2024`, or switch to **Full** if you want AI-powered search and face filters.

**Get Lite:** **Windows** — select **Lite** in `RAWviewer_Setup.exe`. **Mac** — `RAWviewer-v2.4-macOS-Lite.zip`.

---

## Download & install

### Windows

1. Download **`RAWviewer_Setup.exe`** from **[Releases](https://github.com/markyip/RAWviewer/releases/latest)** and run it.
2. In the setup wizard, choose your profile:
   - **Full — CUDA** (NVIDIA GPU with CUDA for fastest AI search)
   - **Full — DirectML** (AMD, Intel, or NVIDIA GPU; recommended for most PCs)
   - **Lite** (browsing and culling only; smaller download, no AI models)
3. Stay online — setup downloads the app runtime and dependencies. **Full** builds also download AI search models (~600 MB); this can take several minutes.
4. Open **RAWviewer** from the Desktop shortcut, Start Menu, or **`RAWviewer.exe`** in your install folder (default: `%LOCALAPPDATA%\RAWviewer`).

Setup also registers RAWviewer in Windows **Open with** for common photo formats, so you can right-click a file in Explorer and choose RAWviewer.

**Uninstall:** **Settings → Apps → RAWviewer → Uninstall**, or run **`uninstall.bat`** in the install folder (`%LOCALAPPDATA%\RAWviewer`). This removes the app, your photo cache (`%USERPROFILE%\.rawviewer_cache`), and logs. Window layout and sort preferences are kept unless you set **`RAWVIEWER_UNINSTALL_FULL=1`** before running **`uninstall.bat`** (that also clears `HKCU\Software\RAWviewer`).

If Windows shows **“Protected your PC”**: click **More info** → **Run anyway**.

> **`RAWviewer_Setup.exe` is the installer only** — it does not open the photo viewer. Use **`RAWviewer.exe`** or the Desktop shortcut.

### macOS (13 Ventura or newer)

1. Download **`RAWviewer-v2.4-macOS.zip`** (Full) or **`RAWviewer-v2.4-macOS-Lite.zip`** (Lite) from **[Releases](https://github.com/markyip/RAWviewer/releases/latest)** and extract the zip.
2. Open **Terminal**, go to the extracted folder (`cd ` then drag the folder onto Terminal), and run:

```bash
bash install_macos_app.sh
```

3. Click **Install**, then **Open** in the dialogs. RAWviewer is copied to **Applications**.

**Full builds only:** the first time you use gallery **Search**, RAWviewer may ask to download offline AI models (~150 MB, one-time, needs internet).

To run from the extracted folder without installing: `bash remove_macos_quarantine.sh`

**Uninstall:** In the extracted release folder (keep the zip or re-download from Releases), run **`bash uninstall_macos_app.sh`** or double-click **`Uninstall RAWviewer.command`** (right-click → **Open** if macOS blocks it). This removes RAWviewer from **Applications** and deletes your photo cache (`~/.rawviewer_cache`), logs, and preferences. Dragging the app to Trash alone does not clear cache.

> **Mac too old?** Prebuilt apps need **macOS 13+**. See [macOS version support](#macos-version-support) below.

### System requirements

| | Requirement |
|---|-------------|
| **Windows** | Windows 10 or newer |
| **macOS** | macOS 13 Ventura or newer |
| **Disk (Lite)** | ~500 MB install + cache as you browse |
| **Disk (Full)** | ~1.5 GB+ (includes AI models after first setup) |
| **RAM** | 8 GB minimum; **16 GB+** recommended for **Full** + large folders (10k+ photos). RAWviewer adjusts how hard it works based on your RAM (see [Automatic memory tuning](#automatic-memory-tuning)). On **macOS**, v2.4 is better at handling very large folders without stuttering. |

### Uninstall vs clear cache

| Action | Windows | macOS |
|--------|---------|-------|
| **Uninstall app** | Settings → Apps, or **`uninstall.bat`** in `%LOCALAPPDATA%\RAWviewer` | **`uninstall_macos_app.sh`** or **`Uninstall RAWviewer.command`** in the release zip folder |
| **Removes app** | Install folder | **`RAWviewer.app`** / **`RAWviewer_Lite.app`** in Applications |
| **Removes photo cache** | `%USERPROFILE%\.rawviewer_cache` | `~/.rawviewer_cache` |
| **Removes preferences** | Only if **`RAWVIEWER_UNINSTALL_FULL=1`** | Always (with uninstall script) |
| **Clear cache only (keep app)** | **`scripts\Launch\bat\clear_cache.bat`** | **`scripts/Launch/shell/clear_cache.sh`** |

---

## Using RAWviewer

1. **Open a folder** — File menu, drag-and-drop, or double-click a photo (Windows: use **Open with → RAWviewer** after install).
2. **Gallery view** — scroll the grid; click a photo for full-screen view.
3. **Check sharpness** — press **`Space`** for 100% zoom; **`←` / `→`** for prev/next.
4. **Search** — click the search icon in gallery view. **Full:** type plain words (`sunset`, `airplane`) or filters (`camera:sony` `iso<800`). **Lite:** metadata filters only (see [Advanced → Gallery search syntax](#gallery-search-syntax)).
5. **Open in another app** — use the bottom external-app button to open the current photo, gallery selection, or bookmarked photos in an editor. On Windows, single files use the native **Open with** picker; on macOS, choose a `.app` once and reuse it.
6. **Reject a shot** — **`↓`** moves it to a **Discard** subfolder; **Delete** removes it (with confirmation).
7. **`Esc`** returns from single view to the gallery.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| **G** | Cycle composition guide (off / rule of thirds / diagonals / both / phi grid) |
| **Space** | Fit ↔ 100% zoom |
| **← / →** | Previous / next image |
| **↓** | Move to Discard folder |
| **Delete** | Delete (with confirm) |
| **Esc** | Back to gallery |
| **H** | Show / hide histogram |
| **F** | Show focus / subject box (when supported) |
| **↑** | Bookmark / unbookmark (single view, or selected thumbnails in gallery) |

### Mouse & trackpad

- **Double-click** — zoom in on a point, or back to fit
- **Drag** — pan when zoomed in
- **Pinch / Ctrl+scroll** — zoom
- **Scroll wheel** — next/previous (single view) or scroll gallery grid
- **Film strip** — move the pointer to the bottom in single view for thumbnails
- **Gallery multi-select** — **Ctrl+click** (Windows) or **Cmd+click** (Mac) to add or remove photos from a selection; **Shift+click** a second photo to select the range in between. Drag any selected thumbnail outward to copy or share all selected files.

### Search tips

- **Full:** type everyday words: `dog on beach`, `crowd`, `portrait`.
- Add filters on the same line: `jet takeoff camera:canon iso<800`.
- Words like **`people`** or **`face`** filter by detected faces (**Full** only) — if nothing matches, try `crowd` or `spectators` instead.
- **Lite:** everyday words won’t run AI search — use filters like `camera:sony`, `iso<800`, or `city:tokyo`.
- Clear the search box to show the whole folder again.

Full search syntax, focus-overlay brands, and power-user options are in **[Advanced reference](#advanced-reference)** below.

---

## What you get

- **Lite or Full** — Lite for fast browse-and-cull; Full adds plain-language AI search and face filters
- **One Windows installer** — `RAWviewer_Setup.exe` lets you pick Full (CUDA / DirectML) or Lite in one wizard
- **Drag photos out** — From single view, gallery, or film strip to Explorer, Finder, Mail, editors, and more
- **Composition guides** — Press **G** for rule-of-thirds, diagonal, or phi grid overlays while reviewing shots
- Fast **RAW + JPEG** viewing (Canon, Nikon, Sony, DNG, and many more)
- **Large folders feel smoother** — especially on Mac, with tens of thousands of photos
- **Portrait photos look correct** — gallery and film-strip thumbnails no longer appear sideways
- **Select several gallery photos** — Ctrl/Cmd+click and Shift+click, then drag them out together
- **Offline AI search** (**Full**) — describe photos in plain language; nothing leaves your PC
- **Metadata filters** — camera, lens, ISO, date, GPS, file type (both Lite and Full)
- **Open in another app** — send the current photo, a gallery selection, or bookmarks to an external editor
- **Focus overlay** (`F`) on many Canon / Nikon / Sony / Olympus / Panasonic files
- **Windows Open with** — registered for common RAW and photo formats after install
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
| Slow AI search (**Full**) | Prefer **DirectML** on most PCs; use **CUDA** only with NVIDIA + CUDA |
| Installer stuck on “Downloading models” (**Full**) | AI models (~600 MB) can take several minutes. Check firewall, VPN, or proxy if it fails — browsing still works; open gallery **Search** later to retry |
| Opened Setup again instead of the app | Launch **`RAWviewer.exe`** or the Desktop shortcut — not **`RAWviewer_Setup.exe`** |
| AI search missing after install (**Full**) | Open gallery **Search** → accept the download prompt |
| RAWviewer not in Open with | Re-run the installer (repair), or reinstall |
| Leftover cache after uninstall | Run **`uninstall.bat`** again, or delete `%USERPROFILE%\.rawviewer_cache` manually |
| Maximize button needs two clicks | Update to **v2.4** — fixed for the custom title bar on Windows |
| Out of memory during AI indexing | See [Automatic memory tuning](#automatic-memory-tuning); use **Lite** on 8 GB PCs or set `RAWVIEWER_MEMORY_TIER_AUTO=0` and lower workers manually |
| Crash | Enable file logging with `RAWVIEWER_FILE_LOG=1`, then check the install folder |

### macOS

| Problem | What to do |
|---------|------------|
| macOS blocks the app (“damaged” / won’t open) | In the extracted folder, run `bash install_macos_app.sh` (see install steps above) |
| `bash: command not found` | Type `cd `, drag the extracted folder onto Terminal, press Return, then run the command again |
| Can’t read Desktop/Documents | System Settings → Privacy → **Full Disk Access** → add RAWviewer |
| Search says models missing (**Full**) | Open gallery search and click **Download** when prompted (needs internet once) |
| Download failed (SSL / certificate error) | Update to **v2.4** or newer (bundles certifi). On a corporate VPN or proxy, add your organization’s root certificate to **Keychain Access** and set it to **Always Trust** |
| Need to uninstall completely | Use **`uninstall_macos_app.sh`** or **`Uninstall RAWviewer.command`** from the release zip — not Trash alone |
| Uninstall scripts missing | Re-download the release zip from [Releases](https://github.com/markyip/RAWviewer/releases/latest); scripts are inside the extracted folder |
| macOS “out of memory” / heavy swap during indexing | See [Automatic memory tuning](#automatic-memory-tuning). On 8 GB Macs, prefer **Lite** or wait for indexing to finish before opening gallery on huge folders |
| Gallery stutters on a huge folder | Update to **v2.4**. If it persists, run **`clear_cache.sh`** and reopen the folder |
| Thumbnails sideways or wrong way up (portrait shots) | Update to **v2.4**. Run **`clear_cache.sh`** once if old thumbnails were cached before the fix |

More detail: [`scripts/Launch/README.md`](scripts/Launch/README.md)

---

## Advanced reference

*Optional — for search power users and troubleshooting.*

### Gallery search syntax

Separate words with spaces. Use `key:value` filters:

| Kind | Example |
|------|---------|
| Free text + filter | `jet takeoff camera:sony iso<800` *(Full: free text uses AI)* |
| Camera / lens | `camera:canon` · `lens:70-200` |
| ISO / year | `iso<=800` · `year>=2024` |
| Place | `city:tokyo` · `country:jp` |
| File name | `filename:_dsc` |
| Format | `format:raw` · `format:jpeg` · `format:cr3` |
| Date | `date:2024-05` |
| GPS / faces | `has:gps` · `has:face` · `no:face` *(face filters: Full only)* |

**Face vs semantic search:** `face`, `people`, `person`, etc. use stored face counts (`has:face`), not the neural search.

**Indexing:** On **Full** builds, semantic search and face counts run in the background on large folders (metadata + AI first, faces after). The gallery becomes searchable before face tagging finishes.

### Automatic memory tuning

On every launch, RAWviewer reads **installed system RAM** (not free RAM at that moment) and applies conservative defaults for load concurrency, preview cache, prefetch, and indexing — **only when you have not already set the same environment variables yourself**.

| Tier | Installed RAM | Typical Mac | What changes (summary) |
|------|---------------|-------------|-------------------------|
| **low** | &lt; 10 GB | 8 GB MacBook Air | Face scan off during indexing; fewer parallel workers; smaller preview cache; less idle prefetch |
| **medium** | 10–14 GB | 12 GB unified | Moderate limits on workers and cache |
| **balanced** | 14–20 GB | 16 GB | Light tuning (default on many laptops) |
| **high** | 20–28 GB | 24 GB | Slightly higher cache / worker caps |
| **ultra** | ≥ 28 GB | 32 GB+ studio machines | Stock app defaults (no overrides) |

**What you might notice**

- Startup log (dev / Terminal): `[PROFILE] memory tier=balanced (16.0 GB RAM)`
- A note file: `~/.rawviewer_cache/memory_tier.json` (tier, RAM, how many defaults were applied)
- **Lite** builds still use Lite profile defaults first; RAM tier only fills in gaps
- **Full** on an 8 GB Mac: semantic AI search can still run, but face indexing is disabled automatically to reduce memory pressure

**Disable auto-tuning** (use only your own env vars or scripts):

```bash
export RAWVIEWER_MEMORY_TIER_AUTO=0
```

**Force a specific override** (wins over auto-tuning — examples for low-RAM machines):

```bash
export RAWVIEWER_ENABLE_FACE_SCAN=0
export RAWVIEWER_SEMANTIC_PREP_WORKERS=2
export RAWVIEWER_MEMORY_PREVIEW_MAX=1280
export RAWVIEWER_IDLE_DISPLAY_PREFETCH=0
```

Semantic batch/chunk size for AI indexing is **auto-tuned separately** on first index pass (Core ML on macOS, ONNX on Windows); results are cached under `~/.rawviewer_cache/semantic_batch_tuning.json`.

### MobileCLIP models (Full — AI search)

| Platform | When downloaded | Change variant (Windows) |
|----------|-----------------|--------------------------|
| **Windows Full** | During setup (~600 MB) | Set `RAWVIEWER_MOBILECLIP_VARIANT` to `s0`, `s2`, `b`, or `l14` |
| **macOS Full** | First gallery search (~150 MB) | Dev helper: `python scripts/download_mobileclip_coreml.py --out-dir models/mobileclip2_coreml` |

**Lite builds** do not use MobileCLIP models.

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
| `RAWVIEWER_MEMORY_TIER_AUTO=1` | **Default.** Tune workers, cache, and prefetch from installed RAM at startup |
| `RAWVIEWER_MEMORY_TIER_AUTO=0` | Disable RAM-tier defaults; only explicit env vars apply |
| `RAWVIEWER_MOBILECLIP_VARIANT` | Windows ONNX model: `b` (default), `s0`, `s2`, `l14` |
| `RAWVIEWER_GPU_VIEW=1` | GPU single-image viewport (OpenGL zoom/pan; on by default in release builds) |
| `RAWVIEWER_GPU_VIEW=0` | Force legacy scroll-area single-image view |
| `RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1` | Same color pipeline for fit vs 100% zoom on RAW (default on) |
| `RAWVIEWER_EXIF_BACKEND=auto` | `auto`, `pyexiv2`, or `exifread` |
| `RAWVIEWER_SHARE_MENU=1` | macOS: Qt share menu (recommended) |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | macOS: try native share sheet first |
| `RAWVIEWER_INDEX_DEFER_FACE_SCAN=1` | Defer face scan until after semantic index (default) |
| `RAWVIEWER_SEMANTIC_PREP_WORKERS` | Parallel CPU workers before AI encode (RAM tier may set this) |
| `RAWVIEWER_SEMANTIC_BATCH_AUTO=1` | Auto-tune AI batch/chunk size on index (default) |
| `RAWVIEWER_SEMANTIC_BATCH_CANDIDATES` | Candidate batch sizes for auto-tune (default `8,16,32,64,128`) |
| `RAWVIEWER_PREVIEW_CACHE_ITEMS` | Cap in-memory preview LRU count |
| `RAWVIEWER_MEMORY_PREVIEW_MAX` | Max long edge for in-memory RAW/JPEG preview (pixels) |
| `RAWVIEWER_IDLE_DISPLAY_PREFETCH=0` | Disable idle neighbor prefetch in single view |

Full list and dev defaults: [`scripts/Launch/README.md`](scripts/Launch/README.md), [`docs/macos-sharing-v21-v22.md`](docs/macos-sharing-v21-v22.md).

</details>

### macOS version support

| Your Mac | Official `.zip` | Build from source |
|----------|-----------------|-------------------|
| macOS 13 Ventura (Intel) | ✅ | `build_macos_full.sh` or Pixi |
| macOS 13 Ventura (Apple Silicon) | ✅ | Use **`build_macos_full.sh`** (Pixi needs 14+) |
| macOS 14 Sonoma+ | ✅ | Pixi or `build_macos.sh` |
| macOS 12 Monterey or older | ❌ | ❌ |

### Upcoming (development branch)

Not in a release yet — tracked on a separate development branch.

**GPU-accelerated RAW decoding** — Early GPU decode works, but **correct color rendering** (matching the current LibRaw / embedded-JPEG pipeline) is still unresolved. We will only ship it if color accuracy and maintenance cost are acceptable.

This is separate from the GPU **viewport** (OpenGL zoom/pan on decoded pixels, on by default in release builds; set `RAWVIEWER_GPU_VIEW=0` to disable); the upcoming work targets **RAW decode** itself.

---

## For developers

Scripts and build matrix: [`scripts/Launch/README.md`](scripts/Launch/README.md)

### Quick start

```bash
pixi install
pixi run start          # full profile (default)
```

**Windows**

| Task | Script |
|------|--------|
| Run (full) | `scripts\Launch\bat\launch_dev_full.bat` |
| Run (lite) | `scripts\Launch\bat\launch_dev_lite.bat` |
| Build Full installers | `scripts\Launch\bat\build_windows_full.bat` (CUDA) or `build_windows_full.bat directml` |
| Build Lite installer | `scripts\Launch\bat\build_windows_lite.bat` |
| Build both Full backends | `scripts\Launch\bat\build_windows_all.bat` |

**macOS**

| Task | Script |
|------|--------|
| Run (full) | `./scripts/Launch/shell/launch_dev_full.sh` |
| Run (lite) | `./scripts/Launch/shell/launch_dev_lite.sh` |
| Build Full | `./scripts/Launch/shell/build_macos_full.sh` → `dist/RAWviewer.app` |
| Build Lite | `./scripts/Launch/shell/build_macos_lite.sh` → `dist/RAWviewer_Lite.app` |

Build outputs:

| Profile | Windows | macOS |
|---------|---------|-------|
| **Full / Unified** | `dist/RAWviewer_Setup.exe` (includes Full & Lite options) | `dist/RAWviewer-v2.4-macOS.zip` |
| **Lite** | (Select Lite option in `RAWviewer_Setup.exe`) | `dist/RAWviewer-v2.4-macOS-Lite.zip` |

Dependencies are in `pixi.toml`. Packaging scripts use a local `rawviewer_env/` venv when building release artifacts.

<details>
<summary><strong>Build from source (commands)</strong></summary>

**Windows**
```batch
scripts\Launch\bat\build_windows_full.bat
scripts\Launch\bat\build_windows_lite.bat
```

**macOS**
```bash
./scripts/Launch/shell/build_macos_full.sh
./scripts/Launch/shell/build_macos_lite.sh
# or: pixi install && pixi run python build.py --profile full
```

</details>

### Architecture (brief)

- **ImageLoadManager** — threaded load queue  
- **UnifiedImageProcessor** — RAW/JPEG/TIFF via one path  
- **Cache** — memory-first; optional disk cache via env; **RAM-tier defaults** at startup (`rawviewer_profile.py`)
- **Semantic index** — SQLite + local embeddings (Core ML on macOS, ONNX on Windows; Full builds only)

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
