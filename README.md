# RAWviewer v2.5

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.5-blue" alt="Version">
  <img src="https://img.shields.io/github/downloads/markyip/RAWviewer/total" alt="Downloads">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <a href="https://www.buymeacoffee.com/markyip">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee" alt="Buy Me a Coffee">
  </a>
</p>

**RAWviewer** is a fast photo viewer for **Windows and macOS**. Browse folders of RAW and JPEG files, check sharpness, cull rejects, and search your library — **on your computer, no cloud upload.**

Download: **[GitHub Releases](https://github.com/markyip/RAWviewer/releases/latest)**

---

## Using RAWviewer

Open a folder (menu, drag-and-drop, or double-click a photo). Scroll the **gallery**; click a thumbnail for full-screen view.

| Key | Action |
|-----|--------|
| **Space** / **Double-click** | Toggle fit-to-window / 100% zoom |
| **Pinch** / **Ctrl+scroll** | Zoom in or out |
| **←** / **→** | Previous / next image |
| **Scroll wheel** | Previous / next (single view, fit mode) |
| **↑** | Bookmark / unbookmark (bottom **star** in single view too) |
| **↓** | Move to Discard folder |
| **Delete** | Delete image(s) |
| **Esc** | Gallery: clear selection → exit bookmark filter · Single view: back to gallery |
| **Ctrl/Cmd+click** | Gallery: toggle selection |
| **Shift+click** | Gallery: select range (visible order) |
| **G** | Cycle composition guide |
| **H** | Show / hide histogram |
| **F** | Show / hide focus overlay (supported files) |
| **M** | Show / hide GPS map overlay (single view, geotagged photos) |

**Gallery bookmarks:** Click the outline **star** (nothing selected) to show bookmarked shots only; gold star = filter on. With photos selected, **↑** or the star toggles bookmarks on the selection.

**Search:** gallery search icon — `camera:sony`, `iso<800`, … (**Full** also accepts `sunset on beach`). **Share:** bottom **Share / Open** button, or drag gallery / film-strip thumbnails out.

Search syntax → [Advanced reference](#advanced-reference).

---

## Lite vs Full

Both editions share the same viewer, culling tools, bookmarks, and metadata search. **Full** adds offline AI search and face filters.

| | Lite | Full |
|---|:--:|:--:|
| Gallery, film strip, zoom, histogram, bookmarks, culling | ✅ | ✅ |
| Metadata search (`camera:`, `iso:`, `date:`, …) | ✅ | ✅ |
| Plain-language search | — | ✅ |
| Face filters (`has:face`, …) | — | ✅ |

Pick **Lite** for a smaller install and browse-by-eye workflow. Pick **Full** to search with everyday words — still 100% offline.

---

## GPS map overlay & geotagging

In **single-image view**, press **M** to toggle an interactive tile-based map card showing the photo's GPS coordinates and a location pin.

Bundled offline databases (`cities500.csv.gz` and `landmarks.csv.gz`, 100,000+ locations) are used during **background indexing** to resolve GPS coordinates into city, region, and country names. These power **gallery search** — search `city:tokyo`, `country:jp`, or similar — with no internet required. The map overlay itself shows the raw coordinates pin; it does not display a resolved place name.

For a dedicated **cluster map** across an entire album and **geotagging photos missing GPS**, see **[LocateIt](https://github.com/markyip/LocateIt)**: open a folder, see where shots were taken on a map, drag-drop to assign coordinates, and save back to JPEG or RAW.

---

## Download & install

### Windows

1. Download **`RAWviewer_Setup.exe`** from [Releases](https://github.com/markyip/RAWviewer/releases/latest).
2. Choose **Full (CUDA)**, **Full (DirectML)**, or **Lite** in the wizard. **Full** also downloads AI models (~600 MB).
3. Launch **`RAWviewer.exe`** or the Desktop shortcut (not the Setup file again).

> **v2.5 new:** Gallery zoom slider, scroll anchoring, GPS map overlay (**M**), and animated GIF/WebP playback.

Registers **Open with** for common photo formats. Uninstall: Settings → Apps, or **`uninstall.bat`** in `%LOCALAPPDATA%\RAWviewer`.

### macOS (13+)

1. Download **`RAWviewer-v2.5-macOS.zip`** (Full) or **`RAWviewer-v2.5-macOS-Lite.zip`** (Lite) from **[Releases](https://github.com/markyip/RAWviewer/releases/latest)** and extract the zip.
2. Open **Terminal**, go to the extracted folder (`cd ` then drag the folder onto Terminal), and run:

```bash
bash install_macos_app.sh
```

3. Click **Install**, then **Open** in the dialogs. RAWviewer is copied to **Applications**.

**Full edition:** The first time you use gallery **Search**, RAWviewer may prompt to download the offline AI models from [Hugging Face](https://huggingface.co/) (~150 MB on macOS, one-time, needs internet). Without a Hugging Face account, that download may take longer. Click **Download** when prompted — progress appears in the search bar as `Downloading... N%`. Windows setup fetches the same models automatically during install.

Uninstall: **`uninstall_macos_app.sh`** or **`Uninstall RAWviewer.command`** in the zip (keeps cache cleared; Trash alone does not).

### Requirements

Windows 10+ · macOS 13+ · 8 GB RAM (16 GB+ recommended for **Full** + large folders) · ~500 MB disk (**Lite**) or ~1.5 GB+ (**Full** with models)

To clear thumbnails only: **`scripts\Launch\bat\clear_cache.bat`** (Windows) · **`scripts/Launch/shell/clear_cache.sh`** (Mac)

---

## Supported formats

**RAW:** CR2, CR3, NEF, ARW, DNG, ORF, RW2, RAF, and other LibRaw types · **Standard:** JPEG, TIFF, HEIF, **GIF** (animated), **WebP** (animated)

---

## Troubleshooting

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
| Out of memory during AI indexing | See [Automatic memory tuning](#automatic-memory-tuning); use **Lite** on 8 GB PCs or set `RAWVIEWER_MEMORY_TIER_AUTO=0` and lower workers manually |
| App slow or exits after reopening last folder | **v2.4.1+** staggers full decode and prefetch on session restore. Still tight on 8 GB? Use **Lite**, or `RAWVIEWER_DISABLE_SESSION_RESTORE=1` |
| GPS map not showing | Press **M** in single-image view; the map only appears when the photo has embedded GPS coordinates |
| Animated GIF/WebP not playing | Update to **v2.5+**; check that the file is a valid animated GIF or WebP |
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
| Killed on relaunch (`Killed: 9` / exit 137 in Terminal) | **v2.4.1+** fixes most session-restore bursts. Try **Lite**, `RAWVIEWER_DISABLE_SESSION_RESTORE=1`, or `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| Gallery stutters on a huge folder | Update to **v2.4**. If it persists, run **`clear_cache.sh`** and reopen the folder |
| Thumbnails sideways or wrong way up (portrait shots) | Update to **v2.4**. Run **`clear_cache.sh`** once if old thumbnails were cached before the fix |
| GPS map not showing | Press **M** in single-image view; the map only appears when the photo has embedded GPS coordinates |
| Animated GIF/WebP not playing | Update to **v2.5+**; check that the file is a valid animated GIF or WebP |

More detail: [`scripts/Launch/README.md`](scripts/Launch/README.md)

---

## Advanced reference

*Optional — for search power users and troubleshooting.*

> **Thumbnail cache notice:** To speed up gallery loading, RAWviewer creates a local thumbnail cache on your device. This cache is **never uploaded or shared** — it stays entirely on your machine. Cache files are automatically deleted after **30 days** of inactivity.

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
- **Relaunch (v2.4.1+):** Session restore staggers full decode and prefetch so reopening the last folder is less likely to OOM; see release notes if fit view stays soft for a few seconds

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
| `RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD=1` | **Default.** After relaunch, delay full decode and neighbor prefetch (see v2.4.1 release notes) |
| `RAWVIEWER_SESSION_RESTORE_FULL_DECODE_DELAY_MS` | Milliseconds to wait after first paint before full decode on session restore (default `2500`) |
| `RAWVIEWER_DISABLE_SESSION_RESTORE=1` | Do not reopen the last folder/file on launch |

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
| **Full / Unified** | `dist/RAWviewer_Setup.exe` (includes Full & Lite options) | `dist/RAWviewer-v2.5-macOS.zip` |
| **Lite** | (Select Lite option in `RAWviewer_Setup.exe`) | `dist/RAWviewer-v2.5-macOS-Lite.zip` |

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

1. Check [Troubleshooting](#troubleshooting) above  
2. Search [existing issues](https://github.com/markyip/RAWviewer/issues)  
3. Open a new issue with OS version, steps, and logs if possible  

## ☕ Buy Me a Coffee

If RAWviewer helps your workflow, you can [buy me a coffee](https://www.buymeacoffee.com/markyip) ☕

---

**Enjoy your photos.** 📸
