# RAWviewer v3.0

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-3.0-blue" alt="Version">
  <img src="https://img.shields.io/github/downloads/markyip/RAWviewer/total" alt="Downloads">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <a href="https://www.buymeacoffee.com/markyip">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee" alt="Buy Me a Coffee">
  </a>
</p>

**Language / 語言：** [English](README.md) · [繁體中文](README.zh-TW.md)

**RAWviewer** is a fast photo viewer for **Windows and macOS**. Browse folders of RAW and JPEG files, check sharpness, cull rejects, rate keepers, and search your library — **on your computer, no cloud upload.**

Download: **[GitHub Releases](https://github.com/markyip/RAWviewer/releases/latest)** · Release notes: [`RELEASE_NOTES.md`](RELEASE_NOTES.md)

---

## Using RAWviewer

Open a folder (menu, drag-and-drop, or double-click a photo). Scroll the **gallery**; click a thumbnail for full-screen view.

On **large folders** (thousands of photos), the **Gallery** button appears once capture-time (EXIF) sorting finishes, so thumbnails are in shooting order as soon as you open gallery. If metadata is already cached, sort is instant.

In **gallery view**, drag the **size slider** in the bottom bar to change thumbnail size. Rows reflow in a justified grid (full width); scroll stays anchored to the upper-left visible photo while you drag.

| Key | Action |
|-----|--------|
| **Space** / **Double-click** | Toggle fit-to-window / 100% zoom |
| **Pinch** / **Ctrl+scroll** | Zoom in or out |
| **←** / **→** | Previous / next image |
| **Scroll wheel** | Previous / next (single view, fit mode) |
| **↑** | Bookmark / unbookmark |
| **1–5** / **0** | Set star rating / clear (single view; also clickable stars in the bottom bar) |
| **↓** | Move to Discard folder |
| **Delete** | Delete image(s) |
| **Esc** | Gallery: clear selection → exit bookmark/rating filter · Single view: back to gallery |
| **Ctrl/Cmd+click** | Gallery: toggle selection |
| **Shift+click** | Gallery: select range (visible order) |
| **C** | Toggle Compare mode on/off (requires multiple images selected) |
| **G** | Cycle composition guide |
| **H** | Show / hide histogram (hidden by default on launch) |
| **J** | Toggle highlight/shadow clipping overlay (RAW single view) |
| **P** | Toggle RAW recovery preview — half-res shadow/highlight recovery (RAW/DNG, session only; fit-only) |
| **F** | Show / hide focus overlay (supported files) |
| **M** | Show / hide GPS map overlay (single view, geotagged photos; hidden by default on launch) |
| **E** | Show / hide Adjust (Develop) panel (single view) |
| **Ctrl+Z** | Undo last slider adjustment (single view) |

**Compare mode shortcuts:**
* **← / →** — Previous / next candidate image
* **↑** — Promote candidate (right pane) to selected (left pane)
* **↓** — Reject candidate and move it to Discard folder (Shift+↓ to reject select)
* **Delete** — Delete candidate image to Recycle Bin/Trash (Shift+Delete to delete select)
* **Space** — Toggle synchronized zoom across both panes (with **F** focus overlays on: zoom each pane to its own focus point)
* **F** — Show / hide focus overlays on both panes (compare mode)
* **J** — Toggle exposure clipping overlay on both panes
* **G** — Cycle composition grid guide on both panes
* **C** / **Esc** — Exit Compare mode

**Gallery bookmarks & ratings:** Click the outline **bookmark star** (nothing selected) to show bookmarked shots only; gold = filter on. With photos selected, **↑** or the bookmark control toggles bookmarks. Use the gallery **rating filter** for stars ≥ N (combine with bookmarks). Ratings save to **XMP** sidecars.

**Search:** gallery search icon — `camera:sony`, `iso<800`, … (**Full** also accepts `sunset on beach`). **Share:** bottom **Share / Open** button, or drag gallery / film-strip thumbnails out.

Search syntax → [Advanced reference](#advanced-reference).

---

## Lite vs Full

Both editions share the same viewer, culling tools, star ratings, bookmarks, and metadata search. **Full** adds offline AI search and face filters.

| | Lite | Full |
|---|:--:|:--:|
| Gallery, film strip, zoom, histogram, bookmarks, ratings, culling | ✅ | ✅ |
| Metadata search (`camera:`, `iso:`, `date:`, …) | ✅ | ✅ |
| Plain-language search | — | ✅ |
| Face filters (`has:face`, …) | — | ✅ |

Pick **Lite** for a smaller install and browse-by-eye workflow. Pick **Full** to search with everyday words — still 100% offline.

### AI Denoise (Optional)

RAWviewer supports an optional neural denoise model (realPLKSR) for high-quality noise reduction during export. To enable AI denoise:
1. Download **`1xDeNoise_realplksr_otf.safetensors`** from [OpenModelDB](https://openmodeldb.info/models/1x-DeNoise-realplksr-otf). (Model by Philip Hofmann, CC-BY-4.0).
2. Place the file in `%LOCALAPPDATA%\RAWviewer\models\` on Windows, or `models/` inside the macOS app bundle (or specify the path via the `RAWVIEWER_NN_DENOISE_MODEL` environment variable).
3. The option "JPEG + AI denoise (realPLKSR)" will appear in the export format dropdown in the Adjust panel if your system has a compatible GPU.

---

## GPS map overlay & geotagging

In **single-image view**, press **M** to toggle an interactive tile-based map card. The card opens immediately with a **Loading map…** state while tiles fetch (no popup on photos without GPS). A **coordinate badge** on the map shows lat/lon; click it to open **Google Maps** in your browser.

Bundled offline databases (`cities500.csv.gz` and `landmarks.csv.gz`, 100,000+ locations) are used during **background indexing** to resolve GPS coordinates into city, region, and country names. These power **gallery search** — search `city:tokyo`, `country:jp`, or similar — with no internet required.

For a dedicated **cluster map** across an entire album and **geotagging photos missing GPS**, see **[LocateIt](https://github.com/markyip/LocateIt)**: open a folder, see where shots were taken on a map, drag-drop to assign coordinates, and save back to JPEG or RAW.

---

## Download & install

### Windows

1. Download **`RAWviewer_Setup.exe`** from [Releases](https://github.com/markyip/RAWviewer/releases/latest).
2. Choose **Full (CUDA)**, **Full (DirectML)**, or **Lite** in the wizard. **Full** also downloads AI models (~600 MB).
3. Launch **`RAWviewer.exe`** or the Desktop shortcut (not the Setup file again).

> **v3.0 new:** **Full Editing Functions** integrated (tone curve, lens correction, XMP sidecars); **Fast RAW decode** verified by multiple testing to deliver massive speed improvements vs 2.5; **1–5 star ratings** (keys **0–5**, gallery filter); Nikon **HE/HE*** NEF support; darkroom chrome. Full changelog: [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

Registers **Open with** for common photo formats. Uninstall: Settings → Apps, or **`uninstall.bat`** in `%LOCALAPPDATA%\RAWviewer`.

### macOS (13+)

1. Download **`RAWviewer-v3.0-macOS.zip`** (Full) or **`RAWviewer-v3.0-macOS-Lite.zip`** (Lite) from **[Releases](https://github.com/markyip/RAWviewer/releases/latest)** and extract the zip.
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

> [!NOTE]
> **EDR Support Removed:** macOS EDR (Extended Dynamic Range) support has been removed at this stage. The updated, highly optimized image loading pipeline is not compatible with EDR at this stage; attempting to combine them results in very slow image decoding and loading. To maintain high-speed browsing and editing performance, EDR support has been disabled/removed. All platforms now display images using standard dynamic range (SDR) with high-fidelity tone-mapping.

**Workflow toggle** (single view): switch between **Embedded JPEG (Fast)** and **RAW (High Quality)**.

**Recovery preview (**P**):** half-res shadow/highlight recovery for judging extreme contrast — session only, does not replace full-res view.

---

## Troubleshooting

### All platforms

| Problem | What to do |
|---------|------------|
| GPS map not showing | Press **M** in single-image view; the map only appears when the photo has embedded GPS coordinates |
| HDR HEIC/TIFF looks flat or too dark | **Windows:** HDR stills are tone-mapped to SDR by design. **macOS:** Needs an EDR-capable display (GPU single-image view is on by default); `RAWVIEWER_DISABLE_EDR=1` forces SDR tone mapping |
| **P** / **J** no effect | **P**/**J** are RAW/DNG single view only; **P** is fit-only half-res preview. **P** recovery also requires scipy + rawpy — check logs if it fails |

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
| App slow or exits after reopening last folder | On 8 GB PCs, use **Lite** or set `RAWVIEWER_DISABLE_SESSION_RESTORE=1` |
| RAW always shows demosaic, not embedded JPEG | Switch to **Embedded JPEG workflow**; RAW EDR re-decodes from LibRaw and overrides embedded preview when **RAW workflow** is on |
| Crash | Enable file logging with `RAWVIEWER_FILE_LOG=1`, then check the install folder |

### macOS

| Problem | What to do |
|---------|------------|
| macOS blocks the app (“damaged” / won’t open) | In the extracted folder, run `bash install_macos_app.sh` (see install steps above) |
| `bash: command not found` | Type `cd `, drag the extracted folder onto Terminal, press Return, then run the command again |
| Can’t read Desktop/Documents | System Settings → Privacy → **Full Disk Access** → add RAWviewer |
| Search says models missing (**Full**) | Open gallery search and click **Download** when prompted (needs internet once) |
| Download failed (SSL / certificate error) | On a corporate VPN or proxy, add your organization’s root certificate to **Keychain Access** and set it to **Always Trust** |
| Need to uninstall completely | Use **`uninstall_macos_app.sh`** or **`Uninstall RAWviewer.command`** from the release zip — not Trash alone |
| Uninstall scripts missing | Re-download the release zip from [Releases](https://github.com/markyip/RAWviewer/releases/latest); scripts are inside the extracted folder |
| macOS “out of memory” / heavy swap during indexing | See [Automatic memory tuning](#automatic-memory-tuning). On 8 GB Macs, prefer **Lite** or wait for indexing to finish before opening gallery on huge folders |
| Killed on relaunch (`Killed: 9` / exit 137 in Terminal) | Try **Lite**, `RAWVIEWER_DISABLE_SESSION_RESTORE=1`, or `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| Gallery still stutters on a huge folder | Run **`clear_cache.sh`** and reopen the folder |
| Gallery button slow on huge folder (first open) | Normal — waits for EXIF capture-time sort so gallery order is correct; instant when metadata is cached |
| RAW EDR but want embedded JPEG | Use **Embedded JPEG workflow** toggle, or `RAWVIEWER_RAW_EDR=0` |

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

**Indexing:** On **Full** builds, semantic search and face counts run in the background on large folders (metadata + AI first, faces after). The **search field stays read-only** until indexing completes for your profile (**Lite:** metadata; **Full:** metadata, embeddings, and face scan when enabled). When you **open a different folder**, indexing and prefetch from the previous folder are cancelled so work does not continue in the background for the old album (**v2.5.0**).

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
| `RAWVIEWER_DISABLE_EDR=1` | macOS: disable EDR viewport and HDR/RAW 16-bit display path; use SDR tone mapping |
| `RAWVIEWER_RAW_EDR=1` | **Default.** macOS: EDR for RAW when **RAW (High Quality)** workflow is selected; `0` to hard-disable. In-app: bottom-bar **EDR** button toggles it per-user, but resets to off every time you switch into RAW workflow. EDR decode is also idle-deferred: rapid navigation shows the fast SDR buffer immediately and only upgrades to EDR after you pause on an image, so browsing speed is unaffected either way |
| `RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1` | Same color pipeline for fit vs 100% zoom on RAW (default on) |
| `RAWVIEWER_FAST_RAW_DECODE=0` | Disable the fast RAW decode path (LibRaw unpack + SIMD demosaic with exact color parity, shared unpack between half/full tiers; default on, auto-falls-back to rawpy for unsupported sensors) |
| `RAWVIEWER_USE_PROCESS_POOL=0` | Disable LibRaw process pool (leave unset for normal use) |
| `RAWVIEWER_EXIF_BACKEND=auto` | `auto`, `pyexiv2`, or `exifread` |
| `RAWVIEWER_SHARE_MENU=1` | macOS: Qt share menu (recommended) |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | macOS: try native share sheet first |
| `RAWVIEWER_INDEX_DEFER_FACE_SCAN=1` | Defer face scan until after semantic index (default) |
| `RAWVIEWER_SEMANTIC_PREP_WORKERS` | Parallel CPU workers before AI encode (RAM tier may set this) |
| `RAWVIEWER_SEMANTIC_BATCH_AUTO=1` | Auto-tune AI batch/chunk size on index (default) |
| `RAWVIEWER_SEMANTIC_BATCH_CANDIDATES` | Candidate batch sizes for auto-tune (default `8,16,32,64,128`) |
| `RAWVIEWER_PREVIEW_CACHE_ITEMS` | Cap in-memory preview LRU count |
| `RAWVIEWER_FULL_IMAGE_CACHE_ITEMS` | Sensor-res buffer LRU slots (default 8, max 32; higher = instant zoomed A↔B revisits at ~100–200 MB per slot) |
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

Not in a release yet — tracked separately.

**Windows HDR / EDR** — v2.5+ ships macOS EDR for HDR stills and RAW (High Quality workflow). Windows still tone-maps HDR HEIC/TIFF and RAW to SDR. A future Windows path would use HDR-capable displays (10-bit / scRGB or HDR10 via Qt QRhi) for extended highlight headroom.

**Multithreaded LibRaw (macOS dev builds)** — The PyPI rawpy wheel bundles a single-threaded LibRaw on macOS/Linux (Windows wheels already ship OpenMP). `scripts/build_libraw_openmp.sh` rebuilds LibRaw with OpenMP and swaps it into the Pixi env — roughly 1.5–2× faster unpack on CR3/RAF/pana8. Local dev-env only; re-apply after `pixi install`. Verify with `scripts/check_libraw_parallelism.py <raw file>`.

**Future Development Plan:**
- **White balance preset support**: Add standard and custom WB presets.
- **LUT support**: Allow users to load and apply custom color lookup tables.
- **Masking for the editor**: Introduce local adjustments and selective masking.
- **VLM connection**: Integrate with Vision-Language Models for automatic, intelligent image adjustments.

**Shipped in 3.0:** Fast RAW decode verified by multiple testing for speed improvements vs 2.5, full Adjust / Develop editing panel integrated for all users, star ratings, burst grouping / Compare (**C**). GPU **viewport** (OpenGL zoom/pan) is on by default (`RAWVIEWER_GPU_VIEW=0` to disable).

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
| **Full / Unified** | `dist/RAWviewer_Setup.exe` (includes Full & Lite options) | `dist/RAWviewer-v3.0-macOS.zip` |
| **Lite** | (Select Lite option in `RAWviewer_Setup.exe`) | `dist/RAWviewer-v3.0-macOS-Lite.zip` |

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

- **ImageLoadManager** — threaded load queue; folder changes cancel in-flight tasks (**v2.5.0**)
- **UnifiedImageProcessor** — RAW/JPEG/TIFF via one path; **Fast RAW decode** shared unpack half/full (**v3.0.0**)
- **Star ratings** — 1–5 + XMP; gallery min-rating filter (**v3.0.0**)
- **Full Editing** — Adjust panel, XMP sidecars, PV2012-style develops (**v3.0.0**)
- **Cache** — memory-first; optional disk cache via env; **RAM-tier defaults** at startup (`rawviewer_profile.py`)
- **Semantic index** — SQLite + local embeddings (Core ML on macOS, ONNX on Windows; Full builds only); background passes abort when folder scope changes (**v2.5.0**)
- **Gallery (JustifiedGallery)** — justified grid with zoom slider (relayout + upper-left scroll anchor); layout cache keyed to folder generation; gallery opens in capture-time order after EXIF sort; tile aspect reconciles decoded thumbnails with container EXIF before justified-row geometry is locked (**v2.5.0**)
- **HDR / EDR (macOS)** — GPU viewport EDR layer + 16-bit HDR still decode; RAW EDR via linear LibRaw when RAW workflow is active (**v2.5.0**)
- **RAW recovery preview** — **P** key, half-res linear decode + local tone recovery (`raw_tone_recovery.py`; **v2.5.0**)
- **Clipping overlay** — **J** key on current pixmap (`exposure_clipping.py`; **v2.5.0**)

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
