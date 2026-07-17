# RAWviewer v3.0

<p align="center">
  <img src="icons/readme-icon.png" alt="RAWviewer Icon" width="256">
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

**RAWviewer** is a fast photo viewer for **Windows and macOS**. Browse folders of RAW and JPEG files, check sharpness, cull rejects, search your library, and (optionally) develop non-destructively to **XMP** — **on your computer, no cloud upload.**

Download: **[GitHub Releases](https://github.com/markyip/RAWviewer/releases/latest)**

> Full changelog: [`RELEASE_NOTES.md`](RELEASE_NOTES.md)
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
| **↑** | Bookmark / unbookmark (bottom **star** in single view too) |
| **↓** | Move to Discard folder |
| **Delete** | Delete image(s) |
| **Esc** | Gallery: clear selection → exit bookmark filter · Single view: back to gallery |
| **Ctrl/Cmd+click** | Gallery: toggle selection |
| **Shift+click** | Gallery: select range (visible order) |
| **C** | Toggle Compare mode on/off (requires multiple images selected) |
| **E** | Show / hide **Adjust** panel (XMP develop; experimental) |
| **0–5** | Star rating (**0** clears) |
| **G** | Cycle composition guide |
| **H** | Show / hide histogram |
| **J** | Toggle highlight/shadow clipping overlay |
| **P** | Toggle RAW recovery preview — half-res shadow/highlight recovery (RAW/DNG, session only; fit-only) |
| **F** | Show / hide focus overlay (supported files) |
| **M** | Show / hide GPS map overlay (single view, geotagged photos; hidden by default on launch) |

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

**Adjust (editor) shortcuts** — while the Adjust panel is open (**E**):

| Key | Action |
|-----|--------|
| **E** / **Esc** | Close Adjust (returns to browse RAW/JPEG mode) |
| **D** / **B** / **X** / **H** | Arm **Dodge** / **Burn** / **Eraser** / **Heal** (press again to disarm) |
| **O** | Toggle **Mask** overlay (when a brush tool is armed) |
| **Two-finger scroll** | Change **Brush Size** when a brush is armed (**Ctrl+scroll** still zooms) |
| **←** / **→** | Nudge the focused slider (or previous/next image if none focused) |
| **Ctrl/Cmd+Z** | Undo last edit step |
| **Space** / **Double-click** | Fit / 100% zoom |
| **J** / **G** / **F** | Clipping / composition guide / focus overlay (same as browse) |

Notes: **Effect Strength** applies only to Dodge/Burn; Heal uses **Size** and **Flow** at full inpaint strength. Browse-only keys (**M**, **P**, histogram **H**) do not apply while Adjust is open — **H** arms Heal instead.

**Gallery bookmarks:** Click the outline **star** (nothing selected) to show bookmarked shots only; gold star = filter on. With photos selected, **↑** or the star toggles bookmarks on the selection.

**Search:** gallery search icon — type plain metadata (`tokyo`, `sony`, `2024`) or filters like `camera:sony` / `iso<800` (**Full** also accepts `sunset on beach`). **Share:** bottom **Share / Open** button, or drag gallery / film-strip thumbnails out.

Search syntax → [Advanced reference](#advanced-reference).

---

## Lite vs Full

Both editions share the same **viewer, culling, bookmarks, ratings, Compare, GPS map, metadata search, and Adjust / Develop panel** (CPU Fast RAW + XMP edits). **Full** adds offline AI search, face filters, and optional GPU/ML tooling. **Lite** is the smaller install for browse-first machines.

| Capability | Lite | Full |
|---|:--:|:--:|
| Gallery, film strip, zoom, histogram, bookmarks, culling, Compare (**C**) | ✅ | ✅ |
| Star ratings (**0–5**) + XMP | ✅ | ✅ |
| Metadata / place search (plain text or `camera:` / `iso:` / `city:` …) | ✅ | ✅ |
| **Adjust** panel (**E**) — tone, WB, crop, D&B, Heal, LUT, vignette/dehaze, XMP | ✅ | ✅ |
| Export (JPEG / WebP / TIFF16) from Adjust | ✅ | ✅ |
| Plain-language AI search (`sunset on beach`) | — | ✅ |
| Face filters (`has:face`, `no:face`) | — | ✅ |
| MobileCLIP / ONNX / Core ML model download | — | ✅ |
| PyTorch / kornia in the package | — | ✅ |
| GPU demosaic (`RAWVIEWER_PREFER_GPU_DECODE`) | — (CPU Fast RAW) | ✅ when backend present |
| AI denoise export path | — | ✅ (Full) |
| Typical install size | ~500 MB | ~1.5 GB+ (models) |
| Recommended RAM | 8 GB+ | 16 GB+ for large folders + AI index |

**Pick Lite** for a lean install and cull-by-eye workflow (Adjust still works). **Pick Full** when you want everyday-language search and face filters — still **100% offline** after models are installed.

Windows installer: choose **Full (CUDA)**, **Full (DirectML)**, or **Lite**. macOS: separate **Full** / **Lite** zips.

---

## GPS map overlay & geotagging

In **single-image view**, press **M** to toggle an interactive tile-based map card. The card opens immediately with a **Loading map…** state while tiles fetch (no popup on photos without GPS). A **coordinate badge** on the map shows lat/lon; click it to open **Google Maps** in your browser.

Bundled offline databases (`cities500.csv.gz` and `landmarks.csv.gz`, 100,000+ locations) are used during **background indexing** to resolve GPS coordinates into city, region, and country names. These power **gallery search** — type a place name like `tokyo` or `Taipei`, or use filters such as `city:tokyo` / `country:jp` — with no internet required.

For a dedicated **cluster map** across an entire album and **geotagging photos missing GPS**, see **[LocateIt](https://github.com/markyip/LocateIt)**: open a folder, see where shots were taken on a map, drag-drop to assign coordinates, and save back to JPEG or RAW.

---

## Download & install

### Windows

1. Download **`RAWviewer_Setup.exe`** from [Releases](https://github.com/markyip/RAWviewer/releases/latest).
2. Choose **Full (CUDA)**, **Full (DirectML)**, or **Lite** in the wizard. **Full** also downloads AI models (~600 MB).
3. Launch **`RAWviewer.exe`** or the Desktop shortcut (not the Setup file again).

> **v3.0 new:** **Adjust / Develop** (tone, WB presets, crop, Dodge/Burn + Heal, vignette/dehaze, Creative LUT / XMP presets); **Fast RAW decode** vs 2.5; **1–5 star ratings**; Nikon **HE/HE*** browse; **Lite without torch**; gallery loading overhaul; OpenMP LibRaw on macOS packages. Full changelog: [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

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

On **macOS** and **Windows**, HDR **HEIC / HEIF / AVIF** and HDR **TIFF** are tone-mapped to SDR for browse speed. **v3.0 removed macOS EDR** so the Fast RAW load path stays fast (see [`RELEASE_NOTES.md`](RELEASE_NOTES.md)).

**Nikon HE / HE\* NEF:** LibRaw cannot demosaic High Efficiency compression today. RAWviewer still opens these files via the **embedded JPEG** (browse / cull works; no spurious “unsupported or corrupt” dialog). **Adjust / RAW develop is disabled** for HE/HE\* until a decoder exists — same limitation as JPEG/HEIC (browse-only). Lossless / standard NEF demosaic and edit as usual.

---

## Single-view tools

**Workflow toggle:** switch between **Embedded JPEG (Fast)** and **RAW (High Quality)**.

**Recovery preview (**P**):** half-res shadow/highlight recovery for judging extreme contrast — session only, does not replace full-res view.

**Adjust (**E**):** non-destructive develops to XMP. By default, **browse surfaces show original pixels**; edits render inside the Adjust panel (`RAWVIEWER_SIDECAR_ADJUST=1` to apply saved edits in browse). Editing is experimental — camera coverage is not guaranteed for every new body. See **Adjust (editor) shortcuts** above.

---

## Troubleshooting

### All platforms

| Problem | What to do |
|---------|------------|
| GPS map not showing | Press **M** in single-image view; the map only appears when the photo has embedded GPS coordinates |
| HDR HEIC/TIFF looks flat or too dark | HDR stills are tone-mapped to SDR by design in v3.0 (macOS EDR removed for Fast RAW) |
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
| Out of memory during AI indexing | See [Automatic memory tuning](#automatic-memory-tuning) under [For developers](#for-developers); use **Lite** on 8 GB PCs or set `RAWVIEWER_MEMORY_TIER_AUTO=0` and lower workers manually |
| App slow or exits after reopening last folder | On 8 GB PCs, use **Lite** or set `RAWVIEWER_DISABLE_SESSION_RESTORE=1` |
| RAW always shows demosaic, not embedded JPEG | Switch to **Embedded JPEG workflow** |
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
| macOS “out of memory” / heavy swap during indexing | See [Automatic memory tuning](#automatic-memory-tuning) under [For developers](#for-developers). On 8 GB Macs, prefer **Lite** or wait for indexing to finish before opening gallery on huge folders |
| Killed on relaunch (`Killed: 9` / exit 137 in Terminal) | Try **Lite**, `RAWVIEWER_DISABLE_SESSION_RESTORE=1`, or `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| Gallery still stutters on a huge folder | Run **`clear_cache.sh`** and reopen the folder |
| Gallery button slow on huge folder (first open) | Normal — waits for EXIF capture-time sort so gallery order is correct; instant when metadata is cached |

More detail: [`scripts/Launch/README.md`](scripts/Launch/README.md)

---

## Advanced reference

*Optional — for search power users and troubleshooting.*

> **Thumbnail cache notice:** To speed up gallery loading, RAWviewer creates a local thumbnail cache on your device. This cache is **never uploaded or shared** — it stays entirely on your machine. Cache files are automatically deleted after **30 days** of inactivity.

### Gallery search syntax

Separate words with spaces. **Prefixes are optional for most metadata** — if a token appears in indexed fields (place, camera, lens, filename, or a date-like `2024` / `2024-05`), it filters without `key:`. Use `key:value` when you want to force a field, run comparisons, or combine with other constraints.

| Kind | Example |
|------|---------|
| Place | `tokyo` · `Taipei` · `hong kong` · `city:tokyo` · `country:jp` |
| Camera / lens | `sony` · `canon` · `70-200` · `camera:canon` · `lens:70-200` |
| File name | `_dsc` · `IMG_1234` · `filename:_dsc` |
| Date | `2024` · `2024-05` · `date:2024-05` |
| ISO / year (comparison) | `iso<=800` · `iso under 800` · `year>=2024` |
| Format | `format:raw` · `format:jpeg` · `format:cr3` |
| GPS / faces | `has:gps` · `has:face` · `no:face` *(face filters: Full only)* |
| Free text + filter | `jet takeoff camera:sony iso<800` *(Full: unmatched free text uses AI)* |

**Face vs semantic search:** `face`, `people`, `person`, etc. use stored face counts (`has:face`), not the neural search.

**Indexing:** On **Full** builds, semantic search and face counts run in the background on large folders (metadata + AI first, faces after). The **search field stays read-only** until indexing completes for your profile (**Lite:** metadata; **Full:** metadata, embeddings, and face scan when enabled). When you **open a different folder**, indexing and prefetch from the previous folder are cancelled so work does not continue in the background for the old album (**v2.5.0**).

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

### macOS version support

| Your Mac | Official `.zip` | Build from source |
|----------|-----------------|-------------------|
| macOS 13 Ventura (Intel) | ✅ | `build_macos_full.sh` or Pixi |
| macOS 13 Ventura (Apple Silicon) | ✅ | Use **`build_macos_full.sh`** (Pixi needs 14+) |
| macOS 14 Sonoma+ | ✅ | Pixi or `build_macos.sh` |
| macOS 12 Monterey or older | ❌ | ❌ |

### Upcoming / remaining work

Tracked in [`RELEASE_NOTES.md`](RELEASE_NOTES.md) (v3.0 Known Issues & Remaining Work).

**Windows HDR / restore Mac EDR safely** — v2.5 shipped macOS EDR; **3.0 removed it** so Fast RAW stays fast. A future path would use HDR-capable displays without regressing browse speed.

**Remaining work (feasibility high → low):**
1. **Cold-folder edited tile regen** — save-from-Adjust already bakes editor-aligned thumbs; optional full `SIDECAR_ADJUST` for never-opened edits.
2. **Broader local masks** — gradients / clone stamp (Dodge/Burn + Heal + crop already ship).
3. **DNG export** — real writer.
4. **ML subject masks** — Full only.
5. **Windows HDR / restore Mac EDR safely**.
6. **VLM-assisted adjust** — large product/model scope.
7. **HE-NEF RAW edit** — blocked until a decoder exists (browse-only today).

**Shipped in 3.0:** Fast RAW vs 2.5, Adjust panel (HSL, Creative LUT, WB presets, crop, D&B + Heal, vignette/dehaze, XMP presets), OpenMP LibRaw in macOS packages, star ratings, burst / Compare (**C**), Lite without torch, gallery loading overhaul. GPU **viewport** on by default (`RAWVIEWER_GPU_VIEW=0` to disable).

---

## For developers

Scripts and build matrix: [`scripts/Launch/README.md`](scripts/Launch/README.md)

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
| `RAWVIEWER_FAST_RAW_DECODE=1` | **Default.** Shared unpack for half/full Fast RAW; `0` falls back toward rawpy |
| `RAWVIEWER_SIDECAR_ADJUST=0` | **Default.** Browse shows original pixels; edits render in Adjust. Set `1` to apply saved XMP in browse |
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
- **Adjust / Develop** — tone, WB, crop, Dodge/Burn, Heal (`cv2.inpaint`), vignette/dehaze, LUT/XMP presets; edits in Adjust by default (**v3.0.0**)
- **Cache** — memory-first; optional disk cache via env; **RAM-tier defaults** at startup (`rawviewer_profile.py`)
- **Semantic index** — SQLite + local embeddings (Core ML on macOS, ONNX on Windows; **Full only**); background passes abort when folder scope changes (**v2.5.0**)
- **Gallery (JustifiedGallery)** — justified grid with zoom slider; capture-time order after EXIF sort (**v2.5.0**+ gallery fill overhaul in **v3.0.0**)
- **RAW recovery preview** — **P** key, half-res linear decode + local tone recovery (**v2.5.0**)
- **Clipping overlay** — **J** key (`exposure_clipping.py`; **v2.5.0**)
- **Lite profile** — no torch/kornia; semantic/face off; CPU Fast RAW + Adjust (**v3.0.0**)

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
