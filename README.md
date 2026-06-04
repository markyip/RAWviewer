# RAWviewer v2.2

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

![Version](https://img.shields.io/badge/version-2.2-blue)
![Downloads](https://img.shields.io/github/downloads/markyip/RAWviewer/total) 
![License](https://img.shields.io/badge/license-MIT-green)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/markyip)

## ✈️ Why This Exists
You're a photographer who just returned from an aviation show, a wildlife safari, or a fast-paced sports event. You took thousands of RAW shots — and now you're facing the real challenge:

- **Sifting through mountains of photos**: Importing everything into heavy catalog tools like **Lightroom** or **Capture One** just to see what's good is slow and frustrating.
- **Checking technical details**: Zooming in to check sharpness or finding where the camera's autofocus locked (`F` key focus overlays) is usually a multi-step chore.
- **Finding specific shots instantly**: Searching your catalog for specific descriptions (e.g., `"takeoff"`, `"sunset"`, `"crowd"`) typically requires manual tagging or cloud-based AI uploads that compromise privacy.
- **Filtering out the noise**: The default system photo viewers are clunky with RAW formats and offer no easy ways to flag or move blurry images in bulk.

I've been there myself — struggling to sort through **500GB of RIAT aviation photos** because of how tedious this process is. That frustration is exactly what inspired me to build **RAWviewer**.

RAWviewer is a lightweight, high-performance local organizer built specifically for photographers who shoot high volumes. It acts as an **AI-powered pre-filtering and discovery workspace**, letting you curate, search, and prepare your images entirely offline **before** you commit them to your editing workflow in Lightroom or Photoshop.

## 🔍 What is RAWviewer?
**RAWviewer** is a fast, AI-powered image viewer and organizer for **Windows and macOS** (official releases), built with PyQt6. 

Beyond instant RAW previewing, 100% single-key zooming, and smooth panning, RAWviewer integrates:
- **Local Semantic AI Search**: Search your photo folder using natural language (e.g., `"airplane in clouds"`, `"portrait of a dog"`) running **100% offline and locally** via MobileCLIP (Core ML on macOS, ONNX on Windows).
- **Structured Metadata Filtering**: Instantly narrow down results using advanced filters like `camera:`, `lens:`, `iso:`, `ext:`, `has:face`, `city:`, or date ranges.
- **Focus Point & Subject Outlines**: Visualize camera autofocus points directly from MakerNote metadata to verify exact focal lock.
- **Non-Destructive Tools & Sharing**: Quick visual rotations, automatic slideshows, and system sharing to jump straight into editing applications.

> **Platform note:** Official prebuilt releases are **Windows** and **macOS** only. There is no official Linux package; running from source on Linux is unsupported and may require manual dependency setup.

## ✨ Features

- **Built for speed**: Open large folders quickly and start culling right away.
- **Easy sharpness check**: Press `Space` for instant 100% zoom, then browse with arrow keys.
- **Smooth gallery browsing**: Scroll a large grid view without waiting for full imports.
- **Search in gallery**: Filter by words and metadata like camera, ISO, date, GPS, and format.
- **Single-view search jump**: Start search from one image and jump directly into filtered gallery results.
- **Film strip navigation**: A bottom thumbnail strip appears when needed for fast jumping.
- **Broad format support**: Works with common RAW types (CR2/CR3, NEF, ARW, DNG, RAF, ORF, RW2, and more) plus JPEG/TIFF/HEIF.
- **Reliable orientation**: Photos display upright consistently across many camera brands.
- **Clean fit and zoom colors**: RAW preview and 100% zoom stay visually consistent.
- **Helpful photo info**: See key shooting details like focal length, aperture, ISO, and capture time.
- **Safe file actions**: Move rejects to Discard or delete with confirmation.
- **Session memory**: Reopen the app and continue from your last folder and image.
- **Modern look and controls**: Keyboard, mouse, trackpad pinch, and polished UI.
- **Windows and macOS releases**: Official prebuilt apps for both platforms.

## 🚀 Quick Start

### Download Executable

Official releases are published for **Windows** and **macOS** only.

#### Windows
1. Download the appropriate version for your system from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest):
   - **`RAWviewer-CUDA.exe`**: Recommended for **NVIDIA GPU** users who have CUDA installed. This provides the fastest indexing and search performance.
   - **`RAWviewer-DirectML.exe`**: Recommended for **AMD, Intel, or NVIDIA** users who want an out-of-the-box hardware-accelerated experience without installing CUDA.
2. Run the downloaded installer directly (no zip extraction needed).
3. Double-click the installer to initiate the installation process. It will automatically set up the dependencies (via Pixi) and download the local AI models to a destination of your choice.
4. Launch RAWviewer from the Desktop shortcut! (You can safely delete the installer afterwards).

#### macOS
> **Minimum supported macOS (official prebuilt release): macOS 13 Ventura or newer.**

1. Download the latest release from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest)
2. Download and extract `RAWviewer-v2.2-macOS.zip`
3. Drag `RAWviewer.app` to your **Applications** folder.
4. **CRITICAL FIRST STEP:** Because this is an open-source app not signed via the paid Apple Developer program, macOS Gatekeeper will incorrectly label it as "Damaged" or block it. **You must run this command in your Terminal once** to remove the download quarantine flag:
   ```bash
   xattr -cr /Applications/RAWviewer.app
   ```
5. You can now double-click to launch it normally from Applications or Launchpad!

## ⌨️ Keyboard Shortcuts

- **Space**: Toggle between fit-to-window and 100% zoom
- **`Esc`**: Return to Gallery View (from Single View)
- **`←`/`→` arrows**: Navigate between images
- **`↓`**: Move current image to Discard folder
- **Delete**: Delete current image (with confirmation)
- **`H`**: Show or hide the single-image histogram strip
- **`F`**: Focus / subject outline

## 🔎 Gallery search (Gallery view)

Open the bottom search panel. The search field placeholder is **Search gallery**.

- On **macOS** with bundled Core ML models, free-text queries run **semantic ranking** over the images that pass any filters below.
- **Important:** Words like **`face`**, **`faces`**, **`people`**, **`person`**, and **`human`** are **not** sent to the neural search: they filter by the **Vision face-detection count** stored at index time (same as `has:face`). If no faces were detected (distant subjects, backs to camera, silhouettes), those photos are excluded—try free-text phrases instead, e.g. `crowd`, `pedestrians`, `spectators`.
- **Formats:** Prefer **`format:jpeg`** · **`format:raw`** (`type jpeg` / `ext raw` with a space also normalize). Loose phrases **`file jpeg`** / **`file raw`** map to **`format:`** so **`.jpg`** matches **JPEG** synonyms and **`raw`** covers typical camera RAW extensions (not only filenames containing the substring `raw`).
- You can combine a description with structured filters on one line (see examples).
- **Clear** the field or use the **×** control to restore the full folder.

### Semantic + face indexing behavior (current default)

- Semantic indexing and face detection are both enabled by default.
- To keep the app responsive on very large RAW folders, indexing runs in two passes:
  1. metadata + semantic embeddings (search-ready first)
  2. face-count backfill in background
- Background face pass starts automatically after semantic indexing is ready and resumes from persisted DB state.
- Thumbnail warm-up before face scan is conservative by default to avoid long "warming" stalls on multi-thousand-image folders.
- Advanced environment switches:
  - `RAWVIEWER_INDEX_DEFER_FACE_SCAN=1` (default): run face scan after semantic pass
  - `RAWVIEWER_FACE_SCAN_WARM_THUMBS=0` (default): disable full warm-up prepass
  - `RAWVIEWER_FACE_SCAN_WARM_MAX_FILES=256` (default): cap warm-up batch size
  - `RAWVIEWER_FACE_SCAN_WARM_MAX_SECONDS=25` (default): cap warm-up wall time

### Gallery search syntax examples

Separate tokens with spaces. Filters use `key:value` or comparison forms.

| Kind | Example |
|------|---------|
| Free text + filter | `jet takeoff camera:sony iso<800` |
| Camera / lens | `camera:canon` · `lens:70-200` |
| ISO / year | `iso<=800` · `year>=2024` |
| Place | `city:tokyo` · `country:jp` · `admin:california` |
| File name | `filename:_dsc` or `name:img_` |
| File format | `format:cr3` · `type:jpeg` · `ext:jpg,png` · `format:raw` (same set as [`src/raw_file_extensions.py`](src/raw_file_extensions.py)) |
| Date prefix | `date:2024-05` |
| GPS / faces | `has:gps` · `no:gps` · `has:face` · `people` · `person` · `no:face` |

Optional: **stronger semantic models** (advanced). The app default is **MobileCLIP2-S0** (vision: ~43MB) for speed and size. You can choose a stronger model by setting the environment variable **`RAWVIEWER_MOBILECLIP_VARIANT`** to one of the following:

- **`s0`** (Default): Vision model ~43MB, fastest indexing, lower memory usage.
- **`s2`**: Vision model ~136MB, better accuracy on complex search terms.
- **`b`**: ViT-B model ~330MB, significantly improved search understanding.
- **`l14`**: ViT-L/14 model ~1.1GB, highest quality, recommended for heavy/large search applications.

When a different variant is selected, the application will automatically download the correct model assets from Hugging Face (`plhery/mobileclip2-onnx` on Windows/Linux) to a separate cache folder.

*Note for macOS users:* Bundled macOS Core ML models are discovered automatically in common app/resource paths:
- Apple Hub naming: `mobileclip_s2_image.mlpackage` + `mobileclip_s2_text.mlpackage`
- App export naming (`--for-app`): `mobileclip2_s0_image.mlpackage` + `mobileclip2_s0_text.mlpackage`

## 🖱️ Mouse Controls

- **Double-click**: Zoom in to the clicked point (from fit), or zoom out to fit
- **Pinch (Mac/Windows Trackpad) or Ctrl+Scroll**: Smoothly zoom in/out with smart cursor anchoring
- **Click and drag**: Pan image when zoomed in
- **Drag and drop**: Open images or folders
- **Scroll Wheel (fit-to-window)**: Navigate images - Scroll down = previous, Scroll up = next
- **Scroll Wheel (zoom mode)**: Pan image vertically
- **Horizontal Wheel (zoom mode)**: Pan image horizontally (left/right)
- **Scroll Wheel (Gallery View)**: Scroll through the image grid
- **Film strip (Single View)**: Move the pointer toward the bottom of the image area to reveal a fading thumbnail strip; move away to dismiss. The strip includes thumbnails from the current folder or active search filter.

When focus/subject indicator is enabled (`F`):

- **Space** from fit-to-window zooms to the focus/subject box center.
- **Double-click** still zooms to your clicked point (same as normal mode).

## 📁 Supported Formats

### RAW Formats
- **Canon**: CR2, CR3
- **Nikon**: NEF
- **Sony**: ARW
- **Adobe**: DNG
- **Olympus**: ORF
- **Panasonic**: RW2
- **Fujifilm**: RAF
- **Hasselblad**: 3FR
- **Pentax**: PEF
- **Samsung**: SRW
- **Sigma**: X3F
- **And many more supported by LibRaw**

### Standard Formats
- **JPEG**: JPG, JPEG
- **TIFF**: TIF, TIFF
- **HEIF**: HEIF

## 🛠️ Development (run from source)

Launch scripts live under [`scripts/Launch/`](scripts/Launch/README.md).

| Platform | Debug run | Build |
|----------|-----------|-------|
| Windows | `scripts\Launch\bat\run_debug.bat` | `scripts\Launch\bat\build_windows.bat` |
| Windows (cache wipe) | `clear_cache.bat` or `scripts\Launch\bat\clear_cache.bat` | — |
| macOS | `./scripts/Launch/shell/launch_dev.sh` | `./scripts/Launch/shell/build_macos.sh` |

**Virtual environments:** `pixi install` → `pixi run start` uses `.pixi/`. Build/debug batch scripts use `rawviewer_env/` (created automatically). `.venv/` is optional for IDE use only.

**Folder sort (capture time):** Gallery and folder load sort by **EXIF** (`metadata_backend` probe when cold; bulk cache when warm). Default order is **oldest first**; use the gallery **⇅ Oldest / Newest** control to toggle (saved in QSettings). Windows Explorer `DateTaken` via Shell was evaluated and rejected for production (slower than EXIF, no benefit on test folders).

**Optional dev toggles:**

| Variable | Effect |
|----------|--------|
| `RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1` | **Default.** Single-image RAW uses LibRaw half-res for fit view and full decode at 100% zoom — same color pipeline, no embedded-JPEG color snap. Set `=0` for faster embedded-preview first paint. |
| `RAWVIEWER_PROGRESSIVE_RAW_LOAD=1` | Show embedded preview first, then upgrade to LibRaw in background (may color-shift). Off by default. |
| `RAWVIEWER_USE_PROCESS_POOL=1` | LibRaw postprocess in worker processes (multi-core). Default on when CPU count ≥ 4. |
| `RAWVIEWER_GPU_VIEW=1` | Use the experimental GPU-accelerated single-image viewport (smoother zoom/pan; default remains the classic scroll area). Space / double-click toggle fit ↔ 100% pixel zoom; arrow keys browse in single view with prefetch. |
| `RAWVIEWER_GPU_VIEW_NO_GL=1` | Force raster viewport when GPU view is enabled (debug / fallback) |
| `RAWVIEWER_NAV_PRELOAD_DISPLAY=1` | Prefetch display-quality buffers for nearby files while navigating in single view (enabled in `run_debug.bat`) |
| `RAWVIEWER_RESOLUTION_CROSSFADE_MS` | Duration (ms) of preview→full crossfade in single view when GPU/legacy crossfade is enabled (default `280`) |
| `RAWVIEWER_DISABLE_CROSSFADE=1` | Disable viewport crossfade on resolution upgrades |
| `RAWVIEWER_PERSISTENT_CACHE=1` | Enable disk/SQLite cache persistence (off by default) |
| `RAWVIEWER_EXIF_BACKEND=auto` | EXIF via pyexiv2 (JPEG/TIFF) + exifread (RAW headers); `exifread` or `pyexiv2` to force one backend |
| `RAWVIEWER_SORT_PROBE_WORKERS` | Parallel EXIF header probes during folder sort (default scales with CPU, up to 12 on local disk; 3 on UNC / `RAWVIEWER_SLOW_STORAGE_PREFIXES`) |
| `RAWVIEWER_INDEX_METADATA_WORKERS` | Semantic index metadata extraction pool (default 2–6; lower on folders &gt;2000 files) |
| `RAWVIEWER_RAW_LOAD_LIMIT` | Max concurrent LibRaw decodes in the load manager (default `4`) |
| `RAWVIEWER_PROCESS_POOL_WORKERS` | LibRaw postprocess process pool size when `RAWVIEWER_USE_PROCESS_POOL=1` |
| `RAWVIEWER_SLOW_STORAGE_PREFIXES` | Comma-separated path prefixes (e.g. `K:\Photos,N:\`) to cap sort-probe parallelism at 3 |

**macOS share (v2.2, single-image view only):**

| Variable | Default in `launch_dev.sh` | Effect |
|----------|------------------------------|--------|
| `RAWVIEWER_SHARE_MENU` | `1` | Qt menu listing `NSSharingService` targets (recommended under Qt6) |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER` | off | Try `NSSharingServicePicker` first, then menu fallback |
| `RAWVIEWER_SHARE_SHOW_AIRDROP` | off | Include AirDrop in the menu (in-app AirDrop is unreliable in the Qt host) |
| `RAWVIEWER_SHARE_DEBUG` | off | Share diagnostics in status bar and `[SHARE]` logs |

Details: [`docs/macos-sharing-v21-v22.md`](docs/macos-sharing-v21-v22.md) and [`scripts/Launch/README.md`](scripts/Launch/README.md#macos--release-smoke-test-manual).

### macOS — build and smoke test (v2.2)

```bash
chmod +x scripts/Launch/shell/*.sh
./scripts/Launch/shell/build_macos.sh    # or: pixi install && pixi run python build.py
xattr -cr dist/RAWviewer.app
open dist/RAWviewer.app
```

**Dev run** (preflight checks for `pyexiv2` and semantic backend):

```bash
./scripts/Launch/shell/launch_dev.sh
# Skip preflight: RAWVIEWER_TEST_PYEXIV2=0 RAWVIEWER_TEST_SEMANTIC=0 ./scripts/Launch/shell/launch_dev.sh
```

Before tagging a macOS release: confirm app version **2.2**, single-view **share** menu works (Mail attach), and bundled `models/mobileclip2_coreml` if semantic search is required in the `.app`.

## 🏗️ Building from Source

### Prerequisites
- [Pixi](https://pixi.sh/latest/) (Package manager for development and dependencies)

### Windows
**Option 1: Using batch script (recommended)**
```batch
# Run the automated build script (manages its own venv)
scripts\Launch\bat\build_windows.bat
```

**Option 2: Manual build with Pixi**
```bash
# Install dependencies
pixi install

# Run application
pixi run start

# Build executable
pixi run python build.py
```

### macOS
**Option 1: Using shell script (recommended)**
```bash
# Run the automated build script (manages its own venv)
./scripts/Launch/shell/build_macos.sh
```

**Option 2: Manual build with Pixi**
```bash
# Install dependencies
pixi install

# Run application
pixi run start

# Build executable
pixi run python build.py
```

### Dependencies
All project dependencies are managed via `pixi.toml` instead of `requirements.txt`. Launch and build scripts live under [`scripts/Launch/`](scripts/Launch/README.md) (`bat/` on Windows, `shell/` on macOS). They install required pip packages into a local `rawviewer_env` virtual environment when building.

## 🐛 Troubleshooting

### Windows
- **"Windows protected your PC"**: Click "More info" → "Run anyway"
- **Antivirus warnings**: Add RAWviewer to your antivirus exclusions
- **Performance issues**: Try running as administrator
- **"Open with another app" / bottom share button**: v2.2 implements the native Open with APIs (`OpenAs_RunDLLW`, `SHOpenWithDialog` + `OAIF_EXEC`), but the bottom-bar control is **hidden on Windows** in current `main`. Use Explorer **Open with** on the file until the in-app button is re-enabled (see `scripts/Launch/README.md`).
- **AttributeError with stdout**: This is normal for windowed builds - the application runs without a console window
- **Installer stuck on "Downloading MobileCLIP ONNX Models" / `No module named 'requests'`**:
  - Fixed in 2.1.0+ (`requests` in `pixi.toml`). Re-run the installer from a fresh build, or in the install folder run `_internal\pixi\pixi.exe install` then retry.
  - Public Hugging Face models download **without** an account or token.

- **Crash code `-1073741819` / `0xC0000005` (access violation)**:
  - This is a native crash (Qt/LibRaw/ONNX/driver layer), not always a Python exception.
  - Check `%LOCALAPPDATA%\RAWviewer\logs\` (persistent crash reports / fatal dumps). Dev runs via `scripts\Launch\bat\run_debug.bat` may also write under `src\logs\`.

### macOS
- **Minimum supported macOS (official prebuilt app): 13.0 (Ventura)**
  - macOS 12 and older may fail to launch the prebuilt binary; use a local Pixi build instead.

- **"App is damaged and should be moved to the Trash" / "Apple could not verify RAWviewer is free of malware"**: 
  - **Why it happens**: Apple heavily restricts apps downloaded outside the App Store that aren't signed with a paid developer certificate. On newer macOS versions (especially Apple Silicon M1/M2/M3), macOS breaks the app's ad-hoc signature and aggressively blocks opening it.
  - **The Fix (Fastest)**: Open your **Terminal** app and run the following command to remove the quarantine flag:
    ```bash
    xattr -cr /Applications/RAWviewer.app
    ```
    *(Note: If you placed the app somewhere other than the Applications folder, update the path accordingly).*

- **"Symbol not found: (_mkfifoat)" or App crashes instantly on macOS 12 (Monterey) or older**:
  - **Why it happens**: The official prebuilt release targets macOS 13+ and newer system SDK/runtime symbols.
  - **The Fix**: Build locally with Pixi (see below).

#### 🛠️ The Ultimate Fix: Build Locally (Solves Both Issues Above)
If you are on macOS 12 or older, OR if you simply want to permanently bypass all Gatekeeper/Quarantine warnings forever, you can build the app directly on your own machine. It takes about 2 minutes and is managed entirely by Pixi, which automatically downloads the correct Python version for you:
1. **Install Pixi** (Open Terminal and run: `curl -fsSL https://pixi.sh/install.sh | bash`).
2. **Open Terminal** and run these commands to download and build:
   ```bash
   git clone https://github.com/markyip/RAWviewer.git
   cd RAWviewer
   pixi run python build.py
   ```
This will automatically create a perfectly compatible, warning-free `RAWviewer.app` inside the `dist/` folder!

#### 🔧 Local Build Troubleshooting
- **Error: "No matching distribution found for pyexiv2" or Massive C++ compilation failures**
  - **Why it happens**: Usually caused by using an unsupported Python version (too old or too bleeding-edge).
  - **The Fix**: Using `pixi` automatically resolves this by pinning a stable, supported Python version (e.g. 3.11) that has pre-compiled wheels for all libraries. Delete your old `rawviewer_env` and use the `pixi` build instructions above instead of the manual shell script.

- **Homebrew delays on macOS 12 Monterey or older**: 
  - Homebrew has officially dropped "binary bottle" support for Monterey. However, **it still works**. When the build script attempts to `brew install inih gettext`, Homebrew will simply compile them from source on your machine. This is completely normal but may take 2-3 extra minutes.

- **Share menu empty or native picker spins**: Use dev defaults (`RAWVIEWER_SHARE_MENU=1` via `launch_dev.sh`). Avoid opening the picker on mouse-up; see [`docs/macos-sharing-v21-v22.md`](docs/macos-sharing-v21-v22.md). For AirDrop, prefer Finder on the file.

- **Permission Denied / Cannot Read Folder**: Modern macOS requires explicit permission for apps to access the Desktop or Documents. 
  1. Go to **System Settings** > **Privacy & Security** > **Full Disk Access**.
  2. Click the **+** button and add `RAWviewer.app`.
  3. Toggle it to **ON**.

- **"Semantic search unavailable" / asks to download models even in packaged app**:
  1. Open `RAWviewer.app/Contents/Resources/models/mobileclip2_coreml/`.
  2. Confirm either **S2** pair (`mobileclip_s2_*`) or **S0 app-export** pair (`mobileclip2_s0_*`) exists, plus `bpe_simple_vocab_16e6.txt.gz`.
  3. If missing, rebuild with `models/mobileclip2_coreml/` present before running `python build.py`.

## 🚧 Upcoming Features

We're continuously working to improve RAWviewer. Here are some features planned for future releases:

- **Batch Operations**: Select and process multiple images at once


## ⚠️ Known Issues

### Platform (v2.2)
- **Windows:** In-app **Open with** picker is implemented but the bottom-bar button is not shown on `win32` in current `main`.
- **macOS:** `NSSharingServicePicker` popover often fails under the Qt6 host; default product path is the **Qt share menu**, not the popover.

### Camera Compatibility
- **Newer camera models**: Support for the latest camera releases may be limited due to LibRaw library compatibility
- **Proprietary RAW formats**: Some manufacturers' newest RAW formats may not be fully supported immediately after camera release
- **Firmware updates**: Camera firmware updates may introduce RAW format changes that require LibRaw updates

## 🏛️ Architecture

RAWviewer uses a modern, optimized architecture:

- **ImageLoadManager**: Manages all image loading tasks using a thread pool and priority queue
- **UnifiedImageProcessor**: Handles all image types (RAW, JPEG, TIFF, etc.) with a unified interface
- **Cache System**: Memory-first cache by default, with optional persistent disk cache via env flag
- **Smart Caching**: Efficient image and video caching for faster navigation
- **Thread Pool**: Reuses threads to avoid creation/destruction overhead
- **Event-Driven**: Permanent signal connections for better performance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information about your problem

## ☕ Thank You / Buy Me a Coffee

If you find RAWviewer useful and it's become part of your workflow, feel free to **buy me a coffee** ☕ or chip in to help fund my **RIAT tickets for next year**

---

**Enjoy viewing your RAW photos with RAWviewer!** 📸
