# RAWviewer v2.0.1

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

![Version](https://img.shields.io/badge/version-2.0.1-blue)
![Downloads](https://img.shields.io/github/downloads/markyip/RAWviewer/total) 
![License](https://img.shields.io/badge/license-MIT-green)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/markyip)

## ✈️ Why This Exists
You're an aviation photographer who just returned from RIAT or spent a day at the Mach Loop. You took thousands of RAW shots of fast jets, helicopters, and flybys — and now you're facing the real challenge:

- You want to **sort through all your photos** and identify the best shots.
- Tools like **Lightroom** or **Capture One** are powerful, but importing and checking each image is **time-consuming**.
- The default **Windows Photos** viewer lets you browse, but:
  - It's clunky with RAW files.
  - You have to zoom in manually to check sharpness.
  - There's no easy way to filter out blurry images.

I've been there myself — I still haven't finished editing my **500GB of RIAT 2024 photos** because of how tedious this process is. That frustration is exactly what inspired me to build **RAWviewer**.

RAWviewer is a lightweight, focused image viewer built specifically for photographers who shoot a lot — especially in aviation, wildlife, or sports.

- Instant file previewing: No import steps — just drag & drop.
- Zoom in with a single key to check sharpness immediately.
- Stay in zoomed mode while browsing with arrow keys.
- Quickly remove blurry photos from the queue with `↓` (moves them to a discard folder).
- No complex controls to memorize — just the essential keys to move fast.

This is a **pre-filtering tool**, letting you go through hundreds of RAW files efficiently **before** committing to editing them in Lightroom or Photoshop.

## 🔍 What is RAWviewer?
**RAWviewer** is a fast, modern, cross-platform image viewer for Windows and macOS, built with PyQt6. It supports advanced zooming, panning, and direct file association, allowing RAW files to be opened with a double-click.

## ✨ Features

- **Cross-platform support**: Windows and macOS
- **Ultra-Fast Performance**: Instant folder loading (scans thousands of images in milliseconds) using optimized algorithms
- **High-Fidelity Thumbnails**: Uses high-quality **LANCZOS resampling** and **2x oversampling** for crystal-clear previews on Retina and 4K displays.
- **Smart Prefetching**: Predictively loads relevant images in the background for zero-latency navigation
- **Memory-First Cache (Default)**: Uses fast in-memory caching by default with no disk/SQLite writes
- **Optional Persistent Cache**: Set `RAWVIEWER_PERSISTENT_CACHE=1` to re-enable disk/SQLite cache persistence
- **Gallery View**: Justified grid layout with virtualized rendering, EXIF-aware ordering, and current-image positioning
- **Gallery search (macOS + Core ML bundle)**: Free-text semantic ranking plus structured metadata filters (`camera:`, ISO, GPS, **`format:`** / **`ext:`**, and more — see README table below)
- **Wide RAW format support**: Canon (CR2, CR3), Nikon (NEF), Sony (ARW), Adobe DNG, and many more
- **Robust Orientation Handling**: Definitive fixes for Sony ARW and other RAW formats, ensuring images are always displayed upright
- **Pillarbox-Free Gallery**: Accurately calculates aspect ratios to prevent black bars in the gallery view
- **macOS File Association**: Fully integrated with macOS Finder; can be set as the default viewer and supports double-click to open
- **Intuitive navigation**: Keyboard shortcuts, mouse controls, and scroll wheel support
- **Zoom functionality**: Fit-to-window and 100% zoom modes with smooth panning, including native Mac trackpad pinch-to-zoom
- **DNG zoom reliability**: Single-view DNG now prioritizes full-resolution decoding to keep Space / double-click 100% zoom behavior consistent
- **File management**: Move images to discard folder or delete permanently
- **EXIF data display**: View camera settings, focal length, ISO, aperture, and capture information with robust metadata extraction
- **Session persistence**: Remembers your last opened folder, image, and view mode
- **Single-image histogram**: Press `H` to show or hide the strip while viewing one image
- **Modern Installer**: Lightweight executable that automatically provisions a self-contained Python environment and downloads AI models on first launch
- **Professional Startup**: Synchronized native and Qt splash screens for a flicker-free, premium launch experience.
- **Modern UI**: Material Design 3 aesthetics with Font Awesome icons (via qtawesome) and non-intrusive loading indicators
- **Platform-specific chrome**: On Windows, the bottom bar omits Share (no stable system share without WinRT interop); **Share** remains on macOS.
- **Non-destructive visual rotate**: Rotate in viewer by 90° steps without modifying original files (including RAW), with gallery-visible tiles refreshed immediately.
- **Precision Focus Area Detection**: Overlays the camera's focus point(s) using manufacturer-specific MakerNote data (Canon, Nikon, Sony) plus EXIF SubjectArea/SubjectLocation with orientation-aware mapping and robust coordinate scaling.

## 🚀 Quick Start

### Download Executable

#### Windows
1. Download the latest release from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest)
2. Download `RAWviewer.exe` directly (no zip extraction needed)
3. Double-click `RAWviewer.exe` to initiate the installation process. It will automatically download the necessary dependencies and AI models to a destination of your choice.
4. Launch RAWviewer from the Desktop shortcut created during installation! (You can safely delete the original `RAWviewer.exe` installer afterwards).

#### macOS
1. Download the latest release from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest)
2. Download and extract `RAWviewer-v2.0.1-macOS.zip`
3. Drag `RAWviewer.app` to your **Applications** folder.
4. **CRITICAL FIRST STEP:** Because this is an open-source app not signed via the paid Apple Developer program, macOS Gatekeeper will incorrectly label it as "Damaged" or block it. **You must run this command in your Terminal once** to remove the download quarantine flag:
   ```bash
   xattr -cr /Applications/RAWviewer.app
   ```
5. You can now double-click to launch it normally from Applications or Launchpad!

## ⌨️ Keyboard Shortcuts

- **Space**: Toggle between fit-to-window and 100% zoom
- **`G`**: Toggle between Gallery View and Single Image View
- **`Esc`**: Return to Gallery View (from Single View)
- **`←`/`→` arrows**: Navigate between images
- **`↓`**: Move current image to Discard folder
- **Delete**: Delete current image (with confirmation)
- **`H`**: Show or hide the single-image histogram strip
- **`F`**: Toggle dashed focus/subject indicator overlay (amber = maker AF, lime = EXIF subject area)

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

Optional: **stronger semantic models** (advanced). The app bundle uses **MobileCLIP2-S0** for speed and size. From the same MobileCLIP2 family, **S2** or **B** checkpoints usually score better on open-vocabulary retrieval at the cost of a larger Core ML package and slower indexing—export with `python scripts/export_mobileclip2_coreml.py --model MobileCLIP2-S2` (today’s `--for-app` flow names files `mobileclip2_s0_*`; you can replace those mlpackages after export or adjust filenames to match). You can also set environment variable **`RAWVIEWER_MOBILECLIP_VARIANT=s2`** to prefer Apple’s downloadable **MobileCLIP S2** Core ML models (a different architecture than MobileCLIP2, but often strong for general photo text queries).

Bundled macOS Core ML models are discovered automatically in common app/resource paths.  
Both naming schemes are supported:

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

## 🏗️ Building from Source

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Windows
**Option 1: Using batch script (recommended)**
```batch
# Run the automated build script
build_windows.bat
```

**Option 2: Manual build**
```bash
# Activate virtual environment (if using one)
rawviewer_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build executable
python build.py
```

### macOS
```bash
# Run the automated build script
./build_macos.sh
```

### Dependencies
All dependencies are listed in `requirements.txt`:
- PyQt6 >= 6.6.0
- rawpy >= 0.25.0
- numpy >= 2.0.0
- Pillow >= 10.0.0
- send2trash >= 1.8.0
- pyinstaller >= 6.0.0
- natsort >= 8.4.0
- exifread >= 3.0.0
- psutil >= 5.9.0
- pyqtgraph >= 0.13.0
- qtawesome >= 1.2.0

## 🐛 Troubleshooting

### Windows
- **"Windows protected your PC"**: Click "More info" → "Run anyway"
- **Antivirus warnings**: Add RAWviewer to your antivirus exclusions
- **Performance issues**: Try running as administrator
- **AttributeError with stdout**: This is normal for windowed builds - the application runs without a console window
- **Crash code `-1073741819` / `0xC0000005` (access violation)**:
  - This is a native crash (Qt/LibRaw/ONNX/driver layer), not always a Python exception.
  - Check logs in both locations:
    - `src/logs/` (project-local when enabled)
    - `%LOCALAPPDATA%\RAWviewer\logs\` (persistent crash reports / fatal dumps)

### macOS
- **"App is damaged and should be moved to the Trash" / "Apple could not verify RAWviewer is free of malware"**: 
  - **Why it happens**: Apple heavily restricts apps downloaded outside the App Store that aren't signed with a paid developer certificate. On newer macOS versions (especially Apple Silicon M1/M2/M3), macOS breaks the app's ad-hoc signature and aggressively blocks opening it.
  - **The Fix (Fastest)**: Open your **Terminal** app and run the following command to remove the quarantine flag:
    ```bash
    xattr -cr /Applications/RAWviewer.app
    ```
    *(Note: If you placed the app somewhere other than the Applications folder, update the path accordingly).*

- **"Symbol not found: (_mkfifoat)" or App crashes instantly on macOS 12 (Monterey) or older**:
  - **Why it happens**: The pre-built release is compiled using a newer macOS 13+ SDK. Older macOS versions do not have the required system files to run it.
  - **The Fix**: You must build the app locally (see the "Ultimate Fix" below).

#### 🛠️ The Ultimate Fix: Build Locally (Solves Both Issues Above)
If you are on macOS 12 or older, OR if you simply want to permanently bypass all Gatekeeper/Quarantine warnings forever, you can build the app directly on your own machine. It takes about 2 minutes:
1. **Install Python 3.10, 3.11, or 3.12** (We recommend the official installer from [python.org](https://www.python.org/downloads/macos/)).
2. **Open Terminal** and run these commands to download and build:
   ```bash
   git clone https://github.com/markyip/RAWviewer.git
   cd RAWviewer
   ./build_macos.sh
   ```
This will automatically create a perfectly compatible, warning-free `RAWviewer.app` inside the `dist/` folder!

#### 🔧 Local Build Troubleshooting
- **Error: "No matching distribution found for pyexiv2"**
  - **Why it happens**: You are using an older version of Python (like macOS Monterey's default Python 3.9) on an Apple Silicon (M1/M2) Mac. `pyexiv2` does not provide pre-compiled packages for that specific combination.
  - **The Fix**: 
    1. Install a newer version of Python (e.g., Python 3.11).
    2. **CRITICAL:** If you previously ran the build script, it created a virtual environment stuck on the old Python version. Delete it by running `rm -rf rawviewer_env`.
    3. Re-run `./build_macos.sh` (If it still uses 3.9, explicitly point to your new Python, e.g., `/usr/local/bin/python3 ./build_macos.sh`).

- **Error: Massive C++ compilation failures / PyQt6 missing wheels**
  - **Why it happens**: You are using a bleeding-edge version of Python (like Python 3.14). It takes the open-source community several months to build pre-compiled packages for brand-new Python versions. Without a wheel, the installer attempts to compile massive UI frameworks like PyQt6 from raw C++ source code, which usually fails.
  - **The Fix**: Roll back to a widely supported "sweet spot" version like **Python 3.11** or **3.12**, where every single required library has highly stable, pre-compiled macOS packages ready to download instantly. Remember to delete your old `rawviewer_env` folder before rebuilding!

- **Homebrew delays on macOS 12 Monterey or older**: 
  - Homebrew has officially dropped "binary bottle" support for Monterey. However, **it still works**. When the build script attempts to `brew install inih gettext`, Homebrew will simply compile them from source on your machine. This is completely normal but may take 2-3 extra minutes.

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
