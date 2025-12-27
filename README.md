# RAWviewer v1.2

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

![Version](https://img.shields.io/badge/version-1.2-blue)
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
- **Smart Prefetching**: Predictively loads relevant images in the background for zero-latency navigation
- **Gallery View**: Justified grid layout with virtualized rendering for smooth scrolling through massive collections
- **Wide RAW format support**: Canon (CR2, CR3), Nikon (NEF), Sony (ARW), Adobe DNG, and many more
- **Automatic orientation correction**: Reads EXIF orientation data and displays images correctly (portrait/landscape)
- **Intuitive navigation**: Keyboard shortcuts, mouse controls, and scroll wheel support
- **Zoom functionality**: Fit-to-window and 100% zoom modes with smooth panning
- **File management**: Move images to discard folder or delete permanently
- **EXIF data display**: View camera settings and capture information
- **Session persistence**: Remembers your last opened folder, image, and view mode
- **Portable executable**: No Python installation required for users
- **Modern UI**: Material Design 3 aesthetics with non-intrusive loading indicators

## 🚀 Quick Start

### Download Executable

#### Windows
1. Download the latest release from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest)
2. Download `RAWviewer.exe` directly (no zip extraction needed)
3. Double-click `RAWviewer.exe` to launch
4. **Optional**: Create a desktop shortcut or pin to taskbar

#### macOS
1. Download the latest release from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest)
2. Download and extract `RAWviewer-v0.5-macOS.zip`
3. Drag `RAWviewer.app` to your Applications folder
4. Double-click to launch from Applications or Launchpad
5. **First launch**: Right-click → "Open" if blocked by Gatekeeper

## ⌨️ Keyboard Shortcuts

- **Space**: Toggle between fit-to-window and 100% zoom
- **`G`**: Toggle between Gallery View and Single Image View
- **`Esc`**: Return to Gallery View (from Single View)
- **`←`/`→` arrows**: Navigate between images
- **`↓`**: Move current image to Discard folder
- **Delete**: Delete current image (with confirmation)

## 🖱️ Mouse Controls

- **Double-click**: Toggle zoom mode
- **Click and drag**: Pan image when zoomed in
- **Drag and drop**: Open images or folders
- **Scroll Wheel**: Navigate images (Single View) or scroll grid (Gallery View)

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

## 🐛 Troubleshooting

### Windows
- **"Windows protected your PC"**: Click "More info" → "Run anyway"
- **Antivirus warnings**: Add RAWviewer to your antivirus exclusions
- **Performance issues**: Try running as administrator
- **AttributeError with stdout**: This is normal for windowed builds - the application runs without a console window

### macOS
- **"App is damaged" error**: Go to System Preferences → Security & Privacy → Allow
- **Gatekeeper warnings**: Right-click the app → Open → Open anyway
- **Performance issues**: Grant Full Disk Access in Privacy settings

## 🚧 Upcoming Features

We're continuously working to improve RAWviewer. Here are some features planned for future releases:

- **Histogram Display**: View RGB and luminance histograms to analyze exposure and color distribution
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
- **Smart Caching**: Efficient image and thumbnail caching for faster navigation
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
