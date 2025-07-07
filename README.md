# RAWviewer v0.5

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

![Version](https://img.shields.io/badge/version-0.5-blue)
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
- **Wide RAW format support**: Canon (CR2, CR3), Nikon (NEF), Sony (ARW), Adobe DNG, and many more
- **Automatic orientation correction**: Reads EXIF orientation data and displays images correctly (portrait/landscape)
- **Intuitive navigation**: Keyboard shortcuts and mouse controls
- **Zoom functionality**: Fit-to-window and 100% zoom modes with smooth panning
- **File management**: Move images to discard folder or delete permanently
- **EXIF data display**: View camera settings and capture information
- **Session persistence**: Remembers your last opened folder and image
- **Portable executable**: No Python installation required for users
- **Threaded processing**: Smooth, non-blocking UI

## 🚀 Quick Start

Download Executable
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
- **`←`/`→` arrows**: Navigate between images
- **`↓`**: Move current image to Discard folder
- **Delete**: Delete current image (with confirmation)

## 🖱️ Mouse Controls

- **Double-click**: Toggle zoom mode
- **Click and drag**: Pan image when zoomed in
- **Drag and drop**: Open images or folders

## 📁 Supported Formats

### RAW Formats
- **Canon**: CR3
- **Nikon**: NEF
- **Sony**: ARW
- **Adobe**: DNG
- **Olympus**: ORF
- **Panasonic**: RW2
- **Fujifilm**: RAF
- **Hasselblad**: 3FR

### Standard Formats
- **JPEG**: JPG, JPEG
- **HEIF**: HEIF

## 🏗️ Building

### Windows
**Option 1: Using batch script (recommended)**
```batch
# Run the automated build script
build_windows.bat
```

**Option 2: Manual build**
```bash
# Activate virtual environment
rawviewer_env\Scripts\activate

# Install dependencies
pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow

# Build executable
python build.py
```

### macOS
```bash
# Run the automated build script
./build_macos.sh
```

## 🐛 Troubleshooting

### Windows
- **"Windows protected your PC"**: Click "More info" → "Run anyway"
- **Antivirus warnings**: Add RAWviewer to your antivirus exclusions
- **Performance issues**: Try running as administrator

### macOS
- **"App is damaged" error**: Go to System Preferences → Security & Privacy → Allow
- **Gatekeeper warnings**: Right-click the app → Open → Open anyway
- **Performance issues**: Grant Full Disk Access in Privacy settings

## ⚠️ Known Issues

### Camera Compatibility
- **Newer camera models**: Support for the latest camera releases may be limited due to LibRaw library compatibility
- **Proprietary RAW formats**: Some manufacturers' newest RAW formats may not be fully supported immediately after camera release
- **Firmware updates**: Camera firmware updates may introduce RAW format changes that require LibRaw updates

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
