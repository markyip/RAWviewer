# RAWviewer

<p align="center">
  <img src="appicon.png" alt="RAWviewer Icon" width="500">
</p>

  
![Version](https://img.shields.io/badge/version-0.4.0-blue)
![Downloads](https://img.shields.io/github/downloads/markyip/RAWviewer/total) 
![License](https://img.shields.io/badge/license-MIT-green)



## ✈️ Why This Exists
You're an aviation photographer who just returned from RIAT or spent a day at the Mach Loop. You took thousands of RAW shots of fast jets, helicopters, and flybys — and now you're facing the real challenge:

- You want to **sort through all your photos** and identify the best shots.
- Tools like **Lightroom** or **Capture One** are powerful, but importing and checking each image is **time-consuming**.
- The default **Windows Photos** viewer lets you browse, but:
  - It’s clunky with RAW files.
  - You have to zoom in manually to check sharpness.
  - There’s no easy way to filter out blurry images.

I’ve been there myself — I still haven’t finished editing my **500GB of RIAT 2024 photos** because of how tedious this process is. That frustration is exactly what inspired me to build **RAWviewer**.


### 💡 The Solution: RAWviewer
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
- ✅ Wide RAW format support (Canon, Nikon, Sony, Fujifilm, Panasonic, Olympus, and more)
- 💡 Supports standard image formats such as **JPEG**, **JPG**, and **HEIF**
- ⚡ Fast image loading with rawpy and numpy
- 🎹 Keyboard shortcuts for speed and efficiency
- 🔗 File association for opening RAWs directly via double-click
- 📝 EXIF display in the status bar
- 📦 Portable executable – No Python installation required for users
- 🗑️ Safe image deletion (with confirmation)
- 🧵 Threaded processing for smooth, non-blocking UI
- 📂 Open entire folders and navigate easily
- ⏸️ Session resume to restore last-viewed folder and image
- 🎯 Smart zoom with double-click or `Space`


## 🚀 Getting Started
### Option 1: Download Executable (Recommended)
#### Windows
1. Download the latest release from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest)
2. Download `RAWviewer.exe` directly (no zip extraction needed)
3. Double-click `RAWviewer.exe` to launch
4. **Optional**: Create a desktop shortcut or pin to taskbar

#### macOS
1. Download the latest release from the [Releases Page](https://github.com/markyip/RAWviewer/releases/latest)
2. Download and extract `RAWviewer-v0.4.0-macOS.zip`
3. Drag `RAWviewer.app` to your Applications folder
4. Double-click to launch from Applications or Launchpad
5. **First launch**: Right-click → "Open" if blocked by Gatekeeper

### Option 2: Run from Source

```bash
# Clone and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```


## 🧭 Usage Guide

- **Drag & Drop** an image to open it
- **Keyboard Navigation**:
  - `←` / `→`: Previous / Next image
  - `Space`: Zoom in/out
  - `↓`: Move image to discard folder
  - `Delete`: Delete image


## 📁 Set as Default Viewer

### Windows:

Right-click a RAW file → Open with → Choose another app → Select `RAWviewer.exe`

### macOS:

Right-click a RAW file → Open with → Other... → Select `RAWviewer.app`


## 📸 Supported Image Formats
### RAW:

`.cr2`, `.cr3`, `.nef`, `.arw`, `.raf`, `.rw2`, `.orf`, `.dng`, `.pef`, `.srw`, `.x3f`, and more

### Standard:

`.jpg`, `.jpeg`, `.heif`, `.heic`


## ⚙️ Build Instructions
### Windows

```bash
python build.py
```

### macOS

```bash
./build_macos.sh
```


## 🆘 Troubleshooting
### Windows
- **App won't open**: Try running as administrator (right-click → "Run as administrator")
- **Windows SmartScreen warning**: Click "More info" → "Run anyway"
- **Antivirus blocking**: Add RAWviewer.exe to your antivirus whitelist

### macOS
- **App won't open**: Right-click `RAWviewer.app` → "Open", then click "Open" in the security dialog
- **"App is damaged" error**: Go to System Preferences → Security & Privacy → Allow
- **Gatekeeper issues**: Run `sudo xattr -rd com.apple.quarantine /Applications/RAWviewer.app` in Terminal

### General
- **Poor performance**: Ensure you have sufficient RAM (8GB+ recommended)
- **Large RAW files slow**: Close other memory-intensive applications
- **File won't open**: Check if the file format is supported in the list above


## 📜 License
This project is licensed under the [MIT License](LICENSE).
