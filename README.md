# RAWviewer

<p align="center">
  <img src="appicon.png" alt="RAWviewer Icon" width="500">
</p>

  
![Version](https://img.shields.io/badge/version-0.4.0-blue)
![Downloads](https://img.shields.io/github/downloads/markyip/RAWviewer/total) 
![License](https://img.shields.io/badge/license-MIT-green)



A fast, modern cross-platform image viewer for Windows and macOS, built with PyQt6. It supports advanced zooming, panning, and direct file association, allowing RAW files to be opened with a double-click.

## Description
RAWviewer is a lightweight desktop application for quickly viewing and navigating image files from a wide range of cameras. It is designed for photographers who need a simple, fast, and user-friendly tool to browse, zoom, and inspect images without the overhead of a full photo editor.

## ✨ Features
- ✅ Wide RAW format support (Canon, Nikon, Sony, Fujifilm, Panasonic, Olympus, and more)
- 💡 Despite the name *RAWviewer*, the application also supports standard image formats such as **JPEG**, **JPG**, and **HEIF**.
- ⚡ Fast image loading with rawpy and numpy
- 🎹 Keyboard shortcuts for speed and efficiency
- 🔗 File association for opening RAWs directly via double-click
- 📝 EXIF display in the status bar
- 📦 Portable executable – No Python installation required for users
- 🗑️ Safe image deletion (with confirmation)
- 🧵 Threaded processing for smooth, non-blocking UI
- 📂 **Open entire folders**: Browse and view all supported images in a selected folder, starting from the first image.
- ⏸️ **Session resume**: Automatically restores your last viewed folder and image when you reopen the app (if they still exist).
- 🎯 **Smart zoom**: Double-click any area to zoom in precisely to that location, or use Space for center zoom.

---

## 🖥️ System Requirements

### Windows (Supported)
- OS: Windows 10 or later
- RAM: 4GB minimum (8GB recommended)
- Storage: ~200MB for app; additional space for image files
- Display: Minimum 1024×768 resolution

### macOS (Supported)
- OS: macOS 11.0 (Big Sur) or later
- Architecture: Intel x64 or Apple Silicon (M1/M2/M3)
- RAM: 4GB minimum (8GB recommended)
- Storage: ~200MB for app; additional space for image files

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

### Option 2: Install from Source
1. Clone or download this repository
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python src/main.py
   ```

---

## 🛠️ Building the Executable

### Windows
1. Ensure your virtual environment is activated.
2. Run:
   ```bash
   python build.py
   ```
3. The standalone executable will appear at `dist/RAWviewer.exe`.

### macOS
1. Run the build script:
   ```bash
   ./build_macos.sh
   ```
   Or manually:
   ```bash
   python3 -m venv rawviewer_env
   source rawviewer_env/bin/activate
   pip install PyQt6 rawpy send2trash pyinstaller natsort exifread
   python build.py
   ```
2. The executable will appear at:
   - **App Bundle**: `dist/RAWviewer.app` (double-click to run)
   - **Command Line**: `dist/RAWviewer`

---

## 🧭 Usage Guide

- **Open images** via:
  - File → Open
  - Drag & drop
  - Double-click from File Explorer (after setting default association)
- **Open a folder of images**:
  - File → Open Folder (or `Ctrl + Shift + O`)
  - The app will display the first image in the folder and allow navigation with arrow keys.
- **Session resume**:
  - When you close and reopen RAWviewer, it will automatically restore the last folder and image you were viewing (as long as they still exist).

- **Keyboard Shortcuts**
  
  **Windows:**
  - `Ctrl + O`: Open file
  - `Ctrl + Shift + O`: Open folder
  - `Space` / Double-click: Toggle zoom
  - `←` / `→`: Previous / Next image
  - `↓` (Down arrow): Move current image to "Discard" folder (non-destructive)
  - `Delete`: Delete current image
  - `Ctrl + Q`: Quit app
  
  **macOS:**
  - `Cmd + O`: Open file
  - `Cmd + Shift + O`: Open folder
  - `Space` / Double-click: Toggle zoom
  - `←` / `→`: Previous / Next image
  - `↓` (Down arrow): Move current image to "Discard" folder (non-destructive)
  - `Delete`: Delete current image
  - `Cmd + Q`: Quit app
💡 The "Discard" folder allows you to filter out unwanted images for later review, without permanently deleting them.

🔍 Zoom in with Space to center the view, or double-click any spot to zoom in right where you clicked.

↔️ Switch images with the arrow keys even while zoomed in — no need to reset the zoom.


---

## 📁 Set as Default App for Image Files

### Windows
1. Right-click an image file (e.g., `.cr2`, `.arw`)
2. Select **Open with → Choose another app**
3. Browse to `RAWviewer.exe`
4. Enable **Always use this app to open .___ files**

### macOS
1. Right-click an image file (e.g., `.cr2`, `.arw`)
2. Select **Open With → Other...**
3. Navigate to Applications and select `RAWviewer.app`
4. Check **Always Open With**
5. Click **Open**

---

## 📸 Supported Image Formats

### ✅ Primary Support (Extensively Tested)
- **Canon**: `.cr2`, `.cr3`
- **Nikon**: `.nef`
- **Sony**: `.arw`, `.srf`
- **Adobe**: `.dng`
- **Fujifilm**: `.raf`
- **Panasonic**: `.rw2`
- **Olympus**: `.orf`
- **Pentax**: `.pef`
- **Samsung**: `.srw`
- **Sigma**: `.x3f`

### 🧪 Extended Support
- **Hasselblad**: `.3fr`, `.fff`
- **Phase One**: `.iiq`, `.cap`
- **Epson**: `.erf`
- **Mamiya**: `.mef`
- **Leaf**: `.mos`
- **Casio**: `.nrw`
- **Leica**: `.rwl`

### 🖼️ Standard Formats:
- **JPEG / JPG
- **HEIF / HEIC (if supported by the system's image codecs)

---

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

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
