# RAWviewer

![Version](https://img.shields.io/badge/version-0.2.0-blue) 
![Downloads](https://img.shields.io/github/downloads/markyip/RAWviewer/total) 
![License](https://img.shields.io/badge/license-MIT-green)

A fast, modern RAW image viewer for Windows, built with PyQt6. Supports advanced zoom, pan, film strip (thumbnail bar), and direct file association for double-click opening of RAW files.

## Description
RAWviewer is a lightweight desktop application for quickly viewing and navigating RAW image files from a wide range of cameras. It is designed for photographers who need a simple, fast, and user-friendly tool to browse, zoom, and inspect RAW images without the overhead of a full photo editor.

## ✨ Features
- ✅ Wide RAW format support (Canon, Nikon, Sony, Fujifilm, Panasonic, Olympus, and more)
- ⚡ Fast image loading with rawpy and numpy
- 🎹 Keyboard shortcuts for speed and efficiency
- 🔗 File association for opening RAWs directly via double-click
- 📝 EXIF display in the status bar
- 📦 Portable executable – No Python installation required for users
- 🗑️ Safe image deletion (with confirmation)
- 🧵 Threaded processing for smooth, non-blocking UI

---

## 🖥️ System Requirements

### Windows (Supported)
- OS: Windows 10 or later
- RAM: 4GB minimum (8GB recommended)
- Storage: ~200MB for app; additional space for image files
- Display: Minimum 1024×768 resolution

### Option 1: Download Executable (Windows - Recommended)
1. Download the latest release from the [Releases Page](https://github.com/yourusername/rawviewer/releases)
2. Extract the ZIP file
3. Run `RAWviewer.exe`

### Option 2: Install from Source
1. Clone or download this repository
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   source .venv/bin/activate # macOS/Linux
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

## 🛠️ Building the Executable (Windows)

1. Ensure your virtual environment is activated.
2. Run:
   ```bash
   python build.py
   ```
3. The standalone executable will appear at `dist/RAWviewer.exe`.

---

## 🧭 Usage Guide

- **Open images** via:
  - File → Open
  - Drag & drop
  - Double-click from File Explorer (after setting default association)

- **Keyboard Shortcuts**
  - `Ctrl + O`: Open file
  - `Space` / Double-click: Toggle zoom
  - `←` / `→`: Previous / Next image
  - `Delete`: Delete current image
  - `Ctrl + Q`: Quit app

- **Film Strip View**
  - Toggle thumbnail bar via status bar button

---

## 📁 Set as Default App for RAW Files

1. Right-click a RAW file (e.g., `.cr2`, `.arw`)
2. Select **Open with → Choose another app**
3. Browse to `RAWviewer.exe`
4. Enable **Always use this app to open .___ files**

---

## 📸 Supported RAW Formats

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

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
