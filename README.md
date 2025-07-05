# RAWviewer

A fast, modern RAW image viewer for Windows, built with PyQt6. Supports advanced zoom, pan, film strip (thumbnail bar), and direct file association for double-click opening of RAW files.

## Description
RAWviewer is a lightweight desktop application for quickly viewing and navigating RAW image files from a wide range of cameras. It is designed for photographers who need a simple, fast, and user-friendly tool to browse, zoom, and inspect RAW images without the overhead of a full photo editor.

## Features
- **Wide RAW format support** (Canon, Nikon, Sony, Fujifilm, Panasonic, Olympus, and more)
- **Fast image loading** using rawpy and numpy
- **Advanced zoom and pan** (double-click to zoom, drag to pan, fit-to-window toggle)
- **Film strip (thumbnail bar)** for quick navigation between images in a folder
- **Keyboard shortcuts** for fast workflow
- **Drag and drop** support for opening files
- **Direct file association**: Double-click a RAW file in Explorer to open it in RAWviewer
- **EXIF info** displayed in the status bar
- **Lightweight, portable executable** (no Python required for end users)

## Installation

### Prerequisites (for building from source)
- Windows 10 or later
- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Install dependencies
1. Open a terminal in the project directory.
2. (Recommended) Create and activate a virtual environment:
   ```sh
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Building the Executable
1. Make sure your virtual environment is activated.
2. Run the build script:
   ```sh
   python build.py
   ```
3. The standalone executable will be created at `dist/RAWviewer.exe` (Windows) or `dist/RAWviewer` (macOS).

## macOS Build & Usage Instructions

### Prerequisites
- macOS 11+ (Apple Silicon or Intel)
- Python 3.8+
- [Homebrew](https://brew.sh/) (recommended for Python and dependencies)
- Install dependencies:
  ```sh
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- You may need to install additional Qt plugins for HEIF/HEIC support (see PyQt6 documentation).

### Creating the App Icon (.icns)
1. Prepare a 1024x1024 PNG image for your app icon.
2. On a Mac, create a folder named `icon.iconset` and place your PNG inside, renaming it to `icon_512x512@2x.png`:
   ```sh
   mkdir icon.iconset
   cp your_icon.png icon.iconset/icon_512x512@2x.png
   ```
3. (Optional) For best results, add PNGs at other sizes (see Apple docs).
4. Run:
   ```sh
   iconutil -c icns icon.iconset
   ```
5. Move the resulting `icon.icns` to your project root as `appicon.icns`.

### Build
- Run:
  ```sh
  python build.py
  ```
- The app will be created at `dist/RAWviewer` (standalone binary) and/or `dist/RAWviewer.app` (app bundle, if you use the right PyInstaller options).
- You can run the app with `open dist/RAWviewer.app` or double-click it in Finder.

### Set as Default App for RAW Files
1. Right-click a RAW file in Finder and choose **Get Info**.
2. Under **Open with**, select `RAWviewer` (browse to the built app if needed).
3. Click **Change All...** to set as default for all files of that type.

### Notes
- The app icon will only show if you use a valid `.icns` file.
- File association must be set by the user in Finder (cannot be set programmatically).
- All features (zoom, pan, delete, discard, navigation) work the same as on Windows.
- If you only have a 1024x1024 PNG, you can use it for the icon, but the icon may look less sharp at small sizes.
- For a true `.app` bundle, you may want to run PyInstaller without `--onefile` for easier distribution.

## Usage
- **Run the app:** Double-click `RAWviewer.exe` or run from the command line.
- **Open images:** Use File > Open, drag and drop, or double-click a RAW file in Explorer (after setting RAWviewer as the default app for that file type).
- **Keyboard shortcuts:**
  - `Ctrl+O`: Open file
  - `Space` or double-click: Toggle zoom
  - `Left/Right arrows`: Previous/next image
  - `Delete`: Delete current image
  - `Ctrl+Q`: Quit
- **Film strip:** Toggle the thumbnail bar from the status bar button.

## Setting as Default App for RAW Files
1. Right-click a RAW file (e.g., `.cr2`, `.arw`) in Explorer.
2. Choose **Open with > Choose another app**.
3. Browse to `RAWviewer.exe` and check **Always use this app to open .CR2 files**.

## Supported RAW Formats
- Canon: `.cr2`, `.cr3`
- Nikon: `.nef`
- Sony: `.arw`, `.srf`
- Adobe: `.dng`
- Olympus: `.orf`
- Panasonic: `.rw2`
- Pentax: `.pef`
- Samsung: `.srw`
- Sigma: `.x3f`
- Fujifilm: `.raf`
- Hasselblad: `.3fr`, `.fff`
- Phase One: `.iiq`, `.cap`
- Epson: `.erf`
- Mamiya: `.mef`
- Leaf: `.mos`
- Casio: `.nrw`
- Leica: `.rwl`

## License
MIT License

## 🚀 Key Features

- **Wide Format Support**: Supports 21+ RAW formats including Canon, Nikon, Sony, Adobe DNG, and more
- **Fast Navigation**: Browse through folders with arrow keys, automatic folder scanning
- **Intelligent Zoom**: Toggle between fit-to-window and 100% zoom with spacebar
- **Drag & Drop**: Simply drag RAW files onto the window to open them
- **Safe Deletion**: Move unwanted images to trash with confirmation dialog
- **Keyboard Shortcuts**: Efficient navigation with intuitive keyboard controls
- **Threaded Processing**: Non-blocking RAW processing for smooth user experience
- **Status Information**: Real-time display of image dimensions, zoom level, and file position

## 📸 Supported File Formats

### Primary Support (Excellent compatibility)
- **Canon**: .cr2, .cr3
- **Nikon**: .nef
- **Sony**: .arw, .srf
- **Adobe**: .dng
- **Olympus**: .orf
- **Panasonic**: .rw2
- **Pentax**: .pef
- **Samsung**: .srw
- **Sigma**: .x3f
- **Fujifilm**: .raf

### Extended Support
- **Hasselblad**: .3fr, .fff
- **Phase One**: .iiq, .cap
- **Epson**: .erf
- **Mamiya**: .mef
- **Leaf**: .mos
- **Casio**: .nrw
- **Leica**: .rwl

For detailed format compatibility information, see [`SUPPORTED_FORMATS.md`](SUPPORTED_FORMATS.md).

## 🖥️ System Requirements

### Windows
- **OS**: Windows 10 or later
- **RAM**: 4GB minimum (8GB recommended for large RAW files)
- **Storage**: 200MB for application, additional space for images
- **Display**: 1024x768 minimum resolution

### Python (Development)
- **Python**: 3.8 or later
- **Dependencies**: Listed in [`requirements.txt`](requirements.txt)

## 📦 Installation

### Option 1: Download Executable (Recommended)
1. Download the latest release from the [releases page](https://github.com/yourusername/rawviewer/releases)
2. Extract the ZIP file to your desired location
3. Run `RAWImageViewer.exe`

### Option 2: Install from Source
1. Clone or download this repository
2. Install Python 3.8 or later
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python src/main.py
   ```

For detailed installation instructions, see [`