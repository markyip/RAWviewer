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
3. The standalone executable will be created at `dist/RAWviewer.exe`.

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

For detailed installation instructions, see [`INSTALL.md`](INSTALL.md).

## 🎯 Quick Start

1. **Launch** the application
2. **Open** a RAW file using:
   - File → Open (Ctrl+O)
   - Drag and drop a RAW file onto the window
3. **Navigate** between images:
   - Use Left/Right arrow keys
   - Application automatically scans the folder
4. **Zoom** control:
   - Press Space to toggle between fit-to-window and 100% zoom
5. **Delete** unwanted images:
   - Press Delete key (with confirmation dialog)

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open file dialog |
| `Space` | Toggle zoom (fit-to-window ↔ 100%) |
| `Left Arrow` | Previous image |
| `Right Arrow` | Next image |
| `Delete` | Delete current image (with confirmation) |
| `Ctrl+Q` | Exit application |

## 🔧 Usage

### Opening Images
- **File Menu**: Use File → Open or Ctrl+O
- **Drag & Drop**: Drag RAW files directly onto the window
- **Folder Navigation**: Opening any image automatically scans the folder

### Viewing Images
- **Fit to Window**: Default mode scales images to fit your screen
- **100% Zoom**: Press Space to view at actual pixel size
- **Automatic Scaling**: Images resize automatically when you resize the window

### Navigation
- **Arrow Keys**: Navigate between images in the current folder
- **Automatic Sorting**: Files are sorted naturally (IMG_1, IMG_2, IMG_10...)
- **Wraparound**: Navigation wraps from last to first image

### File Management
- **Safe Deletion**: Delete key moves files to Recycle Bin with confirmation
- **Status Information**: See filename, dimensions, zoom level, and position

For detailed usage instructions, see [`USER_MANUAL.md`](USER_MANUAL.md).

## 🏗️ Building from Source

### Prerequisites
- Python 3.8+
- All dependencies from `requirements.txt`

### Build Process
1. **Quick Build**: Run `build.bat` (Windows)
2. **Manual Build**: Run `python build.py`
3. **Output**: Executable created in `dist/RAWImageViewer.exe`

For detailed build instructions, see [`BUILD_INSTRUCTIONS.md`](BUILD_INSTRUCTIONS.md).

## 🧪 Testing

The application includes comprehensive testing:

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test suites
python tests/test_suite.py              # Core functionality
python tests/raw_format_compatibility_test.py  # Format support
python tests/keyboard_shortcuts_test.py # Input handling
python tests/edge_case_test.py          # Edge cases
python tests/performance_test.py        # Performance
```

## 📊 Performance

### Typical Performance
- **Startup Time**: < 5 seconds
- **Image Loading**: 1-3 seconds (depends on file size)
- **Navigation**: < 0.05 seconds between images
- **Memory Usage**: 100-200MB base, +100MB per large RAW file

### Optimization Tips
- **SSD Storage**: Faster image loading from solid-state drives
- **RAM**: More RAM allows handling larger RAW files
- **CPU**: Multi-core processors improve RAW processing speed

## 🐛 Troubleshooting

### Common Issues

**Application won't start**
- Ensure Python 3.8+ is installed
- Check all dependencies are installed: `pip install -r requirements.txt`
- Try running from command line to see error messages

**RAW files won't open**
- Verify the file format is supported (see [`SUPPORTED_FORMATS.md`](SUPPORTED_FORMATS.md))
- Check if the file is corrupted
- Ensure sufficient RAM for large files

**Performance issues**
- Close other applications to free memory
- Move images to faster storage (SSD)
- Reduce image file sizes if possible

**Keyboard shortcuts not working**
- Click on the main window to ensure it has focus
- Check if other applications are capturing the same shortcuts

For more troubleshooting help, see [`USER_MANUAL.md`](USER_MANUAL.md).

## 🔒 Security

- **Safe Deletion**: Files are moved to Recycle Bin, not permanently deleted
- **No Network Access**: Application works completely offline
- **No Data Collection**: No telemetry or user data collection
- **Open Source**: Code is available for inspection

## 📈 Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Add docstrings for new functions
- Include tests for new features

For detailed development information, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **rawpy**: RAW image processing library
- **PyQt6**: GUI framework
- **natsort**: Natural sorting for filenames
- **send2trash**: Safe file deletion

## 📞 Support

- **Issues**: Report bugs on the [GitHub issues page](https://github.com/yourusername/rawviewer/issues)
- **Documentation**: Check the [`USER_MANUAL.md`](USER_MANUAL.md) for detailed help
- **FAQ**: Common questions answered in the User Manual

## 🔄 Version History

- **v1.0.0**: Initial release with full RAW format support
- Features: Multi-format support, keyboard navigation, safe deletion
- Testing: Comprehensive QA suite with 2,600+ lines of tests

---

**Made with ❤️ for photographers who need a fast, reliable RAW image viewer**