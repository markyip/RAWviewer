# RAWviewer v0.4.1 Release Notes

## 🚀 What's New

### 🎨 Major Color Correction Improvements
- **Fixed Canon CR3/CR2 red hue issues** - Canon files now display with proper white balance using camera-specific processing
- **Fixed Fujifilm RAF green hue issues** - Fujifilm files now show accurate colors with automatic white balance correction
- **Smart camera detection** - Automatic detection of Canon (.cr2, .cr3) and Fujifilm (.raf) files for optimized processing
- **Performance optimization for large files** - Files over 80MB automatically use fast processing mode (46% speed improvement)
- **Camera-specific white balance** - Uses camera white balance with fallback to auto white balance for better color accuracy

### 🔧 Bug Fixes & Improvements
- **Fixed QImage compatibility issues** with PyQt6 - resolves crashes when loading certain RAW files
- **Enhanced NEF file handling** - improved error recovery for corrupted or incompatible NEF files with smart thumbnail fallback
- **Better error messages** - more informative feedback when files cannot be processed
- **Improved thumbnail fallback** - smoother experience when RAW processing fails
- **Zoom/Pan state preservation** - maintains your zoom and pan position when deleting or discarding images

### 🛠️ Build System Enhancements
- **New Windows build script** (`build_windows.bat`) with automatic dependency management
- **Enhanced cross-platform builds** - streamlined process for both Windows and macOS
- **Updated icon handling** - icons organized in dedicated `icons/` directory
- **Improved build.py script** - better cross-platform icon support

### 📁 Repository Improvements
- **Streamlined file organization** - removed extensive test suite, focused on essential files
- **Enhanced .gitignore** - prevents build artifacts from being committed
- **Updated README** - comprehensive documentation including Known Issues section and camera compatibility information
- **Cleaner project structure** - better organization of assets and build files

### 🎯 User Experience Enhancements
- **Consistent zoom behavior** - zoom and pan state preserved across file operations
- **Better file handling** - improved stability when working with problematic RAW files
- **Enhanced icon display** - proper application icons across all platforms
- **Comprehensive orientation support** - automatic handling for Sony, Leica, and Hasselblad cameras that store pre-rotated RAW data

## 📊 Performance Improvements

### Canon Files
- **Color accuracy**: R/G ratio improved from 1.93 to 1.08 (balanced colors)
- **Processing**: Automatic white balance correction eliminates red color casts

### Fujifilm Files
- **Color accuracy**: G/R ratio improved from 1.21 to 0.94 (natural colors)
- **Processing speed**: 46% faster processing for large files (90MB+ files: 7.37s → 3.96s)
- **Smart optimization**: Automatic half-size processing for files over 80MB

## 📥 Downloads

### macOS
- **RAWviewer-v0.4.1-macOS-Updated.zip** (~37MB)
- Compatible with macOS 10.15 and later
- Universal app bundle with all dependencies included
- Includes all latest color correction improvements

### Windows
- **RAWviewer.exe** (~37MB)
- Compatible with Windows 10 and later
- Single executable file, no installation required
- Build using `build_windows.bat` on Windows machine

## 🔄 Upgrade Notes

This release includes significant color accuracy improvements for Canon and Fujifilm users:

- **Canon CR3/CR2 users**: Red hue issues are now automatically corrected
- **Fujifilm RAF users**: Green tint problems are resolved with much faster processing
- **Large file handling**: Files over 80MB process significantly faster
- **Better crash resistance** - fixes PyQt6 compatibility issues
- **Improved workflow** - zoom/pan state preservation makes image review more efficient

## 🐛 Known Issues

- Some newer NEF files may still trigger thumbnail fallback mode due to LibRaw compatibility limitations
- First launch on macOS may require "Open anyway" due to Gatekeeper security
- PyInstaller deprecation warning for onefile + windowed mode (functionality not affected)
- **Camera compatibility**: Support for newest camera models may be limited due to LibRaw library constraints
- **Processing trade-offs**: RAWviewer prioritizes speed over perfect color accuracy for rapid photo sorting

## 💝 Thank You

Thanks to all users who reported color accuracy issues with Canon and Fujifilm files. Your feedback led to these major improvements that make RAWviewer much more reliable for professional photography workflows!

---

**Full Changelog**: https://github.com/markyip/RAWviewer/compare/v0.4.0...v0.4.1 