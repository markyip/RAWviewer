# RAWviewer v0.4.1 Release Notes

## 🚀 What's New

### 🔧 Bug Fixes & Improvements
- **Fixed QImage compatibility issues** with PyQt6 - resolves crashes when loading certain RAW files
- **Enhanced NEF file handling** - improved error recovery for corrupted or incompatible NEF files
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
- **Updated README** - clearer installation and usage instructions
- **Cleaner project structure** - better organization of assets and build files

### 🎯 User Experience Enhancements
- **Consistent zoom behavior** - zoom and pan state preserved across file operations
- **Better file handling** - improved stability when working with problematic RAW files
- **Enhanced icon display** - proper application icons across all platforms

## 📥 Downloads

### macOS
- **RAWviewer-v0.4.1-macOS-Updated.zip** (~40MB)
- Compatible with macOS 10.15 and later
- Universal app bundle with all dependencies included
- Includes all latest improvements and bug fixes

### Windows
- **RAWviewer.exe** (~45MB)
- Compatible with Windows 10 and later
- Single executable file, no installation required
- Build using `build_windows.bat` on Windows machine

## 🔄 Upgrade Notes

This release focuses on stability, user experience, and repository organization. Key improvements include:

- **Better crash resistance** - fixes PyQt6 compatibility issues
- **Improved workflow** - zoom/pan state preservation makes image review more efficient
- **Cleaner codebase** - removed unnecessary test files and reorganized assets

## 🐛 Known Issues

- Some newer NEF files may still trigger thumbnail fallback mode due to LibRaw compatibility limitations
- First launch on macOS may require "Open anyway" due to Gatekeeper security
- PyInstaller deprecation warning for onefile + windowed mode (functionality not affected)

## 💝 Thank You

Thanks to all users who reported issues and provided feedback. Your input helps make RAWviewer better for everyone!

---

**Full Changelog**: https://github.com/markyip/RAWviewer/compare/v0.4.0...v0.4.1 