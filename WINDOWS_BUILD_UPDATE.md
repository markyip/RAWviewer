# Windows Build Script Update

## Summary

Updated the Windows build system to match the macOS build functionality and ensure all dependencies are properly installed.

## Changes Made

### 1. Enhanced build.py
- **Added dependency installation**: Now automatically installs all required packages including Pillow
- **Added error handling**: Better error messages and dependency checking
- **Added Pillow support**: Critical for NEF thumbnail fallback functionality
- **Improved logging**: More detailed build process information

### 2. New build_windows.bat
- **Automated build script**: Windows equivalent to build_macos.sh
- **Virtual environment management**: Creates and activates venv automatically
- **Dependency installation**: Installs all required packages
- **Clean build process**: Removes previous builds automatically
- **User-friendly output**: Clear instructions and status messages

### 3. Updated README.md
- **Added Windows batch script option**: Recommended automated build method
- **Enhanced build instructions**: Clear step-by-step process
- **Consistent documentation**: Matches macOS build process

## Dependencies Included

All build scripts now install these dependencies:
- PyQt6 (GUI framework)
- rawpy (RAW file processing)
- send2trash (Safe file deletion)
- pyinstaller (Executable creation)
- natsort (Natural sorting)
- exifread (EXIF data reading)
- **Pillow** (Image processing for NEF thumbnails)

## Usage

### Windows Users
**Recommended**: Use the automated batch script
```batch
build_windows.bat
```

**Manual**: Use the Python script directly
```bash
python build.py
```

### Benefits
- **Consistent builds**: Same dependencies across platforms
- **NEF compatibility**: Pillow dependency ensures thumbnail fallback works
- **User-friendly**: Automated scripts handle all setup
- **Error handling**: Clear error messages if build fails
- **Cross-platform**: Same functionality on Windows and macOS

## Files Modified
- `build.py` - Enhanced with dependency installation
- `build_windows.bat` - New automated build script
- `README.md` - Updated build instructions
- `requirements.txt` - Already included Pillow

## Compatibility
- Windows 10/11
- Python 3.8+
- All RAW formats including NEF with thumbnail fallback
- Automatic orientation correction
- Full PyQt6 compatibility 