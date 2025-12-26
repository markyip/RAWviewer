# RAWviewer Release Notes

## 🚀 Version 0.5.1 - Stability & Polish Update

### 🎯 What's New
**Enhanced User Experience and Project Cleanup**

### ✨ Key Improvements

#### 🖼️ User Interface Enhancements
✅ **Startup Splash Screen** – Beautiful splash screen displays app icon during initialization, providing visual feedback while the application loads
✅ **Improved Error Handling** – Fixed AttributeError with stdout in PyInstaller windowed builds, ensuring smooth startup on all platforms

#### 📚 Documentation & Project Maintenance
✅ **Updated README** – Comprehensive documentation refresh with:
  - New "Upcoming Features" section highlighting planned gallery view and histogram display
  - Performance notes for large folder loading scenarios
  - Improved architecture overview
  - Complete dependency list and build instructions

✅ **Project Cleanup** – Removed outdated refactoring documentation and release notes, keeping the repository clean and focused

#### 🏗️ Architecture Improvements
✅ **Optimized Image Loading Architecture** – Continued refinement of the multi-threaded image loading system:
  - Unified image processor for all formats
  - Thread pool-based task management
  - Smart caching with memory awareness

### 🐛 Bug Fixes
- Fixed `AttributeError: 'NoneType' object has no attribute 'reconfigure'` in PyInstaller windowed builds
- Improved error handling for console output in packaged executables

### ⚠️ Known Issues
- **Large folder loading**: When loading a large folder, the application launch speed may become slower as it scans and indexes all image files in the directory

### 📦 Technical Details
- Enhanced startup sequence with splash screen support
- Improved resource path handling for bundled executables
- Better cross-platform compatibility for Windows and macOS

### 🎯 What's Coming Next
Check out the "Upcoming Features" section in the README for a preview of:
- Gallery View with adaptive justified layout
- Histogram Display for exposure analysis
- Batch Operations for multiple image processing
- And more exciting features in development!

---

## Previous Release: v0.5

### 🚀 What's New
🎯 Smarter, Faster, and More Responsive RAW Viewing

### 🎨 Major RAW Processing Engine Overhaul
✅ Brand new multi-threaded RAW processor – Up to 46% faster loading on large RAW files
✅ Progressive image loading – See thumbnails immediately while full image decodes
✅ Fallback thumbnail support – Uses embedded previews when decoding fails

### 📊 Performance & Memory Improvements
✅ Advanced image cache system – Speeds up switching and reduces repeated decoding
✅ Smart preload logic – Preloads adjacent images to improve navigation flow
✅ Memory-aware caching – Automatically clears cache based on system memory
✅ Configurable cache budget – Fine-tune performance for low- or high-spec machines

### 🔍 EXIF & UI Enhancements
✅ Instant EXIF display – Metadata appears immediately in fit-to-window mode
✅ Signal-based EXIF system – Real-time loading with no UI delays

---

**Thank you for using RAWviewer!** 📸



