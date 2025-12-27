# RAWviewer Release Notes

## 🚀 Version 1.1 - Gallery View Release

### 🎯 What's New
**Major Feature Release: Gallery View with Optimized Performance**

### ✨ Key Features

#### 🖼️ Gallery View (NEW!)
✅ **Justified Gallery Layout** – Browse multiple images in an adaptive, justified grid layout that efficiently utilizes screen space
✅ **Virtualized Rendering** – Smooth performance even with thousands of images by rendering only visible items
✅ **Smart Image Loading** – Priority-based loading system that loads visible images first, then preloads adjacent images in the background
✅ **Seamless View Switching** – Toggle between single image view and gallery view with a single keypress (G key)
✅ **Scroll Wheel Navigation** – Navigate between images in single view using mouse scroll wheel
✅ **Session Persistence** – Remembers your view mode preference and automatically opens in the appropriate view

#### 🚀 Performance Improvements
✅ **Optimized Loading Pipeline** – Reduced initial gallery loading delay with priority queue system
✅ **Multi-threaded Thumbnail Loading** – Up to 16 parallel threads for faster thumbnail generation
✅ **Intelligent Caching** – Bucket-based thumbnail caching system for efficient memory usage
✅ **Background Preloading** – Continuously loads images in the background while you browse

#### 🎨 User Experience Enhancements
✅ **Title Bar Updates** – Gallery mode shows folder name in title bar instead of current file
✅ **Keyboard Shortcuts** – Press 'G' to toggle between gallery and single image view, 'Esc' to return to gallery from single view
✅ **Loading Indicators** – Visual feedback during image loading with progress tracking
✅ **Smooth Transitions** – Optimized view switching with minimal delay

### 🐛 Bug Fixes
- Fixed persistent loading message when returning to gallery view
- Fixed gallery loading stopping after ~228 images - now continues loading all images
- Fixed images not displaying in gallery view after loading
- Improved cache key matching for better thumbnail display
- Fixed widget visibility issues in gallery view

### 📦 Technical Details
- Implemented `JustifiedGallery` widget with virtualization support
- Added priority queue system for visible image loading
- Enhanced `ImageLoadTask` with detailed performance logging
- Improved cache management with bucket-based height matching
- Optimized thread pool configuration (16 threads, batch size 8)

### 🎯 What's Coming Next
Check out the "Upcoming Features" section in the README for a preview of:
- Histogram Display for exposure analysis
- Batch Operations for multiple image processing
- And more exciting features in development!

---

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






