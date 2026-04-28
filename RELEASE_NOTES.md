# RAWviewer Release Notes

## 🚀 Version 1.6.0 - Mac Trackpad Pinch-to-Zoom & Gallery Stability
**Release Date: April 28, 2026**

### 🎯 What's New
**Adds seamless pinch-to-zoom support for Mac trackpads, improves large-folder gallery behavior, keeps cacheless mode responsive, and polishes single-image UI controls.**

### ✨ Key Features

#### 🔍 Mac Trackpad Gestures
✅ **Pinch-to-Zoom** – Smoothly zoom in and out of images using the native two-finger pinch gesture on Mac trackpads.
✅ **Smart Cursor Anchoring** – Zooming naturally anchors to the current cursor position, matching the expected behavior of modern macOS applications like Apple Photos.
✅ **Smart Zoom Gesture** – Double-tap with two fingers to instantly toggle between "Fit to Window" and 100% zoom.
✅ **Boundary Limits** – Maximum zoom is capped at 100% to ensure smooth performance, and zooming out is neatly bounded to the "Fit to Window" size.
✅ **Live Percentage** – The status bar actively displays the precise zoom percentage in real-time as you pinch.

#### 🖼️ Gallery View Improvements
✅ **Single-Pass Gallery Rebuilds** – Removed duplicate gallery rebuild/cleanup work that could slow large folders and make thumbnail updates appear unstable.
✅ **EXIF-Aware Gallery Ordering** – Gallery can refresh ordering after background EXIF capture-time extraction, while keeping the UI responsive during the initial load.
✅ **Current Image Positioning** – Switching from single image view back to gallery now scrolls to the current image instead of starting from the beginning.
✅ **Gallery Total Count** – Gallery view now shows the total number of images in the bottom status bar.

#### 📊 Single-Image UI Fixes
✅ **Histogram Reliability** – Histogram visibility now resets correctly when moving to a new image while still staying disabled when no image is loaded.
✅ **Stable Info Button** – The bottom information button remains visible without shifting EXIF metadata text.

#### ⚡ Cacheless Mode Refinements
✅ **Reduced EXIF Re-Reads** – Orientation lookup now checks memory cache first to reduce repeated EXIF reads.
✅ **Cleaner Console Output** – Removed noisy EXIF debug prints and fixed stale cache API usage in preview orientation handling.

---

## 🚀 Version 1.5.5 - Cacheless Trial & Loading Optimizations
**Release Date: April 28, 2026**

### 🎯 What's New
**Introduces a cacheless-by-default trial mode, faster non-RAW thumbnail decoding, and startup safety cleanup for legacy cache folders.**

### ✨ Key Features

#### ⚡ Performance & Cache Behavior
✅ **Memory-Only Default Cache** – Persistent disk/SQLite cache is now disabled by default to prioritize clean trial behavior and reduce local cache side effects.
✅ **Optional Persistent Cache Toggle** – You can re-enable persistent cache with `RAWVIEWER_PERSISTENT_CACHE=1`.
✅ **Legacy Cache Auto-Cleanup on Startup** – In memory-only mode, old `~/.rawviewer_cache` is automatically removed once as a safety cleanup.

#### 🖼️ Image Loading Improvements
✅ **Faster Non-RAW Thumbnail Path** – Standard image thumbnails now use `QImageReader.setScaledSize()` for faster decode at target size.
✅ **Robust Embedded JPEG Fallback** – Improved fallback scanning logic for embedded previews when RAW decode paths fail.

#### 🧭 UX & Stability
✅ **Open Behavior Consistency** – Restored open-flow behavior for choosing gallery folder vs single image loading.
✅ **Histogram UX Guard** – Histogram is hidden/disabled when no image is loaded.
✅ **Background Cleanup Coverage** – Cache cleanup paths were expanded to include preview-related data for better consistency.

---

## 🚀 Version 12.2 - Orientation & Build Optimization
**Release Date: December 30, 2025**

### 🎯 What's New
**Definitive fixes for RAW orientation, gallery layout consistency, and build process reliability.**

This release addresses critical issues with RAW file rotation (especially Sony ARW), resolves pillarboxing in the gallery view, and ensures a seamless build process across Windows and macOS by synchronizing dependencies.

### ✨ Key Features

#### 📸 Perfect RAW Orientation
✅ **Sony ARW Fix** – Resolved inconsistencies where some Sony RAW files appeared sideways despite correct EXIF data.
✅ **Smart Rotation Logic** – Improved orientation handling for all RAW formats, ensuring images are always displayed upright without double-correction.
✅ **Self-Healing Cache** – Implemented logic to detect and re-process cached thumbnails if their orientation doesn't match the source file.

#### 🖼️ Gallery Improvements
✅ **No More Pillarboxes** – Fixed incorrect aspect ratio calculations that caused black bars (pillarboxing) in the gallery view for portrait-oriented RAW files.
✅ **Optimized Dimension Extraction** – Enhanced EXIF extraction logic with a fast `rawpy` fallback for accurate image dimensions.
✅ **BITMAP Thumbnail Safety** – Enforced size limits on BITMAP-format thumbnails to prevent cache corruption.

#### 🛠️ Build & Dependency Sync
✅ **Unified Dependencies** – Synchronized `requirements.txt`, `build.py`, and platform-specific build scripts (`build_windows.bat`, `build_macos.sh`).
✅ **Reliable Executables** – Ensured `qtawesome` and `pyqtgraph` are correctly bundled in all builds.

---

## 🚀 Version 1.2 - Performance Revolution
**Release Date: December 27, 2025**

### 🎯 What's New
**Massive performance overhaul, instant folder loading, and UI modernization.**

This release focuses on eliminating bottlenecks. Large folders now load instantly, gallery scrolling is buttery smooth, and navigation is faster than ever thanks to intelligent prefetching. We've also updated the UI with modern Material Design elements.

### ✨ Key Features

#### ⚡ Ultra-Fast Performance
✅ **Instant Folder Loading** – Replaced `os.walk` with `os.scandir` and optimized metadata gathering. Scanning 7,000+ images is now nearly instant (down from minutes).
✅ **Fixed "Long Loading Time"** – identifying and fixing a 60s freeze caused by missing cache metadata.
✅ **Smart Prefetching** – intelligently pre-loads the next 3 and previous 2 images in background threads, making single-image navigation instant.
✅ **Gallery Scroll Improvements** – Implemented "scroll speed detection". Fast scrolling now defers heavy image loading, keeping the UI responsive even in massive galleries.

#### 🎨 Modernized UI/UX
✅ **Material Design 3 Icons** – Updated the "Open Folder" button with a modern vector icon.
✅ **Window Size Grip** – Added a minimalistic resize grip to the bottom-right corner for easier window management.
✅ **Refined Loading Indicators** – Loading messages are now non-intrusive "toasts" at the bottom of the screen, instead of blocking overlays.

### 🐛 Bug Fixes
- **Fixed "Ghost Image" Bug**: Resolved an issue where an image from the previous folder would persist in the bottom-right corner after switching folders.
- **Fixed Persistent Loading Overlay**: The "Preparing gallery" message no longer gets stuck when switching folders.
- **Fixed Navigation Bugs**: Resolved issues where key navigation could get stuck or behave inconsistently.
- **Fixed Zoom/View Overlaps**: Corrected issues where single view and gallery view could overlap.

### 📦 Technical Details
- **Unified Caching**: All image processing now routes through a central thread-safe cache.
- **Virtualized Gallery**: The gallery widget now strictly recycles widgets, significantly transforming memory usage.
- **Robust Cleanup**: Implemented aggressive widget cleanup to prevent UI artifacts.

---

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
