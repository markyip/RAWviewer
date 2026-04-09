# RAWviewer Release Notes

## Version: v1.5.4 (Current)
**Release Date: April 9, 2026**

### 🎯 What's New
**Gallery reliability, safer navigation when loading folders, RAW preview fallback when LibRaw fails, and clear errors when nothing can be decoded.**

### 🛠️ Fixes & improvements
- **Gallery first layout**: If the scroll viewport has **zero width** during the first layout pass, the justified grid **no longer clears to blank**; it retries after a short delay until geometry is valid.
- **Folder scan / loading UI**: "Scanning folder…" and related loading toasts are **cleared reliably** when the scan finishes; gallery loading errors surface a short **Gallery Error** message when `set_images` fails.
- **Same-file loads**: In-flight decode for the **current path** is no longer cancelled by a duplicate load request (fixes intermittent **blank or stuck** image after changing folders or restoring session).
- **Stale thumbnails**: A **late, lower-resolution** image from the load manager no longer **replaces** an already displayed higher-resolution frame (avoids visual flicker and related logging/UI edge cases).
- **RAW fallback**: When **LibRaw** cannot open or postprocess a RAW file, the app attempts to extract an **embedded JPEG** by scanning the file (thumbnails and preview paths; larger limit when full resolution is requested).
- **Decode failure feedback**: If the **full** decode stage returns nothing for the **current** image (unsupported RAW, corrupt file, or no usable embedded preview), a **Load Error** dialog is shown instead of a silent blank view.

### 📦 Technical
- `extract_embedded_jpeg_by_scan` in `enhanced_raw_processor.py`; optional retry from `unified_image_processor` after RAW processing exceptions.
- `ImageLoadManager` emits `error_occurred` when `process_full_image` yields no result for a current-file, full-stage task.

---

## Version: v1.5.3
**Release Date: April 9, 2026**

### 🎯 What's New
**Gallery keyboard behavior fix: single-image shortcuts no longer run while you are in gallery mode.**

### 🛠️ Fixes
- **View-mode-aware shortcuts**: In **gallery** mode, **↓ / ↑** scroll the thumbnail grid instead of triggering discard or single-image logic. **Space**, **← / →**, **Delete**, and discard (**↓** in single view) apply only in **single-image** view, so the app does not act on a stale `current_file_path` when the grid has focus or the main window still holds focus after switching views.

---

## Previous Releases

## Version: v1.5.2
**Release Date: April 7, 2026**

### 🎯 What's New
**Single-image histogram overlay, full-width viewing, and a compact shortcuts hint on the status bar.**

### ✨ Key Features

#### 📊 Histogram (single-image view)
- **RGB and luminance curves** on a small **16:9** card with a **semi-transparent** background.
- **Overlay layout**: The image area uses the **full width** of the window; the histogram floats on top instead of consuming a side column.
- **Draggable card**: Click and drag to move the histogram within the view; position is kept while resizing (clamped to the visible area).
- **Shortcut `H`**: Toggle the histogram on or off (single-image mode only).

#### ℹ️ Status bar hint
- **`i` control** on the bottom bar: hover to see a **tooltip** with the main keyboard shortcuts (no click dialog).
- **Menu unchanged**: **Keyboard Shortcuts** in the menu bar still opens the full list dialog; tooltip content stays in sync via a shared help string.

### 📦 Technical
- New module `src/image_histogram.py` for sampling, binning, and painting the overlay widget.
- `SingleImageViewOverlay` container: full-bleed `QScrollArea` plus stacked histogram widget.

---

## Version: v1.5.1

### 🚀 Stability & Build Update
This update focuses on eliminating critical crashes during gallery navigation and hardening the macOS build process for reliable app bundle creation.

### 🛠️ Gallery & UI Stability
- **Fixed High-Speed Scrolling Crash**: Resolved a frequent `RuntimeError` where background threads attempted to communicate with a deleted `ImageLoadManager`. The manager is now formally anchored to the `QApplication` lifecycle.
- **Hardened Signal Emitters**: Added safety checks (using `PyQt6.sip.isdeleted`) to all background loading tasks to ensure they only emit signals if the manager is still active.
- **Fixed Transition NameError**: Resolved a crash that occurred when switching from single image view to gallery view due to an undefined variable in the prefetch logic.

### 📦 Build System & macOS Packaging
- **Hermetic Build Process**: Refactored `build.py` to use `sys.executable` for all child processes (`pip`, `PyInstaller`). This ensures the build system always uses the activated virtual environment, preventing conflicts with Homebrew or system Python.
- **Robust App Bundling**: Improved the macOS `.app` bundle structure by explicitly adding the `PkgInfo` signature and performing a deep ad-hoc code sign.
- **Fixed "Forbidden" Icon Issue**: Resolved the circle-slash symbol on the application icon by clearing security quarantine attributes and ensuring proper bundle metadata.

---

## 🚀 Version 1.5 - Gallery Smoothness & Windows EXE Stability
**Release Date: March 18, 2026**

### 🎯 What's New
**A major gallery smoothness upgrade with visible-first loading, plus improved stability for Windows onefile executables.**

This release focuses on large-folder browsing: prioritize thumbnails in the visible viewport, reduce main-thread pressure while scrolling, and improve Windows onefile EXE compatibility and runtime stability.

### ✨ Key Features

#### 🖼️ Gallery Improvements
✅ **Pixel-level smooth scrolling** – Improved wheel scrolling feel to reduce stutter and jitter  
✅ **Visible-first loading + smart prefetch** – Load viewport thumbnails first, then prefetch after scroll settles to avoid wasted work  
✅ **Non-blocking initialization** – Defer heavy gallery setup to the event loop to keep view switching responsive  
✅ **New gallery module packaging** – Introduced the `rawviewer_ui` package to isolate the optimized gallery implementation from legacy UI code

#### 🪟 Windows EXE Packaging Stability
✅ **More stable multiprocessing initialization** – Improved Windows onefile subprocess startup to reduce unexpected runtime behavior  
✅ **Thumbnail output compatibility** – `thumbnail_ready` can now display `QImage` thumbnails reliably, improving pipeline consistency  
✅ **More reliable PyInstaller analysis** – Updated build arguments to ensure UI modules and hidden imports are collected correctly in onefile builds

### 📦 Technical Details
- **Robust path matching**: normalize Windows paths for reliable async-result comparisons
- **Better EXIF cache concurrency**: use thread-local SQLite connections and WAL to reduce contention
- **More precise thumbnail loading**: request work by stage (e.g., `thumbnail`) to avoid unnecessary processing during gallery browsing

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
