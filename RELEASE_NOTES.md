# RAWviewer Release Notes

## 🚀 Version 1.7.0
**Release Date: April 30, 2026**

### What's new
- **Windows UI**: The in-window menu bar (File / Keyboard Shortcuts) is hidden on Windows frameless builds; menu shortcuts remain active via the main window, and the bottom-bar **i** tooltip still lists keyboard help.
- **Windows — no Share button**: The bottom-bar Share control is removed on Windows only. macOS keeps the native share sheet from the same control.
- **Status bar layout**: Share (macOS only), shortcuts hint (`i`), and image counter are grouped with consistent spacing; metadata centering uses the same trailing width math as before.
- **Keyboard shortcuts help**: Tooltip and shortcuts dialog list only keyboard/trackpad actions; slideshow and rotate remain on the bottom bar (not listed as hotkeys).

### Fixes & improvements
- **Windows shell sharing**: Removed legacy `ShellExecute` / `ShellExecuteEx` use of the Explorer **`share`** verb, which often triggered “no application is associated with this file” for common types (e.g. JPEG). Any remaining share-related code path defers to the next event-loop tick, tries Explorer COM verbs when available, then falls back to **file clipboard** (`CF_HDROP`), **PowerShell `Set-Clipboard -LiteralPath`**, and finally path text.
- **Rotate behavior**: Rotate now applies a non-destructive visual 90° step in the viewer (including RAW), without writing EXIF/pixels back to disk.
- **Gallery rotation sync**: Gallery uses viewport-time thumbnail rotation with rotation-aware cache keys, and visible tiles for the active file refresh immediately after a rotate action.
- **Build**: `requirements.txt`, `build.py`, and platform build scripts include optional `pyobjc-framework-Cocoa` (macOS) and `pywin32` (Windows) with matching PyInstaller hidden imports so packaged apps include the APIs needed for macOS Share and other shell features.
- **Gallery mode chrome**: While in gallery view, the gallery toggle, share, slideshow, rotate, and single-image metadata counter behave as expected (only sort + total count + essentials stay visible; return to a photo via the grid).
- **Gallery → single loading**: Fixed cases where the “Loading image” overlay could remain after picking a thumbnail when the image was served from memory cache.
- **Empty-state copy**: Onboarding text refers to bottom-bar controls generically instead of naming slideshow/rotate as keys.

---

## 🚀 Version 1.6.0
**Release Date: April 28, 2026**
🎯 What's New
- macOS Native Integration: The application can now be set as the default viewer for images and RAW files on macOS. Added support for handling `FileOpen` events to instantly display files opened from Finder.
- Seamless pinch-to-zoom support for Mac and Windows trackpads, improved large-folder gallery behavior, a cacheless-by-default trial mode, faster non-RAW thumbnail decoding, and polished single-image UI controls.

> [!IMPORTANT]
> **macOS First Launch Note**: If you see a warning about "Malware" when using *Open With* for the first time, simply Right-click the app in your Applications folder and choose **Open**. This registers the app with macOS, after which *Open With* will work normally.

🛠️ Fixes & improvements
- Pinch-to-Zoom: Smoothly zoom in and out of images using the native two-finger pinch gesture on Mac trackpads, and precision touchpads on Windows (or via Ctrl+Scroll Wheel).
- Smart Cursor Anchoring: Zooming naturally anchors to the current cursor position, matching the expected behavior of modern macOS applications like Apple Photos.
- Smart Zoom Gesture: Double-tap with two fingers to instantly toggle between "Fit to Window" and 100% zoom.
- Boundary Limits: Maximum zoom is capped at 100% to ensure smooth performance, and zooming out is neatly bounded to the "Fit to Window" size.
- Live Percentage: The status bar actively displays the precise zoom percentage in real-time as you pinch.
- Memory-Only Default Cache: Persistent disk/SQLite cache is now disabled by default to prioritize clean trial behavior and reduce local cache side effects.
- Optional Persistent Cache Toggle: You can re-enable persistent cache with `RAWVIEWER_PERSISTENT_CACHE=1`.
- Legacy Cache Auto-Cleanup on Startup: In memory-only mode, old `~/.rawviewer_cache` is automatically removed once as a safety cleanup.
- Reduced EXIF Re-Reads: Orientation lookup now checks memory cache first to reduce repeated EXIF reads.
- Cleaner Console Output: Removed noisy EXIF debug prints and fixed stale cache API usage in preview orientation handling.
- Faster Non-RAW Thumbnail Path: Standard image thumbnails now use `QImageReader.setScaledSize()` for faster decode at target size.
- Robust Embedded JPEG Fallback: Improved fallback scanning logic for embedded previews when RAW decode paths fail.
- Single-Pass Gallery Rebuilds: Removed duplicate gallery rebuild/cleanup work that could slow large folders and make thumbnail updates appear unstable.
- EXIF-Aware Gallery Ordering: Gallery can refresh ordering after background EXIF capture-time extraction, while keeping the UI responsive during the initial load.
- Current Image Positioning: Switching from single image view back to gallery now scrolls to the current image instead of starting from the beginning.
- Gallery Total Count: Gallery view now shows the total number of images in the bottom status bar.
- Open Behavior Consistency: Restored open-flow behavior for choosing gallery folder vs single image loading.
- Histogram UX Guard: Histogram visibility now resets correctly when moving to a new image while still staying disabled when no image is loaded.
- Stable Info Button: The bottom information button remains visible without shifting EXIF metadata text.
- Background Cleanup Coverage: Cache cleanup paths were expanded to include preview-related data for better consistency.

---

**Thank you for using RAWviewer!** 📸
