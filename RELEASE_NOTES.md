# RAWviewer Release Notes

## 🚀 Version 1.6.1
**Release Date: April 28, 2026**
🎯 What's New
- macOS Native Integration: The application can now be set as the default viewer for images and RAW files on macOS. Added support for handling `FileOpen` events to instantly display files opened from Finder.

## 🚀 Version 1.6.0
**Release Date: April 28, 2026**
🎯 What's New
- Seamless pinch-to-zoom support for Mac and Windows trackpads, improved large-folder gallery behavior, a cacheless-by-default trial mode, faster non-RAW thumbnail decoding, and polished single-image UI controls.

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
