# RAWviewer Release Notes

## 🚀 Version 2.3.0
**Release Date: June 6, 2026**

### 🎯 Focus overlay
- **Alignment fix (Canon EOS)**: Maker-note AF uses a center origin with **Y-up**; dashed boxes now line up with the image instead of appearing vertically flipped.
- **Broader maker AF**: Nikon NEF (`AFInfo2`, image-height fallback), Olympus ORF (`AFPointSelected`, `AFFocusArea` / `AFSelectedArea`), Panasonic RW2 (`AFPointPosition`, including decimal `/1024` form).
- **Docs**: README lists focus-overlay support by brand — **Fujifilm RAF**, **Hasselblad 3FR**, typical **Adobe DNG**, Pentax PEF, Samsung SRW, and Sigma X3F are documented as **not supported** in the current parser (JPEG/TIFF may still show CIPA `SubjectArea`).

### 🛠️ Gallery & navigation
- **Filter out composite DNG panoramas**: Handled composite DNG panorama files (e.g., Lightroom/Photoshop HDR panoramas) similarly to unsupported images by hiding them from the gallery and navigation lists entirely to prevent loading errors and improve performance.
- **Navigation prefetch & zoom**: Bidirectional embedded-JPEG prefetch (default radius 6), focus-anchor zoom on resolution upgrades, and more reliable Space / double-click zoom on RAW.
- **RAW ↔ JPEG workflow**: Selective cache invalidation and forced reload when toggling workflow so the current image updates immediately.

### 🔍 Semantic search & indexing
- **Gallery thumbnail reuse**: Semantic warm-up skips paths already in `ImageCache` preview/grid tiers (gallery-loaded thumbnails), avoiding duplicate RAW decode.
- **ONNX Runtime hardening**: Detect broken `onnxruntime` installs early with a clear reinstall message instead of failing every embedding silently.
- **Pixi lock**: Upgraded `pixi.lock` to v7 for reproducible `pixi install`.

### 🏗️ Build
- **Removed unused `mediapipe`** from Windows `build.py` dependencies (face detection uses YuNet ONNX).
- **macOS release zip**: `build_macos.sh` now produces `RAWviewer-v2.3.0-macOS.zip` with **`Start Here.txt`**, **`Install RAWviewer.command`** (Applications + quarantine fix), and **`Remove Quarantine.command`** (run from folder). Bundles **scipy** for GPS reverse geocoding; **`LSMinimumSystemVersion`** set to **13.0**; **pyexiv2** required in the build script.

### 🍎 macOS
- **Startup splash**: Dismisses automatically when the main window is ready (no extra click on macOS).
- **Gallery search crash (macOS 26+)**: Disables NSTextField automatic completion on the search field to avoid ViewBridge / `SPCompletionListServiceViewController` aborts under Qt 6.11.
- **Install docs**: README macOS version support table (13+ prebuilt; Pixi 14+ on Apple Silicon); simplified zip install flow via `Start Here.txt`.

---

## 🚀 Version 2.2
**Release Date: May 30, 2026**

Unified 2.2 release — search, gallery, film strip, frameless window polish, RAW loading consistency, indexing improvements, and macOS share behavior aligned with Qt6.

### 🎯 What's New
- **Search from single-image view**: Search button in single view; submitting a query switches to gallery with filtered results.
- **Fast single-file open**: Opening one image no longer waits for full folder scan and EXIF sort on large libraries.
- **Windows — Open with another app**: Native picker via `OpenAs_RunDLLW` / `SHOpenWithDialog` with `OAIF_EXEC` for Lightroom, Photoshop, etc., directly exposed in the bottom bar via the Share button (bypasses dropdown for instant editing selection).
- **macOS — Share (single image)**: Bottom-bar share in single view uses a **Qt menu** of `NSSharingService` targets (Mail, Messages, …). AirDrop is hidden from the menu by default; use Finder for reliable AirDrop (see `docs/macos-sharing-v21-v22.md`).
- **Experimental GPU single-image view**: Opt in with `RAWVIEWER_GPU_VIEW=1` for smoother zoom/pan on supported hardware (classic scroll area remains the default).
- **Consistent RAW color (fit ↔ zoom)**: Single-image RAW defaults to LibRaw half-res for fit view and full decode at 100% zoom (`RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1`), avoiding embedded-JPEG color snap. Gallery thumbnails still use fast embedded previews.
- **Unified EXIF dual-backend**: Metadata routes through `metadata_backend` — fast header reads for RAW, optional `pyexiv2` for JPEG/TIFF (`RAWVIEWER_EXIF_BACKEND=auto`).
- **Frameless window resize (Windows)**: Drag any window edge (including top strip and bottom-right grip) to resize; gallery scrollbar keeps original 24px / 6px padding with a non-layout overlay for right-edge grip.

### 🛠️ Fixes & improvements

**Search & gallery**
- **Search → gallery navigation**: Clicking a search result opens the correct image; film strip and arrow keys stay within filtered results.
- **Gallery refresh after EXIF sort**: Gallery auto-updates when background capture-time refinement completes.
- **Gallery thumbnails**: Click handler uses the widget’s current path so reordered/filtered grids navigate correctly.
- **Search panel UI**: Collapsing the search field no longer shifts nearby status-bar icons; fixed width jump when clearing the query.
- **Search indexing UX**: No flash of stale `Semantic/Face X/10` progress after search completes; session-aware index status.
- **Semantic indexing**: Skip duplicate RAW companion files when writing to the index; resolved progress bar resets by scaling progress between thumbnail warming (10%) and MobileCLIP neural pass (90%); prevented brief double-count displays by filtering duplicate companion files in start fallbacks.

**Film strip & rotation**
- **Film strip animation**: Smooth fade-in/out when revealing or dismissing the single-image thumbnail strip; extended bottom hot zone.
- **Film strip hover**: Tuned show delays (350ms / 120ms direct) with prefetch so the strip feels responsive without flicker.
- **Film strip sync**: Fixed phantom selection, recursion on hover, and sync to search-filtered lists instead of the full folder.
- **Rotation consistency**: Non-destructive rotation stays aligned across single view, film strip, and gallery after `R` or arrow navigation.

**Performance & image pipeline**
- **Butter-smooth gallery scroll**: Faster scroll-speed detection and throttled prefetch during fast scrolling.
- **Fast-open / EXIF refinement**: Parallel capture-time probing, gated gallery EXIF sort, viewport scroll anchor on manual sort, instant gallery button when sort cache is warm.
- **GPU navigation**: Smoother gallery→single transitions and priority full-resolution decodes; 27% zoom race fix.
- **CPU downscaling**: Replaced LANCZOS with HAMMING for thumbnail downscales (faster, cleaner edges on CPU).
- **GPU viewport scaling**: Experimental GPU view uses hardware-accelerated scaling where enabled.
- **GPU single-view navigation**: Arrow keys and film strip keep the previous frame visible until the next buffer is ready; prefetched preview/full caches paint immediately; thumbnail-only stages are skipped during in-folder navigation to reduce flicker.
- **GPU fit ↔ 100% zoom**: Space and double-click now reach true 100% on the first action on RAW (fixes stale fit-mode flag showing ~fit% until a second toggle); resolution upgrades no longer undo an active 100% zoom.
- **Resolution crossfade (optional)**: Smoother preview→full upgrades in single view (`RAWVIEWER_RESOLUTION_CROSSFADE_MS`, default 280; set `RAWVIEWER_DISABLE_CROSSFADE=1` to disable).
- **Idle display prefetch**: Neighbor images warm preview/full buffers while browsing (`RAWVIEWER_IDLE_DISPLAY_PREFETCH`, `RAWVIEWER_NAV_PRELOAD_*`).
- **Multi-core RAW postprocess**: Process pool for LibRaw when `RAWVIEWER_USE_PROCESS_POOL=1` (default on 4+ cores).
- **Progressive RAW load**: Optional embedded-first path via `RAWVIEWER_PROGRESSIVE_RAW_LOAD=1` (off by default).

**Platform & docs**
- **macOS share under Qt6**: Default dev/shipping path uses Qt menu + `performWithItems_`; picker fallback and AirDrop/Finder behavior documented in `docs/macos-sharing-v21-v22.md`.
- **Folder sort**: Production uses EXIF / probe / birth / mtime only; Windows Shell `DateTaken` POC removed.
- **`clear_cache.bat` / `clear_cache.sh`**: Full dev/session reset; repo-root `clear_cache.bat` forwards to `scripts/Launch/bat/clear_cache.bat`.
- **Windows share helper** (sources retained): .NET `WindowsShareHelper.exe` for WinRT share in dev builds.
- **Launch scripts**: macOS build/test workflow in `scripts/Launch/README.md`; version aligned across `build.py`, `pixi.toml`, and `QApplication`.
- **Environment**: `activation.env` with `PYTHONNOUSERSITE=1` to prevent global package leaks and splash issues.

---

## 🚀 Version 2.1.0
**Release Date: May 28, 2026**

🎯 What's New
- **Film strip (single-image view)**: Bottom thumbnail strip on pointer hover; dismisses when leaving the strip or entering the status bar.
- **Launch scripts**: Debug and build entry points moved to `scripts/Launch/` (`bat/` on Windows, `shell/` on macOS); root scripts forward for compatibility.
- **Semantic + face indexing (Windows)**: Phased indexing (metadata → MobileCLIP embeddings → background face backfill), resume from `semantic_index.db`, DirectML-accelerated ONNX on Windows when available.

🛠️ Fixes & improvements
- **Installer model download**: Added `requests` to Pixi dependencies so `huggingface_hub` can download MobileCLIP ONNX models on first install (no HF account required for public models).
- **Indexing stability**: Stronger RAW thumbnail fallbacks, skip permanently unindexable files, conservative face-scan warm-up defaults, clearer progress phases.
- **Delete confirmation dialog**: Centered on the main window using global coordinates.
- **Face detection threshold**: YuNet / SSD confidence raised to **0.75** (fewer false positives).
- **Logging**: Persistent logs under `%LOCALAPPDATA%\RAWviewer\logs\`; installer no longer bundles old `src/logs`.
- **Dependencies**: `pixi.toml` is the source of truth; legacy `requirements.txt` removed.
- **Docs**: Pixi-first build instructions, minimum macOS 13 for prebuilt releases.

Includes fixes from **2.0.1** (Pixel DNG, gallery aspect ratio, DNG single-view zoom).

---

## 🚀 Version 2.0.1
**Release Date: May 23, 2026**

🛠️ Fixes & improvements
- **Google Pixel DNG Support**: Fixed critical crashes in the `QImageReader` and `EXIFExtractor` fallbacks that prevented Google Pixel DNG files from rendering on macOS.
- **Gallery Aspect Ratio Fix**: Fixed a bug where thumbnail crops were improperly bypassed, ensuring that all gallery tiles now correctly display cropped square previews without distorted aspect ratios or zoomed-in glitches.
- **DNG Single-View Zoom Stability**: Reworked DNG single-image loading to use a full-resolution-first path and tightened pending zoom-state handling, fixing intermittent cases where Space / double-click changed zoom status text without actually zooming the image.

---

## 🚀 Version 2.0.0
**Release Date: May 7, 2026**

🎯 What's New
- **Local Semantic Search**: Cross-platform natural-language gallery search. Harness the power of MobileCLIP (Core ML on macOS, ONNX on Windows) to rank images by meaning (e.g., "sunny landscape" or "vintage portrait").
- **Structured Metadata Filters**: Powerful new query syntax to narrow by `camera:`, `lens:`, `iso:`, `ext:`, and more.
- **Slideshow Mode**: Automatic hands-free playback of your photos with adjustable intervals.
- **macOS Native Share**: Integration with the native macOS share sheet for instant sending via Mail, AirDrop, or Messages.
- **High-Fidelity Rendering**: New LANCZOS resampling and 2x JPEG oversampling for razor-sharp display on 4K and Retina screens.
- **Native macOS & Windows Shell Integration**: Improved Windows shell verbs and deep Finder/Explorer compatibility.
- **Non-Destructive Rotation**: Instantly rotate any image (including RAW) by 90° steps visually without modifying the original file.
- **Massive Location Intelligence**: Added ~150+ world cities to the GPS contradiction filter and improved multi-word place detection (e.g., "Hong Kong").
- **Precision Focus Overlays**: Added focus point visualization using MakerNote data for Canon, Nikon, and Sony.

🛠️ Fixes & improvements
- **High-Quality RAW Fallback**: Automatically triggers high-quality "fast RAW decode" for files with poor-quality embedded previews.
- **Performance Hardening**: Refactored `UnifiedImageProcessor` to open RAW files exactly once, drastically reducing Disk I/O.

### ⌨️ Keyboard & Gesture Map
- **Space / Double-click**: Toggle between "Fit to Window" and 100% zoom.
- **Pinch-to-Zoom**: Smoothly zoom in/out via trackpad or Ctrl+Scroll Wheel.
- **Left / Right Arrow**: Navigate between images (preserves zoom level).
- **Down Arrow**: Move current image to "Discard" folder.
- **Delete**: Remove the current image.
- **H / F**: Toggle Histogram / Focus Subject outlines.
