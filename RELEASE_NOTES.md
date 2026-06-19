# RAWviewer Release Notes

## 🚀 Version 2.4
**Release Date: June 19, 2026**

### Highlights — new in 2.4

- **RAWviewer Lite** — A lighter edition for fast browsing and culling: smaller download, no AI model install, and gallery search by camera, ISO, date, GPS, and filename. On **Windows**, pick **Lite** in the installer; on **Mac**, use **`RAWviewer-v2.4-macOS-Lite.zip`**.
- **One Windows installer** — **`RAWviewer_Setup.exe`** is all you need. In the wizard, choose **Full (CUDA)**, **Full (DirectML)**, or **Lite** — no more hunting for separate setup files.
- **Drag photos out** — Drag a gallery thumbnail or a film-strip thumb to Explorer, Finder, Mail, WhatsApp (where supported), Lightroom, or any app that accepts files. Select several gallery photos first and drag once to export them all.
- **Composition guides** — Press **G** in single-image view to cycle overlays: rule of thirds, diagonals, or phi grid — handy for checking framing while you cull.
- **Bookmarks** — Star keepers per folder; filter the gallery to bookmarked shots only; share or slideshow your picks; works with gallery search (see below).
- **Release highlights image** — The GitHub release includes a new visual summary of v2.4 so you can see what's new at a glance before you download.

### Introducing RAWviewer Lite

**Lite is RAWviewer without the AI search engine** — built for photographers who want the same fast viewer, gallery, and culling workflow, but prefer a smaller install and snappier loading over plain-language search.

**What you keep (same as Full):**
- Open folders of RAW and JPEG, gallery grid, film strip, fit / 100% zoom, histogram, focus overlay, composition guides (**G**)
- Drag photos out to Explorer, Finder, Mail, or your editor
- Bookmarks, Discard folder, keyboard shortcuts
- Gallery search by **metadata** — camera, lens, ISO, date, city, filename, and more (e.g. `camera:sony iso<800`)

**What Lite leaves out:**
- **Semantic (AI) search** — you can't type `sunset on beach` and have the app guess meaning; use metadata filters or your eyes instead
- **Face-based filters** — no `has:face` / `people` style queries
- **AI model download** — no ~600 MB (Windows) or ~150 MB (Mac) model install

**Why Lite feels faster and uses less disk:**
- No background AI indexing eating CPU, GPU, and RAM while you browse
- Tuned prefetch so the next photos in the gallery and film strip load sooner when you scroll or arrow through a folder
- Smaller app footprint (~500 MB install vs ~1.5 GB+ for Full after models)

**How to get Lite:** On **Windows**, choose **Lite** in **`RAWviewer_Setup.exe`**. On **Mac**, download **`RAWviewer-v2.4-macOS-Lite.zip`**. Already on Full? You can install Lite side by side — they don't share the same shortcut name.

Not sure which to pick? **Lite** if you cull by eye and search by camera/date/location. **Full** if you want to describe photos in everyday words or filter by faces.

### Bookmarks

- **Mark keepers per folder** — Press **↑** to toggle. In **single view**, click the **star** on the bottom-right bar too.
- **Single-image view** — White star = bookmarked; outline star = not bookmarked. Click the star or press **↑** to toggle either way.
- **Gallery** — Star badges on thumbnails and in the film strip. **↑** toggles the current selection; with multiple thumbnails selected (**Ctrl/Cmd+click**, **Shift+click**), **↑** or the bottom **star** toggles all selected photos at once.
- **Bookmark-only gallery** — In gallery view, click the **outline star** (nothing selected) to show bookmarked photos only (star turns **gold**). Click again or press **Esc** to return to the full grid. With thumbnails selected, the star toggles bookmarks (same as **↑**).
- **Search + bookmarks** — Search still runs on the whole folder; the bookmark filter narrows what you **see**. Turn off the filter to see all search matches; clear search with the filter still on to see every bookmark.
- **Share / Open** — In **single view**, opens or shares the current photo. In **gallery**, the button appears when you multi-select photos **or** when bookmark filter is on. With filter on and nothing selected, it targets **all visible bookmarked** photos.
- **Slideshow** — Play button in gallery and single view. In bookmark-filter mode, slideshow cycles **bookmarked photos only**.
- **Esc order (gallery)** — Clears multi-select first, then exits bookmark filter (before leaving gallery from single view).
- **Persistence** — Bookmarks are saved per folder locally and restored when you reopen that folder.

### Improvements & fixes

- **Portrait photos look right** — Gallery thumbnails and the film strip no longer show vertical shots sideways or "twisted twice."
- **Smoother on huge folders (especially Mac)** — Large libraries scroll more reliably; the app eases off when your Mac is busy instead of freezing.
- **Gallery multi-select** — **Ctrl+click** (Windows) or **Cmd+click** (Mac) to pick photos; **Shift+click** for a range (follows visible gallery order, including search and bookmark filter). Works with drag-out and the bottom **Share / Open** bar.

### Gallery & everyday workflow

- **Multi-select** — Ctrl/Cmd+click toggles a thumbnail on or off; Shift+click selects a continuous range (great for picking a burst or a run of keepers).
- **Drag out** — One thumbnail drags one file; a multi-selection drags every selected file together.
- **No accidental re-open** — Dragging a photo from RAWviewer and dropping it back on the window no longer opens it again.
- **Search while indexing** — Progress messages stay clearer; Lite won't pretend AI search is ready when only metadata has finished indexing.

### Portrait orientation

- Vertical RAW and JPEG shots appear upright in the gallery grid and film strip.
- Cached thumbnails from older versions are refreshed when needed — run **`clear_cache`** once if you still see old sideways previews.

### Large libraries (Mac)

- Better behavior when a folder has thousands of photos — less stutter, fewer "system busy" moments.
- Background search/indexing pauses while you scroll the gallery and picks up again when you stop.
- On 8 GB Macs, the app stays conservative; on 16 GB+ machines it can work a bit harder automatically.

### Drag & drop

- **Gallery**: Drag one thumbnail, or drag while several are selected to export all of them.
- **Filmstrip**: Drag the active strip thumbnail the same way.
- **Drop in** (unchanged): Drag a folder or image onto the window to open it in RAWviewer.

### Install & uninstall

- **Windows install**: Unified **`RAWviewer_Setup.exe`** — choose **Full (CUDA / DirectML)** or **Lite** in the wizard; launch **`RAWviewer.exe`** (not Setup) after install. Default folder: `%LOCALAPPDATA%\RAWviewer`.
- **Windows uninstall**: **Settings → Apps → RAWviewer → Uninstall**, or **`uninstall.bat`** in the install folder. Removes the app, **`%USERPROFILE%\.rawviewer_cache`**, and **`%LOCALAPPDATA%\RAWviewer`** logs/tiles. Set **`RAWVIEWER_UNINSTALL_FULL=1`** before **`uninstall.bat`** to also clear saved preferences.
- **macOS install**: Extract release zip → **`bash install_macos_app.sh`** (see **`Start Here.txt`** in the zip). Copies the app to **Applications** and clears download quarantine.
- **macOS uninstall**: Release zips include **`uninstall_macos_app.sh`** and **`Uninstall RAWviewer.command`** — removes **Applications** copies, **`~/.rawviewer_cache`**, logs, and preferences. Keep the zip (or re-download) to run uninstall later; Trash alone does not clear cache.
- **Documentation**: README, **`Start Here.txt`**, and **`scripts/Launch/README.md`** updated with install/uninstall steps and an uninstall-vs-**`clear_cache`** table for both platforms.

---

## 🚀 Version 2.3.2
**Release Date: June 8, 2026**

Installer and in-app MobileCLIP download UX, gallery bottom-bar layout, and clearer Hugging Face messaging.

### 🪟 Windows installer
- **Live model download progress**: The ~600 MB Hugging Face download maps to **5–100%** on the installer progress bar (no longer stuck at 75%).
- **Installer log + label**: Progress appears as **`Downloading... N%`** in the step label and install log (every 5%); ASCII-only text avoids console encoding glitches on Windows.
- **Cleaner subprocess output**: Silent Hugging Face tqdm bars; only `@RAWVIEWER_PROGRESS` lines drive the UI (`python -u`, `PYTHONUNBUFFERED`).
- **Byte-level reporting**: `download_mobileclip_onnx.py` streams progress parsed by the bootstrap installer.
- **Optional faster downloads**: Set a **`HF_TOKEN`** environment variable before setup if you use a [Hugging Face](https://huggingface.co/) account (inherits into the download subprocess).

### 🔍 Semantic search (all platforms)
- **In-app download progress in search field**: After you click **Download**, the gallery search bar shows **`Downloading... N%`** (same place as Metadata/Semantic/Face indexing). The prompt dialog closes immediately so the search field stays visible.
- **macOS Core ML downloads**: Per-file `hf_hub_download` with byte progress instead of opaque `snapshot_download`.
- **Shared progress helpers**: New `mobileclip_download_progress.py` used by installer scripts, bootstrap, and the main app.

### 🛠️ Gallery UI
- **Bottom bar button order**: **RAW/JPEG workflow** toggle now sits **before** the search icon so the search field expands right next to the search button (not with RAW sandwiched in between).

### 📖 Documentation
- README and release notes clarify models come from Hugging Face (~600 MB on Windows), installer progress behavior, and that first download may take longer without a Hugging Face account.

---

## 🚀 Version 2.3.1
**Release Date: June 7, 2026**

Focus overlays, RAW ↔ embedded-JPEG workflow, faster gallery indexing, background update checks, Windows installer stability, and macOS release packaging (Terminal install, in-app MobileCLIP download, SSL fix).

### 🔔 Release updates
- **Background check on launch**: Once per app start, RAWviewer quietly compares your version to the latest [GitHub release](https://github.com/markyip/RAWviewer/releases/latest) (offline or unreachable → no UI).
- **MD3 update prompt**: When a newer release is available, a styled dialog shows your version vs the latest tag and offers **Open Download Page** or **Not Now**.
- **Respectful snooze**: **Not Now** hides the prompt for **14 days** for that release; a **newer** release tag will notify you again sooner.
- **Opt out**: Set `RAWVIEWER_SKIP_UPDATE_CHECK=1` to disable the check entirely.

### 🎯 Focus overlay (`F`)
- **Broader maker AF**: Nikon NEF (`AFInfo2`, image-height fallback), Olympus ORF (`AFPointSelected`, `AFFocusArea` / `AFSelectedArea`), Panasonic RW2 (`AFPointPosition`, including decimal `/1024` form), and refined Canon EOS point placement (center origin, Y-up).
- **Brand guide**: README documents which formats support maker-note AF vs CIPA `SubjectArea` only (e.g. Fujifilm RAF, Hasselblad 3FR, typical Adobe DNG, Pentax PEF, Samsung SRW, Sigma X3F).

### 🖼️ Viewing
- **RAW ↔ JPEG workflow (single view)**: One-click toggle on the bottom bar — **embedded JPEG** (fast) vs **full RAW decode** (high quality); switching clears display caches and reloads the current file. Shown in single view only, not in gallery grid.
- **Snappier RAW navigation**: Bidirectional **embedded-JPEG prefetch** (default radius **6**), **focus-anchored zoom** when upgrading resolution.
- **RAW zoom fix**: **Space** and **double-click** reach 100% zoom reliably on RAW when fit-to-window state was out of sync (Ctrl+scroll already worked).

### 🛠️ Gallery & navigation
- **Cleaner libraries**: Composite DNG panoramas (e.g. Lightroom/Photoshop HDR stitches) are hidden from the gallery and navigation lists.
- **Scroll-friendly indexing**: Background metadata and semantic indexing **pause while you scroll** the gallery and resume after idle, keeping large folders responsive.

### 🔍 Semantic search & indexing
- **Gallery thumbnail reuse**: Semantic warm-up reuses paths already in `ImageCache` (preview/grid tiers from loaded tiles), cutting duplicate RAW decodes during indexing.
- **Faster neural pass**: Auto-tuned MobileCLIP **batch size** on your GPU/CPU for higher indexing throughput.
- **Safer setup**: Incomplete ONNX installs report a clear reinstall hint instead of failing silently.

### 🪟 Windows installer & launch
- **Clearer release filenames**: **`RAWviewer_Setup_DirectML.exe`** (recommended) and **`RAWviewer_Setup_CUDA.exe`** replace the old single-file names.
- **Dedicated app launcher**: Install folder **`RAWviewer.exe`** is a small stub that starts the app; **`RAWviewer_Setup.exe`** in the same folder is for repair/reinstall only.
- **More reliable setup**: Pixi and MobileCLIP downloads retry on failure with clearer network/disk/proxy errors; canceling setup removes incomplete install folders; welcome page text simplified (no Ctrl+Shift+O / disk-space hints).
- **AI model install progress**: During the ~600 MB Hugging Face download, the installer progress bar and log show real transfer progress (no longer stuck at 75%).
- **MobileCLIP optional at install**: If AI models fail during setup, browsing still works; download them later from gallery **Search** (MD3 prompt + in-dialog progress, same style as macOS). Models are hosted on Hugging Face; without a Hugging Face account, the first download may take longer.
- **Uninstall fixes**: Settings → Apps and `uninstall.bat` work on Win11; a confirmation message appears when removal finishes.
- **Shortcuts button**: Status-bar **i** opens the keyboard-shortcuts dialog (not just a tooltip).

### 🍎 macOS
- **Release zip**: `RAWviewer-v2.3.1-macOS.zip` (~82 MB); minimum macOS **13.0**; bundles **scipy** (GPS reverse geocoding) and **pyexiv2**. MobileCLIP models are **not** in the zip.
- **Terminal install**: Extract the zip, then `bash install_macos_app.sh` in the extracted folder (see README).
- **In-app model download**: When the user opens **gallery search**, the app prompts to download MobileCLIP from Hugging Face (~150 MB on macOS, one-time, needs internet). Without a Hugging Face account, that download may take longer. No download prompt on app startup.
- **HTTPS / SSL fix**: Packaged app bundles **certifi** CA certificates and configures SSL before Hugging Face / tokenizer downloads, fixing `[SSL: CERTIFICATE_VERIFY_FAILED]` on fresh installs.
- **Dock — single app icon**: LibRaw process pool **off by default** (PyInstaller runtime hook + spawn-safe startup); opt in with `RAWVIEWER_USE_PROCESS_POOL=1` (may bring back extra Dock entries).
- **Startup splash**: Dismisses automatically when the main window is ready (no extra click).
- **Gallery search crash (macOS 26+)**: NSTextField autocomplete disabled on the search field (including after focus and wake-from-sleep) to avoid ViewBridge / `SPCompletionListServiceViewController` aborts under Qt 6.11.

### 🏗️ Build
- **Removed unused `mediapipe`** from Windows `build.py` dependencies (face detection uses YuNet ONNX).
- **macOS SSL bundling**: PyInstaller collects **certifi** and runs an SSL runtime hook for HTTPS downloads in the packaged app.

### 📄 Docs
- README: focus-overlay brand guide, macOS version support table, Setup vs launcher exe, DirectML recommendation, Terminal install, gallery-search model download, uninstall, and troubleshooting.

---

## 🚀 Version 2.2
**Release Date: May 30, 2026**

Unified 2.2 release — search, gallery, film strip, frameless window polish, RAW loading consistency, indexing improvements, and macOS share behavior aligned with Qt6.

### 🎯 What's New
- **Search from single-image view**: Search button in single view; submitting a query switches to gallery with filtered results.
- **Fast single-file open**: Opening one image no longer waits for full folder scan and EXIF sort on large libraries.
- **Windows — Open with another app**: Native picker via `OpenAs_RunDLLW` / `SHOpenWithDialog` with `OAIF_EXEC` for Lightroom, Photoshop, etc., exposed through the bottom external-app button.
- **macOS — Share (single image)**: Bottom-bar share in single view uses a **Qt menu** of `NSSharingService` targets (Mail, Messages, …). AirDrop is hidden from the menu by default; use Finder for reliable AirDrop (see `docs/macos-sharing-v21-v22.md`).
- **Experimental GPU single-image view**: Opt in with `RAWVIEWER_GPU_VIEW=1` for smoother zoom/pan on supported hardware (classic scroll area remains the default).
- **Consistent RAW color (fit ↔ zoom)**: Single-image RAW defaults to LibRaw half-res for fit view and full decode at 100% zoom (`RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1`), avoiding embedded-JPEG color snap. Gallery thumbnails still use fast embedded previews.
- **Unified EXIF dual-backend**: Metadata routes through `metadata_backend` — fast header reads for RAW, optional `pyexiv2` for JPEG/TIFF (`RAWVIEWER_EXIF_BACKEND=auto`).
- **Frameless window resize (Windows)**: Drag any window edge (including top strip and bottom-right grip) to resize; gallery scrollbar keeps original 24px / 6px padding with a non-layout overlay for right-edge grip.

### 🛠️ Fixes & improvements

**Search & gallery**
- **Search → gallery navigation**: Clicking a search result opens the correct image; film strip and arrow keys stay within filtered results.
- **Gallery refresh after EXIF sort**: Gallery auto-updates when background capture-time refinement completes.
- **Gallery thumbnails**: Click handler uses the widget's current path so reordered/filtered grids navigate correctly.
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
- **Installer model download**: Added `requests` to Pixi dependencies so `huggingface_hub` can download MobileCLIP ONNX models on first install (public models on Hugging Face).
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
