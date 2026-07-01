# RAWviewer Release Notes

## 🚀 Version 2.5.0
**Release Date: June 29, 2026**

Major release introducing a custom gallery zoom slider, interactive GPS map display overlay, macOS HDR/EDR viewing (including RAW), animated GIF/WebP playback, RAW tone recovery preview, burst image grouping, a side-by-side comparison pane, and gallery responsiveness fixes for large folders and folder switching.

### Burst Image Grouping
- **Burst image grouping** — Automatically group rapid-fire burst sequences in the gallery (based on capture time intervals). Double-click a collapsed stack cover to enter the burst group view and inspect individual frames.

### Dual-Pane Compare View
- **Dual-pane Compare Mode** — Compare multiple selected images side-by-side (left panel is the selected anchor, right panel displays candidate files).
- **Direct Compare Toggle (C)** — Press **C** to enter Compare mode immediately when multiple images are selected (works from both gallery and single view). Press **C** again while in Compare mode to exit back to the gallery.
- **Compare navigation & culling** — Promote candidate to select (↑), reject candidate to Discard folder (↓), or delete candidate to Recycle Bin/Trash (Delete). Use **Shift + ↓** or **Shift + Delete** to reject or delete the selected (left pane) anchor instead, which automatically promotes the current candidate to the select slot and loads the next image. Supports synchronizing composition guides (G) and clipping overlays (J) on both panes. Unselecting all items automatically hides the compare bottom button.

### RAW tone recovery & clipping (single view, RAW/DNG)
- **P — Recovery preview** — Session-only local shadow/highlight recovery on a half-res linear LibRaw decode (`highlight_mode=Reconstruct`, Reinhard tone map, local polish). Fit-only (~2048 px); press **P** again to exit. Does not change the main full-res pipeline or persist settings.
- **J — Clipping overlay** — Red = highlight clip (any channel ≥252); blue = shadow clip (all channels ≤3) on the current screen buffer. Diagnostic only; distinct from **P** soft recovery.

### Interactive GPS Map Display Overlay
- **GPS Map Overlay** — Press **M** in single-image view to show/hide an interactive tile-based map card. The card appears immediately with a **Loading map…** placeholder while tiles fetch; non-geotagged photos do not pop up a map.
- **Coordinate badge** — A clickable badge shows lat/lon on the map; click it to open the location in **Google Maps** (browser).
- **Offline Reverse-Geocoding for Search** — Bundles preloaded worldwide cities (all cities with population > 500, covering over 100,000+ locations) and landmarks databases (`cities500.csv.gz` and `landmarks.csv.gz`). During background indexing, GPS coordinates are resolved to city, region, and country names and stored locally — enabling gallery search by location (e.g. `city:tokyo`, `country:jp`) with no internet required.

### Gallery Zoom Slider & scroll anchoring
- **Gallery Zoom Slider** — Custom slanted wedge-shaped zoom track and circular thumb in the justified gallery (bottom bar; row height **220–380px**, step 20). The chosen size is saved in `QSettings` (`gallery_row_height`).
- **Justified relayout on zoom** — Dragging the slider runs a full justified grid rebuild at the new target row height so each row still fills the viewport width and thumbnail sizes change visibly.
- **Scroll anchoring** — When you press the slider, the **upper-left visible** thumbnail becomes the anchor for the whole drag. After each relayout, vertical scroll is restored so that photo stays at the same height on screen (horizontal position may shift slightly as rows regroup).

### macOS HDR / EDR display
- **Extended Dynamic Range viewport** — On macOS, the GPU single-image view (`RAWVIEWER_GPU_VIEW=1`, default in release builds) enables EDR on the viewport layer so bright HDR content can use headroom above SDR white on supported displays.
- **HDR still-image decode** — HEIC / HEIF / AVIF and 16-bit HDR TIFF files load as 16-bit on macOS when EDR is enabled; Windows and other platforms tone-map to SDR (Reinhard). Standard JPEG previews are unchanged.
- **RAW EDR (macOS, RAW workflow)** — Linear 16-bit LibRaw decode → extended tone map → `RGBX64` QPixmap. **On by default** (`RAWVIEWER_RAW_EDR=1`); set `RAWVIEWER_RAW_EDR=0` to disable. Respects the **Embedded JPEG / RAW workflow** toggle: embedded-JPEG workflow keeps camera preview pixels (SDR); RAW (High Quality) workflow uses EDR when enabled.
- **EDR status** — Startup status bar and top metadata show `EDR · RAW`, `EDR · HDR`, or `EDR ready · embedded JPEG workflow` when applicable.
- **Opt out** — Set `RAWVIEWER_DISABLE_EDR=1` to force the SDR tone-mapping path on macOS.

### Animated GIF & WebP Playback
- **Animated Previews** — Enhanced the image viewer pipeline to support playing, scaling, and animating GIF and WebP files. Displays playback status messages and handles dynamic window scaling seamlessly.

### Performance & gallery
- **Snappier navigation** — Filmstrip warm-up is staggered and throttled; navigation cancels low-priority prefetch so the current photo wins I/O. RAW sensor dimensions for the status bar resolve off the UI thread. Identical status-bar updates are deduplicated. Filmstrip refresh and prefetch wait for single-view first paint (TTFR) instead of a fixed delay.
- **External drives** — Volume read speed is probed once per mount to tune I/O throttling on slow disks.
- **Gallery folder switching** — Layout and thumbnail loads reset per folder; fixes scattered tiles and blank gaps after changing albums.
- **Huge folders** — Gallery opens in capture-time (EXIF) order; the Gallery button appears once that sort finishes (instant when metadata is fully cached).
- **Background indexing** — Metadata and semantic indexing from the previous folder are cancelled when you open a different album.
- **Fast-open deferrals** — Background folder scan, EXIF sort, and filmstrip prefetch use TTFR-or-fallback timing (~2.5s cap) instead of a hard 5s sleep after fast-open.
- **Search & metadata** — Faster EXIF cache lookups, KD-tree reverse geocoding during indexing, and UI-thread sorting moved to background workers for large folders.

### Windows gallery crash fix & scrolling smoothness
- **Fixed a native crash on gallery entry (Windows)** — Switching from single image to gallery right after viewing a large photo could abort the app on some Windows GPU drivers. The single-image view now finishes hiding before its GPU memory is released, avoiding the driver-level conflict.
- **Fixed a scrollbar-jump stall** — Dragging the gallery scrollbar to a distant position could leave the newly visible area stuck behind stale loading requests for the position you scrolled away from, sometimes for several seconds. It now loads immediately.
- **Smoother scrolling** — Raised gallery thumbnail loading concurrency during active scrolling, including on slower external/network drives, so more thumbnails decode in parallel while you scroll.
- **Much faster gallery loading for Sony & Nikon RAW on external drives** — Sony (ARW) and Nikon (NEF) gallery thumbnails now use a lock-free preview extractor that lets many images decode in parallel instead of one at a time, roughly **3.4× faster** cold gallery loading on a slow external drive in testing. Other RAW formats are unaffected. Set `RAWVIEWER_GALLERY_BYTESCAN_FIRST=0` to revert to the previous behaviour.

---

## 🚀 Version 2.4.1
**Release Date: June 19, 2026**

Bug-fix release for **Canon CR2/CR3** (and similar RAW) orientation and capture-date handling, plus **memory safety on relaunch** when session restore reopens a large folder. Treats **gallery mixed orientation**, **single-image rotation**, and **out-of-memory on restart** as separate issues.

### Gallery — mixed portrait / landscape thumbnails

Some users saw **some vertical shots correct and others sideways** in the gallery after upgrading to v2.4. That pattern usually means **old cached thumbnails or EXIF rows** (from before v2.4 orientation fixes) sitting beside freshly warmed ones — not only a live warm-up clash (semantic vs gallery indexing, addressed in v2.4).

**What's fixed in 2.4.1:**
- **EXIF cache validation** — Persisted orientation is checked against LibRaw flip and embedded-JPEG orientation; stale `orientation=1` rows for portrait Canon RAW are rejected and re-extracted.
- **Cache version bump** (`sensor_meta_ver` 9) — Forces refresh of outdated EXIF metadata on upgrade.
- **Thumbnail load repair** — When a cached gallery thumbnail's pixels don't match container orientation, it is corrected or dropped instead of reused.

**If thumbnails still look wrong once:** Run **`scripts/Launch/shell/clear_cache.sh`** (macOS) or **`clear_cache.bat`** (Windows), then reopen the folder. This clears mixed old/new cache in one step.

### Single-image view — rotation unlike gallery

Gallery and single view used **different preview paths**: gallery favored embedded JPEG previews (with EXIF transpose); opening from gallery could **force immediate full LibRaw decode** with wrong container orientation on Canon CR2/CR3.

**What's fixed in 2.4.1:**
- **Same preview pipeline** — RAW thumbnails prefer LibRaw's embedded JPEG segment; byte-scan fallbacks apply container orientation after decode.
- **Gallery → single** — No longer skips straight to full-resolution LibRaw; uses the same progressive preview → full decode path as normal navigation.
- **Unified orientation lookup** — Single view uses `EXIFExtractor` (LibRaw flip + embedded JPEG), not header-only exifread alone.
- **Canon CR2/CR3 metadata** — When exifread returns sparse tags, pyexiv2 fills in **orientation** and **DateTimeOriginal** (also improves capture-date sorting).

### Memory — relaunch / session restore (macOS & Windows)

On **8–16 GB** machines, closing RAWviewer and reopening it could restore the last folder and file while **full-resolution decode**, **neighbor prefetch**, and **AI/metadata indexing** all started in the same second — sometimes killing the app (macOS jetsam, exit **137**) or freezing Windows under heavy paging.

**What's fixed in 2.4.1:**
- **Staged session restore** — After the first preview paints, full decode for the current image waits **~2.5 s**, then neighbor prefetch waits **~0.8 s** more. Normal folder open and gallery navigation are unchanged.
- **Preview vs full cache tiers** — Full-resolution embedded JPEGs are stored separately from the smaller preview tier used for fit-to-window, reducing peak RAM during upgrades.
- **macOS memory stats fallback** — On newer macOS kernels where `psutil` fails, RAM tier and cache pressure use `sysctl` + `vm_stat` instead of a fake 50% value.

**What you might notice after relaunch:** Fit view may stay on a fast preview for a few seconds before full quality; arrow-key navigation to the next file may be slightly slower for the first ~3 s. **Space / double-click zoom to 100%** still requests full resolution immediately.

**If relaunch still fails on low RAM:** Use **Lite**, set `RAWVIEWER_DISABLE_SESSION_RESTORE=1`, or temporarily disable semantic search — see README troubleshooting.

**Tuning (optional):**

| Variable | Default | Effect |
|----------|---------|--------|
| `RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD` | `1` | Staged heavy loads after session restore |
| `RAWVIEWER_SESSION_RESTORE_FULL_DECODE_DELAY_MS` | `2500` | Delay before full decode on relaunch |
| `RAWVIEWER_SESSION_RESTORE_PRELOAD_DELAY_MS` | `800` | Extra delay before neighbor prefetch |
| `RAWVIEWER_DISABLE_SESSION_RESTORE` | `0` | Skip restoring last folder on launch |

### Recommended test after upgrade

1. Optional but best: **`clear_cache.sh`** once (orientation cache refresh).
2. Open a folder of Canon (or other) portrait RAWs — gallery should be consistent.
3. Click into single view — direction should match the gallery; full quality loads in the background.
4. **Relaunch test:** Close RAWviewer, reopen the same session on a large RAW folder — app should stay up; check logs for `[SESSION] Deferring heavy loads until first paint`.

---

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

---

# RAWviewer 版本發布說明 (繁體中文)

## 🚀 版本 2.5.0
**發布日期：2026 年 6 月 25 日**

主要版本，引入了自訂藝廊縮放滑桿、捲動錨定、互動式 GPS 地圖覆蓋、macOS HDR/EDR 顯示，以及動畫 GIF/WebP 播放功能。

### 藝廊縮放滑桿與捲動錨定
- **藝廊縮放滑桿** —— 在藝廊視圖底部列新增自訂 slanted 楔形縮放軌道與圓形滑桿（列高 **220–380px**，步進 20）。設定儲存於 `QSettings`（`gallery_row_height`）。
- **縮放時 justified 重排** —— 拖曳滑桿會以新的目標列高完整重建 justified 網格，每列仍填滿視窗寬度，縮圖大小明顯改變。
- **捲動錨定** —— 按下滑桿時，以**左上角可見**縮圖為錨點；每次重排後還原垂直捲動，使該照片在畫面上的高度大致不變（列重新分組時水平位置可能略有偏移）。

### 互動式 GPS 地圖顯示覆蓋
- **GPS 地圖覆蓋** —— 在單張影像檢視中按下 **M** 鍵，可顯示/隱藏互動式瓦片地圖卡片。按下後立即顯示 **Loading map…** 占位符；無 GPS 的照片不會彈出地圖。
- **座標徽章** —— 地圖上顯示可點擊的經緯度徽章；點擊可在瀏覽器中開啟 **Google Maps**。
- **離線逆地理編碼（搜尋功能）** —— 隨附預載的全球城市（涵蓋所有人口大於 500 的 10 萬多個城市）與地標資料庫（`cities500.csv.gz` 與 `landmarks.csv.gz`）。在背景索引期間，GPS 座標將被解析為城市、地區和國家名稱並儲存於本機，讓您無需網路即可透過位置進行藝廊搜尋（例如 `city:tokyo`、`country:jp`）。

### macOS HDR / EDR 顯示
- **擴展動態範圍視埠** —— 在 macOS 上，GPU 單圖檢視（`RAWVIEWER_GPU_VIEW=1`，release 預設）於視埠圖層啟用 EDR，支援的顯示器可呈現高於 SDR 白位的 HDR 亮度。
- **HDR 靜態影像解碼** —— HEIC / HEIF / AVIF 與 16-bit HDR TIFF 在 macOS 且 EDR 啟用時以 16-bit 載入；Windows 及其他平台以 Reinhard 壓成 SDR。一般 JPEG 與 RAW 預覽不變。
- **關閉方式** —— 設定 `RAWVIEWER_DISABLE_EDR=1` 可在 macOS 強制走 SDR tone mapping。

### 動畫 GIF 與 WebP 播放
- **動畫預覽** —— 增強了影像檢視器管線，完整支援播放、縮放和播放動畫 GIF 及 WebP 檔案，顯示播放狀態訊息，並無縫處理動態視窗縮放。

### 效能與藝廊
- **更順暢的導覽** —— 底片條預熱分段節流；切換照片時取消低優先級預取。狀態列 RAW 感測器尺寸改在背景執行緒解析；相同狀態列更新會去重。底片條刷新與預取改為等待單圖首次繪製（TTFR）。
- **外接磁碟** —— 每個掛載點探測一次讀取速度，以調整慢速磁碟的 I/O 節流。
- **藝廊切換資料夾** —— 每個資料夾重設版面與縮圖載入，修正切換相簿後縮圖散亂或大片空白。
- **大型資料夾** —— 藝廊以拍攝時間（EXIF）排序；Gallery 按鈕在排序完成後出現（中繼資料已快取時幾乎即時）。
- **背景索引** —— 開啟不同相簿時，取消上一資料夾的中繼資料與語意索引。
- **快速開啟延遲** —— 背景掃描、EXIF 排序與底片條預取改為 TTFR 或約 2.5 秒上限，取代固定 5 秒睡眠。
- **搜尋與中繼資料** —— 更快的 EXIF 快取查詢、索引期間 KD-tree 逆地理編碼，以及大型資料夾的 UI 執行緒排序改至背景 worker。
