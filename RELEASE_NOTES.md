# RAWviewer Release Notes

**Language / 語言：** English · [繁體中文](RELEASE_NOTES.zh-TW.md)


## 🚀 Version 3.0.2
**Release Date: July 21, 2026**

### ✨ Smoother & Faster Editing

- **Much faster slider adjustments:** Dragging sliders for Exposure, Temperature, Denoise, Vignette, and Dodge/Burn is now silky-smooth. The editing pipeline now uses multi-core CPU acceleration across all edit layers, so even heavily retouched photos respond instantly.
- **Faster photo loading:** Opening photos with saved edits is roughly twice as fast. It shows a quick "Loading previous edit…" overlay so you know the app is working without waiting for redundant preview renders.

### 🖌️ Smarter Dodge & Burn Brush

- **Natural, spot-free brush strokes:** Brushing over a photo now paints a smooth, continuous layer. Pausing or slowing down your mouse mid-stroke will no longer leave accidental dark or bright blotches.
- **Fixed brush disarming:** Dodge, Burn, and Eraser tools stay reliably active when moving your mouse between the Adjust panel and the image canvas.
- **Faster mask overlay:** Turning on the mask overlay (<kbd>M</kbd>) now paints under your brush with zero lag, instead of re-rendering the whole photo overlay on every stroke.
- **Smarter Edge Assist:** Edge Assist automatically adapts to image noise, follows sharp object contours, and recognizes color boundaries between regions of equal brightness.

### 🎯 Controls & UI Refinements

- **No accidental scroll/zoom jumps while editing:** Mouse scrolling over the photo is disabled while the Adjust panel is open, preventing accidental photo jumps or getting stuck at 35% zoom mid-edit. (`Ctrl + Scroll` to zoom and scrolling to adjust brush size remain fully active).
- **Inspect masks anytime with <kbd>M</kbd>:** The **Mask (M)** button is now labelled **Mask (M)** and toggles with <kbd>M</kbd>. It stays available whenever a photo has dodge/burn or heal edits — you can toggle it anytime without needing to pick up a brush first.
- **Instant mask clear updates:** Clearing dodge/burn or heal work updates the image immediately.
- **Proper app name in Task Manager:** RAWviewer now clearly identifies itself as "RAWviewer" in your system's Task Manager / Activity Monitor instead of "Python".

### Upgrade

Install over 3.0.1. No cache clear needed for this update.

---

## 🚀 Version 3.0.1
**Release Date: July 20, 2026**

### What’s new for you

- **Smaller Windows Plus (CUDA)** — GPU demosaic now uses **CuPy** instead of a multi-GB PyTorch install. Expect roughly **1.5–2 GB** (+ MobileCLIP models), similar to Plus (DirectML).
- **Clearer edition names** — **Standard**, **Plus (DirectML)**, and **Plus (CUDA)** in the installer and UI.
- **Faster launch** — the app reaches the first screen sooner (AI search libraries load only when you open Search).
- **Export change** — export-time **SCUNet AI denoise** is removed. Use Adjust chroma noise reduction, then export JPEG / WebP / TIFF as usual.

### Fixes and improvements

- **Stable gallery scrolling & loading:**
  - Resolved UI freezes and trackpad stuttering during fast gallery scrolling.
  - Fixed scrolling stalls and black/blank thumbnail holes when jumping between sections on high-resolution displays (e.g., Windows scaled at 125%/150%).
  - Optimized folder scanning and thumbnail checks to prevent temporary system pauses when viewing photos on slower external drives.
  - Fixed gallery loading stalls or frozen thumbnails on certain computers.
- **Reliability:**
  - Fixed a deadlock issue when rebuilding the image cache.
  - Fixed the gallery jumping back to the top of the folder after bulk-deleting or moving photos; the screen now remains anchored to neighboring photos.
  - Fixed semantic search crashes on macOS 26+ beta releases.
  - Improved macOS Plus features, including better Dock icon integration, gallery consistency, and face scanning.
- **Usability & diagnostics:**
  - Gallery images now open using embedded previews first for immediate display.

### Upgrade

Install over 3.0.0. Optionally clear cache once (Setup checkbox or `clear_cache.bat` / **Clear Cache.command**).

---

## 🚀 Version 3.0.0
**Release Date: July 18, 2026**

RAWviewer 3.0 turns the viewer into a complete **cull-and-develop** tool: a fully integrated, non-destructive **Adjust / Develop editor**, **star ratings**, and a rebuilt image-loading pipeline that is **substantially faster than 2.5** — cold tests show gallery entry about **3×** sooner and RAW full-res arrow-key navigation about **1.6–1.7×** sooner (Lite and Full).

### ✨ New features

#### 🎨 Adjust / Develop editor (E)
> [!WARNING]
> **Disclaimer:** Editing is currently experimental. Compatibility with every camera model — especially newly released bodies — is not guaranteed.
- Non-destructive develops saved to **XMP sidecars**; your RAW files are never modified. Browse surfaces show original pixels by default (`RAWVIEWER_SIDECAR_ADJUST=1` to render saved edits in browse).
- **Tone & color:** exposure/contrast/highlights/shadows/whites/blacks, point + parametric **tone curve** (RGB and per-channel), **HSL color mixer**, saturation/vibrance.
- **White balance:** As Shot (from EXIF Kelvin when present), presets (Daylight / Cloudy / Shade / Tungsten / Fluorescent / Flash), eyedropper, and **Auto WB** estimation.
- **Transform:** interactive **crop overlay** with aspect presets (Free / Original / 1:1 / 4:3 / 3:2 / 16:9), **straighten with Auto**, keystone perspective, optional lens correction.
- **Local tools:** **Dodge & Burn** with soft circular stamps, Edge Assist, and mask overlay; **Heal (H)** inpainting for dust/smudge removal. Both work in Lite and Full.
- **Effects & detail:** vignette (Amount/Midpoint), dehaze, sharpness, chroma noise reduction with selectable method.
- **Looks:** **Creative LUT** (`.cube` drag-drop + managed library) and managed **XMP presets** — import, apply, remove.
- **Export** to JPEG / WebP / 16-bit TIFF with a progress dialog and cancel.
- A new **Adjust button** in the single-view bar opens the editor (mirrors the **E** shortcut, highlights while active).

#### ⭐ Star ratings
- Rate **1–5** with the number keys (**0** clears) or the clickable stars in single view. **↑ no longer bookmarks** (gallery ↑ scrolls; Compare ↑ still promotes the candidate). Narrow the gallery with the bottom **rating-filter stars**.
- Ratings persist to **XMP** and appear as gallery tile badges; filter the gallery to **rating ≥ N**, combinable with the bookmark filter.

#### ⚡ Faster than 2.5 (verified cold suite)
**Headline vs 2.5 Lite on the same Windows machine (cache cleared before every run):** gallery ready **8.6s → ~3s (~2.7–2.9× faster)**; RAW full-res navigation median **0.95s → ~0.6s (~1.6–1.7× faster)**. **Lite and Full share the same 3.0 loading pipeline** and both beat 2.5 on the paths that matter for culling; Full can add GPU demosaic (CuPy on Windows CUDA builds) at comparable speed on these tests.

- **Fast RAW decode** on by default: half-size and full sensor tiers share one LibRaw unpack, with verified color parity (±1 8-bit LSB on golden ARW/CR3 sets).
- Cold gallery thumbnail warmup about **3×** faster; Canon CR3 embedded previews no longer read the whole file.
- Heavy optional ML imports deferred until after first paint; session restore staggers full decode and prefetch so relaunch feels instant.
- **GPU demosaic scheduling (Full):** reserve the last heavy decode slot for the on-screen image, preempt neighbor full-decodes when CURRENT is waiting, and release cancelled heavy slots immediately — fixes stalls where neighbor prefetch held `raw_limit=1` and the current file never reached `stages=['full']`.
- **LibRaw-unsupported RAW (e.g. Nikon HE/HE*):** use the embedded preview for that file only — no longer flips the app-wide workflow to embedded JPEG (which previously disabled GPU demosaic for every later file in the folder).
- Near neighbors warm a sensor `full` decode under RAW+GPU once the current full is already in flight (slot reservation keeps CURRENT first). CUDA demosaic concurrency stays at the adaptive default (typically **2** on RTX-class GPUs); raising it further did not improve the mixed-format suite.

##### Cold-start suite vs 2.5 (Windows, Jul 2026)
Same machine, cleared `~/.rawviewer_cache` before **each** test. Gallery: open one RAW in a ~6900-file folder, wait for EXIF sort, hold **Up** 20s. RAW nav: open one file in a ~260-file mixed-format RAW folder, wait for sensor-covering full-res paint, then next (full loop).

| vs 2.5 baseline | Gallery ready | vs 2.5 | RAW nav median | vs 2.5 |
|---------------|---------------|--------|----------------|--------|
| **2.5 Lite** | 8.6s | — | 0.95s | — |
| **3.0 Lite & Full** | **~3s** | **~2.7–2.9× faster** | **~0.6s** | **~1.6–1.7× faster** |

Run-to-run variance between Lite (CPU Fast RAW) and Full (GPU demosaic on supported files) is within a few percent on these cold-suite paths — both editions are in the same class vs 2.5.

| vs 2.5 baseline | RAW nav mean | RAW nav p95 | Tiles / 20s Up |
|---------------|--------------|-------------|----------------|
| **2.5 Lite** | 1.20s | 2.70s | 1043 |
| **3.0 Lite & Full** | **~0.7–0.8s** | **~1.8–2.6s** | **~940** |

**Bottom line:** upgrading from 2.5 to 3.0 is a clear speed win for folder open → gallery and for RAW high-quality browsing. Choose **Standard** for the smallest install, or **Plus** for AI search (and CuPy GPU demosaic on Windows CUDA). Culling speed is in the same ballpark on these paths.

#### 🖼️ Gallery loading & scrolling overhaul
- Main-thread stalls during gallery fill eliminated (worst observed stall **11.7s → <0.6s**): budgeted tile passes, deferred cache-hit delivery, and no per-tile sidecar probes on external drives.
- **Anchor-first entry:** opening the gallery on a mid-folder image fills the rows around that image first.
- Mouse-wheel and trackpad scrolling traverse like key-hold navigation, with configurable gains.
- Tile badges (★ rating / burst ×N / edited ✎) appear together with the thumbnail.

#### 📷 Formats
- **Nikon High Efficiency (HE / HE\*) NEF** now open for browsing, culling, and rating via the embedded JPEG — no more spurious "unsupported or corrupt" dialogs. RAW develop stays disabled for HE files until a decoder exists.

#### 📦 Lite edition, slimmer
- **Lite keeps the full Adjust editor** (CPU Fast RAW) but omits PyTorch / kornia — install stays in the ~500 MB class.
- Full (CUDA) still prefers GPU demosaic when the backend is present.

#### 🎨 Design
- New **darkroom app icon** (gallery tiles + tone-curve tile) with hi-res splash assets.
- Shared darkroom color palette across all widgets and chrome.
- macOS release builds ship an **OpenMP-enabled LibRaw** for parallel decodes.

### 🔧 Fixes since 2.5

- **Orientation:** cold-open / first-paint orientation bugs (sideways, upside-down, or blank first RAW) fixed.
- **Navigation races:** rapid arrow-key navigation no longer sticks on a stale image or a cancelled mid-flight decode.
- **Gallery rebuild scroll:** rebuild/metadata refresh keeps the entry/current image as the scroll anchor instead of jumping to the top-left tile.
- **GPU full-decode starvation:** cancelled heavy RAW tasks now free the concurrency slot immediately; cache hits no longer treat a ≥1024px preview as a finished sensor decode.
- **Gallery click zoom:** clicking a gallery thumbnail always lands at Fit-to-Window instead of inheriting the previous view's zoom.
- **EXIF cache contention:** readers no longer serialize through one global lock (multi-second navigation stalls on large folders fixed).
- **RAW recovery preview (P):** rawpy edge cases fixed.
- **Culling & UI polish:** culling zoom glitches, Windows taskbar flicker on startup, discarded-photo-never-returns, and slow gallery multi-select all fixed.

### ⚠️ Breaking change: macOS EDR removed

The 2.5 macOS EDR (HDR display) path is **removed in 3.0** — it was incompatible with the new fast loading pipeline and made RAW decoding very slow. HDR stills (HEIC/HEIF/AVIF/HDR TIFF) are tone-mapped to SDR. Restoring EDR without regressing speed is tracked under [Looking ahead](README.md#looking-ahead) in the README.

### Environment variables (new / notable)

| Variable | Default | Effect |
|----------|---------|--------|
| `RAWVIEWER_FAST_RAW_DECODE` | `1` | Fast RAW path; `0` falls back toward rawpy |
| `RAWVIEWER_USE_PROCESS_POOL` | auto | Force LibRaw process pool on/off |
| `RAWVIEWER_SIDECAR_ADJUST` | `0` | Apply saved XMP edit sliders to browse/full-res pixels (requires editing enabled). Default off: browse shows original; edits render in Adjust |
| `RAWVIEWER_UNPACK_STASH_SLOTS` | `3` | How many LibRaw unpack mosaics to keep for half→full / A↔B revisit (1–8) |

### Recommended test after upgrade

1. **Existing users (recommended once):** clear cache after installing v3, then reopen your folder. Older installs stay on the legacy search/index performance mode until you clear cache (or set `RAWVIEWER_PERF_V2=1`). **Windows:** check **Clear existing cache** in Setup (or run `clear_cache.bat` next to `RAWviewer.exe`). **macOS:** use **Clear Cache.command** / `clear_macos_cache.sh` from the release zip. This does not delete photos or XMP.
2. Open a mix of ARW / CR3 / NEF (including HE\*): arrow through, zoom to 100%, confirm orientation.
3. Rate with **1–5**, filter gallery by stars, confirm sidecars.
4. Open Adjust (**E** or the sliders button): try WB presets, Crop, Dodge/Burn, Heal, Vignette/Dehaze, a Creative LUT; save and confirm the XMP sidecar appears next to the RAW.

---

## 🚀 Version 2.5.0
**Release Date: July 3, 2026**

RAWviewer 2.5 is built for **serious libraries** — faster culling, smarter search, and a viewer that stays responsive from the first folder open to the ten-thousandth frame.

### 🚀 Key Feature Highlights

#### 🎨 Gallery & View
- **Gallery Zoom Slider** — A sleek wedge-shaped zoom track with **scroll anchoring**: the top-left thumbnail stays locked on screen while you resize the grid.
- **macOS HDR / EDR** — See HDR stills and RAW the way they were meant to look on supported Mac displays (HEIC, AVIF, 16-bit TIFF, and **RAW High Quality** workflow). Embedded JPEG preview stays SDR when you want speed.
- **Animated GIF & WebP** — Play, scale, and browse animated files natively in single-image view.
- **RAW recovery preview (P)** — Press **P** for an instant shadow/highlight recovery preview on RAW/DNG — judge extreme contrast without waiting for a full export. Session-only, fit view.
- **Clipping overlay (J)** — Press **J** to see exactly what's clipping — red for blown highlights, blue for crushed shadows — on both RAW and in Compare mode.

#### 👥 Culling & Comparison
- **Burst Image Grouping** — Rapid-fire sequences stack automatically by capture time. Double-click to expand and pick the keeper.
- **Dual-Pane Compare (C)** — Anchor vs. candidate, side by side. Promote (↑), reject to Discard (↓), or delete — with synced composition guides (**G**) and clipping overlays (**J**) on both panes. **Space** toggles synchronized zoom and pan across both panes; press **F** for focus overlays, then **Space** zooms each pane to its own AF/subject point for side-by-side sharpness checks.

#### 🗺️ Maps & Location Search
- **Interactive GPS Overlay (M)** — A live map card for geotagged shots, with one-click **Google Maps** from the coordinate badge.
- **Offline place search** — 100,000+ cities and landmarks resolved locally during indexing. Search `city:tokyo` or `country:jp` with **no internet** — ever.

#### ⚡ Performance at Scale
Built to stay fluid on **5k–10k+ image libraries**, whether you're on a fast internal SSD or a slow external drive.

**📦 Smarter cache**
- **WebP tiles** — New cache writes are smaller and faster to read; size-based LRU keeps disk use in check.
- **Your existing JPEG cache still works** — no forced migration. Clear cache manually only if you want immediate WebP savings.
- **One decode, many uses** — Preview, grid, and thumbnail tiers fill from a single pass instead of reopening every file.

**🤖 Background indexing that gets out of your way**
- **Two-phase indexing** — Metadata first, then AI embeddings (Full). Search unlocks when indexing is ready for your edition (metadata on Lite; metadata + AI on Full).
- **Scroll-aware** — Indexing pauses while you scroll the gallery and yields to foreground navigation so arrow keys stay smooth.

**⏱️ Snappier navigation**
- **Staggered prefetch** — Switching photos cancels low-priority loads; full-res neighbors wait until the current frame paints.
- **Off the UI thread** — Sorting, heavy conversions, and metadata work move to background workers — fewer micro-stutters.

**🪟 Windows polish**
- Fixed a GPU hand-off race when returning to the gallery after large single-image views.
- Gallery scrollbar jumps load visible tiles immediately.
- Optional stable file picker: `RAWVIEWER_QT_FILE_DIALOG=1`.

### Gallery reliability (large folders)
- **Folder switching** — Layout and thumbnail loads reset per folder; scattered tiles and blank gaps after changing albums are gone.
- **Huge folders** — Gallery opens in capture-time (EXIF) order; the Gallery button appears once that sort finishes (instant when metadata is fully cached).
- **Orientation-aware tiles** — EXIF display orientation is applied consistently when thumbnails enter the gallery grid, so portrait shots keep correct framing when you jump scroll position or enter the gallery from single view.

---

## 🚀 Version 2.4.1
**Release Date: June 19, 2026**

Bug-fix release for **Canon CR2/CR3** (and similar RAW) orientation and capture-date handling, plus **memory safety on relaunch** when session restore reopens a large folder. Treats **gallery mixed orientation**, **single-image rotation**, and **out-of-memory on restart** as separate issues.

### Gallery — mixed portrait / landscape thumbnails

Some users saw **some vertical shots correct and others sideways** in the gallery after upgrading to v2.4. That pattern usually means **old cached thumbnails or preview rows** (from before v2.4 orientation fixes) sitting beside freshly warmed ones — not only a live warm-up clash (semantic vs gallery indexing, addressed in v2.4).

**What's fixed in 2.4.1:**
- **EXIF cache validation** — Persisted orientation is checked against LibRaw flip and embedded-JPEG orientation; stale `orientation=1` rows for portrait Canon RAW are rejected and re-extracted.
- **Cache version bump** (`sensor_meta_ver` 9) — Forces refresh of outdated EXIF metadata on upgrade.
- **Thumbnail load repair** — When a cached gallery thumbnail's pixels don't match container orientation, it is corrected or dropped instead of reused.

**If thumbnails still look wrong once:** Run **`scripts/Launch/macos/clear_cache.sh`** (macOS) or **`clear_cache.bat`** (Windows), then reopen the folder. This clears mixed old/new cache in one step.

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
- **`clear_cache.bat` / `clear_cache.sh`**: Full dev/session reset; repo-root `clear_cache.bat` forwards to `scripts/Launch/windows/clear_cache.bat`.
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

# RAWviewer 版本發布說明

## 🚀 版本 3.0.0
**發布日期：2026 年 7 月 14 日**

RAWviewer 3.0 是一個重大版本，帶來了全面整合的**編輯功能**，並引入了全新的**影像載入邏輯**，以實現極致速度。我們進行了多次測試，以檢查與 2.5 版本相比的速度提升，確認了在各種相機格式上的顯著性能提升。

此版本在 2.5 之上是更快的**瀏覽／篩選**版本：包含 **Fast RAW 解碼**、可寫入 XMP 的星級、Nikon HE/HE* 處理、暗房色票，以及大量導覽／圖庫可靠性修正。

### 🚀 主要功能亮點

#### 🎨 完整編輯功能
- **Adjust / Develop 編輯面板**現已全面整合，並預設對所有使用者開啟。
- 包含色調曲線、鏡頭校正、細節、色彩降噪、加亮/加深，以及 PV2012 風格的顯影。
- 編輯動作為非破壞性，並直接儲存至 **XMP** 附屬檔案。已儲存的編輯只在 **Adjust 面板**內呈現；瀏覽介面（圖庫縮圖、單張預覽、RAW 檢視）顯示原始像素（設 `RAWVIEWER_SIDECAR_ADJUST=1` 可改為在瀏覽時套用編輯）。

#### ⚡ 更新的影像載入邏輯與速度驗證
- **快速 RAW 解碼** 預設開啟（`RAWVIEWER_FAST_RAW_DECODE=1`）：半尺寸與全感測器層級共用一次解包；與先前的管線相比，色彩一致性已驗證（在標準 ARW/CR3 測試集中，誤差為 ±1 8-bit LSB）。
- **多次測試驗證**：針對 2.5 版本的廣泛基準測試證實了大幅的速度提升：
  - **全感測器解碼：** 約快 **1.4×**（中位數）；高階格式在適用 Fast RAW 時約在 **1.3–1.7×** 的範圍。
  - **符合檢視後縮放**（重複使用符合檢視的解包）：約比冷啟動完整 rawpy 解碼快 **2×**。
  - **半尺寸適應檢視瀏覽：** 在典型 ARW 照片集中約快 **1.2–1.3×**；吞吐量提升約 **+30%**。
- 圖庫縮圖冷啟動預熱約快 3 倍；Canon CR3 內嵌預覽不再讀取整個檔案；EXIF 快取不再透過單一全域鎖序列化所有讀取器（修正了長達數秒的導覽卡頓）。
- 笨重的可選 ML 匯入延遲到首次繪製之後（啟動凍結減少約 0.9 秒）。
- 相鄰內嵌 JPEG 預先載入、方向性／懸停圖庫預先載入，以及 RAF/3FR 跳過過於積極的完整去馬賽克相鄰載入。

#### ⭐ 星級評分
- 單張檢視：可點擊的 **1–5 星**（取代原本 UI 中的書籤星號）。鍵盤 **1–5** 評分；**0** 清除。**↑ 不再切換書籤**（圖庫用星級篩選）。
- 評分會儲存到 **XMP** 附屬檔案中。
- 圖庫：縮圖上顯示評分標記；可篩選 **星級 ≥ N**；可與書籤篩選條件合併使用。

#### 📷 格式與解碼可靠性
- **Nikon HE / HE\*** NEF：偵測並避免錯誤的「不支援或損毀」對話框，透過內嵌 JPEG 開啟。
- 冷啟動／首次繪製方向修正（側翻／倒立／第一張 RAW 空白）。
- 快速導覽競態條件（卡在舊影像、取消飛行中的解碼）。
- 修正 RAW 復原預覽（**P**）與 EDR 路徑的 rawpy 錯誤。

#### 🎨 介面優化
- 供小工具與 chrome 使用的共用**暗房**（darkroom）色票（`theme.py`）。
- 圖庫磁碟快取預設改為偏好 **JPEG** 圖磚（WebP 仍可用）。
- 修正篩選縮放故障、Windows 啟動時工作列閃爍、捨棄的相片永遠不返回，以及圖庫多選緩慢的問題。

### 環境變數（新增／值得注意）

| 變數 | 預設 | 效果 |
|----------|---------|--------|
| `RAWVIEWER_FAST_RAW_DECODE` | `1` | 快速 RAW 路徑；`0` 會回退到 rawpy |
| `RAWVIEWER_USE_PROCESS_POOL` | auto | 強制開啟/關閉 LibRaw 程序池 |
| `RAWVIEWER_SIDECAR_ADJUST` | `0` | 將已儲存的 XMP 編輯滑桿套用到瀏覽/全解析度像素（需啟用編輯） |

### 升級後的建議測試

1. **既有使用者（建議執行一次）：** 安裝 v3 後清一次快取，再重開資料夾。縮圖／EXIF 會隨使用自動重建，但舊安裝會維持較舊的搜尋／索引效能模式，直到清快取（或設 `RAWVIEWER_PERF_V2=1`）。**Windows：** 在 Setup 勾選 **Clear existing cache**（或執行安裝目錄內 `clear_cache.bat`）。**macOS：** 使用壓縮檔內 **Clear Cache.command**／`clear_macos_cache.sh`。不會刪除相片或 XMP。
2. 開啟混合的 ARW / CR3 / NEF（包括 HE\*）：使用方向鍵瀏覽，縮放至 100%，確認方向。
3. 使用 **1–5** 評分，在圖庫依星級篩選，確認附屬檔案。
4. 開啟 Adjust（**E**）：試 WB 預設、裁切、Dodge/Burn、暈影／去霧等。

---

## 🚀 版本 2.5.0
**發布日期：2026 年 7 月 3 日**

RAWviewer 2.5 為**大型圖庫**而生——更快的篩選、更聰明的搜尋，從第一個資料夾到第上萬張相片都保持流暢。

### 🚀 重點功能

#### 🎨 圖庫與檢視
- **圖庫縮放滑桿** — 楔形縮放軌道搭配**捲動錨定**：拖曳時左上角縮圖固定在畫面上。
- **macOS HDR / EDR** — 在支援的 Mac 上以延伸動態範圍檢視 HDR 靜態與 **RAW 高品質工作流程**（HEIC、AVIF、16-bit TIFF）。內嵌 JPEG 工作流程仍為快速 SDR 預覽。
- **GIF / WebP 動畫** — 單張檢視原生播放、縮放與瀏覽動畫檔。
- **RAW 復原預覽（P）** — 在 RAW/DNG 上按 **P** 即時預覽陰影/高光復原，無需等待完整匯出。僅本工作階段、符合模式。
- **裁切疊圖（J）** — 按 **J** 標示過曝（紅）與死黑（藍）；比較模式兩側皆可同步顯示。

#### 👥 篩選與比較
- **連拍分組** — 依拍攝時間自動堆疊連拍；雙擊展開挑選最佳張。
- **雙欄比較（C）** — 錨點 vs 候選並排；提升（↑）、拒絕至 Discard（↓）、刪除；可同步構圖輔助線（**G**）與裁切疊圖（**J**）。**空白鍵**切換兩側同步縮放與平移；按 **F** 顯示對焦框後，**空白鍵**可將左右各自縮放至該張的 AF/主體對焦點，方便比對銳利度。

#### 🗺️ 地圖與地點搜尋
- **GPS 地圖疊圖（M）** — 含 GPS 相片的互動地圖；座標徽章一鍵開啟 **Google Maps**。
- **離線地點搜尋** — 背景索引解析 10 萬+ 城市與地標；可搜尋 `city:tokyo`、`country:jp`，**完全離線**。

#### ⚡ 大資料夾效能
針對 **5k–10k+ 張**圖庫優化，內建 SSD 與外接慢碟皆適用。

**📦 更聰明的快取** — 新寫入採 **WebP**、依大小 LRU；舊 **JPEG 快取仍可用**；單次解碼填滿多層縮圖。

**🤖 背景索引** — 先中繼資料、再 AI 嵌入（Full）；搜尋欄在對應版本索引就緒後解鎖；捲動圖庫時暫停索引、讓路給前景導覽。

**⏱️ 更順的導覽** — 分段預載、切換相片取消低優先級載入；排序與重轉換移至背景執行緒。

**🪟 Windows** — 修正大圖單張後切回圖庫的 GPU 競爭；捲軸跳轉即時載入；可選 `RAWVIEWER_QT_FILE_DIALOG=1` 穩定檔案選擇器。

### 藝廊可靠性（大型資料夾）
- **切換資料夾** — 每個資料夾重設版面與縮圖載入。
- **大型資料夾** — 以 EXIF 拍攝時間排序；中繼資料已快取時 Gallery 按鈕幾乎即時出現。
- **方向感知縮圖** — 縮圖進入藝廊時一致套用 EXIF 顯示方向。
