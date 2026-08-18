<p align="center">
  <img src="icons/appicon-master.png" alt="RAWviewer" width="256">
</p>

<h1 align="center">RAWviewer</h1>

<p align="center"><strong>Browse, cull, and rate thousands of RAW photos — fast, private, entirely on your computer.</strong></p>

<p align="center">
  <img src="docs/images/hero-gallery.jpg" alt="RAWviewer gallery view — a folder of RAW photos in a justified grid" width="900">
</p>

<p align="center">
  <a href="https://github.com/markyip/RAWviewer/releases/latest"><strong>⬇&nbsp; Download for Windows</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/markyip/RAWviewer/releases/latest"><strong>⬇&nbsp; Download for macOS</strong></a>
</p>

<p align="center">Free · Open source (MIT) · Works offline — your photos never leave your machine</p>

<p align="center">
  <img src="https://img.shields.io/github/v/release/markyip/RAWviewer?style=flat-square&labelColor=2b2925&color=96897a&label=version" alt="Latest version">
  <img src="https://img.shields.io/github/downloads/markyip/RAWviewer/total?style=flat-square&labelColor=2b2925&color=d9691e&label=downloads" alt="Total downloads">
  <img src="https://img.shields.io/badge/license-MIT-96897a?style=flat-square&labelColor=2b2925" alt="License: MIT">
  <a href="https://www.buymeacoffee.com/markyip">
    <img src="https://img.shields.io/badge/buy%20me%20a%20coffee-☕-d9a441?style=flat-square&labelColor=2b2925" alt="Buy Me a Coffee">
  </a>
</p>

**Language / 語言：** [English](README.md) · [繁體中文](README.zh-TW.md)

---

## What it does

- **Browse at shooting speed.** Open a folder of RAWs and flick through full-screen previews with the arrow keys — no import, no catalog, no waiting.
- **Cull with your fingers, not your mouse.** **1–5** rates a keeper, **↓** moves it to a Discard folder, **0** clears the rating, **C** compares similar frames side by side with synchronized zoom.
- **Find any photo.** Type a place (`tokyo`), a camera (`sony`), or a year (`2024`) in gallery search. The Plus edition also understands plain descriptions like `sunset on beach` — all offline.
- **Develop without touching your files.** Press **E** for the Adjust panel: exposure, white balance, crop, dodge & burn, healing, LUTs, and local masks — brushed by hand or found for you (Plus). Every edit is saved to an XMP sidecar; your RAW is never modified.

<p align="center">
  <img src="docs/images/single-view.jpg" alt="Full-screen single-image view with film strip and star rating" width="900">
  <br><em>Full-screen browsing with the film strip and star ratings</em>
</p>

<p align="center">
  <img src="docs/images/adjust.jpg" alt="Adjust panel with a before/after split view — histogram, tone sliders, and tone curve beside the photo" width="900">
  <br><em>The Adjust panel with before/after split: develop non-destructively, saved to XMP</em>
</p>

## New in 3.0

- **Much faster than 2.5** — cold Windows suite (cache cleared each run): gallery ready **~2.7–2.9×** sooner (**8.6s → ~3s**); RAW full-res navigation median **~1.6–1.7×** sooner (**0.95s → ~0.6s**). Standard and Plus share the same 3.0 pipeline and both beat 2.5 on cull paths — see [`RELEASE_NOTES.md`](RELEASE_NOTES.md)
- **Adjust panel** — develop RAWs non-destructively: tone, white-balance presets, crop, dodge & burn, healing, vignette/dehaze, creative LUTs, and savable presets; export JPEG / WebP / 16-bit TIFF
- **Star ratings** — rate 1–5 with the number keys; filter the gallery by minimum rating
- **Nikon High Efficiency (HE/HE*) files** now open for browsing and culling
- **Lean Standard edition** for machines where install size matters

Full changelog: [`RELEASE_NOTES.md`](RELEASE_NOTES.md)

## Unreleased 3.1.0 (stability + performance vs 3.0.2)

GitHub Releases still ship **3.0.x**. **3.1.0 is a stability and performance patch**, not a feature train, and is **not tagged**. Testers: [`testplan/TEST_PLAN_3.1.md`](testplan/TEST_PLAN_3.1.md). Notes: [`RELEASE_NOTES.md`](RELEASE_NOTES.md#version-310).

What 3.1.0 **claims** vs 3.0.2:

- **RAM-tier Adjust cap** — slower / ≤16 GB machines settle on a smaller working base; export stays full-res. No memory-tier picker.
- **Gallery scroll and load** — viewport-aimed queue, missing tiles return, trackpad sub-pixel, folder change does not wipe thumbs, cancel-once loads.
- **Caches** — byte-budget full-image cache; no wipe of every tier when Adjust is opened and left unchanged; sidecar parse shared.
- **Export reliability** — band seams, silent AI-denoise skip on the banded path, OpenMP/BLAS not taking every core.
- **Crash / thread / cancel** — no `QPixmap` on workers, slot double-free, `id()` registry, CUDA-GL thread, prefetch error storm.
- **HE/HE\* NEF** — Adjust via embedded JPEG instead of refusing the file; undecodable RAW keeps the embedded preview (not a grey tile).

*(Uncommitted, targeted for this patch — not in git yet:)* EXIF blob persist for fields that only lived in RAM; scrollbar date bubble without per-tick flash/decode.

Masks, pairing, MCP, Share to AI, and the rest of the working-tree feature train are **not** 3.1.0 shipping — see [Looking ahead](#looking-ahead) (**3.2+**). Generative editing stays shelved.

---

## Get started

### Windows

1. Download **`RAWviewer_Setup.exe`** from [Releases](https://github.com/markyip/RAWviewer/releases/latest).
2. In the wizard, pick **Plus** (adds photo-description search and AI noise reduction, ~700 MB of models) or **Standard** (smaller). Not sure? See [Standard or Plus?](#standard-or-plus) below.
3. **Upgrading from an older version?** Optionally check **Clear existing cache** in the installer. That unlocks faster search/index defaults without deleting your photos or XMP files.
4. Launch **RAWviewer** from the Desktop shortcut.

Common photo formats get an **Open with** entry automatically. Uninstall any time from Settings → Apps. To clear cache later, run **`clear_cache.bat`** next to `RAWviewer.exe` in the install folder.

### macOS (13 or newer)

1. Download the macOS zip (**Plus** or **Standard**) from [Releases](https://github.com/markyip/RAWviewer/releases/latest) and double-click to extract it.
2. Open **Terminal** (Spotlight → type "Terminal"), type `cd ` (with a space), drag the extracted folder onto the Terminal window, and press Return.
3. Run the installer and follow the two dialogs (**Install**, then **Open**):

```bash
bash install_macos_app.sh
```

4. **Upgrading?** Clear cache once so search/index can use the newer, faster defaults (does not delete photos or XMP):
   - Double-click **Clear Cache.command** (right-click → Open → Open if macOS blocks it), or
   - In Terminal: `bash clear_macos_cache.sh`

RAWviewer is copied to your Applications folder. This one-time Terminal step is what lets macOS trust the app — after that it opens like any other app. To remove it later, use **`Uninstall RAWviewer.command`** from the same zip (dragging to Trash leaves cache files behind). See **Start Here.txt** in the zip for the full checklist.

**Plus edition:** the first time you use gallery **Search**, RAWviewer offers a one-time model download (~150 MB, needs internet once). Click **Download** — after that, search is fully offline.

### Your first ten minutes

Open any folder of photos (drag it onto the window, or double-click a photo). Then:

| Key | Action |
|-----|--------|
| **← / →** | Previous / next photo |
| **Space** or double-click | Fit ↔ 100% zoom |
| **0–5** | Star rating (0 clears; bottom stars work too) |
| **6–9** | Color label (Red / Yellow / Green / Blue; **Shift+6** Purple; same key clears) |
| **K** | Edit keywords (XMP `dc:subject`) |
| **↓** | Move to Discard folder |
| **C** | Compare selected photos side by side |
| **E** | Open the Adjust (develop) panel |
| **Esc** | Back to gallery |

That's the whole culling workflow. Everything else below is optional.

<details>
<summary><strong>All browsing &amp; gallery shortcuts</strong></summary>

Open a folder (menu, drag-and-drop, or double-click a photo). Scroll the **gallery**; click a thumbnail for full-screen view. On **large folders** (thousands of photos), the **Gallery** button appears once capture-time sorting finishes, so thumbnails are in shooting order; if metadata is already cached, sort is instant. In gallery view, drag the **size slider** in the bottom bar to change thumbnail size.

| Key | Action |
|-----|--------|
| **Space** / **Double-click** | Toggle fit-to-window / 100% zoom |
| **Pinch** / **Ctrl+scroll** | Zoom in or out |
| **←** / **→** | Previous / next image |
| **Scroll wheel** | Previous / next (single view, fit mode) |
| **0–5** | Star rating (**0** clears; bottom stars in single view too) |
| **6–9** | Color label (Red / Yellow / Green / Blue; **Shift+6** Purple; same key clears) |
| **K** | Edit keywords |
| **↓** | Move to Discard folder |
| **Delete** | Delete image(s) |
| **Esc** | Gallery: clear selection → exit filters · Single view: back to gallery |
| **Ctrl/Cmd+click** | Gallery: toggle selection |
| **Shift+click** | Gallery: select range (visible order) |
| **↑** | Gallery: scroll up · Compare: promote candidate |
| **C** | Toggle Compare mode on/off (requires multiple images selected) |
| **E** | Show / hide **Adjust** panel |
| **G** | Cycle composition guide |
| **H** | Show / hide histogram |
| **J** | Toggle highlight/shadow clipping overlay |
| **P** | Toggle RAW recovery preview — LibRaw Reconstruct + local polish (RAW/DNG, session only; fit-only). Not the Adjust Highlights/Shadows sliders. |
| **F** | Show / hide focus overlay (supported files) |
| **M** | Show / hide GPS map overlay (single view, geotagged photos) |

**Gallery rating filter:** use the bottom **stars** to show only photos rated N★ and above. Color-label dots sit on tiles when a label is set.

**RAW+JPEG pairing** (on by default) stacks a same-name RAW and JPEG into one `R+J` tile. **Burst grouping** is optional and **off** by default (needs **3+** frames; you pick the cover). Pairing and bursts are separate toggles — a two-file RAW+JPEG pair is never a burst.

**Single-view workflow toggle:** switch between **Embedded JPEG (Fast)** and **RAW (High Quality)** rendering. **Recovery preview (P)** uses Reconstruct + local polish to judge extreme contrast. It is not the Highlights/Shadows sliders (those compress/lift scene-linear luminance and cannot restore hue in fully blown whites).

**Share:** bottom **Share / Open** button, or drag gallery / film-strip thumbnails out. On Mac the menu includes AirDrop (original file) and messaging apps such as WhatsApp / Signal (JPEG proxy for RAW). **File → Share to AI agent** hands a preview JPEG + MCP ticket to an agent (style brief is not a generative prompt). See [`docs/macos-sharing.md`](docs/macos-sharing.md) and [`docs/MCP.md`](docs/MCP.md).

</details>

<details>
<summary><strong>Compare-mode shortcuts</strong></summary>

* **← / →** — Previous / next candidate image
* **↑** — Promote candidate (right pane) to selected (left pane)
* **↓** — Reject candidate and move it to Discard folder (Shift+↓ to reject select)
* **Delete** — Delete candidate image to Recycle Bin/Trash (Shift+Delete to delete select)
* **Space** — Toggle synchronized zoom across both panes (with **F** focus overlays on: zoom each pane to its own focus point)
* **F** — Show / hide focus overlays on both panes
* **J** — Toggle exposure clipping overlay on both panes
* **G** — Cycle composition grid guide on both panes
* **C** / **Esc** — Exit Compare mode

</details>

<details>
<summary><strong>Adjust (develop) shortcuts</strong></summary>

While the Adjust panel is open (**E**):

| Key | Action |
|-----|--------|
| **E** / **Esc** | Close Adjust (returns to browse mode) |
| **D** / **B** / **X** / **H** | Hold to paint **Dodge** / **Burn** / **Eraser** / **Heal** — sweep the pointer, no mouse button. Releasing puts the tool away |
| **P** | On the **Masks** tab: hold/arm to paint into the selected mask (**X** erases). In browse (Adjust closed), **P** is still the Reconstruct *preview* — Auto does not bake that |
| **O** / **M** | Mask overlay (Masks tab: one overlay at a time; **I** invert) |
| **Two-finger scroll** | Change **Brush Size** when a brush is armed (**Ctrl+scroll** still zooms) |
| **←** / **→** | Nudge the focused slider (or previous/next image if none focused) |
| **Ctrl/Cmd+Z** | Undo last edit step |
| **Space** / **Double-click** | Fit / 100% zoom |
| **J** / **G** / **F** | Clipping / composition guide / focus overlay (same as browse) |

Notes: **Effect Strength** applies only to Dodge/Burn; Heal uses **Size** and **Flow** at full inpaint strength. **Brush Feather** sets edge softness for every brush — 0 is a hard circle, 100 fades from the centre. The panel's tool buttons arm persistently if you would rather not hold a key. Browse-only keys (**M**, **P**, histogram **H**) do not apply while Adjust is open — **H** arms Heal instead. By default, edits render inside the Adjust panel and browsing shows original pixels; every edit is written to an XMP sidecar next to the RAW.

</details>

<details>
<summary><strong>Search — how to find photos</strong></summary>

Open gallery search and type words separated by spaces. **No special syntax needed for most things** — a place, camera, lens, filename, or a date like `2024` / `2024-05` just works. Use `key:value` when you want to force a field or compare numbers.

| Kind | Example |
|------|---------|
| Place | `tokyo` · `Taipei` · `hong kong` · `city:tokyo` · `country:jp` |
| Camera / lens | `sony` · `canon` · `70-200` · `camera:canon` · `lens:70-200` |
| File name | `_dsc` · `IMG_1234` · `filename:_dsc` |
| Date | `2024` · `2024-05` · `date:2024-05` |
| ISO / year (comparison) | `iso<=800` · `iso under 800` · `year>=2024` |
| Format | `format:raw` · `format:jpeg` · `format:cr3` |
| GPS / faces | `has:gps` · `has:face` · `no:face` *(face filters: Plus only)* |
| Free text + filter | `jet takeoff camera:sony iso<800` *(Plus: unmatched free text uses AI)* |

Place names work offline: a built-in database of 100,000+ cities and landmarks resolves your photos' GPS into searchable places during background indexing.

The search field stays read-only until indexing of the current folder completes. On Plus builds, semantic and face indexing run in the background; switching folders cancels indexing for the old one.

</details>

---

## Standard or Plus?

Both editions have the complete viewer: gallery, culling, Compare, star ratings, bookmarks, GPS map, metadata search, and the Adjust develop panel with export (JPEG / WebP / 16-bit TIFF).

| | Standard | Plus (DirectML) | Plus (CUDA) |
|---|:--:|:--:|:--:|
| Everything above — browse, cull, rate, compare, develop, export | ✅ | ✅ | ✅ |
| Search by describing the photo (`sunset on beach`) | — | ✅ | ✅ |
| Find photos with people (`has:face`) | — | ✅ | ✅ |
| AI noise reduction (Adjust panel export) | Chroma/luma NR | **SCUNet AI denoise** | **SCUNet AI denoise** |
| RAW demosaic | CPU Fast RAW | CPU Fast RAW | **CuPy** GPU (NVIDIA) |
| Install size (approx.) | ~0.8–1 GB | ~1.5–2 GB + ~700 MB models | ~1.5–2 GB + ~700 MB models |
| Comfortable with | 8 GB RAM | 16 GB RAM | 16 GB RAM + NVIDIA |

**DirectML is for AI search and denoise, not demosaic.** On Windows Plus, photo-description search and AI noise reduction both use ONNX Runtime with DirectML in both Plus editions. Plus (DirectML) and Standard decode RAW with **CPU Fast RAW**; only Plus (CUDA) adds **CuPy** GPU demosaic.

**AI noise reduction** runs a real-image denoise model (SCUNet) via ONNX Runtime — no PyTorch required. It downloads with the other AI models on Plus installs (~100 MB) and applies automatically on export/full-quality renders when noise reduction is enabled in Adjust; Standard keeps the existing chroma/luma noise reduction.

**CuPy is one frame at a time** (serialized), but a single RAW HQ demosaic is still typically **~1.25–1.4×** faster than CPU Fast RAW on a modern NVIDIA GPU — enough to notice when arrow-keying through high-quality RAW, not a multi-fold leap.

**Pick Standard** for a lean install and cull-by-eye. **Pick Plus (DirectML)** for AI search (recommended default on most PCs). **Pick Plus (CUDA)** only if you have NVIDIA and want the modest RAW HQ demosaic speedup (no multi-GB PyTorch wheel).

On Windows, the installer offers **Standard**, **Plus (DirectML)**, and **Plus (CUDA)**. On macOS, Plus uses Core ML for search and CPU Fast RAW for demosaic (no CuPy path).

**Semantic search** does not need PyTorch.


## Cameras & formats

**RAW from all major brands:** Canon (CR2/CR3), Nikon (NEF), Sony (ARW), Fujifilm (RAF), Olympus (ORF), Panasonic (RW2), Adobe DNG, and most other cameras. **Plus:** JPEG, TIFF, HEIF, and animated GIF / WebP.

**Nikon High Efficiency (HE/HE*) NEFs** open using the camera's built-in preview — browsing, culling and rating all work, and Adjust now edits them from that preview. It is an 8-bit starting point rather than sensor data, so white balance and highlight recovery have much less to work with than on a standard NEF; tone, colour, masks, crop and detail behave as they do on any JPEG. Standard and lossless NEFs develop from sensor data as usual.

**HDR stills** (HEIC / HEIF / AVIF / HDR TIFF) are tone-mapped to standard brightness so browsing stays fast.

<details>
<summary><strong>Which cameras show the focus point (F)</strong></summary>

| Brand | Support |
|-------|---------|
| Canon CR2/CR3, Nikon NEF, Sony ARW, Olympus ORF, Panasonic RW2 | Yes |
| JPEG / TIFF / HEIF | Sometimes (when the camera records a subject area) |
| Fujifilm RAF, Hasselblad 3FR, Pentax PEF, Samsung SRW, Sigma X3F | No |
| Typical Adobe DNG | Usually no |

</details>

## Maps & places

Press **M** on any geotagged photo to see where it was taken on an interactive map; click the coordinate badge to open Google Maps. Place names are resolved offline, so you can search `tokyo` or `city:taipei` without an internet connection.

To map an entire album at once, or add locations to photos that don't have GPS, see the companion app **[LocateIt](https://github.com/markyip/LocateIt)**.

## Your photos stay yours

RAWviewer never uploads anything. Search, maps, and AI features all run on your computer. The only network use is optional: the one-time AI model download (Plus edition) and fetching map tiles when you open the map. A local thumbnail cache speeds up the gallery; it stays on your machine and cleans itself up after 30 days of inactivity.

### Upgrading from an older version

Most caches keep working after an update. Stale thumbnails and EXIF rows refresh **automatically** the first time you reopen a folder (may feel slower once).

**Search / indexing speed** is different: if you already have a `~/.rawviewer_cache` from before v3, RAWviewer keeps the older (safer) performance mode so upgrades stay predictable. To unlock the newer, faster defaults:

1. Quit RAWviewer.
2. **Windows:** check **Clear existing cache** in the Setup wizard when reinstalling/upgrading, or run **`clear_cache.bat`** next to `RAWviewer.exe`.
   **macOS:** double-click **Clear Cache.command** from the release zip (or `bash clear_macos_cache.sh`).
3. Reopen your folder — the first index pass rebuilds cache; after that you get the improved performance.

This clears local cache and session state only — **not** your photos or XMP sidecars. Alternatively set `RAWVIEWER_PERF_V2=1` without wiping cache (advanced).

---

## Troubleshooting

**Requirements:** Windows 10+ · macOS 13+ · 8 GB RAM (16 GB+ recommended for Plus with large folders).

<details>
<summary><strong>All platforms</strong></summary>

| Problem | What to do |
|---------|------------|
| GPS map not showing | Press **M** in single-image view; the map only appears when the photo has embedded GPS coordinates |
| HDR HEIC/TIFF looks flat or too dark | HDR stills are tone-mapped to standard brightness by design in v3.0 |
| **P** / **J** no effect | **P**/**J** are RAW/DNG single view only; **P** is a fit-only Reconstruct preview, not the Highlights/Shadows sliders |
| Gallery slow on a huge folder (first open) | Normal — RAWviewer waits for capture-time sorting so gallery order is correct; instant when metadata is cached |
| Upgraded but search / gallery still feels slow | Run **`clear_cache`** once (see [Upgrading from an older version](#upgrading-from-an-older-version)), then reopen the folder |

To clear cache: **`scripts\Launch\windows\clear_cache.bat`** (Windows) · **`scripts/Launch/macos/clear_cache.sh`** (Mac)

</details>

<details>
<summary><strong>Windows</strong></summary>

| Problem | What to do |
|---------|------------|
| SmartScreen warning | More info → Run anyway |
| Slow AI search (**Plus**) | AI search already uses DirectML on both Plus editions — edition choice does not speed up search. For RAW HQ browse speed on NVIDIA, try **Plus (CUDA)** (~1.25–1.4× demosaic vs CPU) |
| Installer stuck on "Downloading models" (**Plus**) | Models (~700 MB) can take several minutes. Check firewall, VPN, or proxy if it fails — browsing still works; open gallery **Search** later to retry |
| Opened Setup again instead of the app | Launch **RAWviewer** from the Desktop shortcut — not **`RAWviewer_Setup.exe`** |
| AI search missing after install (**Plus**) | Open gallery **Search** → accept the download prompt |
| RAWviewer not in Open with | Re-run the installer (repair), or reinstall |
| Leftover cache after uninstall | Run **`uninstall.bat`** again, or delete `%USERPROFILE%\.rawviewer_cache` manually |
| Out of memory during AI indexing | Use **Standard** on 8 GB PCs, or see [memory tuning](docs/DEVELOPING.md#automatic-memory-tuning) |
| Low-RAM PC OOMs when relaunch restores the last folder | Restoring the last folder/file is **intentional**. On 8 GB PCs under pressure, use **Standard**, or opt out with `RAWVIEWER_DISABLE_SESSION_RESTORE=1` |
| RAW always shows demosaic, not embedded JPEG | Switch to **Embedded JPEG workflow** |
| Crash | Normal runs do **not** write session logs. Enable file logging with `RAWVIEWER_FILE_LOG=1` or **`run_with_debug_log.bat`**, then check `%LOCALAPPDATA%\RAWviewer\logs\` |
| Need a freeze / gallery log from the **installed** app | Double-click **`run_with_debug_log.bat`** next to `RAWviewer.exe` (or from the repo root). Reproduce the issue, quit, then send `%LOCALAPPDATA%\RAWviewer\logs\rawviewer_latest.log` |

</details>

<details>
<summary><strong>macOS</strong></summary>

| Problem | What to do |
|---------|------------|
| macOS blocks the app ("damaged" / won't open) | In the extracted folder, run `bash install_macos_app.sh` (see install steps above) |
| `bash: command not found` | Type `cd `, drag the extracted folder onto Terminal, press Return, then run the command again |
| Can't read Desktop/Documents | System Settings → Privacy → **Full Disk Access** → add RAWviewer |
| Search says models missing (**Plus**) | Open gallery search and click **Download** when prompted (needs internet once) |
| No **AI denoise** in Export menu | Expected — export-time SCUNet denoise was removed in 3.0.1; use Adjust chroma NR + JPEG/WebP/TIFF export. |
| Download failed (SSL / certificate error) | On a corporate VPN or proxy, add your organization's root certificate to **Keychain Access** and set it to **Always Trust** |
| Need to uninstall completely | Use **`Uninstall RAWviewer.command`** from the release zip — not Trash alone |
| Uninstall scripts missing | Re-download the release zip from [Releases](https://github.com/markyip/RAWviewer/releases/latest); scripts are inside the extracted folder |
| "Out of memory" / heavy swap during indexing | On 8 GB Macs, prefer **Lite** or wait for indexing to finish; see [memory tuning](docs/DEVELOPING.md#automatic-memory-tuning) |
| Killed on relaunch (`Killed: 9` in Terminal) | Session restore (reopen last folder) is **by design**. On 8 GB Macs under jetsam pressure, try **Standard**, `RAWVIEWER_DISABLE_SESSION_RESTORE=1` (opt-out), or `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| Gallery still stutters on a huge folder | Run **`clear_cache.sh`** and reopen the folder |

</details>

---

## Looking ahead

**3.1.0** is the [stability / performance patch vs 3.0.2](#unreleased-310-stability--performance-vs-302). The items below are **later trains (3.2+)**. Much of this **already exists in the `development` tree** (some only in the dirty working tree). It is **kept**; it is **not** what 3.1.0 claims to ship.

Rule of thumb: **if it can ship in Plus, it counts as feasible** even when Standard must omit it (size / no ML).

### Later trains (3.2+) — in the tree, not 3.1.0

| Area | Where | Notes |
|------|--------|-------|
| **Masks tab** (brush, luminance, colour, invert, grouping) + Plus AI masks / Depth / Blur background | `development` | Not a 3.1.0 product claim |
| **Auto Tone** (WB + Light + straighten; no P-bake / VLM look) | `development` | |
| **Merges** — HDR, panorama, panorama HDR, focus stack | `development` | |
| **Raster develop** (JPEG / TIFF / WebP / PNG / BMP + XMP) | `development` | HE-NEF *refuse* fix is the 3.1.0 exception |
| **ColorChecker**, manual distortion, anamorphic desqueeze | `development` | DPReview sample button is dirty-tree |
| **Share** WhatsApp / Signal (JPEG for RAW) and AirDrop | `development` | |
| **AI Upscale 2×** | `development` | Plus denoise *reliability* is 3.1.0 |
| **Color labels / keywords** | dirty tree | Label **cache persist** is 3.1.0-targeted hardening, not the catalog UI |
| **RAW+JPEG pairing**, Adjust dock L/R, LUT thumbs, splash | dirty tree | |
| **Export template / GPS strip / watermark / queue** | dirty tree | Queue must not drop jobs *when that train ships*; watermark uses a **corner patch** (not a full-page overlay) as a memory choice |
| **Headless MCP**, Share to AI agent | dirty tree | GUI does **not** live-reload MCP sidecars |
| **In-app UI language** | dirty tree (`i18n/`) | Catalog on disk; UI still English until `QTranslator` |

### Still not done

| Rank | Item | Feasibility | Effort | Notes |
|------|------|-------------|--------|-------|
| 1 | **MCP live-reload in the GUI** | **High** | M | Headless MCP already writes XMP + previews. Adjust / gallery do not watch those sidecars — re-open the photo (or Adjust) to see agent edits |
| 2 | **In-app UI language** | **High** | M | `i18n/catalog.json` holds EN / zh-TW / ja wording. The app is not wired to `QTranslator` yet; the UI stays English |
| 3 | **Cold-folder edited tile regen** | **High** | M | Save-from-Adjust already bakes editor-aligned thumbs; a folder you have never opened in Adjust can still show the embedded JPEG |
| 4 | **User-facing upscale strength** | **High** | S | Real-ESRGAN can look over-sharp at full strength; the blend exists as an env var, not a slider |
| 5 | **Negative clicks for AI Selection** | **Medium–High** | M | The model accepts “not this” points; only positive clicks are sent. Trim over-reach with the brush |
| 6 | **Named People / familiar-face clustering** | **Medium** | L | Plus can filter `has:face`. There is no named People library or “this is the same person” clustering |
| 7 | **Generative editing via a local server** | **High** | S | **Shelved** in-app. Backend parked on `generative-editing`. Revival is a local OpenAI-compatible `/v1/images/edits` server, not an embedded 6–7 GB model |
| 8 | **DNG export / round-trip** | **Medium** | L | Writer removed 2026-07; needs a real DNG path, not a stub |
| 9 | **Windows HDR display path** | **Medium** | L | macOS EDR was removed for Fast RAW speed; Windows stays SDR tone-map |
| 10 | **Restore macOS EDR alongside Fast RAW** | **Low–Medium** | L | Previously fought the fast load pipeline |
| 11 | **VLM-assisted Auto** | **Low–Medium** | L | Product Auto is WB + Light + straighten only — no pixel/VLM look. A vision-model Auto is out of scope |
| 12 | **Google Drive browse / edit / XMP sync** | **Medium** | L | Local cache sync + OAuth; not started |
| 13 | **Edit Nikon HE/HE\* NEF as sensor data** | **Low** | L+ | Today: embedded JPEG. True RAW waits on a LibRaw HE mosaic decoder |

**Current limits (not a wishlist):**
- **Cold gallery tiles** for never-opened-in-Adjust edits may still show embedded JPEG.
- **Nikon HE-NEF** develops from the embedded JPEG (8-bit) once Adjust is allowed (3.1.0 refuse fix).
- **No memory-tier picker** — RAM caps are automatic.
- **3.2+ only:** AI mask / upscale weights on first use; Smart Object returns one mask; watermarks JPEG/WebP only; Apply to Burst Group copies tone / colour / noise, not crop, dodge/heal, or local masks.

---

## For developers

Build scripts, environment variables, memory tuning, and architecture notes: **[docs/DEVELOPING.md](docs/DEVELOPING.md)**. Launch helpers live under **[`scripts/Launch/`](scripts/Launch/README.md)** (`windows/` · `macos/` · `macos/release/`). Pull requests welcome.

### Build from source (summary)

| Platform | Command | Output |
|----------|---------|--------|
| **Windows** | `scripts\Launch\windows\build_windows.bat` | `dist/RAWviewer_Setup.exe` (wizard: Standard / Plus DirectML / Plus CUDA) |
| **macOS Plus** | `./scripts/Launch/macos/build_macos_full.sh` | `dist/RAWviewer.app` + release zip |
| **macOS Standard** | `./scripts/Launch/macos/build_macos_lite.sh` | `dist/RAWviewer_Lite.app` + release zip |

Day-to-day without packaging: `pixi install && pixi run start`, or `scripts\Launch\windows\run_debug.bat menu` / `./scripts/Launch/macos/launch_dev_full.sh`.

**Plus (CUDA)** installs `cupy-cuda12x` for GPU demosaic (serialized, typically ~1.25–1.4× vs CPU Fast RAW; no multi-GB PyTorch). **Plus (DirectML)** and **Standard** use CPU Fast RAW. On Windows, AI search uses ONNX DirectML in both Plus editions; macOS Plus uses Core ML.

## Support

1. Check [Troubleshooting](#troubleshooting) above
2. Search [existing issues](https://github.com/markyip/RAWviewer/issues)
3. Open a new issue with OS version, steps, and logs if possible

## Credits

RAWviewer stands on excellent open-source work, including:

- **[LibRaw](https://www.libraw.org/)** / **[rawpy](https://github.com/letmaik/rawpy)** — RAW decoding
- **[MobileCLIP](https://github.com/apple/ml-mobileclip)** (Apple) — on-device photo-description search (Plus edition)
- **[Qt / PyQt6](https://www.riverbankcomputing.com/software/pyqt/)** — application framework
- **[CuPy](https://cupy.dev/)** — optional NVIDIA GPU demosaic on Windows Plus (CUDA)
- **[onnxruntime](https://onnxruntime.ai/)** — MobileCLIP ONNX inference (Windows Plus)

## License

MIT — see [LICENSE](LICENSE).

## ☕ Buy me a coffee

If RAWviewer helps your workflow, you can [buy me a coffee](https://www.buymeacoffee.com/markyip).

---

**Enjoy your photos.** 📸
