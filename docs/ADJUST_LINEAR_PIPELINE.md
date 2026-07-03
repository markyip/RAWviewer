# Adjust Panel — Linear Edit Pipeline (Implementation Plan)

**Status:** Implemented (v2.5+ dev branch)  
**Toggle:** **E** in single-image view (RAW/DNG)  
**Tests:** `PYTHONPATH=src python3 scripts/phase_develop_adjust_linear.py`

## Goal

Replace **8-bit sRGB / gamma-space** adjust preview with a **scene-linear** pipeline so exposure, WB, and tone controls behave predictably and align with Lightroom / Camera Raw ordering at a lite tier. Non-destructive settings are stored in XMP sidecars; exports can bake pixels or copy RAW + XMP for round-trip editing.

---

## Implementation summary

| Area | Status | Notes |
|------|--------|-------|
| Scene-linear decode (AHD, 16-bit) | Done | Half-res preview; full-res export |
| PV2012 tone (ProcessVersion 11.0) | Done | Base curve + HS/W/B in perceptual space |
| Point tone curve UI | **Hidden** | Code in `tone_curve_widget.py`; `_SHOW_TONE_CURVE_UI = False` in panel |
| Parametric tone regions | **Hidden** | PV Shad / Dark / Light / High (same flag) |
| HSL 分色 (8 colors) | **Hidden** | Code in `raw_hsl.py`; `_SHOW_HSL_UI = False` in panel — also has an open cv2 HSV-scale bug (see below) |
| Detail (Sharpness / Clarity / Defringe) | Done | Display-linear, after tone map |
| Chroma / Luma NR | Done | Both bilateral filter, no NLM/uint8 quantization; Luma NR is new (see Performance review #11/#12) |
| Recovery baseline | Done | “Use recovery look” = P-key tone as adjust start |
| Recovery ↔ Adjust UX | Done | P blocked while Adjust open; opening Adjust disables P preview |
| Recovery slider hints | Done | Shadows +40 / Highlights −35 readouts when recovery on |
| Multi-format export | Done | TIFF / JPEG / WebP / DNG (settings or baked) |
| XMP sidecar I/O | Done | Basic + tone curve + HSL; as-shot Temp |
| Background preview worker | Done | 80 ms throttle; generation merge; uint8 encode guard |
| sRGB encode LUT | Done | `_encode_srgb8`/`_encode_srgb16` use a precomputed LUT instead of per-pixel `np.power`; ~2.7× faster (see Performance review) |
| Saturation / Vibrance color fix | Done | Now scales HSV S in gamma-encoded space via a LUT round-trip, not additive chroma in scene-linear space (see Performance review) |
| White balance dropper | Done | Small icon-only button inline in Temperature row; click a neutral area to solve Temperature/Tint (GPU single-view only) |
| Gallery "edited" badge | Done | Lightweight pencil badge on tiles with a saved XMP sidecar; thumbnail pixels unchanged (see Gallery integration) |
| Keyboard / focus fixes | Done | Shortcuts work with panel open |
| Value readout styling | Done | Blue `#90CAF9` (was low-contrast gray on dark card) |

**Not supported:** Lightroom masks, local adjustments, layers, per-channel RGB tone curves.

---

## Pipeline (edit mode)

```
RAW file
  → LibRaw postprocess (recovery_decode_params, demosaic=AHD)
      • 16-bit linear (gamma=(1,1))
      • use_camera_wb=True, highlight reconstruct
      • half_size=True for live preview; full-res for export
  → float32 scene-linear (÷ 65535)
  → WB (Temperature / Tint vs as-shot reference)
  → Exposure2012 (linear multiply)
  → chroma denoise (bilateral, Cb/Cr only; preview skips unless panel NR on)
  → luma denoise (bilateral, Y only; runs whenever Luma NR slider > 0)
  → PV2012 tone (unless recovery baseline — see below)
      • Built-in medium-contrast base curve
      • Point curve LUT (when UI enabled + crs:ToneCurvePV2012)
      • Parametric Shadows / Darks / Lights / Highlights (when UI enabled)
      • Contrast / Highlights / Shadows / Whites / Blacks
  → tone map to display
      • Default: luminance-preserving Reinhard
      • Recovery baseline: linear_tone_map_to_display + local shadow/highlight polish
  → saturation / vibrance (HSV S-scale, round-tripped through sRGB OETF LUT — see below)
  → HSL 分色 (8 colors, UI hidden)
  → sharpness / clarity / defringe
  → sRGB OETF → uint8 (preview) or uint16 (TIFF / baked DNG export)
```

Browse / gallery / non-edit paths **unchanged** (embedded JPEG + fast 8-bit LibRaw).

---

## Adjustments implemented

### Basic — Light

| Control | XMP key | Range | Pipeline stage |
|---------|---------|-------|----------------|
| Exposure | `Exposure2012` | −5 … +5 EV | Scene-linear multiply (fixed range = LR / XMP standard) |
| Contrast | `Contrast2012` | −100 … +100 | PV2012 perceptual |
| Highlights | `Highlights2012` | −100 … +100 | PV2012 perceptual |
| Shadows | `Shadows2012` | −100 … +100 | PV2012 perceptual |
| Whites | `Whites2012` | −100 … +100 | PV2012 perceptual |
| Blacks | `Blacks2012` | −100 … +100 | PV2012 perceptual |

### Tone curve (UI hidden — core validation phase)

| Control | XMP / internal | Notes |
|---------|----------------|-------|
| Point curve | `crs:ToneCurvePV2012` / `_tone_curve_pv2012` | Set `_SHOW_TONE_CURVE_UI = True` in `adjust_panel.py` to re-enable |
| PV Shadows | `ParametricShadows` | Hidden with tone curve flag |
| PV Darks | `ParametricDarks` | |
| PV Lights | `ParametricLights` | |
| PV Highlights | `ParametricHighlights` | |

Pipeline and XMP I/O for point/parametric curves remain in code; only the panel widgets are omitted for now.

### Basic — Color

| Control | XMP key | Range | Pipeline stage |
|---------|---------|-------|----------------|
| Temperature | `Temperature` | As-shot ± slider | Scene-linear WB vs as-shot Kelvin |
| Tint | `Tint` | −150 … +150 | Green–magenta on G channel |
| Saturation | `Saturation` | −100 … +100 | HSV S-scale, gamma-encoded space (see Performance review) |
| Vibrance | `Vibrance` | −100 … +100 | HSV S-scale, low-sat weighted, gamma-encoded space |

**Temperature default:** On first open (no XMP), Temp = **as-shot** from EXIF / RAW metadata (`AsShotTemperature` internal key). LibRaw uses `use_camera_wb=True`, so slider at as-shot = neutral.

### White balance dropper

Small icon-only button (eyedropper, `fa5s.eye-dropper`) inline at the end of the
Temperature slider row (`adjust_panel.py`) — matches the app's existing muted
icon-button language (`#B0B0B0` default / `#90CAF9` accent when armed, same
rgba border/background as the Chroma NR toggle) rather than a labeled button.
Click it, then click a neutral gray/white area in the image; Temperature/Tint
are solved so that pixel becomes neutral, then applied and saved like a normal
slider release. Esc (or the button again) cancels an armed pick without
sampling.

- **Coordinate mapping**: reuses `GpuImageView`'s existing scene-point pattern
  (same one used for double-click-to-zoom). `GpuImageView.set_color_pick_mode(True)`
  arms a one-shot crosshair-cursor click that emits `colorPickRequested(QPointF)`
  in image-pixel coordinates instead of starting a pan/export drag.
- **Sample source**: averages a 7×7 neighborhood from `_adjust_preview_base_rgb`
  (the scene-linear, pre-WB edit-base buffer) at the clicked point — not the
  post-tone-map / post-gamma preview pixmap, so the solve isn't confused by
  tone-mapping or other slider adjustments already applied.
- **Math** (`raw_adjustments.solve_white_balance_from_sample`): an exact inverse of
  `_apply_wb_tint`, not new color science. Temperature scales R/B oppositely
  relative to G, so the R/B ratio depends only on Temperature — solved by
  bisection (monotonic over 2000-12000K). Tint (which only scales G) is then
  solved algebraically so G matches. Re-applying the solved values makes the
  clicked pixel exactly neutral, by construction.
- **Known limitation**: Tint's underlying model only scales G by ±10%
  (`raw_adjustments._apply_wb_tint`: `1 - (tint/150)*0.1`), so samples with a
  strong green/magenta cast clamp at the ±150 Tint boundary without fully
  neutralizing — a pre-existing constraint of the WB model, not something the
  dropper introduces. Samples requiring a Temperature outside 2000-12000K clamp
  the same way.
- **Scope**: GPU single-image view only (`self.gpu_view`); not wired into the
  legacy `QScrollArea`/`QLabel` fallback (`RAWVIEWER_GPU_VIEW=0`) or the Compare
  view, neither of which have the same scene-point click machinery.

### HSL 分色 (UI hidden)

Eight colors: **Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta**.

| Per color | XMP keys | Range |
|-----------|----------|-------|
| Hue | `HueAdjustment{Color}` | −100 … +100 |
| Saturation | `SaturationAdjustment{Color}` | −100 … +100 |
| Luminance | `LuminanceAdjustment{Color}` | −100 … +100 |

UI: color dropdown + three sliders; per-color values cached when switching colors.
**Hidden** — set `_SHOW_HSL_UI = True` in `adjust_panel.py` to re-enable. `raw_hsl.py`'s
`_rgb_to_hsv`/`_hsv_to_rgb` also has an open colorspace-scale bug (see Performance
review #6); fix that before re-enabling.

### Detail

| Control | XMP key | Range | Pipeline stage |
|---------|---------|-------|----------------|
| Sharpness | `Sharpness` | 0 … 150 | Display-linear unsharp |
| Clarity | `Clarity2012` | −100 … +100 | Local contrast (display-linear) |
| Defringe | `Defringe` | 0 … 100 | Purple/green fringe reduction, edge-gated (see Performance review #8) |

### Noise

| Control | XMP key | Range | Notes |
|---------|---------|-------|-------|
| Luma NR | `LuminanceNoiseReduction` | 0 … 100 | Bilateral filter on Y only; chroma untouched |
| Chroma NR (toggle) | `ColorNoiseReduction` | 0 or 50 when On | Bilateral filter on Cb/Cr only; luminance untouched |

Both use an edge-aware bilateral filter directly on float32 (`raw_chroma_denoise.py`)
— no NLM, no uint8 quantization step. Luma NR uses a gentler sigmaColor range
(0.015-0.065) than Chroma NR (0.03-0.15) since luminance carries real image
detail; both keep hard edges sharp regardless of strength (bilateral filter is
edge-aware by construction — see Performance review #11/#12).

Preview: Chroma NR only when toggle On. Export: Chroma NR when toggle On or
`RAWVIEWER_EDIT_CHROMA_DENOISE=1`. Luma NR applies whenever its slider is
non-zero, in both preview and export (same as any other slider, no separate
env-var force-on path).

### Recovery baseline (session UI flag)

| Control | Internal key | Notes |
|---------|--------------|-------|
| Use recovery look | `_recovery_baseline` | Not written to XMP |

When enabled:

- Skips PV2012 tone; uses P-key recovery tone map + local shadow/highlight recovery.
- **Highlights / Shadows sliders** move to hint readouts (`+40` / `−35`) so the panel reflects recovery; hints do not disable recovery tone.
- Moving **Contrast**, **Whites**, **Blacks**, or parametric tone controls clears recovery baseline.
- Moving **Highlights** or **Shadows** off the hints clears recovery and switches to PV2012 tone using the new values.
- **P** recovery preview is disabled while Adjust is open; use **Use recovery look** instead.
- Exposure / WB / Sat / Vibrance / Detail still apply on top of recovery tone (HSL UI hidden — see above).

Constants: `RECOVERY_BASELINE_SHADOWS2012`, `RECOVERY_BASELINE_HIGHLIGHTS2012` in `raw_adjustments.py`.

---

## UI (`rawviewer_ui/adjust_panel.py`)

- Floating panel (**E**), draggable card, scrollable content
- Sliders: immediate value readout in **blue** (`#90CAF9`); **80 ms** throttled preview; XMP on release
- Click value label → reset single slider; **Reset** → all defaults + as-shot Temp
- **Tone curve** graph — **hidden** (`_SHOW_TONE_CURVE_UI = False`) during core-function validation
- **HSL** section — **hidden** (`_SHOW_HSL_UI = False`); pipeline/XMP I/O still present
- **Chroma NR** toggle
- **Use recovery look** button
- **Export…** menu (see below)
- Sliders / buttons use `NoFocus` so app shortcuts keep working

---

## Export formats

| Menu option | Output | Use case |
|-------------|--------|----------|
| 16-bit TIFF (baked) | Full-res AHD + pipeline + optional embedded XMP | Archival / print |
| JPEG (baked) | 8-bit sRGB, Q=92 | Sharing |
| WebP (baked) | 8-bit sRGB, Q=88 | Sharing |
| DNG — copy RAW + XMP settings | Original file copy + sidecar | Lightroom / Camera Raw round-trip |
| DNG — baked 16-bit RGB | Display-referred RGB in DNG container | Editors that open RGB DNG |

Export writes XMP to the **source** file before baking (except settings-only DNG copies to destination + new sidecar).

---

## XMP sidecar

Written on slider / curve release and before export:

- `crs:ProcessVersion` = `11.0`
- `crs:ToneCurveName2012` = `Linear`
- All keys in the table above (non-default values only)
- `crs:ToneCurvePV2012` point list when curve is non-linear
- `DefringePurpleAmount` / `DefringeGreenAmount` when Defringe > 0

**Important:** `write_xmp_adjustments()` **rewrites** the sidecar with supported global fields only. Lightroom **masks / local adjustments** are lost — back up `.xmp` before saving from RAWviewer.

---

## Modules

| Module | Role |
|--------|------|
| `raw_tone_recovery.py` | Decode params; recovery tone map; local S/H recovery |
| `raw_pv2012.py` | PV2012 base curve + parametric tone |
| `raw_tone_curve.py` | Point LUT + parametric regions + point helpers |
| `raw_hsl.py` | 8-color HSL in HSV space |
| `raw_detail_enhance.py` | Sharpness, Clarity, Defringe |
| `raw_chroma_denoise.py` | Chroma (Cb/Cr) + luma (Y) bilateral denoise (OpenCV) |
| `raw_edit_pipeline.py` | Shared pipeline + export dispatch |
| `raw_adjustments.py` | XMP I/O, defaults, `apply_adjustments_to_linear` |
| `rawviewer_ui/adjust_panel.py` | Adjust panel UI |
| `rawviewer_ui/tone_curve_widget.py` | Draggable point curve editor (optional UI) |
| `unified_image_processor.py` | `decode_raw_edit_base()` |
| `main.py` | Edit-base worker, preview worker, export worker, **E** shortcut |
| `scripts/phase_develop_adjust_linear.py` | Smoke tests |

---

## Environment

| Variable | Default | Effect |
|----------|---------|--------|
| `RAWVIEWER_EDIT_CHROMA_DENOISE` | `1` | Chroma NLM on export when panel NR off |

---

## Recent fixes (dev branch)

| Issue | Fix |
|-------|-----|
| Recovery look + P preview conflict | Adjust open disables P; P blocked while Adjust visible |
| Recovery tone resolution | Tone map at 2048 edge, upsample to edit-base size |
| Sharpness → black preview | `float32` bases use linear path; float pixmap encode via `_encode_srgb8` |
| Unsharp halos / blow-out | Detail stage clips display-linear to [0, 1] |
| Shortcut focus stolen by panel | `NoFocus` on widgets; `GpuImageView` routes shortcuts |
| Value numbers invisible on dark card | Default readout color `#90CAF9` |

---

## Verification checklist

1. Open RAW → **E** → wait for edit base (linear demosaic, not embedded JPEG)
2. Drag **Exposure** / **Shadows** — PV2012 tone, not gamma add
3. **Use recovery look** — Shadows/Highlights readouts jump to +40/−35; preview matches recovery tone
4. **Saturation** / **Vibrance** — consistent "pop" across hues (red, green, blue, skin tones); no hard clipping to a flat fully-saturated color on any one hue
5. **Sharpness** / **Clarity** — no black frame; image stays visible
6. **Export…** — TIFF / JPEG / WebP / DNG options
7. Shortcuts (**P**, **J**, arrows, etc.) work with Adjust panel open
8. **WB dropper icon** (end of Temperature row) — click, click a neutral gray/white
   area — Temperature/Tint update and the area turns neutral; Esc cancels an armed
   pick without sampling
9. Save an edit, switch to gallery — the edited file's tile shows the pencil badge
   (top-right corner); tile pixels stay the unedited thumbnail (see Gallery integration)
10. **Defringe** on a purple/green fringed edge (e.g. a backlit branch) — halo shrinks;
    a solid purple/green subject elsewhere in frame (flower, foliage) is not desaturated
11. **Blacks** right (+) lifts/lightens shadows, left (−) crushes them; **Whites** right (+)
    brightens/pushes highlights toward clipping, left (−) recovers them — not the reverse
12. **Shadows** past ~50 on a smooth dark gradient (e.g. a soft shadow falloff) — no
    banding/reversed local contrast in the shadow region
13. **Chroma NR** toggle on a neutral gray/white subject (e.g. a gray card, overcast
    sky) — no green tint after toggling on
14. **Luma NR** slider on a noisy high-ISO shot — grain visibly reduces; hard edges
    (e.g. a horizon line) stay sharp, no smearing
15. `PYTHONPATH=src python3 scripts/phase_develop_adjust_linear.py`

To re-enable tone curve UI: set `_SHOW_TONE_CURVE_UI = True` in `adjust_panel.py`.
To re-enable HSL UI: set `_SHOW_HSL_UI = True` in `adjust_panel.py` (fix the `raw_hsl.py`
colorspace-scale bug first — see Performance review #6).

---

## Known limitations / future work

- Tone curve + parametric PV sliders hidden in UI (pipeline still present)
- No per-channel (R/G/B) tone curves
- No mask / brush / gradient local edits
- Recovery baseline is preview-session UI state, not persisted in XMP
- Baked DNG is RGB container DNG, not a re-encoded sensor RAW
- Point curve uses linear interpolation between knots (not LR cubic spline)

---

## Gallery / cache integration (2026-07-03)

**Background:** single-image view (RAW, Adjust panel closed) already applies saved
XMP adjustments — `UnifiedImageProcessor.process_full_image()` is the sole caller
of the full-resolution decode tier and always runs `_apply_sidecar_if_needed()`
(`unified_image_processor.py`). The gap is the **gallery grid**: thumbnails come
from `ThumbnailExtractor`/embedded-JPEG extraction, which never applies the
sidecar, and none of the cache tiers (`thumbnail`/`grid`/`preview` in
`image_cache.py`) key on an adjustment fingerprint — so a tile never reflects an
edit no matter how long you wait or how many times you reopen the folder.

### Implemented now: lightweight "edited" badge (no pixel re-render)

A small pencil badge (top-right corner of the tile, mirroring the existing
bottom-right bookmark star / top-left burst-count badges) shows on any gallery
tile whose file has a saved XMP sidecar — the thumbnail pixels themselves are
**not** re-rendered through the adjust pipeline.

- `rawviewer_ui/widgets.py` (`ThumbnailLabel`): `_gallery_edited` flag,
  `set_gallery_edited()`, `_update_edited_badge()` / `_position_edited_badge()` —
  copied from the existing `_bookmark_badge` trio, same mechanism (child `QLabel`,
  absolute-positioned, shown/hidden, repositioned in `resizeEvent`).
- `rawviewer_ui/gallery_view.py` (`JustifiedGallery`): `refresh_gallery_edited_visuals()`
  mirrors `refresh_gallery_bookmark_visuals()`; also set at tile-build time
  alongside the existing bookmark/burst badge calls.
- `main.py`: `_is_gallery_path_edited(file_path)` — the state lookup backing the
  badge. Deliberately **not** a full `load_adjustments_for_file()` +
  `is_default_adjustments()` parse (that does EXIF/rawpy calls to resolve
  as-shot temperature — too heavy to run per visible tile). Instead it checks
  whether the `.xmp` sidecar file exists at all
  (`raw_adjustments.resolve_xmp_path` + `os.path.isfile`), which is an exact,
  filesystem-cheap proxy: `write_xmp_adjustments()` already **deletes** the
  sidecar when settings return to default (`raw_adjustments.py`), so existence
  alone means "non-default adjustments saved" — no parsing needed for a hint-only
  badge. `_on_adjust_panel_editing_finished` calls
  `_refresh_gallery_bookmark_visuals()` (now also refreshes the edited badge)
  right after the XMP write, so the badge updates immediately without needing a
  folder reload.
- Caveat: a foreign, non-RAWviewer XMP sidecar with no `crs:` adjustment keys
  (e.g. keywords-only XMP from another tool) would also light the badge — a rare
  false positive acceptable for a hint-only indicator.

### Deferred for future reference: regenerating the actual edited thumbnail pixels

If a literal "gallery shows the edited look" experience is wanted later instead
of (or in addition to) the badge, this is the lowest-risk path — additive only,
does not change the default (unedited) fast path at all:

1. **Invalidate, don't restructure.** On XMP save
   (`_on_adjust_panel_editing_finished`, `main.py`, and the export worker's
   `write_xmp_adjustments_for_file` call) call `image_cache.invalidate_file()`
   for the `thumbnail`/`grid`/`preview` tiers only — leave the `full_image` tier
   alone, since `process_full_image()` already re-applies the sidecar on every
   read of that tier regardless of caching.
2. **Make thumbnail generation sidecar-aware, but only pay for it when needed.**
   `UnifiedImageProcessor.process_thumbnail()`
   (`unified_image_processor.py`) currently never calls
   `_apply_sidecar_if_needed()`. Add the same call `process_full_image()` already
   makes, gated by the existing `is_default_adjustments()` early-exit inside
   `_apply_sidecar_if_needed()` — so files with no saved adjustments (the
   overwhelming majority) take exactly the same code path as today, and only
   files with a non-default XMP sidecar pay the extra cost of running the linear
   pipeline during thumbnail extraction.
3. **Trigger a refresh, not a rebuild.** After step 1's invalidation, call
   something like `gallery_justified.request_thumbnail_refresh(file_path)` (new,
   or reuse the existing `thumbnail_ready` signal path via
   `image_load_manager`) so only the affected tile re-requests its thumbnail —
   the pooled/visible-widget model already used for
   `refresh_gallery_*_visuals()` means this does not require rebuilding the grid.
4. **Cost/risk to flag before building this:** thumbnail generation for edited
   files becomes measurably slower (full linear pipeline vs. a JPEG/half-size
   decode) — acceptable since it is one-time-until-next-edit and scoped to a
   small subset of files, but should be verified against a folder with many
   edited RAWs before shipping, to make sure regenerating dozens of edited
   thumbnails at once (e.g. after a batch XMP edit) doesn't stall the gallery
   thread pool.
5. **Filmstrip/preload** (`_PRELOAD_NEXT_COUNT`/`_PRELOAD_PREV_COUNT`,
   `main.py`) only prefetches `{"thumbnail","exif"}` for normal navigation
   (sidecar-unaware) and `{"exif","full"}` (sidecar-aware) only once zoomed to
   100% — would need the same treatment as step 2 if this is built, or the
   filmstrip will show stale previews even after the gallery grid is fixed.

---

## Performance review (2026-07-03)

Benchmarked on a synthetic 2000×3000 buffer (representative of a half-size decode
of a ~24 MP sensor — the live "E" preview size). The **80 ms preview throttle**
documented above is a scheduling debounce, not the actual per-tick compute cost;
real latency is dominated by full-frame numpy passes with no incremental
recompute.

| # | Finding | Measured | Status |
|---|---------|----------|--------|
| 1 | `_encode_srgb8` / `linear_to_export_uint16_srgb` used `np.where(...,  np.power(x, 1/2.4), ...)` over the whole frame — `np.power` dominated cost even though `raw_pv2012.py` already proved the LUT pattern works for the base curve | Encode alone: **~170–190ms**, ~70% of total no-op preview time | **Fixed** — nearest-index LUT (4096 entries, 8-bit; 65536, 16-bit) in `raw_tone_recovery.py` (`_encode_srgb8` / `_encode_srgb16`); ~65–70ms, 2.6–2.8× faster, <1 LSB error |
| 2 | No incremental recompute — every slider tick reruns the entire WB→exposure→NLM→PV2012→tonemap→sat→HSL→detail chain even when only one late-stage control changed | All-default preview ~150–270ms; typical multi-slider edit ~1.7–1.9s; far past the 80ms throttle window | Open — needs a staged cache (e.g. cache post-tonemap buffer, only rerun sat/HSL/detail when only those move) |
| 3 | Recovery baseline (`_tone_map_recovery_display`) reruns a `scipy.ndimage.gaussian_filter(sigma=22)` + downsample/upsample round-trip on every tick, even though Exposure/Sat/Vibrance/HSL/Detail are layered "on top of" recovery tone and don't need it recomputed | ~2.0–2.3s per tick with recovery on, the slowest path measured | Open — same staged-cache fix as #2, cache the recovery-tone-mapped base once per edit-base load |
| 4 | `UnifiedImageProcessor.decode_raw_edit_base(..., executor=self.process_pool)` accepts `executor` but never uses it — decode always runs in-process on the calling `QRunnable` thread, not offloaded to the process pool | N/A (code review) | Open — wire up or drop the parameter |
| 5 | `phase_develop_adjust_linear.py` only checks correctness on tiny (16×16–64×64) synthetic arrays; no benchmark at realistic preview resolution | N/A | Open — a perf regression (like #1) would not be caught by the existing suite |
| 6 | `raw_hsl.py`'s `_rgb_to_hsv` treats cv2's **float32** `COLOR_BGR2HSV` output (already H:0–360, S/V:0–1) as if it were the **uint8** convention (H:0–179, S/V:0–255) — `h * 2.0` doubles an already-0–360 hue (wraps/misassigns color bands), and `s / 255.0`, `v / 255.0` crush saturation/value to ~1/255 of true scale. `_hsv_to_rgb`'s uint8 packing then largely undoes the S/V crush on the way back out, but the per-color slider deltas (`ds`/`dv`, order ±1.0) are added *before* that undo, so they dominate the crushed (~0.004) true value — HSL Hue/Sat/Lum sliders would behave close to on/off rather than proportional, and hue-band assignment is wrong. Not visible while `_SHOW_HSL_UI = False`, but still runs for any file with pre-existing non-zero `HueAdjustment*`/`SaturationAdjustment*`/`LuminanceAdjustment*` XMP values (browse view, export) | Confirmed via direct `cv2.cvtColor` test; not benchmarked (no user-visible path while hidden) | **Open, not fixed** — out of scope for this pass (user asked to hide HSL, not fix it); fix before setting `_SHOW_HSL_UI = True` again |
| 7 | `_apply_saturation_vibrance` scaled chroma (`img - luma`) additively on **scene-linear** RGB with no upper clip. Linear-light channel spread is highly hue-dependent (green/yellow have large spread relative to luma, red/blue less so), so the same `+50` slider value clipped green/yellow to a flat, fully-saturated block (`ΔS` up to 0.54, hitting the gamut ceiling) while skin tones and blue sky barely moved (`ΔS` ~0.11–0.13) — the reported "colors look off, no pop" | Measured `ΔS` (HSV saturation, encoded output) across 6 test hues at Saturation +50: **0.11–0.54** (7× spread, several hues clipped to `S=1.0`) | **Fixed** — `_apply_saturation_vibrance` now round-trips scene-linear → sRGB-encoded (float LUT, `_encode_srgb_float01`/`_decode_srgb_float01` in `raw_tone_recovery.py`) and scales the HSV **S** channel there, bounded to `[0, 1]` by construction. Same 6 hues at +50 now measure **0.13–0.30** (~2× spread, no clipping). Cost: +~50–80ms per preview tick when Saturation/Vibrance is non-zero (HSV round-trip); no-op path unaffected |
| 8 | `apply_defringe` (`raw_detail_enhance.py`) had no edge/locality gate at all — any pixel whose hue leaned purple/green relative to its own chroma got desaturated, including uniform, unfringed regions, because `fringe / span` is scale-invariant along the purple/green hue axis rather than proportional to cast strength. A uniform purple patch (e.g. a flower petal, no edge anywhere) lost **90% saturation at Defringe=100** with nothing to correct. Separately, `green = g - 0.5*(r+b)` had half the gain of `purple = r+b-2*g` for the same absolute color imbalance — green fringing was flagged as half as severe as equally-strong purple fringing | Uniform purple patch: **90% saturation loss** at Defringe=100 with zero edges present; purple-vs-green raw mask ratio confirmed **exactly 2.00×** at every tested magnitude (d=0.02/0.05/0.10) | **Fixed** — `green` now uses `2*g - r - b` (exact negation of `purple`, so `fringe == \|r + b - 2*g\|`); added `edge_weight` gated on local luminance contrast (same Gaussian-blur trick as clarity/sharpness, sigma=1.5, soft-knee `edge/(edge+0.04)`) so the mask only engages near a real luminance transition. Uniform purple/green patches now measure **0% saturation loss** at Defringe=100; a genuine fringe pixel at a dark/bright edge still loses ~57-75%. 3 new smoke tests added (`test_defringe_*`) |
| 9 | `raw_pv2012._apply_whites_blacks` had Whites/Blacks **sign-inverted**: `white_pt = 1.0 + w*0.12` / `black_pt = b*0.12` made Blacks+100 *crush* shadows and Whites+100 *darken* highlights — backwards from every reference tool and from this app's own legacy gamma-space implementation of the same named controls, which get the direction right. Affects 100% of PV2012 Whites/Blacks usage, not an edge case | End-to-end test on a shadow/highlight ramp: `Blacks=+100` darkened 0.02→0.01 (should lighten); `Blacks=-100` lightened 0.02→0.060 (should darken); `Whites=+100` darkened 0.9→0.804 (should brighten); legacy path's own Blacks/Whites confirmed correct-direction on the identical test | **Fixed** — signs flipped: `white_pt = 1.0 - w*0.12`, `black_pt = -b*0.12`. Re-verified same ramp now moves in the expected direction both ways |
| 10 | `raw_pv2012._apply_highlights_shadows`'s shadow-lift branch (`Shadows2012 > 0`) used coefficient 0.42, but the shadow-region weight's slope reaches ~5.85 at y≈0.11 — any coefficient above ~0.17 makes the combined tone curve **locally decreasing** (a real, visible banding/contrast-reversal defect, not just theoretical). First appeared between Shadows=40 and 50. The Recovery-baseline preset hint (`Shadows=+40`) happened to sit just under the threshold; any user manually pushing Shadows past ~45 hit it. `raw_adjustments.py`'s legacy gamma-space duplicate (`_apply_highlights_shadows`/`_apply_masked_luminance_adjust`, still reachable for non-RAW/uint8 images) had the same class of bug and broke even earlier — Shadows=15 already showed violations — because its coefficient (0.55) is shared across three different region-weight shapes (shadow/black/highlight) of very different steepness (worst slopes -12.36 / -29.57 / -1.0 respectively) | PV2012: 50/1999 backward steps at Shadows=100, worst -0.0069, monotonicity break confirmed to start ~Shadows 40-50 (matches the derivative-based safe-coefficient calc `0.17/0.42×100≈40` exactly). Legacy: up to 596/3999 backward steps at Shadows=100, breaking as early as Shadows=15 (7 bad steps) | **Fixed** — PV2012 coefficient reduced 0.42→0.15 (safe margin under the computed 0.17 ceiling); re-verified monotonic at Shadows=100 and at all-sliders-maxed combos. Legacy: `_apply_masked_luminance_adjust` gained a `lift_up_strength` parameter (previously one hardcoded 0.55 shared by all three masks) so Shadow (0.07), Black (0.03), and Highlight/White (default 0.55, already safe) each get a coefficient sized to their own region-weight steepness rather than one value forced down to the strictest case. Re-verified monotonic for both Shadows and Blacks up to their extremes. 4 new smoke tests added (`test_pv2012_shadows_lift_stays_monotonic`, `test_pv2012_whites_blacks_sign_direction`, `test_legacy_gamma_highlights_shadows_stays_monotonic`, `test_legacy_gamma_blacks_stays_monotonic`) |

| 11 | `raw_chroma_denoise.apply_chroma_nlm` (superseded by `apply_chroma_denoise`, see #12) converted Cb/Cr from float to uint8 via a bare `.astype(np.uint8)`, which **truncates toward zero** rather than rounding. Cb/Cr fractional parts are effectively uniformly distributed, so truncation biases both channels ~-0.5 LSB (≈-0.00196 in [0,1]) low on average -- not randomly, systematically, every pixel, every call. The YCbCr→RGB inverse (`R = Y + 1.5748·Cr0`, `G = Y - 0.1873·Cb0 - 0.4681·Cr0`, `B = Y + 1.8556·Cb0`) turns a uniform negative bias in *both* Cb and Cr into less R, less B, and **more G** — a systematic green cast on every image that goes through Chroma NR, reported by a user as "casts a green shade" | Reproduced the bias in isolation (quantize/dequantize Cb+Cr with **zero** actual NLM denoising applied): mean RGB delta on a neutral-gray test image = `[-0.00308, +0.00128, -0.00363]`, reproducible to 4 decimal places across 15 random-noise seeds (not noise-dependent). Matches the predicted truncation bias (`-0.5/255 = -0.00196`) driven through the same inverse-transform coefficients almost exactly | **Fixed** (then superseded) — added `+ 0.5` before both `.astype(np.uint8)` casts (round-to-nearest, matching the pattern already used in the sRGB encode LUTs from finding #1). Re-verified: residual mean RGB delta dropped to ~1e-5–2e-4. Superseded by #12, which removes the uint8 round-trip entirely |
| 12 | Follow-up to a "is there a better NR than chroma NR" question: (a) chroma NR used NLM (`cv2.fastNlMeansDenoising`), one of the more expensive denoise algorithms, on a full preview/export buffer; (b) there was no luminance noise reduction at all — only Cb/Cr were ever touched, so grainy high-ISO detail noise went untreated | Benchmarked NLM vs. bilateral filter (`cv2.bilateralFilter`, applied directly on float32, no uint8 step) at realistic sizes: **19-22× faster** (1200×1800: 232ms→12ms; 2500×3750: 1105ms→50ms per channel). Bilateral also verified edge-preserving: a hard color-edge transition stays 0px wide even at max strength, while a flat noisy region's std-dev drops 71-81% depending on sigmaColor | **Implemented** — `apply_chroma_nlm` replaced with `apply_chroma_denoise` (bilateral, sigmaColor 0.03-0.15, no uint8 round-trip — also eliminates the whole bug class from #11 structurally, not just patched). Added `apply_luma_denoise` (bilateral on Y only, gentler sigmaColor 0.015-0.065 since luminance carries real detail unlike chroma) with a new **Luma NR** slider (`LuminanceNoiseReduction`, 0-100, `adjust_panel.py`/XMP) sitting above the existing Chroma NR toggle under a new "Noise" section. 6 new smoke tests added covering noise reduction, edge preservation, and pipeline wiring for both filters |

Numbers were captured with `raw_edit_pipeline.process_linear_edit_buffer` +
`linear_to_display_uint8` directly (no Qt/GUI thread involved); see the
conversation history or re-run the benchmark snippet against
`scripts/phase_develop_adjust_linear.py`'s helpers for a repeatable check.
