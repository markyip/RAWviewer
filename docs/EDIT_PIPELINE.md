# Edit Pipeline — Adjust Panel, Metadata & Roadmap

**Status:** Implemented (v2.5+ dev branch)  
**Toggle:** **E** in single-image view (RAW/DNG)  
**Tests:** `PYTHONPATH=src python3 scripts/phase_develop_adjust_linear.py`

> Renamed from `ADJUST_LINEAR_PIPELINE.md` (2026-07-04) — the scene-linear
> tone/color pipeline below is still the core of this doc, but it now also
> tracks feasibility investigations for adjacent editor features (rating/XMP
> compatibility, geometry/perspective tools, local/masked adjustments) that
> don't belong in the pipeline itself. See the **Roadmap investigations**
> section near the end for those.

## Goal

Replace **8-bit sRGB / gamma-space** adjust preview with a **scene-linear** pipeline so exposure, WB, and tone controls behave predictably and align with Lightroom / Camera Raw ordering at a lite tier. Non-destructive settings are stored in XMP sidecars; exports can bake pixels or copy RAW + XMP for round-trip editing.

---

## Implementation summary

| Area | Status | Notes |
|------|--------|-------|
| Scene-linear decode (AHD, 16-bit) | Done | Half-res preview; full-res export |
| PV2012 tone (ProcessVersion 11.0) | Done | Base curve + HS/W/B in perceptual space |
| Point tone curve UI | Done | `tone_curve_widget.py`; `_SHOW_TONE_CURVE_UI = True` in panel (re-enabled 2026-07-04) |
| Parametric tone regions | Done | PV Shad / Dark / Light / High (same flag); monotonicity bug fixed before re-enabling (see Performance review #14) |
| HSL (8 colors) | **Hidden** | Code in `raw_hsl.py`; `_SHOW_HSL_UI = False` in panel — also has an open cv2 HSV-scale bug (see below) |
| Detail (Sharpness / Clarity / Defringe) | Done | Display-linear, after tone map |
| Chroma / Luma NR | Done | Both bilateral filter, no NLM/uint8 quantization; Luma NR is new (see Performance review #11/#12). Chroma NR also gets a downsample/blur/upsample coarse pass for blotchy noise a bilateral kernel can't reach (see Performance review #20) |
| Recovery baseline | Done | “Use recovery look” = P-key tone as adjust start |
| Recovery ↔ Adjust UX | Done | P blocked while Adjust open; opening Adjust disables P preview |
| Recovery slider hints | Done | Shadows +40 / Highlights −35 readouts when recovery on |
| Multi-format export | Done | TIFF16 / JPEG / WebP baked only — DNG options removed (see Export formats) |
| XMP sidecar I/O | Done | Basic + tone curve + HSL; as-shot Temp |
| Background preview worker | Done | 80 ms throttle; generation merge; uint8 encode guard |
| sRGB encode LUT | Done | `_encode_srgb8`/`_encode_srgb16` use a precomputed LUT instead of per-pixel `np.power`; ~2.7× faster (see Performance review) |
| Saturation / Vibrance color fix | Done | Now scales HSV S in gamma-encoded space via a LUT round-trip, not additive chroma in scene-linear space (see Performance review) |
| White balance dropper | Done | Small icon-only button inline in Temperature row; click a neutral area to solve Temperature/Tint (GPU single-view only) |
| Compare with original (split view) | Done | Icon-only toggle in panel header; draggable divider shows original (left) vs live edited (right) over the same `GpuImageView` — see Compare with original section (2026-07-05) |
| Lens profile correction | Done | `lensfunpy` geometric distortion correction; toggle only shown when a profile matches the file's camera+lens; baked in at decode time, not per-tick (see Lens profile correction section, 2026-07-05) |
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
  → HSL (8 colors, UI hidden)
  → sharpness / clarity / defringe
  → sRGB OETF → uint8 (preview) or uint16 (TIFF16 export)
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

### Tone curve

| Control | XMP / internal | Notes |
|---------|----------------|-------|
| Point curve | `crs:ToneCurvePV2012` / `_tone_curve_pv2012` | Draggable graph (`ToneCurveWidget`); non-monotonic shapes are a valid user choice, not a bug |
| PV Shadows | `ParametricShadows` | |
| PV Darks | `ParametricDarks` | |
| PV Lights | `ParametricLights` | |
| PV Highlights | `ParametricHighlights` | |

Re-enabled 2026-07-04 (`_SHOW_TONE_CURVE_UI = True` in `adjust_panel.py`) after
fixing a monotonicity bug in `apply_parametric_tone_curve`
(`raw_tone_curve.py`) found during the re-enable review: `ParametricShadows`
+100 and `ParametricHighlights` -100 both made the combined tone curve
locally decreasing (banding), for the exact same reason as the
`raw_pv2012.py` shadow-lift bug fixed earlier (same `_smooth_weight`/
`_SPLIT_SHADOWS=0.25` region shape, worst slope ~5.89-5.90, needing
coefficient ≤~0.17). Fixed the same way: coefficient reduced 0.35→0.15. See
Performance review #14.

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

### Compare with original (2026-07-05)

Icon-only toggle in the panel header (`fa5s.columns`, same muted-icon-button
language as the WB dropper). When armed, shows a draggable-divider split view
over the current image: the unedited (all-default-adjustments) render on the
left of the divider, the live edited preview on the right. Drag the divider
(or click near it) to move the split; toggle the button again to exit.

- **Architecture**: purely an overlay on `GpuImageView`, no change to the
  existing zoom/pan/fit/pixmap machinery. `self._item` (the always-current
  edited pixmap) is left completely alone — a new `_compare_overlay_item`
  (`QGraphicsPixmapItem`, zValue above `_item`) holds the *original* render
  cropped to the region left of the divider, plus a `QGraphicsLineItem`
  divider and a small `QGraphicsEllipseItem` handle. Because these are normal
  scene items, they zoom/pan with the image for free, and the edited side
  keeps live-updating from ordinary slider drags without any special-casing
  (the overlay just sits on top of whatever `_item` currently shows).
- **Getting the "original"**: `_AdjustCompareOriginalWorker` (`main.py`) runs
  `apply_adjustments_to_rgb(base_rgb, DEFAULT_ADJUSTMENTS)` off the GUI
  thread — this hits `process_linear_edit_buffer`'s existing "adjustments are
  all default" fast path (decode + sRGB encode only, no WB/denoise/tone math),
  so it's cheap without needing its own preview-stage cache. Cached per file
  (`_adjust_compare_original_pixmap`/`_path`) so re-toggling compare on/off
  doesn't recompute; invalidated in `_reset_adjust_compare_state()`, called
  from both `_begin_adjust_editing_session` (new file / fresh editing
  session) and the Adjust-panel-close path in `_set_adjust_panel_visible` —
  the same two places that already reset `_adjust_preview_base_rgb`.
- **Divider interaction**: hit-tested in view-space pixels (`±8px`, mapped
  through `mapFromScene`) so the grab target stays a constant on-screen size
  regardless of zoom, the same cosmetic-pen pattern used for the composition
  grid overlay. Drag handling is intercepted in `mousePressEvent`/
  `mouseMoveEvent`/`mouseReleaseEvent` ahead of the existing pan/export-drag
  logic, mirroring how `_color_pick_mode` already takes priority there for
  the WB dropper.
- **Verified**: headless-Qt tests (`QTest.mousePress/mouseMove/mouseRelease`)
  confirming compare mode refuses to arm without an original pixmap, the
  overlay crop width tracks the split fraction exactly, dragging the divider
  moves the split monotonically in the dragged direction, and toggling off
  hides the overlay/divider/handle; a direct test that
  `_AdjustCompareOriginalWorker`'s output is byte-identical to
  `apply_adjustments_to_linear(base, DEFAULT_ADJUSTMENTS)` and differs from
  an edited render. Not added to `phase_develop_adjust_linear.py` (that suite
  is pixel-math-only and PyQt6-free by design) — this is GUI wiring around
  already-tested pipeline functions, verified the same ad hoc headless-Qt way
  the tone curve widget's drag lifecycle was.
- **Scope**: GPU single-image view only, same as the WB dropper.

### Lens profile correction (2026-07-05)

Automatic geometric distortion correction driven by a `lensfunpy` profile
lookup keyed on camera make/model + lens make/model + focal length +
aperture. New module: `raw_lens_correction.py`. Toggle in the panel (styled
identically to the Chroma NR on/off button) is **hidden by default and only
shown once a matching profile is confirmed** for the current file — no
profile, no button, per the original ask.

- **Dependency**: `lensfunpy` added to `pixi.toml` `[pypi-dependencies]`
  (cross-platform, no per-target section needed — prebuilt wheels bundle
  the lensfun C library + database for Linux/macOS/Windows). `pixi install`
  resolves cleanly; `lensfunpy.Database()` loads 948 cameras / 1304 lenses
  with no separate download step.
- **Lookup key**: `raw_lens_correction.lens_profile_key_from_exif()` builds
  `(camera_make, camera_model, lens_make, lens_model, focal_length,
  aperture)` from the dict `EXIFExtractor.extract_exif_data()` already
  returns — camera fields and formatted focal/aperture strings were already
  parsed there for display; lens model is pulled from the raw EXIF tag dict
  using the same `EXIF LensModel` / `MakerNote LensType` / `Lens` /
  `Image LensModel` / `Composite LensID` fallback chain already used for
  search indexing (`semantic_search.py`). Returns `None` (no lookup
  attempted) if any of the five fields is missing.
- **Matching is strict-only, never `loose_search=True`**: caught during
  development — `lensfunpy`'s `loose_search=True` is not a tighter fuzzy
  match, it's a "return every lens on this camera's mount" fallback when
  nothing matches (confirmed: a made-up lens name returned all ~80 Canon EF
  lenses in the database). Using it would silently apply a *wrong* lens's
  distortion correction to the image — actively corrupting geometry while
  claiming success. `lensfunpy`'s strict mode already normalizes case and
  whitespace on its own (confirmed: `"canon"`/`"Canon"` and
  `"EF16-35mm..."`/`"EF 16-35mm..."` both match), so strict-only is not
  overly brittle.
- **Applied once at decode time, not per-tick**: unlike every tone/color
  slider, this has no continuous "amount" -- it's either the exact profile
  match or nothing. `UnifiedImageProcessor.decode_raw_edit_base()` gained an
  `apply_lens_correction: bool` parameter; when true, the decoded (and
  already orientation-corrected) buffer is undistorted via
  `raw_lens_correction.apply_lens_correction()` before being returned as the
  edit base. This means toggling it triggers a full re-decode
  (`_begin_adjust_editing_session`, which already resets
  `_adjust_preview_base_rgb`/the compare-original cache/`PreviewStageCache`
  correctly via existing base-identity invalidation) rather than a
  pipeline-stage recompute — simpler than threading a new stage through
  `PreviewStageCache`, and correct, since geometry has to be settled before
  any of WB/denoise/tone/detail run on the buffer.
- **Correction math**: `lensfunpy.Modifier(lens, cam.crop_factor, w, h)` +
  `.initialize(focal, aperture, pixel_format=...)` +
  `.apply_geometry_distortion()` returns an `(h, w, 2)` float32 coordinate
  map, fed directly to `cv2.remap()` (already a dependency). Confirmed the
  map is independent of the `pixel_format` argument passed to `initialize()`
  (only affects vignetting, unused here) and that `cv2.remap` preserves
  dtype for both `uint16` (the edit base's actual dtype) and `float32`.
- **Availability check**: `UnifiedImageProcessor.lens_profile_available()`
  does the same EXIF-based lookup (cheap on repeat calls — EXIF is
  SQLite-cached) and is called from `_AdjustEditBaseWorker` alongside the
  decode itself, so there's no extra I/O round-trip just to decide whether
  to show the toggle. Result flows back through `_AdjustEditBaseSignals`
  (gained a third `bool` field) to `_on_adjust_edit_base_ready`, which calls
  `panel.set_lens_correction_available(...)`.
- **Persistence**: `LensCorrectionEnabled` (0.0/1.0) added to
  `DEFAULT_ADJUSTMENTS` — this alone gets it full XMP read/write for free
  via the existing flat-dict `crs:` sidecar mechanism, no new XMP code
  needed. Off by default: an automatic geometry change must never apply
  silently to a file that didn't have it enabled before.
- **Export**: `_AdjustExportWorker` reads `LensCorrectionEnabled` from the
  adjustments dict being exported and passes it to the full-resolution
  `decode_raw_edit_base()` call, so a baked export matches whatever the
  preview showed.
- **Verified**: `raw_lens_correction.py` has real smoke tests in
  `phase_develop_adjust_linear.py` (`test_lens_profile_key_from_exif`,
  `test_lens_correction_gates_and_corrects_only_known_profiles`) — a known
  camera+lens combo actually changes pixel positions (and preserves
  dtype/shape for both `uint16` and `float32`), an unmatched lens returns
  the input completely unchanged (the strict-only-matching regression
  guard), and missing/`None` EXIF is handled without raising. These tests
  skip cleanly (not fail) when run outside the pixi env, since this
  session's ad hoc `python3 scripts/...` invocations use a different,
  older environment than pixi's that doesn't have `lensfunpy` installed —
  run via `pixi run python3 scripts/phase_develop_adjust_linear.py` to
  actually exercise them. Also verified the full signal chain headlessly
  (Qt): toggle hidden by default, shown/hidden correctly as
  `set_lens_correction_available()` is called, toggling emits the dedicated
  `lens_correction_toggled` signal (not the generic preview-refresh path),
  and `_AdjustEditBaseWorker` threads `apply_lens_correction` through to
  `decode_raw_edit_base` correctly.
- **Not implemented**: vignetting and TCA (chromatic aberration) correction
  — `lensfunpy` supports both (`apply_color_modification`,
  `apply_subpixel_distortion`), but "aspect correction" was the ask; a
  reasonable future extension, not a gap in this pass. Auto-crop-to-remove-
  empty-borders is also not implemented — `lensfunpy`'s automatic scale
  (`scale=0.0`, the default) already minimizes empty edges from the geometry
  correction itself, and no border cropping was observed in testing, so this
  wasn't pursued further.

### HSL (UI hidden)

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
- **Tone curve** graph + PV Shad/Dark/Light/High sliders (`_SHOW_TONE_CURVE_UI = True`)
- **HSL** section — **hidden** (`_SHOW_HSL_UI = False`); pipeline/XMP I/O still present
- **Chroma NR** toggle
- **Lens correction** toggle — hidden unless a matching profile is found for the current file
- **Use recovery look** button
- **Compare** toggle (header, split-view icon) — see Compare with original
- **Export…** menu (see below)
- Sliders / buttons use `NoFocus` so app shortcuts keep working

---

## Export formats

| Menu option | Output | Use case |
|-------------|--------|----------|
| 16-bit TIFF (baked) | Full-res AHD + pipeline + optional embedded XMP | Archival / print |
| JPEG (baked) | 8-bit sRGB, Q=92 | Sharing |
| WebP (baked) | 8-bit sRGB, Q=88 | Sharing |

**Removed (2026-07-04):** "DNG — copy RAW + XMP settings" and "DNG — baked
16-bit RGB" were both removed from the export menu — a user reported the
baked-RGB DNG file could not be opened at all. The RGB-container-DNG approach
(a plain RGB TIFF with `DNGVersion`/`UniqueCameraModel` tags bolted on, no
real sensor mosaic/CFA data) was never a real DNG as far as most DNG readers
are concerned; fixing the Pillow crash that blocked it (Performance review
#13) made it *write* successfully but did not make the result *openable*.
Rather than ship a format that produces unopenable files, both DNG options
were dropped. `export_adjusted_dng_rgb`, `export_raw_with_xmp_settings`,
`EXPORT_FORMAT_DNG_RGB`, and `EXPORT_FORMAT_DNG_SETTINGS` no longer exist in
`raw_edit_pipeline.py`. If DNG output is revisited later, it needs a real DNG
writer (e.g. a proper CFA/mosaic + DNG tag set), not this shortcut.

16-bit TIFF writes via **`tifffile`** (`_write_16bit_rgb_tiff` in
`raw_edit_pipeline.py`), not Pillow — Pillow's `"RGB"` mode is defined as
8-bit-per-channel with no 16-bit 3-channel mode at all, so
`Image.fromarray(uint16_array, mode="RGB")` has no valid type-lookup entry
(crashes on newer Pillow; silently truncates to 8-bit via any raw-mode
workaround on older Pillow — see Performance review #13). Compression is
`deflate` (not LZW): LZW requires the separate `imagecodecs` package, which
isn't a project dependency. XMP (tag 700) is written via `extratags`.

Export always writes XMP to the **source** file first (best-effort). The
source-sidecar write cannot fail the export: if it can't be written (e.g. the
source is on read-only media), the requested export to `output_path` still
proceeds, and no (potentially stale) XMP is embedded in the baked TIFF output.

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
| `raw_lens_correction.py` | Lensfun profile lookup + geometric distortion correction |
| `raw_edit_pipeline.py` | Shared pipeline + export dispatch (TIFF16 / JPEG / WebP); 16-bit TIFF via `tifffile` |
| `raw_adjustments.py` | XMP I/O, defaults, `apply_adjustments_to_linear` |
| `rawviewer_ui/adjust_panel.py` | Adjust panel UI |
| `rawviewer_ui/tone_curve_widget.py` | Draggable point curve editor |
| `unified_image_processor.py` | `decode_raw_edit_base()` |
| `main.py` | Edit-base worker, preview worker, export worker, **E** shortcut |
| `scripts/phase_develop_adjust_linear.py` | Smoke tests |

---

## Environment

| Variable | Default | Effect |
|----------|---------|--------|
| `RAWVIEWER_EDIT_CHROMA_DENOISE` | `1` | Chroma denoise (bilateral) on export when panel NR off |

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
6. **Export…** — TIFF / JPEG / WebP options
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
15. **Export…** — every remaining format (TIFF16, JPEG, WebP) completes without
    error; open the 16-bit TIFF in another app and confirm it's genuinely
    16-bit (not silently 8-bit)
16. **Tone curve** — drag a point on the graph, and PV Shad/Dark/Light/High
    past ~50 in either direction — no banding/reversed local contrast anywhere
    on the curve
17. `PYTHONPATH=src python3 scripts/phase_develop_adjust_linear.py`
18. **Compare** toggle (panel header, split-view icon) — split view appears with
    original on the left, edited on the right; drag the divider anywhere across
    the image and it tracks the cursor; adjust a slider while comparing — only
    the edited (right) side updates; toggle off — split view disappears cleanly
19. **Lens correction** toggle — hidden on a file with no matching lens profile
    (check with an unusual/manual lens); shown on a file with a recognized
    camera+lens; toggling on visibly corrects barrel/pincushion distortion and
    re-decodes (brief status message); exported file reflects the same toggle
    state as the preview
20. `PYTHONPATH=src pixi run python3 scripts/phase_develop_adjust_linear.py`
    (run via pixi specifically — the lens-correction tests need `lensfunpy`)

To re-enable HSL UI: set `_SHOW_HSL_UI = True` in `adjust_panel.py` (fix the `raw_hsl.py`
colorspace-scale bug first — see Performance review #6).

---

## Known limitations / future work

- HSL section hidden in UI; `raw_hsl.py` has an open colorspace-scale bug — fix
  before re-enabling (see Performance review #6)
- No per-channel (R/G/B) tone curves
- No mask / brush / gradient / dodge-burn local edits (see Roadmap
  investigations C and D)
- Recovery baseline is preview-session UI state, not persisted in XMP
- No DNG export (removed 2026-07-04; see Export formats) — no non-destructive
  RAW+XMP round-trip or RGB-DNG output until a real DNG writer is built
- `decode_raw_edit_base`'s `executor` parameter is accepted but unused; decode
  always runs in-process (see Performance review #4)

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
| 2 | No incremental recompute — every slider tick reruns the entire WB→exposure→NLM→PV2012→tonemap→sat→HSL→detail chain even when only one late-stage control changed | All-default preview ~150–270ms; typical multi-slider edit ~1.7–1.9s; far past the 80ms throttle window | **Fixed** — see Performance review #18 (`PreviewStageCache`, live preview path only) |
| 3 | Recovery baseline (`_tone_map_recovery_display`) reruns a `scipy.ndimage.gaussian_filter(sigma=22)` + downsample/upsample round-trip on every tick, even though Exposure/Sat/Vibrance/HSL/Detail are layered "on top of" recovery tone and don't need it recomputed | ~2.0–2.3s per tick with recovery on, the slowest path measured | **Partially fixed** — see Performance review #18; the gaussian-filter pass is now skipped when only a sat/vibrance/HSL/detail slider changes with recovery held on, since it lives in the cached tonemap stage, but still reruns on any pre-tone or PV2012-tone-slider change |
| 4 | `UnifiedImageProcessor.decode_raw_edit_base(..., executor=self.process_pool)` accepts `executor` but never uses it — decode always runs in-process on the calling `QRunnable` thread, not offloaded to the process pool | N/A (code review) | Open — wire up or drop the parameter |
| 5 | `phase_develop_adjust_linear.py` only checks correctness on tiny (16×16–64×64) synthetic arrays; no benchmark at realistic preview resolution | N/A | Open — a perf regression (like #1) would not be caught by the existing suite |
| 6 | `raw_hsl.py`'s `_rgb_to_hsv` treats cv2's **float32** `COLOR_BGR2HSV` output (already H:0–360, S/V:0–1) as if it were the **uint8** convention (H:0–179, S/V:0–255) — `h * 2.0` doubles an already-0–360 hue (wraps/misassigns color bands), and `s / 255.0`, `v / 255.0` crush saturation/value to ~1/255 of true scale. `_hsv_to_rgb`'s uint8 packing then largely undoes the S/V crush on the way back out, but the per-color slider deltas (`ds`/`dv`, order ±1.0) are added *before* that undo, so they dominate the crushed (~0.004) true value — HSL Hue/Sat/Lum sliders would behave close to on/off rather than proportional, and hue-band assignment is wrong. Not visible while `_SHOW_HSL_UI = False`, but still runs for any file with pre-existing non-zero `HueAdjustment*`/`SaturationAdjustment*`/`LuminanceAdjustment*` XMP values (browse view, export) | Confirmed via direct `cv2.cvtColor` test; not benchmarked (no user-visible path while hidden) | **Open, not fixed** — out of scope for this pass (user asked to hide HSL, not fix it); fix before setting `_SHOW_HSL_UI = True` again |
| 7 | `_apply_saturation_vibrance` scaled chroma (`img - luma`) additively on **scene-linear** RGB with no upper clip. Linear-light channel spread is highly hue-dependent (green/yellow have large spread relative to luma, red/blue less so), so the same `+50` slider value clipped green/yellow to a flat, fully-saturated block (`ΔS` up to 0.54, hitting the gamut ceiling) while skin tones and blue sky barely moved (`ΔS` ~0.11–0.13) — the reported "colors look off, no pop" | Measured `ΔS` (HSV saturation, encoded output) across 6 test hues at Saturation +50: **0.11–0.54** (7× spread, several hues clipped to `S=1.0`) | **Fixed** — `_apply_saturation_vibrance` now round-trips scene-linear → sRGB-encoded (float LUT, `_encode_srgb_float01`/`_decode_srgb_float01` in `raw_tone_recovery.py`) and scales the HSV **S** channel there, bounded to `[0, 1]` by construction. Same 6 hues at +50 now measure **0.13–0.30** (~2× spread, no clipping). Cost: +~50–80ms per preview tick when Saturation/Vibrance is non-zero (HSV round-trip); no-op path unaffected |
| 8 | `apply_defringe` (`raw_detail_enhance.py`) had no edge/locality gate at all — any pixel whose hue leaned purple/green relative to its own chroma got desaturated, including uniform, unfringed regions, because `fringe / span` is scale-invariant along the purple/green hue axis rather than proportional to cast strength. A uniform purple patch (e.g. a flower petal, no edge anywhere) lost **90% saturation at Defringe=100** with nothing to correct. Separately, `green = g - 0.5*(r+b)` had half the gain of `purple = r+b-2*g` for the same absolute color imbalance — green fringing was flagged as half as severe as equally-strong purple fringing | Uniform purple patch: **90% saturation loss** at Defringe=100 with zero edges present; purple-vs-green raw mask ratio confirmed **exactly 2.00×** at every tested magnitude (d=0.02/0.05/0.10) | **Fixed** — `green` now uses `2*g - r - b` (exact negation of `purple`, so `fringe == \|r + b - 2*g\|`); added `edge_weight` gated on local luminance contrast (same Gaussian-blur trick as clarity/sharpness, sigma=1.5, soft-knee `edge/(edge+0.04)`) so the mask only engages near a real luminance transition. Uniform purple/green patches now measure **0% saturation loss** at Defringe=100; a genuine fringe pixel at a dark/bright edge still loses ~57-75%. 3 new smoke tests added (`test_defringe_*`) |
| 9 | `raw_pv2012._apply_whites_blacks` had Whites/Blacks **sign-inverted**: `white_pt = 1.0 + w*0.12` / `black_pt = b*0.12` made Blacks+100 *crush* shadows and Whites+100 *darken* highlights — backwards from every reference tool and from this app's own legacy gamma-space implementation of the same named controls, which get the direction right. Affects 100% of PV2012 Whites/Blacks usage, not an edge case | End-to-end test on a shadow/highlight ramp: `Blacks=+100` darkened 0.02→0.01 (should lighten); `Blacks=-100` lightened 0.02→0.060 (should darken); `Whites=+100` darkened 0.9→0.804 (should brighten); legacy path's own Blacks/Whites confirmed correct-direction on the identical test | **Fixed** — signs flipped: `white_pt = 1.0 - w*0.12`, `black_pt = -b*0.12`. Re-verified same ramp now moves in the expected direction both ways |
| 10 | `raw_pv2012._apply_highlights_shadows`'s shadow-lift branch (`Shadows2012 > 0`) used coefficient 0.42, but the shadow-region weight's slope reaches ~5.85 at y≈0.11 — any coefficient above ~0.17 makes the combined tone curve **locally decreasing** (a real, visible banding/contrast-reversal defect, not just theoretical). First appeared between Shadows=40 and 50. The Recovery-baseline preset hint (`Shadows=+40`) happened to sit just under the threshold; any user manually pushing Shadows past ~45 hit it. `raw_adjustments.py`'s legacy gamma-space duplicate (`_apply_highlights_shadows`/`_apply_masked_luminance_adjust`, still reachable for non-RAW/uint8 images) had the same class of bug and broke even earlier — Shadows=15 already showed violations — because its coefficient (0.55) is shared across three different region-weight shapes (shadow/black/highlight) of very different steepness (worst slopes -12.36 / -29.57 / -1.0 respectively) | PV2012: 50/1999 backward steps at Shadows=100, worst -0.0069, monotonicity break confirmed to start ~Shadows 40-50 (matches the derivative-based safe-coefficient calc `0.17/0.42×100≈40` exactly). Legacy: up to 596/3999 backward steps at Shadows=100, breaking as early as Shadows=15 (7 bad steps) | **Fixed** — PV2012 coefficient reduced 0.42→0.15 (safe margin under the computed 0.17 ceiling); re-verified monotonic at Shadows=100 and at all-sliders-maxed combos. Legacy: `_apply_masked_luminance_adjust` gained a `lift_up_strength` parameter (previously one hardcoded 0.55 shared by all three masks) so Shadow (0.07), Black (0.03), and Highlight/White (default 0.55, already safe) each get a coefficient sized to their own region-weight steepness rather than one value forced down to the strictest case. Re-verified monotonic for both Shadows and Blacks up to their extremes. 4 new smoke tests added (`test_pv2012_shadows_lift_stays_monotonic`, `test_pv2012_whites_blacks_sign_direction`, `test_legacy_gamma_highlights_shadows_stays_monotonic`, `test_legacy_gamma_blacks_stays_monotonic`) |

| 11 | `raw_chroma_denoise.apply_chroma_nlm` (superseded by `apply_chroma_denoise`, see #12) converted Cb/Cr from float to uint8 via a bare `.astype(np.uint8)`, which **truncates toward zero** rather than rounding. Cb/Cr fractional parts are effectively uniformly distributed, so truncation biases both channels ~-0.5 LSB (≈-0.00196 in [0,1]) low on average -- not randomly, systematically, every pixel, every call. The YCbCr→RGB inverse (`R = Y + 1.5748·Cr0`, `G = Y - 0.1873·Cb0 - 0.4681·Cr0`, `B = Y + 1.8556·Cb0`) turns a uniform negative bias in *both* Cb and Cr into less R, less B, and **more G** — a systematic green cast on every image that goes through Chroma NR, reported by a user as "casts a green shade" | Reproduced the bias in isolation (quantize/dequantize Cb+Cr with **zero** actual NLM denoising applied): mean RGB delta on a neutral-gray test image = `[-0.00308, +0.00128, -0.00363]`, reproducible to 4 decimal places across 15 random-noise seeds (not noise-dependent). Matches the predicted truncation bias (`-0.5/255 = -0.00196`) driven through the same inverse-transform coefficients almost exactly | **Fixed** (then superseded) — added `+ 0.5` before both `.astype(np.uint8)` casts (round-to-nearest, matching the pattern already used in the sRGB encode LUTs from finding #1). Re-verified: residual mean RGB delta dropped to ~1e-5–2e-4. Superseded by #12, which removes the uint8 round-trip entirely |
| 12 | Follow-up to a "is there a better NR than chroma NR" question: (a) chroma NR used NLM (`cv2.fastNlMeansDenoising`), one of the more expensive denoise algorithms, on a full preview/export buffer; (b) there was no luminance noise reduction at all — only Cb/Cr were ever touched, so grainy high-ISO detail noise went untreated | Benchmarked NLM vs. bilateral filter (`cv2.bilateralFilter`, applied directly on float32, no uint8 step) at realistic sizes: **19-22× faster** (1200×1800: 232ms→12ms; 2500×3750: 1105ms→50ms per channel). Bilateral also verified edge-preserving: a hard color-edge transition stays 0px wide even at max strength, while a flat noisy region's std-dev drops 71-81% depending on sigmaColor | **Implemented** — `apply_chroma_nlm` replaced with `apply_chroma_denoise` (bilateral, sigmaColor 0.03-0.15, no uint8 round-trip — also eliminates the whole bug class from #11 structurally, not just patched). Added `apply_luma_denoise` (bilateral on Y only, gentler sigmaColor 0.015-0.065 since luminance carries real detail unlike chroma) with a new **Luma NR** slider (`LuminanceNoiseReduction`, 0-100, `adjust_panel.py`/XMP) sitting above the existing Chroma NR toggle under a new "Noise" section. 6 new smoke tests added covering noise reduction, edge preservation, and pipeline wiring for both filters |

| 13 | Both 16-bit TIFF and baked-DNG-RGB export crashed outright: `Image.fromarray(out, mode="RGB")` with a uint16 3-channel array has no valid entry in Pillow's `fromarray` type-lookup table, because Pillow's `"RGB"` mode is *defined* as 8-bit-per-channel -- there is no Pillow mode for 16-bit 3-channel data at all, on any version. `pixi.toml` pinned `Pillow = ">=10.0.0"` with no upper bound, so whichever Pillow version an environment resolves determines whether this crashes (newer Pillow, confirmed on 12.2.0) or "succeeds" while silently discarding the low byte on every channel (older Pillow / any raw-mode workaround) -- reported by a user as "export failed" | Reproduced with `export_adjusted_tiff16`/`export_adjusted_dng_rgb` on synthetic data: `TypeError: Cannot handle this data type: (1, 1, 3), <u2` (that shape is Pillow's internal lookup-key shorthand, not the actual array shape -- verified the real array was a normal `(200, 300, 3)` right up to the failing call by tracing into Pillow's own `fromarray` source). Separately, JPEG export crashed too: `subsampling=0` + `optimize=True` together triggered a libjpeg "Suspension not allowed here" encoder error, reproduced with a plain synthetic uint8 array and zero RAWViewer-specific code -- either flag alone works | **Fixed, then DNG-RGB removed entirely (2026-07-04)** -- 16-bit TIFF now writes via `tifffile` (`_write_16bit_rgb_tiff`), bypassing Pillow's Image model entirely; `tifffile` added as a project dependency (`pixi.toml`/`pixi.lock`). Compression is `deflate`, not LZW (LZW needs the separate `imagecodecs` package, not a project dependency). XMP (tag 700) moves to `extratags`. Verified true 16-bit output (values >255 present, not silently 8-bit). JPEG export drops `optimize=True` (pure file-size optimization, no visible quality loss) and keeps `subsampling=0` (matters for color fidelity). Baked-DNG-RGB got the same tifffile fix and stopped crashing, but a user then reported the resulting file couldn't be opened at all -- the RGB-container-DNG approach (plain RGB TIFF + `DNGVersion`/`UniqueCameraModel` tags, no real CFA/mosaic data) was never a real DNG to begin with, fixing the crash didn't fix that. Both DNG export options ("copy RAW + XMP settings" and "baked 16-bit RGB") were removed rather than shipping an unopenable format -- see Export formats. 4 new smoke tests added for the surviving formats |

| 14 | Re-enabling the hidden tone curve UI surfaced a monotonicity bug in `apply_parametric_tone_curve` (`raw_tone_curve.py`) before it ever shipped visibly: `ParametricShadows` +100 and `ParametricHighlights` -100 both made the combined curve locally decreasing (banding), for the identical reason as the `raw_pv2012.py` shadow-lift bug fixed earlier in this doc -- the shadows/highlights region masks use the same `_smooth_weight`/`_SPLIT_SHADOWS=0.25` shape, so the same steep-slope-at-the-domain-boundary problem applies, just in a different function that happened to also use an unsafe coefficient (0.35) | Confirmed via the same derivative analysis as the earlier PV2012 fix: shadows-lift and highlights-darken brackets both reach worst slope ~5.89-5.90, requiring coefficient ≤~0.17; 0.35 produced real backward steps (worst -0.000052 at 20k-sample resolution, confirmed non-float32-noise via a 400k-sample float64 re-check showing `d(out)/dy` actually goes to -1.06 at the worst point) | **Fixed** -- coefficient reduced 0.35→0.15 (same value used for the analogous `raw_pv2012.py` fix), matching the identical safety margin. Re-verified monotonic for all 4 regions × both directions × combinations, standalone and composed with the rest of the PV2012 pipeline (Contrast/Highlights/Shadows/Whites/Blacks all maxed simultaneously). `_SHOW_TONE_CURVE_UI` then flipped to `True`. 1 new smoke test added (`test_parametric_tone_curve_stays_monotonic`) |

| 15 | User-reported "the tone curve is unresponsive" right after re-enabling it. Traced the whole chain (widget mouse events -> `get_adjustments()` -> XMP -> pipeline) and every step worked correctly in isolation -- the actual bug was a math error in `apply_pv2012_tone_rgb` (`raw_pv2012.py`): `y0` (meant to be the true pre-curve baseline) got reassigned to the curve-adjusted value (`y0 = apply_tone_curve_perceptual(y0, adj)`), and the final hue-preserving ratio then divided by that *same already-curved* value (`ratio = y1 / y0`). With no PV2012 sliders active, `y1` and the reassigned `y0` end up numerically identical, so `ratio` is exactly 1.0 regardless of what the point curve or PV parametric sliders (Shad/Dark/Light/High) do -- the curve was fully wired end-to-end but its effect was mathematically cancelled out of the image. Note this bug could only manifest once the tone curve UI was reachable (today), and doesn't affect the main Shadows2012/Highlights2012/Whites2012/Blacks2012 sliders, which don't go through `apply_tone_curve_perceptual` at all | Confirmed step by step: `apply_tone_curve_perceptual` alone responded correctly to a test curve; `apply_pv2012_tone_rgb` (the real entry point) called with the identical curve produced byte-identical output to the default curve, both with `ParametricShadows` and with the point-curve serial | **Fixed** -- kept the true baseline in `y0`, moved the curve-adjusted value to a new `y_curve` variable used only to feed `_apply_pv2012_perceptual` and the shadow-lift chroma-damp region test; the ratio's denominator now correctly stays `y0`. Re-verified: point curve and `ParametricShadows` now visibly change `apply_pv2012_tone_rgb` output; existing Shadows2012 behavior unaffected (no regression, as expected since this path never touched it); monotonicity re-confirmed with a curve active, standalone and combined with maxed sliders. 1 new smoke test added (`test_tone_curve_affects_pv2012_tone_rgb_output`) exercising the real entry point, closing the gap that let this ship silently broken |

| 16 | Follow-up user report after the tone curve fix: "shadow/black/white/highlight performance is still questionable... when lifting the black/shadow, the light part also be affected... cannot recover as much highlight/shadow [as] other editing application[s]." Root cause was much bigger than any single slider: `apply_pv2012_tone_rgb`'s `ratio = y1 / np.maximum(y0, _PERCEPTUAL_LUM_FLOOR)` floored *only the denominator*. With every slider at default, `y1 == y0` exactly, so this should be a pure identity operation -- but for any pixel with true perceptual luminance below the 0.03 floor (real deep-shadow/black detail), the floored denominator no longer equals the numerator, silently darkening that pixel up to 50% with **zero user adjustment**, before any slider was ever touched. The same floor then fought against genuine Shadows2012/Blacks2012 recovery attempts in that exact tonal range (the floor doesn't move when a slider tries to lift the pixel out of it, capping how much recovery could show through), and made Highlights2012 incorrectly touch the deepest shadows too (any ratio computed through this formula for a sub-floor pixel was wrong, regardless of which slider produced `y1`) | Confirmed the baseline-identity violation directly: with all 5 PV2012 sliders at default, `scene-linear=0.005` (well within real shadow detail) produced ratio=0.5 (max darkening) and `scene-linear=0.01` produced ratio=0.88 -- not 1.0, despite zero user adjustment. `Highlights2012=-100` also produced ratio=0.5 at `scene-linear=0.005` before the fix, confirming the region leak; after the fix, ratio=1.0 there (correctly untouched) | **Fixed** -- floored both numerator and denominator (`np.maximum(y1, floor) / np.maximum(y0, floor)`), so the floor still prevents a near-zero-denominator blowup but no longer breaks the identity case. Re-verified: baseline is now exact identity (ratio=1.0) at every tested luminance from 0.0001 to 4.0; `Highlights2012=-100` no longer touches deep shadows; `Shadows2012`/`Blacks2012` recovery is now visibly stronger and reaches the full `_MIN_TONE_RATIO`/`_MAX_TONE_RATIO` range in deep shadows instead of being capped short by the floor mismatch; monotonicity re-confirmed for all sliders and combinations. 2 new smoke tests added (`test_pv2012_default_adjustments_are_true_identity`, `test_pv2012_highlights_slider_does_not_touch_deep_shadows`) |

| 17 | Follow-up to the #16 fix: with the baseline-identity bug gone, the user asked whether Whites/Blacks range could be strengthened further -- `_apply_whites_blacks`'s `white_pt = 1.0 - w*0.12` / `black_pt = -b*0.12` meant even `Whites=+100`/`Blacks=-100` (the sliders' hard extremes) only moved the white/black point by 12%, nowhere near enough to visibly clip highlights or crush shadows the way the equivalent controls do in other editing tools | Tested candidate coefficients (0.12/0.2/0.25/0.3/0.35) via the real `apply_pv2012_tone_rgb` entry point; 0.25 gives a worst-case combined span (`Whites=+100` and `Blacks=-100` together) of 0.5 -- still comfortably clear of a degenerate near-zero span -- while producing a clearly stronger, visibly-clipping effect at the extremes | **Fixed** -- coefficient increased 0.12→0.25 in both `white_pt`/`black_pt`. Re-verified: still a pure affine remap + clip, so monotonicity holds unconditionally (confirmed via full derivative sweep at `Whites=+100`, `Blacks=-100`, and the combined worst case, all `bad_steps=0`); through the real pipeline entry point, `Whites=+100` now boosts a bright pixel's ratio to ~1.33× (was ~1.03×) and `Blacks=-100` drops a dark pixel's ratio to the `_MIN_TONE_RATIO` floor (0.5×, was ~0.85×). 1 new smoke test added (`test_pv2012_whites_blacks_strength`) asserting the strengthened magnitude, guarding against silent regression back toward the old weak coefficient |

| 18 | Follow-up to the user's "sluggish/laggy while dragging" report (see #16's investigation): finding #2 above, still open at that point -- every preview tick reran the entire WB -> exposure -> denoise -> PV2012 tone -> tonemap -> sat/vibrance -> HSL -> detail chain from scratch, even when dragging a single late-stage slider (e.g. Sharpness) with everything upstream unchanged | Benchmarked at the documented 2000x3000 preview size: dragging **Sharpness** (last stage) with Exposure/Shadows/Saturation already set, uncached **933ms/tick** vs staged **170ms/tick** (**5.5x**). Dragging an upstream slider (Shadows2012 or Exposure2012, which invalidate everything downstream anyway) showed **0.99-1.00x** -- i.e. no caching overhead penalty when caching can't help | **Implemented** -- `PreviewStageCache` (`raw_edit_pipeline.py`) memoizes 4 checkpoints: pre-tone (WB/Tint/Exposure/denoise), PV2012 tone, tonemap-for-display, and color (sat/vibrance/HSL), with detail (sharpen/clarity/defringe) computed last from the cached color output. Each stage's cache key is `(upstream_stage_key, this_stage's_own_adjustment_keys)` -- chaining to the upstream key is what makes a change to, say, Exposure2012 correctly invalidate every stage after it even though Exposure2012 isn't one of that stage's own keys. Only wired into the live preview path (`_AdjustPreviewWorker` in `main.py`, via `render_adjust_preview_uint8`) -- export always calls the plain uncached functions directly, so a cache bug here could never corrupt a baked export. Cache lives on the main window (`self._adjust_preview_stage_cache`) and self-invalidates on file navigation via base-image identity (`cache.base_ref is not rgb_image`), needing no explicit reset wiring at each of the existing `_adjust_preview_base_rgb = None` call sites. Guarded by a `threading.Lock` since the cache is mutated from the `QThreadPool` worker thread. Two real bugs were caught by property tests before shipping: (1) the tone stage's key initially included only its own PV2012/curve keys, not the upstream pre-tone key, so changing Exposure2012 alone (no PV2012 slider touched) incorrectly reused a tone-stage output computed from the *old* pre-tone buffer; (2) the "adjustments are all default" fast path (which bypasses pre-tone/tone entirely and returns the raw linear buffer) never stamped a tone-stage cache key, so transitioning into or out of all-default state left a stale key in place and the downstream tonemap stage wrongly reused output computed from different pixel data. Both fixed and re-verified. 2 new smoke tests added: `test_preview_stage_cache_matches_full_recompute` (staged output byte-identical to the uncached reference across a 20-step sequence covering single-key drags, multi-key combinations, recovery-baseline toggling, HSL, a fully-loaded combination, repeated returns to default, and a mid-sequence base-image swap) and `test_preview_stage_cache_skips_upstream_recompute` (instruments `apply_pv2012_tone_rgb` with a call counter to directly confirm the tone stage is skipped when only Sharpness changes, and does recompute when Shadows2012 changes) |

| 19 | User asked "should [the point tone curve] be a curve?" — it wasn't: both the widget's rendering and `build_point_curve_lut`'s actual pixel math connected knots with straight line segments (`np.interp`), so the point curve looked and behaved like a polyline rather than the smooth spline every reference tool (Lightroom, Capture One) shows for the same control | N/A (visual/design gap, not a numeric bug) | **Fixed** — switched to `scipy.interpolate.PchipInterpolator` (monotonic piecewise cubic Hermite), not a plain/natural cubic spline: PCHIP is shape-preserving and provably cannot overshoot past the local trend of the input knots, so it can't introduce the ringing/banding artifact a natural spline risks on a steep-then-flat knot layout — the same overshoot failure mode this doc's earlier tone-curve/PV2012 coefficient fixes (#10, #14) were about, just from a different cause (spline math instead of an unsafe linear coefficient). `raw_tone_curve.py` gained `_fit_point_curve` (shared PCHIP fit, deduplicating near-identical x knots since PCHIP requires strictly increasing x) used by both `build_point_curve_lut` (the real 65536-entry LUT) and a new `sample_point_curve_for_display` (a cheap ~96-point sample of the *same* fit, used only for drawing) — so the widget's rendered curve is guaranteed to match what's actually applied to the image, not a separate approximation of it. `tone_curve_widget.py`'s `paintEvent` now draws the sampled spline instead of straight segments between control points. Verified via a headless-Qt screenshot (curve now visibly smooth through the control points) and 2 new smoke tests: `test_point_curve_pchip_no_overshoot` (a steep-rise-then-plateau knot layout stays within `[min(y_i, y_{i+1}), max(y_i, y_{i+1})]` on every segment, catching the exact overshoot a natural spline would produce there) and `test_point_curve_display_matches_applied_lut` (widget-drawn samples match the real LUT to within 0.5/255 at the same x) |

| 20 | User report: "the color noise cannot be removed" — the bilateral filter added in #12 (5-7px kernel diameter) only ever sees noise narrower than itself. Real high-ISO sensor color noise is typically **blotchy**: spatially correlated over several pixels from Bayer demosaic interpolation and sensor readout patterns, not pixel-independent. Confirmed by synthesizing noise with increasing spatial correlation length and measuring the existing bilateral filter's std-dev reduction on each | Pixel-independent noise: 77.4% reduction (bilateral doing its job). Noise blurred with a Gaussian sigma=4 (a representative "blotch" correlation length): only **6.6%** reduction. Sigma=8 (larger blotches): only **1.7%** — the kernel essentially can't see it at all | **Implemented** — added a second, coarser pass in `apply_chroma_denoise`: downsample Cb/Cr (factor 3, 2 in preview), Gaussian-blur at the reduced resolution, upsample back (same principle as JPEG/video 4:2:0 chroma subsampling — human vision resolves color at far lower spatial resolution than luminance, so a softer chroma channel isn't perceptible the way a softer luma channel would be). Blended back with the bilateral (fine) result in proportion to a luma-Sobel-gradient edge mask (`_luma_edge_weight`), so real color edges stay protected while flat/blotchy regions get the much stronger coarse pass. Re-measured with the actual tuned parameters at the strongest reachable slider setting (`strength=1.25`, `ColorNoiseReduction=100`): sigma=4 blotchy noise reduction improved 6.6%→10.9%, sigma=8 improved 1.7%→3.7%, sigma=1.5 (milder, more common) improved 29.5%→37.1%, pixel-independent improved 77.4%→82.9%. Tradeoff: the coarse pass isn't edge-aware on its own, so there is now measurable bleed very close to a hard color edge, bounded by construction (tuned against a synthetic red\|blue edge test) to **~0.04 max channel error at 4px, ~0.004 at 6px, ~0 at 8px+** even at max strength — no user-visible halo beyond a few pixels. Added cost: ~3.7x over the bilateral-only pass at the documented 2000×3000 preview size (63ms → 233ms), but this only matters for slider drags that actually touch `ColorNoiseReduction`/`Temperature`/`Tint`/`Exposure2012` — the `PreviewStageCache` from #18 already skips this stage entirely for any other slider drag (Sharpness, Saturation, PV2012 tone, etc.), and it's still far cheaper than the NLM approach it replaced in #12 (~1100ms+ at a similar size). 2 new smoke tests added: `test_chroma_denoise_removes_blotchy_correlated_noise` (reduction on sigma=4 correlated noise must exceed 8%, guarding against this regressing back toward the ~7% pre-fix baseline) and `test_chroma_denoise_edge_bleed_bounded_near_hard_edge` (bleed at 4/6/8px from a hard edge stays under 0.08/0.03/0.01 respectively at max strength) |

Numbers were captured with `raw_edit_pipeline.process_linear_edit_buffer` +
`linear_to_display_uint8` directly (no Qt/GUI thread involved); see the
conversation history or re-run the benchmark snippet against
`scripts/phase_develop_adjust_linear.py`'s helpers for a repeatable check.

---

## Roadmap investigations (2026-07-04)

Three feasibility investigations, requested as research only (no code changes
in this pass). Each follows Finding → Feasibility → Suggestion.

### A. Lightroom-compatible rating / label system

**Finding.** Ratings exist today (0-5 stars, number keys `0`-`5` in single
view via `rate_current_image()`, `main.py:22779`), but they are
**not written to the file or an XMP sidecar at all**. A rating lives only
inside the `exif_data` blob of the app's local SQLite cache
(`image_cache.py`'s `exif_cache` table — `file_path, file_size, file_mtime,
orientation, camera_make, camera_model, exif_data (BLOB), ...` — no dedicated
`rating` column; it's a key inside the pickled blob, `main.py:22804-22807`
sets `exif_data['rating'] = rating`, `main.py:17069-17073` reads it back).
Rebuilding or deleting that cache silently loses every rating, and ratings
never travel with the file (no round-trip to Lightroom, no round-trip even
to a copy of the file on another machine). There is **no pick/reject flag**
and **no color-label system** anywhere in the codebase — the closest hits are
an unrelated "bookmark for sharing" star icon in the toolbar
(`main.py:7903-7909`) and a "move to Discard subfolder" feature
(`main.py:4839`), neither of which is a metadata flag.

`raw_adjustments.py` already has a working XMP sidecar read/write path
(`write_xmp_adjustments`, `parse_xmp_adjustments`, `resolve_xmp_path`) using
`xml.etree.ElementTree`, but it only ever touches the Camera-Raw-Settings
namespace:
```python
CRS_NS = "http://adobe.com/camera-raw-settings/1.0/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
X_NS = "adobe:ns:meta/"
```
Lightroom's own rating/label/keyword fields live in namespaces this app
never touches: `xmp:Rating` (`http://ns.adobe.com/xap/1.0/`), `xmp:Label`,
and `dc:subject` (`http://purl.org/dc/elements/1.1/`) for keywords. Because
the develop-settings parser already reads/writes the same `crs:` namespace
Lightroom itself uses, **this app is already develop-setting-compatible with
Lightroom XMP** (a sidecar written by either app is readable by the other for
tone/color adjustments) — the gap is specifically rating/label/keywords,
which live in namespaces never touched.

One sharp edge: `write_xmp_adjustments` **deletes the whole sidecar file**
when adjustments equal defaults (so an unedited image has no leftover XMP
clutter). If rating/label moved into the same sidecar file, that delete-on-
default logic would need to change first — a photo with 5 stars and no
develop edits would otherwise have its rating silently deleted the moment
adjustments reset to default.

**Feasibility.** Moderate, well-scoped. Three independent pieces:
1. Add `xmp:Rating` (int, -1 to 5, where -1 = rejected per Lightroom
   convention) and `xmp:Label` (string) read/write to the existing
   `ET`-based sidecar code, namespaced separately from `crs:`.
2. Stop treating "adjustments are default" as "delete the sidecar" — instead
   delete only the `crs:` elements/attributes when they reset to default,
   and keep the file (or the `xmp:`/`dc:` elements within it) if a
   rating/label/keyword is still present.
3. Migrate existing SQLite-cached ratings into XMP sidecars once (a one-time
   background pass, non-destructive — cache stays as a fast read-through
   layer, XMP becomes the source of truth), plus point the gallery/filmstrip
   at a star-rating UI (there's currently no rating UI outside single view
   at all — gallery/filmstrip would need one built).

Risk is concentrated in item 2 (sidecar lifecycle change touches a path
several other features now depend on — export's XMP embedding, the gallery
"edited" badge's existence-check) more than in the XMP schema itself, which
is a small, well-documented addition.

**Suggestion.** Do the sidecar-lifecycle change first, in isolation, with a
regression test asserting current develop-settings behavior is unaffected
(reset-to-default still cleans up `crs:` content; a sidecar with only a
rating survives). Then add `xmp:Rating`/`xmp:Label` read/write. Defer the
SQLite-to-XMP migration and any new gallery/filmstrip rating UI to a
follow-up — they're additive and don't block XMP compatibility itself.

---

### B. Aspect / perspective correction, rotation, straighten

**Finding.** The existing rotate button (`main.py:7924-7930`,
`_rotate_current_image_clockwise_persist`) is **90°-step-only** and purely
cosmetic: it applies a `QTransform.rotate(degrees)` to the displayed
`QPixmap` at render time (`_apply_visual_rotation_for_current`,
`main.py:20479-20490`) and persists the chosen angle per-file in `QSettings`
— it never touches RAW pixel data, never writes XMP, and isn't a continuous
angle. A separate, unrelated code path
(`metadata_backend.rotate_exif_orientation_meta_cw90`) rewrites the EXIF
orientation tag for on-disk rotation, again in 90° steps only. There is
**no crop tool, no straighten tool, and no perspective/lens-correction code
anywhere** in the codebase — a full-repo search for
`perspective|straighten|warp|homography|lens correction` returns zero hits.
`DEFAULT_ADJUSTMENTS`/`SLIDER_SPECS` (`raw_adjustments.py`) have no geometry
keys at all, and `write_xmp_adjustments` writes none of Lightroom's
`crs:CropTop/Left/Bottom/Right`, `crs:StraightenAngle`, or
`crs:PerspectiveVertical/Horizontal` fields. This is confirmed as an
intentional gap, not an oversight — the "Not supported" line earlier in this
doc already lists it.

The building blocks for adding this are already present, though: `cv2` is a
real dependency (`raw_chroma_denoise.py`, `raw_detail_enhance.py`,
`raw_hsl.py` all already import it), so `cv2.warpAffine`/
`cv2.warpPerspective`/`cv2.getPerspectiveTransform` are available with zero
new dependencies. `scipy.ndimage` (used for `zoom`/`gaussian_filter`
elsewhere) is present too, though `cv2`'s warp functions are the more direct
fit. The display path, `gpu_image_view.py`'s `GpuImageView`, is a
`QGraphicsView`/`QGraphicsPixmapItem` with an optional `QOpenGLWidget`
*viewport* for GPU-accelerated compositing of an already-CPU-computed
`QPixmap` — **not** a shader/compute pipeline. That distinction matters for
scope: a **straighten** control (continuous-angle rotation, which is affine)
could get a cheap live-preview path via a `QTransform` applied to the scene
item directly, no pixel resampling needed until the user commits. **True
perspective correction is a projective transform**, which `QTransform`-on-a-
pixmap-item cannot represent correctly for anything beyond a coarse preview
— it needs real pixel resampling (`cv2.warpPerspective`) baked into
`raw_edit_pipeline.py`'s scene-linear buffer, both for preview and for
full-resolution export.

**Feasibility.** Substantial — this is a new subsystem, not an extension of
existing sliders, roughly comparable in size to adding the tone-curve UI was.
Concretely it needs: (1) new adjustment keys (`CropTop/Left/Bottom/Right`,
`StraightenAngle`, `PerspectiveVertical/Horizontal`) in
`DEFAULT_ADJUSTMENTS`; (2) a geometry stage in `raw_edit_pipeline.py` that
runs *before* the tone/color stages (crop/rotate/warp changes the buffer's
shape and pixel positions, so everything downstream — denoise, tone,
detail — must operate on the already-geometry-corrected buffer, not the
other way around); (3) `PreviewStageCache` would need a new, earliest
checkpoint for this stage, since a geometry change invalidates literally
everything downstream (the existing 4-stage chain already handles "upstream
change invalidates downstream" — this just adds one more link at the front);
(4) UI: a crop-rectangle overlay with draggable handles and a straighten
slider/dial on the image canvas itself (`gpu_image_view.py`), which is a
different interaction model than any control this app has today (closest
precedent is the WB dropper's click-to-sample mode, `set_color_pick_mode`);
(5) XMP read/write for the new fields; (6) export-path integration so a
baked TIFF/JPEG/WebP actually reflects the crop/straighten/perspective, not
just the preview.

**Suggestion.** Split into two independent phases rather than one project:
- **Phase 1 — straighten + basic rotate/crop.** Affine-only (no
  perspective), which is both the most commonly used of these tools in
  practice and the cheaper build (no projective math, `QTransform` covers
  live preview). This alone would address "rotation" and "straighten" from
  the request.
- **Phase 2 — perspective/aspect correction.** Projective transform, real
  pixel resampling required even for preview, and meaningfully more UI
  complexity (4 independent corner/edge handles or vertical+horizontal
  sliders with a live keystone preview). Worth gating on whether Phase 1
  proves out the geometry-stage architecture (the "runs before
  everything else + gets its own cache checkpoint" design) cleanly first.

---

### C. Mask / brush / gradient local edits

**Finding.** Confirmed: the entire pipeline is global-only today. Every
stage in `raw_edit_pipeline.py` (`process_linear_edit_buffer`,
`_apply_display_stage`) takes the full frame and one flat
`dict[str, float]` of scalar adjustments — there is no mask, region, ROI, or
per-pixel weight concept anywhere in `raw_edit_pipeline.py`,
`raw_adjustments.py`, `raw_pv2012.py`, `raw_hsl.py`,
`raw_detail_enhance.py`, or `raw_chroma_denoise.py`. This doc's own "Not
supported" line already says so. `gpu_image_view.py` is, as in section B,
GPU-accelerated *display* of a CPU-computed bitmap (`QGraphicsView` +
optional `QOpenGLWidget` viewport), not a shader pipeline — there's no cheap
GPU path to evaluate a mask; it would be rasterized and composited on the
CPU like everything else already is.

Two existing precedents are useful building blocks even though neither is a
mask tool: `tone_curve_widget.py`'s `ToneCurveWidget` has a full drag
lifecycle (press → hit-test → drag with live repaint → release commits,
`editing_finished` signal) that's a reasonable template for "drag a handle,
see it live, commit on release" — just on an abstract curve-graph, not image
pixels. `gpu_image_view.py` already does real image-canvas mouse handling
and scene-to-image-pixel coordinate mapping for the WB dropper's
click-to-sample mode (`set_color_pick_mode`/`colorPickRequested`) — the
coordinate-mapping plumbing a brush/gradient tool needs already exists, but
continuous drag-paint stroke accumulation (brush) or two-point gradient
handles (linear/radial gradient) would be new interaction code built on top
of it.

XMP serialization is the deepest structural gap. `write_xmp_adjustments`
writes almost everything as **flat attributes on a single
`rdf:Description`** — one `dict[str, float]` per image, full stop. The one
exception (`ToneCurvePV2012`, a nested `rdf:Seq` of points) proves nested
XML is usable in principle, but there's exactly one such element, not a
repeatable collection. Lightroom's real local-adjustment XMP
(`crs:MaskGroupBasedCorrections`) is an **indexed array of correction
groups**, each with its own nested mask geometry (brush stroke points +
radii, or gradient endpoints) *and* its own scalar adjustment sub-dict.
Supporting this means the in-memory adjustment model — currently one flat
dict, threaded through `process_linear_edit_buffer`, `PreviewStageCache`,
`adjust_panel.py`, and the XMP read/write functions — would need to become
something like `{"global": {...flat dict...}, "local": [{"mask": ...,
"adjustments": {...}}, ...]}`, which touches essentially every call site
that currently assumes "the adjustments are a flat dict."

The recently-added `PreviewStageCache` (Performance review #18) is not
naturally extensible to per-mask recompute — its whole design is "memoize
whole-frame stage outputs keyed by scalar adjustment tuples," and mask
geometry/values have no representation as a scalar key. The pipeline shape
does suggest a workable approach, though: keep the existing global cache
exactly as-is for the base/global look, and treat each local-adjustment
layer as an **additional final compositing stage** — compute a layer's
effect on a (possibly bbox-cropped-to-the-mask) copy of the cached global
output, blend by the mask's alpha, and stack layers in order. This adds cost
roughly proportional to affected pixels/active masks rather than forcing a
full-frame recompute per mask edit, but it's new architecture layered on top
of the existing cache, not a natural extension of it.

**Feasibility.** Large. Rough size comparison: the core pipeline files
(`raw_edit_pipeline.py`, `raw_adjustments.py`, `adjust_panel.py`,
`gpu_image_view.py`, `tone_curve_widget.py`) total ~3,400 lines today. Full
local-adjustment support — mask rasterization engine, brush stroke math,
linear/radial gradient math, a new XMP schema and parser for indexed
correction groups, a per-layer management UI (add/select/delete/reorder
layers, each with its own scalar sliders), and compositing integration into
both the live preview and full-resolution export paths — is plausibly
**1,500-3,000+ new lines**, i.e. on the order of doubling this subsystem
rather than an incremental addition on top of it.

**Suggestion.** This is the largest of the three asks by a wide margin and
has the least existing infrastructure to build on. If pursued, sequence it
as: (1) radial + linear gradient tools first (two-point/two-circle
parametric masks — no rasterized stroke data, no incremental brush storage,
much simpler math and a much smaller XMP footprint: just endpoint
coordinates + feather, not a point cloud) with the "composite as a final
stage on top of the cached global result" architecture; (2) only build the
adjustment brush (arbitrary rasterized mask, incremental stroke
accumulation, by far the most implementation work of the three tools) once
the gradient tools have validated the compositing/XMP-schema approach
end-to-end. Given the size, this likely warrants being scoped and planned as
its own multi-session project rather than a single follow-up task.

---

### D. Dodge & burn

**Finding.** No existing infrastructure — same zero-mask starting point as
section C (there is no code to build on beyond what C already found).
Classic dodge/burn is functionally a **narrower** special case of a full
adjustment brush, not a separate feature: a single luminance-only intensity
delta (dodge = brighten, burn = darken), applied through an accumulated
grayscale stroke mask, with no color/HSL/detail sliders per stroke the way a
general adjustment brush would need. That narrower scope helps on the
*adjustment* side (one scalar instead of an arbitrary slider set to store
per mask) but does **not** avoid the hard part identified in section C: it
still needs real stroke rasterization and incremental accumulation (repeated
brush passes building up strength, brush hardness/falloff, feathering) —
the same "arbitrary rasterized mask" work section C flagged as the most
expensive of the three tools there, not the gradient tools' cheap parametric
math.

Two things in this codebase are a genuinely good fit for the *adjustment*
side specifically, though: (1) luminance-only math already exists and is
proven — `_luminance()` (`raw_pv2012.py`, `raw_chroma_denoise.py`'s
`_rgb_to_ycbcr`) is exactly the "touch only lightness, preserve hue/chroma"
operation dodge/burn needs, no new color science required; (2) real
dodge/burn tools (Photoshop, Lightroom's Adjustment Brush "Exposure") almost
always protect/target a luminosity range (shadows, midtones, or highlights)
rather than applying a flat delta everywhere a stroke lands, to avoid
blowing out highlights or crushing shadows that are already extreme — and
this exact "smooth region-weight targeting a tonal range" pattern already
exists, twice, in this codebase: `_region_weight_shadows`/
`_region_weight_highlights` in `raw_pv2012.py` and the equivalent
`_region_masks` in `raw_tone_curve.py`, both using the same
`_smooth_weight`/`_SPLIT_SHADOWS` smoothstep shape this doc's monotonicity
fixes (#10, #14) were tuned against. A dodge/burn "range" selector (shadows/
midtones/highlights, matching Photoshop's own tool) could reuse this math
directly instead of inventing new region logic.

The other real design question dodge/burn shares with the general brush: a
rasterized per-pixel stroke mask is a full-resolution single-channel array,
and section C's XMP-schema discussion (moving from a flat scalar dict to
Lightroom's indexed `MaskGroupBasedCorrections`) didn't fully address *how*
such a mask gets serialized. Storing a full-resolution raster as text inside
an XMP sidecar would bloat file sizes considerably; Lightroom's own solution
compresses/encodes mask data compactly within the correction group. This
codebase already has a directly-applicable precedent for "shrink an
auxiliary channel that doesn't need full resolution" — the chroma-denoise
downsample/blur/upsample pass added in Performance review #20 — the same
idea (store the stroke mask at reduced resolution, upsample on the fly) is
a reasonable starting point rather than storing it at full pixel resolution.

**Feasibility.** Same order of magnitude as section C's adjustment brush
(large), because it needs the identical stroke-rasterization/accumulation
architecture — narrowing the adjustment to one luminance scalar plus a
range selector reduces the UI/slider surface area but not the masking
engine work, which is the dominant cost.

**Suggestion.** If a brush tool gets built at all, dodge/burn is the
smallest *useful* version of it — grayscale-only mask, one adjustment
value, one range selector, directly reusing existing luminance and
region-weight code — so it's a reasonable first concrete target for
validating the stroke/mask engine, rather than building a full multi-slider
adjustment brush from the start. This refines, rather than replaces, section
C's suggested order: gradient tools first (cheapest, purely parametric, no
rasterized mask at all), then dodge/burn as the first brush-based tool
(validates stroke rasterization + a compact storage format with the
smallest possible adjustment surface), then the general adjustment brush
once both the mask engine and its storage format are proven.

---

### E. Trackpad pressure controlling brush intensity

**Finding.** The key enabling fact is already true of this codebase: PyObjC
is a real, actively-used dependency, not just an AppleScript shim.
`pixi.toml`/`pixi.lock` pin `pyobjc-framework-Cocoa` and
`pyobjc-framework-Quartz` (both already resolved in the environment), and
`main.py`'s native file-dialog code (`_create_macos_open_panel`,
`_create_macos_save_panel`) does genuine Objective-C bridging today —
`import objc; objc.lookUpClass("NSOpenPanel")`, `from AppKit import NSURL`,
calling `panel.runModal()` directly — with AppleScript (`osascript` via
`subprocess.run`) only as a fallback when the PyObjC path raises. So the
low-level capability needed here (talking to Cocoa classes directly, not
just shelling out to a script) is proven and already shipping.

Qt itself is not the path in, though. PyQt6's installed bindings do have a
pressure API (`QTabletEvent.pressure()`, `QEventPoint.pressure()` — both
confirmed present in the installed `QtGui.pyi` stubs), but it's designed for
stylus/tablet devices reported by the platform plugin (a Wacom tablet, for
example) — a macOS Force Touch trackpad's "deep press" does not arrive
through `QTabletEvent` at all. Trackpad pressure is a Cocoa-level concept
(`NSEvent.pressure`, `NSPressureConfiguration`, and the
`pressureChangeWithEvent:` `NSResponder` method that fires continuously
during a physical press) that Qt's cross-platform event system doesn't
surface. Confirmed no existing code in this app reads pressure anywhere
today (`gpu_image_view.py`'s and `tone_curve_widget.py`'s mouse handlers use
only `button()`/`buttons()`/`position()`/`modifiers()`).

An important hardware/API limitation, independent of implementation effort:
continuous pressure is **only** available on Force Touch trackpads (all
modern MacBook trackpads, Magic Trackpad 2) with the responder opted into
continuous reporting — a plain click, an external mouse, a Magic Mouse, an
older Magic Trackpad 1, and every non-macOS platform report no continuous
pressure signal at all, only a binary click. Any pressure-driven intensity
control therefore cannot be the *only* way to set brush intensity; it has
to be an optional modulation on top of a baseline control (a slider, a
fixed per-stroke opacity, or drag-speed) that works everywhere else.

**Feasibility.** Good for the native-bridge part specifically (this
codebase already proves PyObjC/Cocoa integration works here, for a
different but analogous purpose), but this is a small, platform-specific
addition on top of whatever brush/dodge-burn tool exists — not something to
build before there's a brush to modulate. Concretely it needs: a custom
`NSView`/`NSResponder` override (likely reachable by wrapping the native
view underneath `GpuImageView`'s window via PyObjC, similar in spirit to how
the save/open panel code already reaches into AppKit) overriding
`pressureChangeWithEvent:`, forwarding the continuous pressure float into
Qt (macOS integrates the Cocoa run loop with Qt's event loop on the main
thread, so this shouldn't need complex cross-thread marshaling, but that
needs to be verified in practice, not assumed), and a fallback path (no-op,
or a fixed default intensity) for every device/platform that can't supply
pressure.

**Suggestion.** Treat this strictly as a follow-on refinement, scoped and
built only after a baseline (non-pressure-sensitive) brush/dodge-burn tool
exists and ships — bundling "build the local-adjustment masking engine" and
"also get low-level Cocoa pressure events flowing into Qt" into one project
multiplies risk (two largely-unrelated hard problems) for a payoff (pressure
modulation) that's meaningless without the brush already working. When it
is picked up, prototype the Cocoa bridge in isolation first (e.g. log
pressure values to the console while dragging on a Force Touch trackpad)
before wiring it into any real intensity control, to validate the
main-thread event-forwarding assumption above cheaply.

---

### F. Lens profile automatic aspect/distortion correction — Implemented (2026-07-05)

Originally scoped out here as low-feasibility because no lens-calibration
database existed in the project (`lensfun`/`lensfunpy` unvendored). Re-scoped
and implemented after further research corrected two things: (1) `lensfunpy`
ships **prebuilt wheels bundling the lensfun C library and its full lens
database** for Linux/macOS/Windows (948 cameras / 1304 lenses out of the box,
confirmed by loading `lensfunpy.Database()` directly) — not a from-source
build as first assumed; (2) the lens database itself is CC BY-SA 3.0 (the C
library is LGPLv3, fine to link from closed-source code) — a real license
term to be aware of when bundling, but not a blocker. See the "Lens profile
correction" section below for the implementation. Geometric distortion only
(the "aspect correction" originally asked about) — vignetting/TCA correction
(lensfunpy supports both) are not wired up, a possible future extension.

---
