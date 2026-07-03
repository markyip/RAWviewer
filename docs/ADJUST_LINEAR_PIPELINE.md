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
| Point tone curve UI | Done | Draggable editor + LR `ToneCurvePV2012` import/export |
| Parametric tone regions | Done | PV Shad / Dark / Light / High |
| HSL 分色 (8 colors) | Done | Hue / Sat / Lum per color |
| Detail (Sharpness / Clarity / Defringe) | Done | Display-linear, after tone map |
| Chroma NLM | Done | Toggle in panel; export + optional env |
| Recovery baseline | Done | “Use recovery look” = P-key tone as adjust start |
| Multi-format export | Done | TIFF / JPEG / WebP / DNG (settings or baked) |
| XMP sidecar I/O | Done | Basic + tone curve + HSL; as-shot Temp |
| Background preview worker | Done | 80 ms throttle; generation merge |
| Keyboard / focus fixes | Done | Shortcuts work with panel open |

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
  → chroma NLM (Cb/Cr only; preview skips unless panel NR on)
  → PV2012 tone (unless recovery baseline — see below)
      • Built-in medium-contrast base curve
      • Point curve LUT (interactive editor + crs:ToneCurvePV2012)
      • Parametric Shadows / Darks / Lights / Highlights
      • Contrast / Highlights / Shadows / Whites / Blacks
  → tone map to display
      • Default: luminance-preserving Reinhard
      • Recovery baseline: linear_tone_map_to_display + local shadow/highlight polish
  → saturation / vibrance (luma-preserving)
  → HSL 分色 (8 colors)
  → sharpness / clarity / defringe
  → sRGB OETF → uint8 (preview) or uint16 (TIFF / baked DNG export)
```

Browse / gallery / non-edit paths **unchanged** (embedded JPEG + fast 8-bit LibRaw).

---

## Adjustments implemented

### Basic — Light

| Control | XMP key | Range | Pipeline stage |
|---------|---------|-------|----------------|
| Exposure | `Exposure2012` | −5 … +5 EV | Scene-linear multiply |
| Contrast | `Contrast2012` | −100 … +100 | PV2012 perceptual |
| Highlights | `Highlights2012` | −100 … +100 | PV2012 perceptual |
| Shadows | `Shadows2012` | −100 … +100 | PV2012 perceptual |
| Whites | `Whites2012` | −100 … +100 | PV2012 perceptual |
| Blacks | `Blacks2012` | −100 … +100 | PV2012 perceptual |

### Tone curve

| Control | XMP / internal | Notes |
|---------|----------------|-------|
| Point curve | `crs:ToneCurvePV2012` / `_tone_curve_pv2012` | Draggable UI; drag / click add / double-click remove; **Linear** reset |
| PV Shadows | `ParametricShadows` | −100 … +100, perceptual regions |
| PV Darks | `ParametricDarks` | |
| PV Lights | `ParametricLights` | |
| PV Highlights | `ParametricHighlights` | |

Point curve is applied **before** parametric regions and PV2012 HS/W/B.

### Basic — Color

| Control | XMP key | Range | Pipeline stage |
|---------|---------|-------|----------------|
| Temperature | `Temperature` | As-shot ± slider | Scene-linear WB vs as-shot Kelvin |
| Tint | `Tint` | −150 … +150 | Green–magenta on G channel |
| Saturation | `Saturation` | −100 … +100 | Display-linear, luma-preserving |
| Vibrance | `Vibrance` | −100 … +100 | Display-linear, low-sat weighted |

**Temperature default:** On first open (no XMP), Temp = **as-shot** from EXIF / RAW metadata (`AsShotTemperature` internal key). LibRaw uses `use_camera_wb=True`, so slider at as-shot = neutral.

### HSL 分色

Eight colors: **Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta**.

| Per color | XMP keys | Range |
|-----------|----------|-------|
| Hue | `HueAdjustment{Color}` | −100 … +100 |
| Saturation | `SaturationAdjustment{Color}` | −100 … +100 |
| Luminance | `LuminanceAdjustment{Color}` | −100 … +100 |

UI: color dropdown + three sliders; per-color values cached when switching colors.

### Detail

| Control | XMP key | Range | Pipeline stage |
|---------|---------|-------|----------------|
| Sharpness | `Sharpness` | 0 … 150 | Display-linear unsharp |
| Clarity | `Clarity2012` | −100 … +100 | Local contrast (display-linear) |
| Defringe | `Defringe` | 0 … 100 | Purple/green fringe reduction |

### Noise

| Control | XMP key | Notes |
|---------|---------|-------|
| Chroma NR (toggle) | `ColorNoiseReduction` | 0 or 50 when On; NLM on Cb/Cr only |

Preview: NR only when toggle On. Export: NR when toggle On or `RAWVIEWER_EDIT_CHROMA_DENOISE=1`.

### Recovery baseline (session UI flag)

| Control | Internal key | Notes |
|---------|--------------|-------|
| Use recovery look | `_recovery_baseline` | Not written to XMP |

When enabled: skips PV2012 tone; uses P-key recovery tone map + local shadow/highlight recovery. Exposure / WB / Sat / Vibrance / HSL / Detail still apply. Moving PV tone sliders or editing the point curve clears recovery baseline.

---

## UI (`rawviewer_ui/adjust_panel.py`)

- Floating panel (**E**), draggable card, scrollable content
- Sliders: immediate value readout; **80 ms** throttled preview; XMP on release
- Click value label → reset single slider; **Reset** → all defaults + as-shot Temp
- **Tone curve** graph (`tone_curve_widget.py`) above parametric sliders
- **HSL** section before Detail sliders
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
| `raw_chroma_denoise.py` | Chroma-only NLM (OpenCV) |
| `raw_edit_pipeline.py` | Shared pipeline + export dispatch |
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
| `RAWVIEWER_EDIT_CHROMA_DENOISE` | `1` | Chroma NLM on export when panel NR off |

---

## Verification checklist

1. Open RAW → **E** → wait for edit base (linear demosaic, not embedded JPEG)
2. Drag **Exposure** / **Shadows** — PV2012 tone, not gamma add
3. Edit **point curve** — live preview; release saves XMP; LR round-trip if re-opened
4. **HSL** — switch colors; values persist; preview updates
5. **P** recovery preview → **Use recovery look** → matches recovery tone; tweak Exposure / Sat
6. **Export…** — TIFF / JPEG / WebP / DNG options
7. Shortcuts (**P**, **J**, arrows, etc.) work with Adjust panel open
8. `PYTHONPATH=src python3 scripts/phase_develop_adjust_linear.py`

---

## Known limitations / future work

- No per-channel (R/G/B) tone curves
- No mask / brush / gradient local edits
- Recovery baseline is preview-session UI state, not persisted in XMP
- Baked DNG is RGB container DNG, not a re-encoded sensor RAW
- Point curve uses linear interpolation between knots (not LR cubic spline)
