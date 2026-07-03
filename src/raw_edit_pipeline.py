"""Shared edit decode → denoise → PV2012 adjust → display / 16-bit export."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from raw_adjustments import DEFAULT_ADJUSTMENTS, is_default_adjustments, uses_recovery_tone_map
from raw_chroma_denoise import apply_chroma_denoise, apply_luma_denoise, chroma_denoise_enabled
from raw_pv2012 import apply_pv2012_tone_rgb


def _linear_float_from_buffer(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image.dtype == np.uint16:
        return rgb_image.astype(np.float32) / 65535.0
    if rgb_image.dtype == np.float32:
        return np.clip(rgb_image.astype(np.float32), 0.0, None)
    return rgb_image.astype(np.float32) / 255.0


def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    from raw_adjustments import _kelvin_to_rgb as kelvin

    return kelvin(k)


def _apply_wb_tint(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    from raw_adjustments import wb_reference_temperature

    ref_temp = wb_reference_temperature(merged)
    temp_val = float(merged.get("Temperature", ref_temp))
    tint_val = float(merged.get("Tint", 0.0))
    if temp_val > 1000.0 and abs(temp_val - ref_temp) > 1.0:
        r_tgt, g_tgt, b_tgt = _kelvin_to_rgb(temp_val)
        r_ref, g_ref, b_ref = _kelvin_to_rgb(ref_temp)
        if g_ref > 1e-5 and g_tgt > 1e-5:
            img[:, :, 0] *= (r_tgt / r_ref) / (g_tgt / g_ref)
            img[:, :, 2] *= (b_tgt / b_ref) / (g_tgt / g_ref)
    elif abs(temp_val) > 1e-4 and temp_val <= 1000.0:
        scale = temp_val / 100.0
        img[:, :, 0] *= 1.0 + scale * 0.1
        img[:, :, 2] *= 1.0 - scale * 0.1
    if abs(tint_val) > 1e-4:
        img[:, :, 1] *= 1.0 - (tint_val / 150.0) * 0.1
    return img


def _apply_saturation_vibrance(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    """
    Saturation / vibrance via HSV S-channel scale in gamma-encoded (perceptual) space.

    Scaling chroma (img - luma) directly on scene-linear RGB made the result wildly
    hue-dependent: hues with large linear-light channel spread (green, yellow) blew
    past the gamut boundary and clipped to a flat, fully-saturated block, while hues
    with small linear spread (skin tones, blue sky) barely moved — the "colors look
    off / no pop" symptom. Round-tripping through the sRGB OETF (nearest-index LUT,
    not np.power — see docs/ADJUST_LINEAR_PIPELINE.md) and scaling HSV S there matches
    how a "Saturation" slider is perceived, and S is bounded to [0, 1] by construction
    so it can't overshoot the gamut the way additive chroma scaling did.
    """
    import cv2

    from raw_tone_recovery import _decode_srgb_float01, _encode_srgb_float01

    sat_val = float(merged.get("Saturation", 0.0))
    vib_val = float(merged.get("Vibrance", 0.0))
    if abs(sat_val) < 1e-4 and abs(vib_val) < 1e-4:
        return img

    rgb_lin = np.clip(img.astype(np.float32), 0.0, 1.0)
    rgb_enc = _encode_srgb_float01(rgb_lin)
    bgr = np.ascontiguousarray(rgb_enc[..., ::-1])
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))

    scale = np.ones_like(s)
    if abs(sat_val) > 1e-4:
        scale = scale + sat_val / 100.0
    if abs(vib_val) > 1e-4:
        scale = scale + (vib_val / 100.0) * (1.0 - s)
    s = np.clip(s * scale, 0.0, 1.0)

    bgr_out = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    rgb_enc_out = np.clip(bgr_out[..., ::-1], 0.0, 1.0)
    return _decode_srgb_float01(rgb_enc_out)


def _tone_map_reinhard_display(img: np.ndarray) -> np.ndarray:
    """Scene-linear float → display-linear float in [0, 1] (Reinhard shoulder)."""
    from raw_tone_recovery import _luminance

    lum = _luminance(img)
    mapped = lum / (1.0 + np.maximum(lum, 0.0))
    scale = mapped / np.maximum(lum, 1e-6)
    return np.clip(img * scale[..., np.newaxis], 0.0, None)


def _tone_map_recovery_display(img: np.ndarray) -> np.ndarray:
    """Recovery preview tone path (matches P-key recovery, including edge cap)."""
    from raw_tone_recovery import (
        RECOVERY_MAX_EDGE,
        _cap_max_edge_linear,
        apply_local_shadow_highlight_recovery_display,
        linear_tone_map_to_display,
    )

    h, w = img.shape[:2]
    rgb_f = img.astype(np.float32)
    if max(h, w) <= RECOVERY_MAX_EDGE:
        display = linear_tone_map_to_display(rgb_f)
        return apply_local_shadow_highlight_recovery_display(display)

    small = _cap_max_edge_linear(rgb_f, RECOVERY_MAX_EDGE)
    polished = apply_local_shadow_highlight_recovery_display(
        linear_tone_map_to_display(small)
    )
    try:
        from scipy.ndimage import zoom

        sy = h / polished.shape[0]
        sx = w / polished.shape[1]
        return np.clip(
            zoom(polished, (sy, sx, 1.0), order=1),
            0.0,
            1.0,
        ).astype(np.float32)
    except ImportError:
        from PIL import Image

        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
        out8 = (np.clip(polished, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        up = np.asarray(Image.fromarray(out8, mode="RGB").resize((w, h), resample))
        return up.astype(np.float32) / 255.0


def _tone_map_for_display(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    if uses_recovery_tone_map(merged):
        return _tone_map_recovery_display(img)
    return _tone_map_reinhard_display(img)


def _apply_display_stage(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    """Tone map → sat/vib → HSL → detail (display-linear float [0, 1])."""
    from raw_detail_enhance import apply_detail_enhancements
    from raw_hsl import apply_hsl_adjustments

    display = _tone_map_for_display(img, merged)
    if merged:
        display = _apply_display_color_adjustments(display, merged)
        display = apply_hsl_adjustments(display, merged)
    display = apply_detail_enhancements(display, merged)
    return display


def process_linear_edit_buffer(
    rgb_image: np.ndarray,
    adj: dict[str, float],
    *,
    preview: bool = False,
    chroma_denoise: Optional[bool] = None,
) -> np.ndarray:
    """
    Scene-linear uint16/float → adjusted scene-linear float32.

    Order: WB → exposure → chroma/luma denoise (bilateral) → PV2012 tone → saturation.
    """
    if rgb_image is None:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    img = _linear_float_from_buffer(rgb_image)

    if is_default_adjustments(merged) and not uses_recovery_tone_map(merged):
        return img

    img = _apply_wb_tint(img, merged)

    exp_val = float(merged.get("Exposure2012", 0.0))
    if abs(exp_val) > 1e-4:
        img *= 2.0 ** exp_val

    do_denoise = chroma_denoise if chroma_denoise is not None else chroma_denoise_enabled()
    nr_amount = float(merged.get("ColorNoiseReduction", 0.0))
    if nr_amount > 1e-4:
        img = apply_chroma_denoise(
            img, strength=min(1.5, nr_amount / 100.0 * 1.25), preview=preview
        )
    elif do_denoise and not preview:
        img = apply_chroma_denoise(img, preview=False)

    luma_nr_amount = float(merged.get("LuminanceNoiseReduction", 0.0))
    if luma_nr_amount > 1e-4:
        img = apply_luma_denoise(img, strength=luma_nr_amount / 100.0, preview=preview)

    if not uses_recovery_tone_map(merged):
        img = apply_pv2012_tone_rgb(img, merged)
    return np.clip(img, 0.0, None)


def _apply_display_color_adjustments(
    display_linear: np.ndarray, merged: dict[str, float]
) -> np.ndarray:
    """Saturation / vibrance in display-linear space (after tone mapping)."""
    return _apply_saturation_vibrance(display_linear, merged)


def linear_to_display_uint8(img: np.ndarray, adj: dict[str, float] | None = None) -> np.ndarray:
    from raw_tone_recovery import _encode_srgb8

    merged = dict(adj or {})
    display = _apply_display_stage(img, merged)
    return _encode_srgb8(display)


def linear_to_export_uint16_srgb(img: np.ndarray, adj: dict[str, float] | None = None) -> np.ndarray:
    """Display-referred 16-bit sRGB for TIFF / DNG export."""
    from raw_tone_recovery import _encode_srgb16

    merged = dict(adj or {})
    display = _apply_display_stage(img, merged)
    return _encode_srgb16(display.astype(np.float32))


def _ensure_parent_dir(output_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _chroma_denoise_for_export(adj: dict[str, float]) -> bool:
    return float((adj or {}).get("ColorNoiseReduction", 0.0)) > 0 or chroma_denoise_enabled()


def _process_for_export(rgb_linear: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    return process_linear_edit_buffer(
        rgb_linear,
        adj,
        preview=False,
        chroma_denoise=_chroma_denoise_for_export(adj),
    )


def export_adjusted_tiff16(
    rgb_linear: np.ndarray,
    adj: dict[str, float],
    output_path: str,
    *,
    embed_xmp_path: Optional[str] = None,
) -> None:
    """Bake adjustments to 16-bit sRGB TIFF; optionally embed XMP packet."""
    processed = _process_for_export(rgb_linear, adj)
    out = linear_to_export_uint16_srgb(processed, adj)
    from PIL import Image

    im = Image.fromarray(out, mode="RGB")
    save_kwargs: dict = {"format": "TIFF", "compression": "tiff_lzw"}
    if embed_xmp_path and os.path.isfile(embed_xmp_path):
        with open(embed_xmp_path, "rb") as f:
            xmp_bytes = f.read()
        save_kwargs["xmp"] = xmp_bytes
    _ensure_parent_dir(output_path)
    im.save(output_path, **save_kwargs)


def export_adjusted_jpeg(
    rgb_linear: np.ndarray,
    adj: dict[str, float],
    output_path: str,
    *,
    quality: int = 92,
) -> None:
    """Bake adjustments to 8-bit JPEG."""
    processed = _process_for_export(rgb_linear, adj)
    out = linear_to_display_uint8(processed, adj)
    from PIL import Image

    im = Image.fromarray(out, mode="RGB")
    _ensure_parent_dir(output_path)
    im.save(
        output_path,
        format="JPEG",
        quality=max(1, min(100, int(quality))),
        subsampling=0,
        optimize=True,
    )


def export_adjusted_webp(
    rgb_linear: np.ndarray,
    adj: dict[str, float],
    output_path: str,
    *,
    quality: int = 88,
) -> None:
    """Bake adjustments to 8-bit WebP."""
    processed = _process_for_export(rgb_linear, adj)
    out = linear_to_display_uint8(processed, adj)
    from PIL import Image

    im = Image.fromarray(out, mode="RGB")
    _ensure_parent_dir(output_path)
    im.save(
        output_path,
        format="WEBP",
        quality=max(1, min(100, int(quality))),
        method=4,
    )


def _dng_tiffinfo():
    """Minimal DNGVersion tag so Lightroom / ACR can open baked RGB exports."""
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    info = ImageFileDirectory_v2()
    info[50706] = (1, 4, 0, 0)  # DNGVersion
    info[50741] = "RGB "  # UniqueCameraModel (padded to 4 chars)
    return info


def export_adjusted_dng_rgb(
    rgb_linear: np.ndarray,
    adj: dict[str, float],
    output_path: str,
    *,
    embed_xmp_path: Optional[str] = None,
) -> None:
    """Bake adjustments to 16-bit linear RGB DNG (display-referred sRGB, DNG container)."""
    processed = _process_for_export(rgb_linear, adj)
    out = linear_to_export_uint16_srgb(processed, adj)
    from PIL import Image

    im = Image.fromarray(out, mode="RGB")
    save_kwargs: dict = {
        "format": "TIFF",
        "compression": "tiff_lzw",
        "tiffinfo": _dng_tiffinfo(),
    }
    if embed_xmp_path and os.path.isfile(embed_xmp_path):
        with open(embed_xmp_path, "rb") as f:
            save_kwargs["xmp"] = f.read()
    _ensure_parent_dir(output_path)
    im.save(output_path, **save_kwargs)


def export_raw_with_xmp_settings(
    source_path: str,
    output_path: str,
    adj: dict[str, float],
) -> None:
    """
    Copy the original RAW/DNG and write a Lightroom-compatible XMP sidecar.

    Non-destructive round-trip for Lightroom / Camera Raw (settings only).
    """
    import shutil

    from raw_adjustments import write_xmp_adjustments_for_file

    if not source_path or not os.path.isfile(source_path):
        raise FileNotFoundError("Source RAW file not found")
    _ensure_parent_dir(output_path)
    shutil.copy2(source_path, output_path)
    write_xmp_adjustments_for_file(output_path, adj)


EXPORT_FORMAT_TIFF16 = "tiff16"
EXPORT_FORMAT_JPEG = "jpeg"
EXPORT_FORMAT_WEBP = "webp"
EXPORT_FORMAT_DNG_RGB = "dng_rgb"
EXPORT_FORMAT_DNG_SETTINGS = "dng_settings"


def export_adjusted_image(
    export_format: str,
    *,
    source_path: str,
    output_path: str,
    rgb_linear: Optional[np.ndarray],
    adj: dict[str, float],
    embed_xmp_path: Optional[str] = None,
) -> None:
    """Dispatch baked or settings-only export."""
    fmt = (export_format or EXPORT_FORMAT_TIFF16).strip().lower()
    if fmt == EXPORT_FORMAT_DNG_SETTINGS:
        export_raw_with_xmp_settings(source_path, output_path, adj)
        return
    if rgb_linear is None:
        raise RuntimeError("Full-resolution RAW decode failed")
    if fmt == EXPORT_FORMAT_JPEG:
        export_adjusted_jpeg(rgb_linear, adj, output_path)
    elif fmt == EXPORT_FORMAT_WEBP:
        export_adjusted_webp(rgb_linear, adj, output_path)
    elif fmt == EXPORT_FORMAT_DNG_RGB:
        export_adjusted_dng_rgb(
            rgb_linear, adj, output_path, embed_xmp_path=embed_xmp_path
        )
    else:
        export_adjusted_tiff16(
            rgb_linear, adj, output_path, embed_xmp_path=embed_xmp_path
        )
