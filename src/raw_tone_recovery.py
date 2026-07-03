"""RAW-only shadow / highlight recovery for single-view preview (P key)."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

RECOVERY_MAX_EDGE = 2048
_BLUR_SIGMA = 22.0
_SHADOW_LIFT = 0.28
_HIGHLIGHT_ROLL = 0.62
_SHADOW_LUM_MAX = 0.28
_HIGHLIGHT_LUM_MIN = 0.50
_HIGHLIGHT_BASE_MIN = 0.62
_HIGHLIGHT_KNEE = 0.72
_MIN_HIGHLIGHT_RATIO = 0.48
_MAX_SHADOW_RATIO = 1.85
_LUM_FLOOR = 0.025
_EXPOSURE_ANCHOR_PCT = 92.0
_EXPOSURE_ANCHOR_TARGET = 0.52
_HARD_CLIP_LUM = 0.985
_HARD_CLIP_SPREAD = 0.025
_EDR_PEAK_DISPLAY = 4.0


def _luminance(rgb_f: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb_f[:, :, 0]
        + 0.7152 * rgb_f[:, :, 1]
        + 0.0722 * rgb_f[:, :, 2]
    )


def _to_float01(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 65535.0
    if rgb.dtype != np.float32:
        return rgb.astype(np.float32) / 255.0
    return np.clip(rgb.astype(np.float32), 0.0, 1.0)


def _highlight_reconstruct_mode():
    import rawpy

    try:
        return rawpy.HighlightMode.Reconstruct(5)
    except Exception:
        return rawpy.HighlightMode.ReconstructDefault


def recovery_decode_params(*, half_size: bool = True, demosaic: str | None = None):
    """Linear 16-bit decode with LibRaw highlight reconstruction."""
    import rawpy

    params = rawpy.Params(
        use_camera_wb=True,
        use_auto_wb=False,
        output_bps=16,
        gamma=(1, 1),
        no_auto_bright=True,
        bright=1.0,
        user_flip=0,
        half_size=half_size,
        highlight_mode=_highlight_reconstruct_mode(),
        exp_shift=1.0,
    )
    # exp_preser is not accepted by Params.__init__ on all rawpy builds.
    params.exp_preser = 0.45
    if demosaic:
        try:
            params.demosaic_algorithm = getattr(rawpy.DemosaicAlgorithm, demosaic)
        except (AttributeError, TypeError):
            pass
    return params


def _cap_max_edge_linear(rgb_f: np.ndarray, max_edge: int) -> np.ndarray:
    rgb_f = np.clip(rgb_f.astype(np.float32), 0.0, None)
    if max_edge <= 0:
        return rgb_f
    h, w = rgb_f.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_edge:
        return rgb_f
    scale = max_edge / max_dim
    try:
        from scipy.ndimage import zoom

        rgb_small = zoom(rgb_f, (scale, scale, 1.0), order=1)
        return np.clip(rgb_small.astype(np.float32), 0.0, None)
    except ImportError:
        from PIL import Image

        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
        rgb16 = (np.clip(rgb_f, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = np.asarray(Image.fromarray(rgb16, mode="RGB").resize((nw, nh), resample))
        return small.astype(np.float32) / 65535.0


def _exposure_anchor_scale(lum: np.ndarray, *, percentile: float, target: float) -> float:
    """Scale from a bright-scene percentile so deep shadows do not raise global exposure."""
    valid = lum > 1e-6
    if not np.any(valid):
        return 1.0
    ref = float(np.percentile(lum[valid], percentile))
    return target / max(ref, 1e-6)


def linear_tone_map_to_display(rgb_lin: np.ndarray) -> np.ndarray:
    """
    Map linear scene-referred RGB to display-referred float [0, 1].

    Exposure is anchored to a high luminance percentile (not log-average) so
    backlit shadows do not brighten the whole frame. Shadow lift is deferred to
    apply_local_shadow_highlight_recovery().
    """
    rgb = np.clip(rgb_lin.astype(np.float32), 0.0, None)
    lum = _luminance(rgb)
    rgb = rgb * _exposure_anchor_scale(
        lum, percentile=_EXPOSURE_ANCHOR_PCT, target=_EXPOSURE_ANCHOR_TARGET
    )

    lum = _luminance(rgb)
    lum_mapped = lum / (1.0 + lum)
    lum_safe = np.maximum(lum, 1e-6)
    rgb = rgb * (lum_mapped / lum_safe)[..., np.newaxis]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    linear_part = x <= 0.0031308
    return np.where(
        linear_part,
        x * 12.92,
        1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055,
    )


def linear_tone_map_to_edr(
    rgb_lin: np.ndarray,
    *,
    peak_display: float = _EDR_PEAK_DISPLAY,
) -> np.ndarray:
    """
    Map linear scene-referred RGB to extended display-referred float [0, peak].

    Unlike ``linear_tone_map_to_display``, highlight headroom is preserved above
    SDR white (1.0) up to ``peak_display`` for macOS EDR RGBX64 output.
    """
    peak_display = max(float(peak_display), 1.01)
    rgb = np.clip(rgb_lin.astype(np.float32), 0.0, None)
    lum = _luminance(rgb)
    rgb = rgb * _exposure_anchor_scale(
        lum, percentile=_EXPOSURE_ANCHOR_PCT, target=_EXPOSURE_ANCHOR_TARGET
    )

    lum = _luminance(rgb)
    sdr_part = lum / (1.0 + lum)
    excess = np.maximum(lum - 1.0, 0.0)
    headroom = peak_display - 0.5
    ext_part = 0.5 + excess * headroom / (excess + headroom)
    lum_mapped = np.where(lum <= 1.0, sdr_part, ext_part)
    lum_safe = np.maximum(lum, 1e-6)
    rgb = rgb * (lum_mapped / lum_safe)[..., np.newaxis]

    return np.clip(rgb, 0.0, peak_display).astype(np.float32)


def encode_edr_rgbx64(
    rgb_edr: np.ndarray,
    *,
    peak_display: float = _EDR_PEAK_DISPLAY,
) -> np.ndarray:
    """Pack extended display-referred RGB float into uint16 RGBX for QImage.Format_RGBX64."""
    peak_display = max(float(peak_display), 1.01)
    rgb = np.clip(rgb_edr.astype(np.float32), 0.0, peak_display)
    srgb = _srgb_oetf(rgb)
    peak_srgb = float(_srgb_oetf(np.array(peak_display, dtype=np.float32)))
    scale = 65535.0 / max(peak_srgb, 1e-6)
    u16 = np.clip(srgb * scale + 0.5, 0, 65535).astype(np.uint16)
    h, w = u16.shape[:2]
    padded = np.empty((h, w, 4), dtype=np.uint16)
    padded[:, :, :3] = u16
    padded[:, :, 3] = 65535
    return np.ascontiguousarray(padded)


def process_linear_edr_rgb(
    rgb: np.ndarray,
    *,
    max_edge: int = 0,
    peak_display: float = _EDR_PEAK_DISPLAY,
) -> np.ndarray:
    """Linear decode buffer → extended tone map (float32, not clipped to SDR white)."""
    rgb_f = _cap_max_edge_linear(_to_float01(rgb), max_edge)
    return linear_tone_map_to_edr(rgb_f, peak_display=peak_display)


def _hard_clip_mask(rgb_f: np.ndarray, lum: np.ndarray) -> np.ndarray:
    spread = np.max(rgb_f, axis=-1) - np.min(rgb_f, axis=-1)
    return (lum >= _HARD_CLIP_LUM) & (spread < _HARD_CLIP_SPREAD)


def _encode_srgb8(rgb_f: np.ndarray) -> np.ndarray:
    x = np.clip(rgb_f.astype(np.float32), 0.0, 1.0)
    linear_part = x <= 0.0031308
    srgb = np.where(linear_part, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.ascontiguousarray((np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def apply_local_shadow_highlight_recovery_display(rgb_disp: np.ndarray) -> np.ndarray:
    """
    Local shadow / highlight polish on display-referred float RGB [0, 1].
    Returns float display RGB (same space as ``linear_tone_map_to_display`` output).
    """
    if rgb_disp is None or rgb_disp.ndim != 3 or rgb_disp.shape[2] < 3:
        return rgb_disp
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        logger.warning("scipy unavailable; skipping local RAW recovery")
        return np.clip(_to_float01(rgb_disp), 0.0, 1.0).astype(np.float32)

    rgb_f = _to_float01(rgb_disp)
    lum = _luminance(rgb_f)
    base = gaussian_filter(lum, sigma=_BLUR_SIGMA)
    hard_clip = _hard_clip_mask(rgb_f, lum)

    shadow_w = np.clip((_SHADOW_LUM_MAX - lum) / _SHADOW_LUM_MAX, 0.0, 1.0) ** 1.5
    shadow_room = np.maximum(_SHADOW_LUM_MAX - lum, 0.0)
    lum_lift = shadow_w * _SHADOW_LIFT * shadow_room

    hi_base = np.clip((base - _HIGHLIGHT_BASE_MIN) / (1.0 - _HIGHLIGHT_BASE_MIN), 0.0, 1.0) ** 1.4
    hi_local = np.clip((lum - _HIGHLIGHT_LUM_MIN) / (1.0 - _HIGHLIGHT_LUM_MIN), 0.0, 1.0) ** 1.2
    hi_blown = np.clip((lum - 0.88) / 0.12, 0.0, 1.0) ** 1.5
    highlight_w = np.maximum(hi_base * hi_local, hi_blown * hi_local)
    highlight_w = np.where(hard_clip, 0.0, highlight_w)

    excess = np.maximum(lum - _HIGHLIGHT_KNEE, 0.0)
    lum_roll = highlight_w * (_HIGHLIGHT_ROLL * excess + 0.35 * excess * excess)

    lum_new = np.clip(lum + lum_lift - lum_roll, 0.0, 1.0)
    lum_safe = np.maximum(lum, _LUM_FLOOR)
    ratio = lum_new / lum_safe

    hi_dom = highlight_w >= shadow_w
    sh_dom = shadow_w > highlight_w
    ratio = np.where(
        hi_dom,
        np.clip(ratio, _MIN_HIGHLIGHT_RATIO, 1.0),
        np.where(sh_dom, np.clip(ratio, 1.0, _MAX_SHADOW_RATIO), 1.0),
    )
    active = np.maximum(shadow_w, highlight_w)
    ratio = 1.0 + (ratio - 1.0) * active
    ratio = np.where(hard_clip, 1.0, ratio)

    return np.clip(rgb_f * ratio[..., np.newaxis], 0.0, 1.0).astype(np.float32)


def apply_local_shadow_highlight_recovery(rgb: np.ndarray) -> np.ndarray:
    """
    Local shadow / highlight polish on display-referred float RGB [0, 1].

    Expects input after linear_tone_map_to_display(); skips highlight roll on
    hard-clipped whites to avoid flat gray disks in sensor-saturated areas.
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    out = apply_local_shadow_highlight_recovery_display(rgb)
    return _encode_srgb8(out)


def process_linear_recovery_rgb(
    rgb: np.ndarray,
    *,
    max_edge: int = RECOVERY_MAX_EDGE,
) -> np.ndarray:
    """Full v2 pipeline: linear decode buffer → tone map → local polish → sRGB8."""
    rgb_f = _cap_max_edge_linear(_to_float01(rgb), max_edge)
    rgb_disp = linear_tone_map_to_display(rgb_f)
    return apply_local_shadow_highlight_recovery(rgb_disp)


def decode_and_recover_raw(
    file_path: str,
    *,
    max_edge: int = RECOVERY_MAX_EDGE,
    apply_orientation: Optional[Callable[[np.ndarray, int, Any], np.ndarray]] = None,
    exif_data: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Decode RAW linear with highlight reconstruction, tone-map, return uint8 RGB."""
    import rawpy

    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        from enhanced_raw_processor import _rawpy_global_lock

        with _rawpy_global_lock:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(recovery_decode_params(half_size=True))
    except Exception as exc:
        logger.warning(
            "RAW recovery decode failed for %s: %s",
            os.path.basename(file_path),
            exc,
        )
        return None
    if rgb is None or rgb.size == 0:
        return None

    orientation = 1
    if exif_data:
        orientation = int(exif_data.get("orientation", 1) or 1)
    if apply_orientation is not None and orientation != 1:
        try:
            rgb = apply_orientation(rgb, orientation, exif_data)
        except Exception:
            pass

    return process_linear_recovery_rgb(rgb, max_edge=max_edge)


def decode_raw_for_edr_rgb(
    file_path: str,
    *,
    max_edge: int = 0,
    peak_display: float = _EDR_PEAK_DISPLAY,
    apply_orientation: Optional[Callable[[np.ndarray, int, Any], np.ndarray]] = None,
    exif_data: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Decode RAW linear 16-bit and tone-map for macOS EDR display (float32 RGB)."""
    import rawpy

    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        from enhanced_raw_processor import _rawpy_global_lock

        with _rawpy_global_lock:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(recovery_decode_params(half_size=True))
    except Exception as exc:
        logger.warning(
            "RAW EDR decode failed for %s: %s",
            os.path.basename(file_path),
            exc,
        )
        return None
    if rgb is None or rgb.size == 0:
        return None

    orientation = 1
    if exif_data:
        orientation = int(exif_data.get("orientation", 1) or 1)
    if apply_orientation is not None and orientation != 1:
        try:
            rgb = apply_orientation(rgb, orientation, exif_data)
        except Exception:
            pass

    edge = max_edge if max_edge > 0 else 0
    return process_linear_edr_rgb(rgb, max_edge=edge, peak_display=peak_display)
