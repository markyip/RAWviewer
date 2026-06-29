"""Highlight / shadow clipping masks for single-image view."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


DEFAULT_HI_THRESHOLD = 252
DEFAULT_LO_THRESHOLD = 3
_OVERLAY_MAX_SAMPLE = 1536
_OVERLAY_ALPHA = 190


def _downsample_rgb(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    max_dim = max(h, w)
    if max_dim <= max_side:
        return rgb, 1.0
    scale = max_side / max_dim
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    from PIL import Image

    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
    small = np.asarray(Image.fromarray(rgb).resize((nw, nh), resample))
    return np.ascontiguousarray(small), scale


def clipping_masks(
    rgb: np.ndarray,
    *,
    hi_threshold: int = DEFAULT_HI_THRESHOLD,
    lo_threshold: int = DEFAULT_LO_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (highlight_mask, shadow_mask) boolean arrays matching rgb H×W."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    highlight = (r >= hi_threshold) | (g >= hi_threshold) | (b >= hi_threshold)
    shadow = (r <= lo_threshold) & (g <= lo_threshold) & (b <= lo_threshold)
    return highlight, shadow


def clipping_counts(
    rgb: np.ndarray,
    *,
    hi_threshold: int = DEFAULT_HI_THRESHOLD,
    lo_threshold: int = DEFAULT_LO_THRESHOLD,
) -> tuple[int, int, int]:
    """Return (highlight_pixels, shadow_pixels, total_pixels)."""
    highlight, shadow = clipping_masks(
        rgb, hi_threshold=hi_threshold, lo_threshold=lo_threshold
    )
    total = int(rgb.shape[0] * rgb.shape[1])
    return int(np.count_nonzero(highlight)), int(np.count_nonzero(shadow)), total


def clipping_overlay_rgba(
    rgb: np.ndarray,
    *,
    hi_threshold: int = DEFAULT_HI_THRESHOLD,
    lo_threshold: int = DEFAULT_LO_THRESHOLD,
    alpha: int = _OVERLAY_ALPHA,
    max_sample: int = _OVERLAY_MAX_SAMPLE,
) -> np.ndarray:
    """Build H×W×4 uint8 RGBA overlay aligned to rgb dimensions."""
    h, w = rgb.shape[:2]
    work, _scale = _downsample_rgb(rgb, max_sample)
    hi, lo = clipping_masks(
        work, hi_threshold=hi_threshold, lo_threshold=lo_threshold
    )
    if not np.any(hi) and not np.any(lo):
        return np.zeros((h, w, 4), dtype=np.uint8)

    wh, ww = work.shape[:2]
    small_rgba = np.zeros((wh, ww, 4), dtype=np.uint8)
    if np.any(hi):
        small_rgba[hi, 0] = 255
        small_rgba[hi, 3] = alpha
    if np.any(lo):
        small_rgba[lo, 2] = 255
        small_rgba[lo, 3] = np.maximum(small_rgba[lo, 3], alpha)

    if wh == h and ww == w:
        return np.ascontiguousarray(small_rgba)

    from PIL import Image

    resample = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for ch in range(4):
        plane = Image.fromarray(small_rgba[:, :, ch]).resize((w, h), resample)
        out[:, :, ch] = np.asarray(plane)
    return np.ascontiguousarray(out)


def rgba_array_to_qpixmap(rgba: np.ndarray) -> QPixmap:
    h, w = rgba.shape[:2]
    image = QImage(
        rgba.data,
        w,
        h,
        w * 4,
        QImage.Format.Format_RGBA8888,
    ).copy()
    return QPixmap.fromImage(image)


def clipping_overlay_pixmap(
    rgb: np.ndarray,
    *,
    hi_threshold: int = DEFAULT_HI_THRESHOLD,
    lo_threshold: int = DEFAULT_LO_THRESHOLD,
) -> Optional[QPixmap]:
    rgba = clipping_overlay_rgba(
        rgb,
        hi_threshold=hi_threshold,
        lo_threshold=lo_threshold,
    )
    if not np.any(rgba[:, :, 3]):
        return None
    return rgba_array_to_qpixmap(rgba)


def rgb_array_from_pixmap(pixmap: QPixmap, max_side: int = 0) -> Optional[np.ndarray]:
    if pixmap is None or pixmap.isNull():
        return None
    img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    if w < 1 or h < 1:
        return None
    if max_side > 0 and max(w, h) > max_side:
        img = img.scaled(
            max(1, int(w * max_side / max(w, h))),
            max(1, int(h * max_side / max(w, h))),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    bits = img.constBits()
    if bits is None:
        return None
    try:
        if hasattr(bits, "asstring"):
            raw = bits.asstring(h * bpl)
        else:
            raw = bytes(memoryview(bits)[: h * bpl])
    except (BufferError, TypeError, AttributeError):
        return None
    arr = np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(h, bpl)
    return np.ascontiguousarray(arr[:, : w * 3].reshape(h, w, 3))
