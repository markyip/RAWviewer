"""macOS/Windows EDR display path for RAW loading."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QImage, QPixmap

from .exif_display import apply_container_orientation_to_array
from .helpers import use_libraw_consistent_preview_first

_last_load_used_edr = False
def take_last_load_edr_flag() -> bool:
    """True when the most recent load_pixmap_safe() used macOS HDR/EDR output."""
    global _last_load_used_edr
    used = bool(_last_load_used_edr)
    _last_load_used_edr = False
    return used
def is_macos_edr_enabled() -> bool:
    # EDR support is removed at this stage because the updated image loading pipeline is not compatible with EDR, causing slow loads.
    return False
def format_edr_status_label(
    *,
    display_mode: str | None = None,
    gpu_view=None,
) -> str | None:
    """Short label for metadata / status bar."""
    return None
def summarize_macos_edr_startup(*, gpu_view=None) -> str:
    """One-line startup hint for logs / status bar."""
    import platform

    if platform.system() != "Darwin":
        return "Display: SDR (not macOS)"
    if not is_macos_edr_enabled():
        return "Display: SDR (RAWVIEWER_DISABLE_EDR=1)"
    parts = ["macOS EDR viewport enabled"]
    if use_raw_edr_for_display():
        parts.append("RAW EDR active")
    elif use_raw_edr_display():
        parts.append("RAW EDR on hold (embedded JPEG workflow)")
    else:
        parts.append("RAW EDR off (RAWVIEWER_RAW_EDR=0)")
    if gpu_view is not None and not getattr(gpu_view, "_edr_initialized", False):
        parts.append("viewport initializingΓÇª")
    return " ┬╖ ".join(parts)
def use_raw_edr_display() -> bool:
    """macOS RAW EDR single-view path (on by default; RAWVIEWER_RAW_EDR=0 to disable).

    Also user-toggleable at runtime via the bottom-bar EDR button
    (QSettings ``raw_edr_display_enabled``) -- EDR display re-decodes each RAW
    with LibRaw highlight reconstruction (~12x the cost of a plain decode), so
    turning it off trades highlight headroom for much faster RAW browsing.
    An explicit RAWVIEWER_RAW_EDR=0 env still hard-disables regardless.

    This setting only gates *whether EDR applies at all*; the dispatch point
    (``RAWviewerMainWindow._should_use_raw_edr_display`` in main.py) further
    idle-defers the actual decode so rapid navigation doesn't pay the 12x
    cost on every frame -- see ``_maybe_arm_raw_edr_after_idle``.
    """
    if not is_macos_edr_enabled():
        return False
    v = os.environ.get("RAWVIEWER_RAW_EDR", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False

    settings = QSettings("RAWviewer", "RAWviewer")
    return settings.value("raw_edr_display_enabled", True, type=bool)
def use_raw_edr_for_display() -> bool:
    """True when macOS RAW EDR should replace the normal display path (RAW workflow only)."""
    if not use_raw_edr_display():
        return False
    # Embedded-JPEG workflow must keep camera preview pixels (SDR); do not re-decode via LibRaw.
    return use_libraw_consistent_preview_first()
def is_windows_hdr_display_available() -> bool:
    """Best-effort HDR display probe for future Windows 10-bit work (not wired yet)."""
    import sys

    if sys.platform != "win32":
        return False
    return False
def linear_edr_float_to_qpixmap(
    rgb_edr: np.ndarray,
    *,
    peak_display: float = 4.0,
    max_edge: int = 0,
) -> QPixmap:
    """Convert extended display-referred float RGB to macOS EDR RGBX64 QPixmap."""
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtGui import QColorSpace

    from raw_adjustments.tone_recovery import encode_edr_rgbx64

    if rgb_edr is None or rgb_edr.size == 0:
        return QPixmap()
    arr_64 = encode_edr_rgbx64(rgb_edr, peak_display=peak_display)
    h, w = arr_64.shape[:2]
    qimage = QImage(arr_64.data, w, h, w * 8, QImage.Format.Format_RGBX64)
    qimage.ndarr = arr_64  # Keep array alive
    qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    if max_edge > 0 and max(w, h) > max_edge:
        if w >= h:
            sw = max_edge
            sh = max(1, int(h * max_edge / max(w, 1)))
        else:
            sh = max_edge
            sw = max(1, int(w * max_edge / max(h, 1)))
        qimage = qimage.scaled(
            QSize(sw, sh),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return QPixmap.fromImage(qimage) if not qimage.isNull() else QPixmap()
def decode_raw_edr_pixmap(
    file_path: str,
    *,
    max_edge: int = 0,
    exif_data: dict[str, Any] | None = None,
) -> QPixmap | None:
    """Decode RAW with linear 16-bit pipeline and build macOS EDR QPixmap."""
    if not use_raw_edr_for_display():
        return None
    from raw_adjustments.tone_recovery import decode_raw_for_edr_rgb

    apply_orientation = apply_container_orientation_to_array
    rgb_edr = decode_raw_for_edr_rgb(
        file_path,
        max_edge=max_edge,
        apply_orientation=apply_orientation,
        exif_data=exif_data,
    )
    if rgb_edr is None:
        return None
    pixmap = linear_edr_float_to_qpixmap(rgb_edr, max_edge=max_edge)
    return pixmap if not pixmap.isNull() else None
def pq_to_linear(v: np.ndarray) -> np.ndarray:
    m1 = 2610.0 / 16384.0
    m2 = (2523.0 / 32.0) * 128.0
    c1 = 3424.0 / 4096.0
    c2 = (2413.0 / 4096.0) * 32.0
    c3 = (2392.0 / 4096.0) * 32.0

    v_pow = np.power(v, 1.0 / m2)
    numerator = np.maximum(v_pow - c1, 0.0)
    denominator = c2 - c3 * v_pow
    # Avoid divide by zero
    denominator = np.where(denominator == 0.0, 1e-8, denominator)
    linear = np.power(numerator / denominator, 1.0 / m1)
    return linear
def hlg_to_linear(v: np.ndarray) -> np.ndarray:
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    linear = np.zeros_like(v, dtype=np.float32)
    mask_low = v <= 0.5
    linear[mask_low] = (v[mask_low] ** 2) / 3.0

    mask_high = ~mask_low
    # Avoid domain error in exp
    exponent = np.clip((v[mask_high] - c) / a, -80.0, 80.0)
    linear[mask_high] = (np.exp(exponent) + b) / 12.0
    return linear
def linear_to_srgb(arr: np.ndarray) -> np.ndarray:
    srgb = np.zeros_like(arr, dtype=np.float32)
    mask_low = arr <= 0.0031308
    srgb[mask_low] = 12.92 * arr[mask_low]
    mask_high = ~mask_low
    srgb[mask_high] = 1.055 * np.power(np.maximum(arr[mask_high], 0.0), 1.0 / 2.4) - 0.055
    return np.clip(srgb, 0.0, 1.0)
def tone_map_reinhard(img_float: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    img_exposed = img_float * exposure
    mapped = img_exposed / (1.0 + img_exposed)
    return np.clip(mapped, 0.0, 1.0)

