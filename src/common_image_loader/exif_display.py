"""EXIF display-orientation helpers for image loading and thumbnails."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PyQt6.QtGui import QImage, QPixmap

from image_cache import get_image_cache


def exif_display_dimensions(
    width: int,
    height: int,
    orientation: int,
) -> tuple[int, int]:
    """Pixel width/height after EXIF orientation is applied (display size).

    Sensor dimensions are often stored landscape (w > h) while orientation 6/8
  means the image should display portrait ΓÇö swap only when w > h.
    """
    w, h = int(width or 0), int(height or 0)
    if w <= 0 or h <= 0:
        return w, h
    o = int(orientation or 1)
    if o in (5, 6, 7, 8) and w > h:
        return h, w
    return w, h
def exif_display_aspect_ratio(
    width: int,
    height: int,
    orientation: int,
) -> float:
    """Width/height ratio for layout after EXIF orientation."""
    dw, dh = exif_display_dimensions(width, height, orientation)
    if dh <= 0:
        return 1.5
    return dw / dh
def exif_pixels_display_oriented(exif_data: dict[str, Any] | None) -> bool:
    """True when decode already baked EXIF orientation into pixel data."""
    return bool((exif_data or {}).get("pixels_display_oriented"))
def mark_exif_pixels_display_oriented(exif_data: dict[str, Any] | None) -> dict[str, Any]:
    """Record that container orientation is already applied to cached pixels."""
    data = dict(exif_data or {})
    data["orientation"] = 1
    data["pixels_display_oriented"] = True
    return data
def pixmap_matches_exif_display(
    pixmap_width: int,
    pixmap_height: int,
    original_width: int,
    original_height: int,
    orientation: int,
    *,
    tolerance: float = 0.14,
    pixels_display_oriented: bool = False,
) -> bool:
    """True when pixmap pixels already match EXIF display orientation (no extra rotation)."""
    if pixels_display_oriented:
        return True
    pw, ph = int(pixmap_width or 0), int(pixmap_height or 0)
    if pw <= 0 or ph <= 0:
        return False
    o = int(orientation or 1)
    # 180┬░ / mirror tags do not change aspect ratio ΓÇö never infer correctness from AR alone.
    if o in (2, 3, 4):
        return False
    dw, dh = exif_display_dimensions(original_width, original_height, orientation)
    if dw <= 0 or dh <= 0:
        return False
    if (ph > pw) != (dh > dw):
        return False
    exp_ar = dw / dh
    act_ar = pw / ph
    if exp_ar <= 0 or act_ar <= 0:
        return False
    ratio = exp_ar / act_ar if exp_ar >= act_ar else act_ar / exp_ar
    return ratio <= (1.0 + tolerance)
def array_matches_exif_display(
    pixel_width: int,
    pixel_height: int,
    exif_data: dict[str, Any] | None,
    *,
    tolerance: float = 0.14,
) -> bool:
    """True when RGB/thumbnail pixels already match container EXIF display orientation."""
    if not exif_data:
        return True
    ow = int(exif_data.get("original_width") or 0)
    oh = int(exif_data.get("original_height") or 0)
    if ow <= 0 or oh <= 0:
        return True
    if exif_pixels_display_oriented(exif_data):
        return True
    o = int(exif_data.get("orientation", 1) or 1)
    if o in (2, 3, 4):
        return False
    return pixmap_matches_exif_display(
        pixel_width,
        pixel_height,
        ow,
        oh,
        o,
        tolerance=tolerance,
    )
def apply_container_orientation_to_array(
    image_array: np.ndarray,
    orientation: int,
    exif_data: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Apply container EXIF orientation once; skip when pixels already match display."""
    if image_array is None:
        return None
    o = int(orientation or 1)
    if o == 1:
        return image_array

    h, w = image_array.shape[:2]

    # Make-independent double-rotation guard (single source of truth for orientation):
    # MEASURE the pixels' actual portrait/landscape against the EXIF display orientation
    # and rotate only if they disagree. This is idempotent ΓÇö calling it twice, or on an
    # embedded preview that is already display-oriented (Sony/Leica pre-rotate; many other
    # cameras' JPEG previews are display-oriented too), never double-rotates. We deliberately
    # do NOT branch on camera make: make allowlists are wrong for the many cameras that
    # pre-rotate their preview, which is what produced the "portrait shown sideways" bug.
    # exif_display_dimensions only ever swaps landscape->portrait, so this stays correct
    # whether original_width/height were stored as sensor dims or already-display dims.
    ow = int((exif_data or {}).get("original_width") or 0)
    oh = int((exif_data or {}).get("original_height") or 0)
    if ow <= 0 or oh <= 0:
        ow, oh = 3, 2  # Default to landscape sensor aspect ratio (matches most cameras)

    dw, dh = exif_display_dimensions(ow, oh, o)
    if (dh > dw) == (h > w) and (o not in (2, 3, 4) or exif_pixels_display_oriented(exif_data)):
        import logging
        logging.getLogger(__name__).debug(
            f"apply_container_orientation_to_array: skipping orientation {o} "
            f"because array shape {w}x{h} already matches expected display orientation."
        )
        return image_array

    if o == 2:
        return np.fliplr(image_array)
    if o == 3:
        return np.rot90(image_array, 2)
    if o == 4:
        return np.flipud(image_array)
    if o == 5:
        return np.rot90(np.fliplr(image_array), 1)
    if o == 6:
        return np.rot90(image_array, 3)
    if o == 7:
        return np.rot90(np.fliplr(image_array), 3)
    if o == 8:
        return np.rot90(image_array, 1)
    return image_array
def rgb_array_to_qpixmap(rgb_image: np.ndarray) -> QPixmap:
    """Convert oriented uint8 RGB/grayscale ndarray to QPixmap.

    Builds the QImage from the array's own buffer (no .tobytes() copy):
    QPixmap.fromImage() below deep-copies pixel data into the QPixmap's own
    native storage before this function returns, so the QImage never
    outlives rgb_image's scope -- safe to reference directly rather than
    paying for an extra full-frame copy on every conversion (this path runs
    on every gallery/thumbnail/display repaint).
    """
    if rgb_image is None or not hasattr(rgb_image, "shape"):
        return QPixmap()
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8, copy=False)
    if not rgb_image.flags["C_CONTIGUOUS"]:
        rgb_image = np.ascontiguousarray(rgb_image)
    h, w = rgb_image.shape[:2]
    if len(rgb_image.shape) == 2:
        qimg = QImage(
            rgb_image.data, w, h, w, QImage.Format.Format_Grayscale8
        )
    elif rgb_image.shape[2] >= 3:
        channels = int(rgb_image.shape[2])
        q_format = (
            QImage.Format.Format_RGB888
            if channels == 3
            else QImage.Format.Format_RGBA8888
        )
        qimg = QImage(
            rgb_image.data, w, h, channels * w, q_format
        )
    else:
        return QPixmap()

    qimg.ndarr = rgb_image  # Keep array alive explicitly
    return QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap()
def apply_container_orientation_to_pixmap(
    pixmap: QPixmap,
    exif_data: dict[str, Any] | None,
) -> QPixmap:
    """Apply container EXIF orientation to a QPixmap (idempotent with display-oriented pixels)."""
    from PyQt6.QtGui import QTransform

    if pixmap is None or pixmap.isNull() or not exif_data:
        return pixmap
    if exif_pixels_display_oriented(exif_data):
        return pixmap
    orientation = int(exif_data.get("orientation", 1) or 1)
    if orientation == 1:
        return pixmap

    ow = int(exif_data.get("original_width") or 0)
    oh = int(exif_data.get("original_height") or 0)
    if ow > 0 and oh > 0 and pixmap_matches_exif_display(
        pixmap.width(),
        pixmap.height(),
        ow,
        oh,
        orientation,
        pixels_display_oriented=exif_pixels_display_oriented(exif_data),
    ):
        return pixmap

    transform = QTransform()
    if orientation == 2:
        transform.scale(-1, 1)
    elif orientation == 3:
        transform.rotate(180)
    elif orientation == 4:
        transform.scale(1, -1)
    elif orientation == 5:
        transform.scale(-1, 1)
        transform.rotate(-90)
    elif orientation == 6:
        transform.rotate(90)
    elif orientation == 7:
        transform.scale(-1, 1)
        transform.rotate(90)
    elif orientation == 8:
        transform.rotate(-90)
    else:
        return pixmap
    return pixmap.transformed(transform)
def resolve_container_exif_for_file(
    file_path: str,
    *,
    cache=None,
) -> tuple[dict[str, Any] | None, int]:
    """Return (exif_data, orientation) from ImageCache or container EXIF on disk.

    verify=True: this is the orientation authority for pixel rotation
    (_heal_raw_orientation / finalize_index_thumbnail_array) -- an
    untrustworthy cached orientation here rotates pixels wrongly, so the
    synchronous probe stays. get_exif memoizes the successful probe to disk,
    so the cost is once per file ever, not per call.
    """
    cache = cache or get_image_cache()
    exif_data = cache.get_exif(file_path, verify=True)
    if not exif_data:
        try:
            from enhanced_raw_processor import EXIFExtractor

            exif_data = EXIFExtractor().extract_exif_data(file_path)
        except Exception:
            exif_data = None
    orientation = int((exif_data or {}).get("orientation", 1) or 1)
    return exif_data, orientation
def index_thumbnail_needs_orient_fix(
    file_path: str,
    arr: np.ndarray,
    *,
    cache=None,
) -> bool:
    """True when cached index pixels don't match container EXIF display orientation."""
    if arr is None:
        return True
    exif_data, _ = resolve_container_exif_for_file(file_path, cache=cache)
    if not exif_data:
        return False
    h, w = arr.shape[:2]
    return not array_matches_exif_display(w, h, exif_data)
def finalize_index_thumbnail_array(
    file_path: str,
    arr: np.ndarray,
    *,
    cache=None,
) -> np.ndarray:
    """Apply container EXIF orientation so semantic warm-up matches gallery thumbnails."""
    if arr is None:
        return arr
    exif_data, orientation = resolve_container_exif_for_file(file_path, cache=cache)
    if os.environ.get("RAWVIEWER_ORIENT_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            import logging as _lg
            _h, _w = arr.shape[:2]
            _ow = int((exif_data or {}).get("original_width") or 0)
            _oh = int((exif_data or {}).get("original_height") or 0)
            _src = "cache" if (cache is not None and cache.get_exif(file_path)) else "file"
            _lg.getLogger(__name__).info(
                "[ORIENT] finalize file=%s in=%dx%d(%s) resolved(o=%s ow=%d oh=%d src=%s)",
                os.path.basename(file_path), _w, _h,
                "P" if _h > _w else "L", orientation, _ow, _oh, _src,
            )
        except Exception:
            pass
    if orientation != 1:
        oriented = apply_container_orientation_to_array(arr, orientation, exif_data)
        if oriented is not None:
            return oriented
    return arr
def exif_rotation_degrees_for_pixmap(
    pixmap_width: int,
    pixmap_height: int,
    original_width: int,
    original_height: int,
    orientation: int,
) -> int:
    """Clockwise rotation (0/90/180/270) to align pixmap with EXIF display orientation."""
    pw, ph = int(pixmap_width or 0), int(pixmap_height or 0)
    if pw <= 0 or ph <= 0:
        return 0
    o = int(orientation or 1)
    ow, oh = int(original_width or 0), int(original_height or 0)
    act_portrait = ph > pw

    if ow > 0 and oh > 0:
        dw, dh = exif_display_dimensions(ow, oh, o)
        exp_portrait = dh > dw
        if exp_portrait == act_portrait:
            if o in (2, 3, 4):
                if o == 3:
                    return 180
                return 0
            return 0
        if not exp_portrait and act_portrait:
            # Thumbnail already portrait; metadata still landscape ΓÇö trust pixels.
            return 0
        if o == 3:
            return 180
        if o == 6:
            return 90
        if o == 8:
            return 270
        if o <= 1:
            if dh > dw and not act_portrait:
                return 90
            if dw > dh and act_portrait:
                return 270
        return 0

    if act_portrait and o in (6, 8):
        return 0
    if o == 3:
        return 180
    if o == 6:
        return 90
    if o == 8:
        return 270
    return 0

