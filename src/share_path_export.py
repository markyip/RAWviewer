"""Prepare file paths for external share / open (RAW → shareable JPEG when needed)."""

from __future__ import annotations

import os
import tempfile
from typing import Callable, List, Optional, Sequence, Tuple

# Formats most messaging apps and Windows Share targets accept reliably.
SHARE_AS_IS_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".gif",
        ".bmp",
        ".webp",
        ".heic",
        ".heif",
    }
)

# Batch-open apps that accept camera RAW on the command line (pass through unchanged).
BATCH_RAW_OK_APP_IDS = frozenset(
    {
        "lightroom_classic",
        "lightroom_cc",
        "capture_one",
        "on1_photo_raw",
        "dxo_photolab",
        "photoshop",
        "affinity_photo_2",
    }
)


def path_needs_share_jpeg_export(path: str) -> bool:
    from common_image_loader import is_raw_file

    if not path or not os.path.isfile(path):
        return False
    if is_raw_file(path):
        return True
    ext = os.path.splitext(path)[1].lower()
    return ext not in SHARE_AS_IS_EXTENSIONS


def _jpeg_bytes_for_share(path: str) -> Optional[bytes]:
    """Embedded or cached JPEG suitable for sharing (not full RAW decode)."""
    from image_cache import get_image_cache

    cache = get_image_cache()
    jpeg_data = cache.disk_thumbnail_cache.get(path)
    if jpeg_data:
        return jpeg_data

    preview_jpeg = cache.disk_preview_cache.get(path)
    if preview_jpeg:
        return preview_jpeg

    grid_jpeg = cache.disk_grid_cache.get(path)
    if grid_jpeg:
        return grid_jpeg

    try:
        import rawpy
        from PIL import Image, ImageOps
        import io

        with rawpy.imread(path) as raw:
            thumb = raw.extract_thumb()
            if thumb is None or thumb.format != rawpy.ThumbFormat.JPEG:
                return None
            pil_image = Image.open(io.BytesIO(thumb.data))
            pil_image = ImageOps.exif_transpose(pil_image)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=92)
            return buf.getvalue()
    except Exception:
        return None

    return None


def _write_temp_jpeg(source_path: str, jpeg_data: bytes) -> str:
    base = os.path.splitext(os.path.basename(source_path))[0] or "image"
    fd, temp_path = tempfile.mkstemp(prefix=f"rawviewer_share_{base}_", suffix=".jpg")
    try:
        os.write(fd, jpeg_data)
    finally:
        os.close(fd)
    return os.path.abspath(temp_path)


def prepare_paths_for_external_share(
    paths: Sequence[str],
    *,
    allow_raw: bool = False,
) -> Tuple[List[str], Callable[[], None]]:
    """
    Return paths safe for Share / messaging / editors.

    RAW files are exported to temp JPEG unless *allow_raw* is True (Lightroom, etc.).
    Second return value cleans up any temp files created.
    """
    out: List[str] = []
    temps: List[str] = []

    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        abs_path = os.path.abspath(path)
        if allow_raw or not path_needs_share_jpeg_export(abs_path):
            out.append(abs_path)
            continue
        jpeg_data = _jpeg_bytes_for_share(abs_path)
        if not jpeg_data:
            continue
        temp_path = _write_temp_jpeg(abs_path, jpeg_data)
        temps.append(temp_path)
        out.append(temp_path)

    if temps:
        import logging

        logging.getLogger("rawviewer.share").info(
            "exported %d RAW/unshareable file(s) to temp JPEG for external share",
            len(temps),
        )

    def _cleanup() -> None:
        for temp in temps:
            try:
                if os.path.isfile(temp):
                    os.remove(temp)
            except OSError:
                pass

    return out, _cleanup


def prepare_paths_for_batch_open(
    paths: Sequence[str],
    app_id: str,
) -> Tuple[List[str], Callable[[], None]]:
    allow_raw = app_id in BATCH_RAW_OK_APP_IDS
    return prepare_paths_for_external_share(paths, allow_raw=allow_raw)
