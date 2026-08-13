"""Share payloads: JPEG proxies for messaging apps, original files for AirDrop.

WhatsApp, Signal and similar share extensions typically advertise
``public.jpeg`` / ``public.png``, not camera RAW. Listing those originals
in ``NSSharingService.sharingServicesForItems_`` (or the Windows share
flyout) hides the apps. AirDrop and Mail still accept the RAW.
"""

from __future__ import annotations

import os
from typing import Optional

# Formats messaging apps generally accept as a photo.
SHARE_DIRECT_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".heic",
    ".heif",
    ".bmp",
}


def needs_jpeg_proxy(path: str) -> bool:
    """True when the file should be re-encoded (or an embedded JPEG extracted)."""
    ext = os.path.splitext(path or "")[1].lower()
    if not ext:
        return True
    return ext not in SHARE_DIRECT_EXTS


def extract_embedded_jpeg_bytes(path: str) -> Optional[bytes]:
    """Return the largest LibRaw JPEG preview, or None."""
    try:
        import rawpy
        from rawpy_gil import rawpy_imread_warm

        with rawpy_imread_warm(path) as raw:
            thumb = raw.extract_thumb()
        if thumb is None or getattr(thumb, "format", None) != rawpy.ThumbFormat.JPEG:
            return None
        data = bytes(thumb.data or b"")
        if len(data) >= 32 and data[:2] == b"\xff\xd8":
            return data
    except Exception:
        return None
    return None


def write_jpeg_proxy(src_path: str, dest_path: str) -> bool:
    """Write a JPEG WhatsApp/Signal can open. Prefer the embedded preview."""
    if not src_path or not os.path.isfile(src_path):
        return False
    try:
        embedded = extract_embedded_jpeg_bytes(src_path)
        if embedded:
            with open(dest_path, "wb") as fh:
                fh.write(embedded)
            return os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        pass
    try:
        from PIL import Image, ImageOps

        with Image.open(src_path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")
            im.save(dest_path, format="JPEG", quality=90, optimize=True)
        return os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        return False
