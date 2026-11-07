"""Mipmap tier publish/backfill helpers for the image cache."""
from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np
from PIL import Image
from PyQt6.QtGui import QImage

from image_cache import get_image_cache

_TIER_CACHE_ATTRS = {
    "preview": ("preview_cache", "disk_preview_cache"),
    "grid": ("grid_cache", "disk_grid_cache"),
    "thumbnail": ("thumbnail_cache", "disk_thumbnail_cache"),
}
def _mipmap_publish_debug_enabled() -> bool:
    return os.environ.get("RAWVIEWER_MIPMAP_PUBLISH_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
def _tile_cache_encode_format() -> str:
    """Output format for grid/thumbnail/preview disk-cache tiles.

    JPEG by default. WebP is ~25-35% smaller at equivalent quality with the
    same encode cost, but PIL's WebP *decode* is ~5-6x slower than JPEG at
    these tile sizes (measured directly: ~3ms vs ~0.6ms for a 512x384 tile,
    both synthetic-gradient and photo-like content) -- and this cache tier is
    read synchronously on the gallery's main-thread scroll path
    (get_grid() -> disk_grid_cache), every cache-miss-in-memory tile decode
    happening inline during a scroll frame. That decode cost, not the disk
    footprint, is what mattered here, so WebP was a net loss for this
    specific hot path. RAWVIEWER_TILE_CACHE_FORMAT=webp opts back in.
    Deliberately NOT used for PR-7's raw embedded-JPEG-bytes cache (a
    verbatim byte-slice from the RAW file, not a re-encode -- converting that
    would force a decode+encode pass the whole point of that cache is to
    avoid). Read side is format-agnostic: PIL.Image.open() sniffs format from
    content, so existing WebP entries keep working and this can be toggled
    freely without a migration.
    """
    raw = os.environ.get("RAWVIEWER_TILE_CACHE_FORMAT", "jpeg").strip().lower()
    return "WEBP" if raw == "webp" else "JPEG"
def encode_tile_bytes(pil_image, *, quality: int = 85) -> bytes:
    """Encode a PIL image for a grid/thumbnail/preview disk-cache tile."""
    import io

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    fmt = _tile_cache_encode_format()
    buf = io.BytesIO()
    try:
        pil_image.save(buf, format=fmt, quality=quality)
    except Exception:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
def _bump_mipmap_stat(cache, key: str) -> None:
    """Debug-only counters surfaced via ImageCache.get_cache_stats()['request_stats']."""
    with contextlib.suppress(Exception):
        cache.stats[key] = int(cache.stats.get(key, 0)) + 1
def _tier_really_present(cache, file_path: str, tier: str) -> bool:
    """Direct presence check for ONE tier, bypassing get_preview()/get_grid()'s
    cross-tier "dynamic mipmap fallback" (which synthesizes a tier by up/down-sampling
    a different tier and PERSISTS it as a side effect of merely reading). Calling the
    public get_X() here would let a presence check silently manufacture and cache a
    blurry upsampled tier, defeating the "never upsample" guarantee below -- so tier
    existence must be checked against the memory/disk stores directly.
    """
    mem_attr, disk_attr = _TIER_CACHE_ATTRS[tier]
    try:
        mem_cache = getattr(cache, mem_attr)
        if mem_cache.get(cache._path_key(file_path)) is not None:
            return True
    except Exception:
        pass
    try:
        disk_cache = getattr(cache, disk_attr)
        if hasattr(disk_cache, "has_valid") and disk_cache.has_valid(file_path):
            return True
    except Exception:
        pass
    return False
def publish_mipmap_tiers(
    file_path: str,
    native_array,
    *,
    exif_data: dict[str, Any] | None = None,
    cache=None,
    source: str = "decode",
) -> str:
    """Backfill missing preview/grid/thumbnail ImageCache tiers from one decoded array.

    Single write point for "I just decoded this file, publish it so every other
    consumer (gallery, film strip, semantic/face indexing) can reuse the pixels
    instead of re-extracting/re-decoding the same RAW." Each tier is filled
    independently and NEVER upsampled: a small decode (e.g. a 256px face-scan
    thumbnail) only satisfies the tiers it's large enough for without fabricating
    a fake higher-resolution tier. Tiers that already exist are left untouched.

    Returns the highest tier now present for this path: "preview" > "grid" >
    "thumbnail" > "" (nothing present/written).
    """
    if native_array is None:
        return ""

    if isinstance(native_array, QImage):
        from enhanced_raw_processor import _qimage_to_rgb_array

        native_array = _qimage_to_rgb_array(native_array)
    if native_array is None or not isinstance(native_array, np.ndarray):
        return ""

    try:
        h, w = native_array.shape[:2]
    except Exception:
        return ""
    if h <= 0 or w <= 0:
        return ""
    long_edge = max(h, w)

    cache = cache or get_image_cache()
    _bump_mipmap_stat(cache, "mipmap_publish_count")

    have_preview = _tier_really_present(cache, file_path, "preview")
    have_grid = _tier_really_present(cache, file_path, "grid")
    have_thumb = _tier_really_present(cache, file_path, "thumbnail")

    if have_preview:
        highest = "preview"
    elif have_grid:
        highest = "grid"
    elif have_thumb:
        highest = "thumbnail"
    else:
        highest = ""


    arr = native_array.astype(np.uint8) if native_array.dtype != np.uint8 else native_array
    _pil_cache = []

    def _as_pil():
        if not _pil_cache:
            if len(arr.shape) == 2:
                _pil_cache.append(Image.fromarray(arr, "L").convert("RGB"))
            elif len(arr.shape) == 3:
                _pil_cache.append(Image.fromarray(arr, "RGB"))
            else:
                _pil_cache.append(None)
        return _pil_cache[0]

    def _resized(img, target: int):
        if w <= target and h <= target:
            return img
        scale = min(target / w, target / h)
        return img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.HAMMING
        )

    # A tier is only written when the source is at least ~85% of its target long
    # edge -- the same tolerance used elsewhere (pixmap_matches_exif_display) for
    # "close enough, don't bother re-deriving." Anything smaller is left for a real
    # decode later rather than silently upsampled (upsampled tiers look soft and
    # would otherwise get cached as if they were native-resolution).
    from image_cache import disk_preview_max_edge

    pv_max = disk_preview_max_edge()
    if not have_preview and long_edge >= int(pv_max * 0.85):
        try:
            img = _as_pil()
            if img is not None:
                cache.put_preview(file_path, np.array(_resized(img, pv_max)))
                _bump_mipmap_stat(cache, "tier_backfill_count")
                if highest in ("", "grid", "thumbnail"):
                    highest = "preview"
        except Exception:
            pass

    grid_max = disk_preview_max_edge()
    if not have_grid and long_edge >= int(grid_max * 0.85):
        try:
            img = _as_pil()
            if img is not None:
                grid_pil = _resized(img, grid_max)
                cache.put_grid(file_path, np.array(grid_pil), encode_tile_bytes(grid_pil))
                _bump_mipmap_stat(cache, "tier_backfill_count")
                if highest == "":
                    highest = "grid"
        except Exception:
            pass

    if not have_thumb and long_edge >= int(256 * 0.85):
        try:
            img = _as_pil()
            if img is not None:
                thumb_pil = _resized(img, 256)
                cache.put_thumbnail(file_path, np.array(thumb_pil), encode_tile_bytes(thumb_pil))
                _bump_mipmap_stat(cache, "tier_backfill_count")
                if highest == "":
                    highest = "thumbnail"
        except Exception:
            pass

    if exif_data:
        try:
            cur = cache.get_exif(file_path)
            if cur is None or cur.get("minimal_preview_exif"):
                cache.put_exif(
                    file_path,
                    exif_data,
                    persist_disk=not bool(exif_data.get("minimal_preview_exif")),
                )
        except Exception:
            pass

    if _mipmap_publish_debug_enabled():
        try:
            import logging

            logging.getLogger(__name__).info(
                "[MIPMAP] publish file=%s source=%s in=%dx%d "
                "had(preview=%s,grid=%s,thumb=%s) -> highest=%s",
                os.path.basename(file_path),
                source,
                w,
                h,
                have_preview,
                have_grid,
                have_thumb,
                highest,
            )
        except Exception:
            pass

    return highest

