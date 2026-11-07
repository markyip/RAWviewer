"""Thumbnail, grid, and preview cache operations."""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import pyqtSignal

from cache.keys import _jpeg_bytes_max_edge, disk_preview_max_edge
from cache.memory import LRUCache


class ImageCacheThumbGridPreview:
    """Thumbnail, grid, and preview get/put operations."""

    # Type annotations for attributes inherited from ImageCacheCore / ImageCacheFullExifUtil
    stats: dict[str, int]
    thumbnail_cache: LRUCache
    grid_cache: LRUCache
    preview_cache: LRUCache
    libraw_preview_paths: set[str]
    full_image_embedded_jpeg_paths: set[str]
    cache_hit: pyqtSignal
    cache_miss: pyqtSignal
    pixmap_cache: LRUCache
    full_image_cache: LRUCache
    _path_key: Any
    _note_full_cache_reason: Any
    _load_libraw_preview_paths: Any
    _save_libraw_preview_paths: Any
    _orient_verify_lock: threading.Lock
    _orient_verify_thread: threading.Thread | None
    _orient_verify_attempted: set[str]
    _orient_verify_queue: Any
    _orient_verify_queue_max: int
    _orient_verify_attempted_max: int
    _check_memory_pressure: Any
    disk_thumbnail_cache: Any
    disk_grid_cache: Any
    disk_preview_cache: Any
    persistent_cache_enabled: bool
    _enforce_numpy_cache_byte_budget: Any

    def _heal_raw_orientation(self, file_path: str, arr: np.ndarray | None) -> np.ndarray | None:
        """Self-heal: return arr already in EXIF display orientation, re-orienting if not.

        Every cache tier (thumbnail, grid, preview) must return display-oriented pixels so
        the gallery/single-view never render a stale, wrongly-rotated buffer. Uses the same
        make-independent authority as extraction (finalize_index_thumbnail_array), which is
        idempotent, so calling this on already-correct pixels is a no-op.
        """
        if arr is None:
            return None
        try:
            from common_image_loader import (
                finalize_index_thumbnail_array,
                index_thumbnail_needs_orient_fix,
                is_raw_file,
            )

            if not is_raw_file(file_path):
                return arr
            if not index_thumbnail_needs_orient_fix(file_path, arr, cache=self):
                return arr
            repaired = finalize_index_thumbnail_array(file_path, arr, cache=self)
            return repaired if repaired is not None else arr
        except Exception:
            return arr

    def get_thumbnail(self, file_path: str) -> np.ndarray | None:
        """Get cached thumbnail or return None if not cached."""
        self.stats['thumbnail_requests'] += 1
        self._check_memory_pressure()

        key = self._path_key(file_path)

        def _validate_raw_thumbnail(arr: np.ndarray | None) -> np.ndarray | None:
            return self._heal_raw_orientation(file_path, arr)

        # Check in-memory cache first
        thumbnail = self.thumbnail_cache.get(key)
        if thumbnail is not None:
            validated = _validate_raw_thumbnail(thumbnail)
            if validated is not thumbnail:
                self.thumbnail_cache.put(key, validated.copy())  # type: ignore[union-attr]
                self.disk_thumbnail_cache.remove(file_path)
            self.cache_hit.emit(file_path, 'thumbnail')
            return validated

        # Check disk cache for JPEG thumbnails
        jpeg_data = self.disk_thumbnail_cache.get(file_path)
        if jpeg_data is not None:
            try:
                from common_image_loader import decode_embedded_jpeg_bytes

                thumbnail = decode_embedded_jpeg_bytes(jpeg_data, max_size=0)
                if thumbnail is None:
                    raise ValueError("decode_embedded_jpeg_bytes failed")
                thumbnail = _validate_raw_thumbnail(thumbnail)
                if thumbnail is None:
                    raise ValueError("thumbnail orientation validation failed")
                # Also cache in memory for faster subsequent access
                self.thumbnail_cache.put(key, thumbnail.copy())
                self.cache_hit.emit(file_path, 'thumbnail')
                return thumbnail
            except Exception:
                # If loading from disk cache fails, remove it
                self.disk_thumbnail_cache.remove(file_path)

        # --- Dynamic Mipmap Fallback ---
        # 1. Downsample from Grid cache (512px) if available
        grid = self.grid_cache.get(key)
        if grid is None:
            grid_jpeg = self.disk_grid_cache.get(file_path)
            if grid_jpeg is not None:
                try:
                    import io

                    from PIL import Image, ImageOps
                    pil_image = Image.open(io.BytesIO(grid_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    grid = np.array(pil_image)
                except Exception:
                    pass

        if grid is not None:
            try:
                import io

                from PIL import Image
                if grid.dtype != np.uint8:
                    grid = grid.astype(np.uint8)
                pil_img = Image.fromarray(grid)
                w, h = pil_img.size
                scale = min(256 / w, 256 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                thumb_pil = pil_img.resize((new_w, new_h), Image.Resampling.HAMMING)
                thumbnail = np.array(thumb_pil)

                # Compress and store in disk/memory caches
                from common_image_loader import encode_tile_bytes

                t_jpeg = encode_tile_bytes(thumb_pil)
                self.put_thumbnail(file_path, thumbnail, t_jpeg)
                return thumbnail
            except Exception:
                pass

        # 2. Downsample from preview cache (memory or disk, up to ~512px) if available
        preview = self.preview_cache.get(key)
        if preview is None:
            preview_jpeg = self.disk_preview_cache.get(file_path)
            if preview_jpeg is not None:
                try:
                    import io

                    from PIL import Image, ImageOps
                    pil_image = Image.open(io.BytesIO(preview_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    preview = np.array(pil_image)
                except Exception:
                    pass

        if preview is not None:
            try:
                import io

                from PIL import Image
                if preview.dtype != np.uint8:
                    preview = preview.astype(np.uint8)
                pil_img = Image.fromarray(preview)
                w, h = pil_img.size
                scale = min(256 / w, 256 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                thumb_pil = pil_img.resize((new_w, new_h), Image.Resampling.HAMMING)
                thumbnail = np.array(thumb_pil)

                # Compress and store in disk/memory caches
                from common_image_loader import encode_tile_bytes

                t_jpeg = encode_tile_bytes(thumb_pil)
                self.put_thumbnail(file_path, thumbnail, t_jpeg)
                return thumbnail
            except Exception:
                pass

        self.cache_miss.emit(file_path, 'thumbnail')
        return None

    def put_thumbnail(self, file_path: str, thumbnail: np.ndarray, jpeg_data: bytes | None = None) -> None:
        """Cache a thumbnail image."""
        if thumbnail is not None:
            # Ensure thumbnail is reasonable size (max 1024x1024)
            if thumbnail.shape[0] > 1024 or thumbnail.shape[1] > 1024:
                # This should be handled by the caller, but safety check
                return
            key = self._path_key(file_path)
            # Cache in memory
            self.thumbnail_cache.put(key, thumbnail.copy())

            # If JPEG data is provided, also cache to disk
            if jpeg_data is not None:
                self.disk_thumbnail_cache.put(file_path, jpeg_data)

    def get_preview_memory_only(self, file_path: str) -> np.ndarray | None:
        """In-memory preview only -- never decodes disk tier on the calling thread."""
        key = self._path_key(file_path)
        return self.preview_cache.peek(key)

    def get_grid_memory_only(self, file_path: str) -> np.ndarray | None:
        """In-memory grid only -- never decodes disk tier on the calling thread."""
        key = self._path_key(file_path)
        return self.grid_cache.peek(key)

    def get_thumbnail_memory_only(self, file_path: str) -> np.ndarray | None:
        """In-memory thumbnail only -- never decodes disk tier on the calling thread."""
        key = self._path_key(file_path)
        return self.thumbnail_cache.peek(key)

    def has_warm_thumbnail_in_memory(self, file_path: str) -> bool:
        """True if a grid/thumbnail tier is already in the fast in-memory
        cache -- no disk read, no decode, unlike get_grid()/get_thumbnail().

        Those two fall through to a disk-cache read + synchronous PIL decode
        + EXIF transpose on an in-memory miss. Callers that only want to know
        "is this warm, should I skip prefetching it" (filmstrip prefetch scan
        over dozens of neighbor files) were using get_grid()/get_thumbnail()
        for that check and paying the decode cost as a side effect -- on a
        long session, once the small in-memory tier evicts entries the disk
        tier still has, every such "check" becomes a real synchronous decode.
        Measured via faulthandler stack dumps during a 250+ navigation
        stress test: multi-second main-thread stalls with the current frame
        inside PIL's WebP/JPEG decode, called from exactly this "just
        checking" path.
        """
        key = self._path_key(file_path)
        # peek(): a warmth CHECK must not promote never-displayed entries to MRU.
        return (
            self.grid_cache.peek(key) is not None
            or self.thumbnail_cache.peek(key) is not None
        )

    def get_grid(self, file_path: str) -> np.ndarray | None:
        """Get cached grid image (max 512px) or return None if not cached."""
        if 'grid_requests' not in self.stats:
            self.stats['grid_requests'] = 0
        self.stats['grid_requests'] += 1
        self._check_memory_pressure()

        key = self._path_key(file_path)
        # 1. Check in-memory grid cache
        grid = self.grid_cache.get(key)
        if grid is not None:
            healed = self._heal_raw_orientation(file_path, grid)
            if healed is not grid:
                self.grid_cache.put(key, healed.copy())  # type: ignore[union-attr]
                self.disk_grid_cache.remove(file_path)
            self.cache_hit.emit(file_path, 'grid')
            return healed

        # 2. Check disk grid cache
        jpeg_data = self.disk_grid_cache.get(file_path)
        if jpeg_data is not None:
            try:
                import io

                from PIL import Image, ImageOps
                pil_image = Image.open(io.BytesIO(jpeg_data))
                pil_image = ImageOps.exif_transpose(pil_image)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                grid = np.array(pil_image)
                grid = self._heal_raw_orientation(file_path, grid)
                self.grid_cache.put(key, grid.copy())  # type: ignore[union-attr]
                self.cache_hit.emit(file_path, 'grid')
                return grid
            except Exception:
                self.disk_grid_cache.remove(file_path)

        # 3. Dynamic Mipmap Fallback 1: Downsample from preview tier
        preview = self.preview_cache.get(key)
        if preview is None:
            preview_jpeg = self.disk_preview_cache.get(file_path)
            if preview_jpeg is not None:
                try:
                    import io

                    from PIL import Image, ImageOps
                    pil_image = Image.open(io.BytesIO(preview_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    preview = np.array(pil_image)
                except Exception:
                    pass

        if preview is not None:
            try:
                import io

                from PIL import Image
                if preview.dtype != np.uint8:
                    preview = preview.astype(np.uint8)
                pil_img = Image.fromarray(preview)
                w, h = pil_img.size
                scale = min(512 / w, 512 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                grid_pil = pil_img.resize((new_w, new_h), Image.Resampling.HAMMING)
                grid = np.array(grid_pil)

                # Compress and store in disk/memory caches
                from common_image_loader import encode_tile_bytes

                g_jpeg = encode_tile_bytes(grid_pil)
                self.put_grid(file_path, grid, g_jpeg)
                return grid
            except Exception:
                pass

        # 4. Dynamic Mipmap Fallback 2: Upsample from Thumbnail (256px)
        thumb = self.thumbnail_cache.get(key)
        if thumb is None:
            thumb_jpeg = self.disk_thumbnail_cache.get(file_path)
            if thumb_jpeg is not None:
                try:
                    import io

                    from PIL import Image, ImageOps
                    pil_image = Image.open(io.BytesIO(thumb_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    thumb = np.array(pil_image)
                except Exception:
                    pass

        if thumb is not None:
            try:
                from PIL import Image
                if thumb.dtype != np.uint8:
                    thumb = thumb.astype(np.uint8)
                pil_img = Image.fromarray(thumb)
                w, h = pil_img.size
                scale = min(512 / w, 512 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                grid_pil = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
                grid = np.array(grid_pil)
                self.grid_cache.put(key, grid.copy())
                return grid
            except Exception:
                pass

        self.cache_miss.emit(file_path, 'grid')
        return None

    def put_grid(self, file_path: str, grid: np.ndarray, jpeg_data: bytes | None = None) -> None:
        """Cache a grid image (max 512px)."""
        if grid is not None:
            key = self._path_key(file_path)
            # Ensure grid image is reasonable size (max 512x512)
            if grid.shape[0] > 512 or grid.shape[1] > 512:
                return
            # Cache in memory
            self.grid_cache.put(key, grid.copy())

            # Cache on disk
            if self.persistent_cache_enabled:
                if jpeg_data is not None:
                    self.disk_grid_cache.put(file_path, jpeg_data)
                else:
                    try:
                        from PIL import Image

                        from common_image_loader import encode_tile_bytes

                        if grid.dtype != np.uint8:
                            grid = grid.astype(np.uint8)
                        pil_image = Image.fromarray(grid)
                        self.disk_grid_cache.put(file_path, encode_tile_bytes(pil_image))
                    except Exception:
                        pass

    def get_preview(self, file_path: str) -> np.ndarray | None:
        """Get cached preview (screen size) or return None."""
        self.stats['preview_requests'] += 1
        self._check_memory_pressure()

        key = self._path_key(file_path)
        # Check in-memory cache first
        preview = self.preview_cache.get(key)
        if preview is not None:
            self.cache_hit.emit(file_path, 'preview')
            return preview  # type: ignore[no-any-return]

        # Check disk cache for previews
        jpeg_data = self.disk_preview_cache.get(file_path)
        if jpeg_data is None:
            jpeg_data = self.disk_grid_cache.get(file_path)
        if jpeg_data is not None:
             try:
                # cv2.imdecode over PIL: several times faster on the
                # 1280-1536px entries here, and this path runs on the
                # CALLING thread -- navigation paints call get_preview()
                # from the UI thread, so every ms is a paint stall. EXIF
                # transpose is intentionally absent: disk preview entries
                # are re-encoded from already-oriented pixels with EXIF
                # stripped (_jpeg_bytes_max_edge), and grid tiles are
                # encoded from oriented arrays.
                import cv2

                arr = cv2.imdecode(
                    np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
                )
                if arr is None:
                    raise ValueError("imdecode failed")
                preview = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                # Also cache in memory
                self.preview_cache.put(key, preview)
                self.cache_hit.emit(file_path, 'preview')
                return preview  # type: ignore[no-any-return]
             except Exception:
                self.disk_preview_cache.remove(file_path)

        self.cache_miss.emit(file_path, 'preview')
        return None

    def put_preview(
        self,
        file_path: str,
        preview: np.ndarray,
        jpeg_data: bytes | None = None,
        *,
        libraw_rendered: bool = False,
        copy: bool = True,
    ) -> None:
        """Cache a preview image (memory); optional disk JPEG clamped to disk_preview_max_edge().

        ``libraw_rendered`` marks the preview as LibRaw-pipeline pixels (vs
        embedded-JPEG-derived) so the RAW workflow can refuse to paint
        camera-JPEG interim tiers (session-scoped provenance).

        ``copy=False`` transfers ownership of a freshly allocated buffer the
        caller will not mutate, matching ``put_full_image``.
        """
        if preview is not None:
            key = self._path_key(file_path)
            stored = (
                np.ascontiguousarray(preview) if not copy else preview.copy()
            )
            self.preview_cache.put(key, stored)
            if libraw_rendered:
                self.libraw_preview_paths.add(key)
                self._prune_libraw_preview_paths()
                self._save_libraw_preview_paths()
            else:
                if key in self.libraw_preview_paths:
                    self.libraw_preview_paths.discard(key)
                    self._save_libraw_preview_paths()
            if jpeg_data is not None:
                cap = disk_preview_max_edge()
                jpeg_data = _jpeg_bytes_max_edge(jpeg_data, cap)
                self.disk_preview_cache.put(file_path, jpeg_data)
            self._enforce_numpy_cache_byte_budget(
                self.preview_cache,
                fraction=0.25,
                min_keep=2,
                label="preview",
            )

    def is_libraw_preview(self, file_path: str) -> bool:
        """True when the cached preview holds LibRaw-rendered pixels (this session)."""
        return self._path_key(file_path) in self.libraw_preview_paths

    def _prune_libraw_preview_paths(self, max_paths: int = 2048) -> None:
        """Bound session provenance set (paths only)."""
        n = len(self.libraw_preview_paths)
        if n <= max_paths:
            return
        # Discard arbitrary surplus (set has no order); prefer keeping keys still in preview.
        excess = n - max_paths
        drop = []
        for k in self.libraw_preview_paths:
            if self.preview_cache.peek(k) is None:
                drop.append(k)
            if len(drop) >= excess:
                break
        for k in drop:
            self.libraw_preview_paths.discard(k)
        while len(self.libraw_preview_paths) > max_paths:
            self.libraw_preview_paths.pop()
