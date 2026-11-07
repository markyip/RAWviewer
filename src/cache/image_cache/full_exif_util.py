"""Full image, EXIF, pixmap, orientation verify, and utility operations."""
from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPixmap

from cache.disk import (
    PersistentGridCache,
    PersistentPreviewCache,
    PersistentThumbnailCache,
)
from cache.keys import (
    _env_int,
)
from cache.memory import LRUCache, MemoryMonitor, MemoryOnlyPersistentCache
from cache.metadata import PersistentEXIFCache

logger = logging.getLogger(__name__)

class ImageCacheFullExifUtil:
    """Full image, EXIF, pixmap, orientation verify, and utility operations."""

    # Type annotations for attributes inherited from ImageCacheCore / ImageCacheThumbGridPreview
    # (keep these in sync with the sibling parts so mypy can track cross-part usage)
    _full_cache_reasons: OrderedDict[str, str]
    max_memory_mb: int
    _tier_max_memory_mb: int
    _check_memory_pressure: Any
    _path_key: Any
    _prune_libraw_preview_paths: Any
    stats: dict[str, int]
    full_image_cache: LRUCache
    cache_hit: pyqtSignal
    cache_miss: pyqtSignal
    memory_warning: pyqtSignal
    memory_monitor: MemoryMonitor
    persistent_cache_enabled: bool
    cache_dir: str | None
    thumbnail_cache: LRUCache
    grid_cache: LRUCache
    preview_cache: LRUCache
    pixmap_cache: LRUCache
    libraw_preview_paths: set[str]
    _libraw_preview_paths_file: str | None
    full_image_embedded_jpeg_paths: set[str]
    exif_cache: PersistentEXIFCache | MemoryOnlyPersistentCache
    disk_thumbnail_cache: PersistentThumbnailCache | MemoryOnlyPersistentCache
    disk_grid_cache: PersistentGridCache | MemoryOnlyPersistentCache
    disk_preview_cache: PersistentPreviewCache | MemoryOnlyPersistentCache
    exif_memory_cache: LRUCache
    _orient_verify_lock: threading.Lock
    _orient_verify_thread: threading.Thread | None
    _orient_verify_attempted: set[str]
    _orient_verify_queue: Any  # queue.Queue
    _orient_verify_queue_max: int
    _orient_verify_attempted_max: int

    def _note_full_cache_reason(self, file_path: str, reason: str) -> None:
        """Record why a full image left the cache (diagnostics only)."""
        try:
            self._full_cache_reasons[file_path] = reason
            self._full_cache_reasons.move_to_end(file_path)
            while len(self._full_cache_reasons) > 256:
                self._full_cache_reasons.popitem(last=False)
        except Exception:
            pass

    def get_full_image(self, file_path: str) -> np.ndarray | None:
        """Get cached full image or return None if not cached."""
        self.stats['full_image_requests'] += 1
        self._check_memory_pressure()

        image = self.full_image_cache.get(file_path)
        if image is not None:
            self.cache_hit.emit(file_path, 'full_image')
            return image  # type: ignore[no-any-return]
        else:
            self.cache_miss.emit(file_path, 'full_image')
            if os.environ.get("RAWVIEWER_DEBUG_FULLCACHE"):
                import logging

                logging.getLogger(__name__).info(
                    "[FULLCACHE] miss %s -- %s (cache holds %d/%d)",
                    os.path.basename(file_path or ""),
                    self._full_cache_reasons.get(file_path, "never cached this session"),
                    len(self.full_image_cache.cache),
                    self.full_image_cache.max_size,
                )
            return None

    def put_full_image(
        self, file_path: str, image: np.ndarray, *, copy: bool = True
    ) -> None:
        """Cache a full processed image.

        ``copy=False`` transfers ownership of a freshly allocated buffer the
        caller will not mutate (avoids a second multi-hundred-MB memcpy on
        40–60MP decodes). Contiguous arrays are stored as-is; non-contiguous
        inputs are still compacted with ``ascontiguousarray``.
        """
        if image is not None:
            stored = (
                np.ascontiguousarray(image) if not copy else image.copy()
            )
            self.full_image_cache.put(file_path, stored)

    def mark_full_image_source(self, file_path: str, *, is_embedded_jpeg: bool) -> None:
        """Track whether the cached full_image entry is an embedded-JPEG
        stand-in rather than the app's own RAW-decoded pixels -- see
        full_image_embedded_jpeg_paths. Call alongside every put_full_image()
        so the flag never goes stale: mark True from the embedded-preview
        shortcut, False once a true decode overwrites that same key.
        """
        if not file_path:
            return
        key = self._path_key(file_path)
        if is_embedded_jpeg:
            self.full_image_embedded_jpeg_paths.add(key)
            self._prune_full_image_embedded_jpeg_paths()
        else:
            self.full_image_embedded_jpeg_paths.discard(key)

    def full_image_is_embedded_jpeg(self, file_path: str) -> bool:
        """True when the currently cached full_image for this path is an
        embedded-JPEG stand-in (see mark_full_image_source)."""
        if not file_path:
            return False
        return self._path_key(file_path) in self.full_image_embedded_jpeg_paths

    def _prune_full_image_embedded_jpeg_paths(self, max_paths: int = 2048) -> None:
        """Bound session provenance set (paths only); same shape as
        _prune_libraw_preview_paths."""
        n = len(self.full_image_embedded_jpeg_paths)
        if n <= max_paths:
            return
        excess = n - max_paths
        drop = []
        for k in self.full_image_embedded_jpeg_paths:
            if self.full_image_cache.peek(k) is None:
                drop.append(k)
            if len(drop) >= excess:
                break
        for k in drop:
            self.full_image_embedded_jpeg_paths.discard(k)
        while len(self.full_image_embedded_jpeg_paths) > max_paths:
            self.full_image_embedded_jpeg_paths.pop()

    def _estimate_entry_bytes(self, value: Any) -> int:
        try:
            if hasattr(value, "nbytes"):
                return int(value.nbytes)
            # QPixmap / QImage
            if hasattr(value, "width") and hasattr(value, "height") and callable(value.width):
                w, h = int(value.width()), int(value.height())
                # Assume up to 4 bytes/pixel (RGBA) for budget purposes.
                return max(0, w * h * 4)
        except Exception:
            pass
        return 0

    def _enforce_numpy_cache_byte_budget(
        self,
        lru: LRUCache,
        *,
        fraction: float,
        min_keep: int = 2,
        label: str = "cache",
    ) -> None:
        """Evict LRU entries until sum(nbytes) fits fraction of max_memory_mb."""
        try:
            budget_bytes = float(self.max_memory_mb) * 1024 * 1024 * float(fraction)
            cache = lru.cache
            with lru.lock:
                total = sum(self._estimate_entry_bytes(v) for v in cache.values())
                if total <= budget_bytes:
                    return
                evicted = 0
                while total > budget_bytes and len(cache) > max(1, min_keep):
                    evicted_key, oldest = cache.popitem(last=False)
                    total -= self._estimate_entry_bytes(oldest)
                    evicted += 1
                    if lru.on_evict is not None:
                        with contextlib.suppress(Exception):
                            lru.on_evict(evicted_key)
            if evicted:
                logger.info(
                    "%s cache: evicted %d oldest (budget ~%.0fMB, now ~%.0fMB, %d left)",
                    label,
                    evicted,
                    budget_bytes / 1e6,
                    total / 1e6,
                    len(cache),
                )
        except Exception:
            logger.warning("%s byte-budget enforcement failed", label, exc_info=True)

    def apply_pressure_budget_scale(self, scale: float, *, min_mb: int = 128) -> int:
        """Shrink the full-image byte budget for the rest of the session.

        Never raises the budget (session-sticky). Installed-RAM tier is unchanged.
        """
        original = int(getattr(self, "_tier_max_memory_mb", self.max_memory_mb) or self.max_memory_mb)
        self._tier_max_memory_mb = original
        new_mb = max(int(min_mb), int(original * float(scale)))
        self.max_memory_mb = min(int(self.max_memory_mb), new_mb)
        try:
            new_full = max(2, int(self.full_image_cache.max_size * float(scale)))
            self.full_image_cache.shrink_to_size(min(self.full_image_cache.max_size, new_full))
        except Exception:
            pass
        return int(self.max_memory_mb)

    def get_pixmap(self, file_path: str) -> QPixmap | None:
        """Get cached QPixmap or return None if not cached."""
        self.stats['pixmap_requests'] += 1

        pixmap = self.pixmap_cache.get(file_path)
        if pixmap is not None:
            self.cache_hit.emit(file_path, 'pixmap')
            return pixmap
        else:
            self.cache_miss.emit(file_path, 'pixmap')
            return None

    def put_pixmap(self, file_path: str, pixmap: QPixmap) -> None:
        """Cache a QPixmap."""
        if pixmap is not None and not pixmap.isNull():
            self.pixmap_cache.put(file_path, pixmap)
            self._enforce_numpy_cache_byte_budget(
                self.pixmap_cache,
                fraction=0.20,
                min_keep=2,
                label="pixmap",
            )

    def clear_heavy_memory_tiers(self, keep_path: str | None = None) -> None:
        """Drop full/preview/pixmap RAM tiers (keep grid/thumb for gallery scroll).

        Call on folder change. Optional keep_path retains that file's heavy entries
        when still navigating within open flow.
        """
        keep_key = self._path_key(keep_path) if keep_path else None
        keep_full = None
        keep_preview = None
        keep_pixmap = None
        keep_full_is_embedded_jpeg = False
        if keep_path:
            keep_full = self.full_image_cache.get(keep_path)
            keep_preview = self.preview_cache.get(keep_key)
            keep_pixmap = self.pixmap_cache.get(keep_path)
            keep_full_is_embedded_jpeg = self.full_image_is_embedded_jpeg(keep_path)

        self.full_image_cache.clear()
        self.preview_cache.clear()
        self.pixmap_cache.clear()
        self.libraw_preview_paths.clear()
        self.full_image_embedded_jpeg_paths.clear()

        if keep_path and keep_full is not None:
            self.full_image_cache.put(keep_path, keep_full)
            # Preserve provenance across the clear -- otherwise a kept
            # embedded-JPEG stand-in would look like a true decode afterward
            # and never get upgraded to the real RAW pixels.
            self.mark_full_image_source(keep_path, is_embedded_jpeg=keep_full_is_embedded_jpeg)
        if keep_path and keep_preview is not None and keep_key:
            self.preview_cache.put(keep_key, keep_preview)
        if keep_path and keep_pixmap is not None:
            self.pixmap_cache.put(keep_path, keep_pixmap)

    def on_folder_changed(self, keep_path: str | None = None) -> None:
        """Folder scope changed: free heavy RAM; leave disk thumbs for warm reopen."""
        self.clear_heavy_memory_tiers(keep_path=keep_path)
        try:
            from processing.unified import prune_libraw_unsupported_paths

            prune_libraw_unsupported_paths(clear_all=False, max_keep=512)
        except Exception:
            pass

    def get_exif_memory_only(self, file_path: str) -> dict[str, Any] | None:
        """In-memory EXIF only (no SQLite) — for cache-hit emit during preview-first."""
        return self.exif_memory_cache.get(file_path)

    def get_exif_for_ui(self, file_path: str) -> dict[str, Any] | None:
        """UI hot path: never blocks (see get_exif's verify=False default)."""
        return self.get_exif(file_path)

    def get_exif(
        self, file_path: str, *, verify: bool = False
    ) -> dict[str, Any] | None:
        """Get cached EXIF data or return None if not cached.

        RAW orientation trustworthiness: records lacking ``verified_orientation``
        used to be re-checked synchronously on EVERY call via
        cached_raw_exif_orientation_trustworthy() -> rawpy.imread() -- a real
        file open that stalled whatever thread called this (including the Qt
        main thread: status bar, HUD, prefetch decisions), and whose
        successful result was never memoized, so the same file re-paid the
        probe on every subsequent call too.

        Now: ``verify=False`` (default) returns the record optimistically and
        schedules the probe on a background thread; every pixel-rotation path
        re-derives orientation at decode time, so a briefly-unverified record
        can't rotate pixels wrongly, and the background result either stamps
        verified_orientation=True or evicts the bad record so the next fetch
        re-extracts fresh. ``verify=True`` (decode paths that gate orientation
        decisions on this record, e.g. image_load_manager's deferred-EXIF
        emits) keeps the synchronous probe but now memoizes success.
        """
        self.stats['exif_requests'] += 1

        # 1. Check in-memory cache first
        exif_data = self.exif_memory_cache.get(file_path)
        if exif_data is None:
            # 2. Check persistent cache
            exif_data = self.exif_cache.get(file_path)
            if exif_data is not None:
                self.exif_memory_cache.put(file_path, exif_data)

        if exif_data is None:
            self.cache_miss.emit(file_path, 'exif')
            return None

        try:
            from enhanced_raw_processor import (
                RAW_EXIF_SENSOR_META_VER,
                cached_raw_exif_orientation_trustworthy,
            )
            cached_ver = exif_data.get('raw_exif_sensor_meta_ver', 0)
            if cached_ver < RAW_EXIF_SENSOR_META_VER:
                self.cache_miss.emit(file_path, 'exif')
                return None
            if not exif_data.get("verified_orientation"):
                if verify:
                    if not cached_raw_exif_orientation_trustworthy(
                        file_path, exif_data
                    ):
                        self.cache_miss.emit(file_path, 'exif')
                        return None
                    # Memoize the successful probe so no caller ever re-pays it.
                    exif_data = dict(exif_data)
                    exif_data["verified_orientation"] = True
                    self.put_exif(file_path, exif_data)
                else:
                    self._schedule_orientation_verify(file_path, exif_data)
        except Exception:
            pass

        self.cache_hit.emit(file_path, 'exif')
        return exif_data

    def _schedule_orientation_verify(
        self, file_path: str, exif_data: dict[str, Any]
    ) -> None:
        """Queue a background trustworthiness probe for an unverified RAW record.

        One attempt per path per session; a single lazily-started daemon
        thread drains the queue (probes serialize on the rawpy global lock
        anyway, so more workers would only pile up blocked threads).
        """
        try:
            import common_image_loader

            if not common_image_loader.is_raw_file(file_path):
                return
        except Exception:
            return
        with self._orient_verify_lock:
            if getattr(self, "_closing", False):
                return
            if file_path in self._orient_verify_attempted:
                return
            if len(self._orient_verify_queue) >= self._orient_verify_queue_max:
                return
            if len(self._orient_verify_attempted) >= self._orient_verify_attempted_max:
                self._orient_verify_attempted.clear()
            self._orient_verify_attempted.add(file_path)
            self._orient_verify_queue.append(file_path)
            if (
                self._orient_verify_thread is None
                or not self._orient_verify_thread.is_alive()
            ):
                self._orient_verify_thread = threading.Thread(
                    target=self._orientation_verify_worker,
                    name="exif-orient-verify",
                    daemon=True,
                )
                self._orient_verify_thread.start()

    def _orientation_verify_worker(self) -> None:
        while True:
            with self._orient_verify_lock:
                if not self._orient_verify_queue or getattr(self, "_closing", False):
                    self._orient_verify_thread = None
                    return
                path = self._orient_verify_queue.pop(0)
            try:
                from enhanced_raw_processor import (
                    cached_raw_exif_orientation_trustworthy,
                )

                cached = self.exif_memory_cache.peek(path)
                if not cached:
                    getter = getattr(self.exif_cache, "get", None)
                    cached = getter(path) if callable(getter) else None
                if not cached:
                    continue
                cached = dict(cached)
                if cached_raw_exif_orientation_trustworthy(path, cached):
                    cached["verified_orientation"] = True
                    if not getattr(self, "_closing", False):
                        self.put_exif(path, cached)
                else:
                    # Cached orientation is wrong: drop the record so the next
                    # fetch re-extracts fresh (extraction writes a verified one).
                    if not getattr(self, "_closing", False):
                        self.exif_memory_cache.remove(path)
                    with contextlib.suppress(Exception):
                        self.exif_cache.remove(path)
                    self.cache_miss.emit(path, 'exif')
            except Exception:
                pass

    def put_exif(
        self,
        file_path: str,
        exif_data: dict[str, Any],
        *,
        file_size: int | None = None,
        file_mtime: float | None = None,
        persist_disk: bool = True,
    ) -> None:
        """Cache EXIF data (memory always; disk optional for preview-first minimal stubs)."""
        if exif_data:
            self.exif_memory_cache.put(file_path, exif_data)
            if persist_disk:
                self.exif_cache.put(
                    file_path,
                    exif_data,
                    file_size=file_size,
                    file_mtime=file_mtime,
                )

    def flush_exif_disk_cache(self) -> None:
        """Commit batched EXIF disk writes (call after indexing batches)."""
        if hasattr(self.exif_cache, "flush"):
            self.exif_cache.flush()

    def get_capture_times_for_sort(self, file_paths: list[str]) -> dict[str, str]:
        """capture_time only, for folder sort (ignores stale mtime on network folders)."""
        return self.exif_cache.get_capture_times_bulk(file_paths)

    def record_probed_capture_times(self, timestamps: dict[str, float]) -> None:
        """Persist sort-probe capture times so the next folder open skips re-probing.

        timestamps values are epoch floats (from probe_capture_timestamp_from_file);
        stored as ISO strings to match the capture_time column's existing format.
        """
        if not timestamps or not hasattr(self.exif_cache, "update_capture_time_bulk"):
            return
        import datetime

        iso_times: dict[str, str] = {}
        for path, ts in timestamps.items():
            if not ts or ts <= 0:
                continue
            try:
                iso_times[path] = datetime.datetime.fromtimestamp(ts).strftime(
                    "%Y:%m:%d %H:%M:%S"
                )
            except Exception:
                continue
        self.exif_cache.update_capture_time_bulk(iso_times)

    def get_capture_times_for_folder_sort(self, folder_path: str) -> dict[str, str]:
        """All capture_time rows under folder_path (normalized path keys)."""
        return self.exif_cache.get_capture_times_for_folder(folder_path)

    def get_multiple_exif(self, file_paths: list[str], file_stats: dict[str, tuple[int, float]] | None = None, fast_mode: bool = True) -> dict[str, dict[str, Any]]:
        """Get cached EXIF data for multiple files at once.

        Orientation trust for unverified RAW records is optimistic + deferred
        to a background probe, same as get_exif's verify=False default --
        NOT the old synchronous cached_raw_exif_orientation_trustworthy() ->
        rawpy.imread() call per file. That call, run for every file in a
        folder on the caller's thread (e.g. the main-thread instant-sort
        fast path in main.py's _on_quick_folder_index_ready), turned a large
        folder's sort into a serial rawpy-open storm that froze the UI for
        the whole scan -- get_exif itself was already fixed for this; this
        bulk sibling had the same synchronous check duplicated inline and
        was missed.
        """
        self.stats['exif_requests'] += len(file_paths)

        results = {}
        missing_paths = []

        try:
            from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER
            has_ver = True
        except Exception:
            has_ver = False
            RAW_EXIF_SENSOR_META_VER = 0

        # Check memory cache first
        for path in file_paths:
            exif = self.exif_memory_cache.get(path)
            if exif:
                if has_ver:
                    cached_ver = exif.get('raw_exif_sensor_meta_ver', 0)
                    if cached_ver < RAW_EXIF_SENSOR_META_VER:
                        missing_paths.append(path)
                        continue
                    if not exif.get('verified_orientation'):
                        self._schedule_orientation_verify(path, exif)
                results[path] = exif
            else:
                missing_paths.append(path)

        if missing_paths:
            # Fetch missing from persistent cache
            db_results = self.exif_cache.get_multiple(missing_paths, file_stats, fast_mode)
            for path, exif in db_results.items():
                if has_ver:
                    cached_ver = exif.get('raw_exif_sensor_meta_ver', 0)
                    if cached_ver < RAW_EXIF_SENSOR_META_VER:
                        continue
                    if not exif.get('verified_orientation'):
                        self._schedule_orientation_verify(path, exif)
                self.exif_memory_cache.put(path, exif)
                results[path] = exif

        return results

    def release_full_image_memory(self, file_path: str | None) -> None:
        """Drop in-memory full-res / pixmap tiers; keep preview/grid/thumbnail on disk."""
        if not file_path:
            return
        if self.full_image_cache.remove(file_path):
            self._note_full_cache_reason(file_path, "invalidated (invalidate_file)")
        self.pixmap_cache.remove(file_path)

    def invalidate_file(self, file_path: str) -> None:
        """Remove all cached data for a specific file."""
        key = self._path_key(file_path)
        self.thumbnail_cache.remove(key)
        self.grid_cache.remove(key)
        self.preview_cache.remove(key)
        if self.full_image_cache.remove(file_path):
            self._note_full_cache_reason(file_path, "invalidated (invalidate_file)")
        self.pixmap_cache.remove(file_path)
        self.disk_thumbnail_cache.remove(file_path)
        self.disk_grid_cache.remove(file_path)
        self.disk_preview_cache.remove(file_path)
        self.exif_memory_cache.remove(file_path)
        self.exif_cache.remove(file_path)
        self.full_image_embedded_jpeg_paths.discard(key)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.thumbnail_cache.clear()
        self.grid_cache.clear()
        self.preview_cache.clear()
        for _k in list(self.full_image_cache.cache.keys()):
            self._note_full_cache_reason(_k, "dropped by memory-pressure purge")
        self.full_image_cache.clear()
        self.pixmap_cache.clear()
        # The edit-base LRU lives in processing.unified and is the one
        # heavy cache this purge could not previously reach, so a
        # critical-pressure purge freed everything except a couple of
        # hundred MB of float32 bases.
        try:
            from processing.unified import (
                release_adjusted_full_cache,
                release_edit_base_cache,
                release_unpacked_raw_cache,
            )

            # All three live in processing.unified at module scope so they
            # survive the one-shot workers that build them. That is what makes
            # them useful, and also what puts them outside every cache this
            # class owns -- a critical-pressure purge used to clear everything
            # it could see and leave these whole.
            freed = (
                release_edit_base_cache()
                + release_adjusted_full_cache()
                + release_unpacked_raw_cache()
            )
            if freed:
                import logging

                logging.getLogger(__name__).info(
                    "[MEMORY] released %.0f MB of decode/edit buffers", freed / 1e6
                )
        except Exception:
            pass
        self.libraw_preview_paths.clear()
        if self._libraw_preview_paths_file and os.path.isfile(self._libraw_preview_paths_file):
            with contextlib.suppress(Exception):
                os.remove(self._libraw_preview_paths_file)
        self.full_image_embedded_jpeg_paths.clear()
        self.exif_cache.clear()
        self.exif_memory_cache.clear()
        self.disk_thumbnail_cache.clear()
        self.disk_grid_cache.clear()
        self.disk_preview_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_info = self.memory_monitor.get_memory_info()

        return {
            'thumbnail_cache': self.thumbnail_cache.get_stats(),
            'grid_cache': self.grid_cache.get_stats(),
            'full_image_cache': self.full_image_cache.get_stats(),
            'pixmap_cache': self.pixmap_cache.get_stats(),
            'memory_info': memory_info,
            'request_stats': self.stats.copy(),
            'cache_budget_mb': self.max_memory_mb,
            'persistent_cache_enabled': self.persistent_cache_enabled
        }

    def cleanup_old_cache(self) -> None:
        """Clean up old cache entries."""
        self.exif_cache.cleanup_old_entries()
        self.disk_thumbnail_cache.cleanup_old_entries()
        self.disk_grid_cache.cleanup_old_entries()
        self.disk_preview_cache.cleanup_old_entries()

    def enforce_disk_cache_budget(self, max_mb: int | None = None) -> dict[str, float]:
        """Evict oldest-cached thumbnail/grid/preview disk entries when total usage
        exceeds a budget (RAWVIEWER_DISK_CACHE_MAX_MB, default 4096 MB).

        Age-based cleanup_old_cache() only removes entries older than 30 days --
        nothing previously capped total disk usage, so a long-lived install could
        grow unbounded across every folder ever browsed. This is stat-based (no
        size column is tracked in the DB) and evicts down to 90% of budget
        (hysteresis) so it doesn't re-trigger on every startup right at the
        boundary -- call from a background/startup pass only, never a hot path.
        """
        import logging

        logger = logging.getLogger(__name__)
        if max_mb is None:
            max_mb = _env_int(
                "RAWVIEWER_DISK_CACHE_MAX_MB", 4096, minimum=256, maximum=1024 * 1024
            )
        max_bytes = max_mb * 1024 * 1024
        target_bytes = int(max_bytes * 0.9)

        tiers = {
            "thumbnail": self.disk_thumbnail_cache,
            "grid": self.disk_grid_cache,
            "preview": self.disk_preview_cache,
        }
        usage: dict[str, int] = {}
        for name, cache in tiers.items():
            try:
                usage[name] = cache.get_disk_usage_bytes()
            except Exception:
                usage[name] = 0

        total = sum(usage.values())
        usage_mb = {f"{k}_mb": v / (1024 * 1024) for k, v in usage.items()}
        if total <= max_bytes:
            return usage_mb

        to_free = total - target_bytes
        freed_total = 0
        for name, cache in tiers.items():
            tier_usage = usage.get(name, 0)
            if tier_usage <= 0 or total <= 0:
                continue
            # Evict proportionally to this tier's share of total usage.
            share = int(to_free * (tier_usage / total))
            with contextlib.suppress(Exception):
                freed_total += cache.evict_lru(share)

        logger.info(
            "[CACHE] Disk cache budget exceeded (%.1f MB > %d MB); evicted %.1f MB "
            "(thumbnail=%.1fMB grid=%.1fMB preview=%.1fMB before eviction)",
            total / (1024 * 1024),
            max_mb,
            freed_total / (1024 * 1024),
            usage_mb.get("thumbnail_mb", 0.0),
            usage_mb.get("grid_mb", 0.0),
            usage_mb.get("preview_mb", 0.0),
        )
        return usage_mb

    def _load_libraw_preview_paths(self) -> None:
        try:
            if self._libraw_preview_paths_file and os.path.isfile(self._libraw_preview_paths_file):
                import json
                with open(self._libraw_preview_paths_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                self.libraw_preview_paths = set(loaded) if isinstance(loaded, list) else set()
                self._prune_libraw_preview_paths()
        except Exception:
            self.libraw_preview_paths = set()

    def _save_libraw_preview_paths(self) -> None:
        try:
            if self._libraw_preview_paths_file:
                import json
                with open(self._libraw_preview_paths_file, "w", encoding="utf-8") as f:
                    json.dump(list(self.libraw_preview_paths), f)
        except Exception:
            pass

    def close(self) -> None:
        """Close all persistent cache connections."""
        self._closing = True
        self._save_libraw_preview_paths()

        with self._orient_verify_lock:
            self._orient_verify_queue.clear()
        if self._orient_verify_thread is not None and self._orient_verify_thread.is_alive():
            self._orient_verify_thread.join(timeout=2.0)

        if hasattr(self, 'exif_cache'):
            self.exif_cache.close()
        if hasattr(self, 'disk_thumbnail_cache'):
            self.disk_thumbnail_cache.close()
        if hasattr(self, 'disk_grid_cache'):
            self.disk_grid_cache.close()
        if hasattr(self, 'disk_preview_cache'):
            self.disk_preview_cache.close()
