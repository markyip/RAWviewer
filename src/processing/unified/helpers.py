"""Module-level helpers and state for the UnifiedImageProcessor package."""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import numpy as np
from PyQt6.QtCore import QLoggingCategory

# Re-exports from sibling and external packages
from cache.keys import disk_preview_max_edge, memory_preview_max_edge  # noqa: F401
from common_image_loader import (  # noqa: F401
    array_matches_exif_display,
    image_covers_sensor_resolution,
    is_raw_file,
    load_pixmap_safe,
    use_full_embedded_raw_preview,
    use_libraw_consistent_preview_first,
)
from enhanced_raw_processor import (  # noqa: F401
    EXIFExtractor,
    OptimizedRAWProcessor,
    ThumbnailExtractor,
)
from image_cache import get_image_cache  # noqa: F401
from processing.raw_decode.gil import rawpy_imread_warm

# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats*=false")

import logging

logging.getLogger("exifread").setLevel(logging.ERROR)
# PIL Image, rawpy, and exifread will be imported lazily to avoid import delays



def _verbose_orientation_logs() -> bool:
    return os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# LibRaw cannot open some composite HDR DNGs; avoid repeated preview/decode attempts.
_LIBRAW_UNSUPPORTED_PATHS: set[str] = set()

# In-flight decode_raw_edit_base registry (module-level: workers build their
# own UnifiedImageProcessor instances, so instance state cannot dedup them).
import contextlib
import threading as _ebt  # noqa: E402

_EDIT_BASE_INFLIGHT: dict = {}
_EDIT_BASE_INFLIGHT_LOCK = _ebt.Lock()
# Half-size float32 edit-base LRU (reopen Adjust / revisit without re-unpack).
# Full-res export bases are NOT cached (too large). Cross-platform: keyed with
# normcase(abspath) + mtime_ns/size so Windows path case and file replaces work.
_EDIT_BASE_LRU: OrderedDict = OrderedDict()
_EDIT_BASE_LRU_LOCK = _ebt.Lock()
_EDIT_BASE_LRU_MAX = 2


def _edit_base_max_edge() -> int:
    """Optional long-edge cap for the working (non-export) edit base.

    Unset = LibRaw half-size only. RAM-tier overlay sets this on slower
    machines so settle / Auto Tone do not keep a 12 MP uint16/float working
    copy. Export still uses ``use_full_resolution=True`` and is never capped
    here.
    """
    raw = os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE", "").strip()
    if not raw:
        return 0
    try:
        return max(640, min(int(raw), 8000))
    except ValueError:
        return 0


def _edit_base_lru_max() -> int:
    raw = os.environ.get("RAWVIEWER_EDIT_BASE_LRU_MAX", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 4))
        except ValueError:
            pass
    return _EDIT_BASE_LRU_MAX


def cap_working_edit_base(img, *, use_full_resolution: bool = False):
    """Downsample a working edit base to ``RAWVIEWER_EDIT_BASE_MAX_EDGE``.

    Full-resolution export bases are returned unchanged.
    """
    if img is None or use_full_resolution or not hasattr(img, "shape"):
        return img
    max_edge = _edit_base_max_edge()
    if max_edge <= 0:
        return img
    try:
        h, w = int(img.shape[0]), int(img.shape[1])
        m = max(h, w)
        if m <= max_edge:
            return img
        import cv2

        scale = max_edge / float(m)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        return img


_LIBRAW_UNSUPPORTED_MAX = 2048


def _on_gui_thread() -> bool:
    """True when running on Qt's GUI thread.

    QPixmap, and the pixmap cache the GUI thread reads, are only safe there.
    Returns True if the question cannot be answered (no QApplication yet), so
    single-process scripts and tests behave as before.
    """
    try:
        from PyQt6.QtCore import QThread
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return True
        return QThread.currentThread() is app.thread()
    except Exception:
        return True


# Full-res sidecar-applied buffers, shared across processor instances -- with
# the lock and the in-flight map that go WITH them. Hoisting the list alone
# left concurrent workers mutating one shared list under different locks (lost
# updates), and gave each worker its own empty in-flight map, so the
# coalescing that exists to stop two workers both running a ~2s full-res apply
# for the same file never once coalesced. The state and its guard have to move
# together.
_ADJUSTED_FULL_SLOTS: list = []
_ADJUSTED_FULL_LOCK = _ebt.Lock()
_ADJUSTED_INFLIGHT: dict = {}


def release_adjusted_full_cache() -> int:
    """Drop the full-res adjusted memo. Returns approximate bytes freed.

    Up to six sensor-resolution buffers; at 45MP that is well over a
    gigabyte, and it was invisible to the memory-pressure purge -- which
    cleared every other heavy cache and left this one whole.
    """
    freed = 0
    with _ADJUSTED_FULL_LOCK:
        for entry in _ADJUSTED_FULL_SLOTS:
            with contextlib.suppress(Exception):
                freed += int(getattr(entry[2], "nbytes", 0) or 0)
        _ADJUSTED_FULL_SLOTS.clear()
    return freed


# Unpacked-RAW stash. Module-level for the same reason the in-flight registry
# above is: workers build their own UnifiedImageProcessor, so instance state
# is thrown away with the worker.
_UNPACKED_RAW_LOCK = _ebt.Lock()
_UNPACKED_RAW_SLOTS: OrderedDict = OrderedDict()


def release_unpacked_raw_cache() -> int:
    """Drop the unpacked-mosaic stash. Returns approximate bytes freed."""
    freed = 0
    with _UNPACKED_RAW_LOCK:
        for value in _UNPACKED_RAW_SLOTS.values():
            try:
                arr = value[0] if isinstance(value, tuple) else value
                freed += int(getattr(arr, "nbytes", 0) or 0)
            except Exception:
                pass
        _UNPACKED_RAW_SLOTS.clear()
    return freed


# Serialises the post-decode persist work. Each of these captures the FULL
# sensor buffer (hundreds of MB) in a closure and may re-apply the sidecar at
# full resolution, so fast navigation used to overlap several unbounded
# daemon threads, each holding its own buffer alive. One worker means at most
# one such buffer is pinned, and the generation checks inside the job still
# drop work whose file has moved on.
_PERSIST_POOL = None
_PERSIST_POOL_LOCK = _ebt.Lock()


def _persist_executor():
    global _PERSIST_POOL
    with _PERSIST_POOL_LOCK:
        if _PERSIST_POOL is None:
            from concurrent.futures import ThreadPoolExecutor

            _PERSIST_POOL = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rv-persist"
            )
        return _PERSIST_POOL


def _submit_persist(fn) -> None:
    """Queue background persist work; never raise into the decode path."""
    try:
        _persist_executor().submit(fn)
    except Exception:
        with contextlib.suppress(Exception):
            fn()


# Generation tokens for deferred put_full / preview persist. MODULE-level:
# ImageLoadWorker builds a fresh UnifiedImageProcessor per task, so a
# per-instance gen map could never see another worker's newer decode of the
# same path. Persist jobs check "has the user moved on?" against THIS map;
# if worker B's decode of path X cannot bump the token worker A's in-flight
# job is watching, A's late write poisons the cache after B already put a
# fresher buffer. Same half-hoist failure as lock/list and inflight pairs.
_ASYNC_CACHE_GEN: OrderedDict = OrderedDict()
_ASYNC_CACHE_GEN_LOCK = _ebt.Lock()
_ASYNC_CACHE_GEN_MAX = 4096


def _edit_base_file_ident(file_path: str) -> tuple:
    try:
        st = os.stat(file_path)
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        return (int(mtime_ns), int(st.st_size))
    except OSError:
        return (0, 0)


def _unpack_stash_max_slots() -> int:
    """How many UnpackedRaw mosaics to keep for half→full / A↔B revisit.

    ~2 bytes/sensor-pixel each; default 3 covers current + two neighbors.
    Override with RAWVIEWER_UNPACK_STASH_SLOTS (1–8).
    """
    raw = os.environ.get("RAWVIEWER_UNPACK_STASH_SLOTS", "").strip()
    if raw:
        try:
            return max(1, min(8, int(raw)))
        except ValueError:
            pass
    return 3


def prune_libraw_unsupported_paths(
    *, clear_all: bool = False, max_keep: int = 512
) -> None:
    """Bound or clear the session-global LibRaw skip set (folder change)."""
    global _LIBRAW_UNSUPPORTED_PATHS
    if clear_all:
        _LIBRAW_UNSUPPORTED_PATHS.clear()
        return
    if len(_LIBRAW_UNSUPPORTED_PATHS) <= max_keep:
        return
    # Arbitrary surplus drop; next miss re-probes and re-adds if still bad.
    drop_n = len(_LIBRAW_UNSUPPORTED_PATHS) - max_keep
    for i, k in enumerate(list(_LIBRAW_UNSUPPORTED_PATHS)):
        if i >= drop_n:
            break
        _LIBRAW_UNSUPPORTED_PATHS.discard(k)


def _is_libraw_unsupported_error(exc: BaseException) -> bool:
    """True when LibRaw rejected the file (not a transient decode failure)."""
    try:
        import rawpy

        cls = getattr(rawpy, "LibRawFileUnsupportedError", None)
        if cls is not None and isinstance(exc, cls):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "unsupported" in msg or "not recognized" in msg


def _mark_libraw_unsupported(path_key: str) -> None:
    _LIBRAW_UNSUPPORTED_PATHS.add(path_key)
    if len(_LIBRAW_UNSUPPORTED_PATHS) > _LIBRAW_UNSUPPORTED_MAX:
        prune_libraw_unsupported_paths(max_keep=_LIBRAW_UNSUPPORTED_MAX // 2)


def release_edit_base_cache() -> int:
    """Drop the half-size edit-base LRU. Returns bytes freed (approx).

    Called by the image cache's aggressive purge, and after a LibRaw binary
    swap. This LRU holds up to two float32 half-size bases -- on a 45MP file
    that is ~135 MB each -- and was the one heavy cache the purge could not
    see, so a critical-pressure purge cleared everything else and left
    ~270 MB sitting here.
    """
    freed = 0
    with _EDIT_BASE_LRU_LOCK:
        for value in _EDIT_BASE_LRU.values():
            try:
                arr = value[0] if isinstance(value, tuple) else value
                freed += int(getattr(arr, "nbytes", 0) or 0)
            except Exception:
                pass
        _EDIT_BASE_LRU.clear()
    return freed


def trim_edit_base_lru(max_items: int | None = None) -> int:
    """Drop oldest edit-base LRU entries down to ``max_items`` (keep the newest).

    Used by memory-pressure backoff. Does not change ``RAWVIEWER_EDIT_BASE_MAX_EDGE``
    and does not rebuild the in-use working buffer.
    """
    cap = max(1, int(max_items if max_items is not None else _edit_base_lru_max()))
    freed = 0
    with _EDIT_BASE_LRU_LOCK:
        while len(_EDIT_BASE_LRU) > cap:
            _key, value = _EDIT_BASE_LRU.popitem(last=False)
            try:
                arr = value[0] if isinstance(value, tuple) else value
                freed += int(getattr(arr, "nbytes", 0) or 0)
            except Exception:
                pass
    return freed


def decode_raw_file(file_path: str, params: dict[str, Any]) -> np.ndarray:
    """
    Helper function for process pool to decode RAW files.
    Must be at top level for pickling.
    """
    from enhanced_raw_processor import _heavy_fallback_semaphore, _rawpy_global_lock
    with _rawpy_global_lock:
        raw_ctx = rawpy_imread_warm(file_path)
    with raw_ctx as raw, _heavy_fallback_semaphore:
        return raw.postprocess(**params)
