"""Module-level helpers and state for the ImageLoadManager package."""
from __future__ import annotations

import contextlib
import os
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import ImageLoadManager
    from .worker import ImageLoadTask

# Thread-local storage to track if current thread is executing a CURRENT priority task
worker_thread_local = threading.local()


def _has_running_current_task(manager, *, exclude_task=None) -> bool:
    """True only when a CURRENT task is actively executing (not merely queued).

    Checking queued CURRENT (_active_tasks) caused thread-pool livelock: every
    background worker slept waiting for CURRENT, which could not start because
    those sleeping workers occupied every slot.
    Note: we use a dedicated _running_tasks_lock to avoid contention with
    _schedule_next_task, which holds _queue_lock for the duration of a queue drain.
    """
    running = getattr(manager, "_running_tasks", None)
    if not running:
        return False
    lock = getattr(manager, "_running_tasks_lock", None)
    if lock is None:
        return False
    with lock:
        for t in running:
            if t is exclude_task:
                continue
            if getattr(t, "priority", None) == Priority.CURRENT and not t.is_cancelled():
                return True
    return False

def yield_if_current_task_active() -> None:
    """Yield briefly if another thread is *running* a high-priority CURRENT task.

    BOUNDED and running-only (never waits on merely-queued CURRENT). Silent
    workers + separate _current_thread_pool ensure queued CURRENT can always
    obtain a thread without requiring sleepers to free the background pool.
    """
    if threading.current_thread() is threading.main_thread():
        return
    current_priority = getattr(worker_thread_local, 'priority', None)
    if current_priority == Priority.CURRENT:
        return

    manager = _global_manager
    if not manager:
        return

    own_task = getattr(worker_thread_local, 'task', None)
    max_wait_s = _env_int("RAWVIEWER_YIELD_MAX_MS", 1500, minimum=100) / 1000.0
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        if own_task is not None and own_task.is_cancelled():
            return
        if not _has_running_current_task(manager, exclude_task=own_task):
            break
        time.sleep(0.05)

def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(minimum, default)

def _buffer_max_dim(buf) -> int:
    try:
        if hasattr(buf, "shape"):
            return max(int(buf.shape[0]), int(buf.shape[1]))
        if hasattr(buf, "width") and hasattr(buf, "height"):
            return max(int(buf.width()), int(buf.height()))
    except Exception:
        pass
    return 0

def _display_preview_min_dim() -> int:
    from image_cache import memory_preview_max_edge

    return int(memory_preview_max_edge() * 0.85)

def _min_acceptable_preview_dim(file_path: str) -> int:
    from common_image_loader import dng_prefers_embedded_preview_first

    if dng_prefers_embedded_preview_first(file_path):
        return 1024
    return _display_preview_min_dim()

# Memo for _preview_is_best_possible: HE detection reads the file's EXIF, and
# a folder scan asks about the same paths repeatedly.
_HE_PREVIEW_ONLY_CACHE: dict = {}

def _preview_is_best_possible(processor, file_path: str) -> bool:
    """Whether no larger preview could exist for this file, however we try.

    An undersized preview is normally discarded so a better one can be built
    by demosaicing. That is the wrong move when demosaicing cannot work: the
    embedded JPEG is all there is, and dropping it leaves a grey tile instead
    of a slightly-small thumbnail.

    processor.is_libraw_unsupported() is the existing test, but it only knows
    paths that have ALREADY failed a decode and been registered -- and the
    registry is in-memory and pruned under pressure. During a cold gallery
    scan an HE/HE* NEF has not failed anything yet, so the guard missed
    exactly the files it exists for. Nikon HE is detectable up front, so ask
    directly rather than waiting for a failure to teach us.
    """
    try:
        if processor.is_libraw_unsupported(file_path):
            return True
    except Exception:
        pass
    if not file_path.lower().endswith(".nef"):
        return False
    key = os.path.normcase(os.path.abspath(file_path))
    cached = _HE_PREVIEW_ONLY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from enhanced_raw_processor import _detect_nef_he_compression

        result = _detect_nef_he_compression(file_path) is True
    except Exception:
        result = False
    if len(_HE_PREVIEW_ONLY_CACHE) > 4096:
        _HE_PREVIEW_ONLY_CACHE.clear()
    _HE_PREVIEW_ONLY_CACHE[key] = result
    return result

def _gallery_grid_min_dim() -> int:
    from image_cache import disk_preview_max_edge

    return int(disk_preview_max_edge() * 0.85)

def _gallery_memory_thumb_acceptable(thumb, *, gallery_thumbnail: bool) -> bool:
    """Gallery tiles need grid-tier (~512px), not 256px thumbnail tier."""
    if not gallery_thumbnail:
        return True
    dim = _buffer_max_dim(thumb)
    if dim <= 0:
        return False
    return dim >= _gallery_grid_min_dim()

def _skip_low_res_memory_thumb_for_display_tier(
    thumb,
    wanted: set,
    use_full_resolution: bool,
    *,
    gallery_thumbnail: bool = False,
) -> bool:
    """Do not emit 512px grid-tier thumbs for preview-first single-view loads."""
    if gallery_thumbnail:
        return False
    if use_full_resolution or "full" in wanted:
        return False
    if wanted == {"thumbnail"}:
        return False
    if "thumbnail" not in wanted or "exif" not in wanted:
        return False
    min_dim = _env_int("RAWVIEWER_GALLERY_INSTANT_MIN_DIM", 1400, minimum=400)
    px = _buffer_max_dim(thumb)
    return 0 < px < min_dim

class Priority(Enum):
    """任務優先級"""
    CURRENT = 0  # 當前圖像（最高優先級）
    PRELOAD_NEXT = 1  # 下一個圖像預載入
    PRELOAD_PREV = 2  # 上一個圖像預載入
    BACKGROUND = 3  # 背景任務（最低優先級）

def _task_queue_weight(task: ImageLoadTask) -> int:
    """Lower weight = scheduled sooner within the same Priority band."""
    stages = task.stages or set()
    if stages == {"thumbnail"}:
        return 0
    if stages <= {"thumbnail", "exif"}:
        return 1
    if "full" in stages:
        return 3
    return 2

_global_manager: ImageLoadManager | None = None
_manager_lock = threading.Lock()

def peek_image_load_manager() -> ImageLoadManager | None:
    """Return the manager if already constructed; never create one.

    Cache sizing and other early-startup helpers must use this. Calling
    ``get_image_load_manager()`` from inside ``ImageCache`` / manager
    construction re-enters the singleton lock and deadlocks (splash hang
    after clear-cache / cold start).
    """
    return _global_manager

def get_image_load_manager(max_workers: int = 4) -> ImageLoadManager:
    """獲取全局圖像加載管理器實例（函數級單例模式）

    RAWVIEWER_LOAD_WORKERS overrides max_workers (clamped to [2, 16]); the
    4-thread default has been the gallery tile-fill ceiling on multi-core
    machines even when every thumbnail is disk-cached.
    """
    global _global_manager
    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                raw = os.environ.get("RAWVIEWER_LOAD_WORKERS", "").strip()
                if raw:
                    with contextlib.suppress(ValueError):
                        max_workers = max(2, min(16, int(raw)))
                from .manager import ImageLoadManager

                _global_manager = ImageLoadManager(max_workers)
    return _global_manager

def apply_gpu_decode_profile_to_manager() -> None:
    """Called after main-thread GPU backend import (see torch_bootstrap)."""
    mgr = _global_manager
    if mgr is not None and hasattr(mgr, "apply_gpu_decode_profile"):
        with contextlib.suppress(Exception):
            mgr.apply_gpu_decode_profile()
