"""Unified image load manager: task types, worker, manager, and singleton helpers."""

from . import helpers as _ilm_helpers
from .helpers import (
    _HE_PREVIEW_ONLY_CACHE,
    Priority,
    _buffer_max_dim,
    _display_preview_min_dim,
    _env_int,
    _env_true,
    _gallery_grid_min_dim,
    _gallery_memory_thumb_acceptable,
    _has_running_current_task,
    _min_acceptable_preview_dim,
    _preview_is_best_possible,
    _skip_low_res_memory_thumb_for_display_tier,
    _task_queue_weight,
    apply_gpu_decode_profile_to_manager,
    get_image_load_manager,
    peek_image_load_manager,
    worker_thread_local,
    yield_if_current_task_active,
)
from .manager import ImageLoadManager
from .worker import ImageLoadTask, ImageLoadWorker

__all__ = [
    "ImageLoadManager",
    "ImageLoadTask",
    "ImageLoadWorker",
    "Priority",
    "apply_gpu_decode_profile_to_manager",
    "get_image_load_manager",
    "peek_image_load_manager",
    "yield_if_current_task_active",
    "worker_thread_local",
    "_HE_PREVIEW_ONLY_CACHE",
    "_buffer_max_dim",
    "_display_preview_min_dim",
    "_env_int",
    "_env_true",
    "_gallery_grid_min_dim",
    "_gallery_memory_thumb_acceptable",
    "_has_running_current_task",
    "_min_acceptable_preview_dim",
    "_preview_is_best_possible",
    "_skip_low_res_memory_thumb_for_display_tier",
    "_task_queue_weight",
]


def __getattr__(name: str):
    """Live module-attribute fallback (PEP 562).

    The singleton state (``_global_manager`` / ``_manager_lock``) is rebound by
    :func:`get_image_load_manager`, so it is resolved through the helpers module
    on every access instead of being captured by value here.
    """
    return getattr(_ilm_helpers, name)
