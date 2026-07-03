import sys
import time
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import QWidget, QScrollArea, QLabel
from PyQt6.QtCore import Qt, QTimer, QRect, QEvent, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QTransform

from rawviewer_ui.widgets import ThumbnailLabel
from image_cache import LRUCache
from image_load_manager import get_image_load_manager, Priority
from common_image_loader import is_raw_file

logger = logging.getLogger(__name__)


def _focus_gallery_switch_logs() -> bool:
    return os.environ.get("RAWVIEWER_FOCUS_GALLERY_SWITCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


def _gallery_prefetch_screens(fast: bool) -> int:
    """Viewport heights to prefetch above/below scroll center (embedded JPEG makes this cheap)."""
    if fast:
        return _env_int("RAWVIEWER_GALLERY_PREFETCH_SCREENS_FAST", 5, minimum=1)
    return _env_int("RAWVIEWER_GALLERY_PREFETCH_SCREENS", 6, minimum=1)


def _gallery_viewport_buffer_screens() -> float:
    """Extra viewport heights kept for visible tile widgets above/below the scroll window."""
    raw = os.environ.get("RAWVIEWER_GALLERY_VIEWPORT_BUFFER_SCREENS", "1.25").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.25


def _gallery_scroll_buffer_screens() -> float:
    """Wider widget pool while scrolling to reduce tile churn when thumbnails are warm."""
    base = _gallery_viewport_buffer_screens()
    raw = os.environ.get("RAWVIEWER_GALLERY_SCROLL_BUFFER_SCREENS", "2.5").strip()
    try:
        return max(base, float(raw))
    except ValueError:
        return max(base, 2.5)


def _gallery_current_active_reserve(active_cap: int) -> int:
    """Slots reserved for visible (CURRENT) gallery tiles; prefetch cannot use them."""
    default = max(4, active_cap // 3)
    reserve = _env_int("RAWVIEWER_GALLERY_CURRENT_ACTIVE_RESERVE", default, minimum=2)
    return min(reserve, max(2, active_cap - 4))


def _gallery_scheduling_budgets(fast: bool) -> tuple[int, int, int]:
    """Return (max_widgets, max_tasks, active_cap) for the current scroll mode."""
    if fast:
        return (
            _env_int("RAWVIEWER_GALLERY_MAX_WIDGETS_FAST", 16, minimum=1),
            _env_int("RAWVIEWER_GALLERY_MAX_TASKS_FAST", 24, minimum=1),
            _env_int("RAWVIEWER_GALLERY_ACTIVE_CAP_FAST", 32, minimum=4),
        )
    return (
        _env_int("RAWVIEWER_GALLERY_MAX_WIDGETS", 48, minimum=1),
        _env_int("RAWVIEWER_GALLERY_MAX_TASKS", 64, minimum=1),
        _env_int("RAWVIEWER_GALLERY_ACTIVE_CAP", 64, minimum=4),
    )


def _apply_external_gallery_caps(
    max_widgets: int,
    max_tasks: int,
    active_cap: int,
    folder_path: Optional[str],
) -> tuple[int, int, int]:
    """Lower gallery scheduling on external/USB/network volumes (post-warmup too).

    Applies when the volume is confirmed-slow (all platforms) or, on Windows,
    for any external volume (moderate_external_cap_enabled). macOS fast external
    drives keep the full gallery budget — they never showed the native crash.
    """
    if not folder_path:
        return max_widgets, max_tasks, active_cap
    try:
        from common_image_loader import (
            is_external_or_network_volume,
            moderate_external_cap_enabled,
            volume_speed_tier,
        )

        if not is_external_or_network_volume(folder_path):
            return max_widgets, max_tasks, active_cap
        is_slow = volume_speed_tier(folder_path) == "slow"
        if is_slow:
            cap = _env_int("RAWVIEWER_SLOW_GALLERY_ACTIVE_CAP", 32, minimum=2)
            tasks = _env_int("RAWVIEWER_SLOW_GALLERY_MAX_TASKS", 24, minimum=2)
            widgets = _env_int("RAWVIEWER_SLOW_GALLERY_MAX_WIDGETS", 40, minimum=4)
            return (
                min(max_widgets, widgets),
                min(max_tasks, tasks),
                min(active_cap, cap),
            )
        if not moderate_external_cap_enabled():
            # Stability without throttling: cap fast external on Windows only (GL crash
            # was mode-switch teardown, not this concurrency). macOS keeps full budget.
            if sys.platform != "win32":
                return max_widgets, max_tasks, active_cap
    except Exception:
        return max_widgets, max_tasks, active_cap
    # Fast external (e.g. USB3/TB probing >120 MB/s).
    cap = _env_int("RAWVIEWER_EXTERNAL_GALLERY_ACTIVE_CAP", 32, minimum=2)
    tasks = _env_int("RAWVIEWER_EXTERNAL_GALLERY_MAX_TASKS", 24, minimum=2)
    widgets = _env_int("RAWVIEWER_EXTERNAL_GALLERY_MAX_WIDGETS", 40, minimum=4)
    return (
        min(max_widgets, widgets),
        min(max_tasks, tasks),
        min(active_cap, cap),
    )


def _gallery_warmup_scheduling_budgets(file_count: int) -> tuple[int, int, int]:
    """Brief soft cap during gallery entry (GL teardown is the real crash fix)."""
    if file_count >= 2500:
        return (12, 8, 16)
    if file_count >= 1200:
        return (20, 14, 28)
    if file_count >= 500:
        return (24, 18, 32)
    return (28, 20, 36)


def _batch_tile_apply_enabled() -> bool:
    """Default ON. Set RAWVIEWER_GALLERY_BATCH_TILE_APPLY=0 to apply tile pixmaps inline."""
    return os.environ.get(
        "RAWVIEWER_GALLERY_BATCH_TILE_APPLY", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _tile_apply_batch_size() -> int:
    """Max tile pixmaps applied to the UI per event-loop tick when batching is on."""
    return _env_int("RAWVIEWER_GALLERY_TILE_APPLY_BATCH", 8, minimum=1)


def _orient_debug_enabled() -> bool:
    """Diagnostic only: RAWVIEWER_ORIENT_DEBUG=1 logs per-tile orientation decisions."""
    return os.environ.get("RAWVIEWER_ORIENT_DEBUG", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _gallery_idle_preload_batch() -> int:
    return _env_int("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH", 72, minimum=4)


def _gallery_idle_preload_ms() -> int:
    return _env_int("RAWVIEWER_GALLERY_IDLE_PRELOAD_MS", 250, minimum=50)


def _min_layout_pixmap_dim() -> int:
    """Nav/filmstrip previews below this size must not drive justified layout aspect."""
    return _env_int("RAWVIEWER_GALLERY_MIN_LAYOUT_PIXMAP_DIM", 384, minimum=128)


def _indexing_loads_compete() -> bool:
    """True when semantic/face indexing may flood the load queue at BACKGROUND priority."""
    sem = os.environ.get("RAWVIEWER_ENABLE_SEMANTIC_SEARCH", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    face = os.environ.get("RAWVIEWER_ENABLE_FACE_SCAN", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return sem or face


def _gallery_idle_load_priority() -> Priority:
    """Queue priority for off-screen gallery thumbnail idle preload."""
    raw = os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_PRIORITY", "").strip().lower()
    if raw in ("background", "low"):
        return Priority.BACKGROUND
    if raw in ("preload_prev", "prev"):
        return Priority.PRELOAD_PREV
    if raw in ("preload_next", "preload", "high", "next"):
        return Priority.PRELOAD_NEXT
    if not _indexing_loads_compete():
        return Priority.PRELOAD_NEXT
    return Priority.BACKGROUND


def _thumbnail_data_to_base_pixmap(thumbnail_data) -> Optional[QPixmap]:
    """Convert manager/cache thumbnail payload to a gallery base QPixmap."""
    if thumbnail_data is None:
        return None
    if isinstance(thumbnail_data, QPixmap):
        return thumbnail_data if not thumbnail_data.isNull() else None
    if isinstance(thumbnail_data, np.ndarray):
        arr = np.ascontiguousarray(thumbnail_data)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.ndim == 3:
            h, w, c = arr.shape[:3]
            if h <= 0 or w <= 0:
                return None
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                arr = np.ascontiguousarray(arr)
            if c == 3:
                bytes_per_line = arr.strides[0]
                qimg = QImage(
                    arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                ).copy()
            elif c == 4:
                bytes_per_line = arr.strides[0]
                qimg = QImage(
                    arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
                ).copy()
            else:
                return None
        else:
            return None
        if qimg.isNull():
            return None
        return QPixmap.fromImage(qimg)
    if isinstance(thumbnail_data, QImage):
        if thumbnail_data.isNull():
            return None
        rgb = thumbnail_data.convertToFormat(QImage.Format.Format_RGB888)
        return QPixmap.fromImage(rgb) if not rgb.isNull() else None
    return None


def _rotate_pixmap_cw(pixmap: QPixmap, degrees: int) -> QPixmap:
    """Rotate pixmap clockwise for on-screen display (matches single-image view)."""
    degrees = int(degrees) % 360
    if pixmap is None or pixmap.isNull() or degrees == 0:
        return pixmap
    transform = QTransform()
    transform.rotate(degrees)
    return pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)


class JustifiedGallery(QWidget):
    """
    Adaptive justified gallery layout with high-performance virtualization.
    Optimized for large directories using binary search and widget pooling.
    """

    TARGET_ROW_HEIGHT = 220
    HEIGHT_TOLERANCE = 0.25
    MIN_SPACING = 6
    # Never bulk-read EXIF for entire huge folders on the UI thread during layout build.
    BUILD_EXIF_PREFETCH_MAX = 128

    _background_loading_active: bool = False
    _load_queue: list[Any] = []
    _priority_queue: list[Any] = []
    _gallery_load_start_time: float = 0.0
    _visible_images_to_load: int = 0
    _visible_images_loaded: int = 0

    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.images = images
        self._rebuild_normalized_map()

        self.TARGET_ROW_HEIGHT = 220

        # Virtualization Data
        self._gallery_layout_items = []  # List of {rect, file_path, aspect}
        # Decoded-thumbnail raw aspect per path (filmstrip-style ground truth). Prefer this
        # over EXIF for tile geometry so a portrait image can't be crop-fit into a landscape
        # tile from stale metadata / an evicted pixmap. Applied through _display_aspect.
        self._measured_raw_aspects: "dict[str, float]" = {}
        # Paths whose EXIF was already looked up from the DB this folder (hit or miss);
        # build_gallery must not repeat main-thread bulk SQL for them on every rebuild.
        self._metadata_fetch_attempted: "set[str]" = set()
        self._orientation_flip_paths: set[str] = set()
        self._visible_widgets = {}  # {index: ThumbnailLabel}
        self._widget_pool = []
        self._total_content_height = 0

        # State Management
        self._building = False
        self._gallery_generation = 0
        self._active_tasks = {}  # path -> (started_ts, Priority)
        self._thumb_fail_counts: Dict[str, int] = {}
        self._metadata_cache = {}
        # Thumbnail cache: base + per-bucket scaled
        self._thumbnail_cache = LRUCache(10000)
        self._thumb_base_key = "__base__"
        self._row_height_buckets = list(range(100, 400, 20)) # updated max to 400
        self._width_bucket_px = 64  # quantize widths to avoid mismatched cached pixmaps
        
        # Zoom state variables
        self._is_zooming = False
        self._zoom_anchor_state = None
        
        # Mapping of file_path to list of indices (for when the same file appears multiple times)
        self._path_to_indices = {}
        # Track what thumbnails we've recently requested so we can cancel far-away work
        self._requested_thumbnail_paths = set()
        self._pending_scroll_to_path = None
        self._pending_scroll_anchor_state = None
        # True once the user has jumped far from the entry-point image (see
        # load_visible_images' big-jump handling); reset on each fresh set_images().
        self._entry_prefetch_abandoned = False
        self._is_zooming = False
        self._zoom_anchor_state = None
        self._gallery_zoom_rebuild = False
        self._is_scrollbar_dragging = False
        self._last_scheduled_scroll_y = 0
        self._scroll_area = None
        self._last_layout_viewport_width = -1
        self._last_layout_content_height = 0
        self._layout_image_sequence: tuple = ()
        self._last_build_ts = 0.0
        self._last_force_build_ts = 0.0
        self._last_load_visible_request_ts = 0.0
        self._gallery_set_images_ts = 0.0
        self._first_thumb_ready_after_set = False
        self._gallery_warmup_until = 0.0
        self._thumb_ready_depth = 0
        self._pending_gallery_build = False
        self._pending_build_metadata = None
        self._pending_build_force = False
        self._gallery_folder_token = None
        self._ignore_resize_events = False
        self._gallery_zoom_rebuild = False

        self._loading_label = None
        self._empty_label = None

        # Smooth wheel scrolling (pixel-based) to avoid non-linear jumps and keep thumbnails in sync
        self._wheel_accum_px = 0.0
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(False)
        self._wheel_timer.timeout.connect(self._apply_wheel_scroll_step)
        self._wheel_step_px = 18  # per tick (tuned for smoothness)
        self._wheel_tick_ms = 8   # 125Hz-ish

        # Idle background thumbnail preloader
        self._idle_preload_timer = QTimer(self)
        self._idle_preload_timer.setSingleShot(True)
        self._idle_preload_timer.timeout.connect(self._preload_remaining_thumbnails_background)

        # Debounced retry timer for thumbnail errors to prevent retry storms
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.timeout.connect(self._on_retry_timer_timeout)
        self._max_retry_attempt = 1

        # Batched tile pixmap application: when a burst of thumbnails completes at once,
        # applying each (scale/crop/rotate + setPixmap) inline on the main thread can
        # hitch the scroll frame. Instead, coalesce completed paths and drain a bounded
        # number per event-loop tick so per-frame main-thread work stays smooth. Disable
        # with RAWVIEWER_GALLERY_BATCH_TILE_APPLY=0; tune batch size with
        # RAWVIEWER_GALLERY_TILE_APPLY_BATCH (default 8).
        self._pending_tile_applies: "dict[str, bool]" = {}
        self._tile_apply_timer = QTimer(self)
        self._tile_apply_timer.setSingleShot(True)
        self._tile_apply_timer.timeout.connect(self._drain_tile_applies)

        # Work budget per tick for smooth scrolling (embedded-JPEG thumbnails are cheap).
        self._max_widgets_per_tick = _env_int("RAWVIEWER_GALLERY_MAX_WIDGETS", 20, minimum=1)
        self._max_tasks_per_tick = _env_int("RAWVIEWER_GALLERY_MAX_TASKS", 32, minimum=1)

        # Perf logging (throttled)
        self._perf_last_log_t = 0.0
        self._last_scroll_event_t = 0.0
        self._last_scroll_settle_t = 0.0
        self._thumb_first_after_scroll_t = None
        self._thumb_first_after_settle_t = None

        # Defer heavier init to the next event-loop tick to avoid hangs during construction
        self.load_manager = None
        self._load_timer = None
        self._scroll_settle_timer = None
        self._resize_timer = None
        self._last_scroll_y = -1
        self._last_scroll_time = time.time()
        self._current_scroll_speed = 0.0
        self._is_scrolling_fast = False
        self._scroll_indexing_paused = False
        # Trackpads emit many small deltas; a lower threshold causes "fast scroll" mode
        # and blank tiles. macOS defaults higher; override with RAWVIEWER_GALLERY_FAST_SCROLL_PX_S.
        default_fast = 12000 if sys.platform == "darwin" else 6000
        self._scroll_optimize_threshold = _env_int(
            "RAWVIEWER_GALLERY_FAST_SCROLL_PX_S", default_fast, minimum=2000
        )

        # Metadata tracking for dynamic layout
        self._metadata_changed_paths = set()
        self._last_metadata_change_time = 0.0
        self._failed_thumbnails = set()
        self._metadata_rebuild_timer = QTimer(self)
        self._metadata_rebuild_timer.setSingleShot(True)
        self._metadata_rebuild_timer.timeout.connect(self._handle_metadata_rebuild)

        self._aspects_settle_timer = QTimer(self)
        self._aspects_settle_timer.setSingleShot(True)
        self._aspects_settle_timer.timeout.connect(self._on_aspects_settle_rebuild)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        QTimer.singleShot(0, self._post_init)



    def _post_init(self):
        try:
            # Threading: Prefer viewer's manager to avoid creating a second global manager
            if self.parent_viewer is not None and hasattr(self.parent_viewer, "image_manager") and self.parent_viewer.image_manager:
                self.load_manager = self.parent_viewer.image_manager
            else:
                self.load_manager = get_image_load_manager()

            try:
                self.load_manager.thumbnail_ready.disconnect(self.on_thumbnail_ready)
            except Exception:
                pass
            try:
                self.load_manager.error_occurred.disconnect(self.on_thumbnail_error)
            except Exception:
                pass
            self.load_manager.thumbnail_ready.connect(self.on_thumbnail_ready)
            self.load_manager.error_occurred.connect(self.on_thumbnail_error)
            
            try:
                self.load_manager.exif_data_ready.disconnect(self.on_exif_ready)
            except Exception:
                pass
            self.load_manager.exif_data_ready.connect(self.on_exif_ready)
            
            try:
                self.load_manager.task_completed.disconnect(self.on_task_completed)
            except Exception:
                pass
            self.load_manager.task_completed.connect(self.on_task_completed)

            # Timer Management
            self._load_timer = QTimer(self)
            self._load_timer.setSingleShot(True)
            self._load_timer.timeout.connect(self.load_visible_images)

            self._scroll_settle_timer = QTimer(self)
            self._scroll_settle_timer.setSingleShot(True)
            self._scroll_settle_timer.timeout.connect(self._on_scroll_settled)

            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._handle_resize_rebuild)

            # Finish setup
            QTimer.singleShot(50, self._delayed_initial_build)
        except Exception:
            logger.exception("[GALLERY] _post_init failed")

    def _current_folder_load_token(self):
        pv = self.parent_viewer
        if pv is None:
            return None
        return getattr(pv, "_folder_load_generation", None)

    def _reset_gallery_scroll_position(self) -> None:
        """Scroll to top and clear stale scroll bookkeeping (e.g. after folder change)."""
        scroll = self._scroll_area_widget()
        if scroll is not None:
            sb = scroll.verticalScrollBar()
            sb.setValue(sb.minimum())
        self._last_scheduled_scroll_y = 0
        self._last_scroll_y = -1
        self._is_scrolling_fast = False
        self._current_scroll_speed = 0.0

    def _invalidate_gallery_layout(self) -> None:
        """Drop cached layout geometry so the next pass must rebuild/reposition."""
        self._building = False
        self._pending_gallery_build = False
        self._pending_build_metadata = None
        self._pending_build_force = False
        self._gallery_layout_items.clear()
        self._path_to_indices.clear()
        self._layout_image_sequence = ()
        self._last_layout_content_height = 0
        self._last_layout_viewport_width = -1
        self._last_build_ts = 0.0
        self._is_zooming = False
        self._zoom_anchor_state = None

    def _layout_matches_current_images(self, images: Optional[List[str]] = None) -> bool:
        """True when cached layout rects correspond to the current image list."""
        seq = list(images if images is not None else self.images)
        return bool(
            self._gallery_layout_items
            and len(self._gallery_layout_items) == len(seq)
            and tuple(seq) == self._layout_image_sequence
            and not self._gallery_folder_superseded()
        )

    def _reposition_thumbnail_widgets(self) -> None:
        """Sync visible/pooled thumbnail widgets to current layout item rects."""
        items = self._gallery_layout_items
        if not items:
            return
        tracked: set[int] = set()
        for idx, w in list(self._visible_widgets.items()):
            if idx < 0 or idx >= len(items):
                w.hide()
                self._clear_widget_thumbnail(w)
                w.file_path = None
                self._widget_pool.append(w)
                del self._visible_widgets[idx]
                continue
            item = items[idx]
            path = item.get("file_path")
            rect = item.get("rect")
            if not path or rect is None:
                continue
            tracked.add(id(w))
            if w.file_path != path:
                self._clear_widget_thumbnail(w)
            w.file_path = path
            w.index = idx
            display_rect = self._content_rect_to_viewport(rect)
            w.setGeometry(display_rect)
            w.setFixedSize(display_rect.size())
        for child in self.findChildren(ThumbnailLabel):
            if child.parent() is not self or id(child) in tracked:
                continue
            child.hide()
            self._clear_widget_thumbnail(child)
            child.file_path = None
            if child not in self._widget_pool:
                self._widget_pool.append(child)

    def suspend_thumbnail_display(self) -> None:
        """Hide thumbnail widgets when leaving gallery without discarding layout."""
        for w in list(self._visible_widgets.values()):
            try:
                w.hide()
            except Exception:
                pass
        for w in self._widget_pool:
            try:
                w.hide()
            except Exception:
                pass
        for child in self.findChildren(ThumbnailLabel):
            try:
                child.hide()
            except Exception:
                pass

    def cancel_background_loading(self) -> None:
        """Stop gallery thumbnail scheduling and in-flight decodes (mode switch to single view)."""
        self._background_loading_active = False
        for timer_name in (
            "_load_timer",
            "_resize_timer",
            "_scroll_settle_timer",
            "_metadata_rebuild_timer",
            "_aspects_settle_timer",
            "_idle_preload_timer",
        ):
            timer = getattr(self, timer_name, None)
            if timer is not None and hasattr(timer, "stop"):
                try:
                    timer.stop()
                except Exception:
                    pass
        mgr = getattr(self, "load_manager", None)
        if mgr is not None and hasattr(mgr, "cancel_gallery_tasks"):
            try:
                mgr.cancel_gallery_tasks()
            except Exception:
                pass
        self._active_tasks.clear()
        self._requested_thumbnail_paths.clear()

    def prepare_gallery_display(self) -> None:
        """Ensure layout and widget geometry are valid when gallery becomes visible."""
        if not self.images:
            return
        # _update_gallery_view -> set_images() runs a few ms later on mode switch;
        # skip a duplicate forced build here (same viewport width, double layout).
        if time.time() < float(getattr(self, "_gallery_entry_coalesce_until", 0.0) or 0.0):
            return
        if not self._layout_matches_current_images():
            self.build_gallery(force=True)
            return
        self._sync_content_geometry()
        self._reposition_thumbnail_widgets()
        self._request_load_visible_images(0)

    def prepare_for_folder_change(self) -> None:
        """Drop in-flight gallery thumbnail state when the folder scope changes."""
        self._is_zooming = False
        self._zoom_anchor_state = None
        self._gallery_warmup_until = 0.0
        self._active_tasks.clear()
        self._requested_thumbnail_paths.clear()
        if hasattr(self, "_load_timer") and self._load_timer:
            self._load_timer.stop()
        self._idle_preload_timer.stop()
        self._reset_gallery_scroll_position()
        self._invalidate_gallery_layout()
        self._measured_raw_aspects.clear()
        self._orientation_flip_paths.clear()
        self._metadata_fetch_attempted.clear()
        self.clear_thumbnail_widgets()

    def _evict_stale_active_tasks(
        self,
        *,
        max_age_s: float = 4.0,
        drop_prefetch: bool = False,
    ) -> int:
        """Remove phantom or stale in-flight markers so visible loads can proceed."""
        now = time.time()
        to_remove: list[str] = []
        for path in self._active_tasks:
            ts = self._active_task_started_ts(path) or now
            if now - ts > max_age_s:
                to_remove.append(path)
            elif drop_prefetch and self._active_task_priority(path) != Priority.CURRENT:
                to_remove.append(path)
        for path in to_remove:
            self._active_tasks.pop(path, None)
            try:
                if self.load_manager is not None:
                    self.load_manager.cancel_task(path)
            except Exception:
                pass
        return len(to_remove)

    def _gallery_folder_superseded(self) -> bool:
        captured = getattr(self, "_gallery_folder_token", None)
        current = self._current_folder_load_token()
        if captured is None or current is None:
            return False
        return captured != current

    def _request_load_visible_images(self, delay_ms: int = 30) -> None:
        """Coalesce repeated visible-load requests to avoid event-loop storms."""
        if self._load_timer is None:
            QTimer.singleShot(max(0, delay_ms), self.load_visible_images)
            return
        now = time.time()
        # Prevent back-to-back zero-delay reschedules from thumbnail bursts.
        if (now - self._last_load_visible_request_ts) < 0.05 and self._load_timer.isActive():
            return
        self._last_load_visible_request_ts = now
        self._load_timer.start(max(1, int(delay_ms)))

    def _is_actively_scrolling(self) -> bool:
        """True during an in-progress scroll gesture (trackpad, wheel, or scrollbar)."""
        return (time.time() - self._last_scroll_event_t) < 0.18

    @staticmethod
    def _pixmap_matches(widget, pixmap: Optional[QPixmap]) -> bool:
        """True when the widget already displays the same pixmap instance."""
        if pixmap is None or pixmap.isNull():
            current = widget.pixmap()
            return current is None or current.isNull()
        current = widget.pixmap()
        if current is None or current.isNull():
            return False
        try:
            return int(current.cacheKey()) == int(pixmap.cacheKey())
        except Exception:
            return current is pixmap

    def _paths_have_base_thumbnails(self, paths) -> bool:
        for path in paths:
            if not path:
                continue
            base = self._thumbnail_cache.get((path, self._thumb_base_key))
            if base is None or base.isNull():
                return False
        return bool(paths)

    def _visible_viewport_needs_thumbnails(self) -> bool:
        """True when any on-screen tile still has no painted pixmap."""
        for w in (self._visible_widgets or {}).values():
            path = getattr(w, "file_path", None)
            if not path:
                continue
            if self._thumb_fail_counts.get(path, 0) >= 3:
                continue
            pm = w.pixmap() if hasattr(w, "pixmap") else None
            if pm is None or pm.isNull():
                return True
        return False

    def _apply_pixmap_if_changed(self, widget, pixmap: Optional[QPixmap]) -> None:
        if self._pixmap_matches(widget, pixmap):
            return
        if pixmap is None or pixmap.isNull():
            widget.setPixmap(QPixmap())
        else:
            widget.setPixmap(pixmap)

    def _clear_widget_thumbnail(self, widget) -> None:
        """Drop stale pixmap/text before reassigning a pooled thumbnail widget."""
        widget.clear()
        widget.original_pixmap = None
        widget.setText("")

    def _scale_crop_to_fit(self, pixmap: QPixmap, target_size):
        """Scale pixmap to fully cover target_size, then center-crop to exact size."""
        if not pixmap or pixmap.isNull():
            return pixmap
        tw = int(target_size.width())
        th = int(target_size.height())
        if tw <= 0 or th <= 0:
            return pixmap

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        sw = scaled.width()
        sh = scaled.height()
        if sw == tw and sh == th:
            return scaled

        x = max(0, (sw - tw) // 2)
        y = max(0, (sh - th) // 2)
        return scaled.copy(x, y, tw, th)

    def get_scroll_anchor_path(self) -> Optional[str]:
        """Return the file path at the top of the current viewport (for scroll restore)."""
        items = self._gallery_layout_items
        if not items:
            return None
        p = self._scroll_area
        if p is None:
            p = self.parent()
            while p and not isinstance(p, QScrollArea):
                p = p.parent()
        if not isinstance(p, QScrollArea):
            return None
        anchor_y = p.verticalScrollBar().value() + 12
        low, high = 0, len(items) - 1
        idx = 0
        while low <= high:
            mid = (low + high) // 2
            if items[mid]["rect"].bottom() < anchor_y:
                low = mid + 1
            else:
                idx = mid
                high = mid - 1
        if idx < 0 or idx >= len(items):
            return None
        return items[idx].get("file_path")

    def _capture_scroll_anchor_state(self) -> Optional[dict]:
        """Capture the upper-left visible thumbnail before layout rebuild."""
        scroll = self._scroll_area_widget()
        items = self._gallery_layout_items
        if scroll is None or not items:
            return None
        sb = scroll.verticalScrollBar()
        scroll_y = sb.value()
        idx = self._get_upper_left_visible_image_index()
        if idx < 0:
            idx = self._get_first_visible_image_index()
        if idx < 0 or idx >= len(items):
            return None
        item = items[idx]
        file_path = item.get("file_path")
        if not file_path:
            return None
        rect = item["rect"]
        return {
            "file_path": file_path,
            "offset_px": int(rect.top() - scroll_y),
        }

    def _zoom_scroll_anchor_for_rebuild(self) -> Optional[dict]:
        """Map the zoom-session anchor to build_gallery scroll-restore format."""
        state = self._zoom_anchor_state
        if not state or not state.get("file_path"):
            return None
        offset = state.get("offset_px")
        if offset is None:
            offset = state.get("offset_y_px", 0)
        return {"file_path": state["file_path"], "offset_px": int(offset)}

    def _restore_scroll_anchor_state(self, state: Optional[dict]) -> bool:
        """Restore scroll so the topmost visible tile stays at the same viewport offset."""
        if not state:
            return False
        path = state.get("file_path")
        if not path:
            return False
        if not self._layout_image_sequence or len(self._layout_image_sequence) != len(self.images):
            return False

        resolved = self._resolve_gallery_path(path)
        if not resolved:
            return False
        indices = self._path_to_indices.get(resolved)
        if not indices:
            return False
        idx = indices[0]
        if idx < 0 or idx >= len(self._gallery_layout_items):
            return False

        scroll = self._scroll_area_widget()
        if scroll is None:
            return False
        self._refresh_gallery_scroll_range()
        sb = scroll.verticalScrollBar()
        expected_max = max(0, int(self._total_content_height - scroll.viewport().height()))

        rect = self._gallery_layout_items[idx]["rect"]
        if "offset_px" in state:
            offset_px = int(state["offset_px"])
        else:
            # Legacy center-anchor payloads (pre top-item fix).
            tile_h = max(1, rect.height())
            offset_in_item = float(state.get("offset_in_item", 0.0))
            viewport_fraction = float(state.get("viewport_fraction", 0.0))
            viewport_h = max(1, scroll.viewport().height())
            anchor_y = rect.top() + max(0.0, min(1.0, offset_in_item)) * tile_h
            target = int(
                anchor_y - max(0.0, min(1.0, viewport_fraction)) * viewport_h
            )
            upper = max(sb.maximum(), expected_max)
            sb.setValue(max(sb.minimum(), min(target, upper)))
            return True

        target = int(rect.top() - offset_px)
        upper = max(sb.maximum(), expected_max)
        sb.setValue(max(sb.minimum(), min(target, upper)))
        return True

    def _get_upper_left_visible_image_index(self) -> int:
        """Index of the top-left visible thumbnail (viewport anchor for zoom)."""
        scroll = self._scroll_area_widget()
        items = self._gallery_layout_items
        if scroll is None or not items:
            return -1
        sb = scroll.verticalScrollBar()
        scroll_y = sb.value()
        viewport_bottom = scroll_y + max(1, scroll.viewport().height())

        best_idx = -1
        best_top: int | None = None
        best_left: int | None = None
        for i, item in enumerate(items):
            rect = item.get("rect")
            if rect is None or not item.get("file_path"):
                continue
            if rect.bottom() <= scroll_y or rect.top() >= viewport_bottom:
                continue
            top = int(rect.top())
            left = int(rect.left())
            if (
                best_idx < 0
                or top < best_top
                or (top == best_top and left < best_left)
            ):
                best_idx = i
                best_top = top
                best_left = left
        return best_idx

    def _get_first_visible_image_index(self) -> int:
        idx = self._get_upper_left_visible_image_index()
        if idx >= 0:
            return idx
        scroll = self._scroll_area_widget()
        if scroll is None:
            return -1
        scroll_y = scroll.verticalScrollBar().value()
        for i, item in enumerate(self._gallery_layout_items):
            if item["rect"].bottom() > scroll_y:
                return i
        return -1

    def begin_zoom_session(self) -> bool:
        """Begin interactive zoom; capture upper-left anchor for scroll restore."""
        if not self._gallery_layout_items:
            return False
        scroll = self._scroll_area_widget()
        scroll_y = (
            int(scroll.verticalScrollBar().value()) if scroll is not None else 0
        )
        idx = self._get_upper_left_visible_image_index()
        if idx < 0:
            idx = self._get_first_visible_image_index()

        self._is_zooming = True
        if 0 <= idx < len(self._gallery_layout_items):
            item = self._gallery_layout_items[idx]
            rect = item["rect"]
            self._zoom_anchor_state = {
                "file_path": item.get("file_path"),
                "offset_y_px": int(rect.top() - scroll_y),
            }
        else:
            self._zoom_anchor_state = self._capture_scroll_anchor_state()

        self._stop_layout_rebuild_timers()
        self.hide_loading_message()
        return True

    def apply_gallery_zoom(self, row_height: int) -> None:
        """Justified relayout at the new row height; keep the zoom anchor fixed vertically."""
        row_height = max(220, min(500, int(row_height)))
        if not self._is_zooming:
            self.begin_zoom_session()
        self.TARGET_ROW_HEIGHT = row_height
        self._gallery_zoom_rebuild = True
        try:
            self.build_gallery(force=True)
        finally:
            self._gallery_zoom_rebuild = False
        self._request_load_visible_images(0)

    def end_zoom_session(self) -> None:
        """Commit zoom session; defer any missing thumbs until after slider release."""
        if not self._is_zooming:
            return
        self._is_zooming = False
        self._zoom_anchor_state = None
        self._last_layout_content_height = self._total_content_height
        self._last_build_ts = time.time()
        self._request_load_visible_images(30)

    def _current_image_layout_index(self) -> Optional[int]:
        """Layout index for the viewer's current single-view / navigation file."""
        pv = getattr(self, "parent_viewer", None)
        if pv is None:
            return None
        current = getattr(pv, "current_file_path", None)
        if not current or not self._gallery_layout_items:
            return None
        resolved = self._resolve_gallery_path(current)
        if not resolved:
            return None
        indices = self._path_to_indices.get(resolved)
        if not indices:
            return None
        return indices[0]

    def _layout_index_at_scroll_y(self, anchor_y: int) -> Optional[int]:
        items = self._gallery_layout_items
        if not items:
            return None
        low, high = 0, len(items) - 1
        idx = 0
        while low <= high:
            mid = (low + high) // 2
            if items[mid]["rect"].bottom() < anchor_y:
                low = mid + 1
            else:
                idx = mid
                high = mid - 1
        if idx < 0 or idx >= len(items):
            return None
        return idx

    def _gallery_prefetch_center_index(self, scroll_y: int, viewport_h: int) -> Optional[int]:
        """Center for bidirectional JPEG prefetch: current image, else viewport center."""
        current_idx = self._current_image_layout_index()
        viewport_idx = self._layout_index_at_scroll_y(scroll_y + max(1, viewport_h) // 2)
        if current_idx is None:
            return viewport_idx
        if getattr(self, "_pending_scroll_to_path", None):
            return current_idx
        pending = self._entry_anchor_path()
        if (
            pending
            and not getattr(self, "_entry_prefetch_abandoned", False)
            and self._entry_prefetch_active(current_idx, scroll_y, viewport_h)
        ):
            return current_idx
        if viewport_idx is None:
            return current_idx
        if abs(current_idx - viewport_idx) <= self._gallery_center_prefetch_radius():
            return current_idx
        return viewport_idx

    def _entry_anchor_path(self) -> Optional[str]:
        """File path used to restore scroll when entering gallery."""
        pending = getattr(self, "_pending_scroll_to_path", None)
        if pending:
            return pending
        pv = getattr(self, "parent_viewer", None)
        if pv is not None:
            current = getattr(pv, "current_file_path", None)
            if current:
                return current
            target = getattr(pv, "_gallery_scroll_target_path", None)
            if target:
                return target
        return None

    def _entry_anchor_layout_index(self) -> Optional[int]:
        if not self._gallery_layout_items:
            return None
        path = self._entry_anchor_path()
        if not path:
            return None
        resolved = self._resolve_gallery_path(path)
        if not resolved:
            return None
        indices = self._path_to_indices.get(resolved)
        if not indices:
            return None
        return indices[0]

    def _layout_indices_near(self, center_idx: int, radius: int) -> List[int]:
        n = len(self._gallery_layout_items)
        if n <= 0:
            return []
        center_idx = max(0, min(int(center_idx), n - 1))
        ordered: List[int] = []
        seen: set[int] = set()
        for offset in range(int(radius) + 1):
            for idx in (center_idx + offset, center_idx - offset):
                if 0 <= idx < n and idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
        return ordered

    def _gallery_center_prefetch_radius(self) -> int:
        return _env_int("RAWVIEWER_GALLERY_ENTRY_PREFETCH_RADIUS", 48, minimum=4)

    def _entry_prefetch_radius(self) -> int:
        return self._gallery_center_prefetch_radius()

    def _entry_prefetch_active(
        self, anchor_idx: Optional[int], scroll_y: int, viewport_h: int
    ) -> bool:
        """True when scroll has not yet reached the entry thumbnail neighborhood."""
        if anchor_idx is None:
            return False
        if getattr(self, "_pending_scroll_to_path", None):
            return True
        if anchor_idx < 0 or anchor_idx >= len(self._gallery_layout_items):
            return False
        rect = self._gallery_layout_items[anchor_idx]["rect"]
        margin = max(viewport_h // 3, 80)
        view_top = scroll_y - margin
        view_bottom = scroll_y + viewport_h + margin
        return rect.bottom() < view_top or rect.top() > view_bottom

    def _prioritize_entry_paths(self, paths: List[str], cap: Optional[int] = None) -> List[str]:
        """Order paths so EXIF/layout work near the entry file runs first."""
        if not paths:
            return []
        anchor_path = self._entry_anchor_path()
        if not anchor_path or not self.images:
            return paths[:cap] if cap else list(paths)
        resolved = self._resolve_gallery_path(anchor_path)
        if not resolved:
            return paths[:cap] if cap else list(paths)
        try:
            anchor_list_idx = self.images.index(resolved)
        except ValueError:
            return paths[:cap] if cap else list(paths)

        path_set = set(paths)
        seed_radius = min(max(self.BUILD_EXIF_PREFETCH_MAX // 2, 16), 96)
        ordered: List[str] = []
        seen: set[str] = set()
        for offset in range(seed_radius + 1):
            for idx in (anchor_list_idx + offset, anchor_list_idx - offset):
                if 0 <= idx < len(self.images):
                    p = self.images[idx]
                    if p in path_set and p not in seen:
                        seen.add(p)
                        ordered.append(p)
        for p in paths:
            if p not in seen:
                ordered.append(p)
        if cap is not None:
            return ordered[:cap]
        return ordered

    def _entry_prefetch_paths(self, anchor_idx: int) -> set[str]:
        return self._bidirectional_jpeg_paths_from_center(anchor_idx)

    def _bidirectional_jpeg_paths_from_center(self, center_idx: int) -> set[str]:
        """Paths within ±radius layout indices of center (up/down in the gallery grid)."""
        paths: set[str] = set()
        for idx in self._layout_indices_near(center_idx, self._gallery_center_prefetch_radius()):
            path = self._gallery_layout_items[idx].get("file_path")
            if path:
                paths.add(path)
        return paths

    def _missing_thumbnail_stage(self, path: str) -> bool:
        return not self._thumbnail_cache.get((path, self._thumb_base_key))

    def _missing_load_stages_for_path(
        self, path: str, *, jpeg_only: bool = False
    ) -> set[str]:
        stages: set[str] = set()
        if self._missing_thumbnail_stage(path):
            stages.add("thumbnail")
        if jpeg_only:
            return stages
        meta = self._metadata_cache.get(path)
        if not meta or meta.get("original_width") is None or meta.get("original_height") is None:
            stages.add("exif")
        return stages

    def _path_layout_distance(self, path: str, center_idx: Optional[int]) -> int:
        if center_idx is None or not path:
            return 1_000_000
        indices = self._path_to_indices.get(path)
        if not indices:
            return 1_000_000
        return min(abs(i - center_idx) for i in indices)

    def set_images(
        self,
        images: List[str],
        bulk_metadata: Optional[Dict[str, Any]] = None,
        *,
        force_rebuild: bool = False,
    ):
        """
        Public API used by `main.py` to populate/refresh the gallery.
        Kept compatible with the legacy gallery interface.
        """
        try:
            if self.load_manager is None:
                # Ensure worker wiring exists before first render pass.
                self._post_init()
            new_images = list(images or [])
            self._gallery_folder_token = self._current_folder_load_token()
            if new_images:
                self.hide_empty_message()
            same_order = (
                len(new_images) == len(self.images)
                and all(a == b for a, b in zip(new_images, self.images))
            )
            if _focus_gallery_switch_logs():
                logger.debug(
                    "[MODESWITCH] gallery.set_images called; count=%d load_manager=%s force=%s same_order=%s",
                    len(new_images),
                    self.load_manager is not None,
                    force_rebuild,
                    same_order,
                )
            if not force_rebuild and same_order and self._layout_matches_current_images(new_images):
                if bulk_metadata:
                    self._metadata_cache.update(bulk_metadata)

                # Fetch updated metadata from cache for paths that are missing it or have minimal metadata
                if self.parent_viewer and hasattr(self.parent_viewer, "image_cache"):
                    paths = [img for img in self.images if isinstance(img, str)]
                    missing_paths = [p for p in paths if p not in self._metadata_cache or self._metadata_cache[p].get("minimal_preview_exif")]
                    if missing_paths:
                        fetch_paths = self._prioritize_entry_paths(missing_paths)
                        bulk_fetched = self.parent_viewer.image_cache.get_multiple_exif(fetch_paths)
                        if bulk_fetched:
                            self._metadata_cache.update(bulk_fetched)
                            # If any aspect ratios/rotations changed, trigger a layout rebuild
                            layout_changed = False
                            for path, exif in bulk_fetched.items():
                                if self.on_visual_rotation_changed(path, defer_rebuild=True):
                                    layout_changed = True
                            if layout_changed:
                                self.build_gallery(force=True)
                                return

                self._reposition_thumbnail_widgets()
                if self._pending_scroll_to_path:
                    QTimer.singleShot(0, self._apply_pending_scroll_to_file)
                else:
                    self._request_load_visible_images(20)
                return

            self._gallery_generation += 1
            self.images = new_images
            self._rebuild_normalized_map()
            if not same_order or force_rebuild:
                self._reset_gallery_scroll_position()
            if not same_order or force_rebuild:
                # Order changed: invalidate layout so build_gallery cannot skip via count-only checks.
                self._building = False
                self._pending_gallery_build = False
                self._pending_build_metadata = None
                self._pending_build_force = False
                self.clear_thumbnail_widgets()
                self._gallery_layout_items.clear()
                self._path_to_indices.clear()
                self._layout_image_sequence = ()
                self._last_layout_content_height = 0
            self._active_tasks.clear()
            self._gallery_set_images_ts = time.time()
            self._first_thumb_ready_after_set = False

            if bulk_metadata:
                self._metadata_cache.update(bulk_metadata)

            # Cancel/clear outstanding thumbnail work for previous generation
            try:
                for fp in list(self._requested_thumbnail_paths):
                    try:
                        if self.load_manager is not None:
                            self.load_manager.cancel_task(fp)
                    except Exception:
                        pass
            finally:
                self._requested_thumbnail_paths.clear()

            if not self.images:
                self._building = False
                self._pending_gallery_build = False
                self._pending_build_metadata = None
                self._pending_build_force = False
                self.clear_thumbnail_widgets(destroy=True)
                self._gallery_layout_items.clear()
                self._path_to_indices.clear()
                parent_scroll = self._scroll_area
                if parent_scroll is None:
                    parent_scroll = self.parent()
                    while parent_scroll and not isinstance(parent_scroll, QScrollArea):
                        parent_scroll = parent_scroll.parent()
                viewport_h = parent_scroll.viewport().height() if isinstance(parent_scroll, QScrollArea) else 0
                self._total_content_height = max(1, int(viewport_h))
                self.setMinimumHeight(self._total_content_height)
                self.resize(max(self.width(), self._get_viewport_width()), self._total_content_height)
                if isinstance(parent_scroll, QScrollArea):
                    parent_scroll.verticalScrollBar().setValue(parent_scroll.verticalScrollBar().minimum())
                self.update()
                return

            # Rebuild on next event loop tick (avoid doing heavy UI work inside caller)
            build_gen = self._gallery_generation
            need_force = force_rebuild or not same_order
            build_done = {"gen": -1}

            def _run_build():
                if self._gallery_generation != build_gen:
                    return
                if build_done["gen"] == build_gen:
                    return
                if self._building:
                    return
                if not need_force and self._layout_matches_current_images():
                    build_done["gen"] = build_gen
                    self._reposition_thumbnail_widgets()
                    self._request_load_visible_images(20)
                    return
                self.build_gallery(bulk_metadata=bulk_metadata, force=need_force)
                build_done["gen"] = build_gen

            QTimer.singleShot(0, _run_build)
            # Backup: mode-switch bursts can delay the first tick on large folders.
            QTimer.singleShot(200, _run_build)
        except Exception:
            logger.exception("[GALLERY] set_images() failed")

    def _rebuild_normalized_map(self):
        """Rebuild the O(1) normalized to canonical path mapping."""
        self._normalized_to_canonical = {}
        for img in self.images:
            if isinstance(img, str):
                try:
                    norm = os.path.normcase(os.path.abspath(img))
                except OSError:
                    norm = os.path.normcase(img)
                self._normalized_to_canonical[norm] = img

    def _resolve_gallery_path(self, file_path: Optional[str]) -> Optional[str]:
        """Match file_path to the canonical path string used in ``self.images``."""
        if not file_path or not self.images:
            return None
        try:
            target = os.path.normcase(os.path.abspath(file_path))
        except OSError:
            target = os.path.normcase(file_path)
            
        if hasattr(self, "_normalized_to_canonical") and target in self._normalized_to_canonical:
            return self._normalized_to_canonical[target]
            
        target_base = os.path.normcase(os.path.basename(file_path))
        for img in self.images:
            if not isinstance(img, str):
                continue
            try:
                if os.path.normcase(os.path.abspath(img)) == target:
                    return img
            except OSError:
                pass
            if os.path.normcase(os.path.basename(img)) == target_base:
                return img
        return None

    def scroll_to_file(self, file_path: Optional[str]):
        """Scroll so file_path is visible after the justified layout is built."""
        if not file_path:
            return
        self._pending_scroll_to_path = file_path
        self._pending_scroll_anchor_state = {
            "file_path": file_path,
            "offset_px": 0,
        }
        self._scroll_to_file_attempts = 0
        self._schedule_scroll_to_file_retry(0)

    def _schedule_scroll_to_file_retry(self, delay_ms: int) -> None:
        QTimer.singleShot(max(0, int(delay_ms)), self._apply_pending_scroll_to_file)

    def _apply_pending_scroll_to_file(self):
        anchor_state = getattr(self, "_pending_scroll_anchor_state", None)
        path = self._pending_scroll_to_path
        if not anchor_state and not path:
            return

        # Guard 1: Verify the justified layout is fully built/updated for the current images
        if not self._layout_image_sequence or len(self._layout_image_sequence) != len(self.images):
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            return

        if anchor_state:
            if self._restore_scroll_anchor_state(anchor_state):
                self._pending_scroll_to_path = None
                self._pending_scroll_anchor_state = None
                self._scroll_to_file_attempts = 0
                self._request_load_visible_images(20)
                return
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            return

        if not path:
            return

        resolved = self._resolve_gallery_path(path)
        if not resolved:
            pv = getattr(self, "parent_viewer", None)
            if pv is not None and hasattr(pv, "_gallery_viewport_scroll_path"):
                try:
                    alt = pv._gallery_viewport_scroll_path()
                    if alt:
                        resolved = self._resolve_gallery_path(alt)
                except Exception:
                    pass
        if not resolved:
            pv = getattr(self, "parent_viewer", None)
            cfi = getattr(pv, "current_file_index", -1) if pv is not None else -1
            images = self.images or []
            if 0 <= cfi < len(images):
                resolved = images[cfi]
            elif pv is not None:
                resolved = self._resolve_gallery_path(
                    getattr(pv, "current_file_path", None)
                )
        if not resolved:
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            elif attempts == 40:
                logger.warning(
                    "[GALLERY] scroll_to_file gave up: %s not in layout (%d images)",
                    os.path.basename(path) if path else "?",
                    len(self.images or []),
                )
            return
        indices = self._path_to_indices.get(resolved)
        if not indices:
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            return
        idx = indices[0]
        if idx < 0 or idx >= len(self._gallery_layout_items):
            return
        p = self._scroll_area
        if p is None:
            p = self.parent()
            while p and not isinstance(p, QScrollArea):
                p = p.parent()
        if not isinstance(p, QScrollArea):
            return

        # Guard 2: Verify the scrollbar's maximum range has updated to match content_h
        sb = p.verticalScrollBar()
        expected_max = self._total_content_height - p.viewport().height()
        if expected_max > 0 and sb.maximum() < expected_max - 50:
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            return

        rect = self._gallery_layout_items[idx]["rect"]
        target = max(sb.minimum(), min(rect.top(), sb.maximum()))
        sb.setValue(target)
        self._pending_scroll_to_path = None
        self._pending_scroll_anchor_state = None
        self._scroll_to_file_attempts = 0
        self._request_load_visible_images(20)

    def _scaled_cache_key(self, file_path: str, target_size):
        """Cache key for crop-to-fit thumbnails at exact tile physical size."""
        th = int(target_size.height())
        tw = int(target_size.width())
        rot = self._get_rotation_degrees_for_path(file_path)
        if th <= 0 or tw <= 0:
            return (file_path, self._thumb_base_key, rot)
        return (file_path, tw, th, rot)

    @staticmethod
    def _pixmap_logical_size(pixmap: QPixmap, dpr: float) -> QSize:
        safe_dpr = dpr if dpr > 0 else 1.0
        return QSize(
            int(round(pixmap.width() / safe_dpr)),
            int(round(pixmap.height() / safe_dpr)),
        )

    def _gallery_base_meets_tile(self, base: QPixmap, physical_size: QSize) -> bool:
        """True when cached base pixmap is sharp enough for this tile."""
        if base is None or base.isNull():
            return False
        try:
            from image_load_manager import _gallery_grid_min_dim

            dpr = float(base.devicePixelRatio() or 1.0)
            effective = max(int(base.width() * dpr), int(base.height() * dpr))
            need = max(int(physical_size.width()), int(physical_size.height()))
            floor = min(_gallery_grid_min_dim(), int(need * 0.85))
            return effective >= floor
        except Exception:
            return True

    def _fit_tile_pixmap(
        self, file_path: str, base: QPixmap, physical_size: QSize, dpr: float
    ) -> QPixmap:
        fitted = self._fit_rotated_thumbnail(file_path, base, physical_size)
        fitted.setDevicePixelRatio(dpr)
        self._thumbnail_cache.put(self._scaled_cache_key(file_path, physical_size), fitted)
        return fitted

    def _get_rotation_degrees_for_path(self, file_path: str) -> int:
        """Read per-file visual rotation from parent viewer (0/90/180/270)."""
        pv = getattr(self, "parent_viewer", None)
        if pv is None:
            return 0
        getter = getattr(pv, "_get_visual_rotation_degrees", None)
        if getter is None:
            return 0
        try:
            return int(getter(file_path)) % 360
        except Exception:
            return 0

    def _display_aspect(self, file_path: str, base_aspect: float) -> float:
        """Aspect ratio after non-destructive visual rotation."""
        if base_aspect <= 0:
            return base_aspect
        if self._get_rotation_degrees_for_path(file_path) in (90, 270):
            return 1.0 / base_aspect
        return base_aspect

    def _metadata_display_aspect(self, file_path: str) -> Optional[float]:
        """Layout aspect from EXIF display dimensions (matches index thumbnail pixels)."""
        m = self._metadata_exif_for_path(file_path)
        if not m or not m.get("original_width") or not m.get("original_height"):
            return None
        from common_image_loader import exif_display_dimensions

        w, h = int(m["original_width"]), int(m["original_height"])
        o = int(m.get("orientation", 1) or 1)
        dw, dh = exif_display_dimensions(w, h, o)
        if dh <= 0:
            return None
        return self._display_aspect(file_path, dw / dh)

    @staticmethod
    def _aspect_orientation_flag(ar: float) -> int:
        """-1 portrait, 0 near-square, 1 landscape."""
        if ar <= 0:
            return 0
        if ar < 0.92:
            return -1
        if ar > 1.08:
            return 1
        return 0

    def _raw_aspect_undo_user_rotation(self, file_path: str, display_ar: float) -> float:
        if display_ar <= 0:
            return display_ar
        if self._get_rotation_degrees_for_path(file_path) in (90, 270):
            return 1.0 / display_ar
        return display_ar

    def _raw_aspect_from_pixmap(self, file_path: str, pixmap: QPixmap) -> Optional[float]:
        """Measured width/height for layout, reconciled with EXIF when orientation disagrees."""
        if pixmap is None or pixmap.isNull() or pixmap.height() <= 0:
            return None
        px_ar = pixmap.width() / pixmap.height()
        meta_ar = self._metadata_display_aspect(file_path)
        if meta_ar is not None and meta_ar > 0:
            px_o = self._aspect_orientation_flag(px_ar)
            meta_o = self._aspect_orientation_flag(meta_ar)
            if px_o != 0 and meta_o != 0 and px_o != meta_o:
                return self._raw_aspect_undo_user_rotation(file_path, meta_ar)
            if abs(px_ar - meta_ar) / max(meta_ar, 0.01) > 0.35:
                return self._raw_aspect_undo_user_rotation(file_path, meta_ar)
        return px_ar

    def _orient_gallery_thumbnail_array(self, file_path: str, arr: np.ndarray) -> np.ndarray:
        try:
            from common_image_loader import finalize_index_thumbnail_array
            from image_cache import get_image_cache

            fixed = finalize_index_thumbnail_array(file_path, arr, cache=get_image_cache())
            return fixed if fixed is not None else arr
        except Exception:
            return arr

    def _metadata_exif_for_path(self, file_path: str) -> dict:
        """EXIF dict for layout/orientation — fall back to global cache when local cache is cold."""
        m = self._metadata_cache.get(file_path)
        if m and m.get("original_width") and m.get("original_height"):
            return m
        if file_path in self._metadata_fetch_attempted:
            # build_gallery() already bulk-fetched this folder's EXIF once via
            # get_multiple_exif(); if it's still missing here there's nothing new
            # in the global cache to find. Falling through to get_exif() below would
            # re-run a per-path SQLite SELECT for every still-missing file on every
            # rebuild — with thousands of paths on a large folder, and the background
            # indexer concurrently writing the same DB, that turned build_gallery's
            # per-item layout loop into multi-second stalls (measured 7-13s for 6881
            # items). Wait for the metadata-ready event to populate _metadata_cache.
            return m or {}
        try:
            from image_cache import get_image_cache

            cached = get_image_cache().get_exif(file_path)
            if cached:
                return cached
        except Exception:
            pass
        return m or {}

    def _orient_gallery_base_pixmap(self, file_path: str, pixmap: QPixmap) -> QPixmap:
        """Rotate base thumbnail to match container EXIF display orientation."""
        if pixmap is None or pixmap.isNull():
            return pixmap
        m = self._metadata_exif_for_path(file_path)
        if not m:
            if _orient_debug_enabled():
                logger.info(
                    "[ORIENT] base file=%s pix=%dx%d(%s) EXIF=MISSING -> no rotation",
                    os.path.basename(file_path), pixmap.width(), pixmap.height(),
                    "P" if pixmap.height() > pixmap.width() else "L",
                )
            return pixmap
        from common_image_loader import apply_container_orientation_to_pixmap

        if _orient_debug_enabled():
            from common_image_loader import (
                exif_rotation_degrees_for_pixmap,
                exif_pixels_display_oriented,
                pixmap_matches_exif_display,
            )
            ow = int(m.get("original_width") or 0)
            oh = int(m.get("original_height") or 0)
            o = int(m.get("orientation", 1) or 1)
            _deg = exif_rotation_degrees_for_pixmap(pixmap.width(), pixmap.height(), ow, oh, o)
            _match = pixmap_matches_exif_display(
                pixmap.width(), pixmap.height(), ow, oh, o,
                pixels_display_oriented=exif_pixels_display_oriented(m),
            )
            logger.info(
                "[ORIENT] base file=%s pix=%dx%d(%s) exif(o=%s ow=%d oh=%d) match=%s deg=%s",
                os.path.basename(file_path), pixmap.width(), pixmap.height(),
                "P" if pixmap.height() > pixmap.width() else "L", o, ow, oh, _match, _deg,
            )
        return apply_container_orientation_to_pixmap(pixmap, m)

    def _store_oriented_base_pixmap(self, file_path: str, pixmap: QPixmap) -> QPixmap:
        """Orient, cache base thumbnail, and drop stale scaled variants."""
        oriented = self._orient_gallery_base_pixmap(file_path, pixmap)
        if oriented is None or oriented.isNull():
            return pixmap
        self._thumbnail_cache.put((file_path, self._thumb_base_key), oriented)
        self._invalidate_scaled_thumbnails_for_path(file_path)
        return oriented

    def _global_cache_to_base_pixmap(self, path: str) -> Optional[QPixmap]:
        """Pull an oriented base pixmap from the global ImageCache for gallery tiles."""
        try:
            from image_cache import get_image_cache

            global_cache = get_image_cache()
            global_thumb = global_cache.get_grid(path)
            if global_thumb is None:
                global_thumb = global_cache.get_thumbnail(path)
            if global_thumb is None:
                return None
            arr = self._orient_gallery_thumbnail_array(path, np.ascontiguousarray(global_thumb))
            return _thumbnail_data_to_base_pixmap(arr)
        except Exception:
            return None

    def _layout_aspect_for_path(
        self, file_path: str, pixmap: Optional[QPixmap] = None
    ) -> float:
        """Width/height for justified rows — measured pixels are ground truth once known.

        Like the filmstrip's _measured_widths, we remember the decoded thumbnail's raw
        aspect per path (self._measured_raw_aspects) and prefer it over EXIF. This makes
        tile geometry immune to stale/miscomputed metadata AND to the pixmap being evicted
        from _thumbnail_cache before build_gallery recomputes the aspect — which was how a
        portrait image could end up crop-fit into a landscape tile. User rotation is applied
        at read time via _display_aspect, so a stored raw aspect stays correct across rotates.
        """
        # PIXELS-FIRST (restores the pre-3dcf820 _base_aspect_for_path behaviour). The
        # decoded thumbnail's own dimensions are the ground truth — exactly what the
        # filmstrip still does. EXIF is used ONLY as a placeholder before the thumbnail is
        # decoded; once we've ever measured a pixmap we keep using that (remembered in
        # _measured_raw_aspects) so a rebuild / cache eviction can't fall back to stale
        # metadata and crop-fit a portrait image into a landscape tile. 3dcf820 had
        # inverted this to prefer EXIF, which is the regression behind that bug.
        raw_ar = None
        base = pixmap
        if base is None:
            base = self._thumbnail_cache.get((file_path, self._thumb_base_key))
        min_layout_px = _min_layout_pixmap_dim()
        if base and not base.isNull() and base.height() > 0:
            if max(base.width(), base.height()) >= min_layout_px:
                raw_ar = self._raw_aspect_from_pixmap(file_path, base)
                if raw_ar and raw_ar > 0:
                    self._measured_raw_aspects[file_path] = raw_ar
        else:
            stored = self._measured_raw_aspects.get(file_path)
            if stored and stored > 0:
                raw_ar = stored

        if raw_ar and raw_ar > 0:
            meta_ar = self._metadata_display_aspect(file_path)
            if meta_ar is not None and meta_ar > 0:
                raw_o = self._aspect_orientation_flag(raw_ar)
                meta_o = self._aspect_orientation_flag(meta_ar)
                if raw_o != 0 and meta_o != 0 and raw_o != meta_o:
                    raw_ar = self._raw_aspect_undo_user_rotation(file_path, meta_ar)
            return self._display_aspect(file_path, raw_ar)
        meta_ar = self._metadata_display_aspect(file_path)
        if meta_ar is not None:
            return meta_ar
        return 1.5

    @staticmethod
    def _layout_aspect_needs_rebuild(old_aspect: float, new_aspect: float) -> bool:
        """True when tile geometry must change (including portrait ↔ landscape flip)."""
        old_o = JustifiedGallery._aspect_orientation_flag(old_aspect)
        new_o = JustifiedGallery._aspect_orientation_flag(new_aspect)
        if old_o != 0 and new_o != 0 and old_o != new_o:
            return True
        return abs(old_aspect - new_aspect) > 0.05

    def _frame_matches_pixmap_orientation(
        self, rect, pixmap: QPixmap
    ) -> bool:
        """True when layout rect and pixmap agree on portrait vs landscape."""
        if rect is None or pixmap is None or pixmap.isNull():
            return True
        if rect.width() <= 0 or rect.height() <= 0 or pixmap.width() <= 0 or pixmap.height() <= 0:
            return True
        r_ar = rect.width() / rect.height()
        p_ar = pixmap.width() / pixmap.height()
        r_o = self._aspect_orientation_flag(r_ar)
        p_o = self._aspect_orientation_flag(p_ar)
        if r_o == 0 or p_o == 0:
            return True
        return r_o == p_o

    def _any_visible_tile_frame_mismatch(
        self, file_path: str, pixmap: QPixmap
    ) -> bool:
        for idx in self._path_to_indices.get(file_path, []):
            if idx not in self._visible_widgets:
                continue
            if idx >= len(self._gallery_layout_items):
                continue
            rect = self._gallery_layout_items[idx].get("rect")
            if not self._frame_matches_pixmap_orientation(rect, pixmap):
                return True
        return False

    def _base_aspect_for_path(self, file_path: str) -> float:
        return self._layout_aspect_for_path(file_path)

    def _reconcile_tile_aspect(self, file_path: str, pixmap: Optional[QPixmap]) -> bool:
        """Make tile geometry follow the oriented thumbnail pixels (like the filmstrip).

        Any code path that materializes a decoded base pixmap for a tile must call this,
        not just the on_thumbnail_ready signal: load_visible_images can pull an already-
        oriented thumbnail straight from the global cache (filmstrip/semantic/prior-visit
        warmed it) and crop-fit it into the rect built from stale metadata, leaving a
        portrait image in a landscape frame. Updates the layout-item aspect and arms the
        debounced rebuild (which self-defers while fast-scrolling). Returns True if a
        rebuild was scheduled.
        """
        if pixmap is None or pixmap.isNull() or pixmap.width() <= 0 or pixmap.height() <= 0:
            return False
        aspect = self._layout_aspect_for_path(file_path, pixmap)
        needs_layout_rebuild = False
        orientation_flipped = False
        pre_aspects: dict[int, float] = {}
        for idx in self._path_to_indices.get(file_path, []):
            if idx < len(self._gallery_layout_items):
                pre_aspects[idx] = self._gallery_layout_items[idx].get("aspect", 1.5)
        for idx, old_aspect in pre_aspects.items():
            if abs(old_aspect - aspect) > 0.05:
                if self._layout_aspect_needs_rebuild(old_aspect, aspect):
                    needs_layout_rebuild = True
                    old_o = self._aspect_orientation_flag(old_aspect)
                    new_o = self._aspect_orientation_flag(aspect)
                    if old_o != 0 and new_o != 0 and old_o != new_o:
                        orientation_flipped = True
                self._gallery_layout_items[idx]["aspect"] = aspect
        if needs_layout_rebuild:
            self._add_metadata_changed_path(file_path)
            if orientation_flipped:
                self._orientation_flip_paths.add(file_path)
            if not self._metadata_rebuild_timer.isActive():
                if orientation_flipped and not self._is_scrolling_fast:
                    self._metadata_rebuild_timer.start(0)
                elif orientation_flipped:
                    self._metadata_rebuild_timer.start(250)
                else:
                    self._metadata_rebuild_timer.start(500)
        return needs_layout_rebuild

    def _fit_rotated_thumbnail(self, file_path: str, base: QPixmap, target_size):
        """Apply user rotation then crop-to-fit (tile aspect matches after layout rebuild)."""
        rot = self._get_rotation_degrees_for_path(file_path)
        oriented = _rotate_pixmap_cw(base, rot)
        return self._scale_crop_to_fit(oriented, target_size)

    def invalidate_thumbnails_for_path(self, file_path: str) -> None:
        """Drop cached gallery pixmaps for a path (e.g. after on-disk rotation)."""
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        try:
            self._thumbnail_cache.remove_keys_for_file_path(file_path)
        except Exception:
            pass
        # Drop the remembered aspect so a re-decode (e.g. after on-disk rotation) re-measures.
        self._measured_raw_aspects.pop(file_path, None)

    def _invalidate_scaled_thumbnails_for_path(self, file_path: str) -> None:
        """Drop only scaled variants for a path; keep base thumbnail to speed immediate redraw."""
        try:
            with self._thumbnail_cache.lock:
                for k in list(self._thumbnail_cache.cache.keys()):
                    if (
                        isinstance(k, tuple)
                        and len(k) >= 2
                        and k[0] == file_path
                        and not (len(k) >= 2 and k[1] == self._thumb_base_key)
                    ):
                        del self._thumbnail_cache.cache[k]
        except Exception:
            pass

    def refresh_visible_tile_for_path(self, file_path: str) -> None:
        """Immediately refresh visible widgets for a path after visual-rotation changes."""
        if not file_path:
            return
        resolved = self._resolve_gallery_path(file_path) or file_path
        self._invalidate_scaled_thumbnails_for_path(resolved)
        indices = self._path_to_indices.get(resolved, [])
        if not indices:
            return
        base = self._thumbnail_cache.get((resolved, self._thumb_base_key))
        dpr = self.devicePixelRatio()
        for idx in indices:
            if idx not in self._visible_widgets:
                continue
            w = self._visible_widgets[idx]
            if w.file_path != file_path:
                continue
            logical_size = w.size()
            physical_size = QSize(
                int(logical_size.width() * dpr), int(logical_size.height() * dpr)
            )
            if base and not base.isNull():
                fitted = self._fit_tile_pixmap(resolved, base, physical_size, dpr)
                w.setPixmap(fitted)
                w.setText("")
            elif self.load_manager is not None:
                # No base thumbnail yet: schedule an immediate fetch for this visible tile.
                self.load_manager.load_image(
                    resolved,
                    priority=Priority.CURRENT,
                    cancel_existing=False,
                    stages={"thumbnail"},
                    gallery_thumbnail=True,
                )

    def _active_task_started_ts(self, path: str) -> Optional[float]:
        meta = self._active_tasks.get(path)
        if meta is None:
            return None
        if isinstance(meta, tuple):
            return float(meta[0])
        return float(meta)

    def _active_task_priority(self, path: str) -> Priority:
        meta = self._active_tasks.get(path)
        if meta is None:
            return Priority.PRELOAD_NEXT
        if isinstance(meta, tuple) and len(meta) >= 2:
            return meta[1]
        return Priority.CURRENT

    def _prefetch_active_count(self) -> int:
        return sum(
            1
            for path in self._active_tasks
            if self._active_task_priority(path) != Priority.CURRENT
        )

    def _should_protect_active_raw_thumbnail(
        self, file_path: str, visible_paths: set[str]
    ) -> bool:
        """Keep in-flight visible RAW/DNG decodes when prefetch window moves away."""
        if file_path not in visible_paths or not is_raw_file(file_path):
            return False
        if file_path in self._active_tasks:
            return True
        lm = self.load_manager
        return lm is not None and lm.has_current_priority_work(file_path)

    def _mark_active_task(self, path: str, priority: Priority) -> None:
        self._active_tasks[path] = (time.time(), priority)

    def begin_gallery_warmup(self, file_count: int = 0) -> None:
        """Coalesce duplicate layout builds on gallery entry (no long load delay)."""
        self._entry_prefetch_abandoned = False
        # GL teardown is the real Windows crash fix; avoid multi-second warmup caps that
        # block first tile paint on large external folders.
        self._gallery_warmup_until = 0.0
        self._building = False
        self._gallery_entry_coalesce_until = time.time() + 0.5
        self._pending_gallery_build = False
        self._pending_build_metadata = None
        self._pending_build_force = False
        self._idle_preload_timer.stop()
        if _focus_gallery_switch_logs():
            logger.debug(
                "[MODESWITCH] gallery entry coalesce; files=%d",
                int(file_count or 0),
            )

    def _gallery_warmup_active(self) -> bool:
        return time.time() < float(getattr(self, "_gallery_warmup_until", 0.0) or 0.0)

    def _end_gallery_load_warmup_throttle(self) -> None:
        """Restore ImageLoadManager concurrency after gallery entry warmup."""
        pv = self.parent_viewer
        if pv is None or getattr(pv, "view_mode", "") != "gallery":
            return
        try:
            mgr = getattr(pv, "image_manager", None)
            if mgr is not None and hasattr(mgr, "exit_gallery_warmup_throttle"):
                mgr.exit_gallery_warmup_throttle()
                # Pump the queue: restoring maxThreadCount does not by itself start
                # queued tasks, and if nothing is active there is no completion event
                # to trigger a dispatch — the queue can sit frozen (observed as a
                # 20s+ gallery-entry stall with pending>0 and zero decodes).
                if hasattr(mgr, "_schedule_next_task"):
                    mgr._schedule_next_task()
        except Exception:
            pass
        # Also re-request visible tiles now that full concurrency is available.
        self._request_load_visible_images(0)

    def _stop_layout_rebuild_timers(self) -> None:
        """Cancel pending gallery layout rebuilds (e.g. when leaving gallery mode)."""
        for timer_name in (
            "_metadata_rebuild_timer",
            "_resize_timer",
            "_aspects_settle_timer",
        ):
            timer = getattr(self, timer_name, None)
            if timer is not None and timer.isActive():
                timer.stop()

    def on_visual_rotation_changed(
        self, file_path: str, *, defer_rebuild: bool = False
    ) -> bool:
        """Update layout and visible tiles when single-view rotation changes.

        Returns True when layout aspects changed and a rebuild is needed.
        """
        if not file_path:
            return False
        resolved = self._resolve_gallery_path(file_path)
        if not resolved:
            return False
        file_path = resolved
        self._invalidate_scaled_thumbnails_for_path(file_path)
        if file_path not in self._path_to_indices:
            return False
        display_aspect = self._display_aspect(
            resolved, self._base_aspect_for_path(resolved)
        )
        changed = False
        for idx in self._path_to_indices.get(resolved, []):
            if idx < len(self._gallery_layout_items):
                old_aspect = self._gallery_layout_items[idx].get("aspect", 1.5)
                if abs(old_aspect - display_aspect) > 0.05:
                    self._gallery_layout_items[idx]["aspect"] = display_aspect
                    changed = True
        if changed and self._gallery_layout_items:
            if defer_rebuild:
                return True
            self._add_metadata_changed_path(resolved)
            self.build_gallery(force=True)
        elif changed:
            return True
        else:
            self.refresh_visible_tile_for_path(resolved)
        return changed

    def _add_metadata_changed_path(self, file_path: str) -> None:
        self._metadata_changed_paths.add(file_path)
        self._last_metadata_change_time = time.time()

    def sync_visual_rotations(self) -> None:
        """Refresh gallery tiles/layout for any paths with stored visual rotation."""
        paths = [
            p for p in (self.images or [])
            if isinstance(p, str) and self._get_rotation_degrees_for_path(p)
        ]
        if not paths:
            return
        layout_changed = False
        for path in paths:
            if self.on_visual_rotation_changed(path, defer_rebuild=True):
                layout_changed = True
            self.refresh_visible_tile_for_path(path)
        if layout_changed and self._gallery_layout_items:
            for path in paths:
                self._add_metadata_changed_path(path)
            self._last_metadata_change_time = time.time()
            if not self._metadata_rebuild_timer.isActive():
                self._metadata_rebuild_timer.start(400)

    def apply_visual_rotation_for_path(self, file_path: str) -> None:
        """Apply one file's stored visual rotation when returning to gallery."""
        if not file_path:
            return
        resolved = self._resolve_gallery_path(file_path) or file_path
        layout_changed = self.on_visual_rotation_changed(resolved, defer_rebuild=True)
        self.refresh_visible_tile_for_path(resolved)
        if layout_changed and self._gallery_layout_items:
            self._add_metadata_changed_path(resolved)
            if not self._metadata_rebuild_timer.isActive():
                self._metadata_rebuild_timer.start(400)

    def _schedule_viewport_width_rebuild(self, *, debounce_ms: int = 120) -> None:
        """Rebuild justified rows when the scroll viewport width changes."""
        if getattr(self, "_ignore_resize_events", False):
            return
        if time.time() < float(getattr(self, "_gallery_entry_coalesce_until", 0.0) or 0.0):
            return
        if self._gallery_warmup_active():
            return
        # Rebuilding rows during scroll jumps content height and causes visible glitches.
        if self._is_scrolling_fast or (time.time() - self._last_scroll_event_t) < 0.12:
            debounce_ms = max(debounce_ms, 280)
        try:
            current_viewport_width = self._get_viewport_width()
            if current_viewport_width <= 0:
                return
            if current_viewport_width == self._last_layout_viewport_width:
                return
            # Latch immediately so burst resize events schedule only one rebuild.
            self._last_layout_viewport_width = current_viewport_width
            if self._resize_timer is not None:
                self._resize_timer.start(max(1, int(debounce_ms)))
            else:
                QTimer.singleShot(0, self.build_gallery)
        except Exception:
            pass

    def force_layout_update(self) -> None:
        """Public hook from main window after resize completes (edge drag, maximize, etc.)."""
        self._schedule_viewport_width_rebuild(debounce_ms=0)

    def resizeEvent(self, event):
        # Debounce expensive rebuilds during window resize.
        try:
            self._schedule_viewport_width_rebuild()
        except Exception:
            pass
        try:
            if getattr(self, "_empty_label", None) and self._empty_label.isVisible():
                self._update_empty_label_geometry()
        except Exception:
            pass
        return super().resizeEvent(event)

    def _handle_resize_rebuild(self):
        """Rebuild layout after resize settles."""
        try:
            # Use existing metadata cache; do not block on disk.
            self.build_gallery(bulk_metadata=None)
        except Exception:
            logger.exception("[GALLERY] resize rebuild failed")

    def _delayed_initial_build(self):
        self._setup_scroll_tracking()
        if self.width() > 0:
            # Never call processEvents() here: it can re-enter Qt/Python slots
            # (e.g. ImageLoadManager.thumbnail_ready) while still inside gallery init
            # and cause deep recursion / SIGABRT via pyqt6_err_print.
            self.build_gallery()
        else:
            QTimer.singleShot(50, self._delayed_initial_build)

    def _setup_scroll_tracking(self):
        try:
            p = self.parent()
            while p and not isinstance(p, QScrollArea):
                p = p.parent()
            if p:
                self._scroll_area = p
                scrollbar = p.verticalScrollBar()
                try:
                    scrollbar.valueChanged.disconnect(self._on_scroll)
                except Exception:
                    pass
                scrollbar.valueChanged.connect(self._on_scroll)
                # Dragging the scrollbar thumb should trigger immediate loading on release
                try:
                    scrollbar.sliderPressed.disconnect(self._on_slider_pressed)
                except Exception:
                    pass
                try:
                    scrollbar.sliderReleased.disconnect(self._on_slider_released)
                except Exception:
                    pass
                scrollbar.sliderPressed.connect(self._on_slider_pressed)
                scrollbar.sliderReleased.connect(self._on_slider_released)

                # Install wheel event filter on viewport for smooth scrolling
                try:
                    p.viewport().removeEventFilter(self)
                except Exception:
                    pass
                p.viewport().installEventFilter(self)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        # QScrollArea child width is fixed (widgetResizable=False); viewport width changes
        # on window resize without resizing the gallery widget — watch viewport Resize.
        if (
            event.type() == QEvent.Type.Resize
            and self._scroll_area is not None
            and obj is self._scroll_area.viewport()
        ):
            self._schedule_viewport_width_rebuild()
            return False

        # Wheel: trackpads send pixelDelta — apply immediately (native feel).
        # Mouse wheels send angleDelta only — use the smoothed step timer.
        if event.type() == QEvent.Type.Wheel and self._scroll_area and obj is self._scroll_area.viewport():
            try:
                sb = self._scroll_area.verticalScrollBar()
                if sb is None:
                    return False
                pixel = event.pixelDelta()
                angle = event.angleDelta()
                if not pixel.isNull() and pixel.y() != 0:
                    # Trackpad pixel scrolling: let QScrollArea handle natively (smoother on macOS).
                    return False
                if angle.y() != 0:
                    notches = angle.y() / 120.0
                    delta_y = int(notches * 120)
                    self._wheel_accum_px += -float(delta_y)
                    if not self._wheel_timer.isActive():
                        self._wheel_timer.start(self._wheel_tick_ms)
                    event.accept()
                    return True
            except Exception:
                pass
            event.accept()
            return True

        return super().eventFilter(obj, event)

    def _apply_wheel_scroll_step(self):
        if not self._scroll_area:
            self._wheel_timer.stop()
            self._wheel_accum_px = 0.0
            return

        sb = self._scroll_area.verticalScrollBar()
        if sb is None:
            self._wheel_timer.stop()
            self._wheel_accum_px = 0.0
            return

        # Stop if accumulation is negligible
        if abs(self._wheel_accum_px) < 1.0:
            self._wheel_timer.stop()
            self._wheel_accum_px = 0.0
            return

        # Adaptive step: consume more when accumulation is high (exponential decay)
        # 0.2 means we consume 20% of the remaining distance per tick (8ms)
        # This makes the scroll feel very responsive and stop quickly.
        # But we still enforce a minimum step of _wheel_step_px (18px) to maintain movement.
        adaptive_step = self._wheel_accum_px * 0.2
        if abs(adaptive_step) < self._wheel_step_px:
            step = self._wheel_step_px if self._wheel_accum_px > 0 else -self._wheel_step_px
        else:
            step = adaptive_step

        # Don't over-consume
        if abs(step) > abs(self._wheel_accum_px):
            step = self._wheel_accum_px
        
        self._wheel_accum_px -= step

        current_val = sb.value()
        new_val = int(current_val + step)
        
        # Boundary check: if we hit top/bottom, clear accumulation to stop instantly
        if (new_val <= sb.minimum() and step < 0) or (new_val >= sb.maximum() and step > 0):
            sb.setValue(sb.minimum() if step < 0 else sb.maximum())
            self._wheel_accum_px = 0.0
            self._wheel_timer.stop()
            return

        sb.setValue(new_val)

    def _on_slider_pressed(self):
        self._is_scrollbar_dragging = True

    def _on_slider_released(self):
        self._is_scrollbar_dragging = False
        # Immediately load around the new thumb position (no extra debounce)
        self._on_scroll_settled()

    def _on_scroll(self, value):
        now = time.time()
        
        # Pause background indexing once per scroll burst (not every valueChanged tick).
        if self.parent_viewer is not None and not self._scroll_indexing_paused:
            if hasattr(self.parent_viewer, "_pause_semantic_indexing_deferred"):
                try:
                    self.parent_viewer._pause_semantic_indexing_deferred()
                    self._scroll_indexing_paused = True
                    timer = getattr(self.parent_viewer, "_resume_indexing_timer", None)
                    if timer is not None:
                        timer.stop()
                except Exception:
                    pass

        if self._last_scroll_y >= 0:
            last_time = getattr(self, "_last_scroll_time", 0)
            dt = now - last_time
            # Support higher frequency scrolling (e.g. 8ms wheel timer or precision touchpad)
            if dt > 0.002:
                current_speed = getattr(self, "_current_scroll_speed", 0.0)
                speed = abs(value - self._last_scroll_y) / max(0.001, dt)
                self._current_scroll_speed = (current_speed * 0.4) + (speed * 0.6)
                self._is_scrolling_fast = self._current_scroll_speed > self._scroll_optimize_threshold

        self._last_scroll_y = value
        self._last_scroll_time = now
        self._last_scroll_event_t = now
        if self._thumb_first_after_scroll_t is None:
            self._thumb_first_after_scroll_t = now

        # When the user is dragging the scrollbar thumb, avoid scheduling any image work.
        # We only want to load once after release (sliderReleased → _on_scroll_settled()).
        if self._is_scrollbar_dragging:
            try:
                if self._load_timer and self._load_timer.isActive():
                    self._load_timer.stop()
                if self._scroll_settle_timer and self._scroll_settle_timer.isActive():
                    self._scroll_settle_timer.stop()
                if self._idle_preload_timer and self._idle_preload_timer.isActive():
                    self._idle_preload_timer.stop()
            except Exception:
                pass
            return

        # Throttle (do NOT restart continuously): allow periodic updates while scrolling.
        # The settle timer below guarantees a final update when scrolling stops.
        interval = 50 if not self._is_scrolling_fast else 150
        if self._is_actively_scrolling() and not self._active_tasks:
            interval = max(interval, 120)
        if not self._load_timer.isActive():
            self._load_timer.start(interval)

        # Debounce: after scrolling stops, force a final load near thumb position.
        if self._scroll_settle_timer.isActive():
            self._scroll_settle_timer.stop()
        self._scroll_settle_timer.start(120)

    def _on_scroll_settled(self):
        """Called after scroll events stop; load thumbnails around thumb position."""
        # When the user stops scrolling, treat as "not scrolling fast" so we actually schedule work.
        self._is_scrolling_fast = False
        self._current_scroll_speed = 0
        self._scroll_indexing_paused = False
        self._last_scroll_settle_t = time.time()
        self._thumb_first_after_settle_t = self._last_scroll_settle_t
        self._last_scroll_event_t = 0.0  # Reset scroll event timestamp so we enable DB loading
        pending_layout = bool(self._metadata_changed_paths) or bool(
            self._orientation_flip_paths
        )
        if pending_layout and not self._building:
            if _orient_debug_enabled() and self._orientation_flip_paths:
                logger.info(
                    "[ORIENT] scroll settled — scheduling layout rebuild for %d orientation flip(s)",
                    len(self._orientation_flip_paths),
                )
            self._metadata_rebuild_timer.start(0)
        else:
            self.load_visible_images()
        
        # Defer resuming indexing for 5 seconds of idle gallery time
        if self.parent_viewer is not None:
            try:
                timer = getattr(self.parent_viewer, "_resume_indexing_timer", None)
                if timer is not None:
                    timer.start(5000)
            except Exception:
                pass

    def _get_viewport_width(self):
        p = self.parent()
        while p and not isinstance(p, QScrollArea):
            p = p.parent()
        return p.viewport().width() if p else self.width()

    def _scroll_area_widget(self) -> Optional[QScrollArea]:
        scroll = self._scroll_area
        if scroll is None:
            p = self.parent()
            while p and not isinstance(p, QScrollArea):
                p = p.parent()
            scroll = p
        return scroll if isinstance(scroll, QScrollArea) else None

    def _current_scroll_y(self) -> int:
        scroll = self._scroll_area_widget()
        if scroll is None:
            return 0
        return int(scroll.verticalScrollBar().value())

    def _content_rect_to_viewport(self, rect: QRect) -> QRect:
        """Map layout rect (content coordinates) to widget-local coordinates.
        With native QScrollArea handling, content coordinates are widget-local.
        """
        return rect

    def _refresh_gallery_scroll_range(self) -> None:
        """Resize gallery content and nudge QScrollArea so scrollbar range matches layout height."""
        scroll = self._scroll_area_widget()
        vp_w = max(300, self._get_viewport_width())
        viewport_h = 600
        if scroll is not None:
            viewport_h = max(1, int(scroll.viewport().height()))
        content_h = max(viewport_h, int(self._total_content_height or viewport_h))

        if self.minimumWidth() != vp_w:
            self.setMinimumWidth(vp_w)
        if self.minimumHeight() != content_h:
            self.setMinimumHeight(content_h)
        if self.maximumHeight() != content_h:
            self.setMaximumHeight(content_h)
        if self.width() != vp_w or self.height() != content_h:
            self.resize(vp_w, content_h)

        if scroll is not None:
            scroll.updateGeometry()

    def _apply_scroll_anchor_state(self, state: Optional[dict]) -> bool:
        """Restore viewport scroll; retry on next tick if the scrollbar range is still settling."""
        if not state:
            return False
        self._refresh_gallery_scroll_range()
        if self._restore_scroll_anchor_state(state):
            self._pending_scroll_anchor_state = None
            self._pending_scroll_to_path = None
            self._scroll_to_file_attempts = 0
            return True
        self._pending_scroll_anchor_state = state
        self._pending_scroll_to_path = None
        self._scroll_to_file_attempts = 0
        self._schedule_scroll_to_file_retry(0)
        return False

    def _sync_content_geometry(self) -> None:
        """Sizes the gallery widget to the actual content height to let QScrollArea drive scrolling natively."""
        self._refresh_gallery_scroll_range()

    def build_gallery(self, bulk_metadata=None, force=False):
        """
        Calculate grid layout and place placeholders. 
        Does not load images directly - that's handled by visible range tracking.
        """
        if self._building:
            self._pending_gallery_build = True
            self._pending_build_metadata = bulk_metadata
            self._pending_build_force = self._pending_build_force or bool(force)
            return
        if not self.images:
            return
        if _focus_gallery_switch_logs():
            logger.debug(
                "[MODESWITCH] gallery.build_gallery start; images=%d width=%d",
                len(self.images),
                self._get_viewport_width(),
            )

        # Layout may report width 0 right after gallery_container.show() — do not clear
        # thumbnails/layout in that case or the gallery stays empty ("failed to load").
        viewport_width = self._get_viewport_width()
        layout_unchanged = (
            viewport_width == self._last_layout_viewport_width
            and self._layout_matches_current_images()
            and self._total_content_height == self._last_layout_content_height
        )
        if (
            layout_unchanged
            and not force
            and (time.time() - self._last_build_ts) < 0.8
        ):
            # Skip duplicate rebuilds caused by near-simultaneous resize/layout churn.
            self._reposition_thumbnail_widgets()
            self._request_load_visible_images(20)
            return
        # Layout margins for symmetry with scrollbar
        # Left margin 24px, Right margin 0px + 24px scrollbar gutter = 24px on both sides
        left_margin = 24
        right_margin = 0
        net_width = viewport_width - left_margin - right_margin
        if net_width <= 0:
            n = getattr(self, "_gallery_width_defer_count", 0) + 1
            self._gallery_width_defer_count = n
            if n <= 40:
                QTimer.singleShot(50, lambda m=bulk_metadata: self.build_gallery(bulk_metadata=m))
            else:
                self._gallery_width_defer_count = 0
                logger.warning(
                    "[GALLERY] Viewport width stayed <= 0 after retries; layout may be broken"
                )
            return
        self._gallery_width_defer_count = 0

        anchor_state = None
        if getattr(self, "_is_zooming", False) and self._zoom_anchor_state:
            anchor_state = self._zoom_scroll_anchor_for_rebuild()
        elif self._gallery_layout_items:
            anchor_state = self._capture_scroll_anchor_state()

        self._building = True
        if self._resize_timer is not None and self._resize_timer.isActive():
            self._resize_timer.stop()
        self._last_layout_viewport_width = viewport_width
        should_load_visible = False
        try:


            self._gallery_layout_items.clear()
            self._path_to_indices.clear()

            if bulk_metadata:
                self._metadata_cache.update(bulk_metadata)
            if self.parent_viewer and hasattr(self.parent_viewer, "image_cache"):
                paths = [img for img in self.images if isinstance(img, str)]
                # Fetch each path's EXIF from the DB at most ONCE per folder. Rebuilds
                # (orientation flips, metadata refresh) fire repeatedly on large folders;
                # re-running a 6000+-row SQLite read on the MAIN THREAD every rebuild —
                # while the background indexer may be writing the same DB — blocks paints,
                # signal delivery, and task scheduling (observed as periodic multi-second
                # gallery stalls with idle workers). Paths that were missing stay missing
                # until the metadata-ready event injects fresh rows via bulk_metadata.
                missing_paths = [
                    p
                    for p in paths
                    if p not in self._metadata_cache
                    and p not in self._metadata_fetch_attempted
                ]
                if missing_paths and self.parent_viewer and hasattr(
                    self.parent_viewer, "image_cache"
                ):
                    cap = 100000
                    fetch_paths = self._prioritize_entry_paths(missing_paths, cap=cap)
                    bulk_fetched = self.parent_viewer.image_cache.get_multiple_exif(
                        fetch_paths
                    )
                    self._metadata_fetch_attempted.update(fetch_paths)
                    if bulk_fetched:
                        self._metadata_cache.update(bulk_fetched)

            current_y = 10
            left_margin = 24
            row = []
            aspect_sum = 0

            def commit_row(r, a_sum, is_last):
                nonlocal current_y
                if not r:
                    return
                spacing = (len(r) - 1) * self.MIN_SPACING
                row_h = self.TARGET_ROW_HEIGHT if is_last else (net_width - spacing) / a_sum
                row_h = max(
                    self.TARGET_ROW_HEIGHT * 0.4,
                    min(self.TARGET_ROW_HEIGHT * 2.2, row_h),
                )

                curr_x = left_margin
                for i, (item, aspect) in enumerate(r):
                    w = int(row_h * aspect)
                    # Only justify (stretch) the last tile on full rows; last row keeps natural aspects.
                    if not is_last and i == len(r) - 1:
                        w = net_width - (curr_x - left_margin)
                    self._gallery_layout_items.append(
                        {
                            "rect": QRect(curr_x, int(current_y), int(w), int(row_h)),
                            "file_path": item if isinstance(item, str) else None,
                            "aspect": aspect,
                        }
                    )
                    curr_x += w + self.MIN_SPACING
                current_y += row_h + self.MIN_SPACING

            for item in self.images:
                aspect = 1.5
                if isinstance(item, str):
                    aspect = self._layout_aspect_for_path(item)

                row.append((item, aspect))
                aspect_sum += aspect
                if (aspect_sum * self.TARGET_ROW_HEIGHT + (len(row) - 1) * self.MIN_SPACING) >= net_width:
                    commit_row(row, aspect_sum, False)
                    row, aspect_sum = [], 0

            if row:
                commit_row(row, aspect_sum, True)

            self._path_to_indices = {}
            for i, item in enumerate(self._gallery_layout_items):
                p = item["file_path"]
                if p not in self._path_to_indices:
                    self._path_to_indices[p] = []
                self._path_to_indices[p].append(i)

            self._total_content_height = int(current_y + 20)
            self._sync_content_geometry()
            scroll = self._scroll_area_widget()
            if scroll is not None:
                sb = scroll.verticalScrollBar()
                if sb.value() > sb.maximum():
                    sb.setValue(max(sb.minimum(), sb.maximum()))
            self.update()
            self._last_build_ts = time.time()
            self._last_layout_content_height = self._total_content_height
            self._layout_image_sequence = tuple(self.images)
            self._gallery_folder_token = self._current_folder_load_token()
            if force:
                self._last_force_build_ts = self._last_build_ts
            self._reposition_thumbnail_widgets()
            if anchor_state:
                self._apply_scroll_anchor_state(anchor_state)
            if _focus_gallery_switch_logs():
                logger.debug(
                    "[MODESWITCH] gallery.build_gallery done; items=%d content_h=%d",
                    len(self._gallery_layout_items),
                    self._total_content_height,
                )
            should_load_visible = True
            if len(self.images) > 100 and not (
                getattr(self, "_is_zooming", False)
                or getattr(self, "_gallery_zoom_rebuild", False)
            ):
                # Only schedule a deferred settle when aspects are still drifting;
                # otherwise large folders rebuild every few seconds for no gain.
                if self._metadata_changed_paths:
                    self._schedule_aspects_settle_rebuild()
            if len(self.images) > 1000:
                self.hide_loading_message()
        finally:
            self._building = False
            if getattr(self, "_pending_gallery_build", False):
                self._pending_gallery_build = False
                pending_meta = self._pending_build_metadata
                pending_force = self._pending_build_force
                self._pending_build_metadata = None
                self._pending_build_force = False
                if self._gallery_warmup_active() or time.time() < float(
                    getattr(self, "_gallery_entry_coalesce_until", 0.0) or 0.0
                ):
                    self._reposition_thumbnail_widgets()
                    self._request_load_visible_images(20)
                else:
                    QTimer.singleShot(
                        0,
                        lambda m=pending_meta, f=pending_force: self.build_gallery(
                            bulk_metadata=m, force=f
                        ),
                    )
            elif should_load_visible and not self._gallery_folder_superseded():
                # Apply persisted visual rotations once layout indices exist (set_images()
                # schedules build asynchronously; an earlier sync would no-op).
                self.sync_visual_rotations()
                # Run after _building is cleared, so load_visible_images won't early-return.
                if self._pending_scroll_to_path or self._pending_scroll_anchor_state:
                    QTimer.singleShot(0, self._apply_pending_scroll_to_file)
                self._request_load_visible_images(20)

    def _get_visible_range(self, buffer_rect):
        items = self._gallery_layout_items
        if not items:
            return []

        low = 0
        high = len(items) - 1
        start_idx = 0
        y_top = buffer_rect.top()
        while low <= high:
            mid = (low + high) // 2
            if items[mid]["rect"].bottom() < y_top:
                low = mid + 1
            else:
                start_idx = mid
                high = mid - 1

        visible = []
        for i in range(start_idx, len(items)):
            item = items[i]
            if item["rect"].top() > buffer_rect.bottom():
                break
            visible.append((i, item))
        return visible

    def _get_first_visible_image_index(self):
        idx = self._get_upper_left_visible_image_index()
        if idx >= 0:
            return idx
        p = self._scroll_area
        if p is None:
            p = self.parent()
            while p and not isinstance(p, QScrollArea):
                p = p.parent()
        if not isinstance(p, QScrollArea):
            return -1

        sb = p.verticalScrollBar()
        scroll_y = sb.value()

        # Find the first item whose bottom is below scroll_y
        for i, item in enumerate(self._gallery_layout_items):
            if item["rect"].bottom() > scroll_y:
                return i
        return -1

    def load_visible_images(self):
        # Stop background preloading when loading visible images (e.g. during scroll)
        self._idle_preload_timer.stop()

        if self._building:
            self._request_load_visible_images(50)
            return

        if self._gallery_folder_superseded():
            return

        pv = self.parent_viewer
        if pv is not None and getattr(pv, "view_mode", "") != "gallery":
            return
        gw = getattr(pv, "gallery_widget", None) if pv is not None else None
        if gw is not None and not gw.isVisible():
            return

        p = self.parent()
        while p and not isinstance(p, QScrollArea):
            p = p.parent()
        if not p:
            return
        if self.load_manager is None:
            if _focus_gallery_switch_logs():
                logger.debug("[MODESWITCH] gallery.load_visible_images skipped; load_manager is None")
            return

        # Drop stale in-flight markers so failed/missed thumbnails can be retried.
        self._evict_stale_active_tasks(max_age_s=4.0)

        v_port = p.viewport()
        scrollbar = p.verticalScrollBar()
        scroll_y = scrollbar.value()
        v_h = v_port.height()
        if v_h <= 0:
            self._request_load_visible_images(50)
            return
        actively_scrolling = self._is_actively_scrolling()

        actively_scrolling = self._is_actively_scrolling()
        buffer_screens = (
            _gallery_scroll_buffer_screens()
            if actively_scrolling
            else _gallery_viewport_buffer_screens()
        )
        buffer_px = int(v_h * buffer_screens)
        buffer_top = max(0, scroll_y - buffer_px)
        buffer_h = v_h + (2 * buffer_px)
        buffer_rect = QRect(0, buffer_top, v_port.width(), buffer_h)
        visible_indices_items = self._get_visible_range(buffer_rect)
        if _focus_gallery_switch_logs():
            logger.debug(
                "[MODESWITCH] gallery.load_visible_images visible=%d cached_tasks=%d",
                len(visible_indices_items),
                block_visible_indices := len(self._active_tasks),
            )
        visible_indices_set = {idx for idx, item in visible_indices_items}

        # Dynamic prefetch: follow the scrollbar thumb/viewport center position.
        # Slow scroll: keep more around the thumb; fast scroll: keep less.
        # Note: we already early-return on fast scroll, so this mainly helps "normal" scrolling.
        screens = _gallery_prefetch_screens(self._is_scrolling_fast)
        center_y = scroll_y + (v_h // 2)
        half_span = int((v_h * screens) // 2)
        prefetch_top = max(0, center_y - half_span)
        prefetch_span = max(v_h * screens, buffer_h)
        prefetch_rect = QRect(0, prefetch_top, v_port.width(), prefetch_span)
        prefetch_indices_items = self._get_visible_range(prefetch_rect)
        prefetch_paths = {item["file_path"] for idx, item in prefetch_indices_items if item.get("file_path")}

        center_idx = self._gallery_prefetch_center_index(scroll_y, v_h)
        center_paths: set[str] = set()
        if center_idx is not None:
            center_paths = self._bidirectional_jpeg_paths_from_center(center_idx)

        is_fast = self._is_scrolling_fast
        for idx in list(self._visible_widgets.keys()):
            if idx not in visible_indices_set:
                w = self._visible_widgets.pop(idx)
                w.hide()
                self._clear_widget_thumbnail(w)
                w.file_path = None
                self._widget_pool.append(w)

        # Fast scroll policy:
        # - still keep visible thumbnails loading (small budget)
        # - avoid heavy prefetch
        # First-paint mode: prioritize visible tiles only until first thumbnail arrives
        # (or a short timeout), then enable prefetch.
        warmup_elapsed = time.time() - float(getattr(self, "_gallery_set_images_ts", 0.0) or 0.0)
        warmup_active = self._gallery_warmup_active()
        allow_prefetch = (not is_fast) and (not actively_scrolling) and (not warmup_active) and (
            self._first_thumb_ready_after_set or warmup_elapsed > 1.5
        )

        # If the user jumped far (typical when dragging scrollbar), flush stale queue so new
        # position thumbnails start quickly. Do NOT clear _requested_thumbnail_paths here:
        # flush_queue() only drops the load manager's own queue/active-task bookkeeping, not
        # this gallery's separate self._active_tasks dict (used for the active_cap check
        # below). Clearing _requested_thumbnail_paths would skip the to_cancel diff right
        # after this block, which is what actually pops stale paths out of self._active_tasks
        # -- leaving old, now-offscreen CURRENT-priority entries stuck "active" (they survive
        # the drop_prefetch eviction since it only targets non-CURRENT priority) until the
        # blanket 4s sweep at the top of this function catches them, stalling the new
        # viewport's thumbnails behind a still-counted-active backlog it can no longer see.
        if abs(scroll_y - self._last_scheduled_scroll_y) > (v_h * 3):
            try:
                if hasattr(self.load_manager, "flush_queue"):
                    self.load_manager.flush_queue()
                else:
                    self.load_manager.cancel_all_tasks()
            except Exception:
                pass
            # A deliberate jump away means the user is no longer following the entry
            # point -- stop anchoring center_paths/prefetch to it (see
            # _entry_prefetch_active, which is otherwise purely spatial and would
            # keep reserving active_cap slots around the entry file forever once
            # it's offscreen, since "offscreen" never resolves on its own after a
            # jump the way it does during incremental scrolling).
            self._entry_prefetch_abandoned = True
        self._last_scheduled_scroll_y = scroll_y
        # Determine what we want around current thumb position - paths for prefetch
        visible_paths = {item["file_path"] for idx, item in visible_indices_items if item.get("file_path")}
        wanted_paths = set(visible_paths)
        if allow_prefetch:
            wanted_paths |= {item["file_path"] for idx, item in prefetch_indices_items if item.get("file_path")}
        if center_paths:
            wanted_paths |= center_paths

        if not actively_scrolling:
            # Cancel thumbnail work that is no longer near the scrollbar thumb.
            # Skip cancel for visible RAW/DNG tiles still decoding at CURRENT priority.
            to_cancel = self._requested_thumbnail_paths - wanted_paths
            protected_paths: set[str] = set()
            for fp in list(to_cancel):
                if self._should_protect_active_raw_thumbnail(fp, visible_paths):
                    protected_paths.add(fp)
                    continue
                try:
                    self.load_manager.cancel_task(fp)
                except Exception:
                    pass
                self._active_tasks.pop(fp, None)
            self._requested_thumbnail_paths = wanted_paths | protected_paths
        else:
            self._requested_thumbnail_paths = set(visible_paths)

        load_tasks = []
        created_widgets = 0
        scheduled_tasks = 0

        # Reduce per-tick work when fast scrolling (keep things responsive)
        # RESTORED: Using v1.6.0-style aggressive scheduling for snappier population
        if warmup_active:
            max_widgets, max_tasks, active_cap = _gallery_warmup_scheduling_budgets(
                len(self.images or [])
            )
        else:
            max_widgets, max_tasks, active_cap = _gallery_scheduling_budgets(is_fast)
        folder_path = None
        pv = self.parent_viewer
        if pv is not None:
            folder_path = getattr(pv, "current_folder", None)
        max_widgets, max_tasks, active_cap = _apply_external_gallery_caps(
            max_widgets, max_tasks, active_cap, folder_path
        )
        active_cap = min(
            active_cap,
            _env_int("RAWVIEWER_GALLERY_ACTIVE_CAP_HARD", 80, minimum=8),
        )
        for idx, item in visible_indices_items:
            path = item.get("file_path")
            rect = item.get("rect")
            if not path or rect is None:
                continue

            if idx not in self._visible_widgets:
                if created_widgets >= max_widgets:
                    # Defer widget creation but keep updating slots that already exist.
                    if not self._load_timer.isActive():
                        self._load_timer.start(16)
                    continue
                w = self._widget_pool.pop() if self._widget_pool else ThumbnailLabel(self)
                self._clear_widget_thumbnail(w)
                w.file_path = path
                w.index = idx  # Keep track of index on the widget
                display_rect = self._content_rect_to_viewport(rect)
                w.setGeometry(display_rect)
                w.setFixedSize(display_rect.size())
                def _on_thumb_press(e, _w=w):
                    try:
                        if e.button() == Qt.MouseButton.LeftButton:
                            e.accept()
                            target_path = getattr(_w, "file_path", None)
                            pv = self.parent_viewer
                            if target_path and pv is not None:
                                mods = (
                                    pv._gallery_event_modifiers(e)
                                    if hasattr(pv, "_gallery_event_modifiers")
                                    else e.modifiers()
                                )
                                if hasattr(pv, "_gallery_has_shift_modifier") and pv._gallery_has_shift_modifier(
                                    mods
                                ) and not (
                                    hasattr(pv, "_gallery_has_extend_modifier")
                                    and pv._gallery_has_extend_modifier(mods)
                                ):
                                    pv._gallery_shift_click_selection(target_path)
                                    return
                                if hasattr(pv, "_gallery_has_extend_modifier") and pv._gallery_has_extend_modifier(
                                    mods
                                ):
                                    if hasattr(pv, "_gallery_toggle_path_selection"):
                                        pv._gallery_toggle_path_selection(target_path)
                                    return
                                pv._gallery_item_clicked(target_path)
                            return
                    except Exception:
                        pass
                    try:
                        e.ignore()
                    except Exception:
                        pass

                def _on_thumb_release(e, _w=w):
                    if e.button() != Qt.MouseButton.LeftButton:
                        ThumbnailLabel.mouseReleaseEvent(_w, e)
                        return
                    pv = self.parent_viewer
                    mods = (
                        pv._gallery_event_modifiers(e)
                        if pv is not None and hasattr(pv, "_gallery_event_modifiers")
                        else e.modifiers()
                    )
                    if pv is not None and (
                        (
                            hasattr(pv, "_gallery_has_extend_modifier")
                            and pv._gallery_has_extend_modifier(mods)
                        )
                        or (
                            hasattr(pv, "_gallery_has_shift_modifier")
                            and pv._gallery_has_shift_modifier(mods)
                        )
                    ):
                        _w._drag_start_pos = None
                        _w._drag_started = False
                        e.accept()
                        return
                    ThumbnailLabel.mouseReleaseEvent(_w, e)

                w.mousePressEvent = _on_thumb_press
                w.mouseReleaseEvent = _on_thumb_release
                created_widgets += 1
                self._visible_widgets[idx] = w
            else:
                w = self._visible_widgets[idx]
                if w.file_path != path:
                    self._clear_widget_thumbnail(w)
                w.file_path = path
                w.index = idx
                display_rect = self._content_rect_to_viewport(rect)
                if w.geometry() != display_rect:
                    w.setGeometry(display_rect)
                if w.size() != display_rect.size():
                    w.setFixedSize(display_rect.size())

            cache_hit = False
            # Account for Device Pixel Ratio (Retina/4K)
            dpr = self.devicePixelRatio()
            physical_size = QSize(int(rect.width() * dpr), int(rect.height() * dpr))
            scaled_key = self._scaled_cache_key(path, physical_size)
            cached_scaled = self._thumbnail_cache.get(scaled_key)
            expected_logical = QSize(rect.width(), rect.height())
            if cached_scaled and not cached_scaled.isNull():
                # A scaled entry is rect-shaped by construction, so its orientation is the
                # rect's — if the measured pixels say the image is portrait but this scaled
                # crop (and rect) are landscape, it's a stale crop from before the layout
                # flip. Painting it is exactly the "portrait shown in a landscape frame"
                # artifact; drop it and fall through to the guarded base path instead.
                stale_crop = False
                measured_ar = self._measured_raw_aspects.get(path)
                if measured_ar and measured_ar > 0 and cached_scaled.height() > 0:
                    px_o = self._aspect_orientation_flag(
                        cached_scaled.width() / cached_scaled.height()
                    )
                    m_o = self._aspect_orientation_flag(
                        self._display_aspect(path, measured_ar)
                    )
                    stale_crop = px_o != 0 and m_o != 0 and px_o != m_o
                if stale_crop:
                    self._thumbnail_cache.remove(scaled_key)
                    cached_scaled = None
                elif self._pixmap_logical_size(cached_scaled, dpr) == expected_logical:
                    w.setPixmap(cached_scaled)
                    w.setText("")
                    if not cached_scaled.isNull() and not self._first_thumb_ready_after_set:
                        self._first_thumb_ready_after_set = True
                        self._end_gallery_load_warmup_throttle()
                    cache_hit = True
                else:
                    cached_scaled = None
            if not cache_hit:
                base = self._thumbnail_cache.get((path, self._thumb_base_key))
                if not base:
                    # Avoid main thread SQLite lock contention/stutter by skipping synchronous fetches during active scrolls
                    is_scrolling = (time.time() - self._last_scroll_event_t) < 0.25
                    if is_scrolling:
                        pass
                    else:
                        try:
                            max_dim = max(physical_size.width(), physical_size.height())

                            base = self._global_cache_to_base_pixmap(path)
                            if base is not None and not base.isNull():
                                src_dim = max(base.width(), base.height())
                                # Grid/nav previews are smaller than tile rects but still
                                # usable for instant first paint; sharper tiers load async.
                                min_factor = 0.45 if not self._first_thumb_ready_after_set else 0.85
                                if src_dim < int(max_dim * min_factor):
                                    base = None

                            if base is not None and not base.isNull():
                                base = self._store_oriented_base_pixmap(path, base)
                            else:
                                base = None
                        except Exception as e:
                            logger.debug(f"Sync adaptive mipmap fetch failed for {path}: {e}")
                
                if base is not None and not base.isNull():
                    # Tile geometry must follow the oriented pixels, exactly like the
                    # filmstrip. This base may have been pulled straight from the global
                    # cache above (bypassing on_thumbnail_ready), so a portrait thumbnail
                    # would otherwise be crop-fit into a landscape rect built from stale
                    # metadata. Reconcile before fitting so the debounced rebuild corrects
                    # the frame.
                    self._reconcile_tile_aspect(path, base)
                    if not self._frame_matches_pixmap_orientation(
                        self._gallery_layout_items[idx].get("rect")
                        if idx < len(self._gallery_layout_items)
                        else None,
                        base,
                    ):
                        # Do not `continue` — that skipped w.show() and never queued a
                        # thumbnail load, leaving a permanent blank tile until scroll.
                        if _orient_debug_enabled():
                            logger.info(
                                "[ORIENT] frame mismatch visible tile %s; queue reload",
                                os.path.basename(path),
                            )
                        self._thumbnail_cache.remove((path, self._thumb_base_key))
                        self._apply_pixmap_if_changed(w, QPixmap())
                        w.setText("")
                        if not self._metadata_rebuild_timer.isActive():
                            self._metadata_rebuild_timer.start(
                                0 if not self._is_scrolling_fast else 250
                            )
                    else:
                        scaled = self._fit_tile_pixmap(path, base, physical_size, dpr)
                        w.setPixmap(scaled)
                        w.setText("")
                        if not scaled.isNull() and not self._first_thumb_ready_after_set:
                            self._first_thumb_ready_after_set = True
                            self._end_gallery_load_warmup_throttle()
                        cache_hit = self._gallery_base_meets_tile(base, physical_size)
                else:
                    self._apply_pixmap_if_changed(w, QPixmap())
                    w.setText("")

            w.show()
            if not actively_scrolling:
                pv = self.parent_viewer
                if pv is not None and hasattr(pv, "_is_gallery_path_selected"):
                    if hasattr(w, "set_gallery_selected"):
                        w.set_gallery_selected(pv._is_gallery_path_selected(path))
                if pv is not None and hasattr(pv, "_is_gallery_path_bookmarked"):
                    if hasattr(w, "set_gallery_bookmarked"):
                        w.set_gallery_bookmarked(pv._is_gallery_path_bookmarked(path))
                if pv is not None and hasattr(pv, "_burst_stack_count_for_path"):
                    if hasattr(w, "set_burst_stack_count"):
                        w.set_burst_stack_count(pv._burst_stack_count_for_path(path))
                if pv is not None and hasattr(pv, "_is_gallery_path_edited"):
                    if hasattr(w, "set_gallery_edited"):
                        w.set_gallery_edited(pv._is_gallery_path_edited(path))
            thumb_missing = not cache_hit
            if thumb_missing and self._thumb_fail_counts.get(path, 0) >= 3:
                thumb_missing = False
            
            m = self._metadata_cache.get(path)
            exif_missing = not m or m.get("original_width") is None or m.get("original_height") is None
            
            stages = set()
            if thumb_missing:
                stages.add("thumbnail")
            if exif_missing:
                stages.add("exif")
                
            if stages and path not in self._active_tasks:
                target = QSize(physical_size.width(), physical_size.height())
                load_tasks.append((path, Priority.CURRENT, stages, target))

        if actively_scrolling and not load_tasks and not self._active_tasks:
            viewport_rect = QRect(0, scroll_y, v_port.width(), v_h)
            viewport_items = self._get_visible_range(viewport_rect)
            viewport_paths = {
                item["file_path"] for _, item in viewport_items if item.get("file_path")
            }
            if self._paths_have_base_thumbnails(viewport_paths):
                self._idle_preload_timer.start(1000)
                return

        if center_paths:
            for path in center_paths:
                if not path or path in visible_paths:
                    continue
                stages = self._missing_load_stages_for_path(path, jpeg_only=True)
                if stages and path not in self._active_tasks:
                    load_tasks.append((path, Priority.PRELOAD_NEXT, stages))

        if allow_prefetch:
            for path in prefetch_paths:
                if not path or path in visible_paths:
                    continue
                
                # Preload if either thumbnail or metadata is missing
                thumb_missing = not self._thumbnail_cache.get((path, self._thumb_base_key))
                
                m = self._metadata_cache.get(path)
                exif_missing = not m or m.get("original_width") is None or m.get("original_height") is None
                
                stages = set()
                if thumb_missing:
                    stages.add("thumbnail")
                if exif_missing:
                    stages.add("exif")
                
                if stages and path not in self._active_tasks:
                    load_tasks.append((path, Priority.PRELOAD_NEXT, stages))

        # Closest to prefetch center first (bidirectional up/down), then non-RAW.
        load_tasks.sort(
            key=lambda item: (
                self._path_layout_distance(item[0], center_idx),
                1 if is_raw_file(item[0]) else 0,
            )
        )

        current_reserve = _gallery_current_active_reserve(active_cap)
        prefetch_active_cap = max(0, active_cap - current_reserve)
        has_pending_current = any(
            pri == Priority.CURRENT for _, pri, *_ in load_tasks
        )

        # Schedule with budget and target-sized thumbnails for visible tiles.
        scheduled = 0
        if len(self._active_tasks) >= active_cap:
            if has_pending_current:
                self._evict_stale_active_tasks(drop_prefetch=True)
            elif self._evict_stale_active_tasks(max_age_s=2.0) == 0:
                self._evict_stale_active_tasks(drop_prefetch=True)
            has_pending_current = any(
                pri == Priority.CURRENT for _, pri, *_ in load_tasks
            )
        if len(self._active_tasks) >= active_cap and not has_pending_current:
            logger.info(
                "[GALLERY] load_visible_images deferred; active_cap=%d active=%d visible=%d",
                active_cap,
                len(self._active_tasks),
                len(visible_indices_items),
            )
            if _focus_gallery_switch_logs():
                logger.debug(
                    "[MODESWITCH] gallery.load_visible_images skipped scheduling; active cap reached (%d)",
                    len(self._active_tasks),
                )
            self._request_load_visible_images(150)
            return
        for path, priority, stages, *rest in load_tasks:
            if scheduled >= max_tasks:
                if not self._load_timer.isActive():
                    self._load_timer.start(16)
                break
            if priority != Priority.CURRENT:
                if self._prefetch_active_count() >= prefetch_active_cap:
                    continue
            if len(self._active_tasks) >= active_cap:
                if priority != Priority.CURRENT:
                    continue
                break
            # NEVER pass thumbnail_target_size for gallery tiles. The worker would
            # crop-fit the already display-oriented thumbnail to the tile rect
            # (_handle_thumbnail_result, fit="crop"); when the rect is still the
            # pre-metadata landscape default, that turns an upright portrait into an
            # upright-content LANDSCAPE crop — and since tiles are ~3:2, its aspect is
            # indistinguishable from a genuine sideways full preview, so
            # _orient_gallery_base_pixmap "corrects" it by rotating again -> sideways
            # content inside a correct portrait frame. The gallery caches the payload
            # as its BASE and crop-fits per tile itself (_fit_tile_pixmap), so the
            # worker-side crop was redundant as well as destructive.
            self.load_manager.load_image(
                path,
                priority=priority,
                cancel_existing=False,
                stages=stages,
                gallery_thumbnail=True,
                thumbnail_target_size=None,
            )
            self._mark_active_task(path, priority)
            scheduled += 1
        scheduled_tasks = scheduled

        if scheduled_tasks > 0:
            logger.info(
                "[GALLERY] load_visible_images scheduled=%d visible=%d active=%d pending=%d",
                scheduled_tasks,
                len(visible_indices_items),
                len(self._active_tasks),
                len(load_tasks),
            )
        if _focus_gallery_switch_logs():
            logger.debug(
                "[MODESWITCH] gallery.load_visible_images scheduled=%d visible=%d active=%d",
                scheduled_tasks,
                len(visible_indices_items),
                len(self._active_tasks),
            )

        # While gallery tiles are actively being scheduled, pause background metadata/
        # semantic indexing (same mechanism as the scroll-burst pause): the indexer
        # competes for the same slow drive and writes the EXIF DB the rebuild path
        # reads. Each scheduling burst re-arms the resume timer, so indexing resumes
        # ~5s after tile loading quiets down.
        if scheduled_tasks > 0 and self.parent_viewer is not None:
            try:
                pause = getattr(
                    self.parent_viewer, "_pause_semantic_indexing_deferred", None
                )
                if pause is not None:
                    pause()
                timer = getattr(self.parent_viewer, "_resume_indexing_timer", None)
                if timer is not None:
                    timer.start(5000)
            except Exception:
                pass

        # If everything visible/prefetched is already loaded, schedule idle background preloading
        if scheduled_tasks == 0 and len(self._active_tasks) == 0:
            if self._visible_viewport_needs_thumbnails():
                self._request_load_visible_images(300)
            else:
                self._idle_preload_timer.start(1000)  # 1 second of sustained idle
        elif scheduled_tasks == 0 and len(self._active_tasks) > 0:
            self._idle_preload_timer.stop()
            if not self._load_timer.isActive():
                self._request_load_visible_images(200)
        else:
            self._idle_preload_timer.stop()

    def pause_idle_preload(self):
        """Immediately pause idle background thumbnail generation."""
        if hasattr(self, "_idle_preload_timer") and self._idle_preload_timer.isActive():
            self._idle_preload_timer.stop()
        if self.load_manager:
            idle_priority = _gallery_idle_load_priority()
            self.load_manager.cancel_tasks_by_priority(idle_priority)


    def _preload_remaining_thumbnails_background(self):
        """Silently preload thumbnails for out-of-viewport images during idle stages to smooth out future scrolling."""
        if self._building or not self.images or self.load_manager is None:
            return

        try:
            from common_image_loader import io_pressure_active

            if io_pressure_active():
                return
        except Exception:
            pass
        
        # Guard: do not run if not in gallery view or if there are already active tasks in flight
        pv = getattr(self, "parent_viewer", None)
        if pv and getattr(pv, "view_mode", "single") != "gallery":
            return
        if len(self._active_tasks) > 0:
            return

        # Bidirectional embedded-JPEG prefetch from current image / viewport center.
        start_index = 0
        center_idx = self._gallery_prefetch_center_index(scroll_y=0, viewport_h=400)
        try:
            p = self._scroll_area
            if p is None:
                p = self.parent()
                while p and not isinstance(p, QScrollArea):
                    p = p.parent()
            scroll_y = 0
            viewport_h = 400
            if isinstance(p, QScrollArea):
                scroll_y = p.verticalScrollBar().value()
                viewport_h = max(1, p.viewport().height())
            center_idx = self._gallery_prefetch_center_index(scroll_y, viewport_h)
            if center_idx is not None:
                start_index = center_idx
            elif isinstance(p, QScrollArea):
                for idx, item in enumerate(self._gallery_layout_items):
                    if item["rect"].bottom() > scroll_y:
                        start_index = idx
                        break
        except Exception:
            if center_idx is not None:
                start_index = center_idx

        # Search outwards from start_index: alternately checking next and previous indices
        n_items = len(self._gallery_layout_items)
        outwards_indices = []
        for offset in range(1, n_items):
            nxt = start_index + offset
            prev = start_index - offset
            if nxt < n_items:
                outwards_indices.append(nxt)
            if prev >= 0:
                outwards_indices.append(prev)

        from image_cache import get_image_cache
        global_cache = get_image_cache()

        preload_batch = []
        max_preload_batch = _gallery_idle_preload_batch()
        if len(self.images or []) > 500:
            max_preload_batch = min(max_preload_batch, 24)

        for idx in outwards_indices:
            item = self._gallery_layout_items[idx]
            path = item.get("file_path")
            if not path:
                continue

            # Skip if already in memory thumbnail cache
            if self._thumbnail_cache.get((path, self._thumb_base_key)) is not None:
                continue

            # Skip if already in global cache (either thumbnail or grid)
            if global_cache.get_thumbnail(path) is not None or global_cache.get_grid(path) is not None:
                continue

            # Skip if already being processed or active
            if path in self._active_tasks:
                continue

            preload_batch.append(path)
            if len(preload_batch) >= max_preload_batch:
                break

        if preload_batch:
            idle_priority = _gallery_idle_load_priority()
            # Embedded JPEG only — fast bidirectional scroll cushion above/below center.
            for path in preload_batch:
                self.load_manager.load_image(
                    path,
                    priority=idle_priority,
                    cancel_existing=False,
                    stages={"thumbnail"},
                    gallery_thumbnail=True,
                )
                self._mark_active_task(path, idle_priority)
            
            # Schedule next batch after a short delay (progressive background loading)
            self._idle_preload_timer.start(_gallery_idle_preload_ms())

    def warm_thumbnails_from_global_cache(self, paths: List[str]) -> int:
        """Seed gallery pixmap cache from global ImageCache (e.g. after film strip)."""
        if not paths:
            return 0
        from image_cache import get_image_cache

        global_cache = get_image_cache()
        warmed = 0
        image_set = set(self.images) if self.images else set()
        for path in paths:
            if not path or path not in image_set:
                continue
            if self._thumbnail_cache.get((path, self._thumb_base_key)):
                continue
            thumb = global_cache.get_thumbnail(path)
            if thumb is None:
                thumb = global_cache.get_grid(path)
            if thumb is None:
                continue
            thumb = self._orient_gallery_thumbnail_array(path, thumb)
            pixmap = _thumbnail_data_to_base_pixmap(thumb)
            if pixmap is None or pixmap.isNull():
                continue
            pixmap = self._store_oriented_base_pixmap(path, pixmap)
            self._reconcile_tile_aspect(path, pixmap)
            warmed += 1
        if warmed and _focus_gallery_switch_logs():
            logger.debug(
                "[GALLERY] Warmed %d tile(s) from global thumbnail cache", warmed
            )
        return warmed

    def on_thumbnail_ready(self, file_path, thumbnail_data):
        if self._gallery_folder_superseded():
            return
        # Mark path as no longer in-flight regardless of whether we can render it now.
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]
        # Single-image view uses RAWImageViewer.on_manager_thumbnail_ready; skip gallery
        # work to avoid duplicate handling and any re-entrant UI paths on the same signal.
        if self.parent_viewer is not None and getattr(
            self.parent_viewer, "view_mode", "gallery"
        ) != "gallery":
            return
        if file_path not in self._path_to_indices:
            return
        if self._building or self._gallery_warmup_active() or self._thumb_ready_depth > 0:
            QTimer.singleShot(
                0,
                lambda fp=file_path, td=thumbnail_data: self._apply_thumbnail_ready(fp, td),
            )
            return
        self._apply_thumbnail_ready(file_path, thumbnail_data)

    def _apply_thumbnail_ready(self, file_path, thumbnail_data):
        if self._thumb_ready_depth > 0:
            QTimer.singleShot(
                0,
                lambda fp=file_path, td=thumbnail_data: self._apply_thumbnail_ready(fp, td),
            )
            return
        self._thumb_ready_depth += 1
        try:
            self._apply_thumbnail_ready_impl(file_path, thumbnail_data)
        finally:
            self._thumb_ready_depth = max(0, self._thumb_ready_depth - 1)

    def _apply_thumbnail_ready_impl(self, file_path, thumbnail_data):
        resolved = self._resolve_gallery_path(file_path)
        if not resolved or resolved not in self._path_to_indices:
            return
        file_path = resolved

        pixmap = _thumbnail_data_to_base_pixmap(thumbnail_data)

        if not pixmap or pixmap.isNull():
            self.on_thumbnail_error(file_path, "Null pixmap in on_thumbnail_ready")
            return

        pixmap = self._orient_gallery_base_pixmap(file_path, pixmap)

        existing = self._thumbnail_cache.get((file_path, self._thumb_base_key))
        if existing and not existing.isNull():
            old_dim = max(existing.width(), existing.height())
            new_dim = max(pixmap.width(), pixmap.height())
            if old_dim > new_dim:
                pixmap = existing

        meta_ar = self._metadata_display_aspect(file_path)
        if meta_ar is not None and pixmap.height() > 0:
            px_ar = pixmap.width() / pixmap.height()
            if abs(px_ar - meta_ar) / max(meta_ar, 0.01) > 0.35:
                existing = self._thumbnail_cache.get((file_path, self._thumb_base_key))
                if existing and not existing.isNull() and existing.height() > 0:
                    ex_ar = existing.width() / existing.height()
                    if abs(ex_ar - meta_ar) < abs(px_ar - meta_ar):
                        pixmap = existing
        pixmap = self._store_oriented_base_pixmap(file_path, pixmap)
        self._thumb_fail_counts.pop(file_path, None)
        if not self._first_thumb_ready_after_set:
            self._first_thumb_ready_after_set = True
            self._end_gallery_load_warmup_throttle()

        now = time.time()
        if self._thumb_first_after_scroll_t is not None:
            self._thumb_first_after_scroll_t = None
        if self._thumb_first_after_settle_t is not None and self._last_scroll_settle_t > 0:
            self._thumb_first_after_settle_t = None

        # Record tile geometry from the oriented thumbnail pixels — ground truth, like the
        # filmstrip. This MUST run before the fast-scroll early-return: with fast byte-scan
        # decode many thumbnails complete mid-fast-scroll, and skipping it leaves a portrait
        # image in a landscape frame (the initial metadata/default aspect) after the tile is
        # later filled on settle. Recording the aspect + arming the (debounced) rebuild timer
        # is cheap; _handle_metadata_rebuild defers itself while fast-scrolling, so the actual
        # relayout still waits for scroll to settle.
        self._reconcile_tile_aspect(file_path, pixmap)

        frame_mismatch = self._any_visible_tile_frame_mismatch(file_path, pixmap)

        # Avoid heavy per-widget pixmap work during very fast scrolling; visible tiles are
        # filled on settle via load_visible_images. Tile geometry is already recorded above.
        if self._is_scrolling_fast:
            return

        if frame_mismatch:
            if _orient_debug_enabled():
                logger.info(
                    "[ORIENT] defer paint file=%s until layout rebuild (frame mismatch)",
                    os.path.basename(file_path),
                )
            if not self._metadata_rebuild_timer.isActive():
                self._metadata_rebuild_timer.start(0 if not self._is_scrolling_fast else 250)
            return

        # Apply the pixmap to visible tiles. Under batching, defer the (scale/crop +
        # setPixmap) work to the drain timer so a burst of completions can't stall a
        # scroll frame; the base pixmap is already cached above, so drain re-reads it.
        if _batch_tile_apply_enabled():
            self._pending_tile_applies[file_path] = True
            if not self._tile_apply_timer.isActive():
                self._tile_apply_timer.start(0)
        else:
            self._apply_tile_pixmap_now(file_path)

        # Continue scheduling when new thumbnails arrive so visible blanks are filled quickly.
        if not self._is_scrolling_fast:
            self._request_load_visible_images(25)

    def _apply_tile_pixmap_now(self, file_path):
        """Scale/crop the cached base pixmap onto every visible tile showing file_path."""
        base = self._thumbnail_cache.get((file_path, self._thumb_base_key))
        if not base or base.isNull():
            return
        indices = self._path_to_indices.get(file_path, [])
        if not indices:
            return
        dpr = self.devicePixelRatio()
        for idx in indices:
            if idx in self._visible_widgets:
                w = self._visible_widgets[idx]
                if w.file_path != file_path:
                    continue
                logical_size = w.size()
                physical_size = QSize(int(logical_size.width() * dpr), int(logical_size.height() * dpr))
                if not self._frame_matches_pixmap_orientation(
                    self._gallery_layout_items[idx].get("rect")
                    if idx < len(self._gallery_layout_items)
                    else None,
                    base,
                ):
                    if _orient_debug_enabled():
                        logger.info(
                            "[ORIENT] skip paint file=%s idx=%d (frame mismatch, awaiting rebuild)",
                            os.path.basename(file_path),
                            idx,
                        )
                    continue
                fitted = self._fit_tile_pixmap(file_path, base, physical_size, dpr)
                if _orient_debug_enabled():
                    try:
                        rect = None
                        if idx < len(self._gallery_layout_items):
                            rect = self._gallery_layout_items[idx].get("rect")
                        b_o = "P" if base.height() > base.width() else "L"
                        r_o = "?" if rect is None else ("P" if rect.height() > rect.width() else "L")
                        f_o = "P" if fitted.height() > fitted.width() else "L"
                        flag = " FRAME_MISMATCH" if (rect is not None and r_o != b_o) else ""
                        logger.info(
                            "[ORIENT] paint file=%s base=%dx%d(%s) rect=%s(%s) fitted=%dx%d(%s)%s",
                            os.path.basename(file_path), base.width(), base.height(), b_o,
                            "None" if rect is None else f"{rect.width()}x{rect.height()}", r_o,
                            fitted.width(), fitted.height(), f_o, flag,
                        )
                    except Exception:
                        pass
                w.setPixmap(fitted)
                w.setText("")

    def _drain_tile_applies(self):
        """Apply a bounded batch of pending tile pixmaps per event-loop tick."""
        if self._gallery_folder_superseded():
            self._pending_tile_applies.clear()
            return
        batch = _tile_apply_batch_size()
        count = 0
        while self._pending_tile_applies and count < batch:
            file_path = next(iter(self._pending_tile_applies))
            del self._pending_tile_applies[file_path]
            # Skip while fast-scrolling: tiles are filled on settle via load_visible_images.
            if self._is_scrolling_fast:
                continue
            self._apply_tile_pixmap_now(file_path)
            count += 1
        if self._pending_tile_applies and not self._is_scrolling_fast:
            self._tile_apply_timer.start(0)

    def on_task_completed(self, file_path):
        """Always clear active tasks when the background worker finishes."""
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]
        if not self._building and not self._is_scrolling_fast:
            self._request_load_visible_images(40)

    def _on_retry_timer_timeout(self):
        self._max_retry_attempt = 1
        self._request_load_visible_images(40)

    def on_thumbnail_error(self, file_path, error_msg):
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]

        if file_path not in self.images:
            return

        lower_err = (error_msg or "").lower()
        clearly_unsupported = any(
            term in lower_err
            for term in (
                "unsupported file format",
                "not recognized",
                "unsupported or corrupt",
            )
        )
        attempt = self._thumb_fail_counts.get(file_path, 0) + 1
        self._thumb_fail_counts[file_path] = attempt
        max_retries = 3

        is_emfile = "too many open files" in lower_err or "emfile" in lower_err
        if is_emfile:
            pv = self.parent_viewer
            if pv is not None and hasattr(pv, "_enter_io_pressure_mode"):
                pv._enter_io_pressure_mode("gallery_thumb_emfile")
            self._max_retry_attempt = max(getattr(self, "_max_retry_attempt", 1), 6)
        else:
            self._max_retry_attempt = max(getattr(self, "_max_retry_attempt", 1), attempt)

        # JPEG/PNG/WebP: keep in gallery on thumbnail-only failures; retry transient decode errors.
        if not is_raw_file(file_path):
            if attempt <= max_retries:
                logger.debug(
                    "[GALLERY] Thumbnail failed for %s (attempt %d/%d): %s",
                    os.path.basename(file_path),
                    attempt,
                    max_retries,
                    error_msg,
                )
                if not self._retry_timer.isActive():
                    self._retry_timer.start(min(500 * self._max_retry_attempt, 3000))
                return

            null_pixmap = QPixmap(100, 100)
            null_pixmap.fill(Qt.GlobalColor.darkGray)
            self._thumbnail_cache.put((file_path, self._thumb_base_key), null_pixmap)
            logger.warning(
                "[GALLERY] Keeping %s after repeated thumbnail failures: %s",
                os.path.basename(file_path),
                error_msg,
            )
            return

        if attempt < max_retries and not clearly_unsupported:
            logger.debug(
                "[GALLERY] RAW thumbnail failed for %s (attempt %d/%d): %s",
                os.path.basename(file_path),
                attempt,
                max_retries,
                error_msg,
            )
            if not self._retry_timer.isActive():
                self._retry_timer.start(min(500 * self._max_retry_attempt, 3000))
            return

        if not clearly_unsupported and attempt < max_retries + 1:
            null_pixmap = QPixmap(100, 100)
            null_pixmap.fill(Qt.GlobalColor.darkGray)
            self._thumbnail_cache.put((file_path, self._thumb_base_key), null_pixmap)
            logger.warning(
                "[GALLERY] Keeping RAW %s after thumbnail failures: %s",
                os.path.basename(file_path),
                error_msg,
            )
            return

        # Only drop RAW files that are clearly unsupported after retries are exhausted.
        removed = False
        if file_path in self.images:
            self.images.remove(file_path)
            removed = True

        pv = self.parent_viewer
        if pv is not None:
            if hasattr(pv, "image_files") and file_path in pv.image_files:
                try:
                    pv.image_files.remove(file_path)
                    removed = True
                except ValueError:
                    pass
            if removed and hasattr(pv, "_update_gallery_counter"):
                try:
                    pv._update_gallery_counter()
                except Exception:
                    pass

        if removed:
            logger.info(
                "[GALLERY] Removed unsupported RAW from list: %s (%s)",
                file_path,
                error_msg,
            )
            self.build_gallery(force=True)

        if self.parent_viewer is not None and getattr(
            self.parent_viewer, "view_mode", "gallery"
        ) == "gallery":
            if not self._retry_timer.isActive():
                self._retry_timer.start(100)

    def on_exif_ready(self, file_path, exif_data):
        """Update aspect ratio in layout when metadata arrives."""
        resolved = self._resolve_gallery_path(file_path)
        if not resolved:
            return
            
        if not exif_data:
            # Save dummy data so we don't infinitely request it
            self._metadata_cache[resolved] = {"original_width": 0, "original_height": 0}
            return

        # Store in local metadata cache (keep container orientation + sensor dimensions).
        self._metadata_cache[resolved] = exif_data
        try:
            from image_cache import get_image_cache

            get_image_cache().put_exif(file_path, exif_data)
        except Exception:
            pass
        
        # Calculate real aspect ratio
        w = exif_data.get("original_width")
        h = exif_data.get("original_height")
        if not w or not h or h <= 0:
            return
            
        aspect = self._metadata_display_aspect(file_path)
        if aspect is None:
            return

        raw_meta = self._raw_aspect_undo_user_rotation(file_path, aspect)
        stored_raw = self._measured_raw_aspects.get(resolved)
        if stored_raw is None or (
            self._aspect_orientation_flag(stored_raw) != 0
            and self._aspect_orientation_flag(raw_meta) != 0
            and self._aspect_orientation_flag(stored_raw)
            != self._aspect_orientation_flag(raw_meta)
        ):
            self._measured_raw_aspects[resolved] = raw_meta
            base = self._thumbnail_cache.get((resolved, self._thumb_base_key))
            if base and not base.isNull():
                self._store_oriented_base_pixmap(resolved, base)
        
        # Check if we need to update layout (if it differs from default 1.333 or previous cache)
        # Find all occurrences of this path in layout
        indices = self._path_to_indices.get(file_path, [])
        changed = False
        geometry_fix = False
        for idx in indices:
            if idx < len(self._gallery_layout_items):
                old_aspect = self._gallery_layout_items[idx].get("aspect", 1.5)
                if abs(old_aspect - aspect) > 0.05:
                    if self._layout_aspect_needs_rebuild(old_aspect, aspect):
                        geometry_fix = True
                    self._gallery_layout_items[idx]["aspect"] = aspect
                    changed = True
        
        if changed:
            self.refresh_visible_tile_for_path(file_path)
            self._add_metadata_changed_path(file_path)
            if self._gallery_warmup_active() or self._building:
                return

            large = len(self.images) > 800
            rebuild_threshold = 12 if large else (5 if len(self.images) < 100 else 15)

            if geometry_fix and not self._metadata_rebuild_timer.isActive():
                debounce = 2500 if large else 400
                logger.debug(
                    "[GALLERY_DEBUG] Timer started by on_exif_ready (geometry) for %s",
                    file_path,
                )
                self._metadata_rebuild_timer.start(debounce)
            elif not self._metadata_rebuild_timer.isActive():
                if large:
                    debounce = 4000
                elif len(self._metadata_cache) < (len(self.images) * 0.5):
                    debounce = 2000
                else:
                    debounce = 800
                logger.debug(f"[GALLERY_DEBUG] Timer started by on_exif_ready (long) for {file_path}")
                self._metadata_rebuild_timer.start(debounce)
            elif len(self._metadata_changed_paths) >= rebuild_threshold:
                logger.debug(f"[GALLERY_DEBUG] Timer started by on_exif_ready (short) for {file_path}")
                self._metadata_rebuild_timer.start(1500 if large else 500)

    def _schedule_aspects_settle_rebuild(self, delay_ms: int = 3500) -> None:
        """One deferred full rebuild so justified rows match final EXIF/thumbnail aspects."""
        if len(self.images) < 50:
            return
        if not self._metadata_changed_paths:
            return
        n = len(self.images)
        if n >= 1000:
            delay_ms = max(delay_ms, 12000)
        elif n >= 500:
            delay_ms = max(delay_ms, 8000)
        elif n >= 200:
            delay_ms = max(delay_ms, 5000)
        if self._gallery_warmup_active():
            warmup_left_ms = int(
                max(0.0, float(self._gallery_warmup_until) - time.time()) * 1000
            )
            delay_ms = max(delay_ms, warmup_left_ms + 2000)
        if self._aspects_settle_timer.isActive():
            return
        self._aspects_settle_timer.start(max(500, int(delay_ms)))

    def _on_aspects_settle_rebuild(self) -> None:
        if not self.images or self._building:
            self._schedule_aspects_settle_rebuild(2000)
            return
        if self._is_scrolling_fast:
            self._schedule_aspects_settle_rebuild(1500)
            return
        if self._gallery_warmup_active():
            self._schedule_aspects_settle_rebuild(2000)
            return
        if not self._metadata_changed_paths:
            return
        if _focus_gallery_switch_logs():
            logger.debug(
                "[GALLERY] aspects settle rebuild (%d aspect drift)",
                len(self._metadata_changed_paths),
            )
        self._metadata_changed_paths.clear()
        self.build_gallery(force=True)

    def _image_index_for_path(self, file_path: str) -> int:
        for i, item in enumerate(self.images):
            if item == file_path:
                return i
        return -1

    def _layout_row_start_index(self, layout_idx: int) -> int:
        """First layout-item index in the same justified row as *layout_idx*."""
        if layout_idx <= 0 or layout_idx >= len(self._gallery_layout_items):
            return max(0, layout_idx)
        row_top = self._gallery_layout_items[layout_idx]["rect"].top()
        start = layout_idx
        for i in range(layout_idx - 1, -1, -1):
            if self._gallery_layout_items[i]["rect"].top() < row_top:
                break
            start = i
        return start

    def _partial_rebuild_start_image_index(self, changed_paths: set[str]) -> Optional[int]:
        """Earliest image index whose justified row must be recomputed."""
        if not changed_paths:
            return None
        min_image_idx = len(self.images)
        for path in changed_paths:
            idx = self._image_index_for_path(path)
            if idx < 0:
                continue
            layout_indices = self._path_to_indices.get(path) or []
            if not layout_indices:
                min_image_idx = min(min_image_idx, idx)
                continue
            row_start = self._layout_row_start_index(min(layout_indices))
            row_path = self._gallery_layout_items[row_start].get("file_path")
            if isinstance(row_path, str):
                row_idx = self._image_index_for_path(row_path)
                if row_idx >= 0:
                    min_image_idx = min(min_image_idx, row_idx)
        if min_image_idx >= len(self.images):
            return None
        return min_image_idx

    def _rebuild_layout_from_image_index(
        self, start_image_idx: int, anchor_state: Optional[dict] = None
    ) -> bool:
        """Recompute justified rows from *start_image_idx* onward (partial layout update)."""
        if (
            start_image_idx <= 0
            or not self.images
            or not self._gallery_layout_items
            or self._building
        ):
            return False

        viewport_width = self._get_viewport_width()
        left_margin = 24
        right_margin = 0
        net_width = viewport_width - left_margin - right_margin
        if net_width <= 0:
            return False

        start_path = self.images[start_image_idx]
        if not isinstance(start_path, str):
            return False
        layout_indices = self._path_to_indices.get(start_path) or []
        if not layout_indices:
            return False
        row_start_layout = self._layout_row_start_index(min(layout_indices))
        row_path = self._gallery_layout_items[row_start_layout].get("file_path")
        rebuild_from = start_image_idx
        if isinstance(row_path, str):
            row_image_idx = self._image_index_for_path(row_path)
            if row_image_idx >= 0:
                rebuild_from = row_image_idx

        current_y = self._gallery_layout_items[row_start_layout]["rect"].top()
        self._gallery_layout_items = self._gallery_layout_items[:row_start_layout]

        row: list = []
        aspect_sum = 0.0

        def commit_row(r, a_sum, is_last):
            nonlocal current_y
            if not r:
                return
            spacing = (len(r) - 1) * self.MIN_SPACING
            row_h = self.TARGET_ROW_HEIGHT if is_last else (net_width - spacing) / a_sum
            row_h = max(
                self.TARGET_ROW_HEIGHT * 0.4,
                min(self.TARGET_ROW_HEIGHT * 2.2, row_h),
            )

            curr_x = left_margin
            for i, (item, aspect) in enumerate(r):
                w = int(row_h * aspect)
                if not is_last and i == len(r) - 1:
                    w = net_width - (curr_x - left_margin)
                self._gallery_layout_items.append(
                    {
                        "rect": QRect(curr_x, int(current_y), int(w), int(row_h)),
                        "file_path": item if isinstance(item, str) else None,
                        "aspect": aspect,
                    }
                )
                curr_x += w + self.MIN_SPACING
            current_y += row_h + self.MIN_SPACING

        for item in self.images[rebuild_from:]:
            aspect = 1.5
            if isinstance(item, str):
                aspect = self._layout_aspect_for_path(item)
            row.append((item, aspect))
            aspect_sum += aspect
            if (aspect_sum * self.TARGET_ROW_HEIGHT + (len(row) - 1) * self.MIN_SPACING) >= net_width:
                commit_row(row, aspect_sum, False)
                row, aspect_sum = [], 0.0

        if row:
            commit_row(row, aspect_sum, True)

        self._path_to_indices = {}
        for i, layout_item in enumerate(self._gallery_layout_items):
            p = layout_item["file_path"]
            if p not in self._path_to_indices:
                self._path_to_indices[p] = []
            self._path_to_indices[p].append(i)

        self._total_content_height = int(current_y + 20)
        self._sync_content_geometry()
        scroll = self._scroll_area_widget()
        if scroll is not None:
            sb = scroll.verticalScrollBar()
            if sb.value() > sb.maximum():
                sb.setValue(max(sb.minimum(), sb.maximum()))
        self.update()
        self._last_build_ts = time.time()
        self._last_layout_content_height = self._total_content_height
        self._layout_image_sequence = tuple(self.images)
        self._last_force_build_ts = self._last_build_ts
        self._reposition_thumbnail_widgets()
        if anchor_state:
            self._apply_scroll_anchor_state(anchor_state)
        return True

    def _handle_metadata_rebuild(self):
        """Rebuild layout after metadata changes to settle aspect ratios."""
        large = len(self.images) > 800
        flip_count = len(getattr(self, "_orientation_flip_paths", None) or ())
        flip_pending = flip_count > 0
        if not self._metadata_changed_paths or self._building:
            return
        if self._is_scrolling_fast and not flip_pending:
            # Don't rebuild while scrolling as it blocks the UI thread
            if not self._metadata_rebuild_timer.isActive():
                self._metadata_rebuild_timer.start(1000)
            return
        time_since_last_change = time.time() - getattr(self, "_last_metadata_change_time", 0.0)
        if (
            large
            and not flip_pending
            and len(self._metadata_changed_paths) < 8
            and time_since_last_change < 1.5
        ):
            self._metadata_rebuild_timer.start(1200)
            return
        if not flip_pending and (time.time() - self._last_force_build_ts) < 0.8:
            self._metadata_rebuild_timer.start(400)
            return

        anchor = self._capture_scroll_anchor_state()
        changed_paths = set(self._metadata_changed_paths)
        flip_paths = set(getattr(self, "_orientation_flip_paths", None) or ())
        self._metadata_changed_paths.clear()
        if hasattr(self, "_orientation_flip_paths"):
            if _orient_debug_enabled() and flip_count:
                logger.info(
                    "[ORIENT] metadata rebuild starting (%d orientation flip(s))",
                    flip_count,
                )
            self._orientation_flip_paths.clear()
        if _focus_gallery_switch_logs():
            logger.debug("[GALLERY] metadata rebuild triggered")

        rebuild_t0 = time.time()
        partial_paths = flip_paths or changed_paths
        partial_start = self._partial_rebuild_start_image_index(partial_paths)
        use_partial = (
            partial_start is not None
            and partial_start > 0
            and len(partial_paths) <= max(64, len(self.images) // 50)
        )
        if use_partial:
            if not self._rebuild_layout_from_image_index(partial_start, anchor_state=anchor):
                self.build_gallery(bulk_metadata=None, force=True)
        else:
            self.build_gallery(bulk_metadata=None, force=True)
        rebuild_ms = (time.time() - rebuild_t0) * 1000.0
        if rebuild_ms > 250:
            # Main-thread rebuild cost is the prime suspect for periodic gallery stalls
            # on large folders — surface it so slow rebuilds are visible in logs.
            logger.info(
                "[GALLERY] metadata rebuild took %.0f ms (%d items)",
                rebuild_ms,
                len(self._gallery_layout_items),
            )

    def show_loading_message(self, message="Loading gallery..."):
        """Show loading message overlay - Simplified for better performance"""
        # Remove existing loading label if any
        if self._loading_label:
            # Update text if already visible
            self._loading_label.setText(message)
            self._loading_label.adjustSize()
            self._update_loading_label_geometry()
            return
        
        # Create loading label - smaller, bottom-right toast style
        self._loading_label = QLabel(message, self)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 20, 200);
                color: rgba(255, 255, 255, 220);
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
            }
        """)
        font = QFont()
        font.setPointSize(10)
        self._loading_label.setFont(font)
        self._loading_label.show()
        self._loading_label.raise_()  # Bring to front
        
        # Update geometry
        self._update_loading_label_geometry()
    
    def _update_loading_label_geometry(self):
        """Update loading label geometry - Bottom Center"""
        if self._loading_label and self.parent_viewer and self.parent_viewer.width() > 0:
            self._loading_label.adjustSize()
            w = self._loading_label.width()
            h = self._loading_label.height()
            
            # Position at bottom center of the viewport
            parent_scroll = self.parent_viewer.scroll_area if hasattr(self.parent_viewer, 'scroll_area') else None
            if parent_scroll:
                 # Calculate relative position in the viewport
                 viewport_h = parent_scroll.viewport().height()
                 scroll_y = parent_scroll.verticalScrollBar().value()
                 
                 # Stick to bottom of viewport
                 y = scroll_y + viewport_h - h - 20
                 x = (self.width() - w) // 2
                 
                 self._loading_label.move(x, int(y))
            else:
                 # Fallback
                 x = (self.width() - w) // 2
                 y = self.height() - h - 20
                 self._loading_label.move(x, y)
    
    def hide_loading_message(self):
        """Hide loading message overlay"""
        if self._loading_label:
            self._loading_label.hide()
            self._loading_label.deleteLater()
            self._loading_label = None

    def show_empty_message(self, message):
        """Show empty gallery message centered in the gallery frame."""
        self.hide_loading_message()

        style = """
            QLabel {
                color: #a8a8a8;
                font-size: 15px;
                background-color: transparent;
                padding: 24px;
            }
        """

        if hasattr(self, "_empty_label") and self._empty_label:
            self._empty_label.setText(message)
            self._empty_label.setAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._empty_label.setWordWrap(True)
            self._empty_label.setStyleSheet(style)
            self._update_empty_label_geometry()
            self._empty_label.show()
            self._empty_label.raise_()
            QTimer.singleShot(0, self._update_empty_label_geometry)
            return

        self._empty_label = QLabel(message, self)
        self._empty_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(style)
        font = QFont()
        font.setPointSize(12)
        self._empty_label.setFont(font)
        self._empty_label.show()
        self._empty_label.raise_()
        self._update_empty_label_geometry()
        QTimer.singleShot(0, self._update_empty_label_geometry)

    def hide_empty_message(self):
        """Hide empty gallery message"""
        if hasattr(self, '_empty_label') and self._empty_label:
            self._empty_label.hide()
            self._empty_label.deleteLater()
            self._empty_label = None

    def clear_thumbnail_widgets(self, *, destroy: bool = False):
        """Hide visible thumbnails; pool widgets unless ``destroy`` (folder clear)."""
        try:
            if self.load_manager is not None:
                if hasattr(self.load_manager, "flush_queue"):
                    self.load_manager.flush_queue()
                else:
                    self.load_manager.cancel_all_tasks()
        except Exception:
            pass
        self._active_tasks.clear()
        self._requested_thumbnail_paths.clear()

        for label in list(getattr(self, "_visible_widgets", {}).values()):
            try:
                label.hide()
                label.clear()
                label.setText("")
                label.file_path = None
                label.original_pixmap = None
                if hasattr(label, "set_gallery_selected"):
                    label.set_gallery_selected(False)
                if hasattr(label, "set_gallery_bookmarked"):
                    label.set_gallery_bookmarked(False)
                if destroy:
                    label.deleteLater()
                else:
                    self._widget_pool.append(label)
            except Exception:
                pass
        self._visible_widgets.clear()

        if destroy:
            for label in list(getattr(self, "_widget_pool", [])):
                try:
                    label.hide()
                    label.clear()
                    label.deleteLater()
                except Exception:
                    pass
            self._widget_pool = []
            try:
                for child in self.findChildren(ThumbnailLabel):
                    child.hide()
                    child.deleteLater()
            except Exception:
                pass

    def refresh_gallery_selection_visuals(self) -> None:
        """Sync selection border on visible pooled thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_is_gallery_path_selected"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_gallery_selected"):
                w.set_gallery_selected(pv._is_gallery_path_selected(path))

    def refresh_gallery_bookmark_visuals(self) -> None:
        """Sync bookmark star on visible pooled thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_is_gallery_path_bookmarked"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_gallery_bookmarked"):
                w.set_gallery_bookmarked(pv._is_gallery_path_bookmarked(path))

    def refresh_gallery_burst_visuals(self) -> None:
        """Sync burst stack count badges on visible thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_burst_stack_count_for_path"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_burst_stack_count"):
                w.set_burst_stack_count(pv._burst_stack_count_for_path(path))

    def refresh_gallery_edited_visuals(self) -> None:
        """Sync 'has saved RAW adjustments' badge on visible pooled thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_is_gallery_path_edited"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_gallery_edited"):
                w.set_gallery_edited(pv._is_gallery_path_edited(path))

    def _update_empty_label_geometry(self):
        """Lay out empty-state text across the justified gallery canvas (fills the frame when empty)."""
        if not getattr(self, "_empty_label", None):
            return
        margin_x = 32
        avail_w = max(120, int(self.width() - margin_x * 2))
        avail_h = max(120, self.height())

        label = self._empty_label
        label.setFixedWidth(avail_w)
        try:
            hint_h = int(label.heightForWidth(avail_w))
        except Exception:
            label.adjustSize()
            hint_h = int(label.sizeHint().height())
        label_h = min(max(hint_h, 48), avail_h - 16)
        y = max(8, (avail_h - label_h) // 2)
        x = margin_x
        label.setGeometry(int(x), int(y), avail_w, label_h)

    def _on_thumbnail_clicked(self, target_path, e):
        try:
            pv = self.parent_viewer
            if target_path and pv is not None:
                mods = (
                    pv._gallery_event_modifiers(e)
                    if hasattr(pv, "_gallery_event_modifiers")
                    else e.modifiers()
                )
                shift = (
                    hasattr(pv, "_gallery_has_shift_modifier")
                    and pv._gallery_has_shift_modifier(mods)
                )
                extend = (
                    hasattr(pv, "_gallery_has_extend_modifier")
                    and pv._gallery_has_extend_modifier(mods)
                )
                if shift and not extend:
                    if hasattr(pv, "_gallery_shift_click_selection"):
                        pv._gallery_shift_click_selection(target_path)
                    return
                if extend:
                    if hasattr(pv, "_gallery_toggle_path_selection"):
                        pv._gallery_toggle_path_selection(target_path)
                    return
                pv._gallery_item_clicked(target_path)
        except Exception:
            pass
