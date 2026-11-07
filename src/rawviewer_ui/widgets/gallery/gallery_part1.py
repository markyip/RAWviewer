"""Gallery mixin part 1 of 7 for JustifiedGallery."""
from __future__ import annotations

import contextlib
import logging
import sys
import time
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QScrollArea

from image_cache import LRUCache
from image_load_manager import Priority, get_image_load_manager
from rawviewer_ui.widgets import ThumbnailLabel

from .helpers import _env_int, _env_true

logger = logging.getLogger(__name__)

class GalleryPart1:

    TARGET_ROW_HEIGHT = 220

    HEIGHT_TOLERANCE = 0.25

    MIN_SPACING = 6

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
        self._measured_raw_aspects: dict[str, float] = {}
        # Paths whose EXIF was already looked up from the DB this folder (hit or miss);
        # build_gallery must not repeat main-thread bulk SQL for them on every rebuild.
        self._metadata_fetch_attempted: set[str] = set()
        self._orientation_flip_paths: set[str] = set()
        self._visible_widgets = {}  # {index: ThumbnailLabel}
        self._widget_pool = []
        self._total_content_height = 0

        # State Management
        self._building = False
        self._gallery_generation = 0
        self._active_tasks = {}  # path -> (started_ts, Priority)
        self._thumb_fail_counts: dict[str, int] = {}
        # Negative memo for the synchronous global-cache base fetch. A cold
        # section jump has no cached thumbnails, and the failed disk/SQLite
        # lookup costs ~5ms per tile — re-paying it for the same leading tiles
        # on every 16ms budget pass starved the rest of the viewport (blank
        # tiles until an input event triggered extra passes). Remember misses
        # briefly; async production clears the entry on thumbnail_ready.
        self._base_fetch_miss_ts: dict[str, float] = {}
        self._metadata_cache = {}
        # Thumbnail cache: base + per-bucket scaled. Count-capped via LRUCache
        # and additionally byte-capped by _enforce_thumbnail_cache_budget().
        self._thumbnail_cache = LRUCache(10000)
        self._thumb_cache_puts_since_check = 0
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
        self._last_reaim_t = 0.0
        # Paths showing a grey placeholder after repeated thumbnail failures.
        self._placeholder_thumb_paths: set = set()
        self._pending_scroll_to_path = None
        self._pending_scroll_anchor_state = None
        # True once the user has jumped far from the entry-point image (see
        # load_visible_images' big-jump handling); reset on each fresh set_images().
        self._entry_prefetch_abandoned = False
        self._is_zooming = False
        self._zoom_anchor_state = None
        self._gallery_zoom_rebuild = False
        self._is_scrollbar_dragging = False
        self._scroll_bubble_last_content_t = 0.0
        self._scroll_bubble_cached_text = ""
        self._scroll_bubble_cached_pixmap = None
        self._scroll_bubble_committed_path = None
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
        # Precise, not the default coarse timer: on Windows a coarse 8ms QTimer
        # fires with several ms of jitter (5% tolerance class + 15.6ms system
        # timer coalescing), and with a FIXED step per tick that cadence jitter
        # becomes velocity jitter -- uneven motion users read as "scrolling is
        # not smooth" even at high average FPS.
        self._wheel_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._wheel_timer.timeout.connect(self._apply_wheel_scroll_step)
        self._wheel_step_px = 18  # min px per nominal tick (tuned for smoothness)
        self._wheel_tick_ms = 8   # 125Hz-ish
        # Decay rate for the accumulator: consume 20% per nominal 8ms tick.
        # _apply_wheel_scroll_step scales this by the MEASURED elapsed time,
        # so a late tick consumes proportionally more -- constant on-screen
        # velocity regardless of timer cadence (frame-time independence, the
        # same idea as dt-scaled game-loop integration).
        self._wheel_decay_per_tick = 0.2
        self._wheel_last_tick_t = 0.0
        # Timestamp of the last native momentum-phase wheel event (macOS trackpad
        # decay). Motion from momentum must not push the settle timer out forever.
        self._native_momentum_t = 0.0
        # Input gain: keys / wheel / trackpad share rawviewer_ui.gallery_scroll
        # so arrow-key singleStep*N stays aligned with wheel/trackpad feel.
        from rawviewer_ui.gallery_scroll import (
            trackpad_gain,
            wheel_base_gain,
            wheel_fast_gain,
        )

        self._wheel_base_gain = wheel_base_gain()
        self._wheel_fast_gain = wheel_fast_gain()
        self._trackpad_gain = trackpad_gain()
        # Sub-pixel remainder carried between trackpad wheel events.
        self._trackpad_accum_px = 0.0
        self._wheel_last_notch_t = 0.0

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
        # Also coalesce deferred on_thumbnail_ready work (was one QTimer.singleShot(0)
        # per tile) into the same drain timer.
        self._pending_tile_applies: dict[str, bool] = {}
        self._pending_thumb_ready: dict[str, object] = {}
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

            with contextlib.suppress(Exception):
                self.load_manager.thumbnail_ready.disconnect(self.on_thumbnail_ready)
            with contextlib.suppress(Exception):
                self.load_manager.error_occurred.disconnect(self.on_thumbnail_error)
            self.load_manager.thumbnail_ready.connect(self.on_thumbnail_ready)
            self.load_manager.error_occurred.connect(self.on_thumbnail_error)

            with contextlib.suppress(Exception):
                self.load_manager.exif_data_ready.disconnect(self.on_exif_ready)
            self.load_manager.exif_data_ready.connect(self.on_exif_ready)

            with contextlib.suppress(Exception):
                self.load_manager.task_completed.disconnect(self.on_task_completed)
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

    def _layout_matches_current_images(self, images: list[str] | None = None) -> bool:
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
            with contextlib.suppress(Exception):
                w.hide()
        for w in self._widget_pool:
            with contextlib.suppress(Exception):
                w.hide()
        for child in self.findChildren(ThumbnailLabel):
            with contextlib.suppress(Exception):
                child.hide()

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
                with contextlib.suppress(Exception):
                    timer.stop()
        mgr = getattr(self, "load_manager", None)
        if mgr is not None and hasattr(mgr, "cancel_gallery_tasks"):
            with contextlib.suppress(Exception):
                mgr.cancel_gallery_tasks()
        self._active_tasks.clear()
        self._requested_thumbnail_paths.clear()
        self._placeholder_thumb_paths.clear()

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
        self._placeholder_thumb_paths.clear()
        if hasattr(self, "_load_timer") and self._load_timer:
            self._load_timer.stop()
        self._idle_preload_timer.stop()
        self._reset_gallery_scroll_position()
        self._invalidate_gallery_layout()
        self._measured_raw_aspects.clear()
        self._orientation_flip_paths.clear()
        self._metadata_fetch_attempted.clear()
        self._thumb_fail_counts.clear()
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
            if now - ts > max_age_s or drop_prefetch and self._active_task_priority(path) != Priority.CURRENT:
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

    def _momentum_scroll_active(self) -> bool:
        """True while motion comes from the wheel smoothing timer or native trackpad momentum."""
        if self._wheel_timer.isActive():
            return True
        return (time.time() - self._native_momentum_t) < 0.05

    def _is_actively_scrolling(self) -> bool:
        """True during an in-progress scroll gesture (trackpad, wheel, or scrollbar)."""
        return (time.time() - self._last_scroll_event_t) < 0.18

    def _on_gallery_thumb_hover(self, widget, *, entered: bool) -> None:
        """Warm the sensor-res decode for a tile the cursor rests on (not scrolls past).

        Entering single view from the gallery is the single most expensive
        transition in the app for a RAW file (a full LibRaw decode that was
        never requested while the thumbnail-only tile was on screen). If the
        user's cursor is genuinely resting on a tile -- not just passing
        under it because the content scrolled while the cursor stayed still
        on screen, which fires the exact same enterEvent -- that decode can
        start during the idle moment before the click instead of after it.

        Guarded twice against the scroll case (on entry, and again when the
        debounce timer fires) plus a per-tile debounce so a fast mouse
        sweep across many tiles queues nothing: only a tile the cursor
        stays on past the debounce window actually triggers a decode.
        """
        path = getattr(widget, "file_path", None)
        if not path:
            return
        if not entered:
            if getattr(self, "_hover_prefetch_pending_path", None) == path:
                self._hover_prefetch_pending_path = None
            return
        if self._is_actively_scrolling():
            return
        if not _env_true("RAWVIEWER_GALLERY_HOVER_PREFETCH", default=True):
            return
        self._hover_prefetch_pending_path = path
        delay_ms = _env_int("RAWVIEWER_GALLERY_HOVER_PREFETCH_MS", 220, minimum=50)
        QTimer.singleShot(delay_ms, lambda p=path: self._fire_gallery_hover_prefetch(p))

    def _fire_gallery_hover_prefetch(self, path: str) -> None:
        # Still the same tile (not superseded by a later hover/leave), and
        # no scroll gesture started during the debounce window.
        if getattr(self, "_hover_prefetch_pending_path", None) != path:
            return
        if self._is_actively_scrolling():
            return
        pv = getattr(self, "parent_viewer", None)
        if pv is None or getattr(pv, "view_mode", "") != "gallery":
            return
        if not hasattr(pv, "_maybe_queue_background_full_decode"):
            return
        with contextlib.suppress(Exception):
            pv._maybe_queue_background_full_decode(path)

    def _pixmap_matches(self, widget, pixmap: QPixmap | None) -> bool:
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

    def _apply_pixmap_if_changed(self, widget, pixmap: QPixmap | None) -> None:
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
        widget._tile_hovered = False
        widget._copy_hover = False
        widget._press_on_copy_path = False

    def _scale_crop_to_fit(self, pixmap: QPixmap, target_size, fast: bool = False):
        """Scale pixmap to fully cover target_size, then center-crop to exact size."""
        if not pixmap or pixmap.isNull():
            return pixmap
        tw = int(target_size.width())
        th = int(target_size.height())
        if tw <= 0 or th <= 0:
            return pixmap

        transform_mode = Qt.TransformationMode.FastTransformation if fast else Qt.TransformationMode.SmoothTransformation
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            transform_mode,
        )
        sw = scaled.width()
        sh = scaled.height()
        if sw == tw and sh == th:
            return scaled

        x = max(0, (sw - tw) // 2)
        y = max(0, (sh - th) // 2)
        return scaled.copy(x, y, tw, th)

    def get_scroll_anchor_path(self) -> str | None:
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

    def _capture_scroll_anchor_state(self) -> dict | None:
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

    def _scroll_anchor_state_for_rebuild(self) -> dict | None:
        """Capture scroll anchor for aspect/metadata layout rebuilds.

        Prefer the current (or pending entry) image when it is in the viewport
        or still the entry-scroll target. Pinning the top-left tile lets tiles
        above grow and shove the photo you returned from single-view out of
        sight; anchoring to that photo keeps it stable while new thumbs settle
        off-screen.
        """
        scroll = self._scroll_area_widget()
        items = self._gallery_layout_items
        if scroll is None or not items:
            return None
        sb = scroll.verticalScrollBar()
        scroll_y = int(sb.value())
        viewport_h = max(1, int(scroll.viewport().height()))

        preferred = self._entry_anchor_path()
        if preferred:
            resolved = self._resolve_gallery_path(preferred)
            indices = self._path_to_indices.get(resolved) if resolved else None
            if indices and 0 <= indices[0] < len(items):
                idx = indices[0]
                item = items[idx]
                rect = item["rect"]
                view_top = scroll_y
                view_bottom = scroll_y + viewport_h
                intersects = rect.bottom() > view_top and rect.top() < view_bottom
                pending = getattr(self, "_pending_scroll_to_path", None)
                pending_match = False
                if pending and resolved:
                    pending_resolved = self._resolve_gallery_path(pending)
                    pending_match = bool(
                        pending_resolved and pending_resolved == resolved
                    )
                if intersects or pending_match:
                    offset_px = int(rect.top() - scroll_y) if intersects else 12
                    return {
                        "file_path": item.get("file_path") or preferred,
                        "offset_px": offset_px,
                    }
        return self._capture_scroll_anchor_state()

    def _zoom_scroll_anchor_for_rebuild(self) -> dict | None:
        """Map the zoom-session anchor to build_gallery scroll-restore format."""
        state = self._zoom_anchor_state
        if not state or not state.get("file_path"):
            return None
        offset = state.get("offset_px")
        if offset is None:
            offset = state.get("offset_y_px", 0)
        return {"file_path": state["file_path"], "offset_px": int(offset)}

    def _restore_scroll_anchor_state(self, state: dict | None) -> bool:
        """Restore scroll so the anchored tile stays at the same viewport offset."""
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
