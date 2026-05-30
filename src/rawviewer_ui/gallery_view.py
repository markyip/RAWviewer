import time
import os
import threading
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import QWidget, QScrollArea, QLabel
from PyQt6.QtCore import Qt, QTimer, QRect, QEvent, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QFont, QTransform

from rawviewer_ui.widgets import ThumbnailLabel, ImageLoaded
from image_cache import LRUCache
from image_load_manager import get_image_load_manager, Priority
from common_image_loader import get_image_aspect_ratio, is_raw_file

logger = logging.getLogger(__name__)


def _focus_gallery_switch_logs() -> bool:
    return os.environ.get("RAWVIEWER_FOCUS_GALLERY_SWITCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _thumbnail_data_to_base_pixmap(thumbnail_data) -> Optional[QPixmap]:
    """Convert manager/cache thumbnail payload to a gallery base QPixmap."""
    if thumbnail_data is None:
        return None
    if isinstance(thumbnail_data, QPixmap):
        return thumbnail_data if not thumbnail_data.isNull() else None
    if isinstance(thumbnail_data, np.ndarray):
        arr = np.ascontiguousarray(thumbnail_data)
        h, w = arr.shape[:2]
        if h <= 0 or w <= 0:
            return None
        bytes_per_line = arr.strides[0]
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        if qimg.isNull():
            return None
        return QPixmap.fromImage(qimg)
    if isinstance(thumbnail_data, QImage):
        return QPixmap.fromImage(thumbnail_data) if not thumbnail_data.isNull() else None
    return None


class JustifiedGallery(QWidget):
    """
    Adaptive justified gallery layout with high-performance virtualization.
    Optimized for large directories using binary search and widget pooling.
    """

    TARGET_ROW_HEIGHT = 220
    HEIGHT_TOLERANCE = 0.25
    MIN_SPACING = 6

    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.images = images

        # Virtualization Data
        self._gallery_layout_items = []  # List of {rect, file_path, aspect}
        self._visible_widgets = {}  # {index: ThumbnailLabel}
        self._widget_pool = []
        self._total_content_height = 0

        # State Management
        self._building = False
        self._gallery_generation = 0
        self._active_tasks = {}
        self._metadata_cache = {}
        # Thumbnail cache: base + per-bucket scaled
        self._thumbnail_cache = LRUCache(10000)
        self._thumb_base_key = "__base__"
        self._row_height_buckets = [180, 220, 260, 300, 340]
        self._width_bucket_px = 64  # quantize widths to avoid mismatched cached pixmaps
        # Mapping of file_path to list of indices (for when the same file appears multiple times)
        self._path_to_indices = {}
        # Track what thumbnails we've recently requested so we can cancel far-away work
        self._requested_thumbnail_paths = set()
        self._pending_scroll_to_path = None
        self._is_scrollbar_dragging = False
        self._last_scheduled_scroll_y = 0
        self._scroll_area = None
        self._last_layout_viewport_width = -1
        self._last_build_ts = 0.0
        self._last_load_visible_request_ts = 0.0
        self._gallery_set_images_ts = 0.0
        self._first_thumb_ready_after_set = False

        self._loading_label = None
        self._empty_label = None

        # Smooth wheel scrolling (pixel-based) to avoid non-linear jumps and keep thumbnails in sync
        self._wheel_accum_px = 0.0
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(False)
        self._wheel_timer.timeout.connect(self._apply_wheel_scroll_step)
        self._wheel_step_px = 18  # per tick (tuned for smoothness)
        self._wheel_tick_ms = 8   # 125Hz-ish

        # Work budget per tick for smooth scrolling
        self._max_widgets_per_tick = 10
        self._max_tasks_per_tick = 16

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
        self._scroll_optimize_threshold = 6000

        # Metadata tracking for dynamic layout
        self._metadata_changed_paths = set()
        self._failed_thumbnails = set()
        self._metadata_rebuild_timer = QTimer(self)
        self._metadata_rebuild_timer.setSingleShot(True)
        self._metadata_rebuild_timer.timeout.connect(self._handle_metadata_rebuild)

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

    def set_images(self, images: List[str], bulk_metadata: Optional[Dict[str, Any]] = None):
        """
        Public API used by `main.py` to populate/refresh the gallery.
        Kept compatible with the legacy gallery interface.
        """
        try:
            if self.load_manager is None:
                # Ensure worker wiring exists before first render pass.
                self._post_init()
            new_images = images or []
            if _focus_gallery_switch_logs():
                logger.debug(
                    "[MODESWITCH] gallery.set_images called; count=%d load_manager=%s",
                    len(new_images),
                    self.load_manager is not None,
                )
            if (
                self._gallery_layout_items
                and len(new_images) == len(self.images)
                and all(a == b for a, b in zip(new_images, self.images))
            ):
                if bulk_metadata:
                    self._metadata_cache.update(bulk_metadata)
                if self._pending_scroll_to_path:
                    QTimer.singleShot(0, self._apply_pending_scroll_to_file)
                else:
                    self._request_load_visible_images(20)
                return

            self._gallery_generation += 1
            self.images = new_images
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
                self.clear_thumbnail_widgets()
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
            QTimer.singleShot(0, lambda: self.build_gallery(bulk_metadata=bulk_metadata))
        except Exception:
            logger.exception("[GALLERY] set_images() failed")

    def scroll_to_file(self, file_path: Optional[str]):
        """Scroll so file_path is near the top after the gallery layout is available."""
        if not file_path:
            return
        self._pending_scroll_to_path = file_path
        if self._gallery_layout_items:
            QTimer.singleShot(0, self._apply_pending_scroll_to_file)

    def _apply_pending_scroll_to_file(self):
        path = self._pending_scroll_to_path
        if not path:
            return
        indices = self._path_to_indices.get(path)
        if not indices:
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
        rect = self._gallery_layout_items[idx]["rect"]
        sb = p.verticalScrollBar()
        target = max(sb.minimum(), min(rect.top(), sb.maximum()))
        sb.setValue(target)
        self._pending_scroll_to_path = None
        self._request_load_visible_images(20)

    def _scaled_cache_key(self, file_path: str, target_size):
        """Cache key for scaled thumbnails including width/height buckets."""
        th = int(target_size.height())
        tw = int(target_size.width())
        rot = self._get_rotation_degrees_for_path(file_path)
        if th <= 0 or tw <= 0:
            return (file_path, self._thumb_base_key, rot)
        h_bucket = min(self._row_height_buckets, key=lambda x: abs(x - th))
        w_bucket = max(self._width_bucket_px, int(round(tw / self._width_bucket_px)) * self._width_bucket_px)
        return (file_path, h_bucket, w_bucket, rot)

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

    def invalidate_thumbnails_for_path(self, file_path: str) -> None:
        """Drop cached gallery pixmaps for a path (e.g. after on-disk rotation)."""
        try:
            self._thumbnail_cache.remove_keys_for_file_path(file_path)
        except Exception:
            pass

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
        self._invalidate_scaled_thumbnails_for_path(file_path)
        indices = self._path_to_indices.get(file_path, [])
        if not indices:
            return
        base = self._thumbnail_cache.get((file_path, self._thumb_base_key))
        for idx in indices:
            if idx not in self._visible_widgets:
                continue
            w = self._visible_widgets[idx]
            target_size = w.size()
            if base and not base.isNull():
                fitted = self._scale_crop_to_fit(base, target_size)
                self._thumbnail_cache.put(self._scaled_cache_key(file_path, target_size), fitted)
                w.setPixmap(fitted)
                w.setText("")
            elif self.load_manager is not None:
                # No base thumbnail yet: schedule an immediate fetch for this visible tile.
                self.load_manager.load_image(
                    file_path,
                    priority=Priority.CURRENT,
                    cancel_existing=False,
                    stages={"thumbnail"}
                )

    def resizeEvent(self, event):
        # Debounce expensive rebuilds during window resize.
        try:
            current_viewport_width = self._get_viewport_width()
            width_changed = current_viewport_width > 0 and current_viewport_width != self._last_layout_viewport_width
            # Rebuild justified layout only when width changes; height-only resizes
            # (common after setMinimumHeight/content growth) should not trigger loops.
            if width_changed:
                # Latch width immediately so repeated resize events before timer fires
                # don't keep scheduling duplicate rebuilds.
                self._last_layout_viewport_width = current_viewport_width
                if self._resize_timer is not None:
                    self._resize_timer.start(120)
                else:
                    # If post-init hasn't run yet, schedule a rebuild once.
                    QTimer.singleShot(0, self.build_gallery)
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
        # Smooth wheel scrolling: translate wheel deltas into pixel scrolling
        if event.type() == QEvent.Type.Wheel and self._scroll_area and obj is self._scroll_area.viewport():
            try:
                pixel = event.pixelDelta()
                angle = event.angleDelta()
                if not pixel.isNull():
                    delta_y = pixel.y()
                else:
                    # Typical mouse wheel: 120 units per notch
                    # Map to pixels (negative y = scroll down in Qt)
                    notches = angle.y() / 120.0 if angle.y() else 0.0
                    delta_y = int(notches * 120)  # base magnitude
                # Qt wheel delta is inverted relative to scrollbar value direction
                self._wheel_accum_px += -float(delta_y)
                if not self._wheel_timer.isActive():
                    self._wheel_timer.start(self._wheel_tick_ms)
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
        if self._last_scroll_y >= 0:
            last_time = getattr(self, "_last_scroll_time", 0)
            dt = now - last_time
            if dt > 0.01:
                current_speed = getattr(self, "_current_scroll_speed", 0.0)
                speed = abs(value - self._last_scroll_y) / dt
                self._current_scroll_speed = (current_speed * 0.4) + (speed * 0.6)
                self._is_scrolling_fast = self._current_scroll_speed > self._scroll_optimize_threshold

        self._last_scroll_y = value
        self._last_scroll_time = now
        self._last_scroll_event_t = now
        if self._thumb_first_after_scroll_t is None:
            self._thumb_first_after_scroll_t = now

        # Throttle (do NOT restart continuously): allow periodic updates while scrolling.
        # The settle timer below guarantees a final update when scrolling stops.
        interval = 50 if not self._is_scrolling_fast else 150
        if not self._load_timer.isActive():
            self._load_timer.start(interval)

        # Debounce: after scrolling stops, force a final load near thumb position.
        if self._scroll_settle_timer.isActive():
            self._scroll_settle_timer.stop()
        # When dragging the scrollbar, don't wait long; we handle release explicitly.
        self._scroll_settle_timer.start(60 if self._is_scrollbar_dragging else 120)

    def _on_scroll_settled(self):
        """Called after scroll events stop; load thumbnails around thumb position."""
        # When the user stops scrolling, treat as "not scrolling fast" so we actually schedule work.
        self._is_scrolling_fast = False
        self._current_scroll_speed = 0
        self._last_scroll_settle_t = time.time()
        self._thumb_first_after_settle_t = self._last_scroll_settle_t
        self.load_visible_images()

    def _get_viewport_width(self):
        p = self.parent()
        while p and not isinstance(p, QScrollArea):
            p = p.parent()
        return p.viewport().width() if p else self.width()

    def build_gallery(self, bulk_metadata=None, force=False):
        """
        Calculate grid layout and place placeholders. 
        Does not load images directly - that's handled by visible range tracking.
        """
        if self._building or not self.images:
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
        if (
            not force
            and viewport_width == self._last_layout_viewport_width
            and self._gallery_layout_items
            and len(self._gallery_layout_items) == len(self.images)
            and (time.time() - self._last_build_ts) < 0.8
        ):
            # Skip duplicate rebuilds caused by near-simultaneous resize/layout churn.
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

        self._building = True
        self._last_layout_viewport_width = viewport_width
        should_load_visible = False
        try:
            self._gallery_layout_items.clear()
            self._path_to_indices.clear()

            if bulk_metadata:
                self._metadata_cache.update(bulk_metadata)
            if self.parent_viewer and hasattr(self.parent_viewer, "image_cache"):
                paths = [img for img in self.images if isinstance(img, str)]
                missing_paths = [p for p in paths if p not in self._metadata_cache]
                if missing_paths:
                    bulk_fetched = self.parent_viewer.image_cache.get_multiple_exif(missing_paths)
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
                row_h = max(self.TARGET_ROW_HEIGHT * 0.5, min(self.TARGET_ROW_HEIGHT * 2.0, row_h))

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

            for item in self.images:
                aspect = 1.5
                if isinstance(item, str):
                    base = self._thumbnail_cache.get((item, self._thumb_base_key))
                    if base and not base.isNull() and base.height() > 0:
                        aspect = base.width() / base.height()
                    else:
                        m = self._metadata_cache.get(item)
                        if m and m.get("original_width") and m.get("original_height"):
                            w, h = m["original_width"], m["original_height"]
                            if m.get("orientation", 1) in (5, 6, 7, 8):
                                w, h = h, w
                            aspect = w / h if h > 0 else 1.5
                        else:
                            aspect = 1.5

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
            self.setMinimumHeight(self._total_content_height)
            self.update()
            self._last_build_ts = time.time()
            if _focus_gallery_switch_logs():
                logger.debug(
                    "[MODESWITCH] gallery.build_gallery done; items=%d content_h=%d",
                    len(self._gallery_layout_items),
                    self._total_content_height,
                )
            should_load_visible = True
        finally:
            self._building = False
            if should_load_visible:
                # Run after _building is cleared, so load_visible_images won't early-return.
                if self._pending_scroll_to_path:
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

    def load_visible_images(self):
        if self._building:
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
        now_ts = time.time()
        stale_paths = [p for p, ts in self._active_tasks.items() if (now_ts - ts) > 8.0]
        for sp in stale_paths:
            self._active_tasks.pop(sp, None)

        v_port = p.viewport()
        scrollbar = p.verticalScrollBar()
        scroll_y = scrollbar.value()
        v_h = v_port.height()

        buffer_rect = QRect(0, scroll_y, v_port.width(), v_h)
        visible_indices_items = self._get_visible_range(buffer_rect)
        if _focus_gallery_switch_logs():
            logger.debug(
                "[MODESWITCH] gallery.load_visible_images visible=%d cached_tasks=%d",
                len(visible_indices_items),
                len(self._active_tasks),
            )
        visible_indices_set = {idx for idx, item in visible_indices_items}

        # Dynamic prefetch: follow the scrollbar thumb/viewport center position.
        # Slow scroll: keep more around the thumb; fast scroll: keep less.
        # Note: we already early-return on fast scroll, so this mainly helps "normal" scrolling.
        screens = 2 if not self._is_scrolling_fast else 1
        center_y = scroll_y + (v_h // 2)
        half_span = int((v_h * screens) // 2)
        prefetch_top = max(0, center_y - half_span)
        prefetch_rect = QRect(0, prefetch_top, v_port.width(), v_h * screens)
        prefetch_indices_items = self._get_visible_range(prefetch_rect)
        prefetch_paths = {item["file_path"] for idx, item in prefetch_indices_items if item.get("file_path")}

        for idx in list(self._visible_widgets.keys()):
            if idx not in visible_indices_set:
                w = self._visible_widgets.pop(idx)
                w.hide()
                w.clear()
                w.file_path = None
                w.original_pixmap = None
                self._widget_pool.append(w)

        # Fast scroll policy:
        # - still keep visible thumbnails loading (small budget)
        # - avoid heavy prefetch
        is_fast = self._is_scrolling_fast
        # First-paint mode: prioritize visible tiles only until first thumbnail arrives
        # (or a short timeout), then enable prefetch.
        warmup_elapsed = time.time() - float(getattr(self, "_gallery_set_images_ts", 0.0) or 0.0)
        allow_prefetch = (not is_fast) and (
            self._first_thumb_ready_after_set or warmup_elapsed > 1.5
        )

        # If the user jumped far (typical when dragging scrollbar), flush stale queue so new
        # position thumbnails start quickly.
        if abs(scroll_y - self._last_scheduled_scroll_y) > (v_h * 3):
            try:
                if hasattr(self.load_manager, "flush_queue"):
                    self.load_manager.flush_queue()
                else:
                    self.load_manager.cancel_all_tasks()
            except Exception:
                pass
            self._requested_thumbnail_paths.clear()
        self._last_scheduled_scroll_y = scroll_y
        # Determine what we want around current thumb position - paths for prefetch
        visible_paths = {item["file_path"] for idx, item in visible_indices_items if item.get("file_path")}
        wanted_paths = set(visible_paths)
        if allow_prefetch:
            wanted_paths |= {item["file_path"] for idx, item in prefetch_indices_items if item.get("file_path")}

        # Cancel thumbnail work that is no longer near the scrollbar thumb.
        # This prevents the queue from "finishing the whole folder" in the background.
        # We only cancel tasks we previously requested from this gallery instance.
        to_cancel = self._requested_thumbnail_paths - wanted_paths
        for fp in list(to_cancel):
            try:
                self.load_manager.cancel_task(fp)
            except Exception:
                pass
            if fp in self._active_tasks:
                del self._active_tasks[fp]
        self._requested_thumbnail_paths = wanted_paths

        load_tasks = []
        created_widgets = 0
        scheduled_tasks = 0

        # Reduce per-tick work when fast scrolling (keep things responsive)
        # RESTORED: Using v1.6.0-style aggressive scheduling for snappier population
        max_widgets = 4 if is_fast else 12
        max_tasks = 4 if is_fast else 16

        # Create/update widgets for visible items and schedule loads for missing thumbnails.
        for idx, item in visible_indices_items:
            path = item.get("file_path")
            rect = item.get("rect")
            if not path or rect is None:
                continue

            if idx not in self._visible_widgets:
                if created_widgets >= max_widgets:
                    # Continue on next tick to keep UI responsive
                    if not self._load_timer.isActive():
                        self._load_timer.start(16)
                    break
                w = self._widget_pool.pop() if self._widget_pool else ThumbnailLabel(self)
                w.file_path = path
                w.index = idx  # Keep track of index on the widget
                w.setGeometry(rect)
                w.setFixedSize(rect.size())
                def _on_thumb_click(e, _w=w):
                    try:
                        if e.button() == Qt.MouseButton.LeftButton:
                            e.accept()
                            # Read the widget's CURRENT file_path. Widgets are pooled and
                            # reused by index, so a captured path can go stale when the
                            # image list changes (e.g. after a search filter).
                            target_path = getattr(_w, "file_path", None)
                            if target_path:
                                self.parent_viewer._gallery_item_clicked(target_path)
                            return
                    except Exception:
                        pass
                    try:
                        e.ignore()
                    except Exception:
                        pass
                w.mousePressEvent = _on_thumb_click
                created_widgets += 1
                self._visible_widgets[idx] = w
            else:
                w = self._visible_widgets[idx]
                w.file_path = path
                w.index = idx
                if w.geometry() != rect:
                    w.setGeometry(rect)
                if w.size() != rect.size():
                    w.setFixedSize(rect.size())

            cache_hit = False
            # Account for Device Pixel Ratio (Retina/4K)
            dpr = self.devicePixelRatio()
            physical_size = QSize(int(rect.width() * dpr), int(rect.height() * dpr))
            scaled_key = self._scaled_cache_key(path, physical_size)
            cached_scaled = self._thumbnail_cache.get(scaled_key)
            if cached_scaled:
                w.setPixmap(cached_scaled)
                w.setText("")
                cache_hit = True
            else:
                base = self._thumbnail_cache.get((path, self._thumb_base_key))
                if not base:
                    try:
                        from image_cache import get_image_cache
                        global_thumb = get_image_cache().get_thumbnail(path)
                        if global_thumb is not None:
                            arr = np.ascontiguousarray(global_thumb)
                            h_img, w_img = arr.shape[:2]
                            bytes_per_line = arr.strides[0]
                            qimg = QImage(arr.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888).copy()
                            base = QPixmap.fromImage(qimg)
                            if base and not base.isNull():
                                self._thumbnail_cache.put((path, self._thumb_base_key), base)
                    except Exception as e:
                        logger.debug(f"Sync get_thumbnail failed for {path}: {e}")
                
                if base:
                    scaled = self._scale_crop_to_fit(base, physical_size)
                    scaled.setDevicePixelRatio(dpr)
                    self._thumbnail_cache.put(scaled_key, scaled)
                    w.setPixmap(scaled)
                    w.setText("")
                    cache_hit = True
                else:
                    if w.file_path != path or not w.pixmap() or w.pixmap().isNull():
                        w.setPixmap(QPixmap())
                    w.setText("")

            w.show()
            thumb_missing = not cache_hit
            
            m = self._metadata_cache.get(path)
            exif_missing = not m or m.get("original_width") is None or m.get("original_height") is None
            
            stages = set()
            if thumb_missing:
                stages.add("thumbnail")
            if exif_missing:
                stages.add("exif")
                
            if stages and path not in self._active_tasks:
                load_tasks.append((path, Priority.CURRENT, rect.size(), stages))

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
                    load_tasks.append((path, Priority.PRELOAD_NEXT, None, stages))

        # In mixed RAW/non-RAW folders, render lightweight formats first so the gallery
        # paints quickly while heavier RAW thumbnails continue in background.
        load_tasks.sort(key=lambda item: 1 if is_raw_file(item[0]) else 0)

        # Schedule with budget and target-sized thumbnails for visible tiles.
        scheduled = 0
        # Avoid ballooning in-flight tasks; too many queued "current" requests can
        # delay first visible paint and create long tail stalls.
        active_cap = 16 if not is_fast else 10
        if len(self._active_tasks) >= active_cap:
            if _focus_gallery_switch_logs():
                logger.debug(
                    "[MODESWITCH] gallery.load_visible_images skipped scheduling; active cap reached (%d)",
                    len(self._active_tasks),
                )
            return
        for path, priority, target_size, stages in load_tasks:
            if scheduled >= max_tasks:
                if not self._load_timer.isActive():
                    self._load_timer.start(16)
                break
            if len(self._active_tasks) >= active_cap:
                break
            self.load_manager.load_image(
                path,
                priority=priority,
                cancel_existing=False,
                stages=stages
            )
            self._active_tasks[path] = time.time()
            scheduled += 1
        scheduled_tasks = scheduled

        # (perf logging removed)
        if _focus_gallery_switch_logs():
            logger.debug(
                "[MODESWITCH] gallery.load_visible_images scheduled=%d visible=%d active=%d",
                scheduled_tasks,
                len(visible_indices_items),
                len(self._active_tasks),
            )

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
                continue
            pixmap = _thumbnail_data_to_base_pixmap(thumb)
            if pixmap is None or pixmap.isNull():
                continue
            self._thumbnail_cache.put((path, self._thumb_base_key), pixmap)
            warmed += 1
        if warmed and _focus_gallery_switch_logs():
            logger.debug(
                "[GALLERY] Warmed %d tile(s) from global thumbnail cache", warmed
            )
        return warmed

    def on_thumbnail_ready(self, file_path, thumbnail_data):
        # Mark path as no longer in-flight regardless of whether we can render it now.
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

        pixmap = _thumbnail_data_to_base_pixmap(thumbnail_data)

        if not pixmap or pixmap.isNull():
            self.on_thumbnail_error(file_path, "Null pixmap in on_thumbnail_ready")
            return

        # Cache base thumbnail in original orientation; visual rotation is applied lazily per visible tile.
        self._thumbnail_cache.put((file_path, self._thumb_base_key), pixmap)
        if not self._first_thumb_ready_after_set:
            self._first_thumb_ready_after_set = True

        now = time.time()
        if self._thumb_first_after_scroll_t is not None:
            self._thumb_first_after_scroll_t = None
        if self._thumb_first_after_settle_t is not None and self._last_scroll_settle_t > 0:
            self._thumb_first_after_settle_t = None

        # Avoid heavy UI updates only during very fast scrolling.
        if self._is_scrolling_fast:
            return

        # Update ANY widget displaying this path that is currently visible
        indices = self._path_to_indices.get(file_path, [])
        
        # Ensure layout aspect ratio matches the actual thumbnail
        if pixmap.width() > 0 and pixmap.height() > 0:
            aspect = pixmap.width() / pixmap.height()
            changed = False
            for idx in indices:
                if idx < len(self._gallery_layout_items):
                    old_aspect = self._gallery_layout_items[idx].get("aspect", 1.5)
                    if abs(old_aspect - aspect) > 0.05:
                        self._gallery_layout_items[idx]["aspect"] = aspect
                        changed = True
            
            if changed:
                self._metadata_changed_paths.add(file_path)
                logger.debug(f"[GALLERY_DEBUG] Timer started by on_thumbnail_ready for {file_path}")
                if not self._metadata_rebuild_timer.isActive():
                    self._metadata_rebuild_timer.start(500)

        dpr = self.devicePixelRatio()
        for idx in indices:
            if idx in self._visible_widgets:
                w = self._visible_widgets[idx]
                logical_size = w.size()
                physical_size = QSize(int(logical_size.width() * dpr), int(logical_size.height() * dpr))
                # If worker already emitted a target-fitted image, avoid re-scaling here.
                if pixmap.width() == physical_size.width() and pixmap.height() == physical_size.height():
                    fitted = pixmap
                else:
                    fitted = self._scale_crop_to_fit(pixmap, physical_size)
                
                fitted.setDevicePixelRatio(dpr)
                self._thumbnail_cache.put(self._scaled_cache_key(file_path, physical_size), fitted)
                w.setPixmap(fitted)
                w.setText("")

        # Continue scheduling when new thumbnails arrive so visible blanks are filled quickly.
        if not self._is_scrolling_fast:
            self._request_load_visible_images(25)

    def on_task_completed(self, file_path):
        """Always clear active tasks when the background worker finishes."""
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]

    def on_thumbnail_error(self, file_path, error_msg):
        # Prevent infinite retry loops by putting a dummy grey thumbnail in cache
        null_pixmap = QPixmap(100, 100)
        null_pixmap.fill(Qt.GlobalColor.darkGray)
        # Use target size if known, else default to some key
        self._thumbnail_cache.put((file_path, self._thumb_base_key), null_pixmap)
        
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]
            
        if self.parent_viewer is not None and getattr(
            self.parent_viewer, "view_mode", "gallery"
        ) == "gallery":
            # Delay slightly to prevent spin loops
            QTimer.singleShot(100, lambda: self._request_load_visible_images(40))

    def on_exif_ready(self, file_path, exif_data):
        """Update aspect ratio in layout when metadata arrives."""
        if not exif_data:
            # Save dummy data so we don't infinitely request it
            self._metadata_cache[file_path] = {"original_width": 0, "original_height": 0}
            return
            
        if file_path not in self.images:
            return
            
        # Store in local metadata cache
        self._metadata_cache[file_path] = exif_data
        
        # Calculate real aspect ratio
        w = exif_data.get("original_width")
        h = exif_data.get("original_height")
        if not w or not h or h <= 0:
            return
            
        orientation = exif_data.get("orientation", 1)
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        aspect = w / h
        
        # Check if we need to update layout (if it differs from default 1.333 or previous cache)
        # Find all occurrences of this path in layout
        indices = self._path_to_indices.get(file_path, [])
        changed = False
        for idx in indices:
            if idx < len(self._gallery_layout_items):
                old_aspect = self._gallery_layout_items[idx].get("aspect", 1.5)
                if abs(old_aspect - aspect) > 0.05:
                    self._gallery_layout_items[idx]["aspect"] = aspect
                    changed = True
        
        if changed:
            self._metadata_changed_paths.add(file_path)
            
            # OPTIMIZATION: Only rebuild if a significant number of images have changed,
            # or if it's been a while. This prevents "rebuild storms" in large folders.
            rebuild_threshold = 5 if len(self.images) < 100 else 15
            
            # Debounce layout rebuild
            if not self._metadata_rebuild_timer.isActive():
                # Initial populates get a longer debounce to let things settle
                debounce = 2000 if len(self._metadata_cache) < (len(self.images) * 0.5) else 800
                logger.debug(f"[GALLERY_DEBUG] Timer started by on_exif_ready (long) for {file_path}")
                self._metadata_rebuild_timer.start(debounce)
            elif len(self._metadata_changed_paths) >= rebuild_threshold:
                # If we hit the threshold, force a rebuild sooner
                logger.debug(f"[GALLERY_DEBUG] Timer started by on_exif_ready (short) for {file_path}")
                self._metadata_rebuild_timer.start(500)

    def _handle_metadata_rebuild(self):
        """Rebuild layout after metadata changes to settle aspect ratios."""
        if not self._metadata_changed_paths or self._building or self._is_scrolling_fast:
            # Don't rebuild while scrolling as it blocks the UI thread
            if self._is_scrolling_fast and not self._metadata_rebuild_timer.isActive():
                self._metadata_rebuild_timer.start(1000)
            return
            
        self._metadata_changed_paths.clear()
        if _focus_gallery_switch_logs():
            logger.debug("[GALLERY] metadata rebuild triggered")
            
        # Use existing metadata cache to avoid re-extraction
        self.build_gallery(bulk_metadata=None, force=True)

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

    def clear_thumbnail_widgets(self):
        """Remove all thumbnail widgets from the gallery surface."""
        for label in list(getattr(self, "_visible_widgets", {}).values()):
            try:
                label.hide()
                label.clear()
                label.setText("")
                label.file_path = None
                label.original_pixmap = None
                label.deleteLater()
            except Exception:
                pass
        self._visible_widgets.clear()

        for label in list(getattr(self, "_widget_pool", [])):
            try:
                label.hide()
                label.clear()
                label.setText("")
                label.file_path = None
                label.original_pixmap = None
                label.deleteLater()
            except Exception:
                pass
        self._widget_pool = []

        try:
            for child in self.findChildren(ThumbnailLabel):
                child.hide()
                child.clear()
                child.setText("")
                child.file_path = None
                child.original_pixmap = None
                child.deleteLater()
        except Exception:
            pass
            
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
