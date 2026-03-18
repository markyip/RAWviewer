import time
import os
import threading
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import QWidget, QApplication, QScrollArea
from PyQt6.QtCore import Qt, QTimer, QRect, QEvent, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor

from rawviewer_ui.widgets import ThumbnailLabel, ImageLoaded
from image_cache import LRUCache
from image_load_manager import get_image_load_manager, Priority
from common_image_loader import get_image_aspect_ratio

logger = logging.getLogger(__name__)


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
        self._visible_widgets = {}  # {file_path: ThumbnailLabel}
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
        self._path_to_index = {}
        # Track what thumbnails we've recently requested so we can cancel far-away work
        self._requested_thumbnail_paths = set()
        self._is_scrollbar_dragging = False
        self._last_scheduled_scroll_y = 0
        self._scroll_area = None

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
        self._last_scroll_time = 0
        self._current_scroll_speed = 0
        self._is_scrolling_fast = False
        self._scroll_optimize_threshold = 3000

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
            self.load_manager.thumbnail_ready.connect(self.on_thumbnail_ready)

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
            self._gallery_generation += 1
            self.images = images or []

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

            # Rebuild on next event loop tick (avoid doing heavy UI work inside caller)
            QTimer.singleShot(0, lambda: self.build_gallery(bulk_metadata=bulk_metadata))
        except Exception:
            logger.exception("[GALLERY] set_images() failed")

    def _scaled_cache_key(self, file_path: str, target_size):
        """Cache key for scaled thumbnails including width/height buckets."""
        th = int(target_size.height())
        tw = int(target_size.width())
        if th <= 0 or tw <= 0:
            return (file_path, self._thumb_base_key)
        h_bucket = min(self._row_height_buckets, key=lambda x: abs(x - th))
        w_bucket = max(self._width_bucket_px, int(round(tw / self._width_bucket_px)) * self._width_bucket_px)
        return (file_path, h_bucket, w_bucket)

    def resizeEvent(self, event):
        # Debounce expensive rebuilds during window resize.
        try:
            if self._resize_timer is not None:
                self._resize_timer.start(120)
            else:
                # If post-init hasn't run yet, schedule a rebuild once.
                QTimer.singleShot(0, self.build_gallery)
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
            QApplication.processEvents()
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

        if abs(self._wheel_accum_px) < 0.5:
            self._wheel_timer.stop()
            self._wheel_accum_px = 0.0
            return

        step = self._wheel_step_px if self._wheel_accum_px > 0 else -self._wheel_step_px
        if abs(step) > abs(self._wheel_accum_px):
            step = self._wheel_accum_px
        self._wheel_accum_px -= step

        new_val = int(sb.value() + step)
        new_val = max(sb.minimum(), min(sb.maximum(), new_val))
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
            dt = now - self._last_scroll_time
            if dt > 0.01:
                speed = abs(value - self._last_scroll_y) / dt
                self._current_scroll_speed = (self._current_scroll_speed * 0.4) + (speed * 0.6)
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

    def build_gallery(self, bulk_metadata=None):
        if self._building or not self.images:
            return
        self._building = True
        should_load_visible = False
        try:
            for w in self._visible_widgets.values():
                w.hide()
                self._widget_pool.append(w)
            self._visible_widgets.clear()
            self._gallery_layout_items.clear()

            if bulk_metadata:
                self._metadata_cache.update(bulk_metadata)
            if not self._metadata_cache and self.parent_viewer and hasattr(self.parent_viewer, "image_cache"):
                paths = [img for img in self.images if isinstance(img, str)]
                if paths:
                    self._metadata_cache = self.parent_viewer.image_cache.get_multiple_exif(paths)

            viewport_width = self._get_viewport_width()
            net_width = viewport_width - (self.MIN_SPACING * 2) - 16
            if net_width <= 0:
                return

            current_y = 10
            row = []
            aspect_sum = 0

            def commit_row(r, a_sum, is_last):
                nonlocal current_y
                if not r:
                    return
                spacing = (len(r) - 1) * self.MIN_SPACING
                row_h = self.TARGET_ROW_HEIGHT if is_last else (net_width - spacing) / a_sum
                row_h = max(self.TARGET_ROW_HEIGHT * 0.5, min(self.TARGET_ROW_HEIGHT * 2.0, row_h))

                curr_x = 10
                for i, (item, aspect) in enumerate(r):
                    w = int(row_h * aspect)
                    if not is_last and i == len(r) - 1:
                        w = net_width - (curr_x - 10)
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
                aspect = 1.333
                if isinstance(item, str):
                    m = self._metadata_cache.get(item)
                    if m and m.get("original_width"):
                        w, h = m["original_width"], m["original_height"]
                        if m.get("orientation", 1) in (5, 6, 7, 8):
                            w, h = h, w
                        aspect = w / h
                    else:
                        # IMPORTANT: Keep initial layout build non-blocking.
                        # Avoid calling get_image_aspect_ratio() here because it can hit disk/rawpy
                        # when metadata cache is cold, which blocks UI and delays scroll range setup.
                        aspect = 1.333

                row.append((item, aspect))
                aspect_sum += aspect
                if (aspect_sum * self.TARGET_ROW_HEIGHT + (len(row) - 1) * self.MIN_SPACING) >= net_width:
                    commit_row(row, aspect_sum, False)
                    row, aspect_sum = [], 0

            if row:
                commit_row(row, aspect_sum, True)

            self._path_to_index = {item["file_path"]: i for i, item in enumerate(self._gallery_layout_items)}

            self._total_content_height = int(current_y + 20)
            self.setMinimumHeight(self._total_content_height)
            self.update()
            should_load_visible = True
        finally:
            self._building = False
            if should_load_visible:
                # Run after _building is cleared, so load_visible_images won't early-return.
                QTimer.singleShot(0, self.load_visible_images)

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
            return

        v_port = p.viewport()
        scrollbar = p.verticalScrollBar()
        scroll_y = scrollbar.value()
        v_h = v_port.height()

        buffer_rect = QRect(0, scroll_y, v_port.width(), v_h)
        visible_indices_items = self._get_visible_range(buffer_rect)
        visible_paths = {item["file_path"] for idx, item in visible_indices_items if item.get("file_path")}

        # Dynamic prefetch: follow the scrollbar thumb/viewport center position.
        # Slow scroll: keep more around the thumb; fast scroll: keep less.
        # Note: we already early-return on fast scroll, so this mainly helps "normal" scrolling.
        screens = 4 if not self._is_scrolling_fast else 1
        center_y = scroll_y + (v_h // 2)
        half_span = int((v_h * screens) // 2)
        prefetch_top = max(0, center_y - half_span)
        prefetch_rect = QRect(0, prefetch_top, v_port.width(), v_h * screens)
        prefetch_indices_items = self._get_visible_range(prefetch_rect)
        prefetch_paths = {item["file_path"] for idx, item in prefetch_indices_items if item.get("file_path")}

        for path in list(self._visible_widgets.keys()):
            if path not in visible_paths:
                w = self._visible_widgets.pop(path)
                w.hide()
                self._widget_pool.append(w)

        # Fast scroll policy:
        # - still keep visible thumbnails loading (small budget)
        # - avoid heavy prefetch
        is_fast = self._is_scrolling_fast
        allow_prefetch = not is_fast

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
        # Determine what we want around current thumb position
        wanted_paths = set(visible_paths) | (set(prefetch_paths) if allow_prefetch else set())

        # Cancel thumbnail work that is no longer near the scrollbar thumb.
        # This prevents the queue from "finishing the whole folder" in the background.
        # We only cancel tasks we previously requested from this gallery instance.
        to_cancel = self._requested_thumbnail_paths - wanted_paths
        for fp in list(to_cancel):
            try:
                self.load_manager.cancel_task(fp)
            except Exception:
                pass
        self._requested_thumbnail_paths = wanted_paths

        load_tasks = []
        created_widgets = 0
        scheduled_tasks = 0

        # Reduce per-tick work when fast scrolling (keep things responsive)
        max_widgets = 4 if is_fast else self._max_widgets_per_tick
        max_tasks = 6 if is_fast else self._max_tasks_per_tick

        # Create/update widgets for visible items and schedule loads for missing thumbnails.
        for idx, item in visible_indices_items:
            path = item.get("file_path")
            rect = item.get("rect")
            if not path or rect is None:
                continue

            if path not in self._visible_widgets:
                if created_widgets >= max_widgets:
                    # Continue on next tick to keep UI responsive
                    if not self._load_timer.isActive():
                        self._load_timer.start(16)
                    break
                w = self._widget_pool.pop() if self._widget_pool else ThumbnailLabel(self)
                w.file_path = path
                w.setGeometry(rect)
                w.setFixedSize(rect.size())
                def _on_thumb_click(e, p=path):
                    try:
                        if e.button() == Qt.MouseButton.LeftButton:
                            e.accept()
                            self.parent_viewer._gallery_item_clicked(p)
                            return
                    except Exception:
                        pass
                    try:
                        e.ignore()
                    except Exception:
                        pass
                w.mousePressEvent = _on_thumb_click
                created_widgets += 1

                cache_hit = False
                scaled_key = self._scaled_cache_key(path, rect.size())
                cached_scaled = self._thumbnail_cache.get(scaled_key)
                if cached_scaled:
                    w.setPixmap(cached_scaled)
                    w.setText("")
                    cache_hit = True
                else:
                    base = self._thumbnail_cache.get((path, self._thumb_base_key))
                    if base:
                        scaled = self._scale_crop_to_fit(base, rect.size())
                        self._thumbnail_cache.put(scaled_key, scaled)
                        w.setPixmap(scaled)
                        w.setText("")
                        cache_hit = True
                    else:
                        # Avoid expensive text updates during scroll; placeholder paint will cover.
                        w.setText("")

                w.show()
                self._visible_widgets[path] = w

                if not cache_hit:
                    load_tasks.append((path, Priority.CURRENT, rect.size()))

        if allow_prefetch:
            for path in prefetch_paths:
                if not path or path in visible_paths:
                    continue
                # If neither base nor bucket exists, schedule preload
                # (we don't know exact bucket here; base check is enough)
                if not self._thumbnail_cache.get((path, self._thumb_base_key)):
                    load_tasks.append((path, Priority.PRELOAD_NEXT, None))

        # Schedule with budget and target-sized thumbnails for visible tiles.
        scheduled = 0
        for path, priority, target_size in load_tasks:
            if scheduled >= max_tasks:
                if not self._load_timer.isActive():
                    self._load_timer.start(16)
                break
            if target_size is not None:
                self.load_manager.load_image(
                    path,
                    priority=priority,
                    cancel_existing=False,
                    stages={"thumbnail"},
                    thumbnail_target_size=QSize(target_size.width(), target_size.height()),
                    thumbnail_fit="crop",
                )
            else:
                self.load_manager.load_image(path, priority=priority, cancel_existing=False, stages={"thumbnail"})
            scheduled += 1
        scheduled_tasks = scheduled

        # (perf logging removed)

    def on_thumbnail_ready(self, file_path, thumbnail_data):
        if file_path not in self._path_to_index:
            return

        if isinstance(thumbnail_data, np.ndarray):
            arr = np.ascontiguousarray(thumbnail_data)
            h, w = arr.shape[:2]
            bytes_per_line = arr.strides[0]
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
        elif isinstance(thumbnail_data, QImage):
            pixmap = QPixmap.fromImage(thumbnail_data)
        else:
            pixmap = thumbnail_data

        if not pixmap:
            return

        # Cache base thumbnail (and also cache a scaled version if we can infer current visible tile size)
        self._thumbnail_cache.put((file_path, self._thumb_base_key), pixmap)

        now = time.time()
        if self._thumb_first_after_scroll_t is not None:
            self._thumb_first_after_scroll_t = None
        if self._thumb_first_after_settle_t is not None and self._last_scroll_settle_t > 0:
            self._thumb_first_after_settle_t = None

        # Avoid heavy UI updates only during very fast scrolling.
        if self._is_scrolling_fast:
            return

        if file_path in self._visible_widgets:
            w = self._visible_widgets[file_path]
            if w.file_path == file_path:
                target_size = w.size()
                # If worker already emitted a target-fitted image, avoid re-scaling here.
                if pixmap.width() == target_size.width() and pixmap.height() == target_size.height():
                    fitted = pixmap
                else:
                    fitted = self._scale_crop_to_fit(pixmap, target_size)
                self._thumbnail_cache.put(self._scaled_cache_key(file_path, target_size), fitted)
                w.setPixmap(fitted)
                w.setText("")

        if file_path in self._active_tasks:
            del self._active_tasks[file_path]

