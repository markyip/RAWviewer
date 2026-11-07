"""Gallery mixin part 4 of 7 for JustifiedGallery."""
from __future__ import annotations

import logging
import os
import time

from PyQt6.QtCore import QRect, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QScrollArea

from .helpers import _env_int, _focus_gallery_switch_logs, _orient_debug_enabled, _ScrollDateBubble

logger = logging.getLogger(__name__)

class GalleryPart4:

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

    def _scroll_preview_pixmap(self, path: str) -> QPixmap | None:
        """In-memory thumbnail at ``path``, or None. Never decodes, never hits disk.

        Gallery LRU first (already an oriented QPixmap). Global ImageCache only
        via memory-tier peeks inside ``_global_cache_to_base_pixmap`` — not
        ``get_thumbnail()`` / ``get_grid()`` / ``load_image()``.
        """
        if not path:
            return None
        cache = getattr(self, "_thumbnail_cache", None)
        key = getattr(self, "_thumb_base_key", "__base__")
        if cache is not None:
            try:
                pm = cache.get((path, key))
                if pm is not None and not pm.isNull():
                    return pm
            except Exception:
                pass
        getter = getattr(self, "_global_cache_to_base_pixmap", None)
        if getter is None:
            return None
        try:
            pm = getter(path)
            if pm is not None and not pm.isNull():
                return pm
        except Exception:
            return None
        return None

    def _update_scroll_date_bubble(self, *, force: bool = True) -> None:
        if not getattr(self, "_is_scrollbar_dragging", False):
            return
        scroll = getattr(self, "_scroll_area", None)
        if scroll is None:
            return
        now = time.monotonic()
        last = getattr(self, "_scroll_bubble_last_content_t", 0.0)
        throttle = getattr(self, "_SCROLL_BUBBLE_THROTTLE_S", 0.075)
        refresh = force or (now - last) >= throttle
        if refresh:
            self._scroll_bubble_last_content_t = now
            path = self._scroll_preview_path()
            text = self._scroll_date_text(path)
            committed = getattr(self, "_scroll_bubble_committed_path", None)
            if path != committed:
                # One in-memory lookup per path. A miss stays date-only for
                # the rest of this drag even if a thumbnail arrives later.
                pixmap = self._scroll_preview_pixmap(path) if text else None
                self._scroll_bubble_committed_path = path
                self._scroll_bubble_cached_pixmap = pixmap
            else:
                pixmap = getattr(self, "_scroll_bubble_cached_pixmap", None)
            self._scroll_bubble_cached_text = text
        else:
            text = getattr(self, "_scroll_bubble_cached_text", "")
            pixmap = getattr(self, "_scroll_bubble_cached_pixmap", None)
            path = getattr(self, "_scroll_bubble_committed_path", "") or ""
        bubble = getattr(self, "_scroll_date_bubble", None)
        if bubble is None:
            if not text:
                return
            bubble = _ScrollDateBubble(scroll.viewport())
            self._scroll_date_bubble = bubble
        if not text:
            if bubble._shown_path is not None:
                bubble._shown_path = None
                bubble._preview.clear()
                bubble._preview.hide()
            bubble.hide()
            return
        # Track the thumb: the bubble belongs beside the thing being dragged,
        # not pinned to a corner the eye is not on. Position updates every
        # tick; date + preview content is throttled on the scroll path.
        sb = scroll.verticalScrollBar()
        span = max(1, sb.maximum() - sb.minimum())
        frac = (sb.value() - sb.minimum()) / span
        bubble.show_for(
            text,
            center_y=int(frac * scroll.viewport().height()),
            pixmap=pixmap,
            file_path=path or "",
        )

    def _hide_scroll_date_bubble(self) -> None:
        bubble = getattr(self, "_scroll_date_bubble", None)
        if bubble is not None:
            bubble._shown_path = None
            bubble._preview.clear()
            bubble._preview.hide()
            bubble.hide()
        self._scroll_bubble_cached_text = ""
        self._scroll_bubble_cached_pixmap = None
        self._scroll_bubble_committed_path = None
        self._scroll_bubble_last_content_t = 0.0

    def _on_slider_pressed(self):
        self._is_scrollbar_dragging = True
        self._scroll_bubble_committed_path = None
        self._scroll_bubble_cached_pixmap = None
        self._scroll_bubble_last_content_t = 0.0
        bubble = getattr(self, "_scroll_date_bubble", None)
        if bubble is not None:
            bubble._shown_path = None
        self._update_scroll_date_bubble(force=True)

    def _on_slider_released(self):
        self._is_scrollbar_dragging = False
        self._hide_scroll_date_bubble()
        # Immediately load around the new thumb position (no extra debounce)
        self._on_scroll_settled()

    def _on_scroll(self, value):
        now = time.time()

        # Follows the thumb during a drag. Position tracks every tick; date +
        # cache-hit preview content is throttled (never decode / disk / load).
        if getattr(self, "_is_scrollbar_dragging", False):
            self._update_scroll_date_bubble(force=False)

        # Pause background indexing once per scroll burst (not every valueChanged tick).
        if self.parent_viewer is not None and not self._scroll_indexing_paused and hasattr(self.parent_viewer, "_pause_semantic_indexing_deferred"):
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

        # Re-aim the queue at where the user is NOW, and drop work for tiles
        # they have scrolled well past. Both hang off this throttle rather
        # than the wheel event: a trackpad emits 60-120 of those a second and
        # re-scoring costs ~2 ms for a few hundred tasks.
        #
        # The gallery's own cancel diff only runs once scrolling has settled,
        # which is precisely when the queue is no longer flooded. During a
        # fast scroll -- when it is -- nothing was being dropped, so the tile
        # arriving on screen queued behind everything already left behind.
        self._reaim_queue_at_viewport()

        # Throttle (do NOT restart continuously): allow periodic updates while scrolling.
        # The settle timer below guarantees a final update when scrolling stops.
        interval = 50 if not self._is_scrolling_fast else 150
        if self._is_actively_scrolling() and not self._active_tasks:
            interval = max(interval, 120)
        if self._load_timer is not None and not self._load_timer.isActive():
            self._load_timer.start(interval)

        # Debounce: after scrolling stops, force a final load near thumb position.
        # Momentum motion (wheel smoothing timer ticks, native trackpad momentum)
        # must NOT keep pushing the settle timer out — that starved
        # _on_scroll_settled for the whole decay and left tiles blank until the
        # very end. During momentum, let a pending timer fire (~every 120ms) so
        # interim loads fill tiles while content is still moving.
        if self._momentum_scroll_active():
            if not self._scroll_settle_timer.isActive():
                self._scroll_settle_timer.start(120)
        else:
            if self._scroll_settle_timer.isActive():
                self._scroll_settle_timer.stop()
            self._scroll_settle_timer.start(120)

    _REAIM_MIN_INTERVAL_S = 0.10

    _REAIM_CANCEL_DISTANCE = 24

    def _viewport_row_distance(self, file_path: str) -> int:
        """Rows between ``file_path`` and the visible band. 0 = on screen."""
        try:
            indices = self._path_to_indices.get(file_path)
            if not indices:
                return self._REAIM_CANCEL_DISTANCE + 1
            first, last = self._visible_index_range()
            idx = min(indices)
            if idx < first:
                return first - idx
            if idx > last:
                return idx - last
            return 0
        except Exception:
            return 0

    def _visible_index_range(self) -> tuple:
        """(first, last) image index currently on screen; widest known guess."""
        visible = getattr(self, "_visible_widgets", None)
        if visible:
            keys = [k for k in visible if isinstance(k, int)]
            if keys:
                return min(keys), max(keys)
        return 0, 0

    def _reaim_queue_at_viewport(self) -> None:
        """Re-score queued work against the current viewport, and prune it."""
        now = time.time()
        if now - getattr(self, "_last_reaim_t", 0.0) < self._REAIM_MIN_INTERVAL_S:
            return
        self._last_reaim_t = now
        manager = getattr(self, "load_manager", None)
        if manager is None or not hasattr(manager, "reprioritise"):
            return
        try:
            manager.reprioritise(self._viewport_row_distance)
            if self._is_scrolling_fast:
                manager.cancel_distant_queued_tasks(
                    self._viewport_row_distance, self._REAIM_CANCEL_DISTANCE
                )
        except Exception:
            pass

    def _on_scroll_settled(self):
        """Called after scroll events stop; load thumbnails around thumb position."""
        if self._momentum_scroll_active():
            # Interim pass mid-momentum: fill visible tiles now instead of waiting
            # for the decay tail, but leave scroll-speed state and layout alone
            # (a rebuild here would jump the content under the user).
            self.load_visible_images()
            return
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

    def _scroll_area_widget(self) -> QScrollArea | None:
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

    def _apply_scroll_anchor_state(self, state: dict | None) -> bool:
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
            anchor_state = self._scroll_anchor_state_for_rebuild()

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
            ) and self._metadata_changed_paths:
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
        if os.environ.get("RAWVIEWER_SCROLL_PERF_DEBUG", "").strip() == "1":
            t0 = time.perf_counter()
            try:
                return self._load_visible_images_impl()
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if dt_ms > 4.0:
                    logger.info(
                        "[SCROLL_PERF] load_visible_images took %.1fms (widgets=%d layout_items=%d)",
                        dt_ms, len(self._visible_widgets), len(getattr(self, "_gallery_layout_items", []) or []),
                    )
        if os.environ.get("RAWVIEWER_SCROLL_CPROFILE", "").strip() == "1":
            target_calls = _env_int("RAWVIEWER_SCROLL_CPROFILE_CALLS", 40, minimum=1)
            n = getattr(self, "_scroll_cprofile_calls", 0)
            if n < target_calls:
                import cProfile
                import io as _io
                import pstats
                pr = cProfile.Profile()
                pr.enable()
                result = self._load_visible_images_impl()
                pr.disable()
                self._scroll_cprofile_calls = n + 1
                if not hasattr(self, "_scroll_cprofile_agg"):
                    self._scroll_cprofile_agg = pstats.Stats(pr)
                else:
                    self._scroll_cprofile_agg.add(pr)
                if self._scroll_cprofile_calls == target_calls:
                    s = _io.StringIO()
                    ps = self._scroll_cprofile_agg
                    ps.stream = s
                    ps.sort_stats("cumulative").print_stats(25)
                    logger.info("[SCROLL_CPROFILE] aggregate over %d calls:\n%s", target_calls, s.getvalue())
                return result
        return self._load_visible_images_impl()
