"""Gallery mixin part 3 of 7 for JustifiedGallery."""
from __future__ import annotations

import contextlib
import logging
import time

import numpy as np
from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QScrollArea

from common_image_loader import is_raw_file
from image_load_manager import Priority

from .helpers import (
    _focus_gallery_switch_logs,
    _min_layout_pixmap_dim,
    _rotate_pixmap_cw,
    _thumbnail_data_to_base_pixmap,
)

logger = logging.getLogger(__name__)

class GalleryPart3:

    def _thumb_cache_byte_budget(self) -> int:
        """RAM-tiered QPixmap budget (128–768 MiB), scaled under memory pressure."""
        cached = getattr(self, "_thumb_cache_byte_budget_cached", None)
        if cached is None:
            try:
                from core.profile import classify_memory_tier, system_total_ram_gb

                tier = classify_memory_tier(system_total_ram_gb())
            except Exception:
                tier = "balanced"
            mb = {
                "low": 128,
                "medium": 256,
                "balanced": 384,
                "high": 512,
                "ultra": 768,
            }.get(tier, 384)
            self._thumb_cache_byte_budget_cached = int(mb) * 1024 * 1024
            cached = self._thumb_cache_byte_budget_cached
        try:
            from core.memory_pressure import scale_thumb_budget_bytes

            return scale_thumb_budget_bytes(int(cached))
        except Exception:
            return int(cached)

    def _enforce_thumbnail_cache_budget(self, *, force: bool = False) -> None:
        """Evict LRU thumbnails until the cache is under the byte budget."""
        if not force:
            self._thumb_cache_puts_since_check = getattr(self, "_thumb_cache_puts_since_check", 0) + 1
            if self._thumb_cache_puts_since_check < 64:
                return
        self._thumb_cache_puts_since_check = 0
        cache = self._thumbnail_cache
        with cache.lock:
            total = 0
            sizes = {}
            for k, pm in cache.cache.items():
                try:
                    b = int(pm.width()) * int(pm.height()) * 4
                except Exception:
                    b = 4096
                sizes[k] = b
                total += b
            while total > self._thumb_cache_byte_budget() and cache.cache:
                k, _pm = cache.cache.popitem(last=False)
                total -= sizes.get(k, 0)

    def _store_oriented_base_pixmap(self, file_path: str, pixmap: QPixmap) -> QPixmap:
        """Orient, cache base thumbnail, and drop stale scaled variants."""
        oriented = self._orient_gallery_base_pixmap(file_path, pixmap)
        if oriented is None or oriented.isNull():
            return pixmap
        self._thumbnail_cache.put((file_path, self._thumb_base_key), oriented)
        self._enforce_thumbnail_cache_budget()
        self._invalidate_scaled_thumbnails_for_path(file_path)
        return oriented

    def _global_cache_to_base_pixmap(self, path: str) -> QPixmap | None:
        """Pull an oriented base pixmap from the global ImageCache for gallery tiles.

        Memory tiers only — ``get_grid``/``get_thumbnail`` fall through to disk
        JPEG decode on the UI thread (same stall filmstrip already avoided).
        Disk misses return None so async thumbnail loads fill the tile.
        """
        try:
            from image_cache import get_image_cache

            global_cache = get_image_cache()
            global_thumb = global_cache.get_grid_memory_only(path)
            if global_thumb is None:
                global_thumb = global_cache.get_thumbnail_memory_only(path)
            if global_thumb is None:
                return None
            arr = self._orient_gallery_thumbnail_array(path, np.ascontiguousarray(global_thumb))
            arr = self._apply_edits_to_tile(path, arr)
            return _thumbnail_data_to_base_pixmap(arr)
        except Exception:
            return None

    def _layout_aspect_for_path(
        self, file_path: str, pixmap: QPixmap | None = None
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

    def _layout_aspect_needs_rebuild(self, old_aspect: float, new_aspect: float) -> bool:
        """True when tile geometry must change (including portrait ↔ landscape flip)."""
        old_o = self._aspect_orientation_flag(old_aspect)
        new_o = self._aspect_orientation_flag(new_aspect)
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

    def _reconcile_tile_aspect(self, file_path: str, pixmap: QPixmap | None) -> bool:
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

    def _fit_rotated_thumbnail(self, file_path: str, base: QPixmap, target_size, fast: bool = False):
        """Apply user rotation then crop-to-fit (tile aspect matches after layout rebuild)."""
        rot = self._get_rotation_degrees_for_path(file_path)
        oriented = _rotate_pixmap_cw(base, rot)
        return self._scale_crop_to_fit(oriented, target_size, fast=fast)

    def invalidate_thumbnails_for_path(self, file_path: str) -> None:
        """Drop cached gallery pixmaps for a path (e.g. after on-disk rotation or a saved edit)."""
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        with contextlib.suppress(Exception):
            self._thumbnail_cache.remove_keys_for_file_path(file_path)
        # Drop the remembered aspect so a re-decode (e.g. after on-disk rotation) re-measures.
        self._measured_raw_aspects.pop(file_path, None)
        self._base_fetch_miss_ts.pop(file_path, None)
        # Clearing the cache alone doesn't repaint an already-visible tile --
        # it's still showing the old QPixmap set the last time this widget
        # was laid out. Force it back through the cache-miss path (which
        # re-requests a fresh decode) on the next paint pass so a saved edit
        # shows up in the grid without requiring a scroll/relayout.
        needs_reload = False
        for w in self._visible_widgets.values():
            if getattr(w, "file_path", None) == file_path:
                self._clear_widget_thumbnail(w)
                needs_reload = True
        if needs_reload and not self._is_scrolling_fast:
            with contextlib.suppress(Exception):
                self.load_visible_images()

    def _prune_path_caches(self, keep: set) -> None:
        """Forget cached data for paths that are no longer on show.

        Keeps entries for files still present, so re-sorting or filtering the
        same folder does not throw away work; only a genuine change of
        contents evicts.
        """
        try:
            stale = [p for p in self._metadata_cache if p not in keep]
            for p in stale:
                self._metadata_cache.pop(p, None)
        except Exception:
            self._metadata_cache = {}

        cache = getattr(self, "_thumbnail_cache", None)
        if cache is None:
            return
        try:
            with cache.lock:
                # Keys are (file_path, size_bucket) tuples, so compare the
                # PATH ELEMENT against the keep set.
                #
                # This was `any(str(k).startswith(p) for p in keep)`, which was
                # wrong twice over: str() of a tuple yields "('/path', 512)",
                # so it never matched any path and the whole thumbnail cache
                # was wiped on every folder change -- and being O(keys x paths)
                # it took ~27 SECONDS on the GUI thread for a 3000-file folder.
                # A set lookup on the path element is ~4 ms for the same input.
                stale = [
                    k for k in list(cache.cache.keys())
                    if (k[0] if isinstance(k, tuple) and k else k) not in keep
                ]
                for key in stale:
                    cache.cache.pop(key, None)
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

    def _active_task_started_ts(self, path: str) -> float | None:
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
        with contextlib.suppress(Exception):
            self._schedule_viewport_width_rebuild()
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
                with contextlib.suppress(Exception):
                    scrollbar.valueChanged.disconnect(self._on_scroll)
                scrollbar.valueChanged.connect(self._on_scroll)
                # Dragging the scrollbar thumb should trigger immediate loading on release
                with contextlib.suppress(Exception):
                    scrollbar.sliderPressed.disconnect(self._on_slider_pressed)
                with contextlib.suppress(Exception):
                    scrollbar.sliderReleased.disconnect(self._on_slider_released)
                scrollbar.sliderPressed.connect(self._on_slider_pressed)
                scrollbar.sliderReleased.connect(self._on_slider_released)

                # Install wheel event filter on viewport for smooth scrolling
                with contextlib.suppress(Exception):
                    p.viewport().removeEventFilter(self)
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
                try:
                    if event.phase() == Qt.ScrollPhase.ScrollMomentum:
                        self._native_momentum_t = time.time()
                except Exception:
                    pass
                if not pixel.isNull() and pixel.y() != 0:
                    if abs(self._trackpad_gain - 1.0) < 0.01:
                        # Unity gain: let QScrollArea handle natively (smoother on macOS).
                        return False
                    # Amplified trackpad: scale the pixel delta directly on the
                    # scrollbar. macOS momentum still applies (it arrives as
                    # more pixelDelta events), so the feel stays native.
                    #
                    # Carry the fraction between events. int() per event threw
                    # away everything below a whole pixel, and a trackpad sends
                    # 1-3px deltas: at the default 1.6 gain a stream of
                    # [1,1,2,1,3,...] scrolled 19px where it should have moved
                    # 24, unevenly, and any delta whose product rounded to 0
                    # vanished entirely. That is the choppiness against a mouse
                    # wheel, which reaches the scrollbar through a smoothing
                    # timer instead.
                    self._trackpad_accum_px += -pixel.y() * self._trackpad_gain
                    step = int(self._trackpad_accum_px)
                    self._trackpad_accum_px -= step
                    if step:
                        sb.setValue(
                            max(
                                sb.minimum(),
                                min(sb.maximum(), sb.value() + step),
                            )
                        )
                    self._last_scroll_event_t = time.time()
                    event.accept()
                    return True
                if angle.y() != 0:
                    notches = angle.y() / 120.0
                    # Velocity-scaled notch distance: slow clicks stay precise
                    # (~120px), fast spins ramp toward _wheel_fast_gain so a
                    # flick traverses like holding the Down key. Gains come from
                    # gallery_scroll (same module as key singleStep*N).
                    now_t = time.time()
                    dt_notch = now_t - self._wheel_last_notch_t
                    self._wheel_last_notch_t = now_t
                    gain = self._wheel_base_gain
                    if 0.0 < dt_notch < 0.20:
                        # 200ms→1x ramping to max gain at <=40ms between notches
                        speed = min(1.0, (0.20 - dt_notch) / 0.16)
                        gain *= 1.0 + (self._wheel_fast_gain - 1.0) * speed
                    delta_y = int(notches * 120 * gain)
                    self._wheel_accum_px += -float(delta_y)
                    if not self._wheel_timer.isActive():
                        self._wheel_last_tick_t = 0.0  # fresh burst: dt from nominal tick
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
            # An interim settle may have consumed the pending timer; make sure a
            # real (non-momentum) settle still fires now that motion has ended.
            if self._scroll_settle_timer is not None:
                self._scroll_settle_timer.start(120)
            return

        # Exponential decay of the remaining distance, scaled by MEASURED
        # elapsed time (see _wheel_decay_per_tick): fraction consumed is
        # 1 - (1 - rate)^(dt / nominal_tick), so a tick that arrives late
        # consumes proportionally more and the on-screen velocity stays
        # constant under timer jitter. The old fixed 20%-per-tick step turned
        # every late tick into a visible slowdown-then-jump. The minimum step
        # (which keeps the tail from crawling) is dt-scaled the same way.
        now = time.time()
        dt_ticks = 1.0
        if self._wheel_last_tick_t > 0.0:
            dt_ticks = min(
                4.0, max(0.25, (now - self._wheel_last_tick_t) * 1000.0 / self._wheel_tick_ms)
            )
        self._wheel_last_tick_t = now
        frac = 1.0 - (1.0 - self._wheel_decay_per_tick) ** dt_ticks
        adaptive_step = self._wheel_accum_px * frac
        min_step = self._wheel_step_px * dt_ticks
        step = (min_step if self._wheel_accum_px > 0 else -min_step) if abs(adaptive_step) < min_step else adaptive_step

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
            if self._scroll_settle_timer is not None:
                self._scroll_settle_timer.start(120)
            return

        sb.setValue(new_val)

    def set_capture_date_provider(self, provider) -> None:
        """Install ``path -> datetime | None`` used by the scroll date bubble.

        Injected rather than looked up here: capture times live in the image
        cache on the main window, and the gallery has no business reaching
        into it. With no provider the bubble simply never appears, which is
        the right behaviour for any host that has no dates to give.
        """
        self._capture_date_provider = provider

    _SCROLL_BUBBLE_THROTTLE_S = 0.075  # 75ms; keep overlay work off every scroll tick

    def _scroll_preview_path(self) -> str:
        """``file_path`` of the layout item at the current scrollbar value."""
        scroll = getattr(self, "_scroll_area", None)
        if scroll is None:
            return ""
        try:
            idx = self._layout_index_at_scroll_y(
                int(scroll.verticalScrollBar().value())
            )
            if idx is None:
                return ""
            item = self._gallery_layout_items[idx]
            return item.get("file_path") or ""
        except Exception:
            return ""

    def _scroll_date_text(self, path: str | None = None) -> str:
        """Capture date of the photo at the top of the viewport, if known."""
        provider = getattr(self, "_capture_date_provider", None)
        if provider is None:
            return ""
        if not path:
            path = self._scroll_preview_path()
        if not path:
            return ""
        try:
            when = provider(path)
            if when is None:
                return ""
            # Month and year, with the day: dragging covers months per
            # centimetre, so a timestamp would be false precision and would
            # change on every pixel of travel.
            return when.strftime("%d %b %Y")
        except Exception:
            return ""
