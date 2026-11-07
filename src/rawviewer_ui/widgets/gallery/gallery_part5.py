"""Gallery mixin part 5 of 7 for JustifiedGallery."""
from __future__ import annotations

import contextlib
import logging
import os
import time

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QScrollArea

from common_image_loader import is_raw_file
from image_load_manager import Priority
from rawviewer_ui.widgets import ThumbnailLabel

from .helpers import (
    _apply_external_gallery_caps,
    _env_int,
    _focus_gallery_switch_logs,
    _gallery_current_active_reserve,
    _gallery_prefetch_screens,
    _gallery_scheduling_budgets,
    _gallery_scroll_buffer_screens,
    _gallery_viewport_buffer_screens,
    _gallery_warmup_scheduling_budgets,
    _orient_debug_enabled,
)

logger = logging.getLogger(__name__)

class GalleryPart5:

    def _load_visible_images_impl(self):
        _sp_debug = os.environ.get("RAWVIEWER_SCROLL_PERF_DEBUG", "").strip() == "1"
        _sp_t0 = time.perf_counter() if _sp_debug else 0.0
        # Stop background preloading when loading visible images (e.g. during scroll)
        self._idle_preload_timer.stop()

        if self._building:
            self._request_load_visible_images(50)
            return

        # Entry scroll not applied yet: try once inline, then either load tiles
        # or briefly defer. Never park forever — same-order re-entry used to
        # sit here with pending scroll and zero load_visible passes.
        if getattr(self, "_pending_scroll_to_path", None) or getattr(
            self, "_pending_scroll_anchor_state", None
        ):
            attempts = int(getattr(self, "_scroll_to_file_attempts", 0) or 0)
            if attempts < 40:
                with contextlib.suppress(Exception):
                    self._apply_pending_scroll_to_file()
            if (
                getattr(self, "_pending_scroll_to_path", None)
                or getattr(self, "_pending_scroll_anchor_state", None)
            ) and int(getattr(self, "_scroll_to_file_attempts", 0) or 0) < 40:
                self._request_load_visible_images(50)
                return
            if getattr(self, "_pending_scroll_to_path", None) or getattr(
                self, "_pending_scroll_anchor_state", None
            ):
                self._abandon_pending_scroll_to_file()

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
                len(self._active_tasks),
            )
        visible_indices_set = {idx for idx, item in visible_indices_items}

        # Dynamic prefetch: follow the scrollbar thumb/viewport center position.
        # Slow scroll: keep more around the thumb; fast scroll: keep less.
        # Note: we already early-return on fast scroll, so this mainly helps "normal" scrolling.
        screens = _gallery_prefetch_screens(self._is_scrolling_fast)

        last_y = getattr(self, "_last_scheduled_scroll_y", scroll_y)
        delta_y = scroll_y - last_y

        # Velocity-aware directional prefetch window
        if delta_y > 0:  # Scrolling down
            # Anchor near top of viewport, extend heavily downwards
            prefetch_top = max(0, scroll_y - buffer_h)
            prefetch_span = max(int(v_h * screens), buffer_h) + buffer_h
        elif delta_y < 0:  # Scrolling up
            # Anchor near bottom of viewport, extend heavily upwards
            prefetch_span = max(int(v_h * screens), buffer_h) + buffer_h
            prefetch_top = max(0, scroll_y + v_h + buffer_h - prefetch_span)
        else:  # Stationary / Jump
            center_y = scroll_y + (v_h // 2)
            half_span = int((v_h * screens) // 2)
            prefetch_top = max(0, center_y - half_span)
            prefetch_span = max(int(v_h * screens), buffer_h)

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
            # flush_queue clears the manager's active map but does not emit
            # task_completed — drop the gallery's parallel bookkeeping or
            # phantom CURRENT entries block reschedule until the 4s sweep.
            with contextlib.suppress(Exception):
                self._active_tasks.clear()
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
            # A tile that failed earlier is showing a grey placeholder. Now
            # that scrolling has settled, give the visible ones another go --
            # the failure was often transient (an EMFILE burst, IO pressure,
            # a volume that briefly dropped) and without this the tile stayed
            # grey until the folder was reopened.
            if self._placeholder_thumb_paths:
                with contextlib.suppress(Exception):
                    self.retry_placeholder_thumbnails(visible_paths)

            # Cancel thumbnail work that is no longer near the scrollbar thumb.
            # Skip cancel for visible RAW/DNG tiles still decoding at CURRENT priority.
            to_cancel = self._requested_thumbnail_paths - wanted_paths
            protected_paths: set[str] = set()
            for fp in list(to_cancel):
                if self._should_protect_active_raw_thumbnail(fp, visible_paths):
                    protected_paths.add(fp)
                    continue
                with contextlib.suppress(Exception):
                    self.load_manager.cancel_task(fp)
                self._active_tasks.pop(fp, None)
            self._requested_thumbnail_paths = wanted_paths | protected_paths
        else:
            # Still cancel offscreen CURRENT work while scrolling — otherwise
            # active_cap stays filled with tiles the user already passed until
            # the 4s sweep, starving the new viewport.
            to_cancel = self._requested_thumbnail_paths - visible_paths
            for fp in list(to_cancel):
                with contextlib.suppress(Exception):
                    self.load_manager.cancel_task(fp)
                self._active_tasks.pop(fp, None)
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
        # Main-thread budget per pass: the sync cache path below (disk read +
        # JPEG decode + scale per tile) ran for seconds on a cold gallery,
        # freezing the UI. Overflow tiles resume on the next 16ms tick.
        pass_t0 = time.perf_counter()
        pass_budget_s = (
            _env_int("RAWVIEWER_GALLERY_PASS_BUDGET_MS", 50, minimum=10) / 1000.0
        )
        if actively_scrolling:
            # A live gesture needs ~60fps paints; a 50ms(+2x schedule) pass
            # drops most frames and reads as trackpad lag. Fill catches up on
            # the settle pass.
            pass_budget_s = min(
                pass_budget_s,
                _env_int("RAWVIEWER_GALLERY_SCROLL_PASS_BUDGET_MS", 12, minimum=4)
                / 1000.0,
            )
        pass_processed = 0
        pass_budget_exhausted = False
        for idx, item in visible_indices_items:
            if (
                pass_processed > 0
                and (time.perf_counter() - pass_t0) > pass_budget_s
            ):
                pass_budget_exhausted = True
                break
            pass_processed += 1
            path = item.get("file_path")
            rect = item.get("rect")
            if not path or rect is None:
                continue

            if idx not in self._visible_widgets:
                if created_widgets >= max_widgets:
                    # Defer widget creation but keep updating slots that already exist.
                    if self._load_timer is not None and not self._load_timer.isActive():
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
                                # Check-circle click toggles selection without
                                # opening or starting a drag; remember it so the
                                # release handler skips the mirrored click path.
                                if _w.hits_selection_check(e.position().toPoint()):
                                    _w._press_on_check = True
                                    _w._drag_start_pos = None
                                    if hasattr(pv, "_gallery_toggle_check_path"):
                                        pv._gallery_toggle_check_path(target_path)
                                    elif hasattr(pv, "_gallery_toggle_path_selection"):
                                        pv._gallery_toggle_path_selection(target_path)
                                    return
                                if _w.hits_copy_path_button(e.position().toPoint()):
                                    # Copy-path button: copy the file path and
                                    # swallow the click so the image never opens
                                    # and no drag/selection starts.
                                    _w._press_on_copy_path = True
                                    _w._drag_start_pos = None
                                    try:
                                        pv._copy_paths_to_clipboard([target_path])
                                        if getattr(pv, "status_bar", None) is not None:
                                            pv.status_bar.showMessage(
                                                "Path copied to clipboard", 2500
                                            )
                                    except Exception:
                                        pass
                                    return
                                _w._press_on_check = False
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
                    with contextlib.suppress(Exception):
                        e.ignore()

                def _on_thumb_release(e, _w=w):
                    if getattr(_w, "_press_on_check", False):
                        # Press handled the check toggle; swallow the release so
                        # the plain-click path (open image / re-toggle) never runs.
                        _w._press_on_check = False
                        _w._drag_start_pos = None
                        _w._drag_started = False
                        e.accept()
                        return
                    if getattr(_w, "_press_on_copy_path", False):
                        # Press handled the copy-path button; same swallow.
                        _w._press_on_copy_path = False
                        _w._drag_start_pos = None
                        _w._drag_started = False
                        e.accept()
                        return
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

                def _on_thumb_enter(e, _w=w):
                    self._on_gallery_thumb_hover(_w, entered=True)
                    ThumbnailLabel.enterEvent(_w, e)

                def _on_thumb_leave(e, _w=w):
                    self._on_gallery_thumb_hover(_w, entered=False)
                    ThumbnailLabel.leaveEvent(_w, e)

                w.mousePressEvent = _on_thumb_press
                w.mouseReleaseEvent = _on_thumb_release
                w.enterEvent = _on_thumb_enter
                w.leaveEvent = _on_thumb_leave
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

            is_scrolling = (time.time() - self._last_scroll_event_t) < 0.25
            scaled_key = self._scaled_cache_key(path, physical_size, fast=False)
            cached_scaled = self._thumbnail_cache.get(scaled_key)

            if not cached_scaled and is_scrolling:
                fast_key = self._scaled_cache_key(path, physical_size, fast=True)
                cached_scaled = self._thumbnail_cache.get(fast_key)
                if cached_scaled and not cached_scaled.isNull():
                    scaled_key = fast_key

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
                elif cached_scaled.size() == physical_size:
                    # Compare PHYSICAL pixel sizes. Reconstructing the logical size
                    # via round(px / dpr) disagrees with the int(rect * dpr) used to
                    # build the pixmap on fractional DPR (125%/150% Windows): e.g.
                    # rect 311x207 @1.25 -> pixmap 388x258 -> round back = 310x206.
                    # That off-by-one rejected every cached scaled tile, so each
                    # visible pass re-scaled the same leading tiles, exhausted the
                    # pass budget, and never reached the rest of the viewport after
                    # a section jump (black gallery + infinite 16ms reschedule).
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
                    if is_scrolling:
                        pass
                    elif (
                        time.time() - self._base_fetch_miss_ts.get(path, 0.0)
                    ) < 2.0:
                        # Recently missed: the async thumbnail task will deliver.
                        # Re-fetching every 16ms pass cost ~5ms/tile and starved
                        # the rest of the viewport after a cold section jump.
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
                        if base is None:
                            self._base_fetch_miss_ts[path] = time.time()

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
                        scaled = self._fit_tile_pixmap(path, base, physical_size, dpr, fast=is_scrolling)
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
            pv = self.parent_viewer
            # Selection state applies on every bind (cheap bool + repaint);
            # skipping it while scrolling left stale/missing checks after fast
            # scroll-settles. Badges are QSS-heavy widgets, keep their guard.
            if pv is not None and hasattr(pv, "_is_gallery_path_selected") and hasattr(w, "set_gallery_selected"):
                    w.set_gallery_selected(pv._is_gallery_path_selected(path))
            if not actively_scrolling:
                if pv is not None and hasattr(pv, "_burst_stack_count_for_path") and hasattr(w, "set_burst_stack_count"):
                        w.set_burst_stack_count(pv._burst_stack_count_for_path(path))
                if pv is not None and hasattr(pv, "_is_gallery_path_edited") and hasattr(w, "set_gallery_edited"):
                        w.set_gallery_edited(pv._is_gallery_path_edited(path))
                if pv is not None and hasattr(pv, "_gallery_path_rating") and hasattr(w, "set_gallery_rating"):
                        w.set_gallery_rating(pv._gallery_path_rating(path))
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

        _sp_t_loop = time.perf_counter() if _sp_debug else 0.0
        if pass_budget_exhausted:
            logger.info(
                "[GALLERY] load_visible_images pass budget hit after %d/%d tiles; resuming next tick",
                pass_processed,
                len(visible_indices_items),
            )
            if self._load_timer is not None and not self._load_timer.isActive():
                self._load_timer.start(16)

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

        _sp_t_prep = time.perf_counter() if _sp_debug else 0.0
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
            if has_pending_current or self._evict_stale_active_tasks(max_age_s=2.0) == 0:
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
        for path, priority, stages, *_rest in load_tasks:
            if scheduled >= max_tasks:
                if self._load_timer is not None and not self._load_timer.isActive():
                    self._load_timer.start(16)
                break
            # load_image() can complete synchronously for memory-cache hits
            # (thumbnail_ready emitted inline), so this loop is main-thread
            # work too — share the pass budget with the tile loop above.
            if (
                scheduled > 0
                and (time.perf_counter() - pass_t0) > 2 * pass_budget_s
            ):
                if self._load_timer is not None and not self._load_timer.isActive():
                    self._load_timer.start(16)
                break
            if priority != Priority.CURRENT and self._prefetch_active_count() >= prefetch_active_cap:
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
            _li_t0 = time.perf_counter() if _sp_debug else 0.0
            self.load_manager.load_image(
                path,
                priority=priority,
                cancel_existing=False,
                stages=stages,
                gallery_thumbnail=True,
                thumbnail_target_size=None,
            )
            if _sp_debug:
                _li_ms = (time.perf_counter() - _li_t0) * 1000.0
                if _li_ms > 50.0:
                    logger.info(
                        "[SCROLL_PERF] load_image %.0fms path=%s stages=%s",
                        _li_ms,
                        os.path.basename(path),
                        sorted(stages),
                    )
            self._mark_active_task(path, priority)
            scheduled += 1
        scheduled_tasks = scheduled

        if _sp_debug:
            _sp_t_end = time.perf_counter()
            _sp_total_ms = (_sp_t_end - _sp_t0) * 1000.0
            if _sp_total_ms > 100.0:
                logger.info(
                    "[SCROLL_PERF] impl sections: pre_loop=%.0fms tile_loop=%.0fms "
                    "prep=%.0fms schedule=%.0fms total=%.0fms",
                    (pass_t0 - _sp_t0) * 1000.0,
                    (_sp_t_loop - pass_t0) * 1000.0,
                    (_sp_t_prep - _sp_t_loop) * 1000.0,
                    (_sp_t_end - _sp_t_prep) * 1000.0,
                    _sp_total_ms,
                )

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
            if self._load_timer is None or not self._load_timer.isActive():
                self._request_load_visible_images(200)
        else:
            self._idle_preload_timer.stop()
