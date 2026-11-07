"""gallery mixin, part 1 of 5 (35 methods, 895 lines)."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, cast

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap

from core.config import _env_int, _norm_path

from .. import lazy
from ..lazy import Priority, is_raw_file

lazy.register(sys.modules[__name__])


class GalleryPart1:
    def _sync_composition_grid_display(self) -> None:
        vm = getattr(self, "view_mode", "single")
        mode = getattr(self, "_composition_grid_mode", "off")
        if vm == "compare":
            for view in (
                getattr(self, "compare_select_view", None),
                getattr(self, "compare_candidate_view", None),
            ):
                if view is not None:
                    if getattr(view, "file_path", "") and self.is_animated_image(view.file_path):
                        view.set_composition_grid_mode("off")
                    else:
                        view.set_composition_grid_mode(mode)
                    view.update()
        else:
            path = getattr(self, "current_file_path", None)
            is_anim = path and self.is_animated_image(path)
            gv = getattr(self, "gpu_view", None)
            if gv is not None:
                if is_anim:
                    gv.set_composition_grid_mode("off")
                else:
                    gv.set_composition_grid_mode(mode)
            self._redraw_single_view_pixmap_without_relayout()

    def _wire_filmstrip(self) -> None:
        bar = self._filmstrip_bar()
        if bar is None:
            return
        bar.committed.connect(self._on_filmstrip_committed)
        bar.thumbnails_needed.connect(self._on_filmstrip_thumbnails_needed)
        bar.setEnabled(False)
        bar.hide()
        self._hide_filmstrip_chrome()

    def set_filmstrip_pointer_active(self, active: bool) -> None:
        self._filmstrip_pointer_active = active
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "set_filmstrip_pointer_active"):
            container.set_filmstrip_pointer_active(active)

    def _hide_filmstrip_chrome(self) -> None:
        container = getattr(self, "single_view_container", None)
        if container is None:
            return
        if hasattr(container, "_hide_filmstrip_layer"):
            container._hide_filmstrip_layer()
        if hasattr(container, "_filmstrip_reveal"):
            container._filmstrip_reveal = False
        if hasattr(container, "_filmstrip_opacity"):
            container._filmstrip_opacity.setOpacity(0.0)

    def _filmstrip_bar(self):
        container = getattr(self, "single_view_container", None)
        if container is None or not hasattr(container, "filmstrip_widget"):
            return None
        return container.filmstrip_widget()

    def _filmstrip_handles_input(self) -> bool:
        if getattr(self, "view_mode", "single") != "single":
            return False
        if getattr(self, "_adjust_overlay_visible", False):
            return False
        bar = self._filmstrip_bar()
        if bar is None or not bar.isVisible() or not bar.isEnabled():
            return False
        container = getattr(self, "single_view_container", None)
        if container is None:
            return False
        if getattr(container, "_filmstrip_pointer_active", False):
            return True
        if getattr(self, "_filmstrip_pointer_active", False):
            return True
        return container._pointer_in_bottom_hotzone()

    def _filter_paths_to_bookmarked(self, files: list[str]) -> list[str]:
        bookmarked = getattr(self, "_gallery_bookmarked_paths", set()) or set()
        if not bookmarked:
            return []
        return [p for p in files if p and _norm_path(p) in bookmarked]

    def _filter_burst_rejected_paths(self, files: list[str]) -> list[str]:
        rejected = getattr(self, "_burst_rejected_paths", set()) or set()
        if not rejected:
            return list(files or [])
        return [p for p in files if p and _norm_path(p) not in rejected]

    def _gallery_capture_date(self, path: str):
        """Capture datetime for one photo, for the gallery's scroll date bubble.

        Called once per scrollbar tick while the thumb is dragged, so it reads
        a per-folder dict built once rather than hitting the EXIF cache per
        call. A path it does not know simply has no date, and the bubble stays
        hidden for it rather than blocking the drag to go and find one.

        The map is rebuilt on a folder change, and re-attempted when it came
        back empty or when an unknown path turns up -- both of which happen
        normally and neither of which may be cached as "no dates here".
        """
        import datetime

        if not path:
            return None
        folder = getattr(self, "current_folder", None)
        if getattr(self, "_gallery_capture_date_missing_folder", None) != folder:
            # Misses are per-folder; carrying them across would suppress the
            # retry for a path that a different folder happened to lack, and
            # would grow without bound over a session.
            self._gallery_capture_date_missing = set()
            self._gallery_capture_date_missing_folder = folder
        cached_folder = getattr(self, "_gallery_capture_date_folder", None)
        if cached_folder != folder or not hasattr(self, "_gallery_capture_date_map"):
            raw = {}
            try:
                if folder:
                    raw = self.image_cache.get_capture_times_for_folder_sort(folder) or {}
                if not raw:
                    raw = self._capture_times_for_paths(
                        list(getattr(self, "image_files", []) or [])[:4096]
                    )
            except Exception:
                raw = {}
            parsed = {}
            for p, value in (raw or {}).items():
                when = None
                try:
                    if isinstance(value, (int, float)):
                        when = datetime.datetime.fromtimestamp(float(value))
                    elif isinstance(value, str) and value:
                        # Stored ISO, but older rows used "YYYY:MM:DD hh:mm:ss".
                        try:
                            when = datetime.datetime.fromisoformat(value)
                        except ValueError:
                            when = datetime.datetime.strptime(
                                value, "%Y:%m:%d %H:%M:%S"
                            )
                except Exception:
                    when = None
                if when is not None:
                    parsed[_norm_path(p)] = when
            if not parsed:
                # Do NOT cache an empty result. The first call can land before
                # EXIF indexing has written any rows, and caching {} against
                # this folder would retire the date bubble for the rest of the
                # session -- it would never appear again even once every date
                # was known. Retrying costs one cache read per drag.
                self._gallery_capture_date_map = {}
                self._gallery_capture_date_folder = None
                return None
            self._gallery_capture_date_map = parsed
            self._gallery_capture_date_folder = folder
        key = _norm_path(path)
        cached = self._gallery_capture_date_map.get(key)
        if cached is None:
            # Either a photo written into the open folder after the map was
            # built (HDR merge, panorama, export all appear in the gallery
            # immediately) or one that genuinely carries no capture date.
            #
            # Retry ONCE per unseen path, remembering the misses. Invalidating
            # on every miss instead would rebuild the whole folder map on each
            # scroll tick for any dateless photo -- during a drag, per tick.
            missed = self._gallery_capture_date_missing
            if key not in missed:
                missed.add(key)
                self._gallery_capture_date_folder = None
        return cached

    def _burst_collapsed_gallery_paths(self, files: list[str]) -> list[str]:
        from core.burst_grouping import build_collapsed_display_paths

        if not self._is_burst_grouping_enabled():
            return self._filter_burst_rejected_paths(files)
        if getattr(self, "_burst_group_view_active", False):
            return list(getattr(self, "_burst_group_view_members", []) or [])
        path_to_group = getattr(self, "_burst_path_to_group", {}) or {}
        if not path_to_group:
            return self._filter_burst_rejected_paths(files)
        rejected = self._burst_visibility_exclusion_set()
        covers = dict(getattr(self, "_burst_group_covers", {}) or {})
        covers.update(getattr(self, "_burst_stack_covers", {}) or {})
        return build_collapsed_display_paths(files, path_to_group, covers, rejected)

    def _gallery_norm_path_map(self) -> dict:
        """One norm-key → path dict for the current folder list (O(1) lookups)."""
        return {
            _norm_path(p): p for p in (getattr(self, "image_files", None) or [])
        }

    def _gallery_path_for_norm_key(self, key: str) -> str | None:
        m = getattr(self, "_gallery_norm_map_cache", None)
        if m is None or m[0] is not self.image_files:
            m = (self.image_files, self._gallery_norm_path_map())
            self._gallery_norm_map_cache = m
        return m[1].get(key)

    def _gallery_selection_keys_for_tile(self, file_path: str) -> set:
        """Norm keys toggled when selecting a collapsed stack cover or single tile."""
        canonical = self._gallery_canonical_path(file_path)
        if not canonical:
            return set()
        key = _norm_path(canonical)
        from core.burst_grouping import BURST_MIN_COUNT

        if (
            getattr(self, "view_mode", "") == "gallery"
            and not getattr(self, "_burst_group_view_active", False)
            and self._is_burst_grouping_enabled()
            and self._burst_stack_count_for_path(canonical) >= BURST_MIN_COUNT
        ):
            members = self._burst_members_for_path(canonical)
            if members:
                return {_norm_path(m) for m in members}
        return {key}

    def _gallery_selected_expanded_norm_keys(self) -> set:
        raw = set(getattr(self, "_gallery_selected_paths", set()) or set())
        if not raw:
            return set()
        if getattr(self, "_burst_group_view_active", False) or not self._is_burst_grouping_enabled():
            return raw
        norm_map = self._gallery_norm_path_map()
        expanded: set = set()
        processed_groups: set = set()
        for key in sorted(raw):
            path = norm_map.get(key)
            if path is None:
                expanded.add(key)
                continue
            tile_keys = self._gallery_selection_keys_for_tile(path)
            if len(tile_keys) > 1:
                group_token = min(tile_keys)
                if group_token in processed_groups:
                    continue
                processed_groups.add(group_token)
            expanded |= tile_keys
        return expanded

    def _gallery_selected_canonical_paths_from_keys(self, keys: set) -> list[str]:
        if not keys:
            return []
        return [p for p in (self.image_files or []) if _norm_path(p) in keys]

    def _gallery_effective_files(self) -> list[str]:
        """Current gallery scope: folder/search results, optionally ∩ bookmarks ∩ min rating."""
        files = list(getattr(self, "image_files", []) or [])
        if getattr(self, "_gallery_bookmark_filter_active", False):
            files = self._filter_paths_to_bookmarked(files)
        files = self._filter_paths_to_min_rating(
            files, int(getattr(self, "_gallery_rating_filter_min", 0) or 0)
        )
        files = self._filter_burst_rejected_paths(files)
        if getattr(self, "_burst_group_view_active", False):
            return self._filter_burst_rejected_paths(
                list(getattr(self, "_burst_group_view_members", []) or [])
            )
        if getattr(self, "view_mode", "") == "gallery":
            return self._burst_collapsed_gallery_paths(files)
        return files

    def _adopt_gallery_navigation_path(self) -> None:
        """When leaving gallery, align single-view navigation with the gallery viewport."""
        gj = getattr(self, "gallery_justified", None)
        anchor = None
        if gj is not None and hasattr(gj, "get_scroll_anchor_path"):
            try:
                anchor = gj.get_scroll_anchor_path()
            except Exception:
                anchor = None
        if not anchor:
            anchor = getattr(self, "_gallery_scroll_target_path", None)
        if not anchor:
            return
        target = _norm_path(anchor)
        for i, p in enumerate(self.image_files or []):
            if _norm_path(p) == target:
                self.current_file_index = i
                self.current_file_path = self.image_files[i]
                return
        self.current_file_path = anchor

    def _gallery_thumb_max_dim(self, file_path: str) -> int:
        pm = self._gallery_preview_pixmap_for_path(file_path)
        if pm is None or pm.isNull():
            return 0
        return max(pm.width(), pm.height())

    def _finish_gallery_to_single_without_redisplay(self, file_path: str) -> None:
        """Gallery→single: keep the buffer already on screen; warm neighbors only."""
        self._loading_from_gallery = False
        self._gallery_preview_pending_full = False
        self._clear_pending_resolution_crossfade()
        self._cancel_content_crossfade()
        self._gallery_instant_display_quality = False
        self._last_loaded_path = file_path
        if not self.image_cache.get_exif(file_path):
            self.image_manager.load_image(
                file_path,
                priority=Priority.BACKGROUND,
                cancel_existing=False,
                use_full_resolution=False,
                stages={"exif"},
            )
        # Fit browse: do not immediately pull sensor-covering embedded JPEG
        # (6K–8K). Upgrade 512→display-tier (~2304) as CURRENT, then idle-queue
        # full only after the user pauses -- matches arrow-key nav behavior.
        if self._single_view_is_fit_mode() and not self._preserve_zoom_navigation_active():
            shown = int(getattr(self, "_manager_displayed_max_dim", 0))
            try:
                from image_cache import memory_preview_max_edge

                need_display = shown < int(memory_preview_max_edge() * 0.85)
            except Exception:
                need_display = shown < 1400
            if need_display:
                self.image_manager.load_image(
                    file_path,
                    priority=Priority.CURRENT,
                    cancel_existing=False,
                    use_full_resolution=False,
                    stages={"thumbnail", "exif"},
                )
            self._schedule_idle_full_decode_after_nav_preview(file_path)
        else:
            self._maybe_queue_background_full_decode(file_path)
        self._start_preloading()
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()
        self.setFocus()
        self.schedule_save_session_state()

    def _ensure_filmstrip_synced(self) -> None:
        """Ensure filmstrip matches the current navigation scope (incl. search filter)."""
        bar = self._filmstrip_bar()
        if bar is None:
            return
        nav = self._navigation_files()
        if len(nav) <= 1:
            return
        try:
            stale = tuple(bar.current_files()) != tuple(nav)
        except Exception:
            stale = True
        if not bar.isEnabled() or stale:
            self._sync_filmstrip_to_folder()

    def _sync_filmstrip_to_folder(self) -> None:
        if getattr(self, "_sync_filmstrip_in_progress", False):
            return
        self._sync_filmstrip_in_progress = True
        try:
            bar = self._filmstrip_bar()
            if bar is None:
                return
            if getattr(self, "view_mode", "single") != "single":
                bar.hide()
                bar.setEnabled(False)
                self._hide_filmstrip_chrome()
                return
            if self._adjust_panel_active():
                # Editor is open: the filmstrip is deliberately disabled/hidden
                # (see _set_adjust_panel_visible). This function also runs from
                # background events (e.g. folder-index-ready) that fire while
                # editing, and used to unconditionally re-enable the bar here,
                # which let its hover-reveal logic show it again mid-edit.
                bar.hide()
                bar.setEnabled(False)
                return
            files = self._navigation_files()
            if self._is_semantic_search_filter_active() and files != list(
                getattr(self, "image_files", []) or []
            ):
                self.image_files = list(files)
                path = getattr(self, "current_file_path", None)
                if path and path in self.image_files:
                    self.current_file_index = self.image_files.index(path)
            enabled = len(files) > 1
            bar.setEnabled(enabled)
            if not enabled:
                bar.hide()
                self._hide_filmstrip_chrome()
                return
            bulk = getattr(self, "_gallery_bulk_metadata", None) or {}
            idx = getattr(self, "current_file_index", -1)
            bar.set_files(files, bulk_metadata=bulk, select_index=idx)
            self._refresh_filmstrip_bookmark_visuals()

            # Defer visible-cell refresh / prefetch until TTFR so rawpy.extract_thumb
            # does not compete with the main-thread QPixmap conversion for the GIL.
            def _refresh_filmstrip_after_paint() -> None:
                self._sync_filmstrip_index(center=True)
                self._prefetch_filmstrip_thumbnails()

            if getattr(self, "_single_view_first_render_logged", False):
                _refresh_filmstrip_after_paint()
            else:
                self._defer_until_first_paint(
                    _refresh_filmstrip_after_paint,
                    fallback_ms=_env_int("RAWVIEWER_FILMSTRIP_PREFETCH_DEFER_MS", 800, minimum=0),
                    label="filmstrip",
                )
        finally:
            self._sync_filmstrip_in_progress = False

    def _filmstrip_is_visible_enough(self) -> bool:
        """True when the bottom filmstrip is shown enough to keep selection in view."""
        if getattr(self, "view_mode", "single") != "single":
            return False
        bar = self._filmstrip_bar()
        container = getattr(self, "single_view_container", None)
        if bar is None or container is None or not bar.isEnabled():
            return False
        try:
            if not bar.isVisible():
                return False
            return float(container._filmstrip_opacity_value()) > 0.35
        except Exception:
            return False

    def _on_filmstrip_committed(self, index: int) -> None:
        files = self._navigation_files()
        if index < 0 or index >= len(files):
            return
        path = files[index]
        if _norm_path(path) == _norm_path(getattr(self, "current_file_path", "")):
            return

        self._capture_nav_zoom_state_before_load()
        self.current_file_index = index
        self._loading_from_filmstrip = True
        self.load_raw_image(path)

    def _note_filmstrip_thumbnail_warmed(self, path: str) -> None:
        if path:
            self._filmstrip_warmed_paths.add(path)

    def _filmstrip_cached_thumbnail(self, path: str):
        """Grid tier (~512px) is the film strip's authoritative source now that it
        requests the same tier as gallery tiles; the 256px thumbnail tier is only a
        fallback for files that haven't been decoded at grid resolution yet."""
        cached = self.image_cache.get_grid(path)
        if cached is not None:
            return cached
        return self.image_cache.get_thumbnail(path)

    def _collect_filmstrip_warm_paths(self):
        paths = set(getattr(self, "_filmstrip_warmed_paths", None) or ())
        bar = self._filmstrip_bar()
        if bar is not None and hasattr(bar, "paths_with_displayed_thumbnails"):
            paths.update(bar.paths_with_displayed_thumbnails())
        return list(paths)

    def _warm_gallery_from_filmstrip_cache(self) -> None:
        """Reuse film-strip / global thumbnails when entering gallery view."""
        gj = getattr(self, "gallery_justified", None)
        if gj is None or getattr(self, "view_mode", "") != "gallery":
            return
        paths = self._collect_filmstrip_warm_paths()
        if not paths:
            return
        warmed = gj.warm_thumbnails_from_global_cache(paths)
        if warmed > 0 and hasattr(gj, "_request_load_visible_images"):
            gj._request_load_visible_images(0)

    def _on_filmstrip_thumbnails_needed(self, paths) -> None:
        if getattr(self, "view_mode", "single") != "single":
            return

        # Defer filmstrip thumbnail requests from the UI during fast-open
        # until the first image is actually on screen.
        if not getattr(self, "_single_view_first_render_logged", False):
            if not hasattr(self, "_deferred_filmstrip_paths"):
                self._deferred_filmstrip_paths = set()
            self._deferred_filmstrip_paths.update(paths)
            return

        bar = self._filmstrip_bar()
        to_load = []
        for path in paths:
            if bar is not None:
                cached = self._filmstrip_cached_thumbnail(path)
                if cached is not None:
                    bar.apply_thumbnail(path, cached)
                    self._note_filmstrip_thumbnail_warmed(path)
                    continue
            to_load.append(path)
        # Enqueue uncached thumbnails in small chunks so a large batch (e.g. filmstrip
        # reveal) cannot flood the worker pool / main-thread signal queue at once.
        self._enqueue_filmstrip_thumbnails_chunked(to_load)

    def _flush_deferred_filmstrip_chunked(self) -> None:
        if not (hasattr(self, "_deferred_filmstrip_paths") and self._deferred_filmstrip_paths):
            return
        # Back off while the user is navigating; retry shortly.
        if self._in_nav_quiet_window():
            QTimer.singleShot(200, self._flush_deferred_filmstrip_chunked)
            return
        chunk = _env_int("RAWVIEWER_FILMSTRIP_FLUSH_CHUNK", 4, minimum=1)
        interval_ms = _env_int("RAWVIEWER_FILMSTRIP_FLUSH_INTERVAL_MS", 180, minimum=0)
        batch = []
        for _ in range(chunk):
            if not self._deferred_filmstrip_paths:
                break
            batch.append(self._deferred_filmstrip_paths.pop())
        if batch:
            self._on_filmstrip_thumbnails_needed(batch)
        if self._deferred_filmstrip_paths:
            QTimer.singleShot(interval_ms, self._flush_deferred_filmstrip_chunked)

    def _enqueue_filmstrip_thumbnails_chunked(self, paths) -> None:
        """Queue filmstrip thumbnail loads to be released a few at a time.

        Bulk-enqueueing the whole prefetch radius (e.g. 57 RAWs) saturated the worker
        pool and main-thread signal queue on external drives, stalling the first
        navigation for seconds. We drip-feed instead and pause during navigation.
        """
        if not paths:
            return
        if not hasattr(self, "_filmstrip_prefetch_pending") or self._filmstrip_prefetch_pending is None:
            self._filmstrip_prefetch_pending = []
        existing = set(self._filmstrip_prefetch_pending)
        for p in paths:
            if p not in existing:
                self._filmstrip_prefetch_pending.append(p)
        if not getattr(self, "_filmstrip_prefetch_draining", False):
            self._filmstrip_prefetch_draining = True
            self._drain_filmstrip_prefetch_chunk()

    def _display_low_res_thumbnail_fallback(
        self, file_path: str, thumbnail, max_dim: int
    ) -> None:
        """Paint a small embedded thumb once so TTFR does not stall on upgrade loops."""
        if getattr(self, "view_mode", "single") != "single":
            return
        if self._full_res_only_mode():
            return
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            return
        if self._single_view_pixels_on_screen(file_path):
            return
        import logging

        logger = logging.getLogger(__name__)
        try:
            self._orientation_already_applied = True
            if isinstance(thumbnail, QImage):
                thumb_rgb = thumbnail.convertToFormat(QImage.Format.Format_RGB888)
                self.display_pixmap(QPixmap.fromImage(thumb_rgb))
            else:
                self.display_numpy_image(thumbnail)
            self._on_single_view_content_displayed()
            self._manager_displayed_max_dim = max(
                getattr(self, "_manager_displayed_max_dim", 0), max_dim
            )
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.hide_loading()
            self.status_bar.showMessage(
                f"Low-res preview ({max_dim}px) -- loading higher quality…"
            )
            logger.info(
                "[MANAGER] Showing interim low-res preview (%dpx) for %s",
                max_dim,
                os.path.basename(file_path),
            )
        except Exception as exc:
            logger.debug(
                "[MANAGER] Interim low-res preview failed for %s: %s",
                os.path.basename(file_path),
                exc,
            )
        finally:
            self._orientation_already_applied = False

    def on_manager_thumbnail_ready(self, file_path: str, thumbnail):
        # Re-entrancy guard: displaying a thumbnail can trigger nested manager
        # callbacks (cache hits, preloading) that re-enter this slot synchronously
        # and overflow the C stack. Defer re-entrant calls instead of dropping.
        if getattr(self, "_on_manager_thumbnail_ready_in_progress", False):
            try:
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(
                    0,
                    lambda fp=file_path, th=thumbnail: self.on_manager_thumbnail_ready(
                        fp, th
                    ),
                )
            except Exception:
                pass
            return
        self._on_manager_thumbnail_ready_in_progress = True
        try:
            return self._on_manager_thumbnail_ready_impl(file_path, thumbnail)
        finally:
            self._on_manager_thumbnail_ready_in_progress = False

    def _on_manager_thumbnail_ready_impl(self, file_path: str, thumbnail):
        """處理 ImageLoadManager 的縮圖就緒信號"""
        import logging
        logger = logging.getLogger(__name__)
        from PyQt6.QtGui import QImage, QPixmap

        if getattr(self, "view_mode", "") == "compare":
            self._compare_upgrade_view_image(file_path, thumbnail)
            return

        # In gallery mode, thumbnail rendering is handled by rawviewer_ui.widgets.gallery.
        # Ignore single-view manager callbacks to prevent cross-mode repaint churn.
        if getattr(self, "_suppress_single_manager_callbacks", False):
            return
        if self._adjust_panel_blocks_display_refresh():
            return
        if getattr(self, "view_mode", "single") == "single":
            bar = self._filmstrip_bar()
            if bar is not None:
                if bar.wants_thumbnail(file_path):
                    bar.apply_thumbnail(file_path, thumbnail)
                    self._note_filmstrip_thumbnail_warmed(file_path)
                elif hasattr(bar, "note_cached_thumbnail"):
                    bar.note_cached_thumbnail(file_path, thumbnail)
        if getattr(self, "view_mode", "single") != "single":
            return

        # Only handle current file's thumbnail (normalize for Windows path format differences)
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            logger.debug(f"[MANAGER] Thumbnail for different file: {os.path.basename(file_path)}")
            return

        norm_fp = _norm_path(file_path)
        raw_recovery = (
            getattr(self, "_raw_recovery_active", False) and is_raw_file(file_path)
        )
        if raw_recovery:
            cached = self._raw_recovery_cache.get(norm_fp)
            if cached is not None:
                self._orientation_already_applied = True
                try:
                    self._display_raw_recovery_buffer(file_path, cached)
                finally:
                    self._orientation_already_applied = False
                self._full_resolution_loading = False
                self._finish_pending_zoom_after_full_load(file_path)
                self._on_single_view_content_displayed()
                self._skip_resolution_downgrade_check = False
                return
            else:
                if self._full_res_only_mode():
                    with contextlib.suppress(Exception):
                        self._request_full_res_decode(file_path, priority_current=True, force=True)
                    return
                if isinstance(thumbnail, QImage):
                    self.display_pixmap(QPixmap.fromImage(thumbnail))
                else:
                    self.display_numpy_image(thumbnail)
                self._start_raw_recovery(file_path)
                self._sync_raw_recovery_view_mode()
                self._orientation_already_applied = False
                self._full_resolution_loading = False
                self._finish_pending_zoom_after_full_load(file_path)
                return

        # Speed gate: avoid rendering large thumbnails (double work) when a proper preview is imminent.
        # Also protect UI from accidental oversized "thumbnails" (e.g. RAW BITMAP thumbs).
        try:
            if isinstance(thumbnail, QImage):
                w, h = thumbnail.width(), thumbnail.height()
            elif thumbnail is not None:
                h, w = thumbnail.shape[:2]
            else:
                h, w = 0, 0
            max_dim = max(h, w)
        except Exception:
            max_dim = 0

        # Late thumbnail_ready after preview/full was already shown causes deep re-entrant
        # display + status updates and can trigger RecursionError inside logging.
        # Only skip when THIS file's pixels are actually on screen -- otherwise a
        # leaked high-water mark from the previous file would drop the first
        # paint and leave navigation stuck on the prior image.
        shown_max = getattr(self, "_manager_displayed_max_dim", 0)
        if (
            not getattr(self, "_skip_resolution_downgrade_check", False)
            and max_dim > 0
            and shown_max >= max_dim
            and self._single_view_pixels_on_screen(file_path)
        ):
            logger.debug(
                f"[MANAGER] Skipping stale thumbnail ({max_dim}px max); "
                f"already displayed {shown_max}px for this file"
            )
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.hide_loading()
            return

        min_display = _env_int("RAWVIEWER_GALLERY_INSTANT_MIN_DIM", 1400, minimum=400)
        if 0 < max_dim < min_display:
            norm_fp = _norm_path(file_path)
            upgrade_attempted = norm_fp in getattr(
                self, "_preview_upgrade_attempted_paths", set()
            )
            try:
                from processing.unified import UnifiedImageProcessor

                up = UnifiedImageProcessor()
                upgraded = up.ensure_display_tier_preview(file_path, cast(Any, thumbnail))
                udim = up._preview_buffer_max_dim(upgraded)
                if upgraded is not None and udim >= min_display:
                    logger.info(
                        "[MANAGER] Upgraded low-res thumbnail to %dpx for %s",
                        udim,
                        os.path.basename(file_path),
                    )
                    thumbnail = upgraded
                    max_dim = udim
                elif upgrade_attempted:
                    logger.debug(
                        "[MANAGER] Low-res thumbnail (%dpx) for %s; upgrade already "
                        "queued -- waiting for preview/full",
                        max_dim,
                        os.path.basename(file_path),
                    )
                    self._pending_thumbnail = thumbnail
                    if not self._single_view_pixels_on_screen(file_path):
                        self._display_low_res_thumbnail_fallback(
                            file_path, thumbnail, max_dim
                        )
                    return
                else:
                    self._preview_upgrade_attempted_paths.add(norm_fp)
                    logger.info(
                        "[MANAGER] Skipping low-res thumbnail (%dpx) for %s; "
                        "requesting preview/full decode",
                        max_dim,
                        os.path.basename(file_path),
                    )
                    self._pending_thumbnail = thumbnail
                    from image_load_manager import Priority as _Prio

                    self.image_manager.load_image(
                        file_path,
                        priority=_Prio.CURRENT,
                        cancel_existing=True,
                        use_full_resolution=False,
                        stages={"thumbnail", "exif", "full"},
                    )
                    return
            except Exception:
                if upgrade_attempted:
                    logger.debug(
                        "[MANAGER] Low-res thumbnail (%dpx) for %s; waiting for "
                        "preview/full (upgrade retry already issued)",
                        max_dim,
                        os.path.basename(file_path),
                    )
                    self._pending_thumbnail = thumbnail
                    if not self._single_view_pixels_on_screen(file_path):
                        self._display_low_res_thumbnail_fallback(
                            file_path, thumbnail, max_dim
                        )
                    return
                self._preview_upgrade_attempted_paths.add(norm_fp)
                logger.info(
                    "[MANAGER] Skipping low-res thumbnail (%dpx) for %s; waiting for "
                    "embedded preview/full",
                    max_dim,
                    os.path.basename(file_path),
                )
                self._pending_thumbnail = thumbnail
                from image_load_manager import Priority as _Prio

                with contextlib.suppress(Exception):
                    self.image_manager.load_image(
                        file_path,
                        priority=_Prio.CURRENT,
                        cancel_existing=True,
                        use_full_resolution=False,
                        stages={"thumbnail", "exif", "full"},
                    )
                return

        # If thumbnail is already very large (near full embedded JPEG), skip displaying it and wait for image_ready.
        # Threshold was 1600px which skipped typical 1920px camera embeds -- users saw no preview for seconds.
        # 3840: show normal ~2K/3K embeds immediately; only skip huge thumbs to limit double-render cost.
        if max_dim >= 3840:
            try:
                logger.info(f"[MANAGER] Skipping thumbnail display (size {w}x{h}) to avoid double-render; waiting for preview/full.")
            except Exception:
                logger.info(f"[MANAGER] Skipping thumbnail display (max_dim={max_dim}) to avoid double-render; waiting for preview/full.")
            self._pending_thumbnail = None
            self.status_bar.showMessage("Processing image...")
            return

        # Mark that orientation is already applied (UnifiedImageProcessor applies it to thumbnails)
        # Support both np.ndarray and QImage thumbnails.
        self._orientation_already_applied = True
        try:
            if self._should_show_thumbnail() and not self._full_res_only_mode():
                if self._display_quality_buffer_cached(file_path):
                    if not self._single_view_pixels_on_screen(file_path):
                        if self._try_paint_display_quality_cache(file_path):
                            self._on_single_view_content_displayed()
                            if hasattr(self, "loading_overlay"):
                                self.loading_overlay.hide_loading()
                            return
                        # Tier mismatch: _display_quality_buffer_cached() can report True
                        # from a cache tier (e.g. "thumbnail") that
                        # _try_paint_display_quality_cache() never attempts to paint (it
                        # only checks the "preview" tier). Nothing is on screen yet, so
                        # don't dead-end here -- fall through and paint this thumbnail
                        # directly instead of leaving the view blank.
                    else:
                        logger.debug(
                            f"[MANAGER] Skipping small thumbnail; display-quality buffer cached for "
                            f"{os.path.basename(file_path)}"
                        )
                        return

                # Mid-size thumbnail (below the 3840px hard-skip above) arriving
                # mid-navigation, with a smaller preview already on screen: the
                # navigation that got us here already requested the 'full'
                # decode, which routinely lands well under a second later (see
                # nav_to_display/AUTOTEST perf data) and will immediately
                # supersede this paint. Measured via a 260-image navigation
                # stress test: ~30% of navigations painted this extra tier
                # (preview -> this thumbnail -> full, a visible double-flash)
                # before landing on full-res. Skipping here when something is
                # already visible avoids that extra flash without touching the
                # cold-open path (nothing on screen yet), which still needs
                # this thumbnail to avoid a multi-second blank view.
                if getattr(self, "_navigation_in_progress", False) and self._single_view_pixels_on_screen(
                    file_path
                ):
                    logger.debug(
                        "[MANAGER] Navigation in progress with content already on screen; "
                        "skipping intermediate %dpx thumbnail repaint for %s (full decode "
                        "already requested by navigation)",
                        max_dim,
                        os.path.basename(file_path),
                    )
                    self._pending_thumbnail = thumbnail
                    # Deliberately do NOT advance _manager_displayed_max_dim
                    # here -- this thumbnail was never painted. That field is
                    # the per-file "resolution actually shown on screen"
                    # high-water-mark (see its other call sites), used to drop
                    # late/duplicate deliveries; inflating it for a size that
                    # was skipped would make a real subsequent delivery at or
                    # below this size look "already displayed" and get
                    # dropped too -- if the full decode this skip is banking
                    # on then fails/cancels/stalls, nothing else repaints and
                    # the view sticks on the smaller size that IS on screen.
                    if hasattr(self, "loading_overlay"):
                        self.loading_overlay.hide_loading()
                    return

                # Only hide overlay if we're actually showing the thumbnail
                if hasattr(self, 'loading_overlay'):
                    self.loading_overlay.hide_loading()

                # Full-res-only testing mode: never paint interim thumbnails/previews.
                if self._full_res_only_mode():
                    self._pending_thumbnail = thumbnail
                    with contextlib.suppress(Exception):
                        self._request_full_res_decode(file_path, priority_current=True, force=True)
                    return

                if not getattr(self, "_navigation_in_progress", False):
                    self._mark_pending_resolution_crossfade()
                logger.info(
                    "[MANAGER] Thumbnail ready for %s (%dpx)",
                    os.path.basename(file_path),
                    max_dim,
                )
                if isinstance(thumbnail, QImage):
                    pixmap = QPixmap.fromImage(thumbnail)
                    self.display_pixmap(pixmap)
                else:
                    self.display_numpy_image(thumbnail)

                self._on_single_view_content_displayed()
                self._skip_resolution_downgrade_check = False

                self._manager_displayed_max_dim = max(
                    getattr(self, "_manager_displayed_max_dim", 0), max_dim
                )
                self.status_bar.showMessage("Preview loaded - processing full image...")
            else:
                # Keep overlay visible if we're storing thumbnail as pending (waiting for full image)
                self._pending_thumbnail = thumbnail
                self.status_bar.showMessage("Processing full image for quality evaluation...")
                if self._suppress_jpeg_interim(file_path) and _norm_path(
                    file_path
                ) == _norm_path(getattr(self, "current_file_path", "") or ""):
                    # No interim will ever be painted for this file; without an
                    # explicit request some loads end at the thumbnail stage and
                    # the viewport stays empty until the 5s watchdog fires.
                    try:
                        from image_load_manager import Priority as _Priority

                        # Fit view: half-size LibRaw decode (~0.2-0.6s), not
                        # the sensor-res decode (1.5-4s) -- same tier the
                        # LibRaw-first pipeline uses for fit display.
                        self.image_manager.load_image(
                            file_path,
                            priority=_Priority.CURRENT,
                            cancel_existing=False,
                            use_full_resolution=False,
                            stages={"full"},
                        )
                    except Exception:
                        pass
        finally:
            self._orientation_already_applied = False  # Reset flag

    def _arm_gallery_warmup(self) -> None:
        """Start gallery tile warmup once the justified gallery widget exists."""
        file_count = len(getattr(self, "image_files", []) or [])
        gj = getattr(self, "gallery_justified", None)
        if gj is not None and hasattr(gj, "begin_gallery_warmup"):
            gj.begin_gallery_warmup(file_count)

    def _update_gallery_counter(self):
        if getattr(self, "view_mode", "single") != "gallery":
            return
        self._refresh_image_counter()
