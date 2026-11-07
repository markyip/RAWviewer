
"""misc mixin, part 1 of 4 (31 methods, 827 lines)."""
from __future__ import annotations

import contextlib
import os
import sys
import threading

from PyQt6.QtCore import QPoint, QTimer

from core.config import _env_true, _norm_path

from .. import lazy
from ..lazy import is_raw_file
from ..paths import _require_str_path
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class MiscPart1:
    def _on_gpu_double_clicked(self, scene_pt) -> None:
        """Double-click: zoom to click point from fit, or zoom out when already zoomed."""
        import logging

        logger = logging.getLogger(__name__)
        if self._raw_recovery_preview_active():
            gv = getattr(self, "gpu_view", None)
            if gv is not None and not gv.is_fit_mode():
                gv.fit_to_window()
                self.fit_to_window = True
                self.zoom_center_point = None
                self.update_status_bar()
            else:
                self._notify_raw_recovery_zoom_blocked()
            return
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        if not gv.has_pixmap():
            if getattr(self, "current_file_path", None):
                self._pending_zoom_toggle = True
                self._pending_zoom_target_path = self.current_file_path
                self._pending_gpu_zoom_point = scene_pt
                logger.info(
                    "GPU double-click recorded while image is loading; will zoom once ready."
                )
            return
        gv._sync_fit_mode_flag()
        self._stop_slideshow()
        logger.info(
            "[TRACK] User double-clicked to zoom (GPU) - file: %s",
            os.path.basename(_require_str_path(self.current_file_path))
            if getattr(self, "current_file_path", None)
            else "Unknown",
        )
        if (
            getattr(self, "_focus_subject_outline_active", False)
            and not gv.is_fit_mode()
        ):
            gv.fit_to_window()
            self.fit_to_window = True
            self.zoom_center_point = None
            self._clear_all_zoom_state()
            self._nav_zoom_interim_active = False
        elif gv.wants_zoom_in_toggle():
            self._recenter_on_focus_after_nav = False
            self.zoom_center_point = QPoint(int(scene_pt.x()), int(scene_pt.y()))
            self._allow_full_decode_for_user_zoom()
            # Upgrade from cache before zoom when full is already in memory but
            # the on-screen pixmap is still the preview tier.
            self._try_sync_full_before_zoom()
            # Adjust: restore half-res settle if Fit lite (640) is currently painted.
            self._restore_adjust_hires_before_zoom()
            self._gpu_zoom_in_to_point_finish()
            # Adjust owns the half-res edit buffer -- never queue browse-path
            # sensor decode here (it fought the editor and left zoom on a soft
            # stand-in). Space-key zoom already had this gate; double-click did not.
            if (
                not self._adjust_panel_active()
                and self._needs_full_resolution_upgrade()
            ):
                self._pending_zoom = True
                self._pending_zoom_center = self.zoom_center_point
                self._pending_zoom_thumbnail_size = self.current_pixmap.size() if self.current_pixmap else None
                self._request_full_res_decode(
                    _require_str_path(self.current_file_path),
                    priority_current=True,
                    urgent=True,
                )
            elif self._adjust_panel_active():
                # Prefer the half-res settle buffer over a live-drag 640px tier
                # when the user zooms in mid-edit.
                self._apply_adjust_panel_preview(full_quality=True)
        else:
            gv.fit_to_window()
            self.fit_to_window = True
            self.zoom_center_point = None
            self._clear_all_zoom_state()
            self._nav_zoom_interim_active = False
        self.update_status_bar()
        self.setFocus()

    def _sync_raw_recovery_view_mode(self) -> None:
        active = self._raw_recovery_preview_active()
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv.set_zoom_locked(active)
            if active:
                self._clear_all_zoom_state()
                gv._zoom_intent_100 = False
                if gv.has_pixmap():
                    gv.fit_to_window()
                self.fit_to_window = True
                self.zoom_center_point = None
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "set_recovery_preview_badge"):
            detail = ""
            pm = getattr(self, "current_pixmap", None)
            if pm is not None and not pm.isNull():
                detail = f" ({pm.width()}×{pm.height()})"

            # Check if currently loading recovery preview
            current_path = getattr(self, "current_file_path", None)
            is_loading = False
            if current_path:
                norm_current = _norm_path(current_path)
                worker_path = getattr(self, "_raw_recovery_worker_path", None)
                if worker_path == norm_current:
                    is_loading = True

            container.set_recovery_preview_badge(active, detail=detail, loading=is_loading)
        self._last_status_signature = None
        self.update_status_bar()
        if active:
            self.status_bar.showMessage(
                "Recovery preview ON -- adjusted image, fit-only (not full resolution). Press P to exit.",
                6000,
            )

    def _start_raw_recovery(self, file_path: str) -> None:
        if not file_path or not is_raw_file(file_path):
            return
        if not getattr(self, "_raw_recovery_active", False):
            return
        norm = _norm_path(file_path)
        if norm in self._raw_recovery_cache:
            return
        if getattr(self, "_raw_recovery_worker_path", None) == norm:
            return
        self._raw_recovery_worker_path = norm
        exif_data = {}
        with contextlib.suppress(Exception):
            exif_data = self.image_cache.get_exif(file_path) or {}
        from rawviewer_app.workers import RawRecoveryWorker

        _get_bg_thread_pool().start(
            RawRecoveryWorker(file_path, self._raw_recovery_signals, exif_data=exif_data)
        )

    def _disable_raw_recovery(self) -> None:
        was_active = self._raw_recovery_preview_active()
        self._raw_recovery_active = False
        self._raw_recovery_worker_path = None
        if was_active:
            self._sync_raw_recovery_view_mode()

    def _toggle_raw_recovery(self) -> bool:
        if getattr(self, "view_mode", "single") != "single":
            return False
        path = getattr(self, "current_file_path", None)
        if path and not is_raw_file(path):
            self.status_bar.showMessage("RAW recovery is only available for RAW/DNG", 2500)
            return True
        if self._adjust_panel_active():
            self.status_bar.showMessage(
                "Adjust panel is open -- use «Use recovery look» for recovery tone",
                5000,
            )
            return True
        self._raw_recovery_active = not getattr(self, "_raw_recovery_active", False)
        if self._raw_recovery_active:
            import logging

            self._touch_single_view_interactivity()
            logging.getLogger(__name__).info(
                "RAW recovery ON for %s",
                os.path.basename(path or ""),
            )
            self._cancel_display_pixmap_converter()
            self.status_bar.showMessage(
                "Decoding RAW for local shadow/highlight recovery…",
                4000,
            )
            if path:
                cached = self._raw_recovery_cache.get(_norm_path(path))
                if cached is not None:
                    self._orientation_already_applied = True
                    try:
                        self._display_raw_recovery_buffer(path, cached)
                    finally:
                        self._orientation_already_applied = False
                    self._sync_raw_recovery_view_mode()
                else:
                    self._start_raw_recovery(path)
                    self._sync_raw_recovery_view_mode()
        else:
            import logging

            logging.getLogger(__name__).info(
                "RAW recovery OFF for %s",
                os.path.basename(path or ""),
            )
            self.status_bar.showMessage("RAW recovery preview off", 2000)
            self._refresh_neutral_raw_display()
            self._sync_raw_recovery_view_mode()
        return True

    def _toggle_composition_grid(self) -> bool:
        """Cycle composition guides: off → 3×3 → diagonal → 3×3+diagonal → phi grid."""
        vm = getattr(self, "view_mode", "single")
        if vm not in ("single", "compare"):
            return False
        from rawviewer_ui.overlays.composition_grid import grid_mode_label, next_grid_mode

        current = getattr(self, "_composition_grid_mode", "off")
        new_mode = next_grid_mode(current)
        self._composition_grid_mode = new_mode
        with contextlib.suppress(Exception):
            self.get_settings().setValue("composition_grid_mode", new_mode)
        self.status_bar.showMessage(f"Composition guide: {grid_mode_label(new_mode)}", 2000)
        self._sync_composition_grid_display()
        return True

    def _gpu_update_composition_grid(self) -> None:
        """Reflect the current composition guide onto the GPU scene overlay."""
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        gv.set_composition_grid_mode(getattr(self, "_composition_grid_mode", "off"))

    def _burst_visibility_exclusion_set(self) -> set[str]:
        """Burst-rejected paths, plus anything the active rating/bookmark
        gallery filter hides right now.

        Stack tiles/counts must reflect what's actually visible under the
        current filter, not the full original burst -- otherwise a single
        filtered-in photo from an 11-shot burst shows a misleading "x11"
        badge for a stack where the other 10 aren't even on screen.
        """
        excluded = set(getattr(self, "_burst_rejected_paths", set()) or set())
        bookmark_active = bool(getattr(self, "_gallery_bookmark_filter_active", False))
        min_rating = int(getattr(self, "_gallery_rating_filter_min", 0) or 0)
        if not bookmark_active and min_rating <= 0:
            return excluded
        files = list(getattr(self, "image_files", []) or [])
        visible_files = files
        if bookmark_active:
            visible_files = self._filter_paths_to_bookmarked(visible_files)
        visible_files = self._filter_paths_to_min_rating(visible_files, min_rating)
        visible = {_norm_path(p) for p in visible_files}
        for p in files:
            np = _norm_path(p)
            if np not in visible:
                excluded.add(np)
        return excluded

    def _is_burst_grouping_enabled(self) -> bool:
        return bool(getattr(self, "_burst_grouping_enabled", False))

    def _sync_burst_grouping_button_visibility(self) -> None:
        btn = getattr(self, "burst_grouping_button", None)
        if btn is None:
            return
        vm = getattr(self, "view_mode", "single")
        burst_view = bool(getattr(self, "_burst_group_view_active", False))
        show = (
            vm == "gallery"
            and not burst_view
            and bool(getattr(self, "image_files", None))
        )
        btn.setVisible(show)
        if show:
            self._sync_burst_grouping_button_icon()

    def _on_burst_grouping_toggled(self, checked: bool) -> None:
        self._burst_grouping_enabled = bool(checked)
        try:
            from core.burst_grouping import BURST_GROUPING_SETTINGS_KEY

            self.get_settings().setValue(BURST_GROUPING_SETTINGS_KEY, checked)
        except Exception:
            pass
        self._sync_burst_grouping_button_icon()
        if not checked and getattr(self, "_burst_group_view_active", False):
            self._exit_burst_group_view()
        else:
            self._update_gallery_view()
        self.status_bar.showMessage(
            "Burst grouping on -- rapid captures stack into one tile"
            if checked
            else "Burst grouping off -- every image shown separately",
            2500,
        )

    def _save_burst_rejected_paths(self) -> None:
        import json

        folder = getattr(self, "current_folder", None)
        if not folder:
            return
        try:
            payload = sorted(getattr(self, "_burst_rejected_paths", set()) or set())
            self.get_settings().setValue(
                self._burst_rejected_settings_key(folder),
                json.dumps(payload),
            )
        except Exception:
            pass

    def _rebuild_burst_groups(self) -> None:
        if not self._is_burst_grouping_enabled():
            self._burst_groups = []
            self._burst_path_to_group = {}
            self._burst_group_covers = {}
            return

        from core.burst_grouping import burst_gap_seconds_from_settings, burst_groups_for_folder

        files = list(getattr(self, "image_files", []) or [])
        if not files:
            self._burst_groups = []
            self._burst_path_to_group = {}
            self._burst_group_covers = {}
            return
        folder = getattr(self, "current_folder", None)
        capture_times = {}
        try:
            if folder:
                capture_times = self.image_cache.get_capture_times_for_folder_sort(folder)
            if not capture_times:
                capture_times = self._capture_times_for_paths(files[:1024])
        except Exception:
            capture_times = {}
        gap = burst_gap_seconds_from_settings(self.get_settings())
        rejected = set(getattr(self, "_burst_rejected_paths", set()) or set())
        groups, path_to_group, group_covers = burst_groups_for_folder(
            files,
            capture_times,
            gap_sec=gap,
            rejected=rejected,
        )
        self._burst_groups = groups
        self._burst_path_to_group = path_to_group
        self._burst_group_covers = group_covers
        stack_covers = dict(getattr(self, "_burst_stack_covers", {}) or {})
        for group_id, default_cover in group_covers.items():
            cover = stack_covers.get(group_id)
            if not cover or _norm_path(cover) in rejected:
                stack_covers[group_id] = default_cover
        self._burst_stack_covers = stack_covers

    def _burst_stack_count_for_path(self, file_path: str) -> int:
        from core.burst_grouping import stack_count_for_cover

        if not self._is_burst_grouping_enabled():
            return 0
        if getattr(self, "_burst_group_view_active", False):
            return 0
        rejected = self._burst_visibility_exclusion_set()
        return stack_count_for_cover(
            file_path,
            getattr(self, "_burst_path_to_group", {}) or {},
            rejected,
        )

    def _burst_members_for_path(self, file_path: str) -> list[str] | None:
        if not self._is_burst_grouping_enabled():
            return None
        members = (getattr(self, "_burst_path_to_group", {}) or {}).get(
            _norm_path(file_path)
        )
        if not members:
            return None
        rejected = self._burst_visibility_exclusion_set()
        from core.burst_grouping import BURST_MIN_COUNT

        visible = [p for p in members if _norm_path(p) not in rejected]
        return visible if len(visible) >= BURST_MIN_COUNT else None

    def _burst_group_view_visible_members(self) -> list[str]:
        rejected = set(getattr(self, "_burst_rejected_paths", set()) or set())
        members = list(getattr(self, "_burst_group_view_members", []) or [])
        return [p for p in members if _norm_path(p) not in rejected]

    def _enter_burst_group_view(self, members: list[str], *, cover: str | None = None) -> None:
        self._burst_group_view_active = True
        self._burst_group_view_members = list(members)
        self._clear_gallery_selection()
        if cover:
            group_id = _norm_path(members[0]) if members else ""
            if group_id:
                self._burst_stack_covers[group_id] = cover
        self._stop_slideshow()
        self._sync_burst_group_bottom_chrome()
        self._sync_compare_button_visibility()
        self._update_gallery_view()
        self.update_status_bar()

    def _exit_burst_group_view(self, *, scroll_to: str | None = None) -> None:
        self._burst_group_view_active = False
        self._burst_group_view_members = []
        self._stop_slideshow()
        self._sync_burst_group_bottom_chrome()
        self._sync_compare_button_visibility()
        self._update_gallery_view()
        self.update_status_bar()
        if scroll_to and getattr(self, "gallery_justified", None):
            with contextlib.suppress(Exception):
                self.gallery_justified.scroll_to_file(scroll_to)
        from PyQt6.QtCore import QTimer

        self._restore_keyboard_focus()
        QTimer.singleShot(0, self._restore_keyboard_focus)

    def _path_in_burst_group_view(self, file_path: str | None) -> bool:
        if not file_path or not getattr(self, "_burst_group_view_active", False):
            return False
        key = _norm_path(file_path)
        members = getattr(self, "_burst_group_view_members", []) or []
        return any(_norm_path(m) == key for m in members)

    def _maybe_leave_burst_group_view_for_path(self, file_path: str | None) -> None:
        """Drop burst-group drill-down once single-view navigation leaves that set."""
        if not getattr(self, "_burst_group_view_active", False):
            return
        if getattr(self, "_loading_from_gallery", False):
            return
        if self._path_in_burst_group_view(file_path):
            return
        self._burst_group_view_active = False
        self._burst_group_view_members = []
        self._sync_burst_group_bottom_chrome()
        self._sync_compare_button_visibility()

    def _delete_path_to_trash(self, file_to_delete: str) -> None:
        if not file_to_delete or not os.path.exists(file_to_delete):
            return
        filename = os.path.basename(file_to_delete)
        normalized_path = os.path.normpath(file_to_delete)

        self._cancel_load_and_preload_for_path(file_to_delete)
        from image_cache import get_image_cache
        cache = get_image_cache()
        cache.invalidate_file(file_to_delete)

        from send2trash import send2trash
        send2trash(normalized_path)

        self._drop_discarded_from_semantic_corpus(file_to_delete)
        self._remove_file_from_active_image_list(file_to_delete)
        self.status_bar.showMessage(f"Deleted: {filename}")

    def on_manager_image_ready(self, file_path: str, image):
        if getattr(self, "_on_manager_image_ready_in_progress", False):
            try:
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(
                    0,
                    lambda fp=file_path, im=image: self.on_manager_image_ready(fp, im),
                )
            except Exception:
                pass
            return
        self._on_manager_image_ready_in_progress = True
        try:
            return self._on_manager_image_ready_impl(file_path, image)
        finally:
            self._on_manager_image_ready_in_progress = False

    def on_manager_qimage_ready(self, file_path: str, image):
        """A worker produced a QImage because it could not legally build a
        QPixmap. This slot runs on the GUI thread, so convert here and hand it
        to the existing pixmap path -- including the pixmap cache, which the
        worker deliberately does not touch.
        """
        try:
            from PyQt6.QtGui import QPixmap as _QPixmap

            if image is None or image.isNull():
                return
            # Check staleness BEFORE converting. on_manager_pixmap_ready ends
            # in a path-match guard, so a buffer for a photo the user has
            # already left was converted and then dropped -- and the
            # conversion is 18 ms on the GUI thread for a 33 MP frame, plus a
            # full-size QPixmap allocated only to be freed. The cheap checks
            # are cheap; do them first.
            if getattr(self, "view_mode", "single") not in ("single", "compare"):
                return
            if _norm_path(file_path) != _norm_path(
                getattr(self, "current_file_path", None)
            ) and getattr(self, "view_mode", "single") != "compare":
                return
            pixmap = _QPixmap.fromImage(image)
            if pixmap.isNull():
                return
            with contextlib.suppress(Exception):
                self.image_cache.put_pixmap(file_path, pixmap)
            self.on_manager_pixmap_ready(file_path, pixmap)
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "[LOAD] QImage -> QPixmap failed for %s",
                os.path.basename(file_path or ""),
                exc_info=True,
            )

    def on_manager_error(self, file_path: str, error_message: str):
        """處理 ImageLoadManager 的錯誤信號"""
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()

        import logging
        logger = logging.getLogger(__name__)

        # Remember the failure for the prefetch failure-memo (see
        # _needs_display_buffer_prefetch) regardless of which file is current.
        try:
            import time as _time

            memo = getattr(self, "_prefetch_failure_ts", None)
            if memo is None:
                memo = self._prefetch_failure_ts = {}
            memo[os.path.normcase(os.path.abspath(file_path))] = _time.monotonic()
            if len(memo) > 256:  # bound it; entries are tiny
                oldest = sorted(memo, key=memo.get)[: len(memo) - 256]
                for k in oldest:
                    del memo[k]
        except Exception:
            pass

        # Reset the deferred-sensor-decode gate so a failed decode does not
        # permanently block future full-resolution upgrades for this file.
        try:
            if _norm_path(file_path) == _norm_path(getattr(self, "current_file_path", None)):
                self._sensor_full_decode_queued_norm = None
                self._defer_sensor_full_decode_path = None
        except Exception:
            pass

        # Only handle current file's error (normalize for Windows path format differences)
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            logger.debug(f"[MANAGER] Error for different file: {os.path.basename(file_path)}")
            return

        logger.error(f"[MANAGER] Error loading {os.path.basename(file_path)}: {error_message}")
        if "too many open files" in (error_message or "").lower():
            self._enter_io_pressure_mode("manager_error_emfile")
        if getattr(self, "_defer_sensor_full_decode_path", None):
            self._flush_deferred_sensor_full_decode(file_path)
        if self._retry_composite_dng_embedded_preview(file_path, error_message):
            return
        if self._retry_raw_preview_with_full_decode(file_path, error_message):
            return
        if (
            getattr(self, "view_mode", "single") == "single"
            and not self._single_view_pixels_on_screen(file_path)
        ):
            self._clear_stale_single_view_image()
            self.status_bar.showMessage(
                f"Could not load {os.path.basename(file_path)}: {error_message}",
                8000,
            )
        # Avoid modal-dialog storms from repeated async retries of the same failure.
        # Repeated QMessageBox.exec() blocks user input and can look like folder switching is broken.
        import time
        now = time.time()
        last_error_path = getattr(self, "_last_manager_error_path", None)
        last_t = getattr(self, "_last_manager_error_ts", 0.0)
        norm_p = _norm_path(file_path)
        if last_error_path == norm_p and (now - last_t) < 5.0:
            return
        self._last_manager_error_path = norm_p
        self._last_manager_error_ts = now

        lower_err = error_message.lower()
        from common_image_loader import is_raw_file, load_pixmap_safe

        clearly_unsupported = any(
            term in lower_err
            for term in (
                "unsupported file format",
                "not recognized",
                "libraw unsupported",
                "unsupported or corrupt raw",
                "no usable embedded jpeg",
            )
        )
        if not clearly_unsupported and not is_raw_file(file_path):
            try:
                recovered = load_pixmap_safe(file_path)
                if not recovered.isNull():
                    self.on_manager_pixmap_ready(file_path, recovered)
                    return
            except Exception:
                pass

        # A "clearly unsupported" failure while forced into full RAW-decode
        # mode is very often a LibRaw demosaic-support gap for this specific
        # file's compression (e.g. newer Nikon "High Efficiency" compressed
        # NEF from Z8/Z9/Z6III/Z50II-generation bodies), not a corrupt file --
        # the embedded JPEG preview for the same file typically still decodes
        # fine, since that doesn't need LibRaw to demosaic anything. Mark the
        # path unsupported (per-file) and reload so the embed paints; do not
        # leave RAW mode for the whole app.
        if clearly_unsupported and is_raw_file(file_path):
            if self._auto_revert_to_embedded_jpeg_workflow(
                file_path, reason="manager clearly_unsupported"
            ):
                with contextlib.suppress(Exception):
                    self._invalidate_workflow_pixel_caches([file_path])
                self.load_raw_image(file_path, force_reload=True)
                return
            # Already marked unsupported: a failed LibRaw/full attempt
            # must not modal-storm when a prior (or concurrent) embed paint
            # already put usable pixels on screen for this file.
            if self._single_view_pixels_on_screen(file_path):
                pm = getattr(self, "current_pixmap", None)
                if pm is not None and not pm.isNull() and max(int(pm.width()), int(pm.height())) >= 1400:
                        self.status_bar.showMessage(
                            f"{os.path.basename(file_path)}: keeping embedded preview "
                            f"(decoder cannot demosaic this RAW compression).",
                            5000,
                        )
                        return

        if clearly_unsupported:
            # This error is often just one failed decode *strategy* (e.g. LibRaw
            # cannot demosaic this file's RAW compression) with an embedded-JPEG
            # fallback already in flight for the same "full" request -- that
            # fallback routinely lands ~0.5-1s later and displays the image
            # fine (see processing.unified's preview-tier fallback chain).
            # Showing the modal immediately produced a spurious "unsupported or
            # corrupt" dialog for files that then rendered correctly moments
            # later. Give the fallback a grace period and re-check before
            # actually popping the modal.
            QTimer.singleShot(
                900, lambda fp=file_path: self._show_unsupported_file_dialog_if_still_failed(fp)
            )
        elif (
            is_raw_file(file_path)
            and any(
                marker in lower_err
                for marker in (
                    "thumbnail extraction returned none",
                    "display-tier preview extraction failed",
                )
            )
        ):
            # Preview-only failure in gallery (or after retry): status bar only, no modal.
            pass
        else:
            self.show_error("Load Error", f"Failed to load image: {error_message}")

        # Graceful handling for ejected volumes or missing files
        if not os.path.exists(file_path):
            parent_dir = os.path.dirname(file_path)
            if not os.path.exists(parent_dir):
                self.reset_to_initial_state()

    def _check_for_updates_on_launch(self) -> None:
        """Silent background release check once per app launch (opt-out: RAWVIEWER_SKIP_UPDATE_CHECK=1)."""
        if _env_true("RAWVIEWER_SKIP_UPDATE_CHECK"):
            return
        if getattr(self, "_release_update_launch_check_done", False):
            return
        self._release_update_launch_check_done = True
        self._run_release_update_check()

    def _mobileclip_models_missing(self) -> bool:
        from search import semantic_embeddings_enabled

        if not semantic_embeddings_enabled():
            return False
        if getattr(self, "_semantic_asset_download_in_progress", False):
            return False
        try:
            index = self._get_semantic_index()
            if index.semantic_backend_available():
                return False
            backend_error = index.semantic_backend_error()
            return "Missing MobileCLIP" in backend_error and index.mobileclip_supports_hub_download()
        except Exception:
            return False

    def _should_show_image_counter(self) -> bool:
        """Hide counter until EXIF capture-time sort is applied (index/total can change)."""
        return self._is_exif_sort_ready()

    def _refresh_image_counter(self) -> None:
        """Update or hide the status-bar image counter (gallery total or single index)."""
        if not hasattr(self, "status_counter_label") or self.status_counter_label is None:
            return
        if not self._should_show_image_counter():
            self.status_counter_label.hide()
            self.status_counter_label.setText("")
            return
        if getattr(self, "view_mode", "single") == "gallery":
            if getattr(self, "_gallery_bookmark_filter_active", False) or int(
                getattr(self, "_gallery_rating_filter_min", 0) or 0
            ) > 0:
                total = len(self._gallery_files_for_display())
                qualifier = self._gallery_scope_filter_qualifier()
                img_word = "image" if total == 1 else "images"
                self.status_counter_label.setText(f"{total} {img_word} ({qualifier})")
            elif self._is_semantic_search_filter_active():
                total = len(self.image_files) if self.image_files else 0
                self.status_counter_label.setText(
                    f"{total} results" if total != 1 else "1 result"
                )
            else:
                total = len(self.image_files) if self.image_files else 0
                self.status_counter_label.setText(
                    f"{total} images" if total != 1 else "1 image"
                )
            self.status_counter_label.show()
        elif self.image_files and self.current_file_index >= 0:
            total_files = len(self.image_files)
            current_pos = self.current_file_index + 1
            self.status_counter_label.setText(f"{current_pos} / {total_files}")
            self.status_counter_label.show()
        else:
            self.status_counter_label.hide()
            self.status_counter_label.setText("")

    def _reset_view_for_workflow_toggle(self) -> None:
        """Drop stale zoom anchors and fit before RAW↔JPEG reload."""
        self._clear_focus_zoom_anchor()
        self._pending_zoom = False
        self._pending_zoom_center = None
        self._pending_zoom_thumbnail_size = None
        self._pending_zoom_toggle = False
        if hasattr(self, "_pending_zoom_target_path"):
            with contextlib.suppress(AttributeError):
                delattr(self, "_pending_zoom_target_path")
        self.zoom_center_point = None
        self.fit_to_window = True
        if hasattr(self, "_maintain_zoom_on_navigation"):
            with contextlib.suppress(AttributeError):
                delattr(self, "_maintain_zoom_on_navigation")
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv._zoom_intent_100 = False
            if gv.has_pixmap():
                gv.fit_to_window()
        self._workflow_toggle_fit_until_display = True
        self._workflow_toggle_refit_timer_scheduled = False

    def toggle_raw_jpeg_workflow(self):
        """Toggle between embedded JPEG and raw image workflows."""
        if getattr(self, "view_mode", "single") != "single":
            return
        settings = self.get_settings()
        current_value = settings.value("use_embedded_jpeg_workflow", True, type=bool)
        new_value = not current_value
        settings.setValue("use_embedded_jpeg_workflow", new_value)
        try:
            from common_image_loader import invalidate_libraw_consistent_preview_settings

            invalidate_libraw_consistent_preview_settings()
        except Exception:
            pass
        entering_raw_mode = not new_value

        # Show immediate feedback
        if hasattr(self, "loading_overlay"):
            msg = "Switching to RAW..." if entering_raw_mode else "Switching to Preview..."
            self.loading_overlay.show_loading(msg)

        if entering_raw_mode:
            # RAW EDR re-decodes with LibRaw highlight reconstruction (~12x a
            # plain decode); default it off on entry so switching into RAW
            # mode doesn't silently slow every navigation down. The user can
            # still re-enable it per-session via the EDR button.
            settings.setValue("raw_edr_display_enabled", False)
            self._update_edr_toggle_button_state()
        self._invalidate_workflow_pixel_caches()
        self._skip_resolution_downgrade_check = True
        self._update_raw_toggle_button_state()
        if self.current_file_path:
            self._reset_view_for_workflow_toggle()
            with contextlib.suppress(Exception):
                self.image_manager.cancel_task(self.current_file_path)
            self._manager_displayed_max_dim = 0
            self._displayed_content_path = None
            self._gpu_last_path = None
            norm_cur = _norm_path(self.current_file_path)
            if hasattr(self, "_file_max_dim_map") and norm_cur in self._file_max_dim_map:
                del self._file_max_dim_map[norm_cur]
            self._preview_upgrade_attempted_paths.discard(norm_cur)
            self._raw_workflow_refused_tier_paths.discard(norm_cur)
            self._single_view_display_generation = int(
                getattr(self, "_single_view_display_generation", 0)
            ) + 1
            self._workflow_toggle_reload = True
            try:
                self.load_raw_image(self.current_file_path, force_reload=True)
            finally:
                self._workflow_toggle_reload = False

    def _invalidate_workflow_pixel_caches(self, paths=None) -> None:
        """Drop display-tier pixel caches that depend on RAW vs embedded workflow.

        Memory caches are removed synchronously (fast dict ops -- the very
        next paint must not see stale-workflow pixels). The two DISK caches
        are purged on a background thread: per-file removal is a file-system
        unlink/DB delete each, and doing thousands of them inline made the
        RAW/JPEG workflow toggle stall the UI for seconds on large folders.
        A momentarily stale disk entry is harmless in the gap: the memory
        tiers are already empty, so the next fetch decodes fresh pixels and
        overwrites the disk entry.
        """
        targets = list(paths or getattr(self, "image_files", []) or [])
        cur = getattr(self, "current_file_path", None)
        if cur and cur not in targets:
            targets.append(cur)
        cache = self.image_cache
        with contextlib.suppress(Exception):
            cache.libraw_preview_paths.clear()
        raw_targets = []
        for fp in targets:
            if not fp:
                continue
            try:
                if not is_raw_file(fp):
                    continue
            except Exception:
                continue
            raw_targets.append(fp)
            key = cache._path_key(fp)
            cache.thumbnail_cache.remove(key)
            cache.grid_cache.remove(key)
            cache.preview_cache.remove(key)
            cache.full_image_cache.remove(fp)
            cache.pixmap_cache.remove(fp)

        if raw_targets:
            def _purge_disk(paths_snapshot=tuple(raw_targets)):
                for fp in paths_snapshot:
                    try:
                        cache.disk_preview_cache.remove(fp)
                        cache.disk_thumbnail_cache.remove(fp)
                        cache.disk_grid_cache.remove(fp)
                    except Exception:
                        pass

            threading.Thread(target=_purge_disk, daemon=True).start()
