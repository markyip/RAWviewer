"""adjust mixin, part 2 of 5 (37 methods, 889 lines)."""
from __future__ import annotations

import contextlib
import os
import sys
import time

from PyQt6.QtCore import QTimer

from core.config import _env_int, _norm_path
from rawviewer_app.core.widgets import CustomConfirmDialog

from .. import lazy
from ..lazy import is_raw_file
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class AdjustPart2:
    def _compare_apply_focus_sync_mode(self) -> None:
        select_view = self.compare_select_view
        candidate_view = self.compare_candidate_view
        session = getattr(self, "_comparison_session", None)
        if (
            select_view is None
            or candidate_view is None
            or session is None
            or getattr(self, "_compare_sync_mode", "A") != "B"
        ):
            return
        u_l, v_l = self._compare_focus_point_for_path(session.select, select_view)
        u_r, v_r = self._compare_focus_point_for_path(session.candidate, candidate_view)
        self._compare_sync_guard = True
        try:
            select_view.zoom_to_actual_at(
                u_l * max(0, select_view._img_w - 1),
                v_l * max(0, select_view._img_h - 1),
            )
            candidate_view.zoom_to_actual_at(
                u_r * max(0, candidate_view._img_w - 1),
                v_r * max(0, candidate_view._img_h - 1),
            )
        finally:
            self._compare_sync_guard = False
        self._compare_sync_paused = False

    def _compare_normalized_viewport_center(self, view) -> tuple[float, float]:
        if view is None or not view.has_pixmap():
            return 0.5, 0.5
        try:
            center = view.mapToScene(view.viewport().rect().center())
            return (
                max(0.0, min(1.0, float(center.x()) / max(1, view._img_w))),
                max(0.0, min(1.0, float(center.y()) / max(1, view._img_h))),
            )
        except Exception:
            return 0.5, 0.5

    def _compare_apply_view_sync_from_source(self, source_view, target_view) -> None:
        """Mode A: match relative zoom and normalized viewport center (per-pane fit scale)."""
        if source_view is None or target_view is None:
            return
        if not source_view.has_pixmap() or not target_view.has_pixmap():
            return
        relative = self._compare_relative_zoom(source_view)
        u, v = self._compare_normalized_viewport_center(source_view)
        self._compare_sync_guard = True
        try:
            self._compare_set_relative_zoom(target_view, relative)
            if relative > target_view._FIT_SCALE_EPS:
                target_view.center_on_image_point(
                    u * max(0, target_view._img_w - 1),
                    v * max(0, target_view._img_h - 1),
                )
        finally:
            self._compare_sync_guard = False

    def _on_compare_zoom_changed(self, source_view) -> None:
        if getattr(self, "view_mode", "") != "compare":
            return
        if getattr(self, "_compare_sync_guard", False):
            return
        if getattr(self, "_compare_sync_mode", "A") == "B":
            # Mode B: independent pan/zoom after per-pane focus alignment.
            self._compare_sync_paused = True
            return
        if getattr(self, "_compare_sync_paused", False):
            return
        target = (
            self.compare_candidate_view
            if source_view is self.compare_select_view
            else self.compare_select_view
        )
        self._compare_apply_view_sync_from_source(source_view, target)

    def _on_compare_scroll_changed(self, source_view) -> None:
        if getattr(self, "view_mode", "") != "compare":
            return
        if getattr(self, "_compare_sync_guard", False):
            return
        if getattr(self, "_compare_sync_mode", "A") == "B":
            self._compare_sync_paused = True
            return
        if getattr(self, "_compare_sync_paused", False):
            return
        target = (
            self.compare_candidate_view
            if source_view is self.compare_select_view
            else self.compare_select_view
        )
        self._compare_apply_view_sync_from_source(source_view, target)

    def _compare_handle_promote(self) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None or session.complete:
            return
        select_path = session.promote_candidate()
        group_id = _norm_path(session.members[0]) if session.members else ""
        if group_id and select_path:
            self._burst_stack_covers[group_id] = select_path
        self._refresh_compare_panes(changed="both")

    def _compare_handle_reject_candidate(self) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None or session.complete:
            return
        path = session.candidate
        session.reject_candidate()
        self._reject_compare_path(path)
        self._refresh_compare_panes()

    def _compare_handle_reject_select(self) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None or session.complete:
            return
        path = session.select
        self._handle_compare_select_rejected_transition(path)
        self._reject_compare_path(path)
        self._refresh_compare_panes()

    def _handle_compare_select_rejected_transition(self, rejected_path: str) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None:
            return
        # Reject the select path
        session.rejected.add(rejected_path)

        # Move the current Candidate to the Selected slot
        prev_candidate = session.candidate
        session.select = prev_candidate

        # Retrieve the next candidate
        visible = session.visible_members()
        if len(visible) < 2:
            session.candidate = ""
            session.complete = True
        else:
            next_path = session._pop_queue()
            if next_path and next_path != session.select and next_path not in session.rejected:
                session.candidate = next_path
            else:
                others = [p for p in visible if p != session.select]
                session.candidate = others[0] if others else ""
            if not session.candidate:
                session.complete = True

    def _compare_handle_delete(self, shift_pressed: bool) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None or session.complete:
            return

        path_to_delete = session.select if shift_pressed else session.candidate
        if not path_to_delete:
            return

        # Confirm deletion (custom MD3 confirmation dialog)
        filename = os.path.basename(path_to_delete)
        dialog = CustomConfirmDialog(
            parent=self,
            title="Delete Image",
            message=f"Are you sure you want to move '{filename}' to the Recycle Bin/Trash?",
            confirm_label="Delete",
            cancel_label="Cancel",
        )
        if not dialog.exec():
            return

        try:
            self._delete_path_to_trash(path_to_delete)
            # Now trigger the session discard/reject transition logic on the deleted path
            if shift_pressed:
                self._handle_compare_select_rejected_transition(path_to_delete)
                self._reject_compare_path(path_to_delete)
                self._refresh_compare_panes()
            else:
                session.reject_candidate()
                self._reject_compare_path(path_to_delete)
                self._refresh_compare_panes()
        except Exception as e:
            self.show_error("Delete Error", f"Could not delete file:\n{str(e)}")

    def _gallery_preview_pixmap_for_path(self, file_path: str):
        """Return a gallery-cached thumbnail for instant single-view preview."""
        if not file_path:
            return None
        gj = getattr(self, "gallery_justified", None)
        if gj is None:
            return None
        base_key = getattr(gj, "_thumb_base_key", None)
        if base_key is None:
            return None
        try:
            pm = gj._thumbnail_cache.get((file_path, base_key))
            if pm is not None and not pm.isNull():
                return pm
        except Exception:
            pass
        return None

    def _schedule_idle_full_decode_after_nav_preview(self, file_path: str) -> None:
        """Queue sensor full only after the user pauses on a grid-tier (512px) nav preview."""
        from PyQt6.QtCore import QTimer

        if not file_path:
            return
        if self._preserve_zoom_navigation_active() or self._single_view_is_zoomed_in():
            self._request_full_res_decode(file_path, priority_current=True)
            return
        delay_ms = _env_int("RAWVIEWER_NAV_IDLE_FULL_MS", 1200, minimum=400)
        self._nav_idle_full_generation = int(
            getattr(self, "_nav_idle_full_generation", 0)
        ) + 1
        gen = self._nav_idle_full_generation
        norm = _norm_path(file_path)

        def _tick(g=gen, fp=file_path, n=norm) -> None:
            if g != int(getattr(self, "_nav_idle_full_generation", 0)):
                return
            if _norm_path(getattr(self, "current_file_path", "") or "") != n:
                return
            if not self._needs_full_resolution_upgrade(fp):
                return
            self._maybe_queue_background_full_decode(fp)

        QTimer.singleShot(delay_ms, _tick)

    def _paint_instant_preview_for_path(self, file_path: str, *, prefer_gallery: bool = False) -> bool:
        """Show the best available preview immediately (preview cache, gallery thumb, or pixmap)."""
        if not file_path:
            return False
        if self._suppress_jpeg_interim(file_path) and not self.image_cache.is_libraw_preview(
            file_path
        ):
            # RAW workflow: gallery thumbs / cached previews here are
            # embedded-JPEG pixels; wait for the LibRaw render instead.
            return False

        self._gallery_instant_display_quality = False
        min_quality = _env_int("RAWVIEWER_GALLERY_INSTANT_MIN_DIM", 1400, minimum=400)

        preview_arr = None
        if hasattr(self, "image_cache"):
            try:
                preview_arr = self.image_cache.get_preview(file_path)
            except Exception:
                preview_arr = None

        gal_pm = self._gallery_preview_pixmap_for_path(file_path) if prefer_gallery else None
        gal_max = self._gallery_thumb_max_dim(file_path) if gal_pm is not None else 0
        arr_max = 0
        if preview_arr is not None:
            try:
                arr_max = max(int(preview_arr.shape[0]), int(preview_arr.shape[1]))
            except Exception:
                arr_max = 0

        self._orientation_already_applied = True
        self._skip_resolution_crossfade_once = True

        if preview_arr is not None and arr_max >= min_quality:
            self._gallery_preview_pending_full = False
            self._clear_pending_resolution_crossfade()
            self._gallery_instant_display_quality = True
            self.display_numpy_image(preview_arr)
            return True

        if gal_pm is not None and not gal_pm.isNull() and gal_max >= min_quality and (arr_max <= 0 or gal_max >= int(arr_max * 0.95)):
                self._gallery_preview_pending_full = False
                self._clear_pending_resolution_crossfade()
                self._gallery_instant_display_quality = True
                self.display_pixmap(gal_pm)
                return True

        # Do not paint grid-tier disk preview (~512px); wait for embedded thumbnail/full.

        preview = gal_pm
        if preview is None and hasattr(self, "image_cache"):
            try:
                preview = self.image_cache.get_pixmap(file_path)
            except Exception:
                preview = None
        if preview is None and not prefer_gallery:
            preview = self._gallery_preview_pixmap_for_path(file_path)
        if preview is None or preview.isNull():
            self._skip_resolution_crossfade_once = False
            return False

        if prefer_gallery or getattr(self, "_loading_from_gallery", False):
            self._gallery_preview_pending_full = True
            self._mark_pending_resolution_crossfade()
        self._gallery_instant_display_quality = gal_max >= min_quality
        self.display_pixmap(preview)
        return True

    def _queue_single_view_embedded_preview_load(self, file_path: str) -> None:
        """Request embedded thumbnail / preview upgrade for the current single-view file."""
        if not file_path or not is_raw_file(file_path):
            return
        norm_fp = _norm_path(file_path)
        if norm_fp in getattr(self, "_preview_upgrade_attempted_paths", set()):
            return
        self._preview_upgrade_attempted_paths.add(norm_fp)
        try:
            from image_load_manager import Priority

            self.image_manager.load_image(
                file_path,
                priority=Priority.CURRENT,
                cancel_existing=False,
                use_full_resolution=False,
                stages={"thumbnail", "exif", "full"},
            )
        except Exception:
            pass

    def _try_paint_navigation_cache_preview(self, file_path: str) -> bool:
        if self._full_res_only_mode():
            return False
        """Single-view arrow keys / filmstrip: paint prefetched preview or full before decode."""
        if not file_path or getattr(self, "view_mode", "single") != "single":
            return False
        preserve_zoom = self._preserve_zoom_navigation_active()
        self._skip_resolution_crossfade_once = True
        self._gallery_preview_pending_full = False
        self._clear_pending_resolution_crossfade()
        self._nav_cache_paint_tier = None
        try:
            full = self.image_cache.get_full_image(file_path)
            if full is not None:
                if self._display_cached_numpy_for_path(
                    file_path, full, from_cache="nav_full"
                ):
                    self._nav_cache_paint_tier = "full"
                    return True
                return False
        except Exception:
            pass
        if preserve_zoom:
            return False
        suppress_jpeg = self._suppress_jpeg_interim(file_path)
        try:
            preview = self.image_cache.get_preview_memory_only(file_path)
            if (
                preview is not None
                and (not suppress_jpeg or self.image_cache.is_libraw_preview(file_path))
                and self._preview_acceptable_for_single_view(
                    preview, file_path=file_path, context="navigation"
                )
            ):
                if self._display_cached_numpy_for_path(
                    file_path, preview, from_cache="nav_preview"
                ):
                    self._nav_cache_paint_tier = "preview"
                    return True
                return False
        except Exception:
            pass
        if suppress_jpeg:
            # RAW workflow: no camera-JPEG interim; wait for the LibRaw render.
            return False
        if self._paint_instant_preview_for_path(file_path, prefer_gallery=False):
            self._nav_cache_paint_tier = "instant"
            return True
        return False

    def _retry_composite_dng_embedded_preview(
        self, file_path: str, error_message: str
    ) -> bool:
        """After LibRaw full decode fails on composite HDR DNG, load embedded thumbnail once."""
        from common_image_loader import dng_prefers_embedded_preview_first, is_raw_file
        from image_load_manager import Priority

        if not file_path or not is_raw_file(file_path):
            return False
        if not dng_prefers_embedded_preview_first(file_path):
            return False
        err = str(error_message or "").lower()
        if "unsupported" not in err and "could not decode" not in err:
            return False
        norm = _norm_path(file_path)
        retried = getattr(self, "_composite_dng_embedded_retry_paths", None)
        if retried is None:
            self._composite_dng_embedded_retry_paths = set()
            retried = self._composite_dng_embedded_retry_paths
        if norm in retried:
            return False
        retried.add(norm)
        import logging

        logging.getLogger(__name__).info(
            "[MANAGER] Composite DNG LibRaw failed; retrying embedded preview for %s",
            os.path.basename(file_path),
        )
        self.status_bar.showMessage(
            f"Loading embedded preview for {os.path.basename(file_path)}…",
            4000,
        )
        self.image_manager.load_image(
            file_path=file_path,
            priority=Priority.CURRENT,
            cancel_existing=True,
            use_full_resolution=False,
            stages={"thumbnail"},
        )
        return True

    def _raw_preview_failure_markers(self) -> tuple[str, ...]:
        return (
            "display-tier preview extraction failed",
            "thumbnail extraction returned none",
        )

    def _retry_raw_preview_with_full_decode(
        self, file_path: str, error_message: str
    ) -> bool:
        """Auto-fallback: embedded/thumbnail preview failed -> queue sensor full decode once."""
        from common_image_loader import is_raw_file

        lower_err = str(error_message or "").lower()
        if not any(m in lower_err for m in self._raw_preview_failure_markers()):
            return False
        if not is_raw_file(file_path):
            return False
        if getattr(self, "view_mode", "single") != "single":
            return False

        norm_p = _norm_path(file_path)
        if self.image_manager.has_active_work_for_path(file_path):
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.show_loading("Decoding RAW…")
            self.status_bar.showMessage(
                f"{os.path.basename(file_path)}: preview unavailable, decoding full RAW…",
                4000,
            )
            return True

        retry_token = (
            norm_p,
            int(getattr(self, "_single_view_display_generation", 0)),
        )
        if retry_token == getattr(self, "_display_tier_full_retry_token", None):
            return False
        self._display_tier_full_retry_token = retry_token

        try:
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.show_loading("Decoding RAW…")
            self._request_full_res_decode(file_path, priority_current=True)
            self.status_bar.showMessage(
                f"{os.path.basename(file_path)}: preview unavailable, decoding full RAW…",
                4000,
            )
            return True
        except Exception:
            return False

    def _load_persisted_visual_rotations(self) -> None:
        """Restore non-destructive visual rotations from QSettings (cross-session)."""
        import json

        try:
            raw = self.get_settings().value("visual_rotations", "")
            if not raw:
                return
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return
            loaded = {}
            for k, v in data.items():
                try:
                    deg = int(v) % 360
                    if deg and isinstance(k, str):
                        loaded[_norm_path(k)] = deg
                except (TypeError, ValueError):
                    continue
            self._visual_rotation_degrees = loaded
        except Exception:
            pass

    def _save_persisted_visual_rotations(self) -> None:
        """Persist non-destructive visual rotations to QSettings."""
        import json

        try:
            data = {
                k: int(v) % 360
                for k, v in self._visual_rotation_degrees.items()
                if int(v) % 360
            }
            self.get_settings().setValue(
                "visual_rotations",
                json.dumps(data, separators=(",", ":")),
            )
        except Exception:
            pass

    def _migrate_visual_rotation_path(self, old_path: str, new_path: str) -> None:
        old_k = _norm_path(old_path)
        new_k = _norm_path(new_path)
        if old_k == new_k:
            return
        deg = self._visual_rotation_degrees.pop(old_k, None)
        if deg:
            self._visual_rotation_degrees[new_k] = deg
            self._save_persisted_visual_rotations()

    def _on_zoom_slider_pressed(self):
        if hasattr(self, "gallery_justified") and self.gallery_justified:
            self.gallery_justified.begin_zoom_session()

    def _on_zoom_slider_changed(self, value):
        if hasattr(self, "gallery_justified") and self.gallery_justified:
            gj = self.gallery_justified
            gj.apply_gallery_zoom(value)

    def _on_zoom_slider_released(self):
        if hasattr(self, "gallery_justified") and self.gallery_justified:
            value = self.size_slider.value()
            self.get_settings().setValue("gallery_row_height", value)
            if self._zoom_debounce_timer.isActive():
                self._zoom_debounce_timer.stop()
            gj = self.gallery_justified
            gj.apply_gallery_zoom(value)
            gj.end_zoom_session()

    def _extract_embedded_preview(self, file_path):
        """Extract oriented RAW preview for gallery fallbacks (shared loader)."""
        from common_image_loader import load_raw_preview_pixmap

        pixmap = load_raw_preview_pixmap(file_path, max_size=1024)
        return pixmap if pixmap is not None and not pixmap.isNull() else None

    def _is_gallery_path_edited(self, file_path: str) -> bool:
        """Cheap "has saved RAW adjustments" signal for the gallery badge.

        RAW-only: sidecars resolve by basename, so a JPEG next to an edited
        RAW of the same name would otherwise be badged (and, since edited
        previews landed, rendered) with the RAW's edits.

        Presence of an ``.xmp`` alone is not enough -- rating-only sidecars
        (or foreign LR fields with all our sliders at default) must not
        light the pencil badge after Reset.

        Memoized by sidecar mtime so ``load_visible_images`` does not re-parse
        XML for every tile on every pass (that stalled large-folder gallery
        re-entry).
        """
        from raw_adjustments import editing_features_enabled

        if not editing_features_enabled():
            return False
        if not file_path:
            return False
        try:
            from common_image_loader import is_raw_file
            from raw_adjustments import (
                is_default_adjustments,
                load_adjustments_from_xmp,
                resolve_xmp_path,
            )

            if not is_raw_file(file_path):
                return False
            memo = getattr(self, "_gallery_edited_memo", None)
            if memo is None:
                self._gallery_edited_memo = {}
                memo = self._gallery_edited_memo
            # TTL short-circuit BEFORE any disk stat: load_visible_images calls
            # this per tile per pass, and even the isfile+getmtime pair costs
            # 1-3ms on external volumes -- enough to eat the pass budget and
            # starve tile fill after a section jump. External sidecar changes
            # surface within the TTL; in-app edits invalidate the memo directly.
            now = time.time()
            hit = memo.get(file_path)
            if hit is not None and len(hit) >= 3 and (now - hit[2]) < 2.0:
                return bool(hit[1])
            xmp_path = resolve_xmp_path(file_path)
            if not xmp_path or not os.path.isfile(xmp_path):
                memo[file_path] = (None, False, now)
                return False
            try:
                mtime = os.path.getmtime(xmp_path)
            except OSError:
                memo[file_path] = (None, False, now)
                return False
            if hit is not None and hit[0] == mtime:
                memo[file_path] = (mtime, hit[1], now)
                return bool(hit[1])
            edited = not is_default_adjustments(load_adjustments_from_xmp(xmp_path))
            memo[file_path] = (mtime, edited, now)
            return edited
        except Exception:
            return False

    def _invalidate_gallery_edited_memo(self, file_path: str | None = None) -> None:
        """Drop cached edited-badge answers after save/reset."""
        memo = getattr(self, "_gallery_edited_memo", None)
        if not memo:
            return
        if not file_path:
            memo.clear()
            return
        memo.pop(file_path, None)
        with contextlib.suppress(Exception):
            memo.pop(os.path.normpath(file_path), None)

    def _should_hold_full_until_preview_paint(self, file_path: str) -> bool:
        """preview_first_stages: do not paint full-res before embedded preview lands.

        Do **not** gate on ``_should_defer_full_decode()`` here. That flag only
        delays *starting* extra sensor decodes during session restore. When
        ImageLoadManager already delivered a full buffer (e.g. DNG force-full
        fast-open), refusing to paint while also skipping a too-small cached
        preview (and with ``preview_first_stages=False``) leaves a blank view
        forever -- TTFR never logs, so staged session restore never resumes.
        """
        if self._suppress_jpeg_interim(file_path):
            # RAW workflow suppresses the embedded preview entirely -- the
            # LibRaw full IS the first render; holding it would deadlock the
            # display until the TTFR watchdog.
            return False
        defer = getattr(self, "_defer_sensor_full_decode_path", None)
        if not defer or _norm_path(defer) != _norm_path(file_path or ""):
            return False
        return not getattr(self, "_single_view_first_render_logged", False)

    def _set_adjust_panel_visible(self, visible: bool) -> None:
        self._adjust_overlay_visible = bool(visible)
        self._adjust_user_hidden = not self._adjust_overlay_visible
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None:
            return
        gv = getattr(self, "gpu_view", None)
        if gv is not None and hasattr(gv, "set_export_drag_enabled"):
            # Drag-out + share fight crop/D&B gestures while editing.
            gv.set_export_drag_enabled(not bool(visible))
        if gv is not None and hasattr(gv, "_wheel_navigate_in_fit"):
            # Plain scroll normally navigates prev/next in fit mode (browse
            # convenience) -- while Adjust is open, other wheel-driven actions
            # (slider hover, brush size when D&B isn't the active tool, etc.)
            # could bubble up and silently flip to a different photo mid-edit.
            # Disabled for the editing session, restored on close.
            gv._wheel_navigate_in_fit = not bool(visible)
        if self._adjust_overlay_visible:
            if self._raw_recovery_preview_active():
                self._disable_raw_recovery()
            if getattr(self, "_search_panel_expanded", False):
                self._set_search_panel_expanded(False, animate=False)
            self._maybe_maximize_brightness_for_adjust(True)
        bar = self._filmstrip_bar()
        if self._adjust_overlay_visible:
            # Opening: always show the panel. Do not gate on a display image --
            # opening from embedded-JPEG browse toggles RAW first and can
            # briefly clear on-screen pixels; the old gate fell through to the
            # close branch and left _adjust_overlay_visible stuck True with a
            # hidden panel (editor unreachable).
            # Hide floating browse overlays -- Adjust has its own histogram.
            panel.setVisible(True)
            self.single_image_histogram.hide()
            self._map_hidden_for_adjust = bool(
                getattr(self, "_map_overlay_visible", False)
            )
            if self._map_hidden_for_adjust:
                self._map_overlay_visible = False
                self._clear_location_map()
            # Give the adjust panel its preferred width so slider values
            # aren't clipped (panel min-width alone isn't enough under QSplitter).
            self._apply_adjust_panel_dock(self._adjust_panel_dock_side())
            self._sync_adjust_panel_for_current_file()
            path = getattr(self, "current_file_path", None)
            if path:
                self._begin_adjust_editing_session(path)
            if bar is not None:
                bar.setEnabled(False)
                # Remove (not just grey out) the film strip while editing:
                # it occupies vertical space the image needs, and navigation
                # from it is disabled anyway. Remember whether it was
                # showing so closing the editor restores the user's state.
                self._filmstrip_hidden_for_adjust = bar.isVisible()
                if bar.isVisible():
                    bar.hide()
                container = getattr(self, "single_view_container", None)
                if container is not None and hasattr(container, "set_filmstrip_pointer_active"):
                    container.set_filmstrip_pointer_active(False)
        else:
            gv = getattr(self, "gpu_view", None)
            if gv is not None and gv.is_color_pick_mode():
                self._cancel_wb_picker()
            # Esc / E close the editor without going through Crop Cancel --
            # drop the interactive overlay and D&B brush so they cannot
            # linger on the single-image view after the panel is gone.
            self._crop_preview_uncropped = False
            self._geometry_overlay_preview = False
            self._transform_interacting = False
            self._adjust_transform_source_u8 = None
            self._adjust_transform_source_adj = None
            self._adjust_transform_source_norm = None
            if panel.is_crop_mode():
                panel.set_crop_mode_ui(False)
            if gv is not None and gv.is_crop_mode():
                gv.set_crop_mode(False)
            if hasattr(panel, "disarm_dodge_burn"):
                panel.disarm_dodge_burn()
            if gv is not None and gv.is_dodge_burn_mode():
                gv.set_dodge_burn_mode(False)
                gv.set_dodge_burn_mask_overlay(None, False)
            # Leaving the editor: no in-flight edit render matters any more.
            if getattr(self, "_adjust_loading_edit_norm", None) is not None:
                self._set_edit_loading_overlay(None)
            panel.setVisible(False)
            if getattr(self, "_histogram_overlay_visible", False):
                self.single_image_histogram.show()
            if getattr(self, "_map_hidden_for_adjust", False):
                self._map_hidden_for_adjust = False
                self._map_overlay_visible = True
                self._sync_location_map()
            if not self._adjust_overlay_visible:
                self._adjust_preview_base_rgb = None
                self._adjust_preview_base_rgb_fast = None
                self._adjust_edit_base_path = None
                self._reset_adjust_compare_state()
                for attr in ("_adjust_preview_stage_cache", "_adjust_preview_stage_cache_fast"):
                    cache = getattr(self, attr, None)
                    if cache is not None and hasattr(cache, "reset"):
                        with contextlib.suppress(Exception):
                            cache.reset()
                try:
                    from processing.unified import release_edit_base_cache

                    release_edit_base_cache()
                except Exception:
                    pass
                try:
                    from processing.raw_decode.scunet import release_scunet_sessions

                    release_scunet_sessions()
                except Exception:
                    pass
            if bar is not None:
                bar.setEnabled(True)
                if getattr(self, "_filmstrip_hidden_for_adjust", False):
                    bar.show()
                    self._filmstrip_hidden_for_adjust = False
            self._maybe_maximize_brightness_for_adjust(False)
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "relayout_adjust"):
            container.relayout_adjust()
            if hasattr(container, "_raise_single_view_layers"):
                container._raise_single_view_layers()
        self._last_status_signature = None
        self.update_status_bar()
        if hasattr(self, "shortcuts_hint_button"):
            self.shortcuts_hint_button.setToolTip(self._keyboard_shortcuts_tooltip())
        QTimer.singleShot(0, self._restore_keyboard_focus)

    def _adjust_maximize_brightness_enabled(self) -> bool:
        """Default on; disable with settings or RAWVIEWER_ADJUST_MAX_BRIGHTNESS=0."""
        env = os.environ.get("RAWVIEWER_ADJUST_MAX_BRIGHTNESS", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            return False
        if env in ("1", "true", "yes", "on"):
            return True
        try:
            return bool(
                self.get_settings().value("adjust_maximize_brightness", True, type=bool)
            )
        except Exception:
            return True

    def _adjust_panel_dock_side(self) -> str:
        """Persisted Adjust panel edge: ``left`` or ``right`` (default)."""
        try:
            from rawviewer_ui.widgets.adjust import (
                ADJUST_DOCK_DEFAULT,
                ADJUST_DOCK_SETTINGS_KEY,
                normalize_adjust_dock,
            )
        except Exception:
            return "right"
        try:
            return normalize_adjust_dock(
                self.get_settings().value(ADJUST_DOCK_SETTINGS_KEY, ADJUST_DOCK_DEFAULT)
            )
        except Exception:
            return "right"

    def _apply_adjust_panel_dock(self, side: str) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        splitter = getattr(self, "single_view_splitter", None)
        container = getattr(self, "single_view_container", None)
        if panel is None or splitter is None or container is None:
            return
        try:
            from rawviewer_ui.widgets.adjust import apply_adjust_splitter_dock
        except Exception:
            return
        applied = apply_adjust_splitter_dock(splitter, container, panel, side)
        if hasattr(panel, "set_dock_side"):
            panel.set_dock_side(applied)
        if container is not None and hasattr(container, "relayout_adjust"):
            container.relayout_adjust()

    def _on_adjust_panel_dock_requested(self, side: str) -> None:
        try:
            from rawviewer_ui.widgets.adjust import (
                ADJUST_DOCK_SETTINGS_KEY,
                normalize_adjust_dock,
            )
        except Exception:
            return
        side = normalize_adjust_dock(side)
        with contextlib.suppress(Exception):
            self.get_settings().setValue(ADJUST_DOCK_SETTINGS_KEY, side)
        self._apply_adjust_panel_dock(side)

    def _maybe_maximize_brightness_for_adjust(self, entering: bool) -> None:
        """Raise built-in display brightness while editing; restore on exit."""
        if not self._adjust_maximize_brightness_enabled():
            if not entering:
                self._restore_adjust_brightness()
            return
        if entering:
            if getattr(self, "_adjust_brightness_saved", None) is not None:
                return
            try:
                from platform.display_brightness import get_brightness, set_brightness

                hwnd = self._display_brightness_hwnd()
                current = get_brightness(hwnd=hwnd)
                if current is None:
                    return
                # Already near max -- nothing to do / restore.
                if current >= 0.98:
                    self._adjust_brightness_saved = current
                    return
                if set_brightness(1.0, hwnd=hwnd):
                    self._adjust_brightness_saved = current
            except Exception:
                self._adjust_brightness_saved = None
            return
        self._restore_adjust_brightness()

    def _restore_adjust_brightness(self) -> None:
        saved = getattr(self, "_adjust_brightness_saved", None)
        self._adjust_brightness_saved = None
        if saved is None:
            return
        try:
            from platform.display_brightness import set_brightness

            set_brightness(float(saved), hwnd=self._display_brightness_hwnd())
        except Exception:
            pass

    def _editing_supported_for_file(self, file_path: str | None) -> bool:
        """Whether the Adjust panel can open for this file.

        Supported: RAW files that either demosaic via LibRaw or can fall back
        to an embedded-JPEG edit base (HE/HE* NEF, ARW6 before the packaging
        pin, …). Unsupported: non-RAW rasters, animated images, and DNGs that
        prefer a composite/embedded preview with no usable demosaic path.
        """
        if not file_path:
            return False
        try:
            if self.is_animated_image(_require_str_path(file_path)):
                return False
        except Exception:
            pass
        try:
            if not is_raw_file(file_path):
                return False
            from common_image_loader import dng_prefers_embedded_preview_first

            # Composite/HDR DNGs: browse-only (no demosaic edit base).
            if dng_prefers_embedded_preview_first(file_path):
                return False
            # HE/HE* and other demosaic-failed RAWs stay editable: decode falls
            # back to the embedded-JPEG edit base (see decode_raw_edit_base).
        except Exception:
            return False
        return True

    def _toggle_adjust_panel(self) -> None:
        from raw_adjustments import editing_features_enabled

        if not editing_features_enabled():
            return
        opening = not getattr(self, "_adjust_overlay_visible", False)
        if opening and not self._editing_supported_for_file(
            getattr(self, "current_file_path", None)
        ):
            self._show_auto_dismiss_notice(
                "Editing not available for this file",
                detail=(
                    "This file has neither a demosaicable RAW base nor a "
                    "usable embedded preview to edit."
                ),
                duration_ms=2000,
            )
            return

        if opening:
            settings = self.get_settings()
            use_embedded = settings.value("use_embedded_jpeg_workflow", True, type=bool)
            # Editor needs scene-linear demosaic. Remember if we flipped the
            # global workflow so Esc / E close can restore browse (JPEG) mode.
            if use_embedded:
                self._adjust_forced_raw_workflow = True
                self.toggle_raw_jpeg_workflow()
            else:
                self._adjust_forced_raw_workflow = False
            self._set_adjust_panel_visible(True)
        else:
            self._close_adjust_panel_from_ui()
