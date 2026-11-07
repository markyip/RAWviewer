"""display mixin, part 2 of 6 (32 methods, 775 lines)."""
from __future__ import annotations

import logging
import os
import sys

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QColor, QPixmap, QTransform

from core.config import _norm_path

from .. import lazy
from ..lazy import image_covers_sensor_resolution, is_raw_file

lazy.register(sys.modules[__name__])


class DisplayPart2:
    def _upgrade_display_quality_on_fit_if_needed(self) -> None:
        """After zoom-out to fit, replace a soft buffer with cached or queued sensor full."""
        if self._raw_recovery_preview_active():
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        if not self._needs_full_resolution_upgrade():
            return
        fp = getattr(self, "current_file_path", None)
        if not fp:
            return
        try:
            cached_full = self.image_cache.get_full_image(fp)
            if cached_full is not None and hasattr(cached_full, "shape"):
                ch, cw = cached_full.shape[0], cached_full.shape[1]
                exif = self.image_cache.get_exif(fp)
                if image_covers_sensor_resolution(cw, ch, exif):
                    self._orientation_already_applied = True
                    self.display_numpy_image(cached_full)
                    return
        except Exception:
            pass
        self._request_full_res_decode(fp, priority_current=True)

    def _gpu_zoom_in_to_point_finish(self) -> None:
        """Apply 100% zoom centered on ``zoom_center_point`` in GPU view."""
        if self._raw_recovery_preview_active():
            self._notify_raw_recovery_zoom_blocked()
            return
        gv = getattr(self, "gpu_view", None)
        if gv is None or not gv.has_pixmap():
            return
        pt = getattr(self, "zoom_center_point", None)
        if pt is None:
            w, h = gv.image_size()
            pt = QPoint(max(0, w // 2), max(0, h // 2))
            self.zoom_center_point = pt
        gv.zoom_to_actual_at(float(pt.x()), float(pt.y()))
        self.fit_to_window = False
        self.current_zoom_level = float(gv.current_scale())
        if getattr(self, "_pending_zoom_focus_subject", False) or getattr(
            self, "_focus_zoom_anchor_active", False
        ):
            self._focus_zoom_anchor_active = True
        else:
            self._pending_zoom_focus_subject = False
        self._gpu_request_full_resolution_if_needed()
        self.update_status_bar()

    def _neutral_rgb_for_raw_display(self, file_path: str):
        if not file_path:
            return None
        for getter in (self.image_cache.get_full_image, self.image_cache.get_preview):
            try:
                buf = getter(file_path)
            except Exception:
                buf = None
            if buf is not None and hasattr(buf, "shape"):
                return buf
        return None

    def _display_raw_recovery_buffer(self, file_path: str, rgb) -> None:
        if self._adjust_panel_active():
            return
        self._raw_recovery_skip_display_cache = True
        self._skip_resolution_downgrade_check = True
        try:
            self.display_numpy_image(rgb)
        finally:
            self._raw_recovery_skip_display_cache = False
            self._skip_resolution_downgrade_check = False

    def _refresh_neutral_raw_display(self) -> None:
        path = getattr(self, "current_file_path", None)
        if (
            not path
            or getattr(self, "view_mode", "single") != "single"
            or not is_raw_file(path)
        ):
            return
        neutral = self._neutral_rgb_for_raw_display(path)
        if neutral is None:
            return
        self._orientation_already_applied = True
        # Recovery preview paints leave _manager_displayed_max_dim at the
        # sensor-res high-water-mark from before recovery was toggled on
        # (same file path, so the per-file reset in display_pixmap's
        # _commit_display_metadata never fires). Without this escape hatch,
        # _already_displaying_buffer_for_path() sees that stale high-water
        # mark and thinks the neutral buffer is "already on screen", so the
        # repaint is silently skipped -- state flips to OFF (title/status
        # update correctly) but the recovery-look pixels never actually
        # leave the screen. Mirrors the same guard _display_raw_recovery_buffer
        # uses for the ON path.
        self._skip_resolution_downgrade_check = True
        try:
            self.display_numpy_image(neutral)
        finally:
            self._orientation_already_applied = False
            self._skip_resolution_downgrade_check = False

        if self._needs_full_resolution_upgrade(path):
            self._request_full_res_decode(path, priority_current=True)

    def _build_clipping_overlay_pixmap(self, pm):
        from processing.raw_decode.exposure_clipping import clipping_overlay_pixmap, rgb_array_from_pixmap

        rgb = rgb_array_from_pixmap(pm)
        if rgb is None:
            return None
        return clipping_overlay_pixmap(rgb)

    def _ensure_clipping_overlay_cached(self) -> None:
        if not getattr(self, "_clipping_overlay_active", False):
            self._clipping_overlay_pm = None
            self._clipping_overlay_cache_key = None
            return
        pm = getattr(self, "current_pixmap", None)
        if pm is None or pm.isNull():
            self._clipping_overlay_pm = None
            self._clipping_overlay_cache_key = None
            return
        token = self._clipping_overlay_cache_token(pm)
        if token != getattr(self, "_clipping_overlay_cache_key", None):
            self._clipping_overlay_pm = self._build_clipping_overlay_pixmap(pm)
            self._clipping_overlay_cache_key = token

    def _sync_clipping_overlay(self) -> None:
        vm = getattr(self, "view_mode", "single")
        active = getattr(self, "_clipping_overlay_active", False)

        if vm == "compare":
            for view in (
                getattr(self, "compare_select_view", None),
                getattr(self, "compare_candidate_view", None),
            ):
                if view is not None:
                    is_anim = getattr(view, "file_path", "") and self.is_animated_image(view.file_path)
                    if active and view.has_pixmap() and not is_anim:
                        pm = view.pixmap()
                        if pm is not None and not pm.isNull():
                            mask = self._build_clipping_overlay_pixmap(pm)
                            view.set_clipping_overlay(mask)
                    else:
                        view.clear_clipping_overlay()
                    view.update()
            return

        self._ensure_clipping_overlay_cached()
        gv = getattr(self, "gpu_view", None)
        path = getattr(self, "current_file_path", None)
        is_anim = path and self.is_animated_image(path)
        if gv is not None:
            if active and not is_anim:
                gv.set_clipping_overlay(getattr(self, "_clipping_overlay_pm", None))
            else:
                gv.clear_clipping_overlay()
            return
        if active and not is_anim:
            self._redraw_single_view_pixmap_without_relayout()

    def _toggle_clipping_overlay(self) -> bool:
        vm = getattr(self, "view_mode", "single")
        if vm not in ("single", "compare"):
            return False
        self._clipping_overlay_active = not getattr(
            self, "_clipping_overlay_active", False
        )
        self._clipping_overlay_cache_key = None
        if self._clipping_overlay_active:
            self.status_bar.showMessage("Clipping ON -- red=highlights, blue=shadows. J=off", 4000)
        else:
            self.status_bar.showMessage("Clipping overlay off", 2000)
        self._sync_clipping_overlay()
        return True

    def _resolve_pending_zoom_center_for_pixmap(self, pixmap) -> QPoint | None:
        """Map a deferred zoom target into *pixmap* pixel coordinates."""
        if pixmap is None or pixmap.isNull():
            return None
        if getattr(self, "_pending_zoom_focus_subject", False) and getattr(self, "_focus_subject_outline_active", False):
                self._refresh_focus_subject_rect_from_exif()
                rect = getattr(self, "_focus_subject_rect_image", None)
                if (
                    rect is not None
                    and not rect.isNull()
                    and rect.width() >= 1
                    and rect.height() >= 1
                ):
                    return QPoint(
                        rect.left() + rect.width() // 2,
                        rect.top() + rect.height() // 2,
                    )
        pending = getattr(self, "_pending_zoom_center", None)
        thumb = getattr(self, "_pending_zoom_thumbnail_size", None)
        if (
            pending is not None
            and thumb is not None
            and thumb.width() > 0
            and thumb.height() > 0
        ):
            return QPoint(
                int(pending.x() * pixmap.width() / thumb.width()),
                int(pending.y() * pixmap.height() / thumb.height()),
            )
        if pending is not None:
            return QPoint(pending.x(), pending.y())
        return QPoint(pixmap.width() // 2, pixmap.height() // 2)

    def _pixmap_size_changed(self, old_pm, new_pm) -> bool:
        if old_pm is None or new_pm is None or old_pm.isNull() or new_pm.isNull():
            return False
        return old_pm.size() != new_pm.size()

    def _should_reanchor_focus_on_pixmap_upgrade(self, pixmap) -> bool:
        if getattr(self, "_workflow_toggle_fit_until_display", False):
            return False
        if self._single_view_is_fit_mode():
            return False
        if getattr(self, "_focus_zoom_anchor_active", False):
            return True
        if getattr(self, "_pending_zoom_focus_subject", False):
            return True
        return bool(getattr(self, "_pending_zoom", False) and getattr(self, "zoom_center_point", None) is not None)

    def _focus_zoom_target_for_pixmap(self, pixmap) -> QPoint | None:
        """Image-pixel zoom anchor for *pixmap* (focus box preferred when active)."""
        if pixmap is None or pixmap.isNull():
            return None
        if getattr(self, "_focus_subject_outline_active", False) or getattr(
            self, "_focus_zoom_anchor_active", False
        ):
            self._refresh_focus_subject_rect_from_exif()
            rect = getattr(self, "_focus_subject_rect_image", None)
            if (
                rect is not None
                and not rect.isNull()
                and rect.width() >= 1
                and rect.height() >= 1
            ):
                return QPoint(
                    rect.left() + rect.width() // 2,
                    rect.top() + rect.height() // 2,
                )
        zcp = getattr(self, "zoom_center_point", None)
        old_pm = getattr(self, "current_pixmap", None)
        if zcp is not None and old_pm is not None and not old_pm.isNull():
            ow, oh = old_pm.width(), old_pm.height()
            nw, nh = pixmap.width(), pixmap.height()
            if ow > 0 and oh > 0 and (ow != nw or oh != nh):
                return QPoint(
                    int(zcp.x() * nw / ow),
                    int(zcp.y() * nh / oh),
                )
        if zcp is not None:
            return QPoint(zcp)
        return QPoint(max(0, pixmap.width() // 2), max(0, pixmap.height() // 2))

    def _clear_focus_zoom_anchor(self) -> None:
        self._focus_zoom_anchor_active = False
        self._pending_zoom_focus_subject = False

    def _finish_focus_zoom_anchor_if_sensor_res(self) -> None:
        if not self._needs_full_resolution_upgrade():
            self._clear_focus_zoom_anchor()

    def _apply_gpu_pixmap_with_focus_anchor(
        self, gv, pixmap: QPixmap, *, preserve: bool
    ) -> None:
        """Swap the GPU pixmap without losing a focus-box or pending zoom anchor."""
        reanchor = self._should_reanchor_focus_on_pixmap_upgrade(pixmap)
        if reanchor:
            target = self._focus_zoom_target_for_pixmap(pixmap)
            if target is not None:
                scale = float(gv.current_scale()) if gv.has_pixmap() else 1.0
                if (
                    getattr(gv, "_zoom_intent_100", False)
                    or getattr(self, "_pending_zoom_focus_subject", False)
                    or getattr(self, "_focus_zoom_anchor_active", False)
                    or scale >= 1.0 - 1e-4
                ):
                    scale = max(scale, 1.0)
                gv.set_pixmap_zoomed_at(
                    pixmap,
                    float(target.x()),
                    float(target.y()),
                    scale=scale,
                )
                self.zoom_center_point = target
                self.fit_to_window = False
                self.current_zoom_level = float(gv.current_scale())
                self._finish_focus_zoom_anchor_if_sensor_res()
                return

        gv.set_pixmap(pixmap, preserve_view=True if preserve else None)

    def _clear_pending_point_zoom_state(self) -> None:
        self._pending_zoom = False
        self._pending_zoom_center = None
        self._pending_zoom_thumbnail_size = None
        if not getattr(self, "_focus_zoom_anchor_active", False):
            self._pending_zoom_focus_subject = False
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv._zoom_intent_100 = False

    def _pending_zoom_ready_for_pixmap(self, pixmap, file_path) -> bool:
        """True when a pending Space/double-click zoom may commit on this buffer."""
        if not getattr(self, "_pending_zoom", False):
            return False
        # User already returned to fit (GPU zoom-out): ignore a stale pending
        # zoom left from the previous 100% so navigation does not auto-zoom.
        if self._single_view_is_fit_mode() and not getattr(
            self, "_preserve_nav_zoom_active", False
        ):
            self._clear_pending_point_zoom_state()
            return False
        if not getattr(self, "_full_resolution_loading", False):
            return True
        if pixmap is None or pixmap.isNull() or not file_path:
            return False
        try:
            exif = self.image_cache.get_exif(file_path)
            return image_covers_sensor_resolution(
                pixmap.width(), pixmap.height(), exif
            )
        except Exception:
            return max(pixmap.width(), pixmap.height()) >= 3000

    def _apply_pending_gpu_point_zoom_in_place(self, pixmap, gv) -> bool:
        """Swap to full pixmap and zoom to the pending target in one step (no corner flash)."""
        if not self._pending_zoom_ready_for_pixmap(
            pixmap, getattr(self, "current_file_path", None)
        ):
            return False
        target = self._resolve_pending_zoom_center_for_pixmap(pixmap)
        if target is None:
            return False
        gv.setUpdatesEnabled(False)
        try:
            gv.set_pixmap_zoomed_at(
                pixmap, float(target.x()), float(target.y()), scale=1.0
            )
        finally:
            gv.setUpdatesEnabled(True)
        self.zoom_center_point = target
        self.fit_to_window = False
        self.current_zoom_level = float(gv.current_scale())
        focus_zoom = getattr(self, "_pending_zoom_focus_subject", False)
        self._clear_pending_point_zoom_state()
        if focus_zoom or getattr(self, "_focus_zoom_anchor_active", False):
            self._focus_zoom_anchor_active = True
        self._finish_focus_zoom_anchor_if_sensor_res()
        return True

    def _gpu_zoom_anchor_for_navigation_restore(self) -> QPoint:
        """Viewport-centered image pixel for GPU navigation zoom preservation."""
        if self.zoom_center_point is not None:
            return QPoint(self.zoom_center_point)
        gv = getattr(self, "gpu_view", None)
        pm = getattr(self, "current_pixmap", None)
        if gv is not None and gv.has_pixmap():
            try:
                center_scene = gv.mapToScene(gv.viewport().rect().center())
                w, h = gv.image_size()
                if w > 0 and h > 0:
                    return QPoint(
                        max(0, min(int(center_scene.x()), w - 1)),
                        max(0, min(int(center_scene.y()), h - 1)),
                    )
            except Exception:
                pass
        if pm is not None and not pm.isNull():
            return QPoint(pm.width() // 2, pm.height() // 2)
        return QPoint(0, 0)

    def _apply_gpu_navigation_zoom_restore(
        self, pixmap, gv, *, scale_level_to_pixmap: bool = False
    ) -> bool:
        """Restore zoom level/center on a new file in the GPU view.

        ``scale_level_to_pixmap``: the pixmap is a low-resolution interim tier
        (RAW workflow delivers small previews first); scale the restore level
        by the size ratio so the tier shows at the *same on-screen framing*
        its full-resolution buffer will have -- blurry but geometrically
        continuous, instead of flashing out to fit and re-zooming seconds
        later when the sensor decode lands. The same-file upgrade path then
        preserves on-screen scale, so the final buffer lands at exactly the
        preserved zoom.
        """
        preserve = bool(
            getattr(self, "_preserve_nav_zoom_active", False)
            or getattr(self, "_maintain_zoom_on_navigation", False)
        )
        logging.getLogger(__name__).debug(
            "[ZOOMNAV] restore attempt: preserve=%s scale_mode=%s pm=%sx%s "
            "level=%s old_size=%s",
            preserve,
            scale_level_to_pixmap,
            pixmap.width() if pixmap is not None and not pixmap.isNull() else 0,
            pixmap.height() if pixmap is not None and not pixmap.isNull() else 0,
            getattr(self, "_restore_zoom_level", None),
            getattr(self, "_restore_pixmap_size", None),
        )
        if not preserve:
            return False

        eff_level = getattr(self, "_restore_zoom_level", None)
        if eff_level is None:
            eff_level = getattr(self, "current_zoom_level", 1.0)

        eff_center = getattr(self, "_restore_zoom_center", None)
        if eff_center is None:
            eff_center = getattr(self, "zoom_center_point", None)

        old_size = getattr(self, "_restore_pixmap_size", None)
        level_scale = 1.0
        sizes_differ = bool(
            old_size is not None
            and pixmap is not None
            and not pixmap.isNull()
            and old_size.width() > 0
            and old_size.height() > 0
            and (
                old_size.width() != pixmap.width()
                or old_size.height() != pixmap.height()
            )
        )
        if sizes_differ:
            sx = pixmap.width() / old_size.width()
            sy = pixmap.height() / old_size.height()
            if eff_center is not None:
                eff_center = QPoint(int(eff_center.x() * sx), int(eff_center.y() * sy))
            if scale_level_to_pixmap and sx > 0:
                level_scale = 1.0 / sx
        elif scale_level_to_pixmap and old_size is None:
            # Cannot compute an equivalent framing without the previous
            # buffer's size; let the caller fall back to deferred zoom.
            return False

        if eff_center is None and pixmap is not None and not pixmap.isNull():
            eff_center = QPoint(pixmap.width() // 2, pixmap.height() // 2)

        # Prefer the resolution-independent captured level: re-derive this
        # buffer's scale from EXIF sensor dims, so restoring onto any tier
        # (512px nav preview .. sensor-res) lands at the same on-screen
        # framing. Falls back to the buffer-relative math when EXIF dims are
        # unavailable.
        level = None
        overscale = bool(scale_level_to_pixmap)
        sensor_level = getattr(self, "_restore_sensor_level", None)
        # Prefer this file's own remembered framing over the neighbor's.
        mem = None
        try:
            mem = self._nav_zoom_memory.get(
                _norm_path(getattr(self, "current_file_path", "") or "")
            )
        except Exception:
            mem = None
        if mem is not None and pixmap is not None and not pixmap.isNull():
            fx, fy, mem_level = mem
            eff_center = QPoint(
                int(fx * pixmap.width()), int(fy * pixmap.height())
            )
            sensor_level = mem_level
        if sensor_level is not None and pixmap is not None and not pixmap.isNull():
            try:
                from common_image_loader import sensor_pixel_dimensions

                sdims = sensor_pixel_dimensions(
                    self.image_cache.get_exif(getattr(self, "current_file_path", None))
                )
                pm_max = max(pixmap.width(), pixmap.height())
                if sdims and pm_max > 0:
                    level = float(sensor_level) * max(sdims) / pm_max
                    overscale = pm_max < max(sdims) * 0.98
            except Exception:
                level = None
        if level is None:
            level = float(eff_level or 1.0) * level_scale
            src = "legacy"
        else:
            src = "mem" if mem is not None else "sensor"
        logging.getLogger(__name__).debug(
            "[ZOOMNAV] placing level=%.3f source=%s overscale=%s", level, src, overscale
        )

        gv.setUpdatesEnabled(False)
        try:
            fit = gv.fit_scale()
            if level <= fit * gv._FIT_SCALE_EPS:
                gv.set_pixmap(pixmap, preserve_view=False)
                gv.fit_to_window()
                self.fit_to_window = True
                self.current_zoom_level = float(gv.current_scale())
            else:
                target_scale = max(fit, level) if overscale else max(fit, min(gv.MAX_SCALE, level))
                gv.set_pixmap_zoomed_at(
                    pixmap,
                    float(eff_center.x()),
                    float(eff_center.y()),
                    scale=target_scale,
                    allow_overscale=overscale,
                )
                self.fit_to_window = False
                self.current_zoom_level = float(gv.current_scale())
            self.zoom_center_point = eff_center
        finally:
            gv.setUpdatesEnabled(True)

        self._restore_zoom_center = None
        self._restore_zoom_level = None
        self._restore_pixmap_size = None
        self._restore_sensor_level = None
        self._restore_start_scroll_x = None
        self._restore_start_scroll_y = None
        self._finish_nav_zoom_preserve()
        return True

    def _redraw_single_view_pixmap_without_relayout(self) -> None:
        if getattr(self, "gpu_view", None) is not None:
            # GPU path: focus and composition guides are scene overlays, not painted
            # onto the pixmap, so just refresh those without touching the transform.
            self._gpu_update_focus_overlay()
            self._gpu_update_composition_grid()
            self._gpu_update_clipping_overlay()
            return
        if not self.current_pixmap:
            return
        if self.fit_to_window:
            self.scale_image_to_fit()
        else:
            self.apply_zoom_and_pan()

    def _gpu_update_focus_overlay(self) -> None:
        """Reflect the current focus/subject rect onto the GPU scene overlay."""
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        rect = getattr(self, "_focus_subject_rect_image", None)
        active = getattr(self, "_focus_subject_outline_active", False)
        if active and isinstance(rect, QRect) and not rect.isNull():
            src = getattr(self, "_focus_rect_source", None) or ""
            color = QColor(255, 185, 45) if src == "makernote_af" else QColor(165, 255, 95)
            gv.set_overlay_rect(rect, color)
        else:
            gv.clear_overlay()

    def _gpu_update_clipping_overlay(self) -> None:
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        if not getattr(self, "_clipping_overlay_active", False):
            gv.clear_clipping_overlay()
            return
        self._ensure_clipping_overlay_cached()
        gv.set_clipping_overlay(getattr(self, "_clipping_overlay_pm", None))

    def _navigate_single_view_step(self, delta: int) -> None:
        """Prev/next in single view (filmstrip hotzone or main image list)."""
        if self._filmstrip_handles_input():
            bar = self._filmstrip_bar()
            if bar:
                bar.move_selection(delta)
                return
        self._debounced_navigate("prev" if delta < 0 else "next")

    def _already_displaying_buffer_for_path(self, file_path: str, image) -> bool:
        if _norm_path(getattr(self, "_displayed_content_path", "")) != _norm_path(file_path):
            return False
        if not self._single_view_pixels_on_screen(file_path):
            return False
        try:
            if hasattr(image, "shape"):
                max_dim = max(int(image.shape[0]), int(image.shape[1]))
            elif hasattr(image, "width"):
                max_dim = max(int(image.height()), int(image.width()))
            else:
                return False
        except Exception:
            return False
        shown = int(getattr(self, "_manager_displayed_max_dim", 0))
        return shown > 0 and max_dim <= int(shown * 1.06)

    def _clear_stale_single_view_image(self) -> None:
        """Remove the previous image so navigation does not linger on the wrong file.

        When a folder is already open (``current_file_path`` set), never flash the
        empty-state "No image loaded" help text -- that placeholder is only for
        true cold start / reset-to-initial. Browsing clears stay blank-black.
        """
        self._stop_animations()
        self._displayed_content_path = None
        self._cancel_content_crossfade()
        gv = getattr(self, "gpu_view", None)
        browsing = bool(getattr(self, "current_file_path", None))
        if gv is not None:
            if browsing and hasattr(gv, "clear_image"):
                gv.clear_image(show_placeholder=False)
            else:
                gv.clear()
        if hasattr(self, "image_label") and self.image_label is not None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("")
        self.current_pixmap = None
        self._base_display_pixmap = None

    def on_manager_pixmap_ready(self, file_path: str, pixmap):
        """處理 ImageLoadManager 的 QPixmap 就緒信號（非 RAW 文件）"""
        import logging
        logger = logging.getLogger(__name__)

        if getattr(self, "view_mode", "") == "compare":
            self._compare_upgrade_view_image(file_path, pixmap)
            return
        if getattr(self, "_suppress_single_manager_callbacks", False):
            return
        if self._adjust_panel_blocks_display_refresh():
            return
        if getattr(self, "view_mode", "single") != "single":
            return

        # Only handle current file's pixmap (normalize for Windows path format differences)
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            logger.debug(f"[MANAGER] Pixmap for different file: {os.path.basename(file_path)}")
            return

        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()

        # Check if this is actually a RAW file (shouldn't happen, but log it)
        raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                         '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                         '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
        file_ext = os.path.splitext(file_path)[1].lower()
        is_raw_file = file_ext in raw_extensions

        if is_raw_file and not getattr(self, '_orientation_already_applied', False):
                orientation = self.get_orientation_from_exif(file_path)
                if orientation != 1:
                    pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)

        logger.info(f"[MANAGER] Pixmap ready for {os.path.basename(file_path)}")
        # load_pixmap_safe in common_image_loader.py now uses QImageReader with setAutoTransform
        # This automatically applies EXIF orientation. Orientation is already applied.
        # Set flag so display_pixmap doesn't apply it again
        self._orientation_already_applied = True

        # safe_print(f"[ORIENTATION] on_manager_pixmap_ready: _orientation_already_applied = {self._orientation_already_applied}")
        # Pixmap from manager is always full resolution for non-RAW files
        self._is_half_size_displayed = False
        logger.debug("[MANAGER] Setting _is_half_size_displayed=False for full resolution pixmap")

        # Avoid duplicate repaint churn: thumbnail_ready path may have already displayed the same
        # full-size pixmap through display_numpy_image() cache fast-path just moments earlier.
        # Compare against the un-rotated base buffer -- visual rotation swaps on-screen dimensions.
        if (
            not getattr(self, "_skip_resolution_downgrade_check", False)
            and self._manager_pixmap_is_duplicate_display(file_path, pixmap)
        ):
            logger.debug(
                f"[MANAGER] Skipping duplicate pixmap redraw for "
                f"{os.path.basename(file_path)} ({pixmap.width()}x{pixmap.height()})"
            )
            pm_max = max(pixmap.width(), pixmap.height())
            self._manager_displayed_max_dim = max(
                getattr(self, "_manager_displayed_max_dim", 0), pm_max
            )
            return

        self.display_pixmap(pixmap)
        self._skip_resolution_downgrade_check = False
        self._on_single_view_content_displayed()
        pm_max = max(pixmap.width(), pixmap.height())
        self._manager_displayed_max_dim = max(
            getattr(self, "_manager_displayed_max_dim", 0), pm_max
        )
        self.status_bar.showMessage(f"Loaded {os.path.basename(file_path)}")

        # CRITICAL: Ensure metadata is updated after pixmap is displayed
        # This ensures metadata is shown even if EXIF was ready before pixmap
        logger.info("[MANAGER] Updating status bar after pixmap display to ensure metadata is shown")
        self.update_status_bar()

        self.setFocus()
        self.schedule_save_session_state()
        self._start_preloading()

    def apply_orientation_to_pixmap(self, pixmap, orientation):
        """Apply orientation correction to QPixmap -- idempotent, make-independent.

        Pixmaps coming from extraction/cache are already in display orientation, so this
        must NOT blindly rotate: it measures the pixmap against the EXIF display
        orientation and rotates only when they disagree. This replaces the old make-based
        is_camera_pre_rotated() skip, which only protected Sony/Leica and left every other
        pre-rotated camera's image sideways after a redundant rotation.
        """
        if orientation == 1 or pixmap is None or pixmap.isNull():
            return pixmap

        # Skip when the pixmap already matches the expected display orientation.
        try:
            from common_image_loader import (
                exif_pixels_display_oriented,
                pixmap_matches_exif_display,
            )

            exif = None
            fp = getattr(self, "current_file_path", None)
            if fp and hasattr(self, "image_cache") and self.image_cache is not None:
                exif = self.image_cache.get_exif(fp)
            if exif_pixels_display_oriented(exif):
                return pixmap
            if exif:
                ow = int(exif.get("original_width") or 0)
                oh = int(exif.get("original_height") or 0)
                if ow > 0 and oh > 0 and pixmap_matches_exif_display(
                    pixmap.width(), pixmap.height(), ow, oh, int(orientation or 1)
                ):
                    return pixmap
        except Exception:
            pass

        transform = QTransform()

        if orientation == 2:
            # Mirrored horizontal
            transform.scale(-1, 1)
        elif orientation == 3:
            # Rotated 180 degrees
            transform.rotate(180)
        elif orientation == 4:
            # Mirrored vertical
            transform.scale(1, -1)
        elif orientation == 5:
            # Mirrored horizontal + Rotated 270° CW (k=1 CCW)
            transform.scale(-1, 1)
            transform.rotate(-90)
        elif orientation == 6:
            # Rotate 90° CW (k=3 CCW)
            transform.rotate(90)
        elif orientation == 7:
            # Mirror LR + rotate 90° CW
            transform.scale(-1, 1)
            transform.rotate(90)
        elif orientation == 8:
            # Rotate 270° CW (90° CCW) - need to rotate 90° CCW to correct
            transform.rotate(-90)

        return pixmap.transformed(transform)

    def _schedule_workflow_toggle_fit_complete(self) -> None:
        """Keep fit-to-window across all async paints from a RAW↔JPEG toggle."""
        if getattr(self, "_workflow_toggle_refit_timer_scheduled", False):
            return
        self._workflow_toggle_refit_timer_scheduled = True
        from PyQt6.QtCore import QTimer

        def _done() -> None:
            self._workflow_toggle_refit_timer_scheduled = False
            self._workflow_toggle_fit_until_display = False
            self._skip_resolution_downgrade_check = False

        QTimer.singleShot(300, _done)

    def toggle_raw_edr_display(self):
        """Toggle EDR display for RAW (macOS): highlight headroom vs decode speed.

        EDR display re-decodes each RAW with LibRaw highlight reconstruction
        (~12x the cost of a plain decode); off = standard-dynamic-range display
        straight from the already-decoded buffer, much faster browsing.
        """
        if getattr(self, "view_mode", "single") != "single":
            return
        settings = self.get_settings()
        current = settings.value("raw_edr_display_enabled", True, type=bool)
        settings.setValue("raw_edr_display_enabled", not current)
        self._update_edr_toggle_button_state()
        # Invalidate all workflow pixel caches (memory + disk) since EDR/SDR display encoding has toggled globally
        self._invalidate_workflow_pixel_caches()
        cp = getattr(self, "current_file_path", None)
        if cp and is_raw_file(cp):
            self._displayed_content_path = None
            self._gpu_last_path = None
            # Clear viewport pixmap to avoid format conversion mismatch / solarized flashing during reload
            if getattr(self, "gpu_view", None) is not None:
                self.gpu_view.clear_pixmap_keep_placeholder_hidden()
            else:
                self.image_label.setPixmap(QPixmap())
            self.load_raw_image(cp, force_reload=True)

    def _mark_animated_content_displayed(self, file_path: str) -> None:
        self._displayed_content_path = file_path
        self._on_single_view_content_displayed()
