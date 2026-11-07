"""display mixin, part 6 of 6 (14 methods, 408 lines)."""
from __future__ import annotations

import contextlib
import logging
import sys

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QCursor

from core.config import _norm_path, safe_print

from .. import lazy
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class DisplayPart6:
    def _capture_nav_zoom_state_before_load(self) -> None:
        """Snapshot GPU/legacy zoom so rapid or coalesced navigation can restore it."""
        logging.getLogger(__name__).debug(
            "[ZOOMNAV] capture: zoomed_in=%s", self._single_view_is_zoomed_in()
        )
        self._nav_zoom_interim_active = False
        if not self._single_view_is_zoomed_in():
            self._preserve_nav_zoom_active = False
            if hasattr(self, "_maintain_zoom_on_navigation"):
                with contextlib.suppress(AttributeError):
                    delattr(self, "_maintain_zoom_on_navigation")
            self._restore_zoom_center = None
            self._restore_zoom_level = None
            self._restore_pixmap_size = None
            self._restore_sensor_level = None
            return
        self._preserve_nav_zoom_active = True
        self._maintain_zoom_on_navigation = True
        if getattr(self, "_focus_subject_outline_active", False):
            self._recenter_on_focus_after_nav = True
        gv_nav = getattr(self, "gpu_view", None)
        if gv_nav is not None and gv_nav.has_pixmap():
            self._restore_zoom_center = self._gpu_zoom_anchor_for_navigation_restore()
            self._restore_zoom_level = float(gv_nav.current_scale())
            self.current_zoom_level = self._restore_zoom_level
        else:
            self._restore_zoom_center = self._zoom_anchor_for_navigation_restore()
            self._restore_zoom_level = getattr(self, "current_zoom_level", 1.0)
        if getattr(self, "current_pixmap", None):
            self._restore_pixmap_size = self.current_pixmap.size()
        # Resolution-independent zoom: the displayed buffer may be any tier
        # (including a low-res interim shown overscaled), so the raw view
        # scale is meaningless across buffers. Normalize to "scale on the
        # sensor-resolution buffer" using EXIF dims (max-dim ratio, immune to
        # orientation rotation); the restore side re-derives the per-buffer
        # scale the same way.
        self._restore_sensor_level = None
        try:
            from common_image_loader import sensor_pixel_dimensions

            exif = self.image_cache.get_exif(getattr(self, "current_file_path", None))
            sdims = sensor_pixel_dimensions(exif)
            pm = getattr(self, "current_pixmap", None)
            if sdims and pm is not None and not pm.isNull():
                pm_max = max(pm.width(), pm.height())
                s_max = max(sdims)
                if pm_max > 0 and s_max > 0 and self._restore_zoom_level:
                    self._restore_sensor_level = (
                        float(self._restore_zoom_level) * pm_max / s_max
                    )
        except Exception:
            self._restore_sensor_level = None
        # Remember this file's framing so navigating back restores *its* zoom
        # position, not the neighbor's.
        try:
            gv_mem = getattr(self, "gpu_view", None)
            cur = getattr(self, "current_file_path", None)
            if (
                cur
                and gv_mem is not None
                and gv_mem.has_pixmap()
                and self._restore_sensor_level is not None
            ):
                c = gv_mem.mapToScene(gv_mem.viewport().rect().center())
                fx = c.x() / max(1, gv_mem._img_w)
                fy = c.y() / max(1, gv_mem._img_h)
                self._nav_zoom_memory[_norm_path(cur)] = (
                    float(fx),
                    float(fy),
                    float(self._restore_sensor_level),
                )
        except Exception:
            pass

    def _post_navigation_display_tick(self) -> None:
        """After a navigation step, paint any cached interim preview if the viewport is still empty."""
        if getattr(self, "_navigation_in_progress", False):
            return
        path = getattr(self, "current_file_path", None)
        if not path or getattr(self, "view_mode", "single") != "single":
            return
        if self._single_view_pixels_on_screen(path):
            return
        if self._try_paint_navigation_cache_preview(path):
            self._queue_single_view_embedded_preview_load(path)
            return
        self._try_paint_display_quality_cache(path)

    def _clear_all_zoom_state(self):
        """Clear all variables that could cause accidental zoom restorations."""
        for attr in ['_pending_zoom', '_pending_zoom_center', '_pending_zoom_thumbnail_size',
                     '_pending_zoom_level', '_pending_zoom_restore', '_restore_zoom_center',
                     '_restore_zoom_level', '_preserve_nav_zoom_active', '_maintain_zoom_on_navigation']:
            if hasattr(self, attr):
                delattr(self, attr)
        # GPU-era flags grew after this helper was written and were never
        # added, so "clear all" callers still leaked deferred zooms through
        # the GPU path (gallery -> single landed at the PREVIOUS image's 100%
        # restore). Assign neutral values rather than delattr: unlike the
        # legacy attrs above (always read through getattr), these are normal
        # instance attributes other code reads directly.
        self._pending_zoom_toggle = False
        self._pending_gpu_zoom_point = None
        self._pending_zoom_focus_subject = False
        self._focus_zoom_anchor_active = False
        self._smooth_zoom_full_request_sent = False

    def _on_gpu_wheel_navigate(self, direction: int):
        """Plain wheel while fit-to-window navigates images (legacy parity)."""
        panel = getattr(self, "single_image_adjust_panel", None)
        if (
            panel is not None
            and getattr(self, "_adjust_overlay_visible", False)
            and panel.dodge_burn_mode() is not None
        ):
            # Safety net: D&B should have consumed the wheel in GpuImageView.
            self._on_dodge_burn_brush_size_wheel(120 if direction < 0 else -120)
            return
        import time
        now = time.monotonic()
        last = float(getattr(self, "_last_wheel_nav_ts", 0.0))
        # 250ms cooldown prevents trackpad flick inertia from skipping multiple images
        if now - last < 0.25:
            return
        self._last_wheel_nav_ts = now
        self._schedule_coalesced_navigation(1 if direction > 0 else -1)

    def _on_gpu_fit_mode_changed(self, fit_mode: bool):
        # Keep legacy state roughly in sync for any code that still reads it.
        self.fit_to_window = bool(fit_mode)
        if fit_mode:
            self.zoom_center_point = None
            self._clear_focus_zoom_anchor()
            # Match legacy Space→fit: drop deferred 100% / nav-preserve so the
            # next arrow-key image does not inherit a stale zoom intent.
            self._clear_all_zoom_state()
            self._nav_zoom_interim_active = False
            self._upgrade_display_quality_on_fit_if_needed()
        else:
            self._gpu_request_full_resolution_if_needed()
        self._update_gpu_status()

    def _on_gpu_zoom_changed(self, scale: float):
        self.current_zoom_level = float(scale)
        gv = getattr(self, "gpu_view", None)
        if gv is not None and not gv.is_fit_mode():
            self._touch_single_view_interactivity()
            self._gpu_request_full_resolution_if_needed()
        self._update_gpu_status()

    def toggle_zoom(self):
        """Toggle between fit-to-window and 100% zoom modes"""
        import logging
        logger = logging.getLogger(__name__)

        self._touch_single_view_interactivity()
        if self._raw_recovery_preview_active():
            self._notify_raw_recovery_zoom_blocked()
            return

        # Animated GIF/WebP: fit-to-window only (no Space/double-click zoom).
        if hasattr(self, "current_file_path") and self.is_animated_image(_require_str_path(self.current_file_path)):
            return

        # Route B: GPU view manages its own fit/zoom transform.
        if getattr(self, "gpu_view", None) is not None:
            self._stop_slideshow()
            gv = self.gpu_view
            gv._sync_fit_mode_flag()
            if gv.wants_zoom_in_toggle():
                if (
                    not getattr(self, "_focus_subject_outline_active", False)
                    and getattr(self, "zoom_center_point", None) is None
                ):
                    pm = self.current_pixmap
                    if pm is not None and not pm.isNull():
                        self.zoom_center_point = QPoint(
                            pm.width() // 2, pm.height() // 2
                        )
                    elif gv.has_pixmap():
                        w, h = gv.image_size()
                        self.zoom_center_point = QPoint(
                            max(0, w // 2), max(0, h // 2)
                        )
                self._allow_full_decode_for_user_zoom()
                self._try_sync_full_before_zoom()
                self._restore_adjust_hires_before_zoom()
                self._gpu_zoom_in_to_point_finish()
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
                    self._apply_adjust_panel_preview(full_quality=True)
                self.fit_to_window = gv.is_fit_mode()
                self.current_zoom_level = float(gv.current_scale())
                self.update_status_bar()
                return
            if not gv.has_pixmap():
                if getattr(self, "current_file_path", None):
                    self._pending_zoom_toggle = True
                    self._pending_zoom_target_path = self.current_file_path
                return
            gv.toggle_fit()
            self.fit_to_window = gv.is_fit_mode()
            if not gv.is_fit_mode():
                self.current_zoom_level = float(gv.current_scale())
                self._gpu_request_full_resolution_if_needed()
            else:
                self.zoom_center_point = None
                self._clear_all_zoom_state()
                self._nav_zoom_interim_active = False
            self.update_status_bar()
            return
        # Allow toggle even if pixmap is not ready yet - check if image is loading
        if not self.current_pixmap:
            # If image is currently loading, wait a bit and try again
            if hasattr(self, 'current_file_path') and self.current_file_path:
                # Always set the pending flag so we don't drop the user's spacebar press
                self._pending_zoom_toggle = True
                self._pending_zoom_target_path = self.current_file_path
                logger.debug("Pixmap not ready yet or still loading, will toggle zoom once image is loaded")
                return
            return
        self._stop_slideshow()
        if self._single_view_is_fit_mode():
            # Switch to 100% zoom mode - center on image center
            self.fit_to_window = False

            # Prefer the half-size preview flag -- EXIF embedded-preview WxH often matches pixmap and poison the cache comparison.
            should_load_full_resolution = False
            if self.current_pixmap:
                if hasattr(self, '_is_half_size_displayed') and self._is_half_size_displayed:
                    should_load_full_resolution = True
                    logger.info("User zoomed in -- preview/half-resolution display, loading full resolution")
                else:
                    cached_exif = self.image_cache.get_exif(self.current_file_path)
                    if cached_exif and cached_exif.get('original_width') and cached_exif.get('original_height'):
                        original_width = cached_exif['original_width']
                        original_height = cached_exif['original_height']
                        current_max = max(self.current_pixmap.width(), self.current_pixmap.height())
                        original_max = max(original_width, original_height)
                        if original_max > 0 and current_max < original_max * 0.8:
                            should_load_full_resolution = True
                            logger.info(
                                "User zoomed in -- pixmap smaller than cached original "
                                f"({self.current_pixmap.width()}x{self.current_pixmap.height()} vs {original_width}x{original_height}), loading full resolution"
                            )

            # If currently displaying thumbnail/half_size and user zooms in, load full resolution FIRST
            if should_load_full_resolution:
                if getattr(self, '_full_resolution_loading', False):
                    self._pending_zoom = True
                    self._pending_zoom_center = QPoint(self.current_pixmap.width() // 2, self.current_pixmap.height() // 2) if self.current_pixmap else None
                    self._pending_zoom_thumbnail_size = self.current_pixmap.size() if self.current_pixmap else None
                    # Keep fit preview until full-res decode completes.
                    self.update_status_bar()
                    self.setFocus()
                    return
                if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                    # Check if full resolution is already cached - if so, load it immediately
                    cached_full = self.image_cache.get_full_image(self.current_file_path)
                    if cached_full is not None:
                        cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                        if cached_max_dim >= 3000:
                            logger.info("Full resolution image already cached, loading immediately...")
                            self._full_resolution_loading = True
                            # Set flag to prevent display_pixmap from resetting fit_to_window
                            # We're about to zoom in, so we want to preserve that intent
                            self._maintain_zoom_on_navigation = True
                            self.display_numpy_image(cached_full)
                            self._is_half_size_displayed = False
                            self._full_resolution_loading = False
                            # Clear the flag after display
                            if hasattr(self, '_maintain_zoom_on_navigation'):
                                delattr(self, '_maintain_zoom_on_navigation')
                            # Continue to zoom setup below
                        else:
                            logger.info("Cached image is half_size, processing full resolution...")
                            self._load_full_resolution_on_demand()
                            self._pending_zoom = True
                            self._pending_zoom_center = QPoint(self.current_pixmap.width() // 2, self.current_pixmap.height() // 2) if self.current_pixmap else None
                            self._pending_zoom_thumbnail_size = self.current_pixmap.size() if self.current_pixmap else None
                            logger.debug("Stored pending 100% zoom - full decode first")
                            self.status_bar.showMessage("Loading full resolution for 100% zoom...")
                            # Do not render fake zoom on preview; apply native 100% after full-res.
                            self.update_status_bar()
                            self.setFocus()
                            return
                    else:
                        self._load_full_resolution_on_demand()
                        self._pending_zoom = True
                        self._pending_zoom_center = QPoint(self.current_pixmap.width() // 2, self.current_pixmap.height() // 2) if self.current_pixmap else None
                        self._pending_zoom_thumbnail_size = self.current_pixmap.size() if self.current_pixmap else None
                        logger.debug("Stored pending 100% zoom - full decode first")
                        self.status_bar.showMessage("Loading full resolution for 100% zoom...")
                        # Do not render fake zoom on preview; apply native 100% after full-res.
                        self.update_status_bar()
                        self.setFocus()
                        return

            # Set up zoom parameters
            self.current_zoom_level = 1.0
            # Always center on image center when using space bar
            image_center_x = self.current_pixmap.width() // 2
            image_center_y = self.current_pixmap.height() // 2
            self.zoom_center_point = QPoint(image_center_x, image_center_y)

            if self.current_pixmap:
                available_size = self.scroll_area.size()
                safe_width = max(1, available_size.width() - 20)
                safe_height = max(1, available_size.height() - 20)
                fit_scale_x = safe_width / max(1, self.current_pixmap.width())
                fit_scale_y = safe_height / max(1, self.current_pixmap.height())
                fit_scale = min(fit_scale_x, fit_scale_y)
                if self.current_zoom_level <= fit_scale * 1.05:
                    self.current_zoom_level = fit_scale * 2.0

            if self.current_zoom_level > 1.05:
                self.apply_zoom_and_pan()
            else:
                self.scale_image_to_100_percent()
                self.zoom_to_point()
        else:
            # Switch to fit-to-window mode
            self.fit_to_window = True
            self.current_zoom_level = 1.0
            self.zoom_center_point = None
            self._clear_all_zoom_state()
            self.scale_image_to_fit()
            self._upgrade_display_quality_on_fit_if_needed()
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update_status_bar()
        self.setFocus()

    def center_image_in_scroll_area(self):
        """Center the zoomed image in the scroll area"""
        if not self.current_pixmap:
            return

        # Get viewport size (actual visible area)
        viewport_size = self.scroll_area.viewport().size()
        image_size = self.current_pixmap.size()

        # Calculate center position with proper rounding
        center_x = max(0, (image_size.width() - viewport_size.width()) // 2)
        center_y = max(0, (image_size.height() - viewport_size.height()) // 2)

        # Set scroll position to center the image
        self.scroll_area.horizontalScrollBar().setValue(center_x)
        self.scroll_area.verticalScrollBar().setValue(center_y)

    def _active_vertical_scrollbar(self):
        if getattr(self, "view_mode", "") == "gallery":
            gs = getattr(self, "gallery_scroll", None)
            if gs is not None:
                return gs.verticalScrollBar()
        if hasattr(self, "scroll_area"):
            return self.scroll_area.verticalScrollBar()
        return None

    def _visible_vertical_scrollbar_width(self) -> int:
        sb = self._active_vertical_scrollbar()
        if sb is not None and sb.isVisible():
            return sb.width()
        return 0

    def _scrollbar_blocks_resize_at(self, pos: QPoint) -> bool:
        gutter = self._gallery_frame_right_gutter()
        if pos.x() >= self.width() - max(gutter, self._frame_resize_border()):
            return False
        sb_w = self._visible_vertical_scrollbar_width()
        if sb_w <= 0:
            return False
        status_bar_height = self.status_bar.height() if hasattr(self, "status_bar") else 0
        if pos.y() >= self.height() - status_bar_height:
            return False
        return pos.x() >= self.width() - sb_w

    def mousePressEvent(self, event):
        """Handle mouse press for window resizing"""
        if self._begin_resize_if_edge(event, event.position().toPoint()):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop window resizing"""
        try:
            if self._finish_resize_drag():
                event.accept()
                return
            super().mouseReleaseEvent(event)
        except Exception as e:
            safe_print(f"mouseReleaseEvent error (ignored): {e}")
            self._resize_edge_active = None
            self._is_resizing = False
            try:
                self.unsetCursor()
                self.releaseMouse()
            except Exception:
                pass
            event.accept()

    def _can_pan(self):
        # Only allow panning if the image is larger than the viewport
        if not self.current_pixmap:
            return False
        pixmap_size = self.image_label.pixmap().size()
        viewport_size = self.scroll_area.viewport().size()
        return pixmap_size.width() > viewport_size.width() or pixmap_size.height() > viewport_size.height()
