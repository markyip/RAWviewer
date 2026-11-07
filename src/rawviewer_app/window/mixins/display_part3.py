"""display mixin, part 3 of 6 (21 methods, 830 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtGui import QCursor, QImage, QPixmap

from core import theme
from core.config import _norm_path

from .. import lazy
from ..lazy import Priority, image_covers_sensor_resolution, np, qta
from ..qt_helpers import _qta_icon

lazy.register(sys.modules[__name__])

logger = logging.getLogger(__name__)


class DisplayPart3:
    def _show_single_view(self):
        """Show single image view"""
        import logging
        import os
        import time
        logger = logging.getLogger(__name__)
        if getattr(self, "view_mode", "single") == "gallery":
            logger.debug("[VIEW_MODE] _show_single_view skipped; view_mode is gallery")
            return
        start_time = time.time()
        logger.info(f"[VIEW_MODE] ========== _show_single_view() STARTED at {start_time} ==========")
        if hasattr(self.image_manager, "set_gallery_view_active"):
            self.image_manager.set_gallery_view_active(False)
        self._suppress_single_manager_callbacks = False
        # Gallery entry aborts/defers CuPy import; resume once back in single view.
        with contextlib.suppress(Exception):
            self._schedule_deferred_gpu_backend_import(reason="fallback")
        try:
            from PyQt6.QtCore import QTimer
            self._pause_semantic_indexing_deferred()
            if hasattr(self, "_resume_indexing_timer") and self._resume_indexing_timer:
                self._resume_indexing_timer.stop()

            self._resume_indexing_timer = QTimer(self)
            self._resume_indexing_timer.setSingleShot(True)
            self._resume_indexing_timer.timeout.connect(self._resume_indexing_if_single_view)
            self._resume_indexing_timer.start(5000)
        except Exception:
            pass

        self._end_gallery_indexing_deferral()
        if hasattr(self, "_idle_display_prefetch_timer") and self._idle_display_prefetch_timer:
            self._idle_display_prefetch_timer.stop()
        self._pause_prefetch_for_navigation()
        # Stop all gallery background loading when switching to single view
        if hasattr(self, 'gallery_justified') and self.gallery_justified:
            gj = self.gallery_justified
            gj._background_loading_active = False
            if hasattr(gj, "_stop_layout_rebuild_timers"):
                gj._stop_layout_rebuild_timers()
            if hasattr(gj, "cancel_background_loading"):
                gj.cancel_background_loading()
            elif hasattr(gj, "suspend_thumbnail_display"):
                gj.suspend_thumbnail_display()
            # Cancel any pending background loads
            if hasattr(self.gallery_justified, '_load_queue'):
                self.gallery_justified._load_queue.clear()
            # Drop queued gallery thumbnail decodes (all CURRENT priority) so the
            # image being opened is not stuck behind dozens of them in the worker
            # pool. Protect the foreground file, whose decode may already be in
            # flight (fast-open) or about to be requested in Step 4.
            try:
                mgr = getattr(self, "image_manager", None)
                if mgr is not None:
                    if hasattr(mgr, "cancel_gallery_tasks"):
                        mgr.cancel_gallery_tasks()
                    elif hasattr(mgr, "cancel_all_tasks_except"):
                        mgr.cancel_all_tasks_except(getattr(self, "current_file_path", None))
                if hasattr(self.gallery_justified, "_active_tasks"):
                    keep = getattr(self, "current_file_path", None)
                    for p in list(self.gallery_justified._active_tasks.keys()):
                        if p != keep:
                            self.gallery_justified._active_tasks.pop(p, None)
            except Exception:
                pass
            logger.debug("[VIEW_MODE] Stopped all gallery background loading")

        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()

        # Step 1: Hide gallery widget
        hide_start = time.time()
        if hasattr(self, 'gallery_widget') and self.gallery_widget:
            self.gallery_widget.hide()
            self._position_gallery_frame_grip()
        if hasattr(self, "compare_widget") and self.compare_widget:
            self.compare_widget.hide()
        hide_time = time.time() - hide_start
        logger.info(f"[VIEW_MODE] Step 1: Gallery widget hidden (elapsed: {hide_time:.3f}s)")

        # Step 2: Show single-image area (scroll + histogram)
        show_start = time.time()
        if hasattr(self, 'single_view_container') and self.single_view_container:
            self.single_view_container.show()
            if hasattr(self.single_view_container, "_layout_filmstrip"):
                self.single_view_container._layout_filmstrip()
            self._sync_gpu_single_view_visibility()
        else:
            self.scroll_area.show()
        show_time = time.time() - show_start
        logger.info(f"[VIEW_MODE] Step 2: Single view container shown (elapsed: {show_time:.3f}s)")

        # Step 3: Show UI elements
        ui_start = time.time()
        # Show status bar footer (metadata and image counter)
        if hasattr(self, 'status_bar'):
            self.status_bar.show()
        if hasattr(self, 'status_counter_label'):
            self._refresh_image_counter()

        # In single view mode: hide sort button and size slider, show Gallery button only if EXIF sort is ready
        if hasattr(self, 'sort_toggle_button'):
            self.sort_toggle_button.hide()
        if hasattr(self, 'size_slider'):
            self.size_slider.hide()

        if hasattr(self, 'view_mode_button'):
            self.view_mode_button.setVisible(self._is_exif_sort_ready())
        if hasattr(self, "search_bottom_button"):
            self.search_bottom_button.setVisible(bool(self.image_files) and self._is_exif_sort_ready())
        # Update icon if using qtawesome
        if qta is not None:
            try:
                gallery_icon = _qta_icon('fa5s.th', color=theme.INK_MUTED)
                self.view_mode_button.setIcon(gallery_icon)
                self.view_mode_button.setIconSize(QSize(20, 20))
                self.view_mode_button.setText("")  # Clear text if using icon
            except Exception:
                self.view_mode_button.setText("Gallery")
        else:
            self.view_mode_button.setText("Gallery")
        self._schedule_search_expand_overlay_sync()
        self._sync_compare_button_visibility()

        ui_time = time.time() - ui_start
        logger.info(f"[VIEW_MODE] Step 3: UI elements shown (elapsed: {ui_time:.3f}s)")

        # Step 4: Reload current image if available
        if self.current_file_path:
            load_start = time.time()
            path = self.current_file_path
            logger.info(f"[VIEW_MODE] Step 4: Starting image reload: {os.path.basename(path)}")

            last_path = getattr(self, "_last_loaded_path", None)
            displayed_path = getattr(self, "_displayed_content_path", None)
            from_gallery = getattr(self, "_loading_from_gallery", False)
            path_changed = (
                _norm_path(path) != _norm_path(last_path or "")
                or _norm_path(path) != _norm_path(displayed_path or "")
            )

            gallery_instant = False
            if path_changed or from_gallery:
                if self._animated_playback_active(path):
                    logger.info(
                        "[VIEW_MODE] Keeping active GIF/WebP playback for %s",
                        os.path.basename(path),
                    )
                elif self._paint_instant_preview_for_path(path, prefer_gallery=from_gallery):
                    gallery_instant = getattr(self, "_gallery_instant_display_quality", False)
                elif path_changed:
                    self._clear_stale_single_view_image()
                if not from_gallery and hasattr(self, "loading_overlay") and not gallery_instant:
                    self.loading_overlay.show_loading("Loading Image...")

            if (
                from_gallery
                and gallery_instant
                and _norm_path(getattr(self, "_displayed_content_path", "")) == _norm_path(path)
                and self._single_view_pixels_on_screen(path)
            ):
                logger.info(
                    "[VIEW_MODE] Gallery→single: display-quality preview already painted, "
                    "skipping load_raw_image"
                )
                self._finish_gallery_to_single_without_redisplay(path)
                self.update_status_bar()
            else:
                is_already_loaded = (
                    not path_changed
                    and not from_gallery
                    and _norm_path(last_path or "") == _norm_path(path)
                    and (
                        (
                            getattr(self, "_base_display_pixmap", None) is not None
                            and not self._base_display_pixmap.isNull()
                        )
                        or (
                            getattr(self, "current_pixmap", None) is not None
                            and not self.current_pixmap.isNull()
                        )
                    )
                )

                if is_already_loaded:
                    logger.info("[VIEW_MODE] Image already in memory, skipping reload")
                    base = getattr(self, "_base_display_pixmap", None)
                    if base is not None and not base.isNull():
                        self.display_pixmap(base)
                    elif (
                        getattr(self, "current_pixmap", None) is not None
                        and not self.current_pixmap.isNull()
                        and not self._get_visual_rotation_degrees(self.current_file_path)
                    ):
                        self.display_pixmap(self.current_pixmap)
                    else:
                        self.load_raw_image(self.current_file_path)
                    if hasattr(self, "loading_overlay"):
                        self.loading_overlay.hide_loading()
                    logger.info(
                        "[VIEW_MODE] Updating status bar to ensure metadata is displayed"
                    )
                    self.update_status_bar()
                else:
                    cached_pixmap = None
                    if not from_gallery and not path_changed:
                        try:
                            if hasattr(self, "image_cache"):
                                cached_pixmap = self.image_cache.get_pixmap(path)
                                if cached_pixmap and not cached_pixmap.isNull():
                                    logger.info(
                                        "[VIEW_MODE] Using cached pixmap from image cache "
                                        "for instant display"
                                    )
                                    # Cached pixmaps already have orientation applied, do not re-apply
                                    self._orientation_already_applied = True
                                    self.display_pixmap(cached_pixmap)
                                    if hasattr(self, "loading_overlay"):
                                        self.loading_overlay.hide_loading()
                        except Exception as e:
                            logger.debug(f"[VIEW_MODE] Error using cached pixmap: {e}")

                    if getattr(self, "_skip_next_single_view_load", False):
                        self._skip_next_single_view_load = False
                        logger.info(
                            "[VIEW_MODE] Skipping load_raw_image for %s "
                            "(fast-open decode already started)",
                            os.path.basename(path),
                        )
                    elif (
                        not from_gallery
                        and _norm_path(path)
                        == _norm_path(
                            getattr(self, "_single_view_load_request_path", "") or ""
                        )
                        and getattr(self, "_single_view_first_render_logged", False)
                        and self._single_view_pixels_on_screen(path)
                    ):
                        logger.info(
                            "[VIEW_MODE] Skipping load_raw_image for %s "
                            "(first-render already on screen)",
                            os.path.basename(path),
                        )
                        if hasattr(self, "loading_overlay"):
                            self.loading_overlay.hide_loading()
                        self.update_status_bar()
                    else:
                        logger.info(
                            f"[VIEW_MODE] Triggering load_raw_image for {os.path.basename(path)}"
                        )
                        self.load_raw_image(path)
            load_time = time.time() - load_start
            logger.info(f"[VIEW_MODE] Step 4: Image reload completed (elapsed: {load_time:.3f}s)")
        else:
            # Update status bar to show metadata even if no image is loaded
            self.update_status_bar()
            logger.info("[VIEW_MODE] Step 4: No image to reload, status bar updated")

        self._sync_single_image_histogram()
        self._sync_filmstrip_to_folder()

        total_time = time.time() - start_time
        logger.info("[VIEW_MODE] ========== TIMING BREAKDOWN ==========")
        # GALLERY FUNCTIONALITY COMMENTED OUT
        # if self.gallery_widget:
        #     logger.info(f"[VIEW_MODE] Hide gallery widget: {hide_time:.3f}s")
        logger.info(f"[VIEW_MODE] Show scroll area: {show_time:.3f}s")
        logger.info(f"[VIEW_MODE] Show UI elements: {ui_time:.3f}s")
        if self.current_file_path:
            logger.info(f"[VIEW_MODE] Image reload: {load_time:.3f}s")
        self.setFocus()
        QTimer.singleShot(0, self._restore_keyboard_focus)
        logger.info(f"[VIEW_MODE] ========== SINGLE VIEW RENDERING COMPLETED in {total_time:.3f}s ==========")

    def _numpy_to_qpixmap(self, numpy_array):
        """
        Convert numpy array to QPixmap safely.
        Handles all edge cases and ensures correct format.
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            if numpy_array is None:
                return None

            # Get shape
            shape = numpy_array.shape
            if len(shape) < 2:
                logger.warning(f"[GALLERY] Invalid numpy array shape: {shape}, expected at least 2D (H, W)")
                return None

            height, width = shape[0], shape[1]
            channels = shape[2] if len(shape) > 2 else 1

            if channels == 1:
                plane = numpy_array.reshape(height, width)
                numpy_array = np.stack([plane, plane, plane], axis=-1)
                channels = 3

            if channels not in [3, 4]:
                logger.warning(f"[GALLERY] Unsupported channel count: {channels}, expected 1, 3, or 4")
                return None

            if width <= 0 or height <= 0:
                logger.warning(f"[GALLERY] Invalid dimensions: {width}x{height}")
                return None

            # Ensure contiguous and uint8
            if not numpy_array.flags['C_CONTIGUOUS']:
                numpy_array = np.ascontiguousarray(numpy_array)

            if numpy_array.dtype == np.uint16:
                numpy_array = (numpy_array / 257.0).astype(np.uint8)
            elif np.issubdtype(numpy_array.dtype, np.floating):
                from raw_adjustments.tone_recovery import _encode_srgb8

                numpy_array = _encode_srgb8(
                    np.clip(numpy_array.astype(np.float32), 0.0, None)
                )
            elif numpy_array.dtype != np.uint8:
                numpy_array = np.clip(numpy_array, 0, 255).astype(np.uint8)

            # Calculate bytes per line (must be exact, no padding)
            bytes_per_line = channels * width

            # Wrap the numpy buffer directly -- avoids an extra full-frame
            # tobytes() allocation. QImage does not own the memory, so
            # .copy() once into Qt-owned storage before building the pixmap.
            qimage = QImage(
                numpy_array.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            ).copy()

            if qimage.isNull():
                logger.warning(f"[GALLERY] Failed to create QImage from numpy array {width}x{height}")
                return None

            # Convert to QPixmap
            pixmap = QPixmap.fromImage(qimage)

            if pixmap.isNull():
                logger.warning("[GALLERY] Failed to create QPixmap from QImage")
                return None

            return pixmap

        except Exception as e:
            logger.warning(f"[GALLERY] Error converting numpy to QPixmap: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def _create_macos_save_panel(self):
        """Return an NSSavePanel instance (can fail when Qt owns NSApplication).

        See ``_create_macos_open_panel`` for why this fails fast on a nil
        result instead of retrying with forced foreground activation --
        the retry was observed to hang indefinitely rather than fail fast.
        """
        import objc

        ns_save_panel = objc.lookUpClass("NSSavePanel")
        panel = ns_save_panel.alloc().init()
        if panel is None:
            panel = ns_save_panel.savePanel()
        if panel is None:
            raise RuntimeError("NSSavePanel could not be created")
        with contextlib.suppress(Exception):
            panel.setCanCreateDirectories_(True)
        return panel

    def image_mouse_press_event(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self.current_pixmap:
            if not self.fit_to_window and self._can_pan():
                self._stop_slideshow()
                self.panning = True
                self.last_pan_point = ev.globalPosition().toPoint()
                self.start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                self.start_scroll_y = self.scroll_area.verticalScrollBar().value()
                self.image_label.setCursor(
                    QCursor(Qt.CursorShape.ClosedHandCursor))
            self.setFocus()

    def image_mouse_move_event(self, ev):
        if self.panning and self.current_pixmap and self._can_pan():
            current_pos = ev.globalPosition().toPoint()
            delta = current_pos - self.last_pan_point
            self.last_pan_point = current_pos
            h_scroll = self.scroll_area.horizontalScrollBar()
            v_scroll = self.scroll_area.verticalScrollBar()
            new_x = h_scroll.value() - delta.x()
            new_y = v_scroll.value() - delta.y()
            h_scroll.setValue(max(0, min(new_x, h_scroll.maximum())))
            v_scroll.setValue(max(0, min(new_y, v_scroll.maximum())))
        elif self.current_pixmap and not self.fit_to_window and self._can_pan():
            self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def zoom_to_point(self):
        if not self.current_pixmap:
            return
        self._set_single_view_pixmap(self.current_pixmap)
        self.image_label.adjustSize()  # Ensure label is resized to pixmap
        self.scroll_area.widget().adjustSize()  # Force scroll area to update
        self.scroll_area.updateGeometry()
        self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        # Actually center the view on the zoom point
        # Defer the scroll update to the next event loop iteration to ensure
        # the scroll area layout is fully updated and scrollbars have correct maximums.
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._complete_zoom_to_point)

    def _complete_zoom_to_point(self):
        if self.zoom_center_point is not None:
            viewport_size = self.scroll_area.viewport().size()
            image_size = self.current_pixmap.size()
            # Center the zoom point in the viewport
            target_scroll_x = self.zoom_center_point.x() - (viewport_size.width() // 2)
            target_scroll_y = self.zoom_center_point.y() - (viewport_size.height() // 2)
            max_scroll_x = max(0, image_size.width() - viewport_size.width())
            max_scroll_y = max(0, image_size.height() - viewport_size.height())
            final_scroll_x = max(0, min(target_scroll_x, max_scroll_x))
            final_scroll_y = max(0, min(target_scroll_y, max_scroll_y))
            self.scroll_area.horizontalScrollBar().setValue(final_scroll_x)
            self.scroll_area.verticalScrollBar().setValue(final_scroll_y)
        else:
            self.center_image_in_scroll_area()

    def _zoom_anchor_for_navigation_restore(self):
        """Image-space point to preserve framing when navigating (pinch may leave zoom_center_point unset)."""
        if self.zoom_center_point is not None:
            return QPoint(self.zoom_center_point)
        pm = getattr(self, "current_pixmap", None)
        if pm is None or pm.isNull():
            return QPoint(0, 0)
        if not hasattr(self, "scroll_area") or self.scroll_area is None or not hasattr(self, "image_label"):
            return QPoint(max(0, pm.width() // 2), max(0, pm.height() // 2))
        try:
            vp = self.scroll_area.viewport()
            ctr = vp.rect().center()
            gp = vp.mapToGlobal(ctr)
            lp = self.image_label.mapFromGlobal(gp)
            return self.convert_widget_to_image_coords(lp)
        except Exception:
            return QPoint(max(0, pm.width() // 2), max(0, pm.height() // 2))

    def _finish_nav_zoom_preserve(self) -> None:
        if getattr(self, "_preserve_nav_zoom_active", False):
            import sys as _sys

            logging.getLogger(__name__).debug(
                "[ZOOMNAV] preserve cleared by %s",
                _sys._getframe(1).f_code.co_name,
            )
        self._preserve_nav_zoom_active = False

    def apply_pan_offset(self):
        # Deprecated: direct panning is now handled in mouse events
        pass

    def apply_zoom_and_pan_simple(self):
        """Simple zoom and pan that centers on the clicked point"""
        if not self.current_pixmap:
            return

        # Set the image at 100% zoom
        self._set_single_view_pixmap(self.current_pixmap)

        # If we have a zoom center point, center the scroll area on it
        if self.zoom_center_point is not None:
            # Calculate the position in the full-size image
            # The zoom_center_point is in scroll area coordinates
            # We need to convert it to full-size image coordinates

            # Get the scaling factor from the fit-to-window to full size
            scroll_area_size = self.scroll_area.size()
            image_size = self.current_pixmap.size()

            # Calculate what proportion of the scroll area the click was at
            click_x_ratio = self.zoom_center_point.x() / scroll_area_size.width()
            click_y_ratio = self.zoom_center_point.y() / scroll_area_size.height()

            # Calculate the corresponding position in the full-size image
            target_x = int(click_x_ratio * image_size.width())
            target_y = int(click_y_ratio * image_size.height())

            # Center the scroll area on this point
            scroll_x = target_x - round(scroll_area_size.width() / 2)
            scroll_y = target_y - round(scroll_area_size.height() / 2)

            # Clamp to valid range
            max_scroll_x = max(0, image_size.width() -
                               scroll_area_size.width())
            max_scroll_y = max(0, image_size.height() -
                               scroll_area_size.height())

            scroll_x = max(0, min(scroll_x, max_scroll_x))
            scroll_y = max(0, min(scroll_y, max_scroll_y))

            # Set scroll position
            self.scroll_area.horizontalScrollBar().setValue(scroll_x)
            self.scroll_area.verticalScrollBar().setValue(scroll_y)
        else:
            # Center the image
            self.center_image_in_scroll_area()

        # Update cursor
        self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def apply_zoom_and_pan(self):
        """Apply current zoom level and pan offset to the image"""
        if self._raw_recovery_preview_active():
            self._notify_raw_recovery_zoom_blocked()
            return
        if not self.current_pixmap:
            return

        # Calculate scaled size
        original_size = self.current_pixmap.size()
        scaled_width = int(original_size.width() * self.current_zoom_level)
        scaled_height = int(original_size.height() * self.current_zoom_level)

        # Scale the pixmap
        # Ensure dimensions are at least 1px to prevent crash
        safe_width = max(1, scaled_width)
        safe_height = max(1, scaled_height)

        scaled_pixmap = self.current_pixmap.scaled(
            safe_width, safe_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Set the scaled pixmap
        self._set_single_view_pixmap(scaled_pixmap)

        # Apply zoom center point and panning
        if self.zoom_center_point is not None:
            # Check if we have restored scroll positions from navigation
            if hasattr(self, '_restore_start_scroll_x') and self._restore_start_scroll_x is not None:
                # Use the restored scroll positions directly
                scroll_x = self._restore_start_scroll_x
                scroll_y = self._restore_start_scroll_y

                # Clamp scroll positions to valid range
                viewport_size = self.scroll_area.viewport().size()
                image_size = scaled_pixmap.size()
                max_scroll_x = max(0, image_size.width() - viewport_size.width())
                max_scroll_y = max(0, image_size.height() - viewport_size.height())

                scroll_x = max(0, min(scroll_x, max_scroll_x))
                scroll_y = max(0, min(scroll_y, max_scroll_y))

                # Set scroll position
                self.scroll_area.horizontalScrollBar().setValue(scroll_x)
                self.scroll_area.verticalScrollBar().setValue(scroll_y)

                # Clear the restored scroll positions after use
                self._restore_start_scroll_x = None
                self._restore_start_scroll_y = None
            else:
                # Calculate the position to center the zoom point
                viewport_size = self.scroll_area.viewport().size()
                image_size = scaled_pixmap.size()

                # Convert image coordinates to scaled coordinates
                center_x = int(self.zoom_center_point.x()
                               * self.current_zoom_level)
                center_y = int(self.zoom_center_point.y()
                               * self.current_zoom_level)

                # Calculate scroll position
                if hasattr(self, 'zoom_cursor_offset') and self.zoom_cursor_offset:
                    scroll_x = center_x - self.zoom_cursor_offset.x()
                    scroll_y = center_y - self.zoom_cursor_offset.y()
                    # Clear it after use so it doesn't affect other zoom actions (like double-click)
                    self.zoom_cursor_offset = None
                else:
                    scroll_x = center_x - round(viewport_size.width() / 2)
                    scroll_y = center_y - round(viewport_size.height() / 2)

                # Clamp scroll positions to valid range
                max_scroll_x = max(0, image_size.width() - viewport_size.width())
                max_scroll_y = max(0, image_size.height() - viewport_size.height())

                scroll_x = max(0, min(scroll_x, max_scroll_x))
                scroll_y = max(0, min(scroll_y, max_scroll_y))

                # Set scroll position
                self.scroll_area.horizontalScrollBar().setValue(scroll_x)
                self.scroll_area.verticalScrollBar().setValue(scroll_y)
        else:
            # Center the image if no zoom center point is set
            self.center_image_in_scroll_area()

        # Update cursor
        if not self.fit_to_window:
            self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _fit_zoom_level(self) -> float:
        """``current_zoom_level`` at which the pixmap fits the scroll area (matches scale_image_to_fit)."""
        pm = getattr(self, "current_pixmap", None)
        sa = getattr(self, "scroll_area", None)
        if pm is None or pm.isNull() or sa is None:
            return 1.0
        ow, oh = pm.width(), pm.height()
        if ow <= 0 or oh <= 0:
            return 1.0
        margin = 20
        max_w = max(1, sa.size().width() - margin)
        max_h = max(1, sa.size().height() - margin)
        return min(max_w / ow, max_h / oh)

    def _max_smooth_zoom_level(self) -> float:
        """Ceiling for trackpad pinch / Ctrl+wheel zoom relative to ``current_pixmap`` pixels."""
        cap = self._MAX_PINCH_ZOOM_LEVEL
        if not getattr(self, "_is_half_size_displayed", False):
            return cap
        pm = getattr(self, "current_pixmap", None)
        if pm is None or pm.isNull():
            return max(1.0, cap)
        preview_long = max(pm.width(), pm.height())
        if preview_long <= 0:
            return max(1.0, cap)
        try:
            exif = self.image_cache.get_exif(self.current_file_path)
            if exif:
                ow, oh = exif.get("original_width"), exif.get("original_height")
                if ow and oh:
                    sensor_long = max(int(ow), int(oh))
                    if sensor_long > preview_long * 1.08:
                        r = sensor_long / float(preview_long)
                        return max(1.0, min(r, cap))
        except Exception:
            pass
        return max(1.0, cap)

    def _finish_pending_zoom_after_full_load(self, file_path: str) -> None:
        """Apply a deferred Space/double-click zoom once full-res work completes or stalls."""
        if getattr(self, "_workflow_toggle_fit_until_display", False):
            return
        if _norm_path(file_path or "") != _norm_path(
            getattr(self, "current_file_path", "") or ""
        ):
            return
        gv = getattr(self, "gpu_view", None)
        pm = getattr(self, "current_pixmap", None)
        if gv is not None and pm is not None and not pm.isNull():
            if self._pending_zoom_ready_for_pixmap(pm, file_path) and self._apply_pending_gpu_point_zoom_in_place(pm, gv):
                    return
            if getattr(self, "_focus_zoom_anchor_active", False) or getattr(
                self, "_pending_zoom_focus_subject", False
            ):
                if self._apply_pending_gpu_point_zoom_in_place(pm, gv):
                    return
                if self._single_view_is_fit_mode() or getattr(
                    self, "_pending_zoom", False
                ):
                    target = self._focus_zoom_target_for_pixmap(pm)
                    if target is not None:
                        gv.set_pixmap_zoomed_at(
                            pm,
                            float(target.x()),
                            float(target.y()),
                            scale=1.0,
                        )
                        self.zoom_center_point = target
                        self.fit_to_window = False
                        self.current_zoom_level = float(gv.current_scale())
                self._finish_focus_zoom_anchor_if_sensor_res()
                return
        if getattr(self, "_pending_zoom", False) or (
            gv is not None and getattr(gv, "_zoom_intent_100", False)
        ):
            self._gpu_zoom_in_to_point_finish()

    def _sync_displayed_half_size_flag(self, file_path: str | None = None) -> None:
        """Keep ``_is_half_size_displayed`` aligned with buffer vs sensor size, not file type."""
        fp = file_path or getattr(self, "current_file_path", None)
        pm = getattr(self, "current_pixmap", None)
        if not fp or pm is None or pm.isNull():
            return
        self._is_half_size_displayed = self._needs_full_resolution_upgrade(fp)

    def _try_sync_full_before_zoom(self) -> bool:
        """Paint cached full-res before 100% zoom so the user never sees a soft preview upscale."""
        import logging

        logger = logging.getLogger(__name__)
        if self._raw_recovery_preview_active():
            return True
        if not self._needs_full_resolution_upgrade():
            return True
        fp = getattr(self, "current_file_path", None)
        if not fp:
            return False
        cached_full = self.image_cache.get_full_image(fp)
        if cached_full is None:
            return False
        try:
            if hasattr(cached_full, "shape"):
                ch, cw = cached_full.shape[0], cached_full.shape[1]
            else:
                return False
            exif = self.image_cache.get_exif(fp)
            if not image_covers_sensor_resolution(cw, ch, exif):
                return False
        except Exception:
            return False

        old_size = self.current_pixmap.size() if self.current_pixmap else None
        self._pending_zoom = True
        self._pending_zoom_center = getattr(self, "zoom_center_point", None)
        self._pending_zoom_thumbnail_size = old_size
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv._zoom_intent_100 = True
        self._orientation_already_applied = True
        logger.info(
            "[ZOOM] Upgrading to cached full resolution before 100%% zoom (%s)",
            os.path.basename(fp),
        )
        self.display_numpy_image(cached_full)
        self._is_half_size_displayed = False
        self._smooth_zoom_full_request_sent = False
        if old_size is not None and self.current_pixmap is not None and getattr(self, "zoom_center_point", None) is not None:
            scale_x = (
                self.current_pixmap.width() / old_size.width()
                if old_size.width() > 0
                else 1.0
            )
            scale_y = (
                self.current_pixmap.height() / old_size.height()
                if old_size.height() > 0
                else 1.0
            )
            self.zoom_center_point = QPoint(
                int(self.zoom_center_point.x() * scale_x),
                int(self.zoom_center_point.y() * scale_y),
            )
        return True

    def _defer_zoom_until_full_or_apply(self) -> bool:
        """If zoom needs full-res, load it first; return True when zoom must wait."""
        self._allow_full_decode_for_user_zoom()
        if not self._needs_full_resolution_upgrade():
            return False
        if self._try_sync_full_before_zoom():
            return True
        if getattr(self, "_full_resolution_loading", False):
            self._pending_zoom = True
            self._pending_zoom_center = getattr(self, "zoom_center_point", None)
            self._pending_zoom_thumbnail_size = (
                self.current_pixmap.size() if self.current_pixmap else None
            )
            gv = getattr(self, "gpu_view", None)
            if gv is not None:
                gv._zoom_intent_100 = True
            return True
        self._load_full_resolution_on_demand()
        self._pending_zoom = True
        self._pending_zoom_center = getattr(self, "zoom_center_point", None)
        self._pending_zoom_thumbnail_size = (
            self.current_pixmap.size() if self.current_pixmap else None
        )
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv._zoom_intent_100 = True
        self.status_bar.showMessage("Loading full resolution for 100% zoom...")
        return True

    def _is_single_view_navigation_step(self, requested_file_path):
        """Check if this is a single-view navigation step"""
        _prev_fp = getattr(self, "current_file_path", None)
        _same_path_reload = _prev_fp and _norm_path(_prev_fp) == _norm_path(requested_file_path)
        _is_single_view_step = (
            getattr(self, "view_mode", "single") == "single"
            and not _same_path_reload
            and bool(_prev_fp)
            and _norm_path(_prev_fp) != _norm_path(requested_file_path)
        )
        return _is_single_view_step

    def _display_cached_pixmap(self, requested_file_path, cached_pixmap):
        """Display a cached pixmap (non-RAW files only)"""
        os.path.basename(requested_file_path)
        logger.info("[LOAD] Displaying cached pixmap")
        display_start = time.time()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.display_pixmap(cached_pixmap)
        self._on_single_view_content_displayed()
        display_time = time.time() - display_start
        logger.info(f"[LOAD] Cached pixmap displayed in {display_time:.3f}s")

        self.setFocus()
        self.schedule_save_session_state()

        # Update index for preloading
        try:
            if self.image_files and requested_file_path in self.image_files:
                self.current_file_index = self.image_files.index(requested_file_path)
        except ValueError:
            pass
        self._start_preloading()

        # Ensure EXIF is loaded even on cache hit
        if not self.image_cache.get_exif(requested_file_path):
            logger.info("[LOAD] Cache hit for pixmap, but missing EXIF. Requesting EXIF load.")
            self.image_manager.load_image(
                requested_file_path,
                priority=Priority.BACKGROUND,
                cancel_existing=False,
                use_full_resolution=False,
                stages={"exif"},
            )

    def _should_use_raw_edr_display(self, file_path: str | None = None) -> bool:
        """RAW EDR gate used at the actual decode-dispatch point.

        RAW EDR re-decodes with LibRaw highlight reconstruction (~12x a
        plain decode; see ``use_raw_edr_display()`` docstring) via its own
        separate rawpy call, so triggering it on every navigation frame
        during rapid culling burns that cost on images the user never
        pauses on. Eligibility is eager only when already zoomed in (the
        user is examining this exact image); otherwise it additionally
        requires this file to have been "armed" by
        ``_maybe_arm_raw_edr_after_idle`` after a settle period with no
        further navigation -- mirrors ``_schedule_idle_full_decode_after_
        nav_preview``'s treatment of the sensor-resolution tier.
        """
        path = file_path or getattr(self, "current_file_path", None)
        if not self._raw_edr_eligible_ignoring_idle(path):
            return False
        if self._preserve_zoom_navigation_active() or self._single_view_is_zoomed_in():
            return True
        return getattr(self, "_raw_edr_armed_key", None) == _norm_path(path or "")
