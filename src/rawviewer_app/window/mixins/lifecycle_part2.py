"""lifecycle mixin, part 2 of 2 (5 methods, 591 lines)."""
from __future__ import annotations

import contextlib
import sys

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtWidgets import QApplication

from core.config import _env_int, safe_print

from .. import lazy
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class LifecyclePart2:
    def resizeEvent(self, event):
        """Handle window resize events"""
        import logging
        logger = logging.getLogger(__name__)

        old_size = self.size()
        new_size = event.size()

        # Call base class implementation
        super().resizeEvent(event)

        # Update loading overlay geometry to cover the window
        if hasattr(self, 'loading_overlay') and self.loading_overlay:
            self.loading_overlay.setGeometry(self.rect())
        if getattr(self, "edit_loading_overlay", None) is not None:
            self.edit_loading_overlay.setGeometry(self.rect())
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "_layout_filmstrip"):
            container._layout_filmstrip()
            if hasattr(container, "_raise_filmstrip_chrome"):
                container._raise_filmstrip_chrome()

        # macOS: fullscreen/maximize transitions can trigger re-entrant Qt event delivery.
        # Avoid any expensive work (and especially processEvents() cascades) while the
        # window state is actively changing.
        if sys.platform == "darwin" and getattr(self, "_handling_window_state_change", False):
            return

        # Skip logging and scaling during active resize to prevent intermediate updates
        if getattr(self, '_is_resizing', False):
            grip = getattr(self, "_resize_grip", None)
            if grip is not None:
                grip.hide()
            return

        # Log window size change
        logger.info(f"[WINDOW_RESIZE] Window size changed: {old_size.width()}x{old_size.height()} -> {new_size.width()}x{new_size.height()}")
        safe_print(f"[WINDOW_RESIZE] Window: {old_size.width()}x{old_size.height()} -> {new_size.width()}x{new_size.height()}")

        # Log scroll area (viewing section) size
        if hasattr(self, 'scroll_area') and self.scroll_area:
            scroll_size = self.scroll_area.size()
            logger.info(f"[WINDOW_RESIZE] Scroll area size: {scroll_size.width()}x{scroll_size.height()}")
            safe_print(f"[WINDOW_RESIZE] Scroll area: {scroll_size.width()}x{scroll_size.height()}")

            # Log viewport size
            if hasattr(self.scroll_area, 'viewport') and self.scroll_area.viewport():
                viewport_size = self.scroll_area.viewport().size()
                logger.info(f"[WINDOW_RESIZE] Viewport size: {viewport_size.width()}x{viewport_size.height()}")
                safe_print(f"[WINDOW_RESIZE] Viewport: {viewport_size.width()}x{viewport_size.height()}")

        # Log current image resolution if available
        if hasattr(self, 'current_pixmap') and self.current_pixmap and not self.current_pixmap.isNull():
            pixmap_size = self.current_pixmap.size()
            logger.info(f"[WINDOW_RESIZE] Current image resolution: {pixmap_size.width()}x{pixmap_size.height()}")
            safe_print(f"[WINDOW_RESIZE] Image resolution: {pixmap_size.width()}x{pixmap_size.height()}")

            # Log displayed image size (scaled size)
            if hasattr(self, 'image_label') and self.image_label:
                label_pixmap = self.image_label.pixmap()
                if label_pixmap and not label_pixmap.isNull():
                    displayed_size = label_pixmap.size()
                    logger.info(f"[WINDOW_RESIZE] Displayed image size: {displayed_size.width()}x{displayed_size.height()}")
                    safe_print(f"[WINDOW_RESIZE] Displayed size: {displayed_size.width()}x{displayed_size.height()}")

        # Log gallery view size if in gallery mode
        if hasattr(self, 'view_mode') and self.view_mode == 'gallery':
            if hasattr(self, 'gallery_widget') and self.gallery_widget:
                gallery_size = self.gallery_widget.size()
                logger.info(f"[WINDOW_RESIZE] Gallery widget size: {gallery_size.width()}x{gallery_size.height()}")
                safe_print(f"[WINDOW_RESIZE] Gallery widget: {gallery_size.width()}x{gallery_size.height()}")

            if hasattr(self, 'gallery_scroll') and self.gallery_scroll:
                gallery_scroll_size = self.gallery_scroll.size()
                logger.info(f"[WINDOW_RESIZE] Gallery scroll area size: {gallery_scroll_size.width()}x{gallery_scroll_size.height()}")
                safe_print(f"[WINDOW_RESIZE] Gallery scroll: {gallery_scroll_size.width()}x{gallery_scroll_size.height()}")

                if hasattr(self.gallery_scroll, 'viewport') and self.gallery_scroll.viewport():
                    gallery_viewport_size = self.gallery_scroll.viewport().size()
                    logger.info(f"[WINDOW_RESIZE] Gallery viewport size: {gallery_viewport_size.width()}x{gallery_viewport_size.height()}")
                    safe_print(f"[WINDOW_RESIZE] Gallery viewport: {gallery_viewport_size.width()}x{gallery_viewport_size.height()}")

        # No rounded corners to update
        # Rescale image when window is resized, but only in fit-to-window mode
        if hasattr(self, "current_file_path") and self.is_animated_image(_require_str_path(self.current_file_path)):
            if hasattr(self, "_current_movie") and self._current_movie:
                self._update_gif_scaling()
            else:
                self._update_webp_scaling()
        elif self.current_pixmap and self.fit_to_window:
            self.scale_image_to_fit()
        # GALLERY FUNCTIONALITY COMMENTED OUT
        # Update gallery layout if in gallery view mode (justified layout rebuilds automatically on resize)
        # Update bottom bar responsive elements (hide/show indexing status based on width)
        if hasattr(self, "_gallery_search_status_full"):
            full = (self._gallery_search_status_full or "").strip()
            if (
                getattr(self, "_search_panel_expanded", False)
                and self._is_gallery_search_indexing_active()
            ) or full and not self._should_show_search_index_progress(full):
                self._set_gallery_search_status(full, animate=False)
            elif full:
                self._clear_stale_gallery_search_index_progress()
                self._sync_gallery_search_input_editable()
            else:
                self._set_gallery_search_status("", animate=False)
        self._schedule_search_expand_overlay_sync()
        self._position_resize_grip()
        self._position_gallery_frame_grip()

        # Refresh top metadata label elision on resize
        if hasattr(self, "current_file_path") and self.current_file_path and getattr(self, "view_mode", "single") == "single":
            self.update_status_bar()

        # Gallery child width is fixed; justified rows must rebuild when the scroll viewport grows/shrinks.
        if getattr(self, "view_mode", "") == "gallery" and not getattr(self, "_is_resizing", False):
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "force_layout_update"):
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, gj.force_layout_update)

    def mouseMoveEvent(self, event):
        """Handle mouse move for window resizing and cursor updates"""
        if self._continue_resize_drag(event):
            event.accept()
            return
        self._update_resize_edge_hover(event.position().toPoint())
        super().mouseMoveEvent(event)

    def eventFilter(self, obj, event):
        et = event.type()
        drop_targets = getattr(self, "_file_drop_targets", None)
        if drop_targets and obj in drop_targets:
            if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                if self._accept_drag_urls(event.mimeData()):
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return True
            if et == QEvent.Type.Drop:
                if self._process_dropped_urls(event.mimeData()):
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return True

        if event.type() in (QEvent.Type.MouseMove, QEvent.Type.Enter) and getattr(self, "view_mode", "single") == "single" and self._is_status_bar_chrome_widget(obj):
                container = getattr(self, "single_view_container", None)
                if container is not None:
                    container.set_filmstrip_pointer_active(False)

        if event.type() in (
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            pos = self._resize_pos_from_event(event)
            if self._allow_frame_resize_event(obj, pos) and self._handle_frame_resize_event(
                event
            ):
                return True

        if (
            event.type() == QEvent.Type.Resize
            and obj is getattr(self, "gallery_widget", None)
        ):
            self._position_gallery_frame_grip()

        if event.type() == QEvent.Type.Leave and not getattr(self, "_resize_edge_active", None) and not self._is_status_bar_chrome_widget(obj):
                self.unsetCursor()

        if event.type() == QEvent.Type.KeyPress:
            # Route shortcuts before overlay / OpenGL children consume the event.
            if self._should_route_global_shortcut(obj):
                if self._handle_app_shortcut(event):
                    return True
            # Fallback: active main window but receiver not matched above (e.g. QOpenGLWidget).
            elif not self._shortcut_blocked_by_text_input():
                app = QApplication.instance()
                if app is not None and app.activeWindow() is self and self._handle_app_shortcut(event):
                    return True

        # Film strip hot zone: pointer events hit child widgets, not the overlay itself.
        if event.type() in (
            QEvent.Type.MouseMove,
            QEvent.Type.HoverMove,
            QEvent.Type.Enter,
        ) and getattr(self, "view_mode", "single") == "single" and not getattr(self, "_adjust_overlay_visible", False):
            container = getattr(self, "single_view_container", None)
            gpu_view = getattr(self, "gpu_view", None)
            filmstrip_targets = (
                self.scroll_area.viewport(),
                self.image_label,
                self.scroll_area,
            )
            if container is not None:
                filmstrip_targets = (*filmstrip_targets, container)
            if gpu_view is not None:
                filmstrip_targets = (*filmstrip_targets, gpu_view, gpu_view.viewport())
            if container is not None and obj in filmstrip_targets:
                self._ensure_filmstrip_synced()
                container.handle_pointer_for_filmstrip(event.globalPosition())

        # Handle trackpad pinch-to-zoom on Mac
        if event.type() == QEvent.Type.NativeGesture:
            if (
                hasattr(self, "view_mode")
                and self.view_mode == "single"
                and self._raw_recovery_preview_active()
            ) and event.gestureType() in (
                Qt.NativeGestureType.ZoomNativeGesture,
                Qt.NativeGestureType.SmartZoomNativeGesture,
            ):
                self._notify_raw_recovery_zoom_blocked()
                return True
            if hasattr(self, 'view_mode') and self.view_mode == 'single' and getattr(self, 'current_pixmap', None):
                gpu_view = getattr(self, "gpu_view", None)
                if gpu_view is not None and gpu_view.has_pixmap():
                    if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                        self._stop_slideshow()
                        gpu_view.zoom_by(1.0 + event.value() * 0.5)
                        self.fit_to_window = gpu_view.is_fit_mode()
                        self.current_zoom_level = float(gpu_view.current_scale())
                        if not gpu_view.is_fit_mode():
                            self._gpu_request_full_resolution_if_needed()
                        self.update_status_bar()
                        return True
                    if event.gestureType() == Qt.NativeGestureType.SmartZoomNativeGesture:
                        self._stop_slideshow()
                        self.toggle_zoom()
                        return True
                if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                    self._stop_slideshow()
                    fit_scale = self._fit_zoom_level()

                    if self.fit_to_window:
                        self.fit_to_window = False
                        self.current_zoom_level = fit_scale

                    # Reduce sensitivity and make zoom smooth/proportional
                    self.current_zoom_level *= (1.0 + event.value() * 0.5)

                    # Prevent zooming out beyond fit-to-window scale
                    if self.current_zoom_level <= fit_scale * 1.002:
                        self.fit_to_window = True
                        self.current_zoom_level = fit_scale
                        self.scale_image_to_fit()
                        self.update_status_bar()
                        return True
                    zoom_cap = self._max_smooth_zoom_level()
                    self.current_zoom_level = max(fit_scale, min(self.current_zoom_level, zoom_cap))

                    mouse_global = event.globalPosition().toPoint()
                    self.zoom_cursor_offset = self.scroll_area.viewport().mapFromGlobal(mouse_global)
                    mouse_image = self.image_label.mapFromGlobal(mouse_global)
                    self.zoom_center_point = self.convert_widget_to_image_coords(mouse_image)

                    self.apply_zoom_and_pan()
                    if (
                        not self.fit_to_window
                        and getattr(self, "_is_half_size_displayed", False)
                        and self.current_zoom_level >= 1.0 - 1e-9
                    ):
                        self._maybe_request_full_res_for_smooth_zoom()
                    self.update_status_bar()
                    return True
                elif event.gestureType() == Qt.NativeGestureType.SmartZoomNativeGesture:
                    self._stop_slideshow()
                    # Ensure smart zoom (pinch tap) also triggers full resolution
                    self.toggle_zoom()
                    return True

        # Handle wheel events for navigation in single image view when fit-to-window
        if event.type() == QEvent.Type.Wheel and hasattr(self, 'view_mode') and self.view_mode == 'single' and hasattr(self, 'scroll_area') and obj == self.scroll_area.viewport():
                self._stop_slideshow()
                if self._filmstrip_handles_input():
                    bar = self._filmstrip_bar()
                    if bar is not None:
                        wheel_event = event
                        delta = wheel_event.angleDelta().y() or wheel_event.angleDelta().x()
                        if delta:
                            steps = delta // 120 if abs(delta) >= 120 else (1 if delta > 0 else -1)
                            bar.scroll_by_steps(steps)
                            return True
                wheel_event = event

                # Check vertical wheel (up/down scroll)
                vertical_delta = wheel_event.angleDelta().y()
                # Check horizontal wheel (left/right scroll)
                horizontal_delta = wheel_event.angleDelta().x()

                # Handle Windows trackpad pinch-to-zoom (Ctrl + Wheel)
                if wheel_event.modifiers() & Qt.KeyboardModifier.ControlModifier and getattr(self, 'current_pixmap', None):
                    if vertical_delta != 0:
                        fit_scale = self._fit_zoom_level()

                        if self.fit_to_window:
                            self.fit_to_window = False
                            self.current_zoom_level = fit_scale

                        # Standard mouse wheel delta is 120 per notch. Trackpad pinch may be continuous.
                        # Using vertical_delta / 1200.0 means 120 delta = 10% zoom.
                        self.current_zoom_level *= (1.0 + vertical_delta / 1200.0)

                        if self.current_zoom_level <= fit_scale * 1.002:
                            self.fit_to_window = True
                            self.current_zoom_level = fit_scale
                            self.scale_image_to_fit()
                            self.update_status_bar()
                            return True

                        zoom_cap = self._max_smooth_zoom_level()
                        self.current_zoom_level = max(fit_scale, min(self.current_zoom_level, zoom_cap))

                        mouse_global = wheel_event.globalPosition().toPoint()
                        self.zoom_cursor_offset = self.scroll_area.viewport().mapFromGlobal(mouse_global)
                        mouse_image = self.image_label.mapFromGlobal(mouse_global)
                        self.zoom_center_point = self.convert_widget_to_image_coords(mouse_image)

                        self.apply_zoom_and_pan()
                        if (
                            not self.fit_to_window
                            and getattr(self, "_is_half_size_displayed", False)
                            and self.current_zoom_level >= 1.0 - 1e-9
                        ):
                            self._maybe_request_full_res_for_smooth_zoom()
                        self.update_status_bar()
                    return True

                # Editor mode: block plain wheel events over image area
                if getattr(self, "_adjust_overlay_visible", False):
                    panel = getattr(self, "single_image_adjust_panel", None)
                    if (
                        panel is not None
                        and panel.dodge_burn_mode() is not None
                        and abs(vertical_delta) > 0
                    ):
                        self._on_dodge_burn_brush_size_wheel(int(vertical_delta))
                    return True

                # Browse mode: Only navigate if image is fit-to-window (not zoomed)
                if hasattr(self, 'fit_to_window') and self.fit_to_window:
                    # In fit-to-window mode: only use vertical wheel for navigation
                    # Horizontal wheel is disabled for navigation
                    if abs(vertical_delta) > 0:
                        if vertical_delta > 0:
                            # Wheel up (Qt positive delta) = previous image
                            self._debounced_navigate('prev')
                            return True  # Event handled
                        elif vertical_delta < 0:
                            # Wheel down (Qt negative delta) = next image
                            self._debounced_navigate('next')
                            return True  # Event handled
                else:
                    # In zoom mode: handle horizontal wheel for panning with reversed direction
                    # Vertical wheel is used for normal scrolling (panning)
                    if abs(horizontal_delta) > 0:
                        # Manually handle horizontal wheel to reverse direction
                        # In Qt: angleDelta().x() > 0 = scroll left, < 0 = scroll right
                        # Standard QScrollArea behavior:
                        #   - delta > 0 (left scroll) -> increase scroll value -> viewport right -> image moves left
                        #   - delta < 0 (right scroll) -> decrease scroll value -> viewport left -> image moves right
                        # We want intuitive behavior:
                        #   - Right scroll (delta < 0) -> image moves right -> decrease scroll value (same as standard)
                        #   - Left scroll (delta > 0) -> image moves left -> increase scroll value (same as standard)
                        # But user reports it's reversed, so we reverse it:
                        #   - Right scroll (delta < 0) -> image moves right -> increase scroll value (reversed)
                        #   - Left scroll (delta > 0) -> image moves left -> decrease scroll value (reversed)
                        h_scroll = self.scroll_area.horizontalScrollBar()
                        if h_scroll:
                            scroll_amount = horizontal_delta // 8  # Convert to pixels (standard conversion)
                            current_value = h_scroll.value()
                            # Reverse: add instead of subtract to flip the direction
                            new_value = current_value + scroll_amount  # Reversed: add instead of subtract
                            h_scroll.setValue(max(h_scroll.minimum(), min(new_value, h_scroll.maximum())))
                        return True  # Event handled
                    # For vertical wheel in zoom mode, allow normal scrolling
                    if abs(vertical_delta) > 0:
                        return False  # Let the event pass through for normal vertical panning

        return super().eventFilter(obj, event)

    def changeEvent(self, event):
        """Handle window state changes"""
        if event.type() == QEvent.Type.WindowStateChange:
            if sys.platform == "darwin":
                # Guard against nested WindowStateChange recursion on macOS.
                if getattr(self, "_handling_window_state_change", False):
                    return
                self._handling_window_state_change = True
                try:
                    QTimer.singleShot(
                        0, lambda: setattr(self, "_handling_window_state_change", False)
                    )
                except Exception:
                    self._handling_window_state_change = False
            # Update title bar maximize button to match actual window state.
            if hasattr(self, 'title_bar') and self.title_bar is not None:
                self.title_bar._sync_maximize_state()
        super().changeEvent(event)

    def closeEvent(self, event):
        """Handle application close event with proper cleanup.

        Hide the UI immediately so quit feels instant; then do a short
        best-effort cleanup. Thread-pool waits used to leave the window
        on screen for up to several seconds even when work was already
        cancelled.
        """
        import logging
        import time as _time

        logger = logging.getLogger(__name__)
        t0 = _time.perf_counter()
        logger.info("[CLOSE] Application close event triggered, starting cleanup...")

        # Restore display brightness before teardown if Adjust raised it.
        with contextlib.suppress(Exception):
            self._restore_adjust_brightness()

        # 1) Take the frontend down first -- user-visible quit.
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
            self.hide()
            for w in list(QApplication.topLevelWidgets()):
                try:
                    if w is not self and w.isVisible():
                        w.hide()
                except Exception:
                    pass
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
            logger.info(
                "[CLOSE] Frontend hidden in %.0fms",
                (_time.perf_counter() - t0) * 1000.0,
            )
        except Exception as e:
            logger.warning("[CLOSE] Failed to hide frontend promptly: %s", e)

        try:
            self._stop_animations()
            if getattr(self, "_save_session_debounce_timer", None) is not None:
                self._save_session_debounce_timer.stop()
            # Save session state first (flush debounce -- must be synchronous on quit)
            self.save_session_state()
            logger.info("[CLOSE] Session state saved")

            if getattr(self, "_semantic_indexing_in_progress", False) or getattr(
                self, "_face_indexing_in_progress", False
            ):
                folder = getattr(self, "current_folder", None)
                if folder:
                    self._set_semantic_index_incomplete(folder, True)
                self._semantic_index_active_token = None
                self._semantic_indexing_in_progress = False
                self._face_index_active_token = None
                self._face_indexing_in_progress = False
                logger.info(
                    "[CLOSE] Search indexing interrupted; will resume on next launch"
                )

            # Clean up current processor
            if hasattr(self, 'current_processor') and self.current_processor is not None:
                logger.info("[CLOSE] Cleaning up current processor...")
                self._cleanup_current_processing()
                logger.info("[CLOSE] Current processor cleaned up")

            # Cancel all preload threads
            if hasattr(self, 'preload_manager') and self.preload_manager is not None:
                logger.info("[CLOSE] Cancelling all preload threads...")
                try:
                    self.preload_manager.cancel_all_preloads()
                    logger.info("[CLOSE] All preload threads cancelled")
                except Exception as e:
                    logger.warning(f"[CLOSE] Error cancelling preload threads: {e}", exc_info=True)

            # Stop gallery background loading
            if hasattr(self, 'gallery_justified') and self.gallery_justified:
                logger.info("[CLOSE] Stopping gallery background loading...")
                try:
                    # Stop background loading flag
                    self.gallery_justified._background_loading_active = False
                    # Stop all timers
                    if hasattr(self.gallery_justified, '_load_timer') and self.gallery_justified._load_timer:
                        self.gallery_justified._load_timer.stop()
                        self.gallery_justified._load_timer = None
                    if hasattr(self.gallery_justified, '_scroll_settle_timer') and self.gallery_justified._scroll_settle_timer:
                        self.gallery_justified._scroll_settle_timer.stop()
                        self.gallery_justified._scroll_settle_timer = None
                    if hasattr(self.gallery_justified, '_resize_timer') and self.gallery_justified._resize_timer:
                        self.gallery_justified._resize_timer.stop()
                        self.gallery_justified._resize_timer = None
                    # Clear load queues
                    if hasattr(self.gallery_justified, '_load_queue'):
                        self.gallery_justified._load_queue.clear()
                    if hasattr(self.gallery_justified, '_priority_queue'):
                        self.gallery_justified._priority_queue.clear()
                    logger.info("[CLOSE] Gallery background loading stopped")
                except Exception as e:
                    logger.warning(f"[CLOSE] Error stopping gallery background loading: {e}", exc_info=True)

            # Stop and cancel all image load tasks -- do not wait long; os._exit follows.
            if hasattr(self, 'image_manager') and self.image_manager is not None:
                logger.info("[CLOSE] Stopping and cancelling all image load tasks...")
                try:
                    # First stop accepting new tasks, then cancel existing ones
                    if hasattr(self.image_manager, '_stopped'):
                        self.image_manager._stopped = True
                    self.image_manager.cancel_all_tasks()
                    # Now shutdown worker pools/process pool (app is closing)
                    if hasattr(self.image_manager, 'shutdown'):
                        self.image_manager.shutdown()
                    logger.info("[CLOSE] All image load tasks stopped and cancelled")
                except Exception as e:
                    logger.warning(f"[CLOSE] Error stopping/cancelling image load tasks: {e}", exc_info=True)

            # Brief wait only -- frontend is already hidden.
            _pool_wait_ms = _env_int("RAWVIEWER_CLOSE_POOL_WAIT_MS", 250, minimum=0)
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                logger.info("[CLOSE] Waiting for thread pool to finish...")
                try:
                    self.thread_pool.waitForDone(_pool_wait_ms)
                    logger.info("[CLOSE] Thread pool finished")
                except Exception as e:
                    logger.warning(f"[CLOSE] Error waiting for thread pool: {e}", exc_info=True)

            if hasattr(self, 'image_manager') and self.image_manager is not None and hasattr(self.image_manager, '_thread_pool') and self.image_manager._thread_pool is not None:
                    logger.info("[CLOSE] Waiting for image manager thread pool to finish...")
                    try:
                        self.image_manager._thread_pool.waitForDone(_pool_wait_ms)
                        logger.info("[CLOSE] Image manager thread pool finished")
                    except Exception as e:
                        logger.warning(f"[CLOSE] Error waiting for image manager thread pool: {e}", exc_info=True)

            # Force terminate any remaining threads/processes (no sleeps)
            logger.info("[CLOSE] Force terminating any remaining processes...")
            try:
                if hasattr(self, 'current_processor') and self.current_processor is not None:
                    logger.warning("[CLOSE] Force terminating current processor...")
                    try:
                        if hasattr(self.current_processor, 'terminate'):
                            self.current_processor.terminate()
                        if hasattr(self.current_processor, 'wait'):
                            self.current_processor.wait(50)
                    except Exception as e:
                        logger.warning(f"[CLOSE] Error force terminating processor: {e}")
                    finally:
                        self.current_processor = None

                if hasattr(self, 'preload_manager') and self.preload_manager is not None:
                    try:
                        self.preload_manager.cancel_all_preloads()
                    except Exception as e:
                        logger.warning(f"[CLOSE] Error force cleaning preload manager: {e}")

                if hasattr(self, 'image_manager') and self.image_manager is not None:
                    try:
                        self.image_manager.cancel_all_tasks()
                    except Exception as e:
                        logger.warning(f"[CLOSE] Error force cleaning image manager: {e}")
            except Exception as e:
                logger.warning(f"[CLOSE] Error during force cleanup: {e}", exc_info=True)

            # Close image cache resources (database connections)
            if hasattr(self, 'image_cache') and self.image_cache is not None:
                logger.info("[CLOSE] Closing image cache resources...")
                try:
                    self.image_cache.close()
                    logger.info("[CLOSE] Image cache resources closed")
                except Exception as e:
                    logger.warning(f"[CLOSE] Error closing image cache: {e}")

            logger.info(
                "[CLOSE] Cleanup completed successfully in %.0fms",
                (_time.perf_counter() - t0) * 1000.0,
            )

        except Exception as e:
            logger.error(f"[CLOSE] Error during cleanup: {e}", exc_info=True)
        finally:
            # Always call parent closeEvent to ensure proper Qt cleanup
            super().closeEvent(event)
