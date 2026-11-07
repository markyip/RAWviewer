"""misc mixin, part 4 of 4 (28 methods, 536 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtCore import QEvent, QPoint, QRect, Qt
from PyQt6.QtWidgets import QMessageBox

from core import theme
from core.config import _norm_path, safe_print
from rawviewer_app.core.widgets import CustomConfirmDialog

from .. import lazy
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class MiscPart4:
    def on_processing_error(self, error_message):
        """Handle RAW processing errors"""
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()

        import logging
        import traceback
        logger = logging.getLogger(__name__)

        current_file = getattr(self, 'current_file_path', 'unknown')
        logger.error(f"on_processing_error called for: {os.path.basename(current_file)}, error: {error_message}")

        # Check if this error is for the current file (important for rapid navigation)
        if hasattr(self, 'current_processor') and self.current_processor is not None:
            processor_file = getattr(self.current_processor, 'file_path', None)
            if processor_file and processor_file != current_file:
                logger.warning(f"Error signal mismatch: processor file ({os.path.basename(processor_file)}) != current file ({os.path.basename(current_file)}). This may indicate rapid navigation.")

        try:
            # If we have a pending thumbnail and full processing failed, show it as fallback
            if hasattr(self, '_pending_thumbnail') and self._pending_thumbnail is not None:
                logger.debug("Using pending thumbnail as fallback")
                try:
                    self.display_numpy_image(self._pending_thumbnail)
                    self.status_bar.showMessage("Showing preview", 3000)
                    self._pending_thumbnail = None
                    return
                except Exception as display_error:
                    logger.error(f"Error displaying pending thumbnail: {display_error}", exc_info=True)

            error_msg = f"Error processing RAW file:\n{error_message}"
            logger.debug(f"Showing error message to user: {error_msg}")
            try:
                self.show_error("RAW Processing Error", error_msg)
            except Exception as show_error_ex:
                logger.error(f"Error showing error dialog: {show_error_ex}")

            try:
                self.image_label.setText(
                    "Error loading image\n\nPlease try a different RAW file"
                )
                self._clear_single_image_histogram()
                self.status_bar.showMessage("Error loading image")
                # Reset window title on error
                self.setWindowTitle('RAWviewer')
                # Update custom title bar
                if hasattr(self, 'title_bar') and self.title_bar is not None:
                    self.title_bar.set_title('RAWviewer')
            except Exception as ui_error:
                logger.error(f"Error updating UI on processing error: {ui_error}")
        except Exception as e:
            logger.error(f"Critical error in on_processing_error: {e}", exc_info=True)
            logger.debug(f"on_processing_error error traceback: {traceback.format_exc()}")

    def confirm_deletion(self):
        """Show confirmation dialog for file deletion with custom MD3 design"""
        filename = os.path.basename(_require_str_path(self.current_file_path))

        dialog = CustomConfirmDialog(
            parent=self,
            title="Confirm Delete",
            message="Are you sure you want to delete this file?",
            informative_text=f"File: {filename}\n\nThis will move the file to the Recycle Bin."
        )

        self._is_delete_dialog_open = True
        try:
            dialog.exec()
            return dialog.result_value
        finally:
            self._is_delete_dialog_open = False

    def perform_deletion(self):
        """Perform the actual file deletion"""
        try:
            file_to_delete = _require_str_path(self.current_file_path)
            self._delete_path_to_trash(file_to_delete)
            self.handle_post_deletion_navigation()
        except Exception as e:
            error_msg = f"Could not delete file:\n{str(e)}"
            self.show_error("Delete Error", error_msg)

    def _run_deferred_post_deletion_load(self, generation: int) -> None:
        if generation != self._defer_post_deletion_load_generation:
            return
        if not self.image_files:
            return
        if self.current_file_index >= len(self.image_files):
            self.current_file_index = len(self.image_files) - 1
        if self.current_file_index >= 0:
            self.load_raw_image(self.image_files[self.current_file_index])

    def scale_image_to_fit(self):
        """Scale image to fit the current window size while maintaining aspect ratio"""
        import logging
        logger = logging.getLogger(__name__)

        if getattr(self, "gpu_view", None) is not None:
            self.gpu_view.fit_to_window()
            self.fit_to_window = True
            return

        if not self.current_pixmap:
            return

        # Skip scaling during active resize to prevent intermediate updates
        if self._is_resizing:
            return

        original_size = self.current_pixmap.size()
        available_size = self.scroll_area.size()
        margin = 20
        max_width = available_size.width() - margin
        max_height = available_size.height() - margin

        # Ensure dimensions are at least 1px to prevent crash
        safe_width = max(1, max_width)
        safe_height = max(1, max_height)

        scaled_pixmap = self.current_pixmap.scaled(
            safe_width, safe_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        scaled_size = scaled_pixmap.size()

        # Log scaling information
        logger.info(f"[IMAGE_SCALE] Scaling image: original={original_size.width()}x{original_size.height()}, "
                   f"available={max_width}x{max_height}, scaled={scaled_size.width()}x{scaled_size.height()}")
        safe_print(f"[IMAGE_SCALE] Original: {original_size.width()}x{original_size.height()} -> "
              f"Scaled: {scaled_size.width()}x{scaled_size.height()} (available: {max_width}x{max_height})")

        self._set_single_view_pixmap(scaled_pixmap)
        self.image_label.adjustSize()  # Ensure label is resized to pixmap
        self.scroll_area.widget().adjustSize()  # Force scroll area to update
        self.scroll_area.updateGeometry()
        self.setFocus()

    def scale_image_to_100_percent(self):
        """Display image at 100% zoom (actual pixel size)"""
        import logging
        logger = logging.getLogger(__name__)

        if getattr(self, "gpu_view", None) is not None:
            self.gpu_view.zoom_to_actual()
            self.fit_to_window = False
            self._gpu_request_full_resolution_if_needed()
            return

        if not self.current_pixmap:
            return

        original_size = self.current_pixmap.size()

        # Log 100% zoom information
        logger.info(f"[IMAGE_SCALE] Setting to 100% zoom: {original_size.width()}x{original_size.height()}")
        safe_print(f"[IMAGE_SCALE] 100% zoom: {original_size.width()}x{original_size.height()}")

        # Set original pixmap without scaling
        self._set_single_view_pixmap(self.current_pixmap)

        # Center the image in the scroll area
        self.center_image_in_scroll_area()

        # Ensure main window retains focus for keyboard events
        self.setFocus()

    def _position_resize_grip(self) -> None:
        grip = getattr(self, "_resize_grip", None)
        if grip is None:
            return
        grip.move(
            max(0, self.width() - grip.width()),
            max(0, self.height() - grip.height()),
        )
        if self.isMaximized():
            grip.hide()
        else:
            grip.show()
            grip.raise_()

    def _sync_resize_grip_hint(self, pos: QPoint) -> None:
        grip = getattr(self, "_resize_grip", None)
        if grip is None or self.isMaximized():
            return
        active = bool(getattr(self, "_resize_edge_active", None))
        near = self._pointer_near_bottom_right_grip(pos) or self._get_resize_edge(pos) == "bottom_right"
        if hasattr(grip, "set_hint_visible"):
            grip.set_hint_visible(active or near)
        self._position_resize_grip()

    def _bottom_right_grip_rect(self, *, margin: int = 6) -> QRect:
        """Hit area for the lower-right resize hint."""
        grip = getattr(self, "_resize_grip", None)
        if grip is None:
            return QRect()
        gx = max(0, self.width() - grip.width())
        gy = max(0, self.height() - grip.height())
        pad = max(0, int(margin))
        return QRect(
            gx - pad,
            gy - pad,
            grip.width() + 2 * pad,
            grip.height() + 2 * pad,
        )

    def _pointer_near_bottom_right_grip(self, pos: QPoint) -> bool:
        hit = self._bottom_right_grip_rect()
        return not hit.isNull() and hit.contains(pos)

    def _resize_pos_from_event(self, event) -> QPoint:
        gp = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
        return self.mapFromGlobal(gp)

    def _frame_resize_border(self) -> int:
        return self._FRAME_RESIZE_BORDER

    def _allow_frame_resize_event(self, obj, pos: QPoint) -> bool:
        resizing = bool(getattr(self, "_resize_edge_active", None))
        if resizing:
            return True
        status_chrome = self._is_status_bar_chrome_widget(obj)
        title_chrome = self._is_title_bar_chrome_widget(obj)
        if status_chrome and not self._status_bar_corner_resize_at(pos):
            return False
        return not (title_chrome and not self._title_bar_corner_resize_at(pos))

    def _resize_blocked_at(self, pos: QPoint) -> bool:
        if self.isMaximized():
            return True
        if self._title_bar_blocks_resize_at(pos):
            return True
        if self._status_bar_blocks_resize_at(pos):
            return True
        return bool(self._scrollbar_blocks_resize_at(pos))

    def _begin_resize_at_edge(self, event, edge: str) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self.isMaximized():
            return False
        self._resize_edge_active = edge
        if hasattr(event, "globalPosition"):
            self._resize_start_pos = event.globalPosition().toPoint()
        else:
            self._resize_start_pos = event.globalPos()
        self._resize_start_geometry = self.geometry()
        self._is_resizing = True
        grip = getattr(self, "_resize_grip", None)
        if grip is not None:
            grip.hide()
        if hasattr(self, "view_mode") and self.view_mode == "gallery":
            gallery = getattr(self, "gallery_justified", None)
            if gallery is not None:
                gallery._ignore_resize_events = True
        with contextlib.suppress(Exception):
            self.grabMouse()
        return True

    def _begin_resize_if_edge(self, event, pos: QPoint) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self._resize_blocked_at(pos):
            return False
        edge = self._get_resize_edge(pos)
        if not edge:
            return False
        self._resize_edge_active = edge
        if hasattr(event, "globalPosition"):
            self._resize_start_pos = event.globalPosition().toPoint()
        else:
            self._resize_start_pos = event.globalPos()
        self._resize_start_geometry = self.geometry()
        self._is_resizing = True
        grip = getattr(self, "_resize_grip", None)
        if grip is not None:
            grip.hide()
        if hasattr(self, "view_mode") and self.view_mode == "gallery":
            gallery = getattr(self, "gallery_justified", None)
            if gallery is not None:
                gallery._ignore_resize_events = True
        with contextlib.suppress(Exception):
            self.grabMouse()
        return True

    def _handle_frame_resize_event(self, event) -> bool:
        """Edge resize from the main window or filtered child widgets."""
        et = event.type()
        if et == QEvent.Type.MouseButtonRelease:
            return self._finish_resize_drag()
        if et == QEvent.Type.MouseMove and self._resize_edge_active:
            return self._continue_resize_drag(event)
        pos = self._resize_pos_from_event(event)
        if not self.rect().contains(pos):
            return False
        if et == QEvent.Type.MouseButtonPress:
            return self._begin_resize_if_edge(event, pos)
        if et == QEvent.Type.MouseMove:
            self._update_resize_edge_hover(pos)
        return False

    def _update_resize_edge_hover(self, pos) -> None:
        """Update resize cursor and bottom-right grip while hovering near window edges."""
        if getattr(self, "_resize_edge_active", None):
            return

        grip = getattr(self, "_resize_grip", None)

        if self.isMaximized():
            self.unsetCursor()
            if grip is not None:
                grip.hide()
            return

        if self._title_bar_blocks_resize_at(pos):
            self.unsetCursor()
            if grip is not None and hasattr(grip, "set_hint_visible"):
                grip.set_hint_visible(False)
            return

        if self._scrollbar_blocks_resize_at(pos):
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if grip is not None and hasattr(grip, "set_hint_visible"):
                grip.set_hint_visible(False)
            return

        if self._status_bar_blocks_resize_at(pos):
            self.unsetCursor()
            self._sync_resize_grip_hint(pos)
            return

        border = self._frame_resize_border()
        w = self.width()
        h = self.height()

        if pos.x() <= border and pos.y() <= border:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif pos.x() >= w - border and pos.y() <= border or pos.x() <= border and pos.y() >= h - border:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif pos.x() >= w - border and pos.y() >= h - border:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif pos.x() <= border or pos.x() >= w - border:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif pos.y() <= border or pos.y() >= h - border:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.unsetCursor()

        self._sync_resize_grip_hint(pos)

    def _get_resize_edge(self, pos):
        """Determine which edge the mouse is near"""
        if self.isMaximized():
            return None

        border = self._frame_resize_border()
        w = self.width()
        h = self.height()
        x, y = pos.x(), pos.y()

        if x <= border and y <= border:
            return 'top_left'
        if x >= w - border and y <= border:
            return 'top_right'
        if x <= border and y >= h - border:
            return 'bottom_left'
        if x >= w - border and y >= h - border:
            return 'bottom_right'

        if x <= border:
            return 'left'
        if x >= w - border:
            return 'right'
        if y <= border:
            return 'top'
        if y >= h - border:
            return 'bottom'

        return None

    def get_supported_extensions(self):
        """Get list of supported image file extensions."""
        from raw_adjustments.file_extensions import get_supported_extensions

        return get_supported_extensions()

    def is_image_file(self, file_path):
        """Check if file is a supported image format"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.get_supported_extensions()

    def is_video_file(self, file_path):
        """Check if file is a known video format (for exclusion)"""
        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
        ext = os.path.splitext(file_path)[1].lower()
        return ext in video_exts

    def reset_to_initial_state(self):
        """Reset the UI to its initial state when no image or folder is loaded (e.g. volume ejected)."""
        self.current_pixmap = None
        self.current_image = None
        self.current_file_path = None
        self.image_files = []
        self.current_file_index = -1
        if hasattr(self, 'folder_path'):
            self.folder_path = None
        self._sync_single_image_histogram()

        message = self._no_image_loaded_help_text()

        if hasattr(self, 'view_mode') and self.view_mode == 'gallery' and hasattr(self, 'gallery_justified') and self.gallery_justified:
            self.gallery_justified.show_empty_message("No image loaded\nClick 📁 or drag and drop a folder or image to load it")
        else:
            if hasattr(self, 'image_label'):
                self.image_label.setText(message)
                self.image_label.setStyleSheet("QLabel { color: #666; font-size: 14px; background-color: transparent; }")

        if hasattr(self, 'status_metadata_label'):
            self._apply_top_metadata_text("")
        if hasattr(self, 'status_counter_label'):
            self.status_counter_label.setVisible(False)
        if hasattr(self, 'sort_toggle_button'):
            self.sort_toggle_button.setVisible(False)
        if hasattr(self, 'size_slider'):
            self.size_slider.setVisible(False)

        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage("Ready")
        self.setWindowTitle('RAWviewer')
        if hasattr(self, 'title_bar') and self.title_bar is not None:
            self.title_bar.set_title('RAWviewer')

    def show_no_images_message(self, supported_extensions):
        """Display 'No images found' message in the main viewing area"""
        # Format supported extensions for display
        formats_text = ', '.join(supported_extensions)
        message = f"No images found.\n\nSupported formats: {formats_text}"

        # Clear any existing pixmap
        self.current_pixmap = None
        self.current_image = None
        self._sync_single_image_histogram()

        # Display message in main viewing area
        # Check current view mode to determine where to show the message
        if hasattr(self, 'view_mode') and self.view_mode == 'gallery' and self.gallery_justified:
            self.gallery_justified.show_empty_message(message)
        else:
            # Single view mode - use image_label
            self.image_label.setText(message)
            self.image_label.setStyleSheet(
                f"QLabel {{ color: {theme.INK_MUTED}; font-size: 14px; }}")

        # Clear top metadata when no files
        self._apply_top_metadata_text("")
        if hasattr(self, 'status_counter_label'):
            self.status_counter_label.setVisible(False)
        if hasattr(self, 'sort_toggle_button'):
            self.sort_toggle_button.setVisible(False)
        if hasattr(self, 'size_slider'):
            self.size_slider.setVisible(False)

        # Update status bar
        self.status_bar.showMessage("No images found")

        # Reset window title
        self.setWindowTitle('RAWviewer')
        # Update custom title bar
        if hasattr(self, 'title_bar') and self.title_bar is not None:
            self.title_bar.set_title('RAWviewer')

    def show_error(self, title, message):
        """Show error message dialog"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()

    def _on_sensor_dims_resolved(self, file_path: str) -> None:
        try:
            if _norm_path(file_path) == _norm_path(getattr(self, "current_file_path", None)):
                self.update_status_bar()
        except Exception:
            pass

    def _move_path_to_discard(self, file_to_move: str) -> None:
        if not file_to_move or not os.path.exists(file_to_move):
            return
        try:
            folder_path = os.path.dirname(file_to_move)
            discard_folder = os.path.join(folder_path, "Discard")
            os.makedirs(discard_folder, exist_ok=True)
            filename = os.path.basename(file_to_move)
            target_path = os.path.join(discard_folder, filename)
            # If file with same name exists in Discard, add a suffix
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(
                    discard_folder, f"{base}_discarded_{counter}{ext}")
                counter += 1

            # Before moving, cancel any preload / manager tasks for this file and clear cache
            self._cancel_load_and_preload_for_path(file_to_move)

            # Clear cache for this file
            from image_cache import get_image_cache
            cache = get_image_cache()
            cache.invalidate_file(file_to_move)

            # Now move the file
            os.rename(file_to_move, target_path)
            self._migrate_visual_rotation_path(file_to_move, target_path)

            self._drop_discarded_from_semantic_corpus(file_to_move)
            # Remove from image files list
            self._remove_file_from_active_image_list(file_to_move)
            self.status_bar.showMessage(f"Moved to Discard: {filename}")
            self.schedule_save_session_state()
        except Exception as e:
            error_msg = f"Could not move file to Discard folder:\n{str(e)}"
            self.show_error("Discard Error", error_msg)

    def _enter_io_pressure_mode(self, reason: str = "") -> None:
        """Reduce background I/O when the process hits EMFILE or similar pressure."""
        import logging

        from common_image_loader import note_emfile_pressure

        note_emfile_pressure()
        logger = logging.getLogger(__name__)
        if reason:
            logger.warning("[IO] Entering I/O pressure mode (%s)", reason)
        try:
            lm = getattr(self, "image_manager", None)
            if lm is not None and hasattr(lm, "enter_io_pressure_throttle"):
                lm.enter_io_pressure_throttle()
        except Exception:
            pass
        try:
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "_idle_preload_timer"):
                gj._idle_preload_timer.stop()
        except Exception:
            pass
