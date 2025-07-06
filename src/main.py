import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QFileDialog,
                             QMessageBox, QScrollArea, QSizePolicy, QHBoxLayout, QPushButton, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QEvent, QSettings
from PyQt6.QtGui import (QPixmap, QImage, QAction, QKeySequence,
                         QDragEnterEvent, QDropEvent, QCursor, QIcon, QTransform)
import rawpy
import numpy as np
from natsort import natsorted
from send2trash import send2trash
import exifread
from datetime import datetime
import platform


class RAWProcessor(QThread):
    """Thread for processing RAW images to avoid UI blocking"""
    image_processed = pyqtSignal(object)  # Accepts np.ndarray or None
    error_occurred = pyqtSignal(str)
    # Signal when thumbnail fallback is used
    thumbnail_fallback_used = pyqtSignal(str)

    def __init__(self, file_path, is_raw):
        super().__init__()
        self.file_path = file_path
        self.is_raw = is_raw

    def get_orientation_from_exif(self, file_path):
        """Extract orientation from EXIF data"""
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Check for orientation tag
                orientation_tag = tags.get('Image Orientation')
                if orientation_tag:
                    orientation_str = str(orientation_tag)

                    # Map orientation descriptions to numeric values
                    orientation_map = {
                        'Horizontal (normal)': 1,
                        'Mirrored horizontal': 2,
                        'Rotated 180': 3,
                        'Mirrored vertical': 4,
                        'Mirrored horizontal then rotated 90 CCW': 5,
                        'Rotated 90 CW': 6,
                        'Mirrored horizontal then rotated 90 CW': 7,
                        'Rotated 90 CCW': 8
                    }

                    return orientation_map.get(orientation_str, 1)

                return 1  # Default orientation (no rotation needed)
        except Exception:
            return 1  # Default orientation if EXIF reading fails

    def apply_orientation_correction(self, image_array, orientation):
        """Apply orientation correction to numpy array"""
        if orientation == 1:
            # Normal orientation, no changes needed
            return image_array
        elif orientation == 2:
            # Mirrored horizontal
            return np.fliplr(image_array)
        elif orientation == 3:
            # Rotated 180 degrees
            return np.rot90(image_array, 2)
        elif orientation == 4:
            # Mirrored vertical
            return np.flipud(image_array)
        elif orientation == 5:
            # Mirrored horizontal then rotated 90 CCW
            return np.rot90(np.fliplr(image_array), 1)
        elif orientation == 6:
            # Rotated 90 CW
            return np.rot90(image_array, 3)
        elif orientation == 7:
            # Mirrored horizontal then rotated 90 CW
            return np.rot90(np.fliplr(image_array), 3)
        elif orientation == 8:
            # Rotated 90 CCW
            return np.rot90(image_array, 1)
        else:
            return image_array

    def run(self):
        try:
            if self.is_raw:
                # Get orientation from EXIF data
                orientation = self.get_orientation_from_exif(self.file_path)

                try:
                    # First try to open the RAW file
                    with rawpy.imread(self.file_path) as raw:
                        # Extract thumbnail first as a potential fallback
                        thumbnail_data = None
                        try:
                            thumb = raw.extract_thumb()
                            if thumb.format == rawpy.ThumbFormat.JPEG:
                                # For JPEG thumbnails, decode the JPEG data
                                import io
                                from PIL import Image
                                import numpy as np

                                # Convert JPEG bytes to PIL Image
                                jpeg_image = Image.open(io.BytesIO(thumb.data))
                                # Convert to RGB if needed
                                if jpeg_image.mode != 'RGB':
                                    jpeg_image = jpeg_image.convert('RGB')
                                # Convert to numpy array
                                thumbnail_data = np.array(jpeg_image)
                                # Apply orientation correction to thumbnail
                                thumbnail_data = self.apply_orientation_correction(
                                    thumbnail_data, orientation)
                            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                                # Bitmap thumbnail is already in array format
                                thumbnail_data = thumb.data
                                # Apply orientation correction to thumbnail
                                thumbnail_data = self.apply_orientation_correction(
                                    thumbnail_data, orientation)
                        except Exception:
                            # Thumbnail extraction failed, continue without fallback
                            pass

                        # Now try full RAW processing
                        try:
                            rgb_image = raw.postprocess()
                            # Apply orientation correction to processed RAW image
                            rgb_image = self.apply_orientation_correction(
                                rgb_image, orientation)
                            self.image_processed.emit(rgb_image)
                            return  # Success, exit early
                        except Exception as processing_error:
                            # If RAW processing fails and we have a thumbnail, use it
                            if thumbnail_data is not None:
                                self.thumbnail_fallback_used.emit(
                                    "Using embedded thumbnail due to LibRaw compatibility issue")
                                self.image_processed.emit(thumbnail_data)
                                return  # Success with thumbnail
                            else:
                                # No thumbnail available, re-raise the processing error
                                raise processing_error
                except Exception as e:
                    # Handle file opening errors
                    raise e
            else:
                # For non-RAW, emit None (handled in main thread)
                self.image_processed.emit(None)
        except Exception as e:
            # Provide more specific error messages
            error_msg = str(e)
            if "data corrupted" in error_msg.lower():
                error_msg = f"RAW processing failed due to LibRaw compatibility issue.\n\nThis is a known issue with LibRaw 0.21.3 and certain NEF files.\nTry using a different RAW processor or contact the developer for updates.\n\nOriginal error: {error_msg}"
            elif "unsupported file format" in error_msg.lower():
                error_msg = f"This RAW file format may not be supported by your LibRaw version.\n\nOriginal error: {error_msg}"
            elif "input/output error" in error_msg.lower():
                error_msg = f"Cannot read the file. It may be corrupted or in use by another program.\n\nOriginal error: {error_msg}"
            elif "cannot allocate memory" in error_msg.lower():
                error_msg = f"Not enough memory to process this large RAW file.\n\nOriginal error: {error_msg}"

            self.error_occurred.emit(error_msg)


class RAWImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_pixmap = None

        # Enhanced zoom and pan state tracking
        # Note: Only using simple toggle between fit-to-window and 100% zoom
        self.current_zoom_level = 1.0  # Current zoom level (1.0 = 100%)
        self.fit_to_window = True  # Whether we're in fit-to-window mode
        self.zoom_center_point = None  # Store center point for zooming

        # Panning state
        self.panning = False
        self.last_pan_point = QPoint()
        self.start_scroll_x = 0
        self.start_scroll_y = 0

        # Folder scanning and file list management
        self.current_folder = None
        self.image_files = []  # List of all image files in current folder
        self.current_file_index = -1  # Index of current file in the list
        self.current_file_path = None  # Path of currently loaded file
        self.thumbnail_cache = {}  # Cache for thumbnails
        self.film_strip_visible = False
        self.thumbnail_threads = []  # Track running thumbnail threads

        self.init_ui()
        # Try to restore previous session
        if not self.restore_session_state():
            # If no session, show default message
            pass

    def get_orientation_from_exif(self, file_path):
        """Extract orientation from EXIF data for non-RAW files"""
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Check for orientation tag
                orientation_tag = tags.get('Image Orientation')
                if orientation_tag:
                    orientation_str = str(orientation_tag)

                    # Map orientation descriptions to numeric values
                    orientation_map = {
                        'Horizontal (normal)': 1,
                        'Mirrored horizontal': 2,
                        'Rotated 180': 3,
                        'Mirrored vertical': 4,
                        'Mirrored horizontal then rotated 90 CCW': 5,
                        'Rotated 90 CW': 6,
                        'Mirrored horizontal then rotated 90 CW': 7,
                        'Rotated 90 CCW': 8
                    }

                    return orientation_map.get(orientation_str, 1)

                return 1  # Default orientation (no rotation needed)
        except Exception:
            return 1  # Default orientation if EXIF reading fails

    def apply_orientation_to_pixmap(self, pixmap, orientation):
        """Apply orientation correction to QPixmap"""
        if orientation == 1:
            # Normal orientation, no changes needed
            return pixmap

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
            # Mirrored horizontal then rotated 90 CCW
            transform.scale(-1, 1)
            transform.rotate(-90)
        elif orientation == 6:
            # Rotated 90 CW
            transform.rotate(90)
        elif orientation == 7:
            # Mirrored horizontal then rotated 90 CW
            transform.scale(-1, 1)
            transform.rotate(90)
        elif orientation == 8:
            # Rotated 90 CCW
            transform.rotate(-90)

        return pixmap.transformed(transform)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('RAW Image Viewer')
        # Set icon based on platform and available files
        icon_path = None
        if platform.system() == 'Windows' and os.path.exists(os.path.join('icons', 'appicon.ico')):
            icon_path = os.path.join('icons', 'appicon.ico')
        elif platform.system() == 'Darwin' and os.path.exists(os.path.join('icons', 'appicon.icns')):
            icon_path = os.path.join('icons', 'appicon.icns')
        elif os.path.exists(os.path.join('icons', 'appicon.png')):
            icon_path = os.path.join('icons', 'appicon.png')

        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.create_menu_bar()
        self.scroll_area = QScrollArea()
        # Key: allow scrolling when image is larger
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText(
            "No image loaded\n\n"
            "Use File > Open or press Ctrl+O to load a RAW image\n\n"
            "Press Space to toggle between fit-to-window and 100% zoom\n"
            "Double-click image to zoom in/out\n"
            "Click and drag to pan when zoomed\n"
            "Use Left/Right arrow keys to navigate between images (preserves zoom if zoomed in)\n"
            "Press Down Arrow to move the current image to Discard folder\n"
            "Press Delete to delete the current image"
        )
        self.image_label.setStyleSheet(
            "QLabel { color: #666; font-size: 14px; }")
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.image_mouse_press_event
        self.image_label.mouseMoveEvent = self.image_mouse_move_event
        self.image_label.mouseReleaseEvent = self.image_mouse_release_event
        self.image_label.mouseDoubleClickEvent = self.image_double_click_event
        self.scroll_area.setWidget(self.image_label)
        main_layout.addWidget(self.scroll_area)
        # --- Status bar and toggle button ---
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Install event filter to intercept arrow keys
        self.scroll_area.installEventFilter(self)
        self.image_label.installEventFilter(self)

    def create_menu_bar(self):
        """Create the menu bar with File and Keyboard Shortcuts action"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        # Open action
        open_action = QAction('Open', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip('Open a RAW image file')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Open Folder action
        open_folder_action = QAction('Open Folder', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.setStatusTip('Open a folder of images')
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Keyboard Shortcuts action (direct in menu bar)
        shortcuts_action = QAction('Keyboard Shortcuts', self)
        shortcuts_action.setStatusTip('Show keyboard shortcuts')
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        menubar.addAction(shortcuts_action)

    def get_settings(self):
        return QSettings("RAWviewer", "RAWviewer")

    def open_file(self):
        settings = self.get_settings()
        last_dir = settings.value("last_opened_dir", "")
        file_filters = (
            "Image Files (*.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 "
            "*.pef *.srw *.x3f *.raf *.3fr *.fff *.iiq *.cap *.erf "
            "*.mef *.mos *.nrw *.rwl *.srf *.jpeg *.jpg *.heif);;"
            "JPEG (*.jpeg *.jpg);;"
            "HEIF (*.heif);;"
            "Canon RAW (*.cr2 *.cr3);;"
            "Nikon RAW (*.nef);;"
            "Sony RAW (*.arw *.srf);;"
            "Adobe DNG (*.dng);;"
            "Olympus RAW (*.orf);;"
            "Panasonic RAW (*.rw2);;"
            "Pentax RAW (*.pef);;"
            "Samsung RAW (*.srw);;"
            "Sigma RAW (*.x3f);;"
            "Fujifilm RAW (*.raf);;"
            "Hasselblad RAW (*.3fr *.fff);;"
            "Phase One RAW (*.iiq *.cap);;"
            "Epson RAW (*.erf);;"
            "Mamiya RAW (*.mef);;"
            "Leaf RAW (*.mos);;"
            "Casio RAW (*.nrw);;"
            "Leica RAW (*.rwl);;"
            "All Files (*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RAW Image",
            last_dir,  # Use last opened directory
            file_filters
        )
        if file_path:
            self.load_raw_image(file_path)
            # Save the directory for next time
            settings.setValue("last_opened_dir", os.path.dirname(file_path))

    def open_folder(self):
        settings = self.get_settings()
        last_dir = settings.value("last_opened_dir", "")
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Folder", last_dir)
        if folder_path:
            self.load_folder_images(folder_path)
            settings.setValue("last_opened_dir", folder_path)

    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setText("Available Keyboard Shortcuts:")
        msg_box.setInformativeText(
            "- Ctrl+O - Open image file\n"
            "- Ctrl+Shift+O - Open folder of images\n"
            "- Space - Toggle between fit-to-window and 100% zoom\n"
            "- Double-click - Toggle between fit-to-window and 100% zoom\n"
            "- Click and drag - Pan around zoomed image\n"
            "- Left Arrow - Previous image (preserves zoom if zoomed in)\n"
            "- Right Arrow - Next image (preserves zoom if zoomed in)\n"
            "- Down Arrow - Move current image to Discard folder\n"
            "- Delete - Delete current image\n"
            "- Ctrl+Q - Exit application\n\n"
            "You can also drag and drop image files onto the window."
        )
        msg_box.exec()

    def image_mouse_press_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_pixmap:
            if not self.fit_to_window and self._can_pan():
                self.panning = True
                self.last_pan_point = event.pos()
                self.start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                self.start_scroll_y = self.scroll_area.verticalScrollBar().value()
                self.image_label.setCursor(
                    QCursor(Qt.CursorShape.ClosedHandCursor))

    def image_mouse_move_event(self, event):
        if self.panning and self.current_pixmap and self._can_pan():
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
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

    def image_mouse_release_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            if self.current_pixmap and not self.fit_to_window:
                self.image_label.setCursor(
                    QCursor(Qt.CursorShape.OpenHandCursor))
            elif self.fit_to_window:
                self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def image_double_click_event(self, event):
        if not self.current_pixmap:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.fit_to_window:
                # Zooming in from fit-to-window mode
                click_pos = event.pos()
                displayed_pixmap = self.image_label.pixmap()

                if displayed_pixmap:
                    # Calculate where the click occurred relative to the displayed image
                    label_size = self.image_label.size()
                    displayed_size = displayed_pixmap.size()

                    # Calculate image offset within the label (image is centered in label)
                    image_x_offset = (label_size.width() -
                                      displayed_size.width()) / 2
                    image_y_offset = (label_size.height() -
                                      displayed_size.height()) / 2

                    # Adjust click position relative to the displayed image
                    adjusted_click_x = click_pos.x() - image_x_offset
                    adjusted_click_y = click_pos.y() - image_y_offset

                    # Check if click is within the displayed image bounds
                    if (0 <= adjusted_click_x < displayed_size.width() and
                            0 <= adjusted_click_y < displayed_size.height()):

                        # Calculate the ratio of the click position within the displayed image
                        click_ratio_x = adjusted_click_x / displayed_size.width()
                        click_ratio_y = adjusted_click_y / displayed_size.height()

                        # Map this ratio to the full-size image coordinates
                        full_size = self.current_pixmap.size()
                        image_click_x = int(click_ratio_x * full_size.width())
                        image_click_y = int(click_ratio_y * full_size.height())

                        # Clamp to valid coordinates
                        image_click_x = max(
                            0, min(image_click_x, full_size.width() - 1))
                        image_click_y = max(
                            0, min(image_click_y, full_size.height() - 1))

                        self.zoom_center_point = QPoint(
                            image_click_x, image_click_y)
                    else:
                        # Click outside image, center on image center
                        self.zoom_center_point = QPoint(
                            self.current_pixmap.width() // 2,
                            self.current_pixmap.height() // 2)
                else:
                    # No displayed pixmap, center on image center
                    self.zoom_center_point = QPoint(
                        self.current_pixmap.width() // 2,
                        self.current_pixmap.height() // 2)

                # Switch to 100% zoom mode
                self.fit_to_window = False
                self.current_zoom_level = 1.0
                self.zoom_to_point()
            else:
                # Zooming out to fit-to-window mode
                self.fit_to_window = True
                self.current_zoom_level = 1.0
                self.zoom_center_point = None
                self.scale_image_to_fit()
                self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

            self.update_status_bar()
            self.setFocus()

    def zoom_to_point(self):
        if not self.current_pixmap:
            return
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.resize(self.current_pixmap.size())
        self.image_label.adjustSize()  # Ensure label is resized to pixmap
        self.scroll_area.widget().adjustSize()  # Force scroll area to update
        self.scroll_area.updateGeometry()
        self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        # Actually center the view on the zoom point
        self._complete_zoom_to_point()

    def _complete_zoom_to_point(self):
        if self.zoom_center_point:
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

    def convert_widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to full-resolution image coordinates"""
        if not self.current_pixmap:
            return QPoint(0, 0)

        # Get current displayed image size
        displayed_image = self.image_label.pixmap()
        if not displayed_image:
            return QPoint(0, 0)

        # Calculate scaling factor from displayed to original
        original_size = self.current_pixmap.size()
        displayed_size = displayed_image.size()

        scale_x = original_size.width() / displayed_size.width()
        scale_y = original_size.height() / displayed_size.height()

        # Convert widget coordinates to image coordinates
        image_x = int(widget_pos.x() * scale_x)
        image_y = int(widget_pos.y() * scale_y)

        return QPoint(image_x, image_y)

    def apply_pan_offset(self):
        # Deprecated: direct panning is now handled in mouse events
        pass

    def apply_zoom_and_pan_simple(self):
        """Simple zoom and pan that centers on the clicked point"""
        if not self.current_pixmap:
            return

        # Set the image at 100% zoom
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.resize(self.current_pixmap.size())

        # If we have a zoom center point, center the scroll area on it
        if self.zoom_center_point:
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
        if not self.current_pixmap:
            return

        # Calculate scaled size
        original_size = self.current_pixmap.size()
        scaled_width = int(original_size.width() * self.current_zoom_level)
        scaled_height = int(original_size.height() * self.current_zoom_level)

        # Scale the pixmap
        scaled_pixmap = self.current_pixmap.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Set the scaled pixmap
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

        # Apply zoom center point and panning
        if self.zoom_center_point:
            # Calculate the position to center the zoom point
            viewport_size = self.scroll_area.viewport().size()
            image_size = scaled_pixmap.size()

            # Convert image coordinates to scaled coordinates
            center_x = int(self.zoom_center_point.x()
                           * self.current_zoom_level)
            center_y = int(self.zoom_center_point.y()
                           * self.current_zoom_level)

            # Calculate scroll position to center the zoom point
            scroll_x = center_x - round(viewport_size.width() / 2)
            scroll_y = center_y - round(viewport_size.height() / 2)

            # Apply pan offset
            scroll_x -= self.start_scroll_x - self.scroll_area.horizontalScrollBar().value()
            scroll_y -= self.start_scroll_y - self.scroll_area.verticalScrollBar().value()

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

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for file dropping"""
        if event.mimeData().hasUrls():
            # Check if dragged files are supported image formats
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext in self.get_supported_extensions():
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle drop events for file dropping"""
        urls = event.mimeData().urls()
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in self.get_supported_extensions():
                    self.load_raw_image(file_path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def load_raw_image(self, file_path):
        if not os.path.exists(file_path):
            error_msg = f"The file {file_path} does not exist."
            self.show_error("File not found", error_msg)
            return
        self.current_file_path = file_path
        filename = os.path.basename(file_path)
        self.setWindowTitle(f"RAW Image Viewer - {filename}")
        self.status_bar.showMessage(f"Loading {filename}...")
        self.image_label.setText("Processing image...\nPlease wait...")
        ext = os.path.splitext(file_path)[1].lower()
        raw_exts = [
            '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
            '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', '.cap', '.erf',
            '.mef', '.mos', '.nrw', '.rwl', '.srf'
        ]
        is_raw = ext in raw_exts
        self._is_loading_raw = is_raw  # Track for on_image_processed
        self.raw_processor = RAWProcessor(file_path, is_raw)
        self.raw_processor.image_processed.connect(self.on_image_processed)
        self.raw_processor.error_occurred.connect(self.on_processing_error)
        self.raw_processor.thumbnail_fallback_used.connect(
            self.on_thumbnail_fallback)
        self.raw_processor.start()
        self.setFocus()
        # Save session state when image changes
        self.save_session_state()

    def on_thumbnail_fallback(self, message):
        """Handle when thumbnail fallback is used"""
        self.status_bar.showMessage(
            f"⚠️ {message} - Image quality may be reduced")

    def on_image_processed(self, rgb_image):
        try:
            if rgb_image is None:
                # Non-RAW or RAW with thumbnail fallback: load with QPixmap
                pixmap = QPixmap(self.current_file_path)
                if pixmap.isNull():
                    self.show_error("Display Error",
                                    "Could not load image file.")
                    return

                # Apply orientation correction for non-RAW files
                orientation = self.get_orientation_from_exif(
                    self.current_file_path)
                pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)

                self.current_pixmap = pixmap
                if not hasattr(self, '_maintain_zoom_on_navigation'):
                    self.fit_to_window = True
                    self.current_zoom_level = 1.0
                    self.zoom_center_point = None
                    self.scale_image_to_fit()
                else:
                    if self.fit_to_window:
                        self.scale_image_to_fit()
                    else:
                        self.apply_zoom_and_pan()
                    delattr(self, '_maintain_zoom_on_navigation')
                if self.current_file_path:
                    self.scan_folder_for_images(self.current_file_path)
                self.update_status_bar(pixmap.width(), pixmap.height())
                if hasattr(self, '_restore_zoom_center') and self._restore_zoom_center is not None:
                    self.fit_to_window = False
                    self.current_zoom_level = self._restore_zoom_level or 1.0
                    self.zoom_center_point = self._restore_zoom_center
                    self.start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                    self.start_scroll_y = self.scroll_area.verticalScrollBar().value()
                    self.apply_zoom_and_pan()
                    self._restore_zoom_center = None
                    self._restore_zoom_level = None
                    self._restore_start_scroll_x = None
                    self._restore_start_scroll_y = None
            else:
                # RAW: successful processing with numpy array
                height, width, channels = rgb_image.shape
                bytes_per_line = channels * width

                # Ensure the data is contiguous and convert to bytes for PyQt6 compatibility
                if not rgb_image.flags['C_CONTIGUOUS']:
                    rgb_image = np.ascontiguousarray(rgb_image)

                # Convert to bytes if needed (PyQt6 compatibility)
                image_data = rgb_image.data.tobytes() if hasattr(
                    rgb_image.data, 'tobytes') else bytes(rgb_image.data)

                q_image = QImage(image_data, width, height,
                                 bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.current_pixmap = pixmap
                if not hasattr(self, '_maintain_zoom_on_navigation'):
                    self.fit_to_window = True
                    self.current_zoom_level = 1.0
                    self.zoom_center_point = None
                    self.scale_image_to_fit()
                else:
                    if self.fit_to_window:
                        self.scale_image_to_fit()
                    else:
                        self.apply_zoom_and_pan()
                    delattr(self, '_maintain_zoom_on_navigation')
                if self.current_file_path:
                    self.scan_folder_for_images(self.current_file_path)
                self.update_status_bar(width, height)
                if hasattr(self, '_restore_zoom_center') and self._restore_zoom_center is not None:
                    self.fit_to_window = False
                    self.current_zoom_level = self._restore_zoom_level or 1.0
                    self.zoom_center_point = self._restore_zoom_center
                    self.start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                    self.start_scroll_y = self.scroll_area.verticalScrollBar().value()
                    self.apply_zoom_and_pan()
                    self._restore_zoom_center = None
                    self._restore_zoom_level = None
                    self._restore_start_scroll_x = None
                    self._restore_start_scroll_y = None
        except Exception as e:
            error_msg = f"Error displaying image: {str(e)}"
            self.show_error("Display Error", error_msg)
        self.setFocus()

    def on_processing_error(self, error_message):
        """Handle RAW processing errors"""
        error_msg = f"Error processing RAW file:\n{error_message}"
        self.show_error("RAW Processing Error", error_msg)
        self.image_label.setText(
            "Error loading image\n\nPlease try a different RAW file"
        )
        self.status_bar.showMessage("Error loading image")
        # Reset window title on error
        self.setWindowTitle('RAW Image Viewer')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_zoom()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Left:
            self.navigate_to_previous_image()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Right:
            self.navigate_to_next_image()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Down:
            self.move_current_image_to_discard()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Up:
            # Prevent up arrow from moving the scroll area
            event.accept()
        elif event.key() == Qt.Key.Key_Delete:
            self.delete_current_image()
            event.accept()  # Mark event as handled
        else:
            super().keyPressEvent(event)

    def navigate_to_previous_image(self):
        if not self.image_files or len(self.image_files) <= 1:
            return

        # Calculate previous index with wraparound
        if self.current_file_index <= 0:
            # Wrap to last image
            self.current_file_index = len(self.image_files) - 1
        else:
            self.current_file_index -= 1

        # Set flag to maintain zoom state when navigating
        self._maintain_zoom_on_navigation = True

        # Load the previous image
        self.load_raw_image(self.image_files[self.current_file_index])

        # Save current zoom/pan state for restoration
        if not self.fit_to_window:
            self._restore_zoom_center = self.zoom_center_point
            self._restore_zoom_level = self.current_zoom_level
            self._restore_start_scroll_x = self.start_scroll_x
            self._restore_start_scroll_y = self.start_scroll_y
        else:
            self._restore_zoom_center = None
            self._restore_zoom_level = None
            self._restore_start_scroll_x = None
            self._restore_start_scroll_y = None
        self.save_session_state()

    def navigate_to_next_image(self):
        if not self.image_files or len(self.image_files) <= 1:
            return

        # Calculate next index with wraparound
        if self.current_file_index >= len(self.image_files) - 1:
            # Wrap to first image
            self.current_file_index = 0
        else:
            self.current_file_index += 1

        # Set flag to maintain zoom state when navigating
        self._maintain_zoom_on_navigation = True

        # Load the next image
        self.load_raw_image(self.image_files[self.current_file_index])

        # Save current zoom/pan state for restoration
        if not self.fit_to_window:
            self._restore_zoom_center = self.zoom_center_point
            self._restore_zoom_level = self.current_zoom_level
            self._restore_start_scroll_x = self.start_scroll_x
            self._restore_start_scroll_y = self.start_scroll_y
        else:
            self._restore_zoom_center = None
            self._restore_zoom_level = None
            self._restore_start_scroll_x = None
            self._restore_start_scroll_y = None
        self.save_session_state()

    def delete_current_image(self):
        """Delete the current image after confirmation"""
        if (not self.current_file_path or
                not os.path.exists(self.current_file_path)):
            self.show_error("Delete Error", "No image file to delete.")
            return

        # Show confirmation dialog
        if self.confirm_deletion():
            # --- Preserve zoom/pan state for next image (like navigation/discard) ---
            self._maintain_zoom_on_navigation = True
            if not self.fit_to_window:
                self._restore_zoom_center = self.zoom_center_point
                self._restore_zoom_level = self.current_zoom_level
                self._restore_start_scroll_x = self.start_scroll_x
                self._restore_start_scroll_y = self.start_scroll_y
            else:
                self._restore_zoom_center = None
                self._restore_zoom_level = None
                self._restore_start_scroll_x = None
                self._restore_start_scroll_y = None
            self.perform_deletion()
        self.save_session_state()

    def confirm_deletion(self):
        """Show confirmation dialog for file deletion"""
        filename = os.path.basename(self.current_file_path)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Confirm Delete")
        msg_box.setText("Are you sure you want to delete this file?")
        msg_box.setInformativeText(
            f"File: {filename}\n\n"
            f"This will move the file to the Recycle Bin."
        )
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes |
                                   QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.No)

        result = msg_box.exec()
        return result == QMessageBox.StandardButton.Yes

    def perform_deletion(self):
        """Perform the actual file deletion"""
        try:
            filename = os.path.basename(self.current_file_path)

            # Normalize the file path to handle UNC paths and other issues
            normalized_path = os.path.normpath(self.current_file_path)

            # Move file to trash using send2trash
            send2trash(normalized_path)

            # Remove from image files list
            if self.current_file_path in self.image_files:
                self.image_files.remove(self.current_file_path)

            # Update status
            self.status_bar.showMessage(f"Deleted: {filename}")

            # Handle navigation after deletion
            self.handle_post_deletion_navigation()

        except Exception as e:
            error_msg = f"Could not delete file:\n{str(e)}"
            self.show_error("Delete Error", error_msg)

    def handle_post_deletion_navigation(self):
        """Handle navigation after a file has been deleted"""
        if not self.image_files:
            # No more images in folder
            self.current_file_path = None
            self.current_file_index = -1
            self.current_pixmap = None
            self.image_label.setText(
                "No more images in this folder\n\n"
                "Use File > Open to load another image"
            )
            self.status_bar.showMessage("No images remaining in folder")
            self.setWindowTitle('RAW Image Viewer')
            return

        # Adjust current index if needed
        if self.current_file_index >= len(self.image_files):
            self.current_file_index = len(self.image_files) - 1

        # Load the next image (or previous if we were at the end)
        if self.current_file_index >= 0:
            self.load_raw_image(self.image_files[self.current_file_index])

    def toggle_zoom(self):
        """Toggle between fit-to-window and 100% zoom modes"""
        if not self.current_pixmap:
            return
        if self.fit_to_window:
            # Switch to 100% zoom mode - center on image center
            self.fit_to_window = False
            self.current_zoom_level = 1.0
            # Always center on image center when using space bar
            image_center_x = self.current_pixmap.width() // 2
            image_center_y = self.current_pixmap.height() // 2
            self.zoom_center_point = QPoint(image_center_x, image_center_y)
            self.zoom_to_point()
        else:
            # Switch to fit-to-window mode
            self.fit_to_window = True
            self.current_zoom_level = 1.0
            self.zoom_center_point = None
            self.scale_image_to_fit()
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update_status_bar()
        self.setFocus()

    def scale_image_to_fit(self):
        """Scale image to fit the current window size while maintaining aspect ratio"""
        if not self.current_pixmap:
            return

        available_size = self.scroll_area.size()
        margin = 20
        max_width = available_size.width() - margin
        max_height = available_size.height() - margin
        scaled_pixmap = self.current_pixmap.scaled(
            max_width, max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        self.image_label.adjustSize()  # Ensure label is resized to pixmap
        self.scroll_area.widget().adjustSize()  # Force scroll area to update
        self.scroll_area.updateGeometry()
        self.setFocus()

    def scale_image_to_100_percent(self):
        """Display image at 100% zoom (actual pixel size)"""
        if not self.current_pixmap:
            return

        # Set original pixmap without scaling
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.resize(self.current_pixmap.size())

        # Center the image in the scroll area
        self.center_image_in_scroll_area()

        # Ensure main window retains focus for keyboard events
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

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Rescale image when window is resized, but only in fit-to-window mode
        if self.current_pixmap and self.fit_to_window:
            self.scale_image_to_fit()

    def get_supported_extensions(self):
        """Get list of supported image file extensions"""
        return [
            '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
            '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', '.cap', '.erf',
            '.mef', '.mos', '.nrw', '.rwl', '.srf', '.jpeg', '.jpg', '.heif'
        ]

    def scan_folder_for_images(self, file_path):
        """Scan the folder containing the given file for all image files"""
        try:
            # Get the folder path
            folder_path = os.path.dirname(file_path)
            self.current_folder = folder_path

            # Get supported extensions
            supported_extensions = self.get_supported_extensions()

            # Scan folder for image files
            image_files = []

            try:
                for filename in os.listdir(folder_path):
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in supported_extensions:
                        full_path = os.path.join(folder_path, filename)
                        if os.path.isfile(full_path):
                            image_files.append(full_path)
            except OSError as e:
                error_msg = f"Cannot read folder contents:\n{str(e)}"
                self.show_error("Folder Access Error", error_msg)
                return

            # Sort files using natural sorting (Windows Explorer style)
            self.image_files = natsorted(image_files)

            # Find current file index
            self.current_file_index = -1

            # Normalize paths for comparison by converting to absolute paths
            # This handles both forward/backward slashes and case differences
            try:
                normalized_target = os.path.abspath(file_path)

                for i, img_file in enumerate(self.image_files):
                    normalized_img_file = os.path.abspath(img_file)
                    if normalized_target.lower() == normalized_img_file.lower():
                        self.current_file_index = i
                        break
            except Exception:
                # Fallback to original logic
                if file_path in self.image_files:
                    self.current_file_index = self.image_files.index(file_path)

            # Update status bar after scanning
            self.update_status_bar()

        except Exception as e:
            error_msg = f"Error scanning folder:\n{str(e)}"
            self.show_error("Folder Scan Error", error_msg)

    def show_error(self, title, message):
        """Show error message dialog"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()

    def extract_exif_data(self, file_path):
        """Extract EXIF data from image file"""
        exif_data = {
            'focal_length': None,
            'aperture': None,
            'iso': None,
            'capture_time': None
        }

        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Extract focal length
                if 'EXIF FocalLength' in tags:
                    focal_length_raw = tags['EXIF FocalLength']
                    try:
                        # Handle different focal length formats
                        focal_str = str(focal_length_raw)
                        if '/' in focal_str:
                            # Handle fraction format (e.g., "24/1")
                            num, den = focal_str.split('/')
                            focal_length = round(float(num) / float(den))
                        else:
                            # Handle decimal format
                            focal_length = round(float(focal_str))
                        exif_data['focal_length'] = f"{focal_length}mm"
                    except (ValueError, AttributeError, ZeroDivisionError):
                        pass

                # Extract aperture
                if 'EXIF FNumber' in tags:
                    aperture_raw = tags['EXIF FNumber']
                    try:
                        # Handle different aperture formats
                        aperture_str = str(aperture_raw)
                        if '/' in aperture_str:
                            # Handle fraction format (e.g., "28/10")
                            num, den = aperture_str.split('/')
                            aperture = float(num) / float(den)
                        else:
                            # Handle decimal format
                            aperture = float(aperture_str)
                        exif_data['aperture'] = f"f/{aperture:.1f}"
                    except (ValueError, AttributeError, ZeroDivisionError):
                        pass

                # Extract ISO
                if 'EXIF ISOSpeedRatings' in tags:
                    iso_raw = tags['EXIF ISOSpeedRatings']
                    try:
                        iso = int(str(iso_raw))
                        exif_data['iso'] = f"ISO {iso}"
                    except (ValueError, AttributeError):
                        pass

                # Extract capture time
                datetime_tags = ['EXIF DateTimeOriginal',
                                 'Image DateTime', 'EXIF DateTime']
                for tag_name in datetime_tags:
                    if tag_name in tags:
                        datetime_raw = tags[tag_name]
                        try:
                            datetime_str = str(datetime_raw)
                            # Parse datetime string (format: "YYYY:MM:DD HH:MM:SS")
                            dt = datetime.strptime(
                                datetime_str, "%Y:%m:%d %H:%M:%S")
                            # Format as "HH:MM:SS YYYY-MM-DD"
                            exif_data['capture_time'] = dt.strftime(
                                "%H:%M:%S %Y-%m-%d")
                            break  # Use first available datetime
                        except (ValueError, AttributeError):
                            continue

        except Exception:
            # If any error occurs during EXIF extraction, return empty data
            pass

        return exif_data

    def update_status_bar(self, width=None, height=None):
        """Update status bar with comprehensive information including EXIF data"""
        if not self.current_file_path:
            self.status_bar.showMessage("Ready")
            return

        # Get filename
        filename = os.path.basename(self.current_file_path)

        # Get image dimensions
        if width is None or height is None:
            if self.current_pixmap:
                width = self.current_pixmap.width()
                height = self.current_pixmap.height()
            else:
                width = height = 0

        # Get zoom level (removed "Fit to window" description)
        if self.fit_to_window:
            zoom_level = "Fit"
        else:
            zoom_level = f"{int(self.current_zoom_level * 100)}%"

        # Get file position
        file_position = ""
        if self.image_files and self.current_file_index >= 0:
            total_files = len(self.image_files)
            current_pos = self.current_file_index + 1
            file_position = f" - {current_pos} of {total_files}"

        # Extract EXIF data
        exif_data = self.extract_exif_data(self.current_file_path)

        # Build EXIF info string
        exif_info = []
        if exif_data['focal_length']:
            exif_info.append(exif_data['focal_length'])
        if exif_data['aperture']:
            exif_info.append(exif_data['aperture'])
        if exif_data['iso']:
            exif_info.append(exif_data['iso'])
        if exif_data['capture_time']:
            exif_info.append(exif_data['capture_time'])

        # Construct status message
        status_parts = []

        # Add filename and dimensions
        if width > 0 and height > 0:
            status_parts.append(f"{filename} - {width}x{height}")
        else:
            status_parts.append(filename)

        # Add zoom level
        status_parts.append(zoom_level)

        # Add EXIF info if available
        if exif_info:
            status_parts.extend(exif_info)

        # Add file position
        if file_position:
            status_parts.append(file_position.replace(" - ", ""))

        # Join all parts with separator
        status_msg = " - ".join(status_parts)

        self.status_bar.showMessage(status_msg)

    def _can_pan(self):
        # Only allow panning if the image is larger than the viewport
        if not self.current_pixmap:
            return False
        pixmap_size = self.image_label.pixmap().size()
        viewport_size = self.scroll_area.viewport().size()
        return pixmap_size.width() > viewport_size.width() or pixmap_size.height() > viewport_size.height()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Left:
                self.navigate_to_previous_image()
                return True
            elif key == Qt.Key.Key_Right:
                self.navigate_to_next_image()
                return True
            elif key == Qt.Key.Key_Up or key == Qt.Key.Key_Down:
                # Ignore up/down arrows to prevent panning
                return True
        return super().eventFilter(obj, event)

    def move_current_image_to_discard(self):
        """Move the current image to a 'Discard' folder in the same directory"""
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            self.show_error("Discard Error", "No image file to move.")
            return
        try:
            folder_path = os.path.dirname(self.current_file_path)
            discard_folder = os.path.join(folder_path, "Discard")
            os.makedirs(discard_folder, exist_ok=True)
            filename = os.path.basename(self.current_file_path)
            target_path = os.path.join(discard_folder, filename)
            # If file with same name exists in Discard, add a suffix
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(
                    discard_folder, f"{base}_discarded_{counter}{ext}")
                counter += 1
            os.rename(self.current_file_path, target_path)
            # Remove from image files list
            if self.current_file_path in self.image_files:
                self.image_files.remove(self.current_file_path)
            self.status_bar.showMessage(f"Moved to Discard: {filename}")
            # --- Preserve zoom/pan state for next image (like navigation) ---
            self._maintain_zoom_on_navigation = True
            if not self.fit_to_window:
                self._restore_zoom_center = self.zoom_center_point
                self._restore_zoom_level = self.current_zoom_level
                self._restore_start_scroll_x = self.start_scroll_x
                self._restore_start_scroll_y = self.start_scroll_y
            else:
                self._restore_zoom_center = None
                self._restore_zoom_level = None
                self._restore_start_scroll_x = None
                self._restore_start_scroll_y = None
            self.handle_post_deletion_navigation()
            self.save_session_state()
        except Exception as e:
            error_msg = f"Could not move file to Discard folder:\n{str(e)}"
            self.show_error("Discard Error", error_msg)

    def load_folder_images(self, folder_path, start_file=None):
        # Scan for images in the folder
        extensions = self.get_supported_extensions()
        files = [f for f in natsorted(os.listdir(folder_path))
                 if os.path.splitext(f)[1].lower() in extensions]
        if not files:
            self.show_error("No images found",
                            f"No supported images found in {folder_path}")
            return
        self.current_folder = folder_path
        self.image_files = files
        # Determine which image to start with
        if start_file and start_file in files:
            idx = files.index(start_file)
        else:
            idx = 0
        self.current_file_index = idx
        self.current_file_path = os.path.join(folder_path, files[idx])
        self.load_raw_image(self.current_file_path)
        self.save_session_state()

    def save_session_state(self):
        settings = self.get_settings()
        if self.current_folder and self.current_file_index >= 0 and self.image_files:
            filename = os.path.basename(
                self.image_files[self.current_file_index])
            settings.setValue("last_session_folder", self.current_folder)
            settings.setValue("last_session_file", filename)
        else:
            settings.remove("last_session_folder")
            settings.remove("last_session_file")

    def restore_session_state(self):
        settings = self.get_settings()
        folder = settings.value("last_session_folder", None)
        file = settings.value("last_session_file", None)
        if folder and file and os.path.isdir(folder):
            files = [f for f in natsorted(os.listdir(folder))
                     if os.path.splitext(f)[1].lower() in self.get_supported_extensions()]
            if file in files:
                self.load_folder_images(folder, start_file=file)
                return True
        return False

    def closeEvent(self, event):
        self.save_session_state()
        super().closeEvent(event)


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("RAW Image Viewer")
    app.setApplicationVersion("1.0")

    # Create and show main window
    viewer = RAWImageViewer()
    # Check for file argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            viewer.load_raw_image(file_path)
    viewer.show()

    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
