"""gallery mixin, part 3 of 5 (8 methods, 881 lines)."""
from __future__ import annotations

import contextlib
import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from core import theme

from .. import lazy
from ..lazy import ThumbnailLabel, process_file_from_path

lazy.register(sys.modules[__name__])


class GalleryPart3:
    def _get_gallery_aspect_ratio(self, file_path):
        """
        Get aspect ratio for a file without loading the full image.
        Optimized to use cache, then EXIF tags, then RAW metadata.
        """
        import logging
        import os
        logging.getLogger(__name__)

        # 1. Check in-memory cache
        if file_path in self.gallery_aspect_cache:
            return self.gallery_aspect_cache[file_path]

        # 2. Try EXIF cache (fast)
        try:
            exif_data = self.image_cache.get_exif(file_path)
            if exif_data:
                w = exif_data.get('original_width')
                h = exif_data.get('original_height')
                orientation = exif_data.get('orientation', 1)

                if w and h and h > 0:
                    # Respect orientation (5,6,7,8 are rotated formats)
                    if orientation in (5, 6, 7, 8):
                        w, h = h, w
                    aspect = w / h
                    self.gallery_aspect_cache[file_path] = aspect
                    return aspect
        except Exception:
            pass

        # 3. Extract dimensions and orientation directly from file
        ext = os.path.splitext(file_path)[1].lower()
        raw_extensions = {'.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw',
                        '.pef', '.x3f', '.3fr', '.fff', '.iiq', '.cap', '.erf', '.mef', '.mos', '.nrw', '.srf'}
        is_raw = ext in raw_extensions

        # A. Try EXIF (pyexiv2 preferred, else exifread) for JPEG/TIFF and some RAW formats
        exif_likely_exts = {'.jpg', '.jpeg', '.tif', '.tiff', '.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.heic', '.heif'}
        if ext in exif_likely_exts:
            try:
                tags = process_file_from_path(
                    file_path, details=False, stop_tag="EXIF ExifImageWidth"
                )

                # Search for width and height in various tags
                w = h = None
                for w_tag in ['EXIF ExifImageWidth', 'Image ImageWidth']:
                    if w_tag in tags:
                        try:
                            w = int(str(tags[w_tag]))
                            break
                        except Exception:
                            pass

                for h_tag in ['EXIF ExifImageLength', 'Image ImageLength']:
                    if h_tag in tags:
                        try:
                            h = int(str(tags[h_tag]))
                            break
                        except Exception:
                            pass

                # Handle Orientation
                orientation = 1
                if 'Image Orientation' in tags:
                    try:
                        val = tags['Image Orientation'].values[0]
                        # Handle both integer and string orientation values
                        orientation = val if isinstance(val, int) else int(str(val))
                    except Exception:
                        pass

                if w and h and h > 0:
                    real_w, real_h = (h, w) if orientation in (5, 6, 7, 8) else (w, h)
                    aspect = real_w / real_h
                    self.gallery_aspect_cache[file_path] = aspect

                    # Update cache with un-swapped dimensions but correct orientation
                    try:
                        cached_exif = self.image_cache.get_exif(file_path) or {}
                        cached_exif['original_width'] = w
                        cached_exif['original_height'] = h
                        cached_exif['orientation'] = orientation
                        self.image_cache.put_exif(file_path, cached_exif)
                    except Exception:
                        pass
                    return aspect
            except Exception:
                pass

        # B. Try rawpy for RAW specific metadata (more reliable for some cameras)
        if is_raw:
            try:
                import rawpy
                with rawpy.imread(file_path) as raw:
                    # rawpy dimensions are usually the sensor dimensions (un-rotated)
                    w, h = raw.sizes.width, raw.sizes.height

                    # raw.sizes.flip contains rotation info (usually 0, 3, 5, or 6)
                    # 3 = 180, 5 = 90 CCW, 6 = 90 CW
                    flip = raw.sizes.flip

                    if flip in (5, 6): # Rotated 90 degrees
                        real_w, real_h = h, w
                        aspect = real_w / real_h
                    else:
                        aspect = w / h

                    self.gallery_aspect_cache[file_path] = aspect
                    # Update cache
                    try:
                        cached_exif = self.image_cache.get_exif(file_path) or {}
                        cached_exif['original_width'] = w
                        cached_exif['original_height'] = h
                        # Map rawpy flip to EXIF orientation for consistency if possible
                        # 0->1, 3->3, 6->6, 5->8 (approximate)
                        if flip == 6:
                            cached_exif['orientation'] = 6
                        elif flip == 3:
                            cached_exif['orientation'] = 3
                        elif flip == 5:
                            cached_exif['orientation'] = 8
                        else:
                            cached_exif['orientation'] = 1
                        self.image_cache.put_exif(file_path, cached_exif)
                    except Exception:
                        pass
                    return aspect
            except Exception:
                pass

        # C. Try PIL fallback (for TIFF/JPEG)
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                w, h = img.size
                # orientation handling in PIL
                orientation = 1
                try:
                    exif_getter = getattr(img, "_getexif", None)
                    exif = exif_getter() if callable(exif_getter) else None
                    if isinstance(exif, dict):
                        orientation = int(exif.get(274, 1))  # 274 is Orientation tag
                except Exception:
                    pass

                if h > 0:
                    real_w, real_h = (h, w) if orientation in (5, 6, 7, 8) else (w, h)
                    aspect = real_w / real_h
                    self.gallery_aspect_cache[file_path] = aspect
                    return aspect
        except Exception:
            pass

        return 1.333  # Final fallback

    def _get_gallery_pixmap(self, file_path):
        """Get pixmap for gallery view, loading if necessary - optimized for performance"""
        import logging
        import os
        logger = logging.getLogger(__name__)

        # Check cache first (fastest)
        if file_path in self.gallery_pixmaps:
            pixmap = self.gallery_pixmaps[file_path]
            if pixmap and not pixmap.isNull():
                return pixmap

        # Try to load from image cache (fast, already processed)
        # This includes pixmaps from single view mode, enabling smooth switching
        try:
            cached_pixmap = self.image_cache.get_pixmap(file_path)
            if cached_pixmap and not cached_pixmap.isNull():
                # Also store in gallery_pixmaps for faster subsequent access
                self.gallery_pixmaps[file_path] = cached_pixmap
                # Update aspect cache
                aspect = cached_pixmap.width() / cached_pixmap.height() if cached_pixmap.height() > 0 else 4.0 / 3.0
                self.gallery_aspect_cache[file_path] = aspect
                logger.debug(f"[GALLERY] Using cached pixmap from single view: {os.path.basename(file_path)}")
                return cached_pixmap
        except Exception as e:
            logger.debug(f"Error getting pixmap from cache for {file_path}: {e}")

        # Try to get thumbnail from thumbnail cache (faster than extracting)
        try:
            thumbnail_data = self.image_cache.get_thumbnail(file_path)
            if thumbnail_data is not None:
                # Use unified conversion method
                pixmap = self._numpy_to_qpixmap(thumbnail_data)
                if pixmap and not pixmap.isNull():
                    self.gallery_pixmaps[file_path] = pixmap
                    # Also cache in global image cache for smooth switching between views
                    # DO NOT cache in global image cache - this avoids polluting it with small gallery-sized pixmaps
                    # if they are just thumbnails.
                    # self.image_cache.put_pixmap(file_path, pixmap)
                    # Update aspect cache
                    aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                    self.gallery_aspect_cache[file_path] = aspect
                    return pixmap
        except Exception as e:
            logger.warning(f"Error getting thumbnail from cache for {file_path}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

        # For RAW files, try to extract embedded JPEG preview (fast)
        try:
            if os.path.exists(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw']
                if file_ext in raw_extensions:
                    pixmap = self._extract_embedded_preview(file_path)
                    if pixmap and not pixmap.isNull():
                        self.gallery_pixmaps[file_path] = pixmap
                        # Also cache in global image cache for smooth switching between views
                        # DO NOT cache in global image cache - this avoids polluting it with small previews
                        # self.image_cache.put_pixmap(file_path, pixmap)
                        # Update aspect cache
                        aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                        self.gallery_aspect_cache[file_path] = aspect
                        logger.debug(f"[GALLERY] Using embedded preview for gallery: {os.path.basename(file_path)}")
                        return pixmap
        except Exception as e:
            logger.debug(f"Error extracting embedded preview from RAW {file_path}: {e}")

        # For non-RAW files, try direct QPixmap load (fast for JPEG/PNG)
        try:
            if os.path.exists(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw']:
                    pixmap = self._load_pixmap_safe(file_path)
                    if not pixmap.isNull():
                        self.gallery_pixmaps[file_path] = pixmap
                        # Also cache in global image cache for smooth switching between views
                        # DO NOT cache in global image cache - this avoids polluting it with small previews
                        # self.image_cache.put_pixmap(file_path, pixmap)
                        # Update aspect cache
                        aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                        self.gallery_aspect_cache[file_path] = aspect
                        return pixmap
        except Exception as e:
            logger.debug(f"Error loading pixmap from file {file_path}: {e}")

        # Return None - will be loaded asynchronously
        # This allows layout to proceed without blocking
        return None

    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _add_grid_row(self, row_files, available_width, row_height, item_spacing, row_index=0, start_index=0):
    #     """Create a simple grid row with fixed height - basic gallery layout"""
    #     pass  # Gallery functionality disabled
        # Dummy assignments for old method parameters (gallery disabled)
        available_width, row_height, item_spacing, row_index, start_index = 0, 0, 0, 0, 0
        row_files, thumb_width = [], 0

        # Create row widget with fixed height
        row_widget = QWidget()
        row_widget.setFixedHeight(row_height)
        row_widget.setFixedWidth(available_width)
        row_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {theme.VOID};
                margin: 0px;
                padding: 0px;
            }}
        """)

        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(item_spacing)

        # Calculate item width (distribute available space evenly)
        num_items = len(row_files)
        total_spacing = item_spacing * (num_items - 1) if num_items > 1 else 0
        available_for_items = available_width - total_spacing
        item_width = int(available_for_items / num_items) if num_items > 0 else available_width

        for item_index, file_path in enumerate(row_files):
            # Create thumbnail label
            thumb_label = ThumbnailLabel()
            thumb_label.setFixedSize(item_width, row_height)
            thumb_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            thumb_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {theme.RAISED};
                    border: none;
                    margin: 0px;
                    padding: 0px;
                }}
            """)
            thumb_label.file_path = file_path

            # Store reference
            if not hasattr(self, '_gallery_thumb_labels'):
                self._gallery_thumb_labels = {}
            self._gallery_thumb_labels[file_path] = thumb_label

            # Make clickable
            thumb_label.mousePressEvent = lambda e, fp=file_path: self._gallery_item_clicked(fp)

            # Load thumbnail asynchronously with staggered delay to avoid blocking UI
            # Each item gets a slightly longer delay to spread out the loading
            global_index = start_index + item_index
            delay = 10 + (global_index * 5)  # Stagger delays: 10ms, 15ms, 20ms, etc.
            self._load_gallery_thumbnail_async_justified(file_path, thumb_label, item_width, row_height, delay)

            row_layout.addWidget(thumb_label)

        # Add row to gallery
        self.gallery_content_layout.addWidget(row_widget)

    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _add_justified_row(self, row_items, row_width, available_width, row_height, min_spacing, row_index=0):
    #     """Create a row of thumbnails with equal horizontal spacing between images"""
    #     pass  # Gallery functionality disabled
        # Dummy assignments for old method parameters (gallery disabled)
        row_items, _row_width, min_spacing, row_index = [], 0, 0, 0
        _target_width, target_height, thumb_width = 0, 0, 0

        # Calculate total width needed for all images at base scale
        total_images_width = sum(base_width for _, base_width in row_items)

        if total_images_width <= 0 or len(row_items) == 0:
            return

        # Calculate number of gaps between images (n-1 gaps for n images)
        num_gaps = len(row_items) - 1

        # Calculate equal spacing between images
        # Formula: available_width = scaled_total_width + (num_gaps * spacing_between)
        # We want to ensure spacing is at least min_spacing, then scale images to fit
        # If we have enough space, use min_spacing and scale images up
        # If we don't have enough space, use calculated spacing (which may be less than min_spacing)
        if num_gaps > 0:
            # Calculate maximum spacing we can use
            max_spacing = (available_width - total_images_width) / num_gaps if num_gaps > 0 else 0

            # Use the larger of min_spacing or calculated spacing
            # If we have enough space, use min_spacing and scale images to fill remaining space
            # If we don't have enough space, use calculated spacing (which may be less than min_spacing)
            if max_spacing >= min_spacing:
                # We have enough space, use min_spacing and scale images to fill remaining space
                total_spacing_width = min_spacing * num_gaps
                available_for_images = available_width - total_spacing_width
                scale_factor = available_for_images / total_images_width if total_images_width > 0 else 1.0
                spacing_between = min_spacing
            else:
                # Not enough space, use all available space for spacing and scale images down
                scale_factor = 1.0
                spacing_between = max_spacing if max_spacing > 0 else 0
        else:
            # Only one image, scale it to fit available width
            scale_factor = available_width / total_images_width if total_images_width > 0 else 1.0
            spacing_between = 0

        # Calculate row height: thumbnail height only (no histogram)
        # All items in the row should have the same total height
        first_item_height = int(row_height * scale_factor) if row_items else row_height
        total_row_height = first_item_height

        # Create row widget with fixed height and width to ensure no vertical spacing and no horizontal scroll
        row_widget = QWidget()
        row_widget.setFixedHeight(total_row_height)  # CRITICAL: Set fixed height to prevent spacing
        row_widget.setFixedWidth(available_width)  # CRITICAL: Set fixed width to prevent horizontal scroll
        row_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # Prevent resizing
        row_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {theme.VOID};
                margin: 0px;
                padding: 0px;
            }}
        """)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        row_layout.setSpacing(0)  # No spacing between items (we add spacers manually)

        for i, (file_path, base_width) in enumerate(row_items):
            # CRITICAL OPTIMIZATION: Don't load pixmap here during layout!
            # This is called for every image during layout calculation
            # Loading images synchronously here would block for 30+ seconds
            # Instead, create placeholder and load asynchronously

            # Compute final scaled size (maintain aspect ratio, no stretching)
            new_width = int(base_width * scale_factor)
            new_height = int(row_height * scale_factor)

            # Create thumbnail label
            thumb_label = ThumbnailLabel()
            # Set fixed size - this is the target size for the thumbnail
            thumb_label.setFixedSize(new_width, new_height)
            # Ensure size policy is Fixed (already set in ThumbnailLabel.__init__, but double-check)
            thumb_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            thumb_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {theme.RAISED};
                    border: none;
                    margin: 0px;
                    padding: 0px;
                }}
            """)
            # setScaledContents(False) is set in ThumbnailLabel.__init__
            # We will manually scale pixmap to exact size to prevent stretching

            # Store file path for async loading
            thumb_label.file_path = file_path
            # Store reference to thumb_label for later updates
            if not hasattr(self, '_gallery_thumb_labels'):
                self._gallery_thumb_labels = {}
            self._gallery_thumb_labels[file_path] = thumb_label

            # Make clickable
            thumb_label.mousePressEvent = lambda e, fp=file_path: self._gallery_item_clicked(fp)

            # Create item widget - set fixed height to match row
            item_widget = QWidget()
            item_widget.setFixedHeight(total_row_height)  # CRITICAL: Match row height exactly
            item_widget.setStyleSheet(f"""
                QWidget {{
                    background-color: {theme.VOID};
                    margin: 0px;
                    padding: 0px;
                }}
            """)
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)  # No margins
            item_layout.setSpacing(0)  # No spacing
            # CRITICAL: Align label to center to prevent stretching
            item_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            item_layout.addWidget(thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

            # Load thumbnail asynchronously
            self._load_gallery_thumbnail_async_justified(file_path, thumb_label, new_width, new_height)

            row_layout.addWidget(item_widget)

            # Add equal spacing after each image (except the last one)
            if i < len(row_items) - 1 and spacing_between > 0:
                spacer = QWidget()
                spacer.setFixedWidth(int(spacing_between))
                spacer.setFixedHeight(total_row_height)  # Match row height
                spacer.setStyleSheet("background-color: transparent; margin: 0px; padding: 0px;")
                row_layout.addWidget(spacer)

        # Add row widget with no spacing (vertical spacing is already 0 in gallery_content_layout)
        self.gallery_content_layout.addWidget(row_widget)
        # Store row widget reference
        self._gallery_row_widgets[row_index] = row_widget

        # Ensure no additional spacing is added
        self.gallery_content_layout.setSpacing(0)
        self.gallery_content_layout.setContentsMargins(0, 0, 0, 0)

    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _load_gallery_thumbnail_async_justified(self, file_path, thumb_label, target_width, target_height, delay=10):
    #     """Load thumbnail for justified gallery item asynchronously"""
    #     pass  # Gallery functionality disabled
        import logging
        import os
        import time

        from PyQt6.QtCore import QTimer
        logger = logging.getLogger(__name__)

        # Validate inputs
        if not file_path or not thumb_label:
            logger.warning(f"[GALLERY] Invalid inputs for thumbnail load: file_path={file_path}, thumb_label={thumb_label}")
            return

        # Track loading start time
        if file_path in self._gallery_load_tracking:
            self._gallery_load_tracking[file_path]['start_time'] = time.time()
        else:
            logger.warning(f"[GALLERY] File {os.path.basename(file_path)} not in load tracking")

        # Store target dimensions in local variables to ensure they're available in closure
        final_target_height = target_height

        def load_thumbnail():
            load_start = time.time()
            # Use local variables for target dimensions
            local_target_height = final_target_height

            try:
                logger.debug(f"[GALLERY] Starting load for {os.path.basename(file_path)}")
                # Get or load pixmap
                pixmap = self._get_gallery_pixmap(file_path)
                if not pixmap or pixmap.isNull():
                    # Try to load asynchronously if not in cache
                    logger.debug(f"[GALLERY] Pixmap not in cache for {os.path.basename(file_path)}, loading async")
                    if file_path in self._gallery_load_tracking:
                        self._gallery_load_tracking[file_path]['loaded'] = False
                    # Load pixmap asynchronously - it will update the label when ready
                    self._load_gallery_pixmap_async(file_path)
                    # Return early - label will be updated by _load_gallery_pixmap_async
                    return

                # Mark as loaded
                load_time = time.time() - load_start
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['loaded'] = True
                    self._gallery_load_tracking[file_path]['load_time'] = load_time
                logger.info(f"[GALLERY] [PIXMAP] {os.path.basename(file_path)} - Loaded from cache in {load_time:.3f}s")

                # Get actual image dimensions
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                if pixmap_width <= 0 or pixmap_height <= 0:
                    logger.warning(f"[GALLERY] Invalid pixmap size for {os.path.basename(file_path)}: {pixmap_width}x{pixmap_height}")
                    return

                # Update aspect cache with actual image dimensions
                actual_aspect = pixmap_width / pixmap_height
                self.gallery_aspect_cache[file_path] = actual_aspect

                # Log original image dimensions and aspect ratio
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Original: {pixmap_width}x{pixmap_height}, Aspect: {actual_aspect:.3f}")

                # For fixed height gallery: scale to fixed height while preserving aspect ratio
                # Reference code: resized = pixmap.scaled(QSize(w, ROW_HEIGHT), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # We use the target_height as the fixed constraint (this is the row_height from layout)
                target_height = local_target_height if local_target_height > 0 else self.gallery_row_height

                # Get label size for logging
                label_width = thumb_label.width()
                label_height = thumb_label.height()
                label_aspect = label_width / label_height if label_height > 0 else 0
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Label: {label_width}x{label_height}, Label Aspect: {label_aspect:.3f}")

                # Calculate width based on actual image aspect ratio and fixed height
                scaled_width = int(target_height * actual_aspect)
                scaled_height = target_height
                scaled_aspect = scaled_width / scaled_height if scaled_height > 0 else 0
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Target: {scaled_width}x{scaled_height}, Scaled Aspect: {scaled_aspect:.3f}")

                # Scale pixmap to fixed height while preserving aspect ratio
                scaled_pixmap = pixmap.scaled(
                    QSize(scaled_width, scaled_height),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                if scaled_pixmap.isNull():
                    logger.warning(f"[GALLERY] Failed to scale pixmap for {os.path.basename(file_path)}")
                    return

                # Log actual scaled pixmap dimensions
                actual_scaled_width = scaled_pixmap.width()
                actual_scaled_height = scaled_pixmap.height()
                actual_scaled_aspect = actual_scaled_width / actual_scaled_height if actual_scaled_height > 0 else 0
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Result: {actual_scaled_width}x{actual_scaled_height}, Result Aspect: {actual_scaled_aspect:.3f}")

                # CRITICAL: Resize label to match scaled pixmap size (like reference code: thumb.setFixedSize(resized.size()))
                # This ensures the label matches the pixmap dimensions exactly, preventing compression
                # The label size must match the pixmap size to avoid stretching/compression
                display_start = time.time()
                thumb_label.setFixedSize(actual_scaled_width, actual_scaled_height)
                thumb_label.setPixmap(scaled_pixmap)
                thumb_label.set_original_pixmap(pixmap)

                # Log final state
                final_label_width = thumb_label.width()
                final_label_height = thumb_label.height()
                final_pixmap_width = scaled_pixmap.width()
                final_pixmap_height = scaled_pixmap.height()
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Final Label: {final_label_width}x{final_label_height}, Pixmap: {final_pixmap_width}x{final_pixmap_height}")
                display_time = time.time() - display_start

                # Mark as displayed
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['displayed'] = True
                    self._gallery_load_tracking[file_path]['display_time'] = display_time
                    total_time = time.time() - self._gallery_load_tracking[file_path]['start_time']
                    logger.info(f"[GALLERY] [IMAGE] {os.path.basename(file_path)} - Loaded in {load_time:.3f}s, Displayed in {display_time:.3f}s, Total: {total_time:.3f}s")
            except Exception as e:
                logger.warning(f"Error loading justified thumbnail for {os.path.basename(file_path)}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['loaded'] = False
                    self._gallery_load_tracking[file_path]['displayed'] = False

        # Use a small delay to allow UI to update first
        QTimer.singleShot(10, load_thumbnail)

    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _load_gallery_pixmap_async(self, file_path):
    #     """Load pixmap asynchronously and update specific thumbnail when ready"""
    #     pass  # Gallery functionality disabled
        import logging
        import time

        from PIL import Image
        from PyQt6.QtCore import QTimer
        logger = logging.getLogger(__name__)

        # Track loading start time if not already tracked
        if file_path not in self._gallery_load_tracking:
            self._gallery_load_tracking[file_path] = {
                'start_time': time.time(),
                'loaded': False,
                'displayed': False,
                'load_time': None,
                'display_time': None
            }
        elif self._gallery_load_tracking[file_path]['start_time'] is None:
            self._gallery_load_tracking[file_path]['start_time'] = time.time()

        def load_pixmap():
            load_start = time.time()
            try:
                # Check if already loaded by another thread/method (avoid duplicate loading)
                if file_path in self.gallery_pixmaps:
                    pixmap = self.gallery_pixmaps[file_path]
                    if pixmap and not pixmap.isNull():
                        # Already loaded, skip reloading but still update label if needed
                        load_time = time.time() - load_start
                        if file_path in self._gallery_load_tracking and not self._gallery_load_tracking[file_path]['loaded']:
                                self._gallery_load_tracking[file_path]['loaded'] = True
                                self._gallery_load_tracking[file_path]['load_time'] = load_time
                        # Continue to update label below
                        # (label update logic is at the end of this function)
                    else:
                        # Pixmap exists but is null, try to reload
                        pixmap = None

                # Try to load from cache or file if not already loaded
                if not pixmap or pixmap.isNull():
                    pixmap = self._get_gallery_pixmap(file_path)

                # If not in cache, try to load directly

                if (not pixmap or pixmap.isNull()) and os.path.exists(file_path):
                        file_ext = os.path.splitext(file_path)[1].lower()
                        raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw']

                        # For RAW files, try to extract embedded JPEG preview (fast)
                        if file_ext in raw_extensions:
                            try:
                                pixmap = self._extract_embedded_preview(file_path)
                                if pixmap and not pixmap.isNull():
                                    logger.debug(f"[GALLERY] Loaded embedded preview for gallery: {os.path.basename(file_path)}")
                            except Exception as raw_error:
                                logger.debug(f"Error extracting embedded preview for {os.path.basename(file_path)}: {raw_error}")
                                pixmap = None

                        # For non-RAW files, try direct QPixmap load
                        if (not pixmap or pixmap.isNull()) and file_ext not in raw_extensions:
                            pixmap = self._load_pixmap_safe(file_path)
                            if pixmap.isNull():
                                # If QPixmap fails, try PIL Image
                                try:
                                    pil_image = Image.open(file_path)
                                    if pil_image.mode != 'RGB':
                                        pil_image = pil_image.convert('RGB')
                                    # Convert PIL Image to QPixmap
                                    width, height = pil_image.size
                                    image_bytes = pil_image.tobytes('raw', 'RGB')
                                    # CRITICAL: Calculate bytes_per_line for PIL Image
                                    bytes_per_line = 3 * width  # RGB = 3 channels
                                    qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                                    pixmap = QPixmap.fromImage(qimage) if not qimage.isNull() else None
                                except Exception as pil_error:
                                    logger.debug(f"Error loading with PIL for {os.path.basename(file_path)}: {pil_error}")
                                    pixmap = None

                        # If still no pixmap, try to get thumbnail from image cache
                        if (not pixmap or pixmap.isNull()) and hasattr(self, 'image_cache'):
                            try:
                                thumbnail_data = self.image_cache.get_thumbnail(file_path)
                                if thumbnail_data is not None:
                                    # Use unified conversion method
                                    pixmap = self._numpy_to_qpixmap(thumbnail_data)
                            except Exception as thumb_error:
                                logger.debug(f"Error getting thumbnail from cache for {os.path.basename(file_path)}: {thumb_error}")
                                import traceback
                                logger.debug(f"Traceback: {traceback.format_exc()}")

                if pixmap and not pixmap.isNull():
                    # Store in gallery cache
                    self.gallery_pixmaps[file_path] = pixmap
                    # Also cache in global image cache for smooth switching between views
                    with contextlib.suppress(Exception):
                        self.image_cache.put_pixmap(file_path, pixmap)
                    # Update aspect cache
                    aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                    self.gallery_aspect_cache[file_path] = aspect
                    load_time = time.time() - load_start

                    # Mark as loaded
                    if file_path in self._gallery_load_tracking:
                        self._gallery_load_tracking[file_path]['loaded'] = True
                        self._gallery_load_tracking[file_path]['load_time'] = load_time
                        logger.debug(f"[GALLERY] [PIXMAP] {os.path.basename(file_path)} - Loaded pixmap in {load_time:.3f}s")

                    # Update the specific thumbnail label directly instead of refreshing entire gallery
                    if hasattr(self, '_gallery_thumb_labels') and file_path in self._gallery_thumb_labels:
                        thumb_label = self._gallery_thumb_labels[file_path]
                        if thumb_label:
                            # Get actual image dimensions
                            pixmap_width = pixmap.width()
                            pixmap_height = pixmap.height()
                            if pixmap_width <= 0 or pixmap_height <= 0:
                                return

                            # Update aspect cache with actual image dimensions
                            actual_aspect = pixmap_width / pixmap_height
                            self.gallery_aspect_cache[file_path] = actual_aspect

                            # Log original image dimensions and aspect ratio
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Original: {pixmap_width}x{pixmap_height}, Aspect: {actual_aspect:.3f}")

                            # For fixed height gallery: scale to fixed height while preserving aspect ratio
                            # Reference code: resized = pixmap.scaled(QSize(w, ROW_HEIGHT), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            # Get label's fixed height (row_height)
                            label_height = thumb_label.height()
                            if label_height <= 0:
                                label_height = self.gallery_row_height

                            label_width = thumb_label.width()
                            label_aspect = label_width / label_height if label_height > 0 else 0
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Label: {label_width}x{label_height}, Label Aspect: {label_aspect:.3f}")

                            # Calculate width based on actual image aspect ratio and fixed height
                            scaled_width = int(label_height * actual_aspect)
                            scaled_height = label_height
                            scaled_aspect = scaled_width / scaled_height if scaled_height > 0 else 0
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Target: {scaled_width}x{scaled_height}, Scaled Aspect: {scaled_aspect:.3f}")

                            # Scale pixmap to fixed height while preserving aspect ratio
                            scaled_pixmap = pixmap.scaled(
                                QSize(scaled_width, scaled_height),
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation
                            )

                            if scaled_pixmap.isNull():
                                logger.warning(f"[GALLERY] Failed to scale pixmap for {os.path.basename(file_path)}")
                                return

                            # Log actual scaled pixmap dimensions
                            actual_scaled_width = scaled_pixmap.width()
                            actual_scaled_height = scaled_pixmap.height()
                            actual_scaled_aspect = actual_scaled_width / actual_scaled_height if actual_scaled_height > 0 else 0
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Result: {actual_scaled_width}x{actual_scaled_height}, Result Aspect: {actual_scaled_aspect:.3f}")

                            # CRITICAL: Resize label to match scaled pixmap size (like reference code: thumb.setFixedSize(resized.size()))
                            # This ensures the label matches the pixmap dimensions exactly, preventing compression
                            # The label size must match the pixmap size to avoid stretching/compression
                            display_start = time.time()
                            thumb_label.setFixedSize(actual_scaled_width, actual_scaled_height)
                            thumb_label.setPixmap(scaled_pixmap)
                            thumb_label.set_original_pixmap(pixmap)

                            # Log final state
                            final_label_width = thumb_label.width()
                            final_label_height = thumb_label.height()
                            final_pixmap_width = scaled_pixmap.width()
                            final_pixmap_height = scaled_pixmap.height()
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Final Label: {final_label_width}x{final_label_height}, Pixmap: {final_pixmap_width}x{final_pixmap_height}")
                            display_time = time.time() - display_start

                            # Mark as displayed
                            if file_path in self._gallery_load_tracking:
                                self._gallery_load_tracking[file_path]['displayed'] = True
                                self._gallery_load_tracking[file_path]['display_time'] = display_time
                                total_time = time.time() - self._gallery_load_tracking[file_path]['start_time']
                                logger.debug(f"[GALLERY] [IMAGE] {os.path.basename(file_path)} - Displayed in {display_time:.3f}s, Total: {total_time:.3f}s")
                else:
                    # Failed to load
                    logger.warning(f"[GALLERY] [PIXMAP] Failed to load pixmap for {os.path.basename(file_path)}")
                    if file_path in self._gallery_load_tracking:
                        self._gallery_load_tracking[file_path]['loaded'] = False
                        self._gallery_load_tracking[file_path]['displayed'] = False
            except Exception as e:
                logger.warning(f"Error loading pixmap async for {os.path.basename(file_path)}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['loaded'] = False
                    self._gallery_load_tracking[file_path]['displayed'] = False

        QTimer.singleShot(50, load_pixmap)

    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _create_gallery_item(self, file_path, thumb_width=200):
    #     """Create a single gallery item with thumbnail"""
    #     pass  # Gallery functionality disabled

        from PyQt6.QtWidgets import QLabel


        item = QWidget()
        item.setStyleSheet(f"""
            QWidget {{
                background-color: {theme.VOID};
            }}
        """)
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)  # No margins for tight packing
        item_layout.setSpacing(0)  # No spacing

        # Thumbnail label
        thumb_label = QLabel()
        thumb_height = int(thumb_width * 0.75)  # 4:3 aspect ratio
        thumb_label.setFixedSize(thumb_width, thumb_height)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setStyleSheet(f"""
            QLabel {{
                background-color: {theme.RAISED};
                border: none;
            }}
        """)
        # CRITICAL: Use setScaledContents(False) to prevent stretching
        thumb_label.setScaledContents(False)
        thumb_label.setText("")  # No placeholder text for cleaner look

        item_layout.addWidget(thumb_label)

        # Make item clickable
        thumb_label.clicked.connect(self._gallery_item_clicked)

        # Store references
        item.thumb_label = thumb_label
        item.file_path = file_path

        return item

    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _load_gallery_thumbnail_async(self, file_path, item_widget):
    #     """Load thumbnail for gallery item asynchronously (non-blocking)"""
    #     pass  # Gallery functionality disabled
        # Use QThread or simple delayed loading to avoid blocking UI
        from PyQt6.QtCore import QTimer

        def load_thumbnail():
            try:
                self._load_gallery_thumbnail(file_path, item_widget.thumb_label)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Error in async thumbnail load: {e}")

        # Use a small delay to allow UI to update first
        QTimer.singleShot(10, load_thumbnail)

    def _gallery_event_modifiers(self, event=None):
        """Reliable modifier read for gallery clicks (macOS press events often omit Cmd)."""
        from PyQt6.QtWidgets import QApplication

        mods = Qt.KeyboardModifier.NoModifier
        if event is not None:
            mods |= event.modifiers()
        mods |= QApplication.keyboardModifiers()
        return mods

    def _gallery_has_extend_modifier(self, modifiers) -> bool:
        """Ctrl (Windows/Linux) or Cmd (macOS) for gallery extend-selection."""
        return bool(
            modifiers
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )

    def _gallery_has_shift_modifier(self, modifiers) -> bool:
        return bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

    def _gallery_refresh_selection_ui(self) -> None:
        gj = getattr(self, "gallery_justified", None)
        if gj is not None and hasattr(gj, "refresh_gallery_selection_visuals"):
            gj.refresh_gallery_selection_visuals()
        self._sync_gallery_selection_chrome()
        self._sync_share_button_visibility()
        self._sync_compare_button_visibility()

    def _gallery_set_selection_keys(self, keys: set) -> None:
        self._gallery_selected_paths = set(keys)
        self._gallery_refresh_selection_ui()

    def _gallery_selection_ordered_paths(self) -> list[str]:
        """Ordered paths used for Shift+click range selection (matches on-screen gallery)."""
        if getattr(self, "view_mode", "") == "gallery":
            gj = getattr(self, "gallery_justified", None)
            if gj is not None:
                visible = list(getattr(gj, "images", []) or [])
                if visible:
                    return visible
            return self._gallery_files_for_display()
        return list(getattr(self, "image_files", []) or [])
