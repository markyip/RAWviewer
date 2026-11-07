'''PixmapConverter and RawEdrPixmapConverter.'''
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


class PixmapConverter(QThread):
    """Background thread to convert numpy array to QPixmap and cache it"""
    pixmap_ready = pyqtSignal(str, QPixmap)  # file_path, pixmap

    def __init__(self, file_path, rgb_image, image_cache):
        super().__init__()
        self.file_path = file_path
        # Not copied here: this class is a QThread, but __init__ runs
        # synchronously on whatever thread constructs it (always the main/
        # GUI thread for this class -- see on_manager_image_ready). A
        # full-resolution RAW buffer can be 100+MB; copying it here blocked
        # the main event loop for the copy's full duration on every image
        # display, confirmed via a live SIGUSR1/faulthandler stack dump
        # during a multi-second UI stall (main thread parked inside this
        # __init__). The copy still happens -- just moved to the start of
        # run(), on the background thread, which is what QThread is for.
        self.rgb_image = rgb_image
        self.image_cache = image_cache
        self._should_stop = False

    def stop_processing(self):
        """Request processing to stop"""
        self._should_stop = True

    def run(self):
        """Convert numpy array to QPixmap in background"""
        try:
            if self._should_stop:
                return

            # Copy here (background thread), not in __init__ (main thread).
            # See __init__ docstring comment for why this moved.
            if hasattr(self.rgb_image, 'copy'):
                self.rgb_image = self.rgb_image.copy()

            if not hasattr(self.rgb_image, 'shape'):
                if hasattr(self.rgb_image, 'width') and hasattr(self.rgb_image, 'height'):
                    height, width = self.rgb_image.height(), self.rgb_image.width()
                    channels = 3
                else:
                    return
            else:
                shape = self.rgb_image.shape
                height, width = shape[0], shape[1]
                channels = shape[2] if len(shape) > 2 else 1
                if self.rgb_image.dtype == np.uint16:
                    self.rgb_image = (self.rgb_image / 257.0).astype(np.uint8)
                elif np.issubdtype(self.rgb_image.dtype, np.floating):
                    from raw_adjustments.tone_recovery import _encode_srgb8
                    self.rgb_image = _encode_srgb8(
                        np.clip(self.rgb_image.astype(np.float32), 0.0, None)
                    )
                elif self.rgb_image.dtype != np.uint8:
                    self.rgb_image = np.clip(self.rgb_image, 0, 255).astype(np.uint8)

            bytes_per_line = channels * width

            # Ensure the data is contiguous
            if not self.rgb_image.flags['C_CONTIGUOUS']:
                self.rgb_image = np.ascontiguousarray(self.rgb_image)

            if self._should_stop:
                return

            # Create QImage directly from numpy array without copying memory
            q_format = QImage.Format.Format_RGB888
            if channels == 1:
                q_format = QImage.Format.Format_Grayscale8
            elif channels == 4:
                q_format = QImage.Format.Format_RGBA8888

            q_image = QImage(self.rgb_image.data, width, height,
                             bytes_per_line, q_format)
            q_image.ndarr = self.rgb_image  # Keep numpy array alive
            pixmap = QPixmap.fromImage(q_image)

            if self._should_stop:
                return

            # Cache the pixmap
            if self.image_cache:
                self.image_cache.put_pixmap(self.file_path, pixmap)

            # Emit signal if not stopped
            if not self._should_stop:
                self.pixmap_ready.emit(self.file_path, pixmap)

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in PixmapConverter for {self.file_path}: {e}")


class RawEdrPixmapConverter(QThread):
    """Background thread: linear RAW decode → macOS EDR RGBX64 QPixmap."""

    pixmap_ready = pyqtSignal(str, QPixmap)

    def __init__(
        self,
        file_path: str,
        *,
        max_edge: int = 0,
        exif_data=None,
    ):
        super().__init__()
        self.file_path = file_path
        self.max_edge = int(max_edge or 0)
        self.exif_data = exif_data
        self._should_stop = False

    def stop_processing(self):
        self._should_stop = True

    def run(self):
        try:
            if self._should_stop:
                return
            from common_image_loader import decode_raw_edr_pixmap

            pixmap = decode_raw_edr_pixmap(
                self.file_path,
                max_edge=self.max_edge,
                exif_data=self.exif_data,
            )
            if self._should_stop:
                return
            if pixmap is None or pixmap.isNull():
                self.pixmap_ready.emit(self.file_path, QPixmap())
                return
            self.pixmap_ready.emit(self.file_path, pixmap)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(
                "Error in RawEdrPixmapConverter for %s: %s",
                self.file_path,
                e,
            )
            if not self._should_stop:
                self.pixmap_ready.emit(self.file_path, QPixmap())

