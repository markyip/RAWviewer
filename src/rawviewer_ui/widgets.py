from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import pyqtSignal, Qt, QObject
from PyQt6.QtGui import QPixmap


class ThumbnailLabel(QLabel):
    """
    Thumbnail widget - keeps original pixmap and rescales cleanly.
    Based on reference implementation: simple and reliable.
    """

    _STYLE_DEFAULT = "ThumbnailLabel { background: transparent; }"
    _STYLE_SELECTED = (
        "ThumbnailLabel { background: transparent; border: 3px solid #4A9EFF; }"
    )

    def __init__(self, parent=None, pixmap=None):
        super().__init__(parent)
        self.original_pixmap = None
        self._gallery_selected = False

        # Optimize property settings for performance
        self.setScaledContents(False)  # We manually scale for better quality
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(self._STYLE_DEFAULT)

        if pixmap:
            self.set_original_pixmap(pixmap)

    def set_original_pixmap(self, pixmap):
        """Store the original pixmap for rescaling"""
        self.original_pixmap = pixmap
        if pixmap:
            self.setPixmap(pixmap)

    def get_original_pixmap(self):
        """Get the original pixmap"""
        return self.original_pixmap

    def set_gallery_selected(self, selected: bool) -> None:
        self._gallery_selected = bool(selected)
        self.setStyleSheet(
            self._STYLE_SELECTED if self._gallery_selected else self._STYLE_DEFAULT
        )


class ImageLoaded(QObject):
    """Signal carrier for image loading - thread to UI communication"""

    # index (if needed), pixmap/image, generation
    loaded = pyqtSignal(int, object, int)

