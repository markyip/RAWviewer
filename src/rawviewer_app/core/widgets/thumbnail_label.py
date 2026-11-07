"""ThumbnailLabel: keeps the original pixmap and rescales cleanly."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QSizePolicy


class ThumbnailLabel(QLabel):
    """
    Thumbnail widget - keeps original pixmap and rescales cleanly.
    Based on reference implementation: simple and reliable.
    """
    clicked = pyqtSignal(str) # file_path

    def __init__(self, parent=None, pixmap=None, file_path=None):
        super().__init__(parent)
        self.file_path = file_path
        self.original_pixmap = pixmap
        if pixmap:
            self.setPixmap(pixmap)
        else:
            self.setText("Loading...")  # Consistent with check in JustifiedGallery
        # Use setScaledContents(False) - like reference code for JustifiedGallery
        self.setScaledContents(False)
        # Use Fixed size policy - prevents layout from resizing
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.file_path:
            self.clicked.emit(self.file_path)
            event.accept()
        else:
            super().mousePressEvent(event)

    def set_original_pixmap(self, pixmap):
        """Store the original pixmap for rescaling"""
        self.original_pixmap = pixmap

    def get_original_pixmap(self):
        """Get the original pixmap"""
        return self.original_pixmap
