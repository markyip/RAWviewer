from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import pyqtSignal, Qt, QObject
from PyQt6.QtGui import QPixmap

class ThumbnailLabel(QLabel):
    """
    Thumbnail widget - keeps original pixmap and rescales cleanly.
    Based on reference implementation: simple and reliable.
    """
    def __init__(self, parent=None, pixmap=None):
        super().__init__(parent)
        self.original_pixmap = None
        self._is_internal_set = False
        
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use Fixed size policy for consistent layout - like 1.2.1 reference
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        if pixmap:
            self.setPixmap(pixmap)
        
    def setPixmap(self, pixmap):
        """Override to ensure scaling is always applied, preserving aspect ratio."""
        if not self._is_internal_set:
            self.original_pixmap = pixmap
            self._update_display()
        else:
            # When internal, we trust the pixmap is already scaled to label size
            super().setPixmap(pixmap)

    def _update_display(self, target_size=None):
        """Scales the original pixmap to fit the current widget size while preserving aspect ratio."""
        if self.original_pixmap and not self.original_pixmap.isNull():
            size = target_size if target_size else self.size()
            if size.width() > 0 and size.height() > 0:
                # PERFORMANCE: Only scale if size actually differs or it's first set
                # For now, always scale for reliability
                scaled = self.original_pixmap.scaled(
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self._is_internal_set = True
                self.setPixmap(scaled)
                self._is_internal_set = False
            else:
                # If size not ready, set a placeholder or original (clipped)
                # This will be fixed by the next resizeEvent
                self._is_internal_set = True
                super().setPixmap(self.original_pixmap)
                self._is_internal_set = False

    def resizeEvent(self, event):
        """Handle resize to rescale pixmap while preserving aspect ratio"""
        # CRITICAL: Use event.size() as it is the most accurate target
        super().resizeEvent(event)
        if self.original_pixmap and not self.original_pixmap.isNull():
            self._update_display(event.size())
            
    def get_original_pixmap(self):
        """Get the original pixmap"""
        return self.original_pixmap

class ImageLoaded(QObject):
    """Signal carrier for image loading - thread to UI communication"""
    # index (if needed), pixmap/image, generation
    loaded = pyqtSignal(int, object, int)
