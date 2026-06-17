import os
from PyQt6.QtWidgets import QLabel, QApplication
from PyQt6.QtCore import pyqtSignal, Qt, QObject, QPoint, QUrl, QMimeData
from PyQt6.QtGui import QPixmap, QDrag


class ThumbnailLabel(QLabel):
    """
    Thumbnail widget - keeps original pixmap and rescales cleanly.
    Based on reference implementation: simple and reliable.
    """
    clicked = pyqtSignal(str, object)  # file_path, QMouseEvent

    _STYLE_DEFAULT = "ThumbnailLabel { background: transparent; }"
    _STYLE_SELECTED = (
        "ThumbnailLabel { background: transparent; border: 3px solid #4A9EFF; }"
    )

    def __init__(self, parent=None, pixmap=None):
        super().__init__(parent)
        self.original_pixmap = None
        self._gallery_selected = False
        self._gallery_bookmarked = False
        self.file_path = None
        self._drag_start_pos = None
        self._drag_started = False

        # Optimize property settings for performance
        self.setScaledContents(False)  # We manually scale for better quality
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(self._STYLE_DEFAULT)

        if pixmap:
            self.set_original_pixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.file_path:
            self._drag_start_pos = event.position().toPoint()
            self._drag_started = False
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton) or not self.file_path:
            super().mouseMoveEvent(event)
            return

        if self._drag_start_pos is not None and not self._drag_started:
            dist = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._drag_started = True
                self._start_drag()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            drag_started = self._drag_started
            self._drag_start_pos = None
            self._drag_started = False
            if not drag_started and self.file_path:
                self.clicked.emit(self.file_path, event)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _start_drag(self):
        if not self.file_path or not os.path.exists(self.file_path):
            return
        
        # Check if the parent/viewer has a selection and if this thumbnail is part of it
        drag_paths = [self.file_path]
        try:
            # ThumbnailLabel -> JustifiedGallery -> RAWImageViewer
            pv = None
            p = self.parent()
            while p is not None:
                if p.__class__.__name__ == "RAWImageViewer":
                    pv = p
                    break
                p = p.parent()
            
            if pv is not None and hasattr(pv, "_is_gallery_path_selected") and hasattr(pv, "_gallery_selected_canonical_paths"):
                if pv._is_gallery_path_selected(self.file_path):
                    selected = pv._gallery_selected_canonical_paths()
                    if selected:
                        drag_paths = selected
        except Exception:
            pass

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(p) for p in drag_paths])
        drag.setMimeData(mime_data)
        
        px = self.pixmap()
        if px and not px.isNull():
            drag_px = px.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            drag.setPixmap(drag_px)
            drag.setHotSpot(QPoint(drag_px.width() // 2, drag_px.height() // 2))
            
        drag.exec(Qt.DropAction.CopyAction)

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

    def set_gallery_bookmarked(self, bookmarked: bool) -> None:
        self._gallery_bookmarked = bool(bookmarked)
        self._update_bookmark_badge()

    def _update_bookmark_badge(self) -> None:
        badge = getattr(self, "_bookmark_badge", None)
        if self._gallery_bookmarked:
            if badge is None:
                badge = QLabel("★", self)
                badge.setStyleSheet(
                    "color: #FFD700; font-size: 14px; font-weight: 700;"
                    " background: transparent; border: none;"
                )
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._bookmark_badge = badge
            badge.show()
            self._position_bookmark_badge()
        elif badge is not None:
            badge.hide()

    def _position_bookmark_badge(self) -> None:
        badge = getattr(self, "_bookmark_badge", None)
        if badge is None or not badge.isVisible():
            return
        size = 16
        margin = 2
        badge.setFixedSize(size, size)
        badge.move(
            max(0, self.width() - size - margin),
            max(0, self.height() - size - margin),
        )
        badge.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_bookmark_badge()


class ImageLoaded(QObject):
    """Signal carrier for image loading - thread to UI communication"""

    # index (if needed), pixmap/image, generation
    loaded = pyqtSignal(int, object, int)

