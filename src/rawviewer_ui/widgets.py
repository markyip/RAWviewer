import os
from PyQt6.QtWidgets import QLabel, QApplication, QSlider, QStyle, QStyleOptionSlider
from PyQt6.QtCore import pyqtSignal, Qt, QObject, QPoint, QUrl, QMimeData, QPointF
from PyQt6.QtGui import QPixmap, QDrag, QPainter, QColor, QPolygonF

RAWVIEWER_INTERNAL_FILE_DRAG_MIME = "application/x-rawviewer-internal-file-drag"


def stamp_rawviewer_export_drag(mime_data: QMimeData) -> None:
    """Mark drags started inside RAWviewer so drops on the app do not reopen files."""
    mime_data.setData(RAWVIEWER_INTERNAL_FILE_DRAG_MIME, b"1")


def is_rawviewer_export_drag(mime_data) -> bool:
    try:
        return mime_data is not None and mime_data.hasFormat(
            RAWVIEWER_INTERNAL_FILE_DRAG_MIME
        )
    except Exception:
        return False


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
        self.setScaledContents(True)   # Enable scaling for smooth thumbnail transitions during zoom
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
        stamp_rawviewer_export_drag(mime_data)
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


class GalleryZoomSlider(QSlider):
    """
    Custom-styled slider with a slanted wedge-shaped track and circular thumb handle.
    Represents thumbnail sizes scaling from smaller to larger.
    """
    ZOOM_STEP = 20

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet("background: transparent; border: none;")
        if orientation == Qt.Orientation.Horizontal:
            self.setFixedHeight(28)
            self.setMinimumWidth(120)
            self.setMaximumWidth(180)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSingleStep(self.ZOOM_STEP)
        self.setPageStep(self.ZOOM_STEP)

    def setValue(self, val):
        min_val = self.minimum()
        max_val = self.maximum()
        
        # Calculate possible step values
        steps = list(range(min_val, max_val + 1, self.ZOOM_STEP))
        if not steps or steps[-1] != max_val:
            steps.append(max_val)
            
        # Find closest step
        snapped = min(steps, key=lambda x: abs(x - val))
        old = self.value()
        super().setValue(snapped)
        if snapped != old:
            self.valueChanged.emit(snapped)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self.sliderPressed.emit()
            val = self._value_for_pos(event.pos())
            self.setValue(val)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            val = self._value_for_pos(event.pos())
            self.setValue(val)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(False)
            self.sliderReleased.emit()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _value_for_pos(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
        )
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )
        
        hw = handle_rect.width()
        x_start = groove_rect.left() + hw // 2
        x_end = groove_rect.right() - hw // 2
        
        if x_end <= x_start:
            return self.minimum()
            
        fraction = (pos.x() - x_start) / (x_end - x_start)
        fraction = max(0.0, min(1.0, fraction))
        
        return int(self.minimum() + fraction * (self.maximum() - self.minimum()))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        # Get subcontrol rects
        groove_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
        )
        handle_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )

        # Use widget mid-point for y_center: the native Windows style engine may
        # report a groove_rect centre that is offset from the actual widget
        # centre, causing the track and thumb to appear visually off-centre.
        y_center = self.height() // 2
        
        # Calculate horizontal range for the wedge (center of handle at min to center of handle at max)
        hw = handle_rect.width()
        x_start = groove_rect.left() + hw // 2
        x_end = groove_rect.right() - hw // 2
        current_x = handle_rect.center().x()

        # Clamp current_x to [x_start, x_end] to prevent division by zero or out of bounds
        if x_end <= x_start:
            x_start = 10
            x_end = self.width() - 10
        current_x = max(x_start, min(current_x, x_end))

        # Wedge thickness configurations
        min_thick = 3.0
        max_thick = 12.0

        # Calculate current thickness at current_x
        fraction = 0.0
        if x_end > x_start:
            fraction = (current_x - x_start) / (x_end - x_start)
        current_thick = min_thick + (max_thick - min_thick) * fraction

        # Draw inactive part (from current_x to x_end)
        inactive_poly = QPolygonF([
            QPointF(current_x, y_center - current_thick / 2.0),
            QPointF(x_end, y_center - max_thick / 2.0),
            QPointF(x_end, y_center + max_thick / 2.0),
            QPointF(current_x, y_center + current_thick / 2.0)
        ])
        
        # Inactive color: subtle dark-mode transparent gray
        inactive_color = QColor(255, 255, 255, 30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(inactive_color)
        painter.drawPolygon(inactive_poly)

        # Draw active part (from x_start to current_x)
        active_poly = QPolygonF([
            QPointF(x_start, y_center - min_thick / 2.0),
            QPointF(current_x, y_center - current_thick / 2.0),
            QPointF(current_x, y_center + current_thick / 2.0),
            QPointF(x_start, y_center + min_thick / 2.0)
        ])
        
        # Active color: clean white to match overall style
        active_color = QColor("#E0E0E0")
        painter.setBrush(active_color)
        painter.drawPolygon(active_poly)

        # Draw active start dot (small white dot)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(QPointF(x_start, y_center), min_thick / 2.0, min_thick / 2.0)

        # Draw thumb handle (solid white circle)
        painter.setBrush(QColor("#FFFFFF"))
        thumb_radius = 9.0
        painter.drawEllipse(QPointF(current_x, y_center), thumb_radius, thumb_radius)

