import os
from PyQt6.QtWidgets import QLabel, QApplication, QSlider
from PyQt6.QtCore import pyqtSignal, Qt, QObject, QPoint, QUrl, QMimeData, QPointF
from PyQt6.QtGui import QPixmap, QDrag, QPainter, QColor, QPolygonF

import theme

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
        f"ThumbnailLabel {{ background: transparent; border: 3px solid {theme.EMBER}; }}"
    )

    def __init__(self, parent=None, pixmap=None):
        super().__init__(parent)
        self.original_pixmap = None
        self._gallery_selected = False
        self._gallery_edited = False
        self._gallery_rating = 0
        self._burst_stack_count = 0
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
        selected = bool(selected)
        # setStyleSheet forces a full style re-polish/repaint -- one of the
        # more expensive Qt calls. refresh_gallery_selection_visuals() used
        # to call this unconditionally on every visible tile per click, so
        # a single shift/ctrl-click on a dense gallery view repainted tiles
        # whose selection state hadn't even changed. Skip the no-op case.
        if selected == self._gallery_selected:
            return
        self._gallery_selected = selected
        self.setStyleSheet(
            self._STYLE_SELECTED if self._gallery_selected else self._STYLE_DEFAULT
        )

    def set_gallery_edited(self, edited: bool) -> None:
        self._gallery_edited = bool(edited)
        self._update_edited_badge()

    def set_gallery_rating(self, rating: int) -> None:
        self._gallery_rating = max(0, min(5, int(rating or 0)))
        self._update_rating_badge()

    def _update_rating_badge(self) -> None:
        badge = getattr(self, "_rating_badge", None)
        rating = int(getattr(self, "_gallery_rating", 0) or 0)
        if rating >= 1:
            if badge is None:
                badge = QLabel(self)
                # Rating is a mark already decided about a photo -- dodge,
                # same family as the edited badge (rule 4).
                badge.setStyleSheet(
                    f"color: {theme.DODGE}; font-size: 9px; font-weight: 700;"
                    f" background: rgba({theme.VOID_RGB[0]}, {theme.VOID_RGB[1]}, {theme.VOID_RGB[2]}, 0.72);"
                    " border-radius: 3px; padding: 1px 4px;"
                )
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._rating_badge = badge
            badge.setText("★" * rating)
            badge.adjustSize()
            badge.show()
            self._position_rating_badge()
        elif badge is not None:
            badge.hide()

    def _position_rating_badge(self) -> None:
        badge = getattr(self, "_rating_badge", None)
        if badge is None or not badge.isVisible():
            return
        margin = 4
        badge.move(max(0, margin), max(0, self.height() - badge.height() - margin))
        badge.raise_()

    def set_burst_stack_count(self, count: int) -> None:
        self._burst_stack_count = max(0, int(count))
        self._update_burst_stack_badge()

    def _update_burst_stack_badge(self) -> None:
        badge = getattr(self, "_burst_stack_badge", None)
        count = int(getattr(self, "_burst_stack_count", 0) or 0)
        if count >= 2:
            if badge is None:
                badge = QLabel(self)
                # Burst count is informational, not a status mark or an active
                # state -- kept neutral (ink), not dodge or ember.
                badge.setStyleSheet(
                    f"color: {theme.INK}; font-size: 11px; font-weight: 700;"
                    f" background: rgba({theme.VOID_RGB[0]}, {theme.VOID_RGB[1]}, {theme.VOID_RGB[2]}, 0.72);"
                    f" border: 1px solid rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.25);"
                    " border-radius: 9px; padding: 0px 5px;"
                )
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._burst_stack_badge = badge
            badge.setText(f"×{count}")
            badge.show()
            self._position_burst_stack_badge()
        elif badge is not None:
            badge.hide()

    def _position_burst_stack_badge(self) -> None:
        badge = getattr(self, "_burst_stack_badge", None)
        if badge is None or not badge.isVisible():
            return
        badge.adjustSize()
        margin = 4
        badge.move(max(0, margin), max(0, margin))
        badge.raise_()

    def _update_edited_badge(self) -> None:
        """Lightweight 'has saved RAW adjustments' hint (pencil dot) — does not
        change the thumbnail pixels; see docs/EDIT_PIPELINE.md for why
        (gallery thumbnails are not re-rendered through the adjust pipeline)."""
        badge = getattr(self, "_edited_badge", None)
        if self._gallery_edited:
            if badge is None:
                badge = QLabel("✎", self)
                # Edited is a mark, not a live action -- dodge (rule 4).
                badge.setStyleSheet(
                    f"color: {theme.DODGE}; font-size: 12px; font-weight: 700;"
                    f" background: rgba({theme.VOID_RGB[0]}, {theme.VOID_RGB[1]}, {theme.VOID_RGB[2]}, 0.55);"
                    " border-radius: 8px;"
                )
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                badge.setToolTip("Has saved adjustments (Adjust panel / XMP sidecar)")
                self._edited_badge = badge
            badge.show()
            self._position_edited_badge()
        elif badge is not None:
            badge.hide()

    def _position_edited_badge(self) -> None:
        badge = getattr(self, "_edited_badge", None)
        if badge is None or not badge.isVisible():
            return
        size = 16
        margin = 2
        badge.setFixedSize(size, size)
        badge.move(max(0, self.width() - size - margin), max(0, margin))
        badge.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_burst_stack_badge()
        self._position_edited_badge()
        self._position_rating_badge()


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
    THUMB_RADIUS = 9.0

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

    def _track_bounds(self):
        """Horizontal track inset so the thumb circle is never clipped at the edges."""
        margin = self.THUMB_RADIUS
        x_start = margin
        x_end = max(x_start + 1.0, float(self.width()) - margin)
        y_center = self.height() / 2.0
        return x_start, x_end, y_center

    def _value_to_x(self, value):
        x_start, x_end, _ = self._track_bounds()
        span = self.maximum() - self.minimum()
        if span <= 0:
            return x_start
        fraction = (value - self.minimum()) / span
        return x_start + fraction * (x_end - x_start)

    def _x_to_value(self, x):
        x_start, x_end, _ = self._track_bounds()
        span = x_end - x_start
        if span <= 0:
            return self.minimum()
        fraction = (x - x_start) / span
        fraction = max(0.0, min(1.0, fraction))
        return int(self.minimum() + fraction * (self.maximum() - self.minimum()))

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
        return self._x_to_value(float(pos.x()))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        x_start, x_end, y_center = self._track_bounds()
        current_x = self._value_to_x(self.value())

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
        inactive_color = QColor(*theme.INK_RGB, 30)
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
        
        # Active color: neutral ink-muted, not ember -- resizing thumbnails
        # isn't a selection or an armed tool (rule 2 stays reserved for those).
        active_color = QColor(theme.INK_MUTED)
        painter.setBrush(active_color)
        painter.drawPolygon(active_poly)

        # Draw active start dot
        painter.setBrush(QColor(theme.INK_MUTED))
        painter.drawEllipse(QPointF(x_start, y_center), min_thick / 2.0, min_thick / 2.0)

        # Draw thumb handle
        painter.setBrush(QColor(theme.INK))
        thumb_radius = self.THUMB_RADIUS
        painter.drawEllipse(QPointF(current_x, y_center), thumb_radius, thumb_radius)

