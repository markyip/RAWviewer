"""GPU-accelerated single-image view (Route B).

A self-contained ``QGraphicsView`` that renders one image and performs fit / zoom /
pan as GPU matrix transforms instead of re-scaling the pixmap on the CPU every frame.
This gives QuickLook-style smoothness for wheel/pinch zoom and drag pan.

The optional ``QOpenGLWidget`` viewport is vendor-agnostic (NVIDIA / AMD / Intel on
Windows; Qt typically uses a Metal-backed GL stack on macOS).

Opt-in via the ``RAWVIEWER_GPU_VIEW=1`` environment variable; the legacy
``QScrollArea`` + ``QLabel`` path remains the default. The widget is intentionally
decoupled from the main window: it exposes a small API plus a few signals so the
host can keep navigation / status / histogram behaviour identical.

Environment toggles:
- ``RAWVIEWER_GPU_VIEW=1``       Enable this view in the main window.
- ``RAWVIEWER_GPU_VIEW_NO_GL=1`` Use the raster viewport (debug / fallback).
"""

import os

from PyQt6.QtCore import Qt, QRect, QRectF, QPointF, pyqtSignal, QEvent
from PyQt6.QtGui import QKeyEvent, QPixmap, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QLabel,
)


def _env_true(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


class GpuImageView(QGraphicsView):
    """Hardware-accelerated single image viewport.

    Coordinate model: scene units == image pixels. ``transform().m11()`` is therefore
    "view pixels per image pixel"; a scale of 1.0 means 100% (actual pixels).
    """

    # bool: True when showing fit-to-window, False when zoomed
    fitModeChanged = pyqtSignal(bool)
    # float: current scale (view px per image px); 1.0 == 100%
    zoomChanged = pyqtSignal(float)
    # int: +1 request next image, -1 request previous image (plain wheel in fit mode)
    wheelNavigate = pyqtSignal(int)
    # Image pixel coordinates (scene space) where the user double-clicked.
    doubleClickedAt = pyqtSignal(QPointF)

    # Absolute floor for set_scale; wheel zoom cannot go below fit_scale() (fit-to-window).
    MIN_SCALE = 0.01
    # Pinch / wheel cap (400%). Space and double-click use zoom_to_actual() at 100%.
    MAX_SCALE = 4.0
    _FIT_SCALE_EPS = 1.002  # treat within ~0.2% of fit as fit-to-window

    def __init__(self, parent=None, background="#1E1E1E"):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)

        # Dashed focus / subject overlay rectangle (scene coords == image pixels). Uses a
        # cosmetic pen so the on-screen line width stays constant at any zoom level.
        self._overlay_item = None

        self._fit_mode = True
        self._zoom_intent_100 = False
        self._has_pixmap = False
        self._img_w = 0
        self._img_h = 0

        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setBackgroundBrush(QColor(background))
        self.setStyleSheet(f"border: none; background-color: {background};")
        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True
        )
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Set by RAWImageViewer: callable(QKeyEvent) -> bool
        self._shortcut_handler = None

        self._maybe_enable_opengl()

        # Centered placeholder text shown when no image is loaded (parity with the
        # legacy QLabel instruction screen, which the GPU view would otherwise cover).
        self._placeholder = QLabel(self.viewport())
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "QLabel { color: #666; font-size: 14px; background-color: transparent; }"
        )
        self._placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._placeholder.hide()

    # ------------------------------------------------------------------ setup
    def _maybe_enable_opengl(self) -> None:
        if _env_true("RAWVIEWER_GPU_VIEW_NO_GL"):
            return
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            gl = QOpenGLWidget()
            gl.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setViewport(gl)
            # Re-enable tracking on the new viewport.
            self.viewport().setMouseTracking(True)
            self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception:
            # Raster fallback keeps the feature working without a GL context.
            pass

    # ------------------------------------------------------------------ state
    def has_pixmap(self) -> bool:
        return self._has_pixmap

    def is_fit_mode(self) -> bool:
        return self._fit_mode

    def is_at_fit_scale(self) -> bool:
        """True when the view transform matches fit-to-window (within tolerance)."""
        if not self._has_pixmap:
            return True
        return self.current_scale() <= self.fit_scale() * self._FIT_SCALE_EPS

    def wants_zoom_in_toggle(self) -> bool:
        """Whether Space / double-click should enter 100% zoom (not zoom out to fit)."""
        return self._fit_mode or self.is_at_fit_scale()

    def _sync_fit_mode_flag(self) -> None:
        """Align ``_fit_mode`` with the live transform (fixes stale-flag toggle bugs)."""
        if not self._has_pixmap:
            return
        if self.is_at_fit_scale() and not getattr(self, "_zoom_intent_100", False):
            self._fit_mode = True
        elif self.current_scale() >= 1.0 - 1e-4:
            self._fit_mode = False

    def current_scale(self) -> float:
        return float(self.transform().m11())

    def fit_scale(self) -> float:
        """View pixels per image pixel when the image is scaled to fit the viewport."""
        if not self._has_pixmap or self._img_w <= 0 or self._img_h <= 0:
            return self.MIN_SCALE
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        return min(vw / self._img_w, vh / self._img_h)

    def image_size(self):
        return (self._img_w, self._img_h)

    # ------------------------------------------------------------- placeholder
    def set_placeholder_text(self, text: str) -> None:
        self._placeholder.setText(text or "")
        self._update_placeholder()

    def _update_placeholder(self) -> None:
        if self._has_pixmap or not self._placeholder.text():
            self._placeholder.hide()
            return
        vp = self.viewport().rect()
        margin = 24
        self._placeholder.setGeometry(
            margin, margin, max(1, vp.width() - 2 * margin), max(1, vp.height() - 2 * margin)
        )
        self._placeholder.show()
        self._placeholder.raise_()

    # ----------------------------------------------------------------- overlay
    def set_overlay_rect(self, rect, color: QColor, line_width: int = 2) -> None:
        """Draw a dashed rectangle in image (scene) coordinates."""
        if rect is None:
            self.clear_overlay()
            return
        rectf = QRectF(rect)
        if rectf.isNull():
            self.clear_overlay()
            return
        if self._overlay_item is None:
            self._overlay_item = QGraphicsRectItem()  # no brush by default (outline only)
            self._overlay_item.setZValue(10)
            self._scene.addItem(self._overlay_item)
        pen = QPen(color)
        pen.setCosmetic(True)  # constant on-screen width regardless of zoom
        pen.setWidth(max(1, int(line_width)))
        pen.setStyle(Qt.PenStyle.DashLine)
        self._overlay_item.setPen(pen)
        self._overlay_item.setRect(rectf)
        self._overlay_item.show()

    def clear_overlay(self) -> None:
        if self._overlay_item is not None:
            self._overlay_item.hide()

    # ------------------------------------------------------------------ image
    def set_pixmap(self, pixmap: QPixmap, preserve_view=None) -> None:
        """Set the displayed image.

        ``preserve_view`` keeps the on-screen framing across a resolution upgrade
        (e.g. thumbnail -> full) while zoomed. Defaults to True when zoomed.
        """
        if pixmap is None or pixmap.isNull():
            self._item.setPixmap(QPixmap())
            self._has_pixmap = False
            self._img_w = self._img_h = 0
            self.clear_overlay()
            self._update_placeholder()
            return

        new_w, new_h = pixmap.width(), pixmap.height()
        old_w, old_h = self._img_w, self._img_h
        had_pixmap = self._has_pixmap

        if preserve_view is None:
            preserve_view = not self._fit_mode

        capture = preserve_view and had_pixmap and old_w > 0 and old_h > 0
        s_old = fx = fy = None
        if capture:
            s_old = self.current_scale()
            center_scene = self.mapToScene(self.viewport().rect().center())
            fx = center_scene.x() / old_w
            fy = center_scene.y() / old_h

        self._item.setPixmap(pixmap)
        self._item.setOffset(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, new_w, new_h))
        self._img_w, self._img_h = new_w, new_h
        self._has_pixmap = True
        self._update_placeholder()

        if self._fit_mode or not capture:
            if self._fit_mode and not getattr(self, "_zoom_intent_100", False):
                self.fit_to_window()
            elif not capture:
                # Stale state: _fit_mode False but transform still at fit scale (e.g. zoom
                # requested before the first pixmap arrived). Refit so Space/double-click
                # toggles predictably instead of staying stuck at ~fit%.
                if self.is_at_fit_scale() and not getattr(self, "_zoom_intent_100", False):
                    self.fit_to_window()
            self._sync_fit_mode_flag()
            return

        # If user intended to zoom to 100% (actual pixels), do not scale down to preserve the thumbnail's screen size
        if getattr(self, "_zoom_intent_100", False) or (s_old is not None and s_old >= 1.0 - 1e-4):
            s_new = 1.0
        else:
            # Preserve on-screen image scale: s_new * new_w == s_old * old_w
            s_new = max(self.MIN_SCALE, min(self.MAX_SCALE, s_old * old_w / new_w))
        self.resetTransform()
        self.scale(s_new, s_new)
        self.centerOn(QPointF(fx * new_w, fy * new_h))
        self._sync_fit_mode_flag()
        self.zoomChanged.emit(self.current_scale())

    def clear(self) -> None:
        self.set_pixmap(QPixmap())

    def capture_viewport_pixmap(self) -> QPixmap | None:
        """Capture the visible view for resolution cross-fade (OpenGL-safe)."""
        if not self._has_pixmap:
            return None
        vp = self.viewport()
        w = max(1, vp.width())
        h = max(1, vp.height())
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            if isinstance(vp, QOpenGLWidget):
                fb = vp.grabFramebuffer()
                if fb is not None and not fb.isNull():
                    return fb
        except Exception:
            pass
        try:
            target = QRect(0, 0, w, h)
            pix = QPixmap(w, h)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            self.render(painter, target, target)
            painter.end()
            if not pix.isNull():
                return pix
        except Exception:
            pass
        try:
            snap = vp.grab()
            if snap is not None and not snap.isNull():
                return snap
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ zoom
    def fit_to_window(self) -> None:
        self._fit_mode = True
        self._zoom_intent_100 = False
        if not self._has_pixmap:
            return
        self.resetTransform()
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self.fitModeChanged.emit(True)
        self.zoomChanged.emit(self.current_scale())

    def zoom_to_actual(self) -> None:
        """100% — one image pixel per view pixel, centered on the image middle."""
        w, h = self._img_w, self._img_h
        cx = w * 0.5 if w > 0 else 0.0
        cy = h * 0.5 if h > 0 else 0.0
        self.zoom_to_actual_at(cx, cy)

    def zoom_to_actual_at(self, scene_x: float, scene_y: float) -> None:
        """100% zoom with the given image pixel centered in the viewport."""
        if not self._has_pixmap:
            return
        self._fit_mode = False
        self._zoom_intent_100 = True
        x = max(0.0, min(float(scene_x), max(0, self._img_w - 1)))
        y = max(0.0, min(float(scene_y), max(0, self._img_h - 1)))
        self.resetTransform()
        self.scale(1.0, 1.0)
        
        # Ensure scrollbars are updated based on the new scale
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
        
        self.centerOn(QPointF(x, y))
            
        self.fitModeChanged.emit(False)
        self.zoomChanged.emit(1.0)

    def set_pixmap_zoomed_at(
        self,
        pixmap: QPixmap,
        scene_x: float,
        scene_y: float,
        *,
        scale: float = 1.0,
    ) -> None:
        """Replace the image and apply zoom in one step (avoids a wrong-frame flash)."""
        if pixmap is None or pixmap.isNull():
            self.set_pixmap(QPixmap())
            return
        new_w, new_h = pixmap.width(), pixmap.height()
        self._item.setPixmap(pixmap)
        self._item.setOffset(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, new_w, new_h))
        self._img_w, self._img_h = new_w, new_h
        self._has_pixmap = True
        self._fit_mode = False
        intent_100 = float(scale) >= 1.0 - 1e-4
        self._zoom_intent_100 = intent_100
        self._update_placeholder()
        fit = self.fit_scale()
        s = float(scale)
        if s <= fit * self._FIT_SCALE_EPS:
            self.fit_to_window()
            return
        s = max(fit, min(self.MAX_SCALE, s))
        x = max(0.0, min(float(scene_x), max(0, new_w - 1)))
        y = max(0.0, min(float(scene_y), max(0, new_h - 1)))
        self.resetTransform()
        self.scale(s, s)
        
        # Ensure scrollbars are updated based on the new scale
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
        
        self.centerOn(QPointF(x, y))
            
        self.fitModeChanged.emit(False)
        self.zoomChanged.emit(self.current_scale())

    def center_on_image_point(self, scene_x: float, scene_y: float) -> None:
        """Pan so an image pixel is centered without changing zoom scale."""
        if not self._has_pixmap:
            return
        x = max(0.0, min(float(scene_x), max(0, self._img_w - 1)))
        y = max(0.0, min(float(scene_y), max(0, self._img_h - 1)))
        self.centerOn(QPointF(x, y))

    def toggle_fit(self) -> None:
        if not self._has_pixmap:
            return
        self._sync_fit_mode_flag()
        if self.wants_zoom_in_toggle():
            self.zoom_to_actual()
        else:
            self.fit_to_window()

    def set_scale(self, scale: float) -> None:
        fit = self.fit_scale()
        scale = float(scale)
        if scale <= fit * self._FIT_SCALE_EPS:
            self.fit_to_window()
            return
        scale = max(fit, min(self.MAX_SCALE, scale))
        self.resetTransform()
        self.scale(scale, scale)
        self._fit_mode = False
        self.fitModeChanged.emit(False)
        self.zoomChanged.emit(self.current_scale())

    def zoom_by(self, factor: float) -> None:
        if not self._has_pixmap or factor <= 0:
            return
        fit = self.fit_scale()
        cur = self.current_scale()
        new = cur * factor
        if factor < 1.0 and new <= fit * self._FIT_SCALE_EPS:
            self.fit_to_window()
            return
        new = max(fit, min(self.MAX_SCALE, new))
        if abs(new - cur) < 1e-9:
            return
        self.scale(new / cur, new / cur)
        self._fit_mode = False
        self.fitModeChanged.emit(False)
        self.zoomChanged.emit(self.current_scale())

    # ------------------------------------------------------------------ events
    def mouseMoveEvent(self, event) -> None:
        host = self.parentWidget()
        if host is not None and hasattr(host, "handle_pointer_for_filmstrip"):
            host.handle_pointer_for_filmstrip(event.globalPosition())
        super().mouseMoveEvent(event)

    def event(self, event) -> bool:
        """Trackpad pinch (macOS) and smart-zoom gestures."""
        if event.type() == QEvent.Type.NativeGesture and self._has_pixmap:
            gtype = event.gestureType()
            if gtype == Qt.NativeGestureType.ZoomNativeGesture:
                self.zoom_by(1.0 + event.value() * 0.5)
                event.accept()
                return True
            if gtype == Qt.NativeGestureType.SmartZoomNativeGesture:
                self.toggle_fit()
                event.accept()
                return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        handler = getattr(self, "_shortcut_handler", None)
        if callable(handler) and handler(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl:
            # Control + Scroll: Zoom smoothly
            self.zoom_by(1.25 if delta > 0 else 0.8)
            event.accept()
            return

        # Plain wheel (no Control Modifier)
        if self._fit_mode:
            # Fit-to-window: navigate images (Qt: delta > 0 = up, delta < 0 = down)
            self.wheelNavigate.emit(-1 if delta > 0 else 1)
            event.accept()
            return

        # Zoomed in: pan vertically by delegating to QGraphicsView scroll behavior
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._has_pixmap:
            scene_pt = self.mapToScene(event.position().toPoint())
            self.doubleClickedAt.emit(
                QPointF(
                    max(0.0, min(scene_pt.x(), max(0, self._img_w - 1))),
                    max(0.0, min(scene_pt.y(), max(0, self._img_h - 1))),
                )
            )
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode and self._has_pixmap:
            self.fit_to_window()
        self._update_placeholder()
