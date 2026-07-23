"""GPU-accelerated single-image view (Route B).

A self-contained ``QGraphicsView`` that renders one image and performs fit / zoom /
pan as GPU matrix transforms instead of re-scaling the pixmap on the CPU every frame.
This gives QuickLook-style smoothness for wheel/pinch zoom and drag pan.

The optional ``QOpenGLWidget`` viewport is vendor-agnostic (NVIDIA / AMD / Intel on
Windows; Qt typically uses a Metal-backed GL stack on macOS).

Enabled by default in release builds; set ``RAWVIEWER_GPU_VIEW=0`` for the legacy
``QScrollArea`` + ``QLabel`` path. The widget is intentionally
decoupled from the main window: it exposes a small API plus a few signals so the
host can keep navigation / status / histogram behaviour identical.

Environment toggles:
- ``RAWVIEWER_GPU_VIEW=0``       Disable; use legacy scroll-area single-image view.
- ``RAWVIEWER_GPU_VIEW_NO_GL=1`` Use the raster viewport (debug / fallback).
- ``RAWVIEWER_GPU_GL_TEX=1``     Paint RGB via ``QImage`` on the OpenGL viewport
  (skip ``QPixmap``). Opt-in Phase 2b step toward CUDA↔GL.
- ``RAWVIEWER_GPU_CUDA_GL=1``    Keep decode on CUDA and upload via
  ``cudaGraphicsGLRegisterImage`` (Phase 2c; falls back to GL_TEX / pixmap).
"""

import os
import sys
from typing import Any

from PyQt6.QtCore import Qt, QRect, QRectF, QPoint, QPointF, pyqtSignal, QEvent, QMimeData, QUrl
from PyQt6.QtGui import QKeyEvent, QPixmap, QPainter, QColor, QPen, QDrag, QBrush, QImage, QTransform
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
    QGraphicsObject,
    QLabel,
    QApplication,
)

from rawviewer_ui.composition_grid import CompositionGridGraphicsItem
from rawviewer_ui.widgets import stamp_rawviewer_export_drag


class CompareHandleItem(QGraphicsEllipseItem):
    """A custom comparison view split divider handle.
    It renders a prominent white circle with a dark border, and draws two vertical
    grip lines in the center to clearly convey that it can be dragged.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptHoverEvents(True)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        rect = self.rect()
        cx = rect.center().x()
        cy = rect.center().y()
        h = rect.height()
        
        # Pen for white vertical grip lines
        pen = QPen(QColor(255, 255, 255, 200), 1.2)
        painter.setPen(pen)
        
        offset = 2.0
        line_h = h * 0.4
        painter.drawLine(QPointF(cx - offset, cy - line_h / 2.0), QPointF(cx - offset, cy + line_h / 2.0))
        painter.drawLine(QPointF(cx + offset, cy - line_h / 2.0), QPointF(cx + offset, cy + line_h / 2.0))


def _env_true(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def gpu_gl_tex_enabled() -> bool:
    """Opt-in QImage→OpenGL paint path (skips QPixmap); see set_rgb_numpy."""
    return _env_true("RAWVIEWER_GPU_GL_TEX")


def gpu_cuda_gl_enabled() -> bool:
    """Opt-in CUDA↔GL zero-copy display; see set_device_rgb."""
    return _env_true("RAWVIEWER_GPU_CUDA_GL")


class RgbGlImageItem(QGraphicsObject):
    """Scene item that paints RGB via OpenGL (QImage or CUDA-registered texture).

    * Host path: ``QPainter.drawImage`` (Phase 2b).
    * Device path: GUI-thread GL texture + ``cudaGraphicsGLRegisterImage``
      upload, painted with ``beginNativePainting`` (Phase 2c).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._w = 0
        self._h = 0
        self._arr = None
        self._qimage = None  # Optional[QImage]
        self._device_rgb = None
        self._cuda_slot = None  # Optional[CudaGlTextureSlot]
        self._rgba_device = None
        self._cuda_upload_failed = False
        self._cuda_uploaded_key = None  # (id(device_rgb), generation) after upload
        self._gl_fns = None

    def set_rgb_uint8(self, rgb) -> None:
        import numpy as np

        self._release_cuda_gl(None)
        arr = np.ascontiguousarray(rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"expected HxWx3+ RGB, got shape={getattr(arr, 'shape', None)}")
        if arr.dtype == np.uint16:
            arr = (arr / 257.0).astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.floating):
            from raw_tone_recovery import _encode_srgb8
            arr = _encode_srgb8(np.clip(arr.astype(np.float32), 0.0, None))
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[2] > 3:
            arr = np.ascontiguousarray(arr[:, :, :3])
        h, w = int(arr.shape[0]), int(arr.shape[1])
        self.prepareGeometryChange()
        self._arr = arr
        self._qimage = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        # Keep numpy alive for as long as QImage borrows its buffer.
        self._qimage._ndarray = arr  # type: ignore[attr-defined]
        self._w, self._h = w, h
        self._device_rgb = None
        self._cuda_uploaded_key = None
        self.update()

    def set_device_rgb(self, device_rgb) -> None:
        """Display a CUDA ``DeviceRgb`` via CUDA↔GL (lazily registered on paint)."""
        self.prepareGeometryChange()
        self._arr = None
        self._qimage = None
        self._device_rgb = device_rgb
        self._cuda_upload_failed = False
        self._cuda_uploaded_key = None
        self._w = int(device_rgb.width)
        self._h = int(device_rgb.height)
        # Drop previous interop slot; recreated on next paint with GL current.
        self._release_cuda_gl(None)
        self.update()

    def clear_rgb(self) -> None:
        self.prepareGeometryChange()
        self._release_cuda_gl(None)
        self._arr = None
        self._qimage = None
        self._device_rgb = None
        self._cuda_uploaded_key = None
        self._w = self._h = 0
        self.update()

    def image_size(self) -> tuple[int, int]:
        return self._w, self._h

    def to_qimage(self) -> QImage:
        if self._qimage is not None:
            return self._qimage
        if self._device_rgb is not None:
            try:
                arr = self._device_rgb.to_numpy()
                h, w = arr.shape[:2]
                q = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                q._ndarray = arr  # type: ignore[attr-defined]
                return q
            except Exception:
                return QImage()
        return QImage()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, float(self._w), float(self._h))

    def _release_cuda_gl(self, widget) -> None:
        slot = self._cuda_slot
        self._cuda_slot = None
        self._rgba_device = None
        self._cuda_uploaded_key = None
        if slot is None:
            return
        try:
            import ctypes

            def _del_tex(tex_id: int) -> None:
                fns = self._gl_fns
                if fns is None:
                    return
                fns.glDeleteTextures(1, [tex_id])

            if widget is not None:
                try:
                    widget.makeCurrent()
                except Exception:
                    pass
            slot.release(gl_delete_fn=_del_tex)
            if widget is not None:
                try:
                    widget.doneCurrent()
                except Exception:
                    pass
        except Exception:
            try:
                slot.release()
            except Exception:
                pass

    def _device_upload_key(self):
        d = self._device_rgb
        if d is None:
            return None
        return (id(d), int(getattr(d, "generation", 0)), self._w, self._h)

    def _ensure_cuda_gl(self, widget, *, context_already_current: bool = False) -> bool:
        if self._cuda_upload_failed or self._device_rgb is None:
            return False
        if not self._device_rgb.is_cuda():
            self._cuda_upload_failed = True
            return False
        try:
            from gpu_gl_bridge import (
                CudaGlTextureSlot,
                ensure_rgba_u8,
                gl_tex_image_2d_rgba_empty,
                register_gl_texture,
                upload_device_rgba_to_gl,
            )
            from PyQt6.QtOpenGL import QOpenGLFunctions_2_0
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
        except Exception:
            self._cuda_upload_failed = True
            return False

        if not isinstance(widget, QOpenGLWidget):
            return False

        try:
            if not context_already_current:
                widget.makeCurrent()
            if self._gl_fns is None:
                fns = QOpenGLFunctions_2_0()
                if not fns.initializeOpenGLFunctions():
                    self._cuda_upload_failed = True
                    return False
                self._gl_fns = fns
            fns = self._gl_fns

            need_new = (
                self._cuda_slot is None
                or self._cuda_slot.width != self._w
                or self._cuda_slot.height != self._h
                or not self._cuda_slot.tex_id
            )
            if need_new:
                if self._cuda_slot is not None:
                    # Keep GL context current when painting; only makeCurrent when
                    # this helper is called outside beginNativePainting.
                    self._release_cuda_gl(None if context_already_current else widget)
                    if not context_already_current:
                        widget.makeCurrent()

                gen = fns.glGenTextures(1)
                tex_id = int(gen[0] if isinstance(gen, (tuple, list)) else gen)
                GL_TEXTURE_2D = 0x0DE1
                GL_LINEAR = 0x2601
                GL_CLAMP_TO_EDGE = 0x812F
                GL_TEXTURE_MIN_FILTER = 0x2801
                GL_TEXTURE_MAG_FILTER = 0x2800
                GL_TEXTURE_WRAP_S = 0x2802
                GL_TEXTURE_WRAP_T = 0x2803
                fns.glBindTexture(GL_TEXTURE_2D, tex_id)
                fns.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                fns.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                fns.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                fns.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                gl_tex_image_2d_rgba_empty(self._w, self._h)
                resource = register_gl_texture(tex_id, target=GL_TEXTURE_2D)
                self._cuda_slot = CudaGlTextureSlot(
                    tex_id=tex_id,
                    resource=resource,
                    width=self._w,
                    height=self._h,
                    rgba=True,
                )
                self._cuda_uploaded_key = None

            upload_key = self._device_upload_key()
            if upload_key != self._cuda_uploaded_key or need_new:
                rgba = ensure_rgba_u8(self._device_rgb)
                self._rgba_device = rgba  # keep alive across map
                upload_device_rgba_to_gl(self._cuda_slot, rgba)
                self._cuda_uploaded_key = upload_key
            return True
        except Exception as e:
            import logging

            logging.getLogger(__name__).info(
                "[CUDA_GL] register/upload failed; falling back: %s", e
            )
            self._cuda_upload_failed = True
            try:
                self._release_cuda_gl(None if context_already_current else widget)
            except Exception:
                pass
            return False

    def _apply_painter_gl_matrices(self, painter: QPainter, fns) -> None:
        """Map item pixel space through QPainter's transform (Compatibility profile)."""
        from PyQt6.QtGui import QTransform

        GL_PROJECTION = 0x1701
        GL_MODELVIEW = 0x1700
        device = painter.device()
        dpr = 1.0
        try:
            dpr = float(device.devicePixelRatioF())
        except Exception:
            try:
                dpr = float(device.devicePixelRatio())
            except Exception:
                dpr = 1.0
        dw = max(1.0, float(device.width()) * dpr)
        dh = max(1.0, float(device.height()) * dpr)
        fns.glMatrixMode(GL_PROJECTION)
        fns.glLoadIdentity()
        # Top-left origin to match QPainter device coords.
        fns.glOrtho(0.0, dw, dh, 0.0, -1.0, 1.0)
        fns.glMatrixMode(GL_MODELVIEW)
        fns.glLoadIdentity()
        t = painter.transform()
        if dpr != 1.0:
            # Logical (painter) -> device pixels: apply the painter transform
            # FIRST, then the DPR scale. Qt's row-vector convention means
            # `A * B` applies A first, so the DPR factor must be on the RIGHT.
            #
            # The reverse (fromScale(dpr, dpr) * t) scales the item's geometry
            # by dpr but leaves the painter's TRANSLATION in logical pixels --
            # the quad then lands at t.dx() instead of t.dx() * dpr, i.e. too
            # far left/up by (1 - 1/dpr) of the offset, while still being the
            # correct SIZE. That was the "image sits off-center with a black
            # gap on one side" bug: invisible at 100% scaling, ~115px of left
            # shift at 125% on a 1400px view, and invisible to Qt's own
            # hit-testing (mapFromScene still reported the image centered,
            # because only the GL draw was wrong -- not the scene geometry).
            t = t * QTransform.fromScale(dpr, dpr)
        # Column-major 4×4 for column vector GL transforms.
        mat = [
            t.m11(),
            t.m12(),
            0.0,
            0.0,
            t.m21(),
            t.m22(),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            t.dx(),
            t.dy(),
            0.0,
            1.0,
        ]
        fns.glLoadMatrixf(mat)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        # Device path: CUDA→GL texture then native textured quad.
        if self._device_rgb is not None and not self._cuda_upload_failed:
            vp = widget
            if vp is None:
                # QGraphicsView usually passes the viewport as widget.
                try:
                    sc = self.scene()
                    if sc is not None:
                        for v in sc.views():
                            vp = v.viewport()
                            break
                except Exception:
                    vp = None
            if vp is not None:
                try:
                    painter.beginNativePainting()
                    # Context is current after beginNativePainting — avoid makeCurrent.
                    if not self._ensure_cuda_gl(vp, context_already_current=True):
                        painter.endNativePainting()
                        raise RuntimeError("cuda-gl ensure failed")
                    fns = self._gl_fns
                    GL_TEXTURE_2D = 0x0DE1
                    GL_BLEND = 0x0BE2
                    GL_DEPTH_TEST = 0x0B71
                    GL_TEXTURE_ENV = 0x2300
                    GL_TEXTURE_ENV_MODE = 0x2200
                    GL_REPLACE = 0x1E01
                    fns.glDisable(GL_BLEND)
                    fns.glDisable(GL_DEPTH_TEST)
                    self._apply_painter_gl_matrices(painter, fns)
                    fns.glEnable(GL_TEXTURE_2D)
                    try:
                        fns.glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
                    except Exception:
                        pass
                    try:
                        fns.glColor4f(1.0, 1.0, 1.0, 1.0)
                    except Exception:
                        pass
                    fns.glBindTexture(GL_TEXTURE_2D, self._cuda_slot.tex_id)
                    w, h = float(self._w), float(self._h)
                    # DeviceRgb row0 = image top. cudaMemcpy2DToArray writes that to
                    # GL texture row y=0. With Y-down glOrtho (top-left item origin),
                    # sample V=0 at the top edge — do not invert V.
                    fns.glBegin(0x0007)  # GL_QUADS
                    fns.glTexCoord2f(0.0, 0.0)
                    fns.glVertex2f(0.0, 0.0)
                    fns.glTexCoord2f(1.0, 0.0)
                    fns.glVertex2f(w, 0.0)
                    fns.glTexCoord2f(1.0, 1.0)
                    fns.glVertex2f(w, h)
                    fns.glTexCoord2f(0.0, 1.0)
                    fns.glVertex2f(0.0, h)
                    fns.glEnd()
                    fns.glBindTexture(GL_TEXTURE_2D, 0)
                    fns.glDisable(GL_TEXTURE_2D)
                    painter.endNativePainting()
                    return
                except Exception as e:
                    try:
                        painter.endNativePainting()
                    except Exception:
                        pass
                    import logging

                    logging.getLogger(__name__).info(
                        "[CUDA_GL] native paint failed; falling back: %s", e
                    )
                    self._cuda_upload_failed = True

            # Fallback: host QImage path once CUDA-GL fails.
            if self._qimage is None:
                try:
                    arr = self._device_rgb.to_numpy()
                    self.set_rgb_uint8(arr)
                except Exception:
                    return

        if self._qimage is None or self._qimage.isNull():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QPointF(0.0, 0.0), self._qimage)


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
    # Image pixel coordinates (scene space) where the user clicked while color-pick
    # mode was armed (see set_color_pick_mode). One-shot: mode disarms after firing.
    colorPickRequested = pyqtSignal(QPointF)
    # Dodge & burn brush: (scene_x, scene_y, pressure 0..1, is_stroke_end).
    # Emitted continuously while painting (set_dodge_burn_mode(True)) so the
    # host can stamp+repaint per point; is_stroke_end=True on release, the
    # host's cue to edge-snap the touched region and trigger the exact
    # (worker-thread) re-render.
    dodgeBurnStroke = pyqtSignal(QPointF, float, bool)
    # Mouse left the photo (or the view) while a brush tool was armed — host
    # should disarm Dodge/Burn/Eraser/Heal so the tool does not stay sticky.
    brushToolLeftImage = pyqtSignal()
    # Vertical wheel/trackpad delta while D&B is armed — host nudges Brush Size
    # instead of navigating images.
    dodgeBurnBrushSizeWheel = pyqtSignal(int)
    # Fingers lifted after a mid-stroke two-finger resize: host resets its
    # per-stroke stamp anchor so the resumed stamp is not flow-scaled against
    # the pre-pause position.
    dodgeBurnResumeAfterResize = pyqtSignal()
    # Crop overlay: insets changed / drag finished (CropLeft/Right/Top/Bottom).
    cropInsetsChanged = pyqtSignal(float, float, float, float)
    cropEditingFinished = pyqtSignal()

    # Absolute floor for set_scale; wheel zoom cannot go below fit_scale() (fit-to-window).
    MIN_SCALE = 0.01
    # Pinch / wheel cap (400%). Space and double-click use zoom_to_actual() at 100%.
    MAX_SCALE = 4.0
    _FIT_SCALE_EPS = 1.002  # treat within ~0.2% of fit as fit-to-window
    _shortcut_handler: Any

    def __init__(self, parent=None, background="#14120F"):
        # Attributes read from event()/viewportEvent() must exist before super().__init__()
        # because Qt may deliver events during construction.
        self._fit_mode = True
        self._wheel_navigate_in_fit = True
        self._zoom_intent_100 = False
        self._zoom_locked = False
        self._has_pixmap = False
        self._img_w = 0
        self._img_h = 0
        self._overlay_item = None
        self._clipping_item = None
        self._grid_mode = "off"
        self._shortcut_handler = None
        self._edr_initialized = False
        self.file_path = None
        self._dodge_burn_mode = False
        # True once the pointer has landed on the image at least once since
        # the tool was armed. Guards _maybe_emit_brush_tool_left_image so
        # arming (D/B/X, or clicking the tool button) while the cursor is
        # still over the Adjust panel / letterbox margin can't immediately
        # disarm itself on the very next mouse-move -- only a genuine
        # "was on the image, now isn't" transition should disarm.
        self._dodge_burn_confirmed_on_image = False
        # Hold-to-paint: True while a brush hotkey is physically held (paint
        # gate, decoupled from the persistent _dodge_burn_mode context).
        self._db_key_held = False
        # True between a two-finger-scroll ScrollBegin and ScrollEnd while a
        # hold-to-paint stroke is live: stamping is suspended so the user can
        # resize the brush mid-stroke without painting, and the stroke stays
        # open (one undo unit) until the fingers lift and painting resumes.
        self._brush_resizing = False
        self._crop_mode = False
        self._export_drag_enabled = True
        self._drag_start_pos = None
        self._drag_started = False
        self._color_pick_mode = False
        self._compare_active = False
        self._compare_dragging_divider = False

        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)

        # Sibling of the image (not a child of _item) so Show Mask stays
        # visible on both the QPixmap and RGB-GL display paths.
        self._mask_item = QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.45)
        self._mask_item.setZValue(5)
        self._mask_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._mask_item.hide()
        self._mask_overlay_wanted = False
        self._mask_overlay_mask = None
        self._scene.addItem(self._mask_item)

        # Dodge/burn brush preview: soft radial falloff (matches stamp kernel).
        self._brush_cursor_item = QGraphicsPixmapItem()
        self._brush_cursor_item.setZValue(25)
        self._brush_cursor_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self._brush_cursor_item.hide()
        self._dodge_burn_brush_radius = 40.0
        self._dodge_burn_brush_flow = 0.55
        self._brush_cursor_pixmap_r = -1.0
        self._brush_cursor_pixmap_flow = -1.0
        self._scene.addItem(self._brush_cursor_item)

        # Opt-in RGB→OpenGL paint path (RAWVIEWER_GPU_GL_TEX=1): skips QPixmap.
        self._rgb_item = RgbGlImageItem()
        self._rgb_item.setZValue(0)
        self._rgb_item.hide()
        self._scene.addItem(self._rgb_item)
        self._use_rgb_gl = False

        # Dashed focus / subject overlay rectangle (scene coords == image pixels). Uses a
        # cosmetic pen so the on-screen line width stays constant at any zoom level.
        self._grid_item = CompositionGridGraphicsItem()
        self._scene.addItem(self._grid_item)

        self._clipping_item = QGraphicsPixmapItem()
        self._clipping_item.setZValue(15)
        self._clipping_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self._clipping_item.hide()
        self._scene.addItem(self._clipping_item)

        # Compare-with-original split view (Adjust panel only). The original
        # (unedited) render sits in an overlay item above the always-current
        # edited pixmap in self._item, cropped to only the region left of a
        # draggable divider -- so the main item/zoom/pan/fit machinery needs
        # no changes at all; this is purely an additional overlay.
        self._compare_split_frac = 0.5
        self._compare_original_pixmap = QPixmap()
        self._compare_overlay_item = QGraphicsPixmapItem()
        self._compare_overlay_item.setZValue(12)
        self._compare_overlay_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._compare_overlay_item.hide()
        self._scene.addItem(self._compare_overlay_item)

        from rawviewer_ui.crop_overlay import CropOverlayItem

        self._crop_item = CropOverlayItem()
        self._crop_item.hide()
        self._crop_item.insetsChanged.connect(self.cropInsetsChanged.emit)
        self._crop_item.editingFinished.connect(self.cropEditingFinished.emit)
        self._scene.addItem(self._crop_item)

        self._compare_divider_line = QGraphicsLineItem()
        self._compare_divider_line.setZValue(13)
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self._compare_divider_line.setPen(pen)
        self._compare_divider_line.hide()
        self._scene.addItem(self._compare_divider_line)

        self._compare_divider_handle = CompareHandleItem()
        self._compare_divider_handle.setZValue(14)
        self._compare_divider_handle.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        self._compare_divider_handle.setBrush(QColor(0, 0, 0, 140))
        self._compare_divider_handle.hide()
        self._scene.addItem(self._compare_divider_handle)

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
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._maybe_enable_opengl()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, self._enable_macos_edr)

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

    def set_file_path(self, file_path: str) -> None:
        self.file_path = file_path

    # ------------------------------------------------------------------ setup
    def _maybe_enable_opengl(self) -> None:
        if _env_true("RAWVIEWER_GPU_VIEW_NO_GL"):
            return
        try:
            from PyQt6.QtGui import QSurfaceFormat
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            gl = QOpenGLWidget()
            # CUDA↔GL Phase 2c paints with fixed-function textured quads
            # (glBegin / GL_TEXTURE_2D). Qt6's default Core Profile strips
            # those; without Compatibility the item fills with a solid colour.
            if gpu_cuda_gl_enabled():
                fmt = QSurfaceFormat()
                fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
                fmt.setVersion(3, 3)
                fmt.setProfile(
                    QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile
                )
                fmt.setDepthBufferSize(24)
                fmt.setStencilBufferSize(8)
                fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
                gl.setFormat(fmt)
            gl.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setViewport(gl)
            # Re-enable tracking on the new viewport (HoverMove works before first click).
            self.viewport().setMouseTracking(True)
            self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            self.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        except Exception:
            # Raster fallback keeps the feature working without a GL context.
            pass


    def _enable_macos_edr(self) -> None:
        """Enable macOS EDR (Extended Dynamic Range) support on the viewport's CALayer."""
        if getattr(self, "_edr_initialized", False):
            return
        import platform
        if platform.system() == "Darwin":
            from common_image_loader import is_macos_edr_enabled
            if not is_macos_edr_enabled():
                return
            try:
                import objc
                vp = self.viewport()
                if vp is None:
                    return
                ptr = int(vp.winId())
                if not ptr:
                    return
                view = objc.objc_object(c_void_p=ptr)
                view.setWantsLayer_(True)
                layer = view.layer()
                if layer is not None:
                    layer.setWantsExtendedDynamicRangeContent_(True)
                    self._edr_initialized = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Failed to enable EDR on CALayer: {e}")



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

    def _zoom_in_blocked(self) -> bool:
        if not self._has_pixmap or self._item.pixmap() is None:
            return False
        return self.current_scale() >= self.MAX_SCALE - 1e-9

    def set_dodge_burn_mask_overlay(self, mask, show: bool) -> None:
        self._mask_overlay_wanted = bool(show)
        self._mask_overlay_mask = mask if show else None
        if not show or mask is None or getattr(mask, "is_empty", True):
            self._mask_item.hide()
            self._mask_item.setPixmap(QPixmap())
            return
        self.update_dodge_burn_mask(mask)
        self._mask_item.show()

    @staticmethod
    def _dodge_burn_overlay_rgba(data, is_heal: bool):
        """uint8 RGBA overlay for a (sub-)region of mask.data (float32)."""
        import numpy as np

        h, w = data.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        if is_heal:
            cov = np.clip(data, 0.0, 1.0)
            on = cov > 0.02
            overlay[on, 1] = 220  # G
            overlay[on, 3] = np.clip(cov[on] * 200.0 + 40.0, 0, 255).astype(np.uint8)
        else:
            # Red for Dodge (positive), Blue for Burn (negative)
            pos_mask = data > 0
            overlay[pos_mask, 0] = 255  # R
            overlay[pos_mask, 3] = np.clip(
                data[pos_mask] / 1.5 * 180.0 + 40.0, 0, 255
            ).astype(np.uint8)

            neg_mask = data < 0
            overlay[neg_mask, 2] = 255  # B
            overlay[neg_mask, 3] = np.clip(
                -data[neg_mask] / 1.5 * 180.0 + 40.0, 0, 255
            ).astype(np.uint8)
        return overlay

    def update_dodge_burn_mask(self, mask, dirty_bbox=None) -> None:
        """Refresh the red/blue mask overlay when Show Mask is on.

        Callers may invoke this during painting even if the item was just
        shown; skip only when the user has Show Mask off.

        ``dirty_bbox`` (x0, y0, x1, y1, mask-pixel coords) lets a live brush
        stroke repaint only the touched region instead of rebuilding +
        re-uploading the full mask-resolution RGBA buffer on every throttled
        tick -- that full rebuild (even after dropping the old cv2.resize-to-
        display-size step) was still O(full mask image) per tick, not O(brush
        area) like the actual stamping code, and was the remaining cause of
        brush lag with Show Mask on for large edit-base resolutions.
        """
        if not getattr(self, "_mask_overlay_wanted", False):
            return
        self._mask_overlay_mask = mask
        if mask is None or getattr(mask, "is_empty", True):
            self._mask_item.hide()
            self._mask_item.setPixmap(QPixmap())
            self._mask_overlay_shape = None
            return

        h, w = mask.data.shape

        try:
            from raw_spot_heal import HealMask

            is_heal = isinstance(mask, HealMask)
        except Exception:
            is_heal = False

        cur_pixmap = self._mask_item.pixmap()
        can_incremental = (
            dirty_bbox is not None
            and getattr(self, "_mask_overlay_shape", None) == (h, w)
            and getattr(self, "_mask_overlay_is_heal", None) == is_heal
            and cur_pixmap is not None
            and not cur_pixmap.isNull()
        )

        if can_incremental:
            x0, y0, x1, y1 = dirty_bbox
            pad = 2
            x0 = max(0, int(x0) - pad)
            y0 = max(0, int(y0) - pad)
            x1 = min(w, int(x1) + pad)
            y1 = min(h, int(y1) + pad)
            if x1 <= x0 or y1 <= y0:
                return
            patch = self._dodge_burn_overlay_rgba(mask.data[y0:y1, x0:x1], is_heal)
            import numpy as np

            patch = np.ascontiguousarray(patch)
            qimg = QImage(
                patch.data,
                patch.shape[1],
                patch.shape[0],
                patch.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()
            pm = cur_pixmap
            painter = QPainter(pm)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.drawImage(x0, y0, qimg)
            painter.end()
            self._mask_item.setPixmap(pm)
        else:
            overlay = self._dodge_burn_overlay_rgba(mask.data, is_heal)
            import numpy as np

            # Keep the overlay at the mask's own (usually much lower)
            # resolution and let the graphics item scale it up via
            # QTransform instead of cv2.resize-ing a full-display-size
            # buffer -- that resize was the original dominant cost here.
            overlay = np.ascontiguousarray(overlay)
            qimg = QImage(
                overlay.data,
                overlay.shape[1],
                overlay.shape[0],
                overlay.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()
            self._mask_item.setPixmap(QPixmap.fromImage(qimg))
            self._mask_overlay_shape = (h, w)
            self._mask_overlay_is_heal = is_heal

        self._mask_item.setOffset(0, 0)
        if self._img_w > 0 and self._img_h > 0 and (w != self._img_w or h != self._img_h):
            self._mask_item.setTransform(
                QTransform().scale(self._img_w / w, self._img_h / h)
            )
        else:
            self._mask_item.setTransform(QTransform())
        if not self._mask_item.isVisible():
            self._mask_item.show()

    def _refresh_dodge_burn_mask_overlay_size(self) -> None:
        """Re-fit the mask overlay after the displayed pixmap size changes."""
        if not getattr(self, "_mask_overlay_wanted", False):
            return
        mask = getattr(self, "_mask_overlay_mask", None)
        if mask is not None:
            self.update_dodge_burn_mask(mask)

    def _mouse_stroke_pressure(self, event) -> float:
        """Pressure for mouse/trackpad dodge-burn stamps (0.05..1).

        Stylus pressure comes from ``tabletEvent``. Mouse/trackpad usually
        report a flat 1.0 via Qt; on macOS Force Touch trackpads we also
        read ``NSEvent.pressure()`` when the current event carries real
        force data. If neither varies, fall back to full strength.
        """
        try:
            p = float(event.pressure())
            if 0.02 < p < 0.999:
                return max(0.05, min(1.0, p))
        except Exception:
            pass
        if sys.platform == "darwin":
            try:
                from AppKit import NSApp

                nev = NSApp.currentEvent()
                if nev is not None:
                    p = float(nev.pressure())
                    # Non-force events often report 0.0; treat as "no data".
                    if p > 0.02:
                        return max(0.05, min(1.0, p))
            except Exception:
                pass
        return 1.0

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

    def set_clipping_overlay(self, pixmap: QPixmap | None) -> None:
        item = self._clipping_item
        if item is None:
            return
        if pixmap is None or pixmap.isNull():
            item.hide()
            return
        item.setPixmap(pixmap)
        item.setOffset(0, 0)
        item.show()

    def clear_clipping_overlay(self) -> None:
        if self._clipping_item is not None:
            self._clipping_item.hide()

    def pixmap(self) -> QPixmap:
        if getattr(self, "_use_rgb_gl", False):
            img = self._rgb_item.to_qimage()
            if img is not None and not img.isNull():
                return QPixmap.fromImage(img)
            return QPixmap()
        return self._item.pixmap() if getattr(self, "_has_pixmap", False) else QPixmap()

    def set_composition_grid_mode(self, mode: str) -> None:
        """Show composition guide lines in image (scene) coordinates."""
        self._grid_mode = mode if mode else "off"
        self._grid_item.set_grid(self._img_w, self._img_h, self._grid_mode)

    # ------------------------------------------------------------------ image
    def clear_pixmap_keep_placeholder_hidden(self) -> None:
        """Clear the displayed pixmap image but keep placeholder text hidden (e.g. during reload)."""
        self._item.setPixmap(QPixmap())

    def set_pixmap(self, pixmap: QPixmap, preserve_view=None, *, exact_framing: bool = False) -> None:
        """Set the displayed image.

        ``preserve_view`` keeps the on-screen framing across a resolution upgrade
        (e.g. thumbnail -> full) while zoomed. Defaults to True when zoomed.

        ``exact_framing`` (implies preserving) additionally disables the
        snap-to-100% heuristics: used for same-content re-renders at varying
        resolutions (Adjust-panel live preview), where "100% of the current
        pixmap" is meaningless -- the tiers are stand-ins for the same image
        and only the on-screen magnification should ever be preserved.
        """
        if pixmap is None or pixmap.isNull():
            self._use_rgb_gl = False
            self._rgb_item.hide()
            self._rgb_item.clear_rgb()
            self._item.show()
            self._item.setPixmap(QPixmap())
            self._has_pixmap = False
            self._img_w = self._img_h = 0
            self.clear_overlay()
            self.clear_clipping_overlay()
            self.set_compare_mode(False)
            self._grid_item.set_grid(0, 0, self._grid_mode)
            self._update_placeholder()
            return

        # Pixmap path owns the scene image; hide the RGB-GL item.
        self._use_rgb_gl = False
        self._rgb_item.hide()
        self._rgb_item.clear_rgb()
        self._item.show()

        new_w, new_h = pixmap.width(), pixmap.height()
        old_w, old_h = self._img_w, self._img_h
        had_pixmap = self._has_pixmap

        if preserve_view is None:
            preserve_view = not self._fit_mode

        capture = preserve_view and had_pixmap and old_w > 0 and old_h > 0
        s_old = fx = fy = None
        was_fit_scale = False
        if capture:
            s_old = self.current_scale()
            # "Clearly above fit", not merely >= 100%: fit scale itself
            # exceeds 1.0 for content smaller than the viewport, and the
            # viewport can CHANGE between paints (gallery->single shows the
            # container mid-transition), which made a plain is_at_fit_scale()
            # comparison unreliable -- a fitted tile read as "user at 100%"
            # and snapped, poisoning the zoom-intent machinery.
            was_fit_scale = not (s_old > self.fit_scale() * 1.1)
            center_scene = self.mapToScene(self.viewport().rect().center())
            fx = center_scene.x() / old_w
            fy = center_scene.y() / old_h

        self._item.setPixmap(pixmap)
        self._item.setOffset(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, new_w, new_h))
        self._img_w, self._img_h = new_w, new_h
        self._has_pixmap = True
        self._grid_item.set_grid(new_w, new_h, self._grid_mode)
        if getattr(self, "_crop_mode", False) and getattr(self, "_crop_item", None) is not None:
            self._crop_item.set_image_size(new_w, new_h)
        if (new_w, new_h) != (old_w, old_h):
            self._refresh_dodge_burn_mask_overlay_size()
        self._update_placeholder()
        if self._compare_active and (new_w, new_h) != (old_w, old_h):
            # The compare-with-original crop/divider are computed from _img_w/_img_h
            # (see _update_compare_overlay); resync them whenever the displayed edited
            # pixmap's size changes while comparing, so a resolution swap (e.g. the
            # native-resolution preview being replaced by the Adjust panel's own,
            # differently-sized render) can never leave the overlay cropped against a
            # stale size until the user happens to toggle Compare off and back on.
            self._update_compare_overlay()

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

        # If user intended to zoom to 100% (actual pixels), do not scale down to
        # preserve the thumbnail's screen size. Only valid for a resolution
        # UPGRADE (new pixmap >= old): snapping a *smaller* same-content render
        # to scale 1.0 shrinks the on-screen image by old_w/new_w -- observed as
        # "editing a parameter zooms the image out" in the Adjust panel, whose
        # live preview swaps the full-res display buffer for a half-size (or
        # smaller fast-base) render. For downgrades, always preserve on-screen
        # framing, and skip the MAX_SCALE clamp for the same reason
        # set_pixmap_zoomed_at has allow_overscale: matching the previous
        # on-screen magnification on a low-res stand-in may need > MAX_SCALE,
        # and the later full-quality swap lands back inside the normal range.
        is_upgrade = new_w >= old_w - 1
        if (
            not exact_framing
            and is_upgrade
            and (
                getattr(self, "_zoom_intent_100", False)
                # s_old >= 1 means "user was at/above 100%" ONLY when the view
                # was not simply fitted: content smaller than the viewport has
                # a fit scale >= 1, and treating that as 100%-intent snapped a
                # freshly fitted gallery tile to "100%" on gallery->single
                # ("clicking a gallery image lands on a zoomed-in view").
                or (s_old is not None and s_old >= 1.0 - 1e-4 and not was_fit_scale)
            )
        ):
            s_new = 1.0
        else:
            # Preserve on-screen image scale: s_new * new_w == s_old * old_w
            s_new = max(self.MIN_SCALE, s_old * old_w / new_w)
            if is_upgrade and not exact_framing:
                s_new = min(self.MAX_SCALE, s_new)
        self.resetTransform()
        self.scale(s_new, s_new)
        self.centerOn(QPointF(fx * new_w, fy * new_h))
        self._log_framing("preserve/pixmap")
        self._sync_fit_mode_flag()
        self.zoomChanged.emit(self.current_scale())

    def set_rgb_numpy(self, rgb, preserve_view=None, *, exact_framing: bool = False) -> bool:
        """Display HxWx3 uint8 RGB via OpenGL ``drawImage`` (no ``QPixmap``).

        Returns False if the buffer is unsuitable (caller should fall back to
        ``set_pixmap``). Framing semantics match ``set_pixmap``.
        """
        try:
            import numpy as np

            if rgb is None:
                return False
            arr = np.asarray(rgb)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return False
            new_h, new_w = int(arr.shape[0]), int(arr.shape[1])
            if new_w <= 0 or new_h <= 0:
                return False

            old_w, old_h = self._img_w, self._img_h
            had_pixmap = self._has_pixmap
            if preserve_view is None:
                preserve_view = not self._fit_mode
            capture = preserve_view and had_pixmap and old_w > 0 and old_h > 0
            s_old = fx = fy = None
            was_fit_scale = False
            if capture:
                s_old = self.current_scale()
                # See set_pixmap: "clearly above fit", not merely >= 100%.
                was_fit_scale = not (s_old > self.fit_scale() * 1.1)
                center_scene = self.mapToScene(self.viewport().rect().center())
                fx = center_scene.x() / old_w
                fy = center_scene.y() / old_h

            self._item.hide()
            self._item.setPixmap(QPixmap())
            self._rgb_item.set_rgb_uint8(arr)
            self._rgb_item.show()
            self._use_rgb_gl = True

            self._scene.setSceneRect(QRectF(0, 0, new_w, new_h))
            self._img_w, self._img_h = new_w, new_h
            self._has_pixmap = True
            self._grid_item.set_grid(new_w, new_h, self._grid_mode)
            if (new_w, new_h) != (old_w, old_h):
                self._refresh_dodge_burn_mask_overlay_size()
            self._update_placeholder()
            if self._compare_active and (new_w, new_h) != (old_w, old_h):
                self._update_compare_overlay()

            if self._fit_mode or not capture:
                if self._fit_mode and not getattr(self, "_zoom_intent_100", False):
                    self.fit_to_window()
                elif not capture:
                    if self.is_at_fit_scale() and not getattr(self, "_zoom_intent_100", False):
                        self.fit_to_window()
                self._sync_fit_mode_flag()
                return True

            is_upgrade = new_w >= old_w - 1
            if (
                not exact_framing
                and is_upgrade
                and (
                    getattr(self, "_zoom_intent_100", False)
                    # See set_pixmap: fit scale >= 1 for content smaller
                    # than the viewport must not read as 100%-intent.
                    or (s_old is not None and s_old >= 1.0 - 1e-4 and not was_fit_scale)
                )
            ):
                s_new = 1.0
            else:
                s_new = max(self.MIN_SCALE, s_old * old_w / new_w)
                if is_upgrade and not exact_framing:
                    s_new = min(self.MAX_SCALE, s_new)
            self.resetTransform()
            self.scale(s_new, s_new)
            self.centerOn(QPointF(fx * new_w, fy * new_h))
            self._log_framing("preserve/rgb_numpy")
            self._sync_fit_mode_flag()
            self.zoomChanged.emit(self.current_scale())
            return True
        except Exception:
            return False

    def set_device_rgb(self, device_rgb, preserve_view=None, *, exact_framing: bool = False) -> bool:
        """Display a CUDA ``DeviceRgb`` via CUDA↔GL (no host download for paint).

        Falls back to ``set_rgb_numpy`` if interop is unavailable. Framing
        matches ``set_pixmap`` / ``set_rgb_numpy``.
        """
        try:
            if device_rgb is None or not hasattr(device_rgb, "shape"):
                return False
            new_h = int(device_rgb.shape[0])
            new_w = int(device_rgb.shape[1])
            if new_w <= 0 or new_h <= 0:
                return False
            if not self.is_opengl_viewport():
                # No GL context → download and use host path.
                return self.set_rgb_numpy(
                    device_rgb.to_numpy(),
                    preserve_view=preserve_view,
                    exact_framing=exact_framing,
                )

            old_w, old_h = self._img_w, self._img_h
            had_pixmap = self._has_pixmap
            if preserve_view is None:
                preserve_view = not self._fit_mode
            capture = preserve_view and had_pixmap and old_w > 0 and old_h > 0
            s_old = fx = fy = None
            was_fit_scale = False
            if capture:
                s_old = self.current_scale()
                # See set_pixmap: "clearly above fit", not merely >= 100%.
                was_fit_scale = not (s_old > self.fit_scale() * 1.1)
                center_scene = self.mapToScene(self.viewport().rect().center())
                fx = center_scene.x() / old_w
                fy = center_scene.y() / old_h

            self._item.hide()
            self._item.setPixmap(QPixmap())
            self._rgb_item.set_device_rgb(device_rgb)
            self._rgb_item.show()
            self._use_rgb_gl = True

            self._scene.setSceneRect(QRectF(0, 0, new_w, new_h))
            self._img_w, self._img_h = new_w, new_h
            self._has_pixmap = True
            self._grid_item.set_grid(new_w, new_h, self._grid_mode)
            if (new_w, new_h) != (old_w, old_h):
                self._refresh_dodge_burn_mask_overlay_size()
            self._update_placeholder()
            if self._compare_active and (new_w, new_h) != (old_w, old_h):
                self._update_compare_overlay()

            if self._fit_mode or not capture:
                if self._fit_mode and not getattr(self, "_zoom_intent_100", False):
                    self.fit_to_window()
                elif not capture:
                    if self.is_at_fit_scale() and not getattr(self, "_zoom_intent_100", False):
                        self.fit_to_window()
                self._sync_fit_mode_flag()
                return True

            is_upgrade = new_w >= old_w - 1
            if (
                not exact_framing
                and is_upgrade
                and (
                    getattr(self, "_zoom_intent_100", False)
                    # See set_pixmap: fit scale >= 1 for content smaller
                    # than the viewport must not read as 100%-intent.
                    or (s_old is not None and s_old >= 1.0 - 1e-4 and not was_fit_scale)
                )
            ):
                s_new = 1.0
            else:
                s_new = max(self.MIN_SCALE, s_old * old_w / new_w)
                if is_upgrade and not exact_framing:
                    s_new = min(self.MAX_SCALE, s_new)
            self.resetTransform()
            self.scale(s_new, s_new)
            self.centerOn(QPointF(fx * new_w, fy * new_h))
            self._log_framing("preserve/device_rgb")
            self._sync_fit_mode_flag()
            self.zoomChanged.emit(self.current_scale())
            return True
        except Exception:
            return False

    def clear(self) -> None:
        self.clear_image(show_placeholder=True)

    def clear_image(self, *, show_placeholder: bool = True) -> None:
        """Drop the current image; optionally show the empty-state placeholder."""
        self._use_rgb_gl = False
        try:
            self._rgb_item.hide()
            self._rgb_item.clear_rgb()
        except Exception:
            pass
        self._item.show()
        self._item.setPixmap(QPixmap())
        self._has_pixmap = False
        self._img_w = self._img_h = 0
        self.clear_overlay()
        self.clear_clipping_overlay()
        self.set_compare_mode(False)
        self._grid_item.set_grid(0, 0, self._grid_mode)
        if show_placeholder:
            self._update_placeholder()
        else:
            self._placeholder.hide()

    def has_heavy_pixmap(self) -> bool:
        return bool(getattr(self, "_has_pixmap", False)) and max(
            int(getattr(self, "_img_w", 0) or 0),
            int(getattr(self, "_img_h", 0) or 0),
        ) >= 2048

    def is_opengl_viewport(self) -> bool:
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            return isinstance(self.viewport(), QOpenGLWidget)
        except Exception:
            return False

    def release_for_gallery_entry(self) -> None:
        """Drop single-view image resources when switching to gallery.

        On Windows with an OpenGL viewport, clearing a large GL-backed pixmap
        while the widget is still *visible* can abort the process. Once the
        single-view container is hidden, clearing is safe and needed — otherwise
        a 30+ MP texture stays on the GPU while gallery tiles decode.
        """
        self.file_path = None
        if getattr(self, "_use_rgb_gl", False):
            try:
                self._rgb_item.hide()
                self._rgb_item.clear_rgb()
                self._use_rgb_gl = False
                self._item.show()
            except Exception:
                pass
        if sys.platform == "win32" and self.is_opengl_viewport():
            if self._viewport_hidden_for_teardown():
                self._safe_clear_opengl_pixmap()
            else:
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(0, self._clear_for_gallery_if_hidden)
                QTimer.singleShot(50, self._clear_for_gallery_if_hidden)
                QTimer.singleShot(150, self._clear_for_gallery_if_hidden)
            return
        self.clear()

    def _viewport_hidden_for_teardown(self) -> bool:
        """True when this view (or an ancestor) is hidden — safe for GL pixmap drop."""
        w = self
        while w is not None:
            try:
                if not w.isVisible():
                    return True
            except Exception:
                return True
            w = w.parentWidget()
        return False

    def _safe_clear_opengl_pixmap(self) -> None:
        """Release GL texture backing the pixmap item (Windows gallery entry)."""
        vp = self.viewport()
        gl_widget = None
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            if isinstance(vp, QOpenGLWidget):
                gl_widget = vp
                gl_widget.makeCurrent()
        except Exception:
            gl_widget = None
        try:
            self.clear()
        except Exception:
            pass
        finally:
            if gl_widget is not None:
                try:
                    gl_widget.doneCurrent()
                except Exception:
                    pass

    def _clear_for_gallery_if_hidden(self) -> None:
        if not self.has_heavy_pixmap():
            return
        if not self._viewport_hidden_for_teardown():
            return
        import logging

        logging.getLogger(__name__).info(
            "[GPU_VIEW] Clearing hidden OpenGL pixmap (%dx%d) for gallery entry",
            int(getattr(self, "_img_w", 0) or 0),
            int(getattr(self, "_img_h", 0) or 0),
        )
        self._safe_clear_opengl_pixmap()

    def capture_viewport_pixmap(self) -> QPixmap | None:
        """Capture the visible view for resolution cross-fade (OpenGL-safe).

        ANY snapshot of a live QOpenGLWidget surface — grabFramebuffer(),
        QGraphicsView.render() over a GL viewport, or viewport.grab() — can abort
        the process (SIGABRT/SIGSEGV) on several GL drivers when background
        decodes are in flight. This was originally guarded on Windows only, but
        reproduced identically on macOS (a real crash: grabFramebuffer() segfaulted
        on the main thread while a background thread was mid-decode in
        extract_thumbnail_from_image, right after a gallery->single-view click) --
        so the guard is unconditional now. Return None so the caller falls back to
        the cached on-screen pixmap for the crossfade instead of touching the GL
        surface at all.
        """
        if not self._has_pixmap:
            return None
        vp = self.viewport()
        w = max(1, vp.width())
        h = max(1, vp.height())
        is_gl_viewport = False
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            is_gl_viewport = isinstance(vp, QOpenGLWidget)
        except Exception:
            is_gl_viewport = False

        if is_gl_viewport:
            return None
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

    def set_zoom_locked(self, locked: bool) -> None:
        """When True, keep fit-to-window; block pinch/wheel/100% zoom."""
        self._zoom_locked = bool(locked)

    def zoom_locked(self) -> bool:
        return bool(getattr(self, "_zoom_locked", False))

    def _zoom_in_blocked(self) -> bool:
        return self.zoom_locked()

    # ------------------------------------------------------------------ zoom
    def fit_to_window(self) -> None:
        self._fit_mode = True
        self._zoom_intent_100 = False
        if not self._has_pixmap:
            return
        self.resetTransform()
        target = self._rgb_item if getattr(self, "_use_rgb_gl", False) else self._item
        self.fitInView(target, Qt.AspectRatioMode.KeepAspectRatio)
        self._log_framing("fit")
        self.fitModeChanged.emit(True)
        self.zoomChanged.emit(self.current_scale())

    def _log_framing(self, where: str) -> None:
        """Diagnostic for the 'image sits off-center with a black gap' report.

        Off by default; set RAWVIEWER_DEBUG_FRAMING=1 to trace where the image
        lands after a fit. Logs the three rects that must agree: the item's own
        rect, the scene rect the view scrolls within, and where the item
        actually ends up in viewport pixels. A left/right gap means the mapped
        item rect is not centered in the viewport -- and the offending rect
        (item vs scene) says which side of the pipeline put it there.
        """
        if not _env_true("RAWVIEWER_DEBUG_FRAMING"):
            return
        try:
            import logging

            target = self._rgb_item if getattr(self, "_use_rgb_gl", False) else self._item
            vp = self.viewport().rect()
            mapped = self.mapFromScene(target.boundingRect()).boundingRect()
            sr = self._scene.sceneRect()
            logging.getLogger(__name__).info(
                "[FRAMING] %s path=%s img=%dx%d scene=(%.0f,%.0f %.0fx%.0f) "
                "viewport=%dx%d item_on_screen=(x=%d y=%d %dx%d) "
                "gap_left=%d gap_right=%d gap_top=%d gap_bottom=%d scale=%.4f",
                where,
                "rgb_gl" if getattr(self, "_use_rgb_gl", False) else "pixmap",
                self._img_w, self._img_h,
                sr.x(), sr.y(), sr.width(), sr.height(),
                vp.width(), vp.height(),
                mapped.x(), mapped.y(), mapped.width(), mapped.height(),
                mapped.left() - vp.left(), vp.right() - mapped.right(),
                mapped.top() - vp.top(), vp.bottom() - mapped.bottom(),
                self.current_scale(),
            )
        except Exception:
            pass

    def zoom_to_actual(self) -> None:
        """100% — one image pixel per view pixel, centered on the image middle."""
        w, h = self._img_w, self._img_h
        cx = w * 0.5 if w > 0 else 0.0
        cy = h * 0.5 if h > 0 else 0.0
        self.zoom_to_actual_at(cx, cy)

    def zoom_to_actual_at(self, scene_x: float, scene_y: float) -> None:
        """100% zoom with the given image pixel centered in the viewport."""
        if not self._has_pixmap or self._zoom_in_blocked():
            return
        self._fit_mode = False
        self._zoom_intent_100 = True
        x = max(0.0, min(float(scene_x), max(0, self._img_w - 1)))
        y = max(0.0, min(float(scene_y), max(0, self._img_h - 1)))
        self.resetTransform()
        self.scale(1.0, 1.0)

        # Force this view's own deferred geometry-update event (posted by
        # scale()/resetTransform(), otherwise processed on the next event-loop
        # iteration) so centerOn() below sees up-to-date scrollbar ranges
        # instead of the pre-zoom ones -- centerOn's setValue() calls clamp
        # against stale bounds, mis-centering until Qt catches up on its own.
        # Scoped to this widget only: a full QApplication.processEvents()
        # here pumped EVERY queued event for the whole app (timers, other
        # widgets' input/paint, queued signal/slot deliveries) while
        # mid-transform, risking reentrancy into this method or navigation
        # code from unrelated handlers. sendPostedEvents(self, 0) only
        # flushes events already queued for this view.
        QApplication.sendPostedEvents(self, 0)

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
        allow_overscale: bool = False,
    ) -> None:
        """Replace the image and apply zoom in one step (avoids a wrong-frame flash).

        ``allow_overscale``: skip the MAX_SCALE clamp. Used when displaying a
        low-resolution interim tier at the on-screen magnification its full
        resolution buffer will have (e.g. a 512px nav preview standing in for
        a 7000px sensor buffer at 100% needs ~13x) -- the subsequent same-file
        upgrade preserves on-screen scale, landing exactly back inside the
        normal range. Interactive zooming still clamps as usual.
        """
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
        self._grid_item.set_grid(new_w, new_h, self._grid_mode)
        self._update_placeholder()
        fit = self.fit_scale()
        s = float(scale)
        if s <= fit * self._FIT_SCALE_EPS:
            self.fit_to_window()
            return
        if not allow_overscale:
            s = min(self.MAX_SCALE, s)
        s = max(fit, s)
        x = max(0.0, min(float(scene_x), max(0, new_w - 1)))
        y = max(0.0, min(float(scene_y), max(0, new_h - 1)))
        self.resetTransform()
        self.scale(s, s)

        # See zoom_to_actual_at: flush only this view's own posted geometry
        # update so centerOn() below uses fresh scrollbar ranges, without
        # pumping the whole application's event queue.
        QApplication.sendPostedEvents(self, 0)

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
            if self._zoom_in_blocked():
                return
            self.zoom_to_actual()
        else:
            self.fit_to_window()

    def set_scale(self, scale: float) -> None:
        fit = self.fit_scale()
        scale = float(scale)
        if scale <= fit * self._FIT_SCALE_EPS:
            self.fit_to_window()
            return
        if self._zoom_in_blocked() and scale > fit * self._FIT_SCALE_EPS:
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
        if self._zoom_in_blocked() and new > fit * self._FIT_SCALE_EPS:
            return
        new = max(fit, min(self.MAX_SCALE, new))
        if abs(new - cur) < 1e-9:
            return
        self.scale(new / cur, new / cur)
        self._fit_mode = False
        self.fitModeChanged.emit(False)
        self.zoomChanged.emit(self.current_scale())

    # ------------------------------------------------------------- color pick
    def set_color_pick_mode(self, enabled: bool) -> None:
        """Arm/disarm one-shot pixel picking (e.g. white-balance dropper).

        While armed, the next left click samples an image-pixel coordinate via
        ``colorPickRequested`` instead of starting a pan drag or export drag, then
        disarms itself. Call again with ``enabled=False`` to cancel without picking.
        """
        self._color_pick_mode = bool(enabled)
        if self._color_pick_mode:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().unsetCursor()

    def is_color_pick_mode(self) -> bool:
        return bool(getattr(self, "_color_pick_mode", False))

    def set_dodge_burn_mode(self, enabled: bool) -> None:
        """Arm/disarm brush painting: mouse/tablet drags stamp dodgeBurnStroke
        instead of panning or driving the compare divider."""
        self._dodge_burn_mode = bool(enabled)
        if self._dodge_burn_mode:
            self._dodge_burn_confirmed_on_image = False
            self.set_crop_mode(False)
            # Hide OS cursor — the circular brush preview is the hit target.
            self.viewport().setCursor(Qt.CursorShape.BlankCursor)
            self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)
            self.setMouseTracking(True)
            self._brush_cursor_item.show()
            self._sync_brush_cursor_at_view_center()
        else:
            # Exiting the context also closes any in-progress hold-stroke.
            self.end_key_paint()
            self._brush_cursor_item.hide()
            self.viewport().unsetCursor()
            self.setMouseTracking(False)

    def begin_key_paint(self) -> None:
        """Hold-to-paint: a brush hotkey went down.

        Opens the key-driven paint gate. Painting itself is decoupled from the
        brush *context* (``_dodge_burn_mode``, which drives the cursor preview
        and wheel-to-resize and persists across holds): this only controls
        whether pointer movement stamps. If the pointer is already over the
        image, stamp immediately so a stationary tap leaves a single dab.
        """
        if not self._dodge_burn_mode or not self._has_pixmap:
            return
        self._db_key_held = True
        # The pointer is deliberately on the image for a hold; never let the
        # leave-image auto-disarm fire mid-hold.
        self._dodge_burn_confirmed_on_image = True
        from PyQt6.QtGui import QCursor

        view_pos = self.viewport().mapFromGlobal(QCursor.pos())
        if self._view_pos_on_image(view_pos):
            pt = self._clamped_scene_point(view_pos)
            self._place_brush_cursor(pt)
            self._dodge_burn_painting = True
            # No button is down, so Force Touch pressure is unavailable here by
            # design (accepted trade-off) -- stamp at full strength.
            self.dodgeBurnStroke.emit(pt, 1.0, False)

    def end_key_paint(self) -> None:
        """Hotkey released (or focus lost): close any active hold-stroke.

        Idempotent -- safe to call when no hold is in progress, which is what
        the focus-loss safety net relies on.
        """
        self._db_key_held = False
        self._brush_resizing = False
        if getattr(self, "_dodge_burn_painting", False):
            self._dodge_burn_painting = False
            from PyQt6.QtGui import QCursor

            view_pos = self.viewport().mapFromGlobal(QCursor.pos())
            pt = self._clamped_scene_point(view_pos)
            self.dodgeBurnStroke.emit(pt, 1.0, True)

    def set_dodge_burn_brush_radius(self, radius_px: float) -> None:
        """Brush radius in *display/scene* pixels (matches the Size slider)."""
        self._dodge_burn_brush_radius = max(2.0, float(radius_px))
        self._refresh_brush_cursor_placement()

    def set_dodge_burn_brush_flow(self, flow: float) -> None:
        """Per-stroke flow 0..1 — drives preview opacity (matches Brush Flow)."""
        self._dodge_burn_brush_flow = max(0.05, min(1.0, float(flow)))
        self._refresh_brush_cursor_placement()

    def _refresh_brush_cursor_placement(self) -> None:
        center = None
        if not self._brush_cursor_item.pixmap().isNull():
            off = self._brush_cursor_item.offset()
            pm = self._brush_cursor_item.pixmap()
            center = QPointF(
                off.x() + pm.width() * 0.5, off.y() + pm.height() * 0.5
            )
        self._ensure_brush_cursor_pixmap()
        if center is not None and self._brush_cursor_item.isVisible():
            self._place_brush_cursor(center)
        elif self._brush_cursor_item.isVisible():
            self._sync_brush_cursor_at_view_center()

    def _ensure_brush_cursor_pixmap(self) -> None:
        """Soft radial preview: size from radius, opacity from flow. No hard ring."""
        r = float(getattr(self, "_dodge_burn_brush_radius", 40.0))
        flow = float(getattr(self, "_dodge_burn_brush_flow", 0.55))
        # Rebuild when radius moves by >=0.5px or flow by >=2%.
        if (
            abs(r - float(getattr(self, "_brush_cursor_pixmap_r", -1.0))) < 0.5
            and abs(flow - float(getattr(self, "_brush_cursor_pixmap_flow", -1.0)))
            < 0.02
        ):
            if not self._brush_cursor_item.pixmap().isNull():
                return
        try:
            import numpy as np
            from raw_dodge_burn import circular_brush_falloff

            ri = max(2, int(np.ceil(r)))
            size = 2 * ri + 1
            falloff = circular_brush_falloff(0, size, 0, size, float(ri), float(ri), r)
            rgba = np.zeros((size, size, 4), dtype=np.uint8)
            # Soft white fill; peak alpha scales with Brush Flow so the preview
            # matches how strong each stamp will land.
            peak_alpha = 40.0 + flow * 160.0  # ~48 at flow 0.05 … ~200 at 1.0
            rgba[..., 0:3] = 255
            rgba[..., 3] = np.clip(falloff * peak_alpha, 0, 255).astype(np.uint8)
            rgba = np.ascontiguousarray(rgba)
            qimg = QImage(
                rgba.data,
                size,
                size,
                rgba.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()
            self._brush_cursor_item.setPixmap(QPixmap.fromImage(qimg))
            self._brush_cursor_pixmap_r = r
            self._brush_cursor_pixmap_flow = flow
        except Exception:
            # Fallback: empty — cursor still hidden OS-side while armed.
            self._brush_cursor_item.setPixmap(QPixmap())
            self._brush_cursor_pixmap_r = -1.0
            self._brush_cursor_pixmap_flow = -1.0

    def _place_brush_cursor(self, scene_pt: QPointF) -> None:
        self._ensure_brush_cursor_pixmap()
        pm = self._brush_cursor_item.pixmap()
        if pm.isNull():
            return
        self._brush_cursor_item.setOffset(
            scene_pt.x() - pm.width() * 0.5,
            scene_pt.y() - pm.height() * 0.5,
        )

    def _sync_brush_cursor_at_view_center(self) -> None:
        if not self._has_pixmap:
            return
        self._place_brush_cursor(
            self._clamped_scene_point(self.viewport().rect().center())
        )

    def is_dodge_burn_mode(self) -> bool:
        return bool(getattr(self, "_dodge_burn_mode", False))

    def set_crop_mode(
        self,
        enabled: bool,
        *,
        insets: tuple[float, float, float, float] | None = None,
        aspect: float | None = None,
        refit: bool = True,
    ) -> None:
        """Show/hide the interactive crop overlay (mutually exclusive with D&B)."""
        enabled = bool(enabled) and self._has_pixmap
        was_on = bool(getattr(self, "_crop_mode", False))
        self._crop_mode = enabled
        if enabled:
            self.set_dodge_burn_mode(False)
            self._crop_item.set_image_size(self._img_w, self._img_h)
            if insets is not None:
                self._crop_item.set_insets(*insets)
            self._crop_item.set_aspect_ratio(aspect)
            self._crop_item.show()
            # Skip fit when already in crop mode — Transform-slider ticks
            # re-enter overlay often; refitting every time feels laggy.
            if refit and not was_on:
                self.fit_to_window()
        else:
            self._crop_item.hide()
            self._crop_item._apply_hover_cursor("")
            self.viewport().unsetCursor()

    def is_crop_mode(self) -> bool:
        return bool(getattr(self, "_crop_mode", False))

    def set_crop_aspect_ratio(self, aspect: float | None) -> None:
        if getattr(self, "_crop_item", None) is not None:
            self._crop_item.set_aspect_ratio(aspect)

    def set_crop_insets(self, left: float, right: float, top: float, bottom: float) -> None:
        if getattr(self, "_crop_item", None) is not None:
            self._crop_item.set_insets(left, right, top, bottom)

    def crop_insets(self) -> tuple[float, float, float, float]:
        item = getattr(self, "_crop_item", None)
        if item is None:
            return (0.0, 0.0, 0.0, 0.0)
        return item.insets()

    def set_export_drag_enabled(self, enabled: bool) -> None:
        """Enable/disable fit-mode file drag-out (disabled while Adjust is open)."""
        self._export_drag_enabled = bool(enabled)
        if not self._export_drag_enabled:
            self._drag_start_pos = None
            self._drag_started = False

    def is_export_drag_enabled(self) -> bool:
        return bool(getattr(self, "_export_drag_enabled", True))

    def _export_drag_allowed(self) -> bool:
        """File drag-out is blocked in Adjust, crop, dodge/burn, and color-pick."""
        if not getattr(self, "_export_drag_enabled", True):
            return False
        if getattr(self, "_crop_mode", False):
            return False
        if getattr(self, "_dodge_burn_mode", False):
            return False
        if getattr(self, "_color_pick_mode", False):
            return False
        return True

    def _clamped_scene_point(self, view_pos) -> QPointF:
        scene_pt = self.mapToScene(view_pos)
        return QPointF(
            max(0.0, min(scene_pt.x(), max(0, self._img_w - 1))),
            max(0.0, min(scene_pt.y(), max(0, self._img_h - 1))),
        )

    def _view_pos_on_image(self, view_pos) -> bool:
        """True when ``view_pos`` maps onto the pixmap (not letterbox / off-view)."""
        if not getattr(self, "_has_pixmap", False):
            return False
        if self._img_w <= 0 or self._img_h <= 0:
            return False
        scene_pt = self.mapToScene(view_pos)
        return (
            0.0 <= scene_pt.x() < float(self._img_w)
            and 0.0 <= scene_pt.y() < float(self._img_h)
        )

    def _maybe_emit_brush_tool_left_image(self, view_pos=None) -> None:
        if not getattr(self, "_dodge_burn_mode", False):
            return
        if getattr(self, "_dodge_burn_painting", False):
            return
        if view_pos is not None and self._view_pos_on_image(view_pos):
            return
        if not getattr(self, "_dodge_burn_confirmed_on_image", False):
            # Never disarm before the pointer has landed on the image at
            # least once since arming -- see the flag's definition in
            # __init__ for why (avoids a spurious disarm on the very first
            # move/press after toggling the tool on).
            return
        self.brushToolLeftImage.emit()

    # --------------------------------------------------------- compare split
    def set_compare_original_pixmap(self, pixmap: QPixmap | None) -> None:
        """Provide the unedited render to show on the left of the divider."""
        self._compare_original_pixmap = pixmap if pixmap is not None else QPixmap()
        if self._compare_active:
            self._update_compare_overlay()

    def set_compare_mode(self, enabled: bool) -> None:
        """Show/hide the before/after split view over the current edited pixmap."""
        enabled = bool(enabled) and self._has_pixmap and not self._compare_original_pixmap.isNull()
        self._compare_active = enabled
        if not enabled:
            self._compare_dragging_divider = False
            self._compare_overlay_item.hide()
            self._compare_divider_line.hide()
            self._compare_divider_handle.hide()
            self.viewport().unsetCursor()
            return
        self._update_compare_overlay()

    def is_compare_mode(self) -> bool:
        return bool(getattr(self, "_compare_active", False))

    def _compare_divider_scene_x(self) -> float:
        return max(0.0, min(1.0, self._compare_split_frac)) * self._img_w

    def _update_compare_overlay(self) -> None:
        if not self._compare_active or self._img_w <= 0 or self._img_h <= 0:
            return
        src = self._compare_original_pixmap
        if src.isNull():
            return
        if src.width() != self._img_w or src.height() != self._img_h:
            # Prefer letterboxed KeepAspectRatio over IgnoreAspectRatio so a
            # full-frame original is not stretched into a cropped edited frame.
            src = src.scaled(
                self._img_w,
                self._img_h,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            if src.width() != self._img_w or src.height() != self._img_h:
                x0 = max(0, (src.width() - self._img_w) // 2)
                y0 = max(0, (src.height() - self._img_h) // 2)
                src = src.copy(x0, y0, self._img_w, self._img_h)
        split_x = int(round(self._compare_divider_scene_x()))
        split_x = max(0, min(self._img_w, split_x))
        if split_x <= 0:
            self._compare_overlay_item.hide()
        else:
            crop_w = min(split_x, src.width())
            crop_h = min(self._img_h, src.height())
            cropped = src.copy(0, 0, crop_w, crop_h)
            self._compare_overlay_item.setPixmap(cropped)
            self._compare_overlay_item.setOffset(0, 0)
            self._compare_overlay_item.show()
        self._compare_divider_line.setLine(split_x, 0, split_x, self._img_h)
        self._compare_divider_line.show()
        handle_r = 12.0
        self._compare_divider_handle.setRect(
            split_x - handle_r, self._img_h / 2.0 - handle_r, handle_r * 2, handle_r * 2
        )
        self._compare_divider_handle.show()

    def _compare_divider_hit(self, view_pos: QPoint) -> bool:
        """Hit-test the divider with a generous on-screen tolerance (cosmetic pen width)."""
        if not self._compare_active:
            return False
        scene_x = self._compare_divider_scene_x()
        divider_view_x = self.mapFromScene(QPointF(scene_x, 0)).x()
        return abs(view_pos.x() - divider_view_x) <= 16

    def _set_compare_split_from_view_x(self, view_x: int) -> None:
        scene_pt = self.mapToScene(QPoint(view_x, 0))
        frac = 0.0 if self._img_w <= 0 else scene_pt.x() / self._img_w
        self._compare_split_frac = max(0.0, min(1.0, frac))
        self._update_compare_overlay()

    # ------------------------------------------------------------------ events
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dodge_burn_mode:
            if self._has_pixmap:
                view_pos = event.position().toPoint()
                if not self._view_pos_on_image(view_pos):
                    self._maybe_emit_brush_tool_left_image(view_pos)
                    event.accept()
                    return
                self._dodge_burn_confirmed_on_image = True
                self._dodge_burn_painting = True
                pt = self._clamped_scene_point(view_pos)
                self._place_brush_cursor(pt)
                self.dodgeBurnStroke.emit(pt, self._mouse_stroke_pressure(event), False)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._color_pick_mode:
            if self._has_pixmap:
                scene_pt = self.mapToScene(event.position().toPoint())
                pt = QPointF(
                    max(0.0, min(scene_pt.x(), max(0, self._img_w - 1))),
                    max(0.0, min(scene_pt.y(), max(0, self._img_h - 1))),
                )
                self.set_color_pick_mode(False)
                self.colorPickRequested.emit(pt)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._compare_active:
            pos = event.position().toPoint()
            if self._compare_divider_hit(pos):
                self._compare_dragging_divider = True
                self._set_compare_split_from_view_x(pos.x())
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton and self._fit_mode and self.file_path:
            if self._export_drag_allowed():
                self._drag_start_pos = event.position().toPoint()
                self._drag_started = False
            else:
                self._drag_start_pos = None
                self._drag_started = False
        elif event.button() == Qt.MouseButton.LeftButton and not self._fit_mode:
            self._manual_pan_start = event.position().toPoint()
            self._manual_pan_scroll_h = self.horizontalScrollBar().value()
            self._manual_pan_scroll_v = self.verticalScrollBar().value()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dodge_burn_mode and self._has_pixmap:
            view_pos = event.position().toPoint()
            button_painting = (event.buttons() & Qt.MouseButton.LeftButton) and getattr(
                self, "_dodge_burn_painting", False
            )
            key_held = getattr(self, "_db_key_held", False)
            on_image = self._view_pos_on_image(view_pos)
            if on_image:
                self._dodge_burn_confirmed_on_image = True
            # A held brush key keeps the tool live even off-image (the stroke
            # just doesn't stamp until the pointer is back over the photo), so
            # a mid-hold excursion must not trip the leave-image auto-disarm.
            if not button_painting and not key_held and not on_image:
                self._maybe_emit_brush_tool_left_image(view_pos)
                event.accept()
                return
            pt = self._clamped_scene_point(view_pos)
            self._place_brush_cursor(pt)
            # Mid-stroke resize gesture in progress: keep the cursor tracking but
            # stamp nothing, and leave the stroke open (no is_stroke_end) so the
            # whole paint+resize+paint sequence remains a single undo unit.
            if getattr(self, "_brush_resizing", False):
                event.accept()
                return
            if button_painting:
                self.dodgeBurnStroke.emit(pt, self._mouse_stroke_pressure(event), False)
                event.accept()
                return
            if key_held and on_image:
                # Starts the stroke on first on-image move when the key went
                # down off-image; the host captures the undo baseline on the
                # first non-end stamp, so this stays one undo unit per hold.
                self._dodge_burn_painting = True
                self.dodgeBurnStroke.emit(pt, 1.0, False)
                event.accept()
                return
            if key_held:
                # Held but off-image: keep the tool live, stamp nothing.
                event.accept()
                return
        host = self.parentWidget()
        if host is not None:
            if hasattr(host, "_ensure_filmstrip_enabled"):
                host._ensure_filmstrip_enabled()
            if hasattr(host, "handle_pointer_for_filmstrip"):
                host.handle_pointer_for_filmstrip(event.globalPosition())

        if self._compare_active and not self._color_pick_mode:
            pos = event.position().toPoint()
            if self._compare_dragging_divider:
                if event.buttons() & Qt.MouseButton.LeftButton:
                    self._set_compare_split_from_view_x(pos.x())
                    event.accept()
                    return
                self._compare_dragging_divider = False
            if self._compare_divider_hit(pos):
                self.viewport().setCursor(Qt.CursorShape.SplitHCursor)
            elif not (event.buttons() & Qt.MouseButton.LeftButton):
                self.viewport().unsetCursor()

        if (
            (event.buttons() & Qt.MouseButton.LeftButton)
            and self._fit_mode
            and self.file_path
            and getattr(self, "_drag_start_pos", None) is not None
            and self._export_drag_allowed()
        ):
            if not self._drag_started:
                dist = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
                if dist >= QApplication.startDragDistance():
                    self._drag_started = True
                    self._start_drag()
            event.accept()
            return
            
        if (event.buttons() & Qt.MouseButton.LeftButton) and not self._fit_mode and getattr(self, "_manual_pan_start", None) is not None:
            delta = event.position().toPoint() - self._manual_pan_start
            self.horizontalScrollBar().setValue(self._manual_pan_scroll_h - delta.x())
            self.verticalScrollBar().setValue(self._manual_pan_scroll_v - delta.y())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and getattr(
            self, "_dodge_burn_painting", False
        ):
            self._dodge_burn_painting = False
            view_pos = event.position().toPoint()
            pt = self._clamped_scene_point(view_pos)
            self.dodgeBurnStroke.emit(pt, self._mouse_stroke_pressure(event), True)
            if not self._view_pos_on_image(view_pos):
                self._maybe_emit_brush_tool_left_image(view_pos)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._compare_dragging_divider:
            self._compare_dragging_divider = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._fit_mode:
            self._drag_start_pos = None
            self._drag_started = False
        elif event.button() == Qt.MouseButton.LeftButton and not self._fit_mode:
            self._manual_pan_start = None
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        # Leaving the view (e.g. into the Adjust panel) disarms brush tools.
        self._maybe_emit_brush_tool_left_image(None)
        super().leaveEvent(event)

    def _start_drag(self) -> None:
        if not self._export_drag_allowed():
            return
        if not self.file_path or not os.path.exists(self.file_path):
            return
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(self.file_path)])
        stamp_rawviewer_export_drag(mime_data)
        drag.setMimeData(mime_data)

        # Try to capture the current viewport pixmap to use as the drag thumbnail
        try:
            drag_px = self.capture_viewport_pixmap()
            if drag_px and not drag_px.isNull():
                drag_px = drag_px.scaled(140, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                drag.setPixmap(drag_px)
                drag.setHotSpot(QPoint(drag_px.width() // 2, drag_px.height() // 2))
        except Exception:
            pass

        drag.exec(Qt.DropAction.CopyAction)

    def _handle_native_gesture(self, event) -> bool:
        """Trackpad pinch (macOS) and smart-zoom gestures."""
        if not self._has_pixmap or event.type() != QEvent.Type.NativeGesture:
            return False
        gtype = event.gestureType()
        if gtype == Qt.NativeGestureType.ZoomNativeGesture:
            if self._zoom_in_blocked():
                event.accept()
                return True
            self.zoom_by(1.0 + event.value() * 0.5)
            event.accept()
            return True
        if gtype == Qt.NativeGestureType.SmartZoomNativeGesture:
            self.toggle_fit()
            event.accept()
            return True
        return False

    def viewportEvent(self, event) -> bool:
        # macOS delivers pinch/double-click to the OpenGL viewport, not the view's event().
        if self._handle_native_gesture(event):
            return True
        if event.type() == QEvent.Type.MouseButtonDblClick:
            self.mouseDoubleClickEvent(event)
            return True
        return super().viewportEvent(event)

    def event(self, event) -> bool:
        if self._handle_native_gesture(event):
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        handler = getattr(self, "_shortcut_handler", None)
        if callable(handler) and handler(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def tabletEvent(self, event) -> None:
        """Real pressure from a graphics tablet/stylus (Wacom, iPad+Sidecar, ...).

        Trackpads do not deliver ``QTabletEvent`` (stylus-only). Mouse /
        trackpad strokes use ``mousePressEvent`` / ``mouseMoveEvent`` and
        ``_mouse_stroke_pressure`` (Qt pressure when available, else macOS
        Force Touch via ``NSEvent.pressure``, else full strength).
        """
        if not self._dodge_burn_mode or not self._has_pixmap:
            event.ignore()
            return
        from PyQt6.QtCore import QEvent as _QEvent

        pressure = max(0.0, min(1.0, float(event.pressure())))
        view_pos = event.position().toPoint()
        etype = event.type()
        painting = getattr(self, "_dodge_burn_painting", False)
        if self._view_pos_on_image(view_pos):
            self._dodge_burn_confirmed_on_image = True
        elif (
            etype == _QEvent.Type.TabletMove
            and not painting
        ):
            self._maybe_emit_brush_tool_left_image(view_pos)
            event.accept()
            return
        pt = self._clamped_scene_point(view_pos)
        self._place_brush_cursor(pt)
        if etype == _QEvent.Type.TabletPress:
            if not self._view_pos_on_image(view_pos):
                event.accept()
                return
            self._dodge_burn_painting = True
            self.dodgeBurnStroke.emit(pt, pressure, False)
        elif etype == _QEvent.Type.TabletMove and painting:
            self.dodgeBurnStroke.emit(pt, pressure, False)
        elif etype == _QEvent.Type.TabletRelease:
            self._dodge_burn_painting = False
            self.dodgeBurnStroke.emit(pt, pressure, True)
            if not self._view_pos_on_image(view_pos):
                self._maybe_emit_brush_tool_left_image(view_pos)
        event.accept()

    def wheelEvent(self, event) -> None:
        # Mid-stroke resize gesture (hold-to-paint only): a macOS trackpad
        # brackets a two-finger scroll with ScrollBegin/ScrollEnd, so we can
        # suspend stamping for the duration, resize on the updates, and resume
        # the same stroke when the fingers lift. This is handled BEFORE the
        # delta==0 guard below because ScrollEnd on macOS carries zero delta --
        # dropping it there would strand the brush in resize-suspend forever
        # (fingers lifted but painting never resumes). Platforms/devices without
        # phase report NoScrollPhase and fall through to plain per-tick resize.
        if (
            getattr(self, "_dodge_burn_mode", False)
            and getattr(self, "_db_key_held", False)
            and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            phase = event.phase()
            if phase == Qt.ScrollPhase.ScrollBegin:
                self._brush_resizing = True
            elif phase == Qt.ScrollPhase.ScrollEnd:
                # Fingers lifted -- resume painting and reset the stamp anchor.
                self._brush_resizing = False
                self.dodgeBurnResumeAfterResize.emit()
                event.accept()
                return
            elif phase == Qt.ScrollPhase.ScrollMomentum:
                # Inertia after the fingers lifted: ignore so the brush does not
                # keep growing once the gesture is physically over.
                event.accept()
                return

        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        # Dodge/Burn armed: two-finger scroll changes Brush Size, not image.
        if getattr(self, "_dodge_burn_mode", False) and not (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.dodgeBurnBrushSizeWheel.emit(int(delta))
            event.accept()
            return
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if ctrl:
            if self._zoom_in_blocked():
                event.accept()
                return
            # Control + Scroll: Zoom smoothly
            self.zoom_by(1.25 if delta > 0 else 0.8)
            event.accept()
            return

        # Plain wheel (no Control Modifier)
        if not getattr(self, "_wheel_navigate_in_fit", True):
            # Editor / Adjust panel mode: disable plain wheel scrolling over image entirely
            event.accept()
            return

        if self._fit_mode:
            # Fit-to-window: navigate images (Qt: delta > 0 = up, delta < 0 = down)
            self.wheelNavigate.emit(-1 if delta > 0 else 1)
            event.accept()
            return

        # Zoomed in: pan vertically by delegating to QGraphicsView scroll behavior
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._has_pixmap:
                scene_pt = self.mapToScene(event.position().toPoint())
                pt = QPointF(
                    max(0.0, min(scene_pt.x(), max(0, self._img_w - 1))),
                    max(0.0, min(scene_pt.y(), max(0, self._img_h - 1))),
                )
            else:
                pt = QPointF(0.0, 0.0)
            self.doubleClickedAt.emit(pt)
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode and self._has_pixmap:
            self.fit_to_window()
        self._update_placeholder()

    def viewport_scroll_state(self) -> tuple[float, int, int]:
        return (
            self.current_scale(),
            self.horizontalScrollBar().value(),
            self.verticalScrollBar().value(),
        )

    def apply_viewport_scroll_state(
        self, scale: float, h_scroll: int, v_scroll: int
    ) -> None:
        if not self._has_pixmap:
            return
        fit = self.fit_scale()
        self.resetTransform()
        if scale > fit * self._FIT_SCALE_EPS:
            self._fit_mode = False
            self._zoom_intent_100 = scale >= 1.0 - 1e-4
            self.scale(scale, scale)
            self.horizontalScrollBar().setValue(int(h_scroll))
            self.verticalScrollBar().setValue(int(v_scroll))
            self.fitModeChanged.emit(False)
        else:
            self.fit_to_window()
