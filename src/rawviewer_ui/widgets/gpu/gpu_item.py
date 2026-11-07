"""GPU-accelerated scene items: OpenGL RGB image item + comparison split handle."""

from __future__ import annotations

import contextlib

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QTransform
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsObject,
)


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
        # Slots dropped while no GL context was current; released at the next
        # paint. See _defer_cuda_gl_release.
        self._pending_cuda_release: list = []
        self._rgba_device = None
        self._cuda_upload_failed = False
        self._cuda_uploaded_key = None  # (id(device_rgb), generation) after upload
        self._gl_fns = None

    def set_rgb_uint8(self, rgb) -> None:
        import numpy as np

        self._defer_cuda_gl_release()
        arr = np.ascontiguousarray(rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"expected HxWx3+ RGB, got shape={getattr(arr, 'shape', None)}")
        if arr.dtype == np.uint16:
            arr = (arr / 257.0).astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.floating):
            from raw_adjustments.tone_recovery import _encode_srgb8
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
        self._defer_cuda_gl_release()
        self.update()

    def clear_rgb(self) -> None:
        self.prepareGeometryChange()
        self._defer_cuda_gl_release()
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

    def _defer_cuda_gl_release(self) -> None:
        """Queue the interop slot for release at the next paint.

        set_rgb_uint8 / set_device_rgb / clear_rgb used to call
        _release_cuda_gl(None) directly. With no widget there is no
        makeCurrent(), so glDeleteTextures ran with NO CURRENT CONTEXT --
        which does not delete the texture, it just clears our handle, leaking
        the GL object while still unregistering the CUDA resource.

        This item is a QGraphicsObject; the GL widget only reaches it as
        paint()'s ``widget`` argument. So the release is deferred to there,
        where a context is guaranteed current. Freeing a frame later is
        strictly better than not freeing at all.
        """
        slot = self._cuda_slot
        self._cuda_slot = None
        self._rgba_device = None
        self._cuda_uploaded_key = None
        if slot is not None:
            self._pending_cuda_release.append(slot)
            # paint() is the reliable place to free this, but it is not a
            # guaranteed one: the item can be destroyed, the scene cleared, or
            # the view switched away without this item ever painting again --
            # and then the texture and the CUDA registration leak. Schedule a
            # fallback for the next event-loop turn, which finds the GL widget
            # itself. If paint gets there first this is a no-op.
            self._schedule_pending_cuda_drain()

    def _gl_widget(self):
        """The QOpenGLWidget viewport backing this item's scene, if any."""
        try:
            scene = self.scene()
            if scene is None:
                return None
            for view in scene.views():
                vp = view.viewport()
                if vp is not None and hasattr(vp, "makeCurrent"):
                    return vp
        except Exception:
            pass
        return None

    def _schedule_pending_cuda_drain(self) -> None:
        from PyQt6.QtCore import QTimer

        def _drain():
            try:
                if not self._pending_cuda_release:
                    return
                widget = self._gl_widget()
                if widget is None:
                    # No context reachable. Drop the CUDA registration, which
                    # does not need GL, and let the texture go with the
                    # context when it is destroyed.
                    self._drain_pending_cuda_releases(None)
                    return
                try:
                    widget.makeCurrent()
                    self._drain_pending_cuda_releases(widget)
                finally:
                    with contextlib.suppress(Exception):
                        widget.doneCurrent()
            except RuntimeError:
                # Underlying C++ item already deleted; nothing left to free
                # through it.
                pass
            except Exception:
                pass

        with contextlib.suppress(Exception):
            QTimer.singleShot(0, _drain)

    def _drain_pending_cuda_releases(self, widget) -> None:
        """Release deferred interop slots. Call from paint(), context current."""
        if not self._pending_cuda_release:
            return
        pending, self._pending_cuda_release = self._pending_cuda_release, []
        for slot in pending:
            self._release_cuda_slot(slot, widget)

    def _release_cuda_gl(self, widget) -> None:
        slot = self._cuda_slot
        self._cuda_slot = None
        self._rgba_device = None
        self._cuda_uploaded_key = None
        if slot is None:
            return
        self._release_cuda_slot(slot, widget)

    def _release_cuda_slot(self, slot, widget) -> None:
        if slot is None:
            return
        try:

            def _del_tex(tex_id: int) -> None:
                fns = self._gl_fns
                if fns is None:
                    return
                fns.glDeleteTextures(1, [tex_id])

            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.makeCurrent()
            slot.release(gl_delete_fn=_del_tex)
            if widget is not None:
                with contextlib.suppress(Exception):
                    widget.doneCurrent()
        except Exception:
            with contextlib.suppress(Exception):
                slot.release()

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
            from PyQt6.QtOpenGL import QOpenGLFunctions_2_0
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            from processing.raw_decode.gpu_bridge import (
                CudaGlTextureSlot,
                ensure_rgba_u8,
                gl_tex_image_2d_rgba_empty,
                register_gl_texture,
                upload_device_rgba_to_gl,
            )
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
            with contextlib.suppress(Exception):
                self._release_cuda_gl(None if context_already_current else widget)
            return False

    def _apply_painter_gl_matrices(self, painter: QPainter, fns) -> None:
        """Map item pixel space through QPainter's transform (Compatibility profile)."""

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
        # A GL context is current here, which is the only place it reliably
        # is -- release anything dropped while it was not.
        if self._pending_cuda_release:
            with contextlib.suppress(Exception):
                self._drain_pending_cuda_releases(widget)
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
                    with contextlib.suppress(Exception):
                        fns.glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
                    with contextlib.suppress(Exception):
                        fns.glColor4f(1.0, 1.0, 1.0, 1.0)
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
                    with contextlib.suppress(Exception):
                        painter.endNativePainting()
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
