"""GPU view part 1: scene item hosting, GL enable, overlays, fit/zoom state."""

from __future__ import annotations

import contextlib
import sys

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import (
    QGraphicsRectItem,
)

from rawviewer_ui.widgets.gpu.gpu_state import _env_true, gpu_cuda_gl_enabled


class GpuImageViewPart1:

    def set_file_path(self, file_path: str) -> None:
        self.file_path = file_path
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
    def _dodge_burn_overlay_rgba(self, data, is_heal: bool):
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
            from raw_adjustments.spot_heal import HealMask

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
            self._zoom_anchor = None
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
        anchor_dx = anchor_dy = 0.0
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
            anchor = self._zoom_anchor
            if anchor is not None:
                fx, fy, anchor_dx, anchor_dy = anchor
                self._zoom_anchor = None

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
            elif not capture and self.is_at_fit_scale() and not getattr(self, "_zoom_intent_100", False):
                    # Stale state: _fit_mode False but transform still at fit scale (e.g. zoom
                    # requested before the first pixmap arrived). Refit so Space/double-click
                    # toggles predictably instead of staying stuck at ~fit%.
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
        # The anchor's screen offset from the viewport centre is kept fixed, so the
        # point under the pointer does not move when the sharper buffer swaps in.
        self.centerOn(QPointF(
            fx * new_w - anchor_dx / s_new,
            fy * new_h - anchor_dy / s_new,
        ))
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
                if self._fit_mode and not getattr(self, "_zoom_intent_100", False) or not capture and self.is_at_fit_scale() and not getattr(self, "_zoom_intent_100", False):
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
                if self._fit_mode and not getattr(self, "_zoom_intent_100", False) or not capture and self.is_at_fit_scale() and not getattr(self, "_zoom_intent_100", False):
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
                with contextlib.suppress(Exception):
                    gl_widget.doneCurrent()
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
