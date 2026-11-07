"""GPU view part 2: interactions, events, compare overlay, scroll state."""

from __future__ import annotations

import os

from PyQt6.QtCore import QEvent, QMimeData, QPoint, QPointF, QRectF, Qt, QUrl
from PyQt6.QtGui import QDrag, QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
)

from rawviewer_ui.widgets import stamp_rawviewer_export_drag
from rawviewer_ui.widgets.gpu.gpu_state import _env_true


class GpuImageViewPart2:

    def set_zoom_locked(self, locked: bool) -> None:
        """When True, keep fit-to-window; block pinch/wheel/100% zoom."""
        self._zoom_locked = bool(locked)
    def zoom_locked(self) -> bool:
        return bool(getattr(self, "_zoom_locked", False))
    def _zoom_in_blocked(self) -> bool:
        return self.zoom_locked()
    def fit_to_window(self) -> None:
        self._fit_mode = True
        self._zoom_intent_100 = False
        self._zoom_anchor = None
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
        self._zoom_anchor = None
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
        self._zoom_anchor = None
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
    def zoom_by(self, factor: float, view_pos: QPoint | None = None) -> None:
        """Zoom by ``factor``, keeping the image point under ``view_pos`` fixed.

        ``view_pos`` is in viewport coordinates (pointer position for wheel and
        pinch). Without it the zoom is about the viewport centre.
        """
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
        if view_pos is None:
            view_pos = self.viewport().rect().center()
        vc = self.viewport().rect().center()
        under_pointer = self.mapToScene(view_pos)
        self.scale(new / cur, new / cur)
        self._fit_mode = False
        QApplication.sendPostedEvents(self, 0)
        # Re-centre so the scene point under the pointer maps back to the same
        # viewport pixel. This replaces Qt's AnchorUnderMouse, which depends on a
        # mouse position the view may not have seen for this wheel event.
        dx = view_pos.x() - vc.x()
        dy = view_pos.y() - vc.y()
        self.centerOn(QPointF(
            under_pointer.x() - dx / new,
            under_pointer.y() - dy / new,
        ))
        iw, ih = max(1, self._img_w), max(1, self._img_h)
        self._zoom_anchor = (
            under_pointer.x() / iw,
            under_pointer.y() / ih,
            float(dx),
            float(dy),
        )
        self.fitModeChanged.emit(False)
        self.zoomChanged.emit(self.current_scale())
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
            self._brush_cursor_item.hide()
            self.viewport().unsetCursor()
            self.setMouseTracking(False)
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
        ) and not self._brush_cursor_item.pixmap().isNull():
            return
        try:
            import numpy as np

            from raw_adjustments.dodge_burn import circular_brush_falloff

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
        return not getattr(self, "_color_pick_mode", False)
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
    def mousePressEvent(self, event) -> None:
        # A drag-pan moves the view, so a stored wheel-zoom anchor is stale.
        self._zoom_anchor = None
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
            painting = (event.buttons() & Qt.MouseButton.LeftButton) and getattr(
                self, "_dodge_burn_painting", False
            )
            on_image = self._view_pos_on_image(view_pos)
            if on_image:
                self._dodge_burn_confirmed_on_image = True
            if not painting and not on_image:
                self._maybe_emit_brush_tool_left_image(view_pos)
                event.accept()
                return
            pt = self._clamped_scene_point(view_pos)
            self._place_brush_cursor(pt)
            if painting:
                self.dodgeBurnStroke.emit(pt, self._mouse_stroke_pressure(event), False)
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
            pointer = self.viewport().mapFromGlobal(event.globalPosition().toPoint())
            self.zoom_by(1.0 + event.value() * 0.5, pointer)
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
            # Control + Scroll: zoom about the pointer
            self.zoom_by(1.25 if delta > 0 else 0.8, event.position().toPoint())
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
