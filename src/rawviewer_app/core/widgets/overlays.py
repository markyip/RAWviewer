"""Overlays built on the shared Qt widget layer."""
from __future__ import annotations

import contextlib

from PyQt6.QtCore import QAbstractAnimation, QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QSizePolicy, QWidget

from core import theme

from .helpers import _filmstrip_ui_env_int


class SingleImageViewOverlay(QWidget):
    """Scroll area fills the widget; histogram + film strip float on top."""

    _HIST_MARGIN = 8
    _FILMSTRIP_HEIGHT = 88
    # Hot zone extends above the strip so reveal starts before the cursor reaches thumbnails.
    _FILMSTRIP_HOTZONE_EXTRA = 40
    _BOTTOM_HOTZONE = _FILMSTRIP_HEIGHT + _FILMSTRIP_HOTZONE_EXTRA
    _FILMSTRIP_FADE_MS = 200
    _FILMSTRIP_HIDE_DELAY_MS = 90
    _FILMSTRIP_SHOW_DELAY_MS = _filmstrip_ui_env_int(
        "RAWVIEWER_FILMSTRIP_SHOW_DELAY_MS", 500
    )
    _FILMSTRIP_SHOW_DELAY_DIRECT_MS = _filmstrip_ui_env_int(
        "RAWVIEWER_FILMSTRIP_SHOW_DELAY_DIRECT_MS", 70
    )

    def __init__(self, scroll_area, histogram_widget, viewer=None, parent=None,
                 gpu_view=None, map_widget=None, adjust_widget=None):
        super().__init__(parent)
        self._viewer = viewer
        self._scroll = scroll_area
        self._gpu_view = gpu_view
        self._hist = histogram_widget
        self._map = map_widget
        self._adj = adjust_widget
        self._hist_user_placed = False
        self._map_user_placed = False
        self._adj_user_placed = False
        self._filmstrip_pointer_active = False
        self._filmstrip_reveal = False
        scroll_area.setParent(self)
        if gpu_view is not None:
            gpu_view.setParent(self)
            gpu_view.installEventFilter(self)
            gpu_view.viewport().installEventFilter(self)
            # Legacy scroll area sits under the OpenGL surface; on macOS it can steal
            # mouse/gesture input unless hidden (see docs/macos-sharing-v21-v22.md).
            self.set_gpu_view_active(True)
        histogram_widget.setParent(self)
        if map_widget is not None:
            map_widget.setParent(self)
            map_widget.hide()
        if adjust_widget is not None:
            adjust_widget.setParent(self)
            adjust_widget.hide()
        self.setObjectName("single_view_container")
        self.setStyleSheet(f"#single_view_container {{ background-color: {theme.VOID}; }}")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)

        from rawviewer_ui.widgets.filmstrip import FilmStripBar
        # Host on the main window (not beside QOpenGLWidget) so the strip overlays the image.
        film_host = viewer if viewer is not None else self
        self._filmstrip_layer = QWidget(film_host)
        self._filmstrip_layer.setObjectName("filmstrip_layer")
        self._filmstrip_layer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._filmstrip_layer.hide()
        self._filmstrip = FilmStripBar(self._filmstrip_layer, viewer=viewer)
        self._filmstrip_opacity = QGraphicsOpacityEffect(self._filmstrip_layer)
        self._filmstrip_layer.setGraphicsEffect(self._filmstrip_opacity)
        self._filmstrip_opacity.setOpacity(0.0)
        self._filmstrip_fade_anim = None
        self._filmstrip_fade_gen = 0
        self._filmstrip.hide()
        self._hide_filmstrip_layer()
        self._hide_filmstrip_timer = QTimer(self)
        self._hide_filmstrip_timer.setSingleShot(True)
        self._hide_filmstrip_timer.timeout.connect(self._hide_filmstrip_if_inactive)
        self._show_filmstrip_timer = QTimer(self)
        self._show_filmstrip_timer.setSingleShot(True)
        self._show_filmstrip_timer.timeout.connect(self._show_filmstrip_if_still_in_hotzone)

        from PyQt6.QtWidgets import QLabel

        self.recovery_badge = QLabel(self)
        self.recovery_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Recovery preview literally lifts shadows / holds back highlights --
        # the two darkroom operations BURN's cool slate-blue represents.
        self.recovery_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {theme.rgba(theme.BURN_RGB, 200)};
                color: {theme.INK};
                font-size: 12px;
                font-weight: 600;
                border-radius: 4px;
                padding: 5px 10px;
                border: 1px solid {theme.rgba(theme.BURN_RGB, 140)};
            }}
        """)
        self.recovery_badge.hide()

    def set_recovery_preview_badge(self, visible: bool, *, detail: str = "", loading: bool = False) -> None:
        badge = getattr(self, "recovery_badge", None)
        if badge is None:
            return
        if visible:
            if loading:
                text = "Loading recovery preview…"
            else:
                text = "Recovery preview"
                if detail:
                    text = f"{text}{detail}"
                text = f"{text} · fit only · P to exit"
            badge.setText(text)
            badge.adjustSize()
            badge.show()
            self.relayout_recovery_badge()
            badge.raise_()
        else:
            badge.hide()

    def relayout_recovery_badge(self) -> None:
        badge = getattr(self, "recovery_badge", None)
        if badge is None or not badge.isVisible():
            return
        margin = 14
        badge.move(max(margin, self.width() - badge.width() - margin), margin)
        badge.raise_()

    def _filmstrip_layer_visible(self) -> bool:
        layer = getattr(self, "_filmstrip_layer", None)
        return layer is not None and layer.isVisible()

    def _show_filmstrip_layer(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is not None:
            layer.show()
        self._filmstrip.show()
        self._layout_filmstrip_inner()
        self._raise_filmstrip_chrome()

    def _hide_filmstrip_layer(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is not None:
            layer.hide()
        self._filmstrip.hide()

    def _layout_filmstrip_inner(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return
        self._filmstrip.setGeometry(0, 0, max(1, layer.width()), max(1, layer.height()))

    def _raise_filmstrip_chrome(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return
        if not layer.isVisible() and self._filmstrip_opacity_value() <= 0.0:
            return
        layer.raise_()
        layer.update()

    def _stop_filmstrip_fade(self) -> None:
        self._filmstrip_fade_gen += 1
        anim = getattr(self, "_filmstrip_fade_anim", None)
        if anim is not None:
            # Disconnect all signals BEFORE stop() to prevent the synchronous
            # finished emission from triggering _finish_hide during cancellation.
            with contextlib.suppress(TypeError, RuntimeError):
                anim.finished.disconnect()
            with contextlib.suppress(TypeError, RuntimeError):
                anim.valueChanged.disconnect()
            if anim.state() == QAbstractAnimation.State.Running:
                anim.stop()
        self._filmstrip_fade_anim = None

    def _filmstrip_opacity_value(self) -> float:
        try:
            return float(self._filmstrip_opacity.opacity())
        except Exception:
            return 0.0

    def _filmstrip_accepts_pointer(self) -> bool:
        return (
            self._filmstrip_layer_visible()
            and self._filmstrip_opacity_value() > 0.35
        )

    def _update_filmstrip_hit_testing(self) -> None:
        accept = self._filmstrip_accepts_pointer()
        self._filmstrip.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, not accept
        )
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is not None:
            layer.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, not accept
            )

    def _animate_filmstrip_opacity(self, target: float, on_finished=None) -> None:
        target = max(0.0, min(1.0, float(target)))
        self._stop_filmstrip_fade()

        self._filmstrip_opacity.setOpacity(target)
        self._update_filmstrip_hit_testing()
        self._raise_filmstrip_chrome()

        if on_finished is not None:
            on_finished()

    def _fade_in_filmstrip(self) -> None:
        if not self._filmstrip.isEnabled():
            return
        self._show_filmstrip_timer.stop()
        self._hide_filmstrip_timer.stop()

        self._stop_filmstrip_fade()
        self._filmstrip_reveal = True

        if self._filmstrip_opacity_value() >= 0.99:
            return

        needs_setup = (
            not self._filmstrip_layer_visible()
            or self._filmstrip_opacity_value() <= 0.01
        )
        if needs_setup:
            self._filmstrip_opacity.setOpacity(0.0)
            self._show_filmstrip_layer()
            self._layout_filmstrip()
            if hasattr(self._filmstrip, "center_on_current"):
                QTimer.singleShot(0, lambda: self._filmstrip.center_on_current())
            if hasattr(self._filmstrip, "refresh_visible_thumbnails"):
                QTimer.singleShot(0, lambda: self._filmstrip.refresh_visible_thumbnails(
                    refresh_cache=True
                ))
        self._raise_filmstrip_chrome()
        self._animate_filmstrip_opacity(1.0)

    def _fade_out_filmstrip(self) -> None:
        if not self._filmstrip_layer_visible():
            return

        def _finish_hide():
            self._hide_filmstrip_layer()
            self._filmstrip_opacity.setOpacity(0.0)
            self._filmstrip_reveal = False
            self._update_filmstrip_hit_testing()
            if self._viewer is not None:
                self._viewer._filmstrip_pointer_active = False

        self._animate_filmstrip_opacity(0.0, on_finished=_finish_hide)

    def filmstrip_widget(self):
        return getattr(self, "_filmstrip", None)

    def set_filmstrip_pointer_active(self, active: bool) -> None:
        if active:
            if self._filmstrip_pointer_active:
                return
            self._filmstrip_pointer_active = True
            self._hide_filmstrip_timer.stop()
            self._fade_in_filmstrip()
            if self._viewer is not None:
                self._viewer._filmstrip_pointer_active = True
            return
        self._filmstrip_pointer_active = False
        if self._viewer is not None:
            self._viewer._filmstrip_pointer_active = False
        self._schedule_hide_filmstrip()

    def _schedule_hide_filmstrip(self) -> None:
        self._hide_filmstrip_timer.start(self._FILMSTRIP_HIDE_DELAY_MS)

    def _cancel_show_filmstrip(self) -> None:
        self._show_filmstrip_timer.stop()
        if (
            self._filmstrip_layer_visible()
            and self._filmstrip_opacity_value() <= 0.01
            and not self._filmstrip_pointer_active
        ):
            self._hide_filmstrip_layer()
            self._filmstrip_reveal = False
            self._update_filmstrip_hit_testing()

    def _prepare_filmstrip_reveal(self) -> None:
        """Warm layout and thumbnails while the hover delay runs (opacity stays 0)."""
        if not self._filmstrip.isEnabled():
            return
        if self._filmstrip_layer_visible() and self._filmstrip_opacity_value() > 0.01:
            return
        self._filmstrip_reveal = True
        self._filmstrip_opacity.setOpacity(0.0)
        self._show_filmstrip_layer()
        self._layout_filmstrip()
        self._update_filmstrip_hit_testing()
        if hasattr(self._filmstrip, "center_on_current"):
            self._filmstrip.center_on_current()
        if hasattr(self._filmstrip, "refresh_visible_thumbnails"):
            self._filmstrip.refresh_visible_thumbnails(refresh_cache=True)
        viewer = self._viewer
        if viewer is not None and hasattr(viewer, "_prefetch_filmstrip_thumbnails"):
            viewer._prefetch_filmstrip_thumbnails()

    def _schedule_show_filmstrip(self, delay_ms: int | None = None) -> None:
        if self._filmstrip_opacity_value() >= 0.99:
            return
        delay = (
            int(self._FILMSTRIP_SHOW_DELAY_MS)
            if delay_ms is None
            else max(0, int(delay_ms))
        )
        if self._show_filmstrip_timer.isActive():
            remaining = self._show_filmstrip_timer.remainingTime()
            if remaining >= 0 and remaining <= delay:
                return
        self._prepare_filmstrip_reveal()
        self._show_filmstrip_timer.start(delay)

    def _show_filmstrip_if_still_in_hotzone(self) -> None:
        if not self._filmstrip.isEnabled():
            return
        if self._filmstrip.underMouse() or self._pointer_in_bottom_hotzone():
            self._fade_in_filmstrip()

    def _hide_filmstrip_if_inactive(self) -> None:
        if self._filmstrip_pointer_active:
            return
        if self._pointer_in_bottom_hotzone():
            return
        self._filmstrip_reveal = False
        self._fade_out_filmstrip()

    def _pointer_in_bottom_hotzone(self) -> bool:
        pos = self.mapFromGlobal(QCursor.pos())
        return self._pointer_in_bottom_hotzone_at(pos)

    def _pointer_in_bottom_hotzone_at(self, pos: QPoint) -> bool:
        return self.rect().contains(pos) and (
            pos.y() >= self.height() - self._BOTTOM_HOTZONE
        )

    def _pointer_in_filmstrip_band_at(self, pos: QPoint) -> bool:
        """Bottom strip band (where thumbnails appear), not the wider approach zone."""
        return self.rect().contains(pos) and (
            pos.y() >= self.height() - self._FILMSTRIP_HEIGHT
        )

    def _local_pos_from_global(self, global_pos) -> QPoint:
        if hasattr(global_pos, "toPoint"):
            global_pos = global_pos.toPoint()
        return self.mapFromGlobal(global_pos)

    def _filmstrip_rect_in_container(self) -> QRect:
        """Filmstrip geometry mapped into this overlay's coordinate system."""
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return QRect()
        if (
            not layer.isVisible()
            and self._filmstrip_opacity_value() <= 0.0
            and not self._filmstrip_reveal
        ):
            return QRect()
        if layer.parentWidget() is self:
            return layer.geometry()
        global_top_left = layer.mapToGlobal(QPoint(0, 0))
        local_top_left = self.mapFromGlobal(global_top_left)
        return QRect(local_top_left, layer.size())

    def _ensure_filmstrip_enabled(self) -> None:
        """Enable/sync filmstrip before hover handling (GPU path may skip main eventFilter)."""
        viewer = self._viewer
        if viewer is not None and hasattr(viewer, "_ensure_filmstrip_synced"):
            viewer._ensure_filmstrip_synced()

    def handle_pointer_for_filmstrip(self, global_pos) -> None:
        """Reveal/hide film strip when pointer moves over scroll area (child widgets)."""
        self._ensure_filmstrip_enabled()
        if not self._filmstrip.isEnabled():
            return
        pos = self._local_pos_from_global(global_pos)
        in_hot = self._pointer_in_bottom_hotzone_at(pos)
        self._pointer_in_filmstrip_band_at(pos)
        strip_rect = self._filmstrip_rect_in_container()
        over_strip = (
            self._filmstrip_accepts_pointer()
            and strip_rect.isValid()
            and strip_rect.contains(pos)
        )
        strip_visible = self._filmstrip_opacity_value() > 0.35

        # If the filmstrip is already visible and we are hovering over it, keep it visible instantly.
        is_hovering_visible_strip = strip_visible and (
            over_strip or self._filmstrip.underMouse() or (
                getattr(self, "_filmstrip_layer", None) is not None
                and self._filmstrip_layer.underMouse()
            )
        )

        if is_hovering_visible_strip:
            self._cancel_show_filmstrip()
            self._hide_filmstrip_timer.stop()
            self._fade_in_filmstrip()
        elif in_hot:
            self._hide_filmstrip_timer.stop()
            if strip_visible:
                # Keep it visible without delay if we are in the hotzone
                self._cancel_show_filmstrip()
                self._fade_in_filmstrip()
            else:
                # Force the full 500ms delay when the filmstrip is not visible,
                # ignoring the 70ms in_strip_band shortcut so it doesn't accidentally trigger.
                self._schedule_show_filmstrip(self._FILMSTRIP_SHOW_DELAY_MS)
        else:
            self._cancel_show_filmstrip()
            if not self._filmstrip_pointer_active:
                self._schedule_hide_filmstrip()

    def eventFilter(self, watched, event):
        from PyQt6.QtCore import QEvent

        gpu = self._gpu_view
        if gpu is not None and watched in (gpu, gpu.viewport()) and event.type() in (
            QEvent.Type.MouseMove,
            QEvent.Type.HoverMove,
            QEvent.Type.Enter,
        ):
            self.handle_pointer_for_filmstrip(event.globalPosition())
        return False

    def _reveal_filmstrip(self) -> None:
        """Backward-compatible alias for fade-in."""
        self._fade_in_filmstrip()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Scroll area stays full size; film strip overlays the bottom (no image reflow).
        self._scroll.setGeometry(0, 0, self.width(), self.height())
        self._scroll.lower()
        if self._gpu_view is not None:
            self._gpu_view.setGeometry(0, 0, self.width(), self.height())
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "_try_apply_pending_overlay_positions"):
            viewer._try_apply_pending_overlay_positions()
        self._layout_filmstrip()
        self._layout_histogram()
        self._layout_map()
        self._layout_adjust()
        self.relayout_recovery_badge()
        self._raise_single_view_layers()

    def set_gpu_view_active(self, active: bool) -> None:
        """Toggle GPU vs legacy scroll input/display (GIF/WebP use scroll only)."""
        gpu = self._gpu_view
        if gpu is None:
            return
        if active:
            gpu.show()
            self._scroll.hide()
        else:
            gpu.hide()
            self._scroll.show()
        self._raise_single_view_layers()

    def _raise_single_view_layers(self) -> None:
        """Keep image view under filmstrip / histogram chrome."""
        self._scroll.lower()
        if self._gpu_view is not None and self._gpu_view.isVisible():
            self._gpu_view.raise_()
        if hasattr(self, "recovery_badge") and self.recovery_badge.isVisible():
            self.recovery_badge.raise_()
        if self._map is not None and self._map.isVisible():
            self._map.raise_()
        if self._hist.isVisible():
            self._hist.raise_()
        if self._adj is not None and self._adj.isVisible():
            self._adj.raise_()
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "_restore_keyboard_focus"):
            QTimer.singleShot(0, viewer._restore_keyboard_focus)

    def _layout_adjust(self):
        panel = self._adj
        if panel is None or not panel.isVisible():
            return
        pw, ph = self.width(), self.height()
        aw, ah = panel.width(), panel.height()
        if pw < 1 or ph < 1:
            return
        if not self._adj_user_placed:
            x = max(self._HIST_MARGIN, pw - aw - self._HIST_MARGIN)
            y = max(self._HIST_MARGIN, (ph - ah) // 2)
            x = min(max(0, x), max(0, pw - aw))
            y = min(max(0, y), max(0, ph - ah))
            panel.move(x, y)
        else:
            x = min(max(0, panel.x()), max(0, pw - aw))
            y = min(max(0, panel.y()), max(0, ph - ah))
            panel.move(x, y)
        panel.raise_()

    def mark_adjust_user_moved(self):
        self._adj_user_placed = True
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "schedule_save_session_state"):
            viewer.schedule_save_session_state()

    def relayout_adjust(self):
        self._layout_adjust()

    def _layout_map(self):
        m = self._map
        if m is None or not m.isVisible():
            return
        pw, ph = self.width(), self.height()
        mw, mh = m.width(), m.height()
        if pw < 1 or ph < 1:
            return
        if not self._map_user_placed:
            x = self._HIST_MARGIN
            y = self._HIST_MARGIN
            x = min(max(0, x), max(0, pw - mw))
            y = min(max(0, y), max(0, ph - mh))
            m.move(x, y)
        else:
            x = min(max(0, m.x()), max(0, pw - mw))
            y = min(max(0, m.y()), max(0, ph - mh))
            m.move(x, y)
        m.raise_()

    def relayout_map(self):
        self._layout_map()

    def mark_map_user_moved(self):
        self._map_user_placed = True
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "schedule_save_session_state"):
            viewer.schedule_save_session_state()

    def overlay_session_snapshot(self) -> dict:
        """Normalized overlay positions (0–1) for cross-session restore."""
        pw, ph = max(1, self.width()), max(1, self.height())
        out: dict = {}
        if self._hist_user_placed:
            out["histogram"] = {"x": self._hist.x() / pw, "y": self._hist.y() / ph}
        if self._map is not None and self._map.isVisible() and self._map_user_placed:
            out["map"] = {"x": self._map.x() / pw, "y": self._map.y() / ph}
        if self._adj is not None and self._adj.isVisible() and self._adj_user_placed:
            out["adjust"] = {"x": self._adj.x() / pw, "y": self._adj.y() / ph}
        return out

    def apply_overlay_session_positions(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        pw, ph = max(1, self.width()), max(1, self.height())
        hist = data.get("histogram")
        if isinstance(hist, dict):
            try:
                x = int(float(hist["x"]) * pw)
                y = int(float(hist["y"]) * ph)
                hw, hh = self._hist.width(), self._hist.height()
                self._hist.move(
                    min(max(0, x), max(0, pw - hw)),
                    min(max(0, y), max(0, ph - hh)),
                )
                self._hist_user_placed = True
            except (KeyError, TypeError, ValueError):
                pass
        map_pos = data.get("map")
        if map_pos and self._map is not None:
            try:
                x = int(float(map_pos["x"]) * pw)
                y = int(float(map_pos["y"]) * ph)
                mw, mh = self._map.width(), self._map.height()
                self._map.move(
                    min(max(0, x), max(0, pw - mw)),
                    min(max(0, y), max(0, ph - mh)),
                )
                self._map_user_placed = True
            except (KeyError, TypeError, ValueError):
                pass
        adjust_pos = data.get("adjust")
        if adjust_pos and self._adj is not None:
            try:
                x = int(float(adjust_pos["x"]) * pw)
                y = int(float(adjust_pos["y"]) * ph)
                aw, ah = self._adj.width(), self._adj.height()
                self._adj.move(
                    min(max(0, x), max(0, pw - aw)),
                    min(max(0, y), max(0, ph - ah)),
                )
                self._adj_user_placed = True
            except (KeyError, TypeError, ValueError):
                pass

    def _layout_filmstrip(self):
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return
        if (
            not layer.isVisible()
            and self._filmstrip_opacity_value() <= 0.0
            and not self._filmstrip_reveal
        ):
            return
        fh = self._FILMSTRIP_HEIGHT
        w = max(1, self.width())
        host = layer.parentWidget()
        if host is not None and host is not self:
            top_left = self.mapToGlobal(QPoint(0, max(0, self.height() - fh)))
            bottom_right = self.mapToGlobal(QPoint(w, self.height()))
            origin = host.mapFromGlobal(top_left)
            corner = host.mapFromGlobal(bottom_right)
            layer.setGeometry(
                QRect(origin, corner).normalized().intersected(host.rect())
            )
        else:
            layer.setGeometry(0, self.height() - fh, w, fh)
        self._layout_filmstrip_inner()
        self._raise_filmstrip_chrome()

    def mouseMoveEvent(self, event):
        self.handle_pointer_for_filmstrip(event.globalPosition())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._cancel_show_filmstrip()
        self._schedule_hide_filmstrip()
        super().leaveEvent(event)

    def _layout_histogram(self):
        h = self._hist
        if not h.isVisible():
            return
        pw, ph = self.width(), self.height()
        hw, hh = h.width(), h.height()
        if pw < 1 or ph < 1:
            return
        if not self._hist_user_placed:
            x = max(self._HIST_MARGIN, pw - hw - self._HIST_MARGIN)
            y = self._HIST_MARGIN
            x = min(max(0, x), max(0, pw - hw))
            y = min(max(0, y), max(0, ph - hh))
            h.move(x, y)
        else:
            x = min(max(0, h.x()), max(0, pw - hw))
            y = min(max(0, h.y()), max(0, ph - hh))
            h.move(x, y)
        h.raise_()

    def mark_histogram_user_moved(self):
        self._hist_user_placed = True
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "schedule_save_session_state"):
            viewer.schedule_save_session_state()

    def relayout_histogram(self):
        self._layout_histogram()


class LoadingOverlay(QWidget):
    """Semi-transparent loading overlay with spinner-like message"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setVisible(False)
        self._message = "Loading Image..."

    def set_message(self, message):
        self._message = message
        self.update()

    def show_loading(self, message=None):
        if message:
            self._message = message
        self.setVisible(True)
        self.raise_()
        self.update()

    def hide_loading(self):
        self.setVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent background
        painter.setBrush(QColor(*theme.VOID_RGB, 120))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

        # Draw loading box
        box_width = 240
        box_height = 80
        x = (self.width() - box_width) // 2
        y = (self.height() - box_height) // 2

        painter.setBrush(QColor(*theme.RAISED_RGB, 230))
        painter.setPen(QPen(QColor(*theme.INK_FAINT_RGB), 1))
        painter.drawRoundedRect(x, y, box_width, box_height, 10, 10)

        # Draw text
        painter.setPen(QColor(theme.INK))
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x, y, box_width, box_height, Qt.AlignmentFlag.AlignCenter, self._message)

    def resizeEvent(self, event):
        # Already covers parent because we resize it in parent's resizeEvent
        super().resizeEvent(event)
