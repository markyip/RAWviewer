"""Draggable RAW adjustment panel for single-image view (E to toggle)."""

from __future__ import annotations

from PyQt6.QtCore import (
    QObject,
    QRectF,
    QRunnable,
    QSettings,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from core import theme
from raw_adjustments.tone_curve import (
    TONE_CURVE_BLUE_KEY,
    TONE_CURVE_GREEN_KEY,
    TONE_CURVE_RED_KEY,
    TONE_CURVE_SERIAL_KEY,
)

_CHANNEL_CURVE_KEYS_BY_NAME = {
    "RGB": TONE_CURVE_SERIAL_KEY,
    "R": TONE_CURVE_RED_KEY,
    "G": TONE_CURVE_GREEN_KEY,
    "B": TONE_CURVE_BLUE_KEY,
}

# Point-curve + parametric PV rows.
_SHOW_TONE_CURVE_UI = True

_TRANSFORM_SLIDER_KEYS = frozenset(
    {"CropAngle", "PerspectiveVertical", "PerspectiveHorizontal", "Distortion"}
)

# Session-wide copy/paste clipboard for edit settings (survives navigation and
# panel rebuilds; intentionally not persisted across app restarts).
_EDIT_SETTINGS_CLIPBOARD: dict | None = None


def _edit_settings_clipboard() -> dict | None:
    return _EDIT_SETTINGS_CLIPBOARD


def _adjust_combo_stylesheet() -> str:
    """Shared QComboBox chrome: no separate drop-down button (click anywhere)."""
    return f"""
        QComboBox {{
            background-color: {theme.RAISED};
            border: 1px solid {theme.LINE};
            border-radius: 3px;
            color: {theme.INK};
            font-size: 11px;
            padding: 3px 8px;
            min-height: 18px;
        }}
        QComboBox:hover {{
            border-color: {theme.rgba(theme.INK_RGB, 70)};
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 0px;
            border: none;
            background: transparent;
        }}
        QComboBox::down-arrow {{
            image: none;
            width: 0px;
            height: 0px;
            border: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {theme.SURFACE};
            border: 1px solid {theme.LINE};
            color: {theme.INK};
            selection-background-color: {theme.rgba(theme.EMBER_RGB, 140)};
            selection-color: {theme.INK};
            outline: none;
            font-size: 11px;
            padding: 2px;
        }}
        QComboBox QAbstractItemView::item {{
            min-height: 22px;
            padding: 2px 6px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {theme.rgba(theme.EMBER_RGB, 90)};
            color: {theme.INK};
        }}
    """


# HSL section — hidden pending a saturation/vibrance review (see docs).
_SHOW_HSL_UI = True
# Dodge & burn brush — disabled pending a brush-shape/intensity-accumulation
# rework (brush currently paints a hard square instead of a soft circular
# falloff, and stops accumulate strength too fast). Widgets are still built
# (so main.py's wiring/attribute access stays valid) but hidden.
# Hidden until soft-brush + accumulation were tuned (see stamp strength in
# main._on_dodge_burn_stroke). Re-enabled 2026-07 with lower per-stamp delta
# and live patch preview so full pipeline isn't paid on every mouse move.
_SHOW_DODGE_BURN_UI = True

if _SHOW_TONE_CURVE_UI:
    pass

_PARAMETRIC_TONE_KEYS = frozenset(
    {
        "ParametricShadows",
        "ParametricDarks",
        "ParametricLights",
        "ParametricHighlights",
    }
)

# Dim background hint of what direction a slider pushes -- kept muted (not
# the saturated hues a web mockup can get away with) so it reads as a subtle
# cue under the solid accent fill, not a second, competing signal.
_SLIDER_TRACK_GRADIENTS: dict[str, list[tuple[float, str]]] = {
    "Temperature": [(0.0, "#4A73B5"), (0.5, theme.LINE), (1.0, "#C98A46")],
    "Tint": [(0.0, "#4A9B5E"), (0.5, theme.LINE), (1.0, "#B457A0")],
}


def _qta_icon_safe(name: str, *, color: str) -> QIcon:
    """qtawesome icon that degrades to an empty (invisible) QIcon if unavailable."""
    try:
        import qtawesome as qta

        return qta.icon(name, color=color)
    except Exception:
        return QIcon()


class AdjustValueLabel(QLabel):
    """Clickable value readout — resets the slider to its default."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


_SECTION_EXPANDED_SETTINGS_PREFIX = "adjust_panel/section_expanded/"


class _FileDropFrame(QFrame):
    """Drop target for files ending in ``suffixes``.

    Always accepts URL drags so unsupported drops can surface a warning
    instead of the OS silently refusing the drop.
    """

    files_dropped = pyqtSignal(list)
    unsupported_dropped = pyqtSignal(list)  # rejected local paths (basenames ok)

    def __init__(self, suffixes: tuple[str, ...], parent=None):
        super().__init__(parent)
        self._suffixes = tuple(s.lower() for s in suffixes)

    def _accepts(self, path: str) -> bool:
        pl = path.lower()
        return any(pl.endswith(s) for s in self._suffixes)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        ok: list[str] = []
        bad: list[str] = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if not p:
                continue
            if self._accepts(p):
                ok.append(p)
            else:
                bad.append(p)
        if ok:
            self.files_dropped.emit(ok)
        if bad:
            self.unsupported_dropped.emit(bad)
        if ok or bad:
            event.acceptProposedAction()
        else:
            event.ignore()


# Back-compat alias
class _LutDropFrame(_FileDropFrame):
    def __init__(self, parent=None):
        super().__init__((".cube",), parent)



_LOOKS_NAME_ROLE = int(Qt.ItemDataRole.UserRole) + 1
_LOOKS_THUMB_PX = 72
_LUT_THUMB_CACHE_MAX = 160
_LUT_THUMB_DEBOUNCE_MS = 140

_GLOBAL_SECTION_ORDER_SETTING = "adjust_panel/global_section_order"
_GLOBAL_SECTION_DEFAULT_ORDER = (
    "light",
    "color",
    "curve",
    "hsl",
    "detail",
    "noise",
    "effects",
    "local",
    "lut",
    "transform",
)

ADJUST_DOCK_SETTINGS_KEY = "adjust/dock"
ADJUST_DOCK_DEFAULT = "right"


def _sanitize_section_order(saved) -> list:
    """A saved order → a usable one, dropping junk and appending what is new."""
    known = list(_GLOBAL_SECTION_DEFAULT_ORDER)
    if not isinstance(saved, (list, tuple)):
        return known
    seen = set()
    order = []
    for key in saved:
        if isinstance(key, str) and key in known and key not in seen:
            order.append(key)
            seen.add(key)
    order.extend(k for k in known if k not in seen)
    return order


def normalize_adjust_dock(value) -> str:
    """``left`` or ``right`` (default). Unknown values fall back to right."""
    side = str(value or "").strip().lower()
    return "left" if side == "left" else ADJUST_DOCK_DEFAULT


def apply_adjust_splitter_dock(splitter: QSplitter, image_widget, panel, side: str) -> str:
    """Put the Adjust panel on the left or right of a horizontal QSplitter."""
    side = normalize_adjust_dock(side)
    if splitter is None or image_widget is None or panel is None:
        return side
    if splitter.indexOf(image_widget) < 0 or splitter.indexOf(panel) < 0:
        return side
    sizes = splitter.sizes()
    pan_i = splitter.indexOf(panel)
    old_panel_w = sizes[pan_i] if 0 <= pan_i < len(sizes) else 0
    target = 0 if side == "left" else 1
    if splitter.indexOf(panel) != target:
        splitter.insertWidget(target, panel)
    img_i = splitter.indexOf(image_widget)
    pan_i = splitter.indexOf(panel)
    splitter.setStretchFactor(img_i, 1)
    splitter.setStretchFactor(pan_i, 0)
    if panel.isVisible() and splitter.width() > 1:
        panel_w = max(
            int(panel.minimumWidth() or 0),
            int(getattr(panel, "_PANEL_W", 440) or 440),
            int(old_panel_w or 0),
        )
        image_w = max(120, max(splitter.width(), 1) - panel_w)
        if side == "left":
            splitter.setSizes([panel_w, image_w])
        else:
            splitter.setSizes([image_w, panel_w])
    return side


class _LutThumbSignals(QObject):
    finished = pyqtSignal(int, object)  # token, dict[lut_id -> ndarray]|None


class _LutThumbWorker(QRunnable):
    """Apply each .cube to a tiny as-shot display copy. No RAW decode, no user adj."""

    def __init__(
        self,
        token: int,
        display_u8,
        lut_names: list[str],
        amount: float,
        space: str,
        signals: _LutThumbSignals,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.token = int(token)
        self.display_u8 = display_u8
        self.lut_names = list(lut_names)
        self.amount = float(amount)
        self.space = str(space)
        self.signals = signals

    def run(self) -> None:
        out = None
        try:
            import numpy as np

            from raw_adjustments.lut import render_lut_chooser_previews

            base = self.display_u8
            if base is None:
                self.signals.finished.emit(self.token, None)
                return
            tiny = np.asarray(base)
            if str(getattr(tiny, "dtype", "")) != "uint8":
                tiny = np.clip(
                    np.asarray(tiny, dtype=np.float32) * 255.0 + 0.5, 0, 255
                ).astype(np.uint8)
            out = render_lut_chooser_previews(
                tiny, self.lut_names, amount=self.amount, space=self.space
            )
        except Exception:
            out = None
        self.signals.finished.emit(self.token, out)


def _u8_rgb_to_pixmap(rgb) -> QPixmap | None:
    """GUI-thread QPixmap from contiguous uint8 HxWx3."""
    try:
        import numpy as np

        if rgb is None or getattr(rgb, "ndim", 0) != 3:
            return None
        arr = np.ascontiguousarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if h < 1 or w < 1 or int(arr.shape[2]) < 3:
            return None
        img = QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format.Format_RGB888)
        return QPixmap.fromImage(img.copy())
    except Exception:
        return None


class _LooksRowWidget(QWidget):
    """One Looks list row: optional LUT thumb + name + type badge + remove (×)."""

    remove_clicked = pyqtSignal()

    def __init__(self, name: str, kind: str, parent=None):
        super().__init__(parent)
        self._suppress_item_click = False
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 2, 2)
        row.setSpacing(6)

        self._preview = QLabel()
        self._preview.setFixedSize(_LOOKS_THUMB_PX, _LOOKS_THUMB_PX)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet(
            f"background: {theme.VOID}; border-radius: 3px; border: 1px solid {theme.LINE};"
        )
        self._preview.hide()
        row.addWidget(self._preview)

        title = QLabel(name)
        title.setStyleSheet(f"color: {theme.INK}; font-size: 11px; background: transparent;")
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row.addWidget(title, 1)

        badge = QLabel(".cube" if kind == "cube" else ".xmp")
        badge.setStyleSheet(
            f"color: {theme.INK_FAINT}; font-size: 10px; background: transparent;"
        )
        row.addWidget(badge)

        rem = QPushButton("×")
        rem.setFixedSize(22, 22)
        rem.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rem.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        rem.setToolTip("Remove from library")
        rem.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {theme.INK_MUTED};
                border: none;
                border-radius: 4px;
                font-size: 14px;
                padding: 0;
            }}
            QPushButton:hover {{
                background: {theme.RAISED_HI};
                color: {theme.INK};
            }}
            """
        )
        rem.clicked.connect(self._on_remove_clicked)
        row.addWidget(rem)

    def set_preview(self, pixmap: QPixmap | None) -> None:
        """Show a LUT thumb; hide when there is no current image."""
        if pixmap is None or pixmap.isNull():
            self._preview.hide()
            self._preview.clear()
            self._sync_list_item_size()
            return
        scaled = pixmap.scaled(
            _LOOKS_THUMB_PX,
            _LOOKS_THUMB_PX,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)
        self._preview.show()
        self._sync_list_item_size()

    def _sync_list_item_size(self) -> None:
        parent = self.parent()
        while parent is not None and not isinstance(parent, QListWidget):
            parent = parent.parent()
        if parent is None:
            return
        hint = QSize(0, max(28, self.sizeHint().height()))
        for i in range(parent.count()):
            item = parent.item(i)
            if parent.itemWidget(item) is self:
                item.setSizeHint(hint)
                break

    def _on_remove_clicked(self) -> None:
        self._suppress_item_click = True
        self.remove_clicked.emit()


class CollapsibleSection(QWidget):
    """A clean, Lightroom-style collapsible accordion section for PyQt6.

    When constructed with a ``settings_key``, the expanded/collapsed state
    persists across sessions (QSettings) -- so a user who never touches HSL
    doesn't have to keep collapsing it every time they open the editor.
    """
    def __init__(self, title: str, parent=None, *, settings_key: str | None = None):
        super().__init__(parent)
        self._settings_key = settings_key
        self._expanded = self._load_expanded_default()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        # Header button/panel
        self.header = QWidget()
        self.header.setObjectName("accordion_header")
        self.header.setStyleSheet(f"""
            QWidget#accordion_header {{
                background-color: transparent;
                border: none;
                border-bottom: 1px solid {theme.LINE};
            }}
            QWidget#accordion_header:hover {{
                background-color: {theme.rgba(theme.INK_RGB, 10)};
            }}
            QWidget#accordion_header QLabel {{
                background: transparent;
                border: none;
            }}
        """)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(8)

        # Arrow label
        self.arrow = QLabel("▼")
        self.arrow.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.arrow.setAutoFillBackground(False)
        self.arrow.setStyleSheet(
            f"color: {theme.INK_FAINT}; font-size: 10px; font-weight: bold;"
            " background: transparent; border: none;"
        )
        header_layout.addWidget(self.arrow)

        # Title label
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.title_lbl.setAutoFillBackground(False)
        self.title_lbl.setStyleSheet(f"""
            color: {theme.INK};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            background: transparent;
            border: none;
        """)
        header_layout.addWidget(self.title_lbl, 1)

        # Enable clicking on header (left toggles; right-click is section order)
        self.header.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.header.mousePressEvent = self._on_header_pressed

        main_layout.addWidget(self.header)

        # Content container — opaque Surface so macOS light-mode scroll
        # viewports cannot wash out cream INK labels (white-on-white).
        self.content = QWidget()
        self.content.setObjectName("accordion_content")
        self.content.setAutoFillBackground(True)
        self.content.setStyleSheet(
            f"""
            QWidget#accordion_content {{
                background-color: {theme.SURFACE};
            }}
            """
        )
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 8, 12, 8)
        self.content_layout.setSpacing(10)
        main_layout.addWidget(self.content)

        self.content.setVisible(self._expanded)
        self.arrow.setText("▼" if self._expanded else "▶")

    def _load_expanded_default(self) -> bool:
        if not self._settings_key:
            return True
        return bool(
            QSettings("RAWviewer", "RAWviewer").value(
                _SECTION_EXPANDED_SETTINGS_PREFIX + self._settings_key,
                True,
                type=bool,
            )
        )

    def _on_header_pressed(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_expanded(not self._expanded)
            event.accept()
            return
        QWidget.mousePressEvent(self.header, event)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.content.setVisible(self._expanded)
        self.arrow.setText("▼" if self._expanded else "▶")
        if self._settings_key:
            QSettings("RAWviewer", "RAWviewer").setValue(
                _SECTION_EXPANDED_SETTINGS_PREFIX + self._settings_key, self._expanded
            )

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def add_layout(self, layout: QHBoxLayout | QVBoxLayout) -> None:
        self.content_layout.addLayout(layout)


class AdjustSlider(QSlider):
    """
    Custom-painted slider: fill grows from a center reference point (0 for a
    bipolar -X..+X range) rather than from the left edge, plus a compact
    rectangular thumb -- the Lightroom-style visual language a stock
    QSlider/QSS combination can't express (QSS's sub-page/add-page can only
    fill from one edge, never an interior point). An optional background
    gradient hints at a slider's effect (warm/cool for Temperature, etc.).

    Hit-testing/dragging is untouched -- this only overrides paintEvent, and
    positions everything from the *same* QStyle rects
    (subControlRect/sliderPositionFromValue) Qt's own default mouse handling
    already uses internally, so the painted thumb and the actual click/drag
    target stay pixel-exact without reimplementing any mouse logic.
    """

    _TRACK_HEIGHT = 4
    _THUMB_W = 10
    _THUMB_H = 14

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setMinimumHeight(22)
        self.setMouseTracking(True)
        # Click-to-focus: while a slider has focus, Left/Right arrows nudge
        # its value by one step (the main window's image-navigation shortcuts
        # defer to a focused AdjustSlider -- see _nav_shortcut_defers_to_
        # slider in main.py). Clicking anywhere else returns focus, and with
        # it arrow-key navigation.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        # Hover-focus: pointing at a slider is enough for Left/Right (and the
        # wheel) to address it; moving the pointer off hands arrow keys back
        # to image navigation. Focus follows the mouse only over sliders --
        # nothing else in the app uses hover focus, so navigation is never
        # stolen without the pointer visibly resting on a control.
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._center_value: float | None = None  # None -> auto (0 if bipolar, else minimum)
        self._track_gradient: list[tuple[float, str]] | None = None
        self._accent = QColor(theme.EMBER)

    def enterEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self.hasFocus() and not self.isSliderDown():
            self.clearFocus()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        # One minimum step per wheel notch: Qt's default is
        # singleStep * wheelScrollLines (3x) which overshoots fine-grained
        # sliders like Straighten's 0.1-degree steps.
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0:
            event.ignore()
            return
        self.setValue(self.value() + (self.singleStep() if delta > 0 else -self.singleStep()))
        # Wheel adjustments never pass through sliderReleased, so persist the
        # same way a click-release does.
        self.sliderReleased.emit()
        event.accept()

    def set_center_value(self, value: float | None) -> None:
        """Override the fill's zero/reference point (e.g. as-shot Temperature
        instead of a literal 0, which isn't in-range for an absolute-Kelvin
        slider). None restores the automatic bipolar/left-edge default."""
        self._center_value = value
        self.update()

    def set_track_gradient(self, stops: list[tuple[float, str]] | None) -> None:
        """stops: [(0.0, '#4080ff'), (0.5, '#3A332A'), (1.0, '#ffb040')] --
        a dim always-visible hint of what direction does what, independent
        of the solid accent fill drawn on top for the actual value."""
        self._track_gradient = stops
        self.update()

    def _effective_center(self) -> float:
        if self._center_value is not None:
            return max(float(self.minimum()), min(float(self.maximum()), self._center_value))
        if self.minimum() < 0 < self.maximum():
            return 0.0
        return float(self.minimum())

    def _handle_center_x(self, value: float, groove: QRectF, handle_w: float, upside_down: bool) -> float:
        span = max(1, int(round(groove.width() - handle_w)))
        pos = QStyle.sliderPositionFromValue(
            self.minimum(), self.maximum(), int(round(value)), span, upside_down
        )
        return groove.x() + pos + handle_w / 2.0

    def paintEvent(self, _event) -> None:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self
        )
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        track_y = groove.center().y() - self._TRACK_HEIGHT / 2.0
        # Inset the painted track to the thumb's actual travel range
        # [groove.x + handle_w/2, groove.right - handle_w/2]: Qt's value
        # mapping reserves half a *style* handle width (much wider than the
        # 10px painted thumb on macOS) at each end, so a full-groove track
        # left dead-looking zones the thumb could never reach -- reported as
        # "cannot slide to the edge of the slider".
        half_handle = handle.width() / 2.0
        track_rect = QRectF(
            groove.x() + half_handle,
            track_y,
            max(1.0, groove.width() - handle.width()),
            self._TRACK_HEIGHT,
        )

        painter.setPen(Qt.PenStyle.NoPen)
        if self._track_gradient:
            grad = QLinearGradient(track_rect.left(), 0, track_rect.right(), 0)
            for pos, color_str in self._track_gradient:
                grad.setColorAt(pos, QColor(color_str))
            painter.setBrush(grad)
        else:
            painter.setBrush(QColor(theme.LINE))
        painter.drawRoundedRect(track_rect, 2, 2)

        value_x = self._handle_center_x(self.value(), QRectF(groove), handle.width(), opt.upsideDown)
        center_x = self._handle_center_x(
            self._effective_center(), QRectF(groove), handle.width(), opt.upsideDown
        )
        fill_left, fill_right = (value_x, center_x) if value_x < center_x else (center_x, value_x)
        if fill_right - fill_left > 0.5:
            fill_rect = QRectF(fill_left, track_y, fill_right - fill_left, self._TRACK_HEIGHT)
            painter.setBrush(self._accent)
            painter.drawRoundedRect(fill_rect, 2, 2)

        thumb_rect = QRectF(
            value_x - self._THUMB_W / 2.0,
            groove.center().y() - self._THUMB_H / 2.0,
            self._THUMB_W,
            self._THUMB_H,
        )
        painter.setPen(QPen(QColor(*theme.LINE_RGB), 1))
        painter.setBrush(self._accent if self.underMouse() else QColor(theme.INK))
        painter.drawRoundedRect(thumb_rect, 1.5, 1.5)
        painter.end()
