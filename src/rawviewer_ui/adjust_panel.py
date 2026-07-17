"""Draggable RAW adjustment panel for single-image view (E to toggle)."""

from __future__ import annotations

import os
from typing import Callable, Dict

from PyQt6.QtCore import QRectF, Qt, QSettings, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QLinearGradient,
    QPainter,
    QPalette,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QStyle,
    QStyleOptionSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import theme
from raw_adjustments import (
    AS_SHOT_TEMP_KEY,
    CHROMA_NR_ON_VALUE,
    DEFAULT_ADJUSTMENTS,
    RECOVERY_BASELINE_KEY,
    SLIDER_SPECS,
    WB_PRESETS,
    is_pv2012_tone_slider,
    recovery_baseline_slider_hints,
)
from raw_dodge_burn import DEFAULT_STRENGTH as DB_DEFAULT_STRENGTH
from raw_dodge_burn import STRENGTH_KEY as DB_STRENGTH_KEY
from raw_hsl import HSL_COLOR_NAMES
from raw_tone_curve import (
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
    {"CropAngle", "PerspectiveVertical", "PerspectiveHorizontal"}
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
    from rawviewer_ui.tone_curve_widget import ToneCurveEditorRow, ToneCurveWidget

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


class _LooksRowWidget(QWidget):
    """One Looks list row: name + type badge + remove (×)."""

    remove_clicked = pyqtSignal()

    def __init__(self, name: str, kind: str, parent=None):
        super().__init__(parent)
        self._suppress_item_click = False
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 2, 2)
        row.setSpacing(6)

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
        
        # Enable clicking on header
        self.header.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
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


class ImageAdjustPanelWidget(QWidget):
    """Floating adjustment card; drag anywhere except sliders / Reset.

  Slider tracking (UI vs preview):
  - ``valueChanged`` (tracking on): numeric readout updates immediately; preview is throttled.
  - Throttled ``preview_changed`` while dragging (image work runs off the GUI thread).
  - ``sliderReleased``: final preview + ``editing_finished`` (XMP write).
    """

    # Wide enough for label + slider + value (+ AUTO / picker) without
    # clipping or forcing horizontal scroll on typical layouts.
    _PANEL_W = 440
    _PANEL_H = 520
    _PREVIEW_THROTTLE_MS = 80

    editing_finished = pyqtSignal(dict)
    preview_changed = pyqtSignal(dict)
    reset_requested = pyqtSignal()
    export_requested = pyqtSignal(str, dict)  # format id, adjustments
    recovery_baseline_requested = pyqtSignal()
    wb_picker_toggled = pyqtSignal(bool)  # True: arm the WB dropper; False: cancel
    compare_toggled = pyqtSignal(bool)  # True: show compare-with-original split view
    # True/False: re-decode the edit base with/without lens-profile correction
    # baked in (see main.py._on_adjust_lens_correction_toggled) -- unlike other
    # toggles, this needs a full re-decode, not just a preview-pipeline rerun.
    lens_correction_toggled = pyqtSignal(bool)
    # "dodge" / "burn" / None (disarmed) -- see main.py._on_dodge_burn_mode_changed.
    # True while the user is interacting with a Transform slider (straighten/
    # perspective); the host shows an alignment grid overlay for the duration.
    transform_interaction = pyqtSignal(bool)
    # One-shot auto adjustments (see raw_auto_adjust.py); the host computes
    # from the edit base and writes the result back through the sliders.
    auto_wb_requested = pyqtSignal()
    auto_straighten_requested = pyqtSignal()
    dodge_burn_mode_changed = pyqtSignal(object)
    dodge_burn_clear_requested = pyqtSignal()
    spot_heal_clear_requested = pyqtSignal()
    dodgeBurnMaskToggled = pyqtSignal(bool)
    dodge_burn_brush_changed = pyqtSignal()  # size/flow changed — refresh brush cursor
    # Fired when an XMP preset is applied so the host can drop per-image local
    # state (dodge/burn mask) that should not ride along with a global look.
    xmp_preset_applied = pyqtSignal()
    # User dropped files that are not .cube / .xmp onto the Looks panel.
    looks_drop_rejected = pyqtSignal(str)
    # Crop overlay (Transform): enter/exit mode, aspect lock, apply/cancel/reset.
    crop_mode_changed = pyqtSignal(bool)
    crop_aspect_changed = pyqtSignal(object)  # float|None
    crop_apply_requested = pyqtSignal()
    crop_cancel_requested = pyqtSignal()
    crop_reset_requested = pyqtSignal()

    def __init__(self, parent=None, histogram_widget=None):
        super().__init__(parent)
        self._sliders: Dict[str, QSlider] = {}
        self._value_labels: Dict[str, QLabel] = {}
        self._block_emit = False
        self._as_shot_temperature = float(DEFAULT_ADJUSTMENTS["Temperature"])
        self._recovery_baseline = False
        self._current_hsl_color = HSL_COLOR_NAMES[0]
        self._hsl_cache: Dict[str, float] = {
            k: float(v)
            for k, v in DEFAULT_ADJUSTMENTS.items()
            if k.startswith("HueAdjustment")
            or k.startswith("SaturationAdjustment")
            or k.startswith("LuminanceAdjustment")
        }
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._emit_live_preview)

        self.setMinimumWidth(self._PANEL_W)
        self.setMaximumWidth(560)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Inherit darkroom palette so any child without an explicit color
        # still gets cream ink on Surface — never system light WindowText.
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(*theme.SURFACE_RGB))
        pal.setColor(QPalette.ColorRole.Base, QColor(*theme.SURFACE_RGB))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(*theme.RAISED_RGB))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(*theme.INK_RGB))
        pal.setColor(QPalette.ColorRole.Text, QColor(*theme.INK_RGB))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(*theme.INK_RGB))
        pal.setColor(QPalette.ColorRole.BrightText, QColor(*theme.INK_RGB))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        card = QWidget(self)
        card.setObjectName("adjust_panel_card")
        card.setStyleSheet(
            f"""
            QWidget#adjust_panel_card {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.rgba(theme.INK_RGB, 45)};
                border-radius: 8px;
            }}
            QLabel#adjust_panel_title {{
                color: {theme.INK};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel[class="adjust_slider_label"] {{
                color: {theme.INK};
                font-size: 11px;
            }}
            QLabel[class="adjust_slider_value"] {{
                color: {theme.EMBER};
                font-size: 11px;
                min-width: 48px;
            }}
            QLabel[class="adjust_slider_value"]:hover {{
                color: {theme.INK};
            }}
            QPushButton#adjust_reset_btn {{
                color: {theme.INK};
                font-size: 11px;
                border: none;
                background: transparent;
                padding: 2px 6px;
            }}
            QPushButton#adjust_reset_btn:hover {{
                color: {theme.EMBER};
            }}
            QPushButton#adjust_export_btn {{
                color: {theme.INK};
                font-size: 13px;
                font-weight: 600;
                border: 1px solid {theme.rgba(theme.EMBER_RGB, 120)};
                border-radius: 6px;
                background: {theme.rgba(theme.EMBER_RGB, 70)};
                padding: 10px 14px;
                min-height: 22px;
            }}
            /* setMenu() draws a native caret; on dark chrome it reads as a
               stray white speck on the right. Hide it — whole button still opens the menu. */
            QPushButton#adjust_export_btn::menu-indicator {{
                image: none;
                width: 0px;
                height: 0px;
                padding: 0px;
                border: none;
            }}
            QPushButton#adjust_export_btn:hover {{
                background: {theme.rgba(theme.EMBER_RGB, 110)};
                color: {theme.INK};
                border-color: {theme.EMBER};
            }}
            QPushButton#adjust_export_btn:disabled {{
                color: {theme.INK_FAINT};
                border-color: {theme.rgba(theme.INK_RGB, 20)};
                background: transparent;
                font-weight: 400;
            }}
            QPushButton#adjust_nr_btn,
            QPushButton#adjust_db_btn,
            QPushButton#adjust_db_clear_btn,
            QPushButton#adjust_db_show_mask_btn {{
                color: {theme.INK};
                font-size: 11px;
                border: 1px solid {theme.rgba(theme.INK_RGB, 35)};
                border-radius: 4px;
                background: {theme.rgba(theme.INK_RGB, 12)};
                padding: 4px 8px;
            }}
            QPushButton#adjust_nr_btn:checked,
            QPushButton#adjust_db_btn:checked,
            QPushButton#adjust_db_show_mask_btn:checked {{
                color: {theme.EMBER};
                border-color: {theme.rgba(theme.EMBER_RGB, 90)};
                background: {theme.rgba(theme.EMBER_RGB, 30)};
            }}
            QPushButton#adjust_nr_btn:hover,
            QPushButton#adjust_db_btn:hover,
            QPushButton#adjust_db_clear_btn:hover,
            QPushButton#adjust_db_show_mask_btn:hover {{
                color: {theme.INK};
                background: {theme.rgba(theme.INK_RGB, 24)};
            }}
            QPushButton#adjust_db_clear_btn:disabled,
            QPushButton#adjust_db_show_mask_btn:disabled,
            QPushButton#adjust_db_btn:disabled {{
                color: {theme.INK_FAINT};
                border-color: {theme.rgba(theme.INK_RGB, 20)};
                background: transparent;
            }}
            QPushButton#adjust_wb_picker_btn, QPushButton#adjust_auto_wb_btn {{
                border: 1px solid {theme.rgba(theme.INK_RGB, 35)};
                border-radius: 4px;
                background: {theme.rgba(theme.INK_RGB, 12)};
                padding: 0px;
            }}
            QPushButton#adjust_auto_wb_btn {{
                padding: 0px 4px;
                color: {theme.INK};
                font-size: 10px;
                font-weight: 600;
            }}
            QPushButton#adjust_wb_picker_btn:checked, QPushButton#adjust_auto_wb_btn:checked {{
                border-color: {theme.rgba(theme.EMBER_RGB, 90)};
                background: {theme.rgba(theme.EMBER_RGB, 30)};
                color: {theme.EMBER};
            }}
            QPushButton#adjust_wb_picker_btn:hover, QPushButton#adjust_auto_wb_btn:hover {{
                background: {theme.rgba(theme.INK_RGB, 24)};
            }}
            QPushButton#adjust_compare_btn {{
                border: 1px solid {theme.rgba(theme.INK_RGB, 35)};
                border-radius: 4px;
                background: {theme.rgba(theme.INK_RGB, 12)};
                padding: 0px;
            }}
            QPushButton#adjust_compare_btn:checked {{
                border-color: {theme.rgba(theme.EMBER_RGB, 90)};
                background: {theme.rgba(theme.EMBER_RGB, 30)};
            }}
            QPushButton#adjust_compare_btn:hover {{
                background: {theme.rgba(theme.INK_RGB, 24)};
            }}
            """
        )
        # AdjustSlider paints its own groove/fill/handle (paintEvent override,
        # not QSS) so its center-out fill can start from an interior point --
        # QSS's sub-page/add-page can only fill from one edge. See its
        # docstring / docs/EDIT_PIPELINE.md "Slider visual language".
        outer.addWidget(card)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        scroll = QScrollArea(card)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # macOS Qt leaves the scroll viewport on the system light Base color
        # when only the QScrollArea itself is styled transparent — cream INK
        # labels then vanish. Paint Surface all the way through.
        scroll.setAutoFillBackground(True)
        scroll.setStyleSheet(
            f"""
            QScrollArea {{
                background-color: {theme.SURFACE};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {theme.SURFACE};
                width: 10px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {theme.LINE};
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            """
        )
        vp = scroll.viewport()
        vp.setAutoFillBackground(True)
        vp.setStyleSheet(f"background-color: {theme.SURFACE};")
        card_layout.addWidget(scroll)

        inner = QWidget()
        inner.setObjectName("adjust_panel_inner")
        inner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        inner.setAutoFillBackground(True)
        inner.setStyleSheet(
            f"""
            QWidget#adjust_panel_inner {{
                background-color: {theme.SURFACE};
            }}
            """
        )
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Adjust")
        title.setObjectName("adjust_panel_title")
        header.addWidget(title)
        header.addStretch(1)
        self._build_compare_button(header)
        copy_btn = QPushButton("Copy")
        copy_btn.setObjectName("adjust_reset_btn")  # share Reset's compact style
        copy_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        copy_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        copy_btn.setToolTip("Copy this image's edit settings (session clipboard)")
        copy_btn.clicked.connect(self._on_copy_settings_clicked)
        header.addWidget(copy_btn)
        self._paste_btn = QPushButton("Paste")
        self._paste_btn.setObjectName("adjust_reset_btn")
        self._paste_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._paste_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._paste_btn.setToolTip("Apply the copied edit settings to this image")
        self._paste_btn.setEnabled(_edit_settings_clipboard() is not None)
        self._paste_btn.clicked.connect(self._on_paste_settings_clicked)
        header.addWidget(self._paste_btn)
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("adjust_reset_btn")
        reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        reset_btn.clicked.connect(self._on_reset_clicked)
        header.addWidget(reset_btn)
        layout.addLayout(header)

        hint = QLabel(
            "E / Esc — close · D/B/X/H brush tools · two-finger scroll = Brush Size"
        )
        hint.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")
        hint.setWordWrap(True)
        hint.setToolTip(
            "While Adjust is open:\n"
            "• E or Esc closes the editor (restores browse RAW/JPEG mode)\n"
            "• D / B / X / H arm Dodge / Burn / Eraser / Heal (press again to disarm; also disarms when the cursor leaves the image)\n"
            "• With a brush tool armed, two-finger scroll changes Brush Size "
            "(Ctrl+scroll still zooms)\n"
            "• Brush Flow changes how opaque the brush preview looks"
        )
        layout.addWidget(hint)

        self._tone_curve_row = None

        # Build Collapsible Sections
        self.histogram_widget = histogram_widget
        self.sect_histogram = CollapsibleSection("Histogram", settings_key="histogram")
        if self.histogram_widget:
            # Add some styling wrapper or just add it directly
            # Set minimum height for the histogram
            self.histogram_widget.setMinimumHeight(120)
            self.sect_histogram.content_layout.addWidget(self.histogram_widget)
        else:
            self.sect_histogram.hide()

        self.sect_light = CollapsibleSection("Light", settings_key="light")
        self.sect_color = CollapsibleSection("Color / WB", settings_key="color")
        self.sect_transform = CollapsibleSection("Transform", settings_key="transform")

        self.sect_curve = CollapsibleSection("Tone Curve", settings_key="curve")
        if not _SHOW_TONE_CURVE_UI:
            self.sect_curve.hide()

        self.sect_hsl = CollapsibleSection("HSL / Color Mixer", settings_key="hsl")
        if not _SHOW_HSL_UI:
            self.sect_hsl.hide()

        self.sect_detail = CollapsibleSection("Detail", settings_key="detail")
        self.sect_noise = CollapsibleSection("Noise Reduction", settings_key="noise")
        self.sect_effects = CollapsibleSection("Effects", settings_key="effects")
        self.sect_local = CollapsibleSection("Local", settings_key="local")
        if not _SHOW_DODGE_BURN_UI:
            self.sect_local.hide()
        self.sect_lut = CollapsibleSection("Looks (.cube / .xmp)", settings_key="lut")

        # Add Collapsible Sections to main scroll layout
        layout.addWidget(self.sect_histogram)
        layout.addWidget(self.sect_light)
        layout.addWidget(self.sect_color)
        layout.addWidget(self.sect_curve)
        layout.addWidget(self.sect_hsl)
        layout.addWidget(self.sect_detail)
        layout.addWidget(self.sect_noise)
        layout.addWidget(self.sect_effects)
        layout.addWidget(self.sect_local)
        layout.addWidget(self.sect_lut)
        layout.addWidget(self.sect_transform)

        # Build tone curve editor row inside the curve section first
        if _SHOW_TONE_CURVE_UI:
            from PyQt6.QtWidgets import QStackedWidget
            
            self._tone_curve_tabs = QWidget()
            tabs_layout = QVBoxLayout(self._tone_curve_tabs)
            tabs_layout.setContentsMargins(0, 0, 0, 0)
            
            # Segmented control
            seg_layout = QHBoxLayout()
            seg_layout.setSpacing(0)
            
            self._btn_point = QPushButton("Point")
            self._btn_param = QPushButton("Parametric")
            
            seg_style = """
                QPushButton {
                    background: #272219;
                    color: #96897A;
                    border: 1px solid #404040;
                    padding: 6px 12px;
                    font-size: 11px;
                }
                QPushButton:checked {
                    background: #404040;
                    color: #FFFFFF;
                }
                QPushButton#btn_point {
                    border-top-left-radius: 4px;
                    border-bottom-left-radius: 4px;
                    border-right: none;
                }
                QPushButton#btn_param {
                    border-top-right-radius: 4px;
                    border-bottom-right-radius: 4px;
                }
            """
            
            self._btn_point.setObjectName("btn_point")
            self._btn_param.setObjectName("btn_param")
            self._btn_point.setCheckable(True)
            self._btn_param.setCheckable(True)
            self._btn_point.setChecked(True)
            self._btn_point.setStyleSheet(seg_style)
            self._btn_param.setStyleSheet(seg_style)
            self._btn_point.setCursor(Qt.CursorShape.PointingHandCursor)
            self._btn_param.setCursor(Qt.CursorShape.PointingHandCursor)
            
            seg_layout.addWidget(self._btn_point)
            seg_layout.addWidget(self._btn_param)
            tabs_layout.addLayout(seg_layout)
            
            self._tone_curve_stack = QStackedWidget()
            
            self._tone_curve_point_tab = QWidget()
            self._tone_curve_point_layout = QVBoxLayout(self._tone_curve_point_tab)
            self._tone_curve_point_layout.setContentsMargins(0, 8, 0, 0)
            
            self._tone_curve_param_tab = QWidget()
            self._tone_curve_param_layout = QVBoxLayout(self._tone_curve_param_tab)
            self._tone_curve_param_layout.setContentsMargins(0, 8, 0, 0)
            
            self._tone_curve_stack.addWidget(self._tone_curve_point_tab)
            self._tone_curve_stack.addWidget(self._tone_curve_param_tab)
            tabs_layout.addWidget(self._tone_curve_stack)
            
            self._btn_point.clicked.connect(lambda: (self._btn_point.setChecked(True), self._btn_param.setChecked(False), self._tone_curve_stack.setCurrentIndex(0)))
            self._btn_param.clicked.connect(lambda: (self._btn_param.setChecked(True), self._btn_point.setChecked(False), self._tone_curve_stack.setCurrentIndex(1)))
            
            self.sect_curve.add_widget(self._tone_curve_tabs)

            # Standard-mode R/G/B channel selector (Lightroom/RawTherapee
            # "Standard" style: each channel independently remapped, only
            # meaningful for the Point curve, not the Parametric sliders).
            self._channel_curve_cache: Dict[str, str] = {
                "RGB": "",
                "R": "",
                "G": "",
                "B": "",
            }
            self._current_curve_channel = "RGB"
            self._channel_btns: Dict[str, QPushButton] = {}

            channel_layout = QHBoxLayout()
            channel_layout.setSpacing(0)
            channel_style = """
                QPushButton {
                    background: #272219;
                    color: #96897A;
                    border: 1px solid #404040;
                    padding: 4px 10px;
                    font-size: 10px;
                }
                QPushButton:checked {
                    background: #404040;
                    color: #FFFFFF;
                }
                QPushButton#chan_RGB {
                    border-top-left-radius: 4px;
                    border-bottom-left-radius: 4px;
                }
                QPushButton#chan_B {
                    border-top-right-radius: 4px;
                    border-bottom-right-radius: 4px;
                    border-left: none;
                }
                QPushButton#chan_R, QPushButton#chan_G {
                    border-left: none;
                }
            """
            for name in ("RGB", "R", "G", "B"):
                btn = QPushButton(name)
                btn.setObjectName(f"chan_{name}")
                btn.setCheckable(True)
                btn.setChecked(name == "RGB")
                btn.setStyleSheet(channel_style)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(lambda checked, n=name: self._on_curve_channel_selected(n))
                channel_layout.addWidget(btn)
                self._channel_btns[name] = btn
            self._tone_curve_point_layout.addLayout(channel_layout)

            self._tone_curve_row = ToneCurveEditorRow()
            self._tone_curve_row.points_changed.connect(self._on_tone_curve_changed)
            self._tone_curve_row.editing_finished.connect(self._on_tone_curve_finished)
            self._tone_curve_point_layout.addWidget(self._tone_curve_row)

        # Build HSL mixer inside the HSL section
        if _SHOW_HSL_UI:
            self._build_hsl_section(self.sect_hsl.content_layout)

        # WB situation presets (Daylight/Cloudy/…) sit above Temp/Tint.
        self._build_wb_preset_row(self.sect_color)

        # Loop through sliders and assign to sections
        for spec in SLIDER_SPECS:
            if not _SHOW_TONE_CURVE_UI and spec.key in _PARAMETRIC_TONE_KEYS:
                continue
                
            target_layout = None
            if spec.key in {"Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012"}:
                target_sect = self.sect_light
            elif spec.key in {"Temperature", "Tint", "Saturation", "Vibrance"}:
                target_sect = self.sect_color
            elif spec.key in {"ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"}:
                if _SHOW_TONE_CURVE_UI:
                    target_sect = None
                    target_layout = self._tone_curve_param_layout
                else:
                    target_sect = self.sect_curve
            elif spec.key in {"Sharpness", "Clarity2012", "Defringe"}:
                target_sect = self.sect_detail
            elif spec.key in {"LuminanceNoiseReduction"}:
                target_sect = self.sect_noise
            elif spec.key in {
                "CropAngle", "PerspectiveVertical", "PerspectiveHorizontal",
            }:
                target_sect = self.sect_transform
            elif spec.key in {
                "PostCropVignetteAmount",
                "PostCropVignetteMidpoint",
                "Dehaze",
            }:
                target_sect = self.sect_effects
            else:
                target_sect = None
                
            if target_sect is None and target_layout is None:
                continue

            row = QHBoxLayout()
            row.setSpacing(6)
            name_lbl = QLabel(spec.label)
            name_lbl.setProperty("class", "adjust_slider_label")
            name_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
            name_lbl.setFixedWidth(82)
            name_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            row.addWidget(name_lbl)

            slider = AdjustSlider(Qt.Orientation.Horizontal)
            slider.setTracking(True)
            slider.setRange(spec.minimum, spec.maximum)
            slider.setSingleStep(spec.single_step)
            slider.setPageStep(max(spec.single_step, (spec.maximum - spec.minimum) // 20))
            # ClickFocus (from AdjustSlider.__init__) stays: a clicked slider
            # owns Left/Right for value nudging until focus moves elsewhere.
            gradient = _SLIDER_TRACK_GRADIENTS.get(spec.key)
            if gradient is not None:
                slider.set_track_gradient(gradient)
            if spec.key == "Temperature":
                slider.set_center_value(self._as_shot_temperature)
                self._temperature_slider = slider
            elif spec.key == "PostCropVignetteMidpoint":
                # Default Midpoint is 50 (LR); fill from the middle like Amount.
                slider.set_center_value(50.0)
            slider.sliderMoved.connect(
                lambda _v, k=spec.key, fmt=spec.format_value: self._update_slider_label(
                    k, fmt
                )
            )
            slider.valueChanged.connect(
                lambda _v, k=spec.key, fmt=spec.format_value: self._on_slider_value_changed(
                    k, fmt
                )
            )
            slider.sliderReleased.connect(self._on_slider_released)
            if spec.key in _TRANSFORM_SLIDER_KEYS:
                # Alignment grid while straightening / correcting perspective:
                # pressed -> grid on; released -> grid off (host-side).
                slider.sliderPressed.connect(
                    lambda: self.transform_interaction.emit(True)
                )
                slider.sliderReleased.connect(
                    lambda: self.transform_interaction.emit(False)
                )
            row.addWidget(slider, 1)

            if spec.key == "CropAngle":
                auto_st = QPushButton("AUTO")
                auto_st.setObjectName("adjust_auto_wb_btn")
                auto_st.setFixedHeight(20)
                auto_st.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                auto_st.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                auto_st.setToolTip(
                    "Auto straighten — detect dominant near-horizontal/vertical lines"
                )
                auto_st.clicked.connect(lambda: self.auto_straighten_requested.emit())
                row.addWidget(auto_st)

            val_lbl = AdjustValueLabel()
            val_lbl.setProperty("class", "adjust_slider_value")
            val_lbl.setStyleSheet(f"color: {theme.EMBER}; font-size: 11px;")
            val_lbl.setFixedWidth(52)
            val_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setToolTip("Click to reset")
            val_lbl.clicked.connect(lambda k=spec.key: self._reset_slider(k))
            row.addWidget(val_lbl)

            if spec.key == "Temperature":
                self._build_wb_picker_button(row)

            if target_layout is not None:
                target_layout.addLayout(row)
            elif target_sect is not None:
                target_sect.add_layout(row)
            self._sliders[spec.key] = slider
            self._value_labels[spec.key] = val_lbl

        self._build_crop_controls(self.sect_transform)

        # Chroma NR
        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_lbl = QLabel("Chroma NR")
        method_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        method_lbl.setMinimumWidth(78)
        method_row.addWidget(method_lbl)
        
        self._denoise_method_combo = QComboBox()
        self._denoise_method_combo.addItems(["Off", "Bilateral filter", "Guided filter"])
        self._denoise_method_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._denoise_method_combo.currentIndexChanged.connect(self._on_denoise_method_changed)
        self._denoise_method_combo.setToolTip("Chroma-only denoise (luminance preserved)")
        self._denoise_method_combo.setStyleSheet(_adjust_combo_stylesheet())
        method_row.addWidget(self._denoise_method_combo, 1)
        self.sect_noise.add_layout(method_row)

        chroma_amt_row = QHBoxLayout()
        chroma_amt_row.setSpacing(6)
        chroma_amt_lbl = QLabel("Chroma Amount")
        chroma_amt_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        chroma_amt_lbl.setMinimumWidth(78)
        chroma_amt_row.addWidget(chroma_amt_lbl)
        self._chroma_nr_amount_slider = AdjustSlider(Qt.Orientation.Horizontal)
        self._chroma_nr_amount_slider.setRange(1, 100)
        self._chroma_nr_amount_slider.setValue(int(CHROMA_NR_ON_VALUE))
        self._chroma_nr_amount_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._chroma_nr_amount_slider.setToolTip(
            "Chroma denoise strength (0–100). Only applies when a method is selected."
        )
        self._chroma_nr_amount_value = AdjustValueLabel()
        self._chroma_nr_amount_value.setProperty("class", "adjust_slider_value")
        self._chroma_nr_amount_value.setStyleSheet(f"color: {theme.EMBER}; font-size: 11px;")
        self._chroma_nr_amount_value.setFixedWidth(52)
        self._chroma_nr_amount_value.setText(str(int(CHROMA_NR_ON_VALUE)))
        self._chroma_nr_amount_slider.valueChanged.connect(self._on_chroma_nr_amount_changed)
        self._chroma_nr_amount_slider.sliderReleased.connect(self._on_slider_released)
        chroma_amt_row.addWidget(self._chroma_nr_amount_slider, 1)
        chroma_amt_row.addWidget(self._chroma_nr_amount_value)
        self._chroma_nr_amount_row = QWidget()
        self._chroma_nr_amount_row.setLayout(chroma_amt_row)
        self._chroma_nr_amount_row.setVisible(False)
        self.sect_noise.add_widget(self._chroma_nr_amount_row)

        # Lens correction (optics — lives with Transform / crop geometry)
        self._lens_correction_row = QHBoxLayout()
        self._lens_correction_row.setSpacing(6)
        
        lbl_vbox = QVBoxLayout()
        lbl_vbox.setSpacing(1)
        lbl_vbox.setContentsMargins(0, 0, 0, 0)
        
        lens_label = QLabel("Lens correction")
        lens_label.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        lbl_vbox.addWidget(lens_label)
        
        self._lens_profile_lbl = QLabel("")
        self._lens_profile_lbl.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")
        self._lens_profile_lbl.setWordWrap(True)
        lbl_vbox.addWidget(self._lens_profile_lbl)
        
        lbl_container = QWidget()
        lbl_container.setLayout(lbl_vbox)
        lbl_container.setMinimumWidth(78)
        
        self._lens_correction_row.addWidget(lbl_container)
        self._lens_correction_btn = QPushButton("Off")
        self._lens_correction_btn.setObjectName("adjust_nr_btn")
        self._lens_correction_btn.setCheckable(True)
        self._lens_correction_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lens_correction_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._lens_correction_btn.setToolTip(
            "Automatic distortion correction from a matched lens profile"
        )
        self._lens_correction_btn.toggled.connect(self._on_lens_correction_toggled)
        self._lens_correction_row.addWidget(self._lens_correction_btn, 1)
        self._lens_correction_row_widget = QWidget()
        self._lens_correction_row_widget.setLayout(self._lens_correction_row)
        self._lens_correction_row_widget.hide()  # shown only once a profile match is confirmed
        self.sect_transform.add_widget(self._lens_correction_row_widget)

        # Recovery-look button removed (2026-07): the editor's default
        # display transform now matches the browse render exactly (dcraw
        # BT.709 + highlight clip, see raw_edit_pipeline), which eliminated
        # the tonal gap the recovery baseline existed to bridge. The
        # pipeline still honors the flag for sessions/dicts that carry it
        # (set_recovery_baseline stays as a no-op-capable setter); nothing
        # in the UI can arm it anymore.
        self._recovery_btn = None

        # Dodge & burn brush (own Local accordion): mutually-exclusive
        # Dodge/Burn + Size/Flow. Mask persists via XMP; live strokes use a
        # cheap region patch (main._on_dodge_burn_stroke) so unrelated sliders
        # don't re-pay brush cost (gain map cached in raw_dodge_burn).
        db_container = QWidget()
        db_container_layout = QVBoxLayout(db_container)
        db_container_layout.setContentsMargins(0, 0, 0, 0)
        db_container_layout.setSpacing(6)

        db_row = QHBoxLayout()
        db_row.setSpacing(6)
        db_mode_lbl = QLabel("Mode")
        db_mode_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        db_mode_lbl.setMinimumWidth(78)
        db_row.addWidget(db_mode_lbl)
        self._dodge_btn = QPushButton("Dodge (D)")
        self._burn_btn = QPushButton("Burn (B)")
        self._erase_btn = QPushButton("Eraser (X)")
        self._heal_btn = QPushButton("Heal (H)")
        for btn, tip in (
            (
                self._dodge_btn,
                "Dodge (D) — brush to brighten; soft falloff, edge-snaps on release.\n"
                "Two-finger scroll changes Brush Size; Ctrl+scroll zooms.",
            ),
            (
                self._burn_btn,
                "Burn (B) — brush to darken; soft falloff, edge-snaps on release.\n"
                "Two-finger scroll changes Brush Size; Ctrl+scroll zooms.",
            ),
            (
                self._erase_btn,
                "Eraser (X) — remove dodge/burn or heal paint under the brush.\n"
                "Two-finger scroll changes Brush Size; Ctrl+scroll zooms.",
            ),
        ):
            btn.setObjectName("adjust_db_btn")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setToolTip(tip)
            db_row.addWidget(btn, 1)
        self._dodge_btn.toggled.connect(lambda on: self._on_dodge_burn_toggled(self._dodge_btn, on))
        self._burn_btn.toggled.connect(lambda on: self._on_dodge_burn_toggled(self._burn_btn, on))
        self._erase_btn.toggled.connect(lambda on: self._on_dodge_burn_toggled(self._erase_btn, on))
        db_container_layout.addLayout(db_row)

        heal_row = QHBoxLayout()
        heal_row.setSpacing(6)
        heal_mode_lbl = QLabel("Heal")
        heal_mode_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        heal_mode_lbl.setMinimumWidth(78)
        heal_row.addWidget(heal_mode_lbl)
        self._heal_btn.setObjectName("adjust_db_btn")
        self._heal_btn.setCheckable(True)
        self._heal_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._heal_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._heal_btn.setToolTip(
            "Heal (H) — brush to remove smudges / dust (OpenCV inpaint at full strength).\n"
            "Only Brush Size and Brush Flow apply; Effect Strength is for Dodge/Burn only.\n"
            "Paint the defect; release to fill from neighbors.\n"
            "Two-finger scroll changes Brush Size; Ctrl+scroll zooms."
        )
        self._heal_btn.toggled.connect(lambda on: self._on_dodge_burn_toggled(self._heal_btn, on))
        heal_row.addWidget(self._heal_btn, 1)
        self._heal_clear_btn = QPushButton("Clear Heal")
        self._heal_clear_btn.setObjectName("adjust_db_clear_btn")
        self._heal_clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._heal_clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._heal_clear_btn.setToolTip("Erase the spot-heal mask for this image")
        self._heal_clear_btn.setEnabled(False)
        self._heal_clear_btn.clicked.connect(self.spot_heal_clear_requested.emit)
        heal_row.addWidget(self._heal_clear_btn, 1)
        db_container_layout.addLayout(heal_row)

        db_actions_row = QHBoxLayout()
        db_actions_row.setSpacing(6)
        db_actions_spacer = QLabel("")
        db_actions_spacer.setMinimumWidth(78)
        db_actions_row.addWidget(db_actions_spacer)
        self._db_clear_btn = QPushButton("Clear")
        self._db_clear_btn.setObjectName("adjust_db_clear_btn")
        self._db_clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._db_clear_btn.setToolTip("Erase the dodge/burn brush mask for this image")
        self._db_clear_btn.setEnabled(False)
        self._db_clear_btn.clicked.connect(self.dodge_burn_clear_requested.emit)
        db_actions_row.addWidget(self._db_clear_btn, 1)

        self._db_show_mask_btn = QPushButton("Mask")
        self._db_show_mask_btn.setObjectName("adjust_db_show_mask_btn")
        self._db_show_mask_btn.setCheckable(True)
        self._db_show_mask_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_show_mask_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._db_show_mask_btn.setEnabled(False)
        self._db_show_mask_btn.setToolTip(
            "Overlay the active brush mask (red/blue = dodge/burn, green = heal).\n"
            "Turns on automatically when a brush tool is armed. Shortcut: O"
        )
        self._db_show_mask_btn.toggled.connect(self.dodgeBurnMaskToggled.emit)
        db_actions_row.addWidget(self._db_show_mask_btn, 1)

        self._db_edge_btn = QPushButton("Edge Assist")
        self._db_edge_btn.setObjectName("adjust_db_show_mask_btn")
        self._db_edge_btn.setCheckable(True)
        self._db_edge_btn.setChecked(True)
        self._db_edge_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_edge_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._db_edge_btn.setToolTip(
            "Keep dodge/burn paint on the subject under the cursor — flood-fills "
            "within similar luminance so strokes stop at subject edges "
            "instead of spilling onto neighbors (not used by Heal)"
        )
        db_actions_row.addWidget(self._db_edge_btn, 1)
        db_container_layout.addLayout(db_actions_row)

        db_size_row = QHBoxLayout()
        db_size_row.setSpacing(6)
        db_size_lbl = QLabel("Brush Size")
        db_size_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        db_size_lbl.setMinimumWidth(78)
        db_size_row.addWidget(db_size_lbl)
        self._db_size_slider = AdjustSlider(Qt.Orientation.Horizontal)
        self._db_size_slider.setRange(8, 400)
        self._db_size_slider.setValue(80)
        self._db_size_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_size_slider.setToolTip(
            "Brush diameter in pixels.\n"
            "With Dodge/Burn armed: two-finger scroll up/down also changes size "
            "(does not navigate images). Ctrl+scroll still zooms."
        )
        self._db_size_slider.valueChanged.connect(
            lambda _v: self.dodge_burn_brush_changed.emit()
        )
        db_size_row.addWidget(self._db_size_slider, 1)
        db_container_layout.addLayout(db_size_row)

        db_strength_row = QHBoxLayout()
        db_strength_row.setSpacing(6)
        db_strength_lbl = QLabel("Brush Flow")
        db_strength_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        db_strength_lbl.setMinimumWidth(78)
        db_strength_row.addWidget(db_strength_lbl)
        self._db_strength_slider = AdjustSlider(Qt.Orientation.Horizontal)
        self._db_strength_slider.setRange(5, 100)
        # Default was 28 * 0.12 stamp scale ≈ too subtle after full settle;
        # raise flow so a short stroke remains clearly visible after apply.
        self._db_strength_slider.setValue(55)
        self._db_strength_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_strength_slider.setToolTip(
            "Per-stroke flow (low = build up gradually; high = stronger each pass).\n"
            "Also sets how opaque the on-image brush preview looks.\n"
            "In Heal mode this is the only intensity control (Effect Strength is off)."
        )
        self._db_strength_slider.valueChanged.connect(
            lambda _v: self.dodge_burn_brush_changed.emit()
        )
        db_strength_row.addWidget(self._db_strength_slider, 1)
        db_container_layout.addLayout(db_strength_row)

        # Stops-per-mask-unit (persisted as DodgeBurnStrength). Distinct from
        # Brush Flow, which only controls per-stroke accumulation while painting.
        # Heal mode disables this row — inpaint always runs at full strength.
        self._db_mask_strength_row = QWidget()
        db_mask_str_row = QHBoxLayout(self._db_mask_strength_row)
        db_mask_str_row.setContentsMargins(0, 0, 0, 0)
        db_mask_str_row.setSpacing(6)
        db_mask_str_lbl = QLabel("Effect Strength")
        db_mask_str_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        db_mask_str_lbl.setMinimumWidth(78)
        db_mask_str_row.addWidget(db_mask_str_lbl)
        self._db_mask_strength_slider = AdjustSlider(Qt.Orientation.Horizontal)
        # 0.50–3.00 stops in 0.05 steps → slider 10–60
        self._db_mask_strength_slider.setRange(10, 60)
        self._db_mask_strength_slider.setValue(int(round(DB_DEFAULT_STRENGTH * 20.0)))
        # While Heal is armed the slider is forced to max for display; this
        # holds the previous dodge/burn stops so leaving Heal restores them.
        self._db_mask_strength_saved_for_heal: int | None = None
        self._db_mask_strength_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_mask_strength_slider.setToolTip(
            "How many stops of exposure the painted dodge/burn mask applies at full strength.\n"
            "Not used by Heal (inpaint always runs at full strength)."
        )
        self._db_mask_strength_value = AdjustValueLabel()
        self._db_mask_strength_value.setProperty("class", "adjust_slider_value")
        self._db_mask_strength_value.setStyleSheet(f"color: {theme.EMBER}; font-size: 11px;")
        self._db_mask_strength_value.setFixedWidth(52)
        self._db_mask_strength_value.setText(f"{DB_DEFAULT_STRENGTH:.2f}")
        self._db_mask_strength_slider.valueChanged.connect(self._on_db_mask_strength_changed)
        self._db_mask_strength_slider.sliderReleased.connect(self._on_slider_released)
        db_mask_str_row.addWidget(self._db_mask_strength_slider, 1)
        db_mask_str_row.addWidget(self._db_mask_strength_value)
        db_container_layout.addWidget(self._db_mask_strength_row)
        self.sect_local.add_widget(db_container)

        self._build_looks_section(self.sect_lut)

        export_btn = QPushButton("Export")
        export_btn.setObjectName("adjust_export_btn")
        export_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        export_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        export_btn.setMinimumHeight(40)
        export_menu = QMenu(export_btn)
        # Bare QMenu renders with OS-native chrome, clashing with the app's
        # frameless dark-chrome dialogs (see release_update_dialog.py) -- the
        # only other QMenu-style popup in the app. Match those tokens.
        export_menu.setStyleSheet(
            f"""
            QMenu {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 8px;
                padding: 4px;
                color: {theme.INK};
                font-size: 12px;
            }}
            QMenu::item {{
                padding: 6px 20px 6px 12px;
                border-radius: 5px;
                background-color: transparent;
            }}
            QMenu::item:selected {{
                background-color: {theme.EMBER_DIM};
                color: {theme.INK};
            }}
            QMenu::item:disabled {{
                color: {theme.INK_FAINT};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {theme.LINE};
                margin: 4px 8px;
            }}
            """
        )
        export_menu.addAction(
            "16-bit TIFF (baked)",
            lambda: self._request_export("tiff16"),
        )
        export_menu.addAction(
            "JPEG (baked)",
            lambda: self._request_export("jpeg"),
        )
        export_menu.addAction(
            "WebP (baked)",
            lambda: self._request_export("webp"),
        )
        try:
            from raw_nn_denoise import nn_denoise_available

            if nn_denoise_available():
                export_menu.addSeparator()
                export_menu.addAction(
                    "JPEG + AI denoise (SCUNet)",
                    lambda: self._request_export("jpeg_nn"),
                )
                export_menu.addAction(
                    "16-bit TIFF + AI denoise (SCUNet)",
                    lambda: self._request_export("tiff16_nn"),
                )
        except Exception:
            pass
        export_btn.setMenu(export_menu)
        export_btn.setToolTip("Export baked image (TIFF / JPEG / WebP)")
        layout.addWidget(export_btn)
        self._export_btn = export_btn

        layout.addStretch(1)
        self.set_adjustments(dict(DEFAULT_ADJUSTMENTS))

    def _build_compare_button(self, row: QHBoxLayout) -> None:
        """Small icon-only toggle in the header: split-view compare with the original."""
        btn = QPushButton()
        btn.setObjectName("adjust_compare_btn")
        btn.setCheckable(True)
        btn.setFixedSize(22, 22)
        btn.setIconSize(QSize(13, 13))
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(
            "Compare with original — split view; drag the divider."
            " Click again to exit."
        )
        self._compare_icon_default = _qta_icon_safe("fa5s.columns", color=theme.INK_MUTED)
        self._compare_icon_active = _qta_icon_safe("fa5s.columns", color=theme.EMBER)
        btn.setIcon(self._compare_icon_default)
        btn.toggled.connect(self._on_compare_toggled)
        row.addWidget(btn)
        self._compare_btn = btn

    def _on_compare_toggled(self, checked: bool) -> None:
        self._sync_compare_icon(checked)
        self.compare_toggled.emit(bool(checked))

    def set_compare_active(self, active: bool) -> None:
        """Programmatically arm/disarm the compare toggle (e.g. on navigation)."""
        btn = getattr(self, "_compare_btn", None)
        if btn is None:
            return
        btn.blockSignals(True)
        try:
            btn.setChecked(bool(active))
        finally:
            btn.blockSignals(False)
        self._sync_compare_icon(active)

    def _sync_compare_icon(self, active: bool) -> None:
        btn = getattr(self, "_compare_btn", None)
        if btn is None:
            return
        icon = (
            getattr(self, "_compare_icon_active", None)
            if active
            else getattr(self, "_compare_icon_default", None)
        )
        if icon is not None:
            btn.setIcon(icon)

    def _build_wb_preset_row(self, sect: CollapsibleSection) -> None:
        """Dropdown of common illuminants + As Shot (EXIF Kelvin when available)."""
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel("WB")
        lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        lbl.setFixedWidth(82)
        row.addWidget(lbl)
        combo = QComboBox()
        combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        combo.setToolTip(
            "White-balance preset (Kelvin). As Shot uses the camera/EXIF "
            "color temperature when present; other entries are standard "
            "illuminant targets (not EXIF named modes)."
        )
        for name, kelvin in WB_PRESETS:
            if kelvin is None:
                combo.addItem(name, None)
            else:
                combo.addItem(f"{name} ({int(kelvin)}K)", float(kelvin))
        combo.addItem("Custom", -1.0)
        combo.setStyleSheet(_adjust_combo_stylesheet())
        combo.currentIndexChanged.connect(self._on_wb_preset_changed)
        row.addWidget(combo, 1)
        sect.add_layout(row)
        self._wb_preset_combo = combo
        self._wb_preset_block = False

    def _on_wb_preset_changed(self, _index: int) -> None:
        if getattr(self, "_wb_preset_block", False):
            return
        combo = getattr(self, "_wb_preset_combo", None)
        if combo is None:
            return
        data = combo.currentData()
        name = combo.currentText()
        if data == -1.0 or (isinstance(name, str) and name.startswith("Custom")):
            return
        if data is None:
            kelvin = float(
                getattr(self, "_as_shot_temperature", DEFAULT_ADJUSTMENTS["Temperature"])
            )
        else:
            kelvin = float(data)
        slider = self._sliders.get("Temperature")
        if slider is None:
            return
        spec = next(s for s in SLIDER_SPECS if s.key == "Temperature")
        self._block_emit = True
        try:
            slider.setValue(spec.value_to_slider(kelvin))
            self._update_slider_label("Temperature", spec.format_value)
        finally:
            self._block_emit = False
        self._emit_preview_and_save()

    def _sync_wb_preset_combo(self, temperature: float | None = None) -> None:
        """Match the WB dropdown to the current Kelvin (or Custom)."""
        combo = getattr(self, "_wb_preset_combo", None)
        if combo is None:
            return
        if temperature is None:
            slider = self._sliders.get("Temperature")
            if slider is None:
                return
            spec = next(s for s in SLIDER_SPECS if s.key == "Temperature")
            temperature = float(spec.slider_to_value(slider.value()))
        # Keep the current selection when it already matches (Daylight and Flash
        # share 5500K — don't flip the label on every slider tick).
        cur_data = combo.currentData()
        cur_text = str(combo.currentText())
        as_shot = float(
            getattr(self, "_as_shot_temperature", DEFAULT_ADJUSTMENTS["Temperature"])
        )
        if cur_data is None and not cur_text.startswith("Custom"):
            if abs(temperature - as_shot) <= 25.0:
                return
        elif cur_data is not None and cur_data != -1.0:
            if abs(temperature - float(cur_data)) <= 25.0:
                return
        target_idx = None
        for i in range(combo.count()):
            data = combo.itemData(i)
            text = combo.itemText(i)
            if text.startswith("Custom"):
                continue
            if data is None:
                if abs(temperature - as_shot) <= 25.0:
                    target_idx = i
                    break
            elif data != -1.0 and abs(temperature - float(data)) <= 25.0:
                target_idx = i
                break
        if target_idx is None:
            for i in range(combo.count()):
                if str(combo.itemText(i)).startswith("Custom"):
                    target_idx = i
                    break
        if target_idx is None:
            return
        self._wb_preset_block = True
        try:
            combo.setCurrentIndex(target_idx)
        finally:
            self._wb_preset_block = False

    def _build_crop_controls(self, sect: CollapsibleSection) -> None:
        """Crop mode entry + aspect pills + Apply/Cancel/Reset.

        Ratio and action rows stay hidden until Crop is toggled on — visible but
        disabled pills felt unresponsive. No ratio is pre-selected: entering crop
        starts unconstrained (free) until the user picks a lock.
        """
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        self._crop_btn = QPushButton("Crop")
        self._crop_btn.setObjectName("adjust_nr_btn")
        self._crop_btn.setCheckable(True)
        self._crop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._crop_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._crop_btn.setToolTip(
            "Interactive crop on the image — drag handles; Apply writes Crop insets"
        )
        self._crop_btn.toggled.connect(self._on_crop_toggled)
        layout.addWidget(self._crop_btn)

        self._crop_ratio_wrap = QWidget()
        ratio_row = QHBoxLayout(self._crop_ratio_wrap)
        ratio_row.setContentsMargins(0, 0, 0, 0)
        ratio_row.setSpacing(4)
        self._crop_ratio_btns: dict[str, QPushButton] = {}
        # label -> aspect (width/height) or None for free; "original" is special.
        self._crop_ratio_defs = (
            ("Free", None),
            ("Original", "original"),
            ("1:1", 1.0),
            ("4:3", 4.0 / 3.0),
            ("3:2", 3.0 / 2.0),
            ("16:9", 16.0 / 9.0),
        )
        for label, _aspect in self._crop_ratio_defs:
            btn = QPushButton(label)
            btn.setObjectName("adjust_nr_btn")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setAutoExclusive(True)
            btn.clicked.connect(lambda _=False, lab=label: self._on_crop_ratio(lab))
            ratio_row.addWidget(btn, 1)
            self._crop_ratio_btns[label] = btn
        self._crop_ratio_wrap.setVisible(False)
        layout.addWidget(self._crop_ratio_wrap)

        self._crop_action_wrap = QWidget()
        action_row = QHBoxLayout(self._crop_action_wrap)
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)
        self._crop_apply_btn = QPushButton("Apply")
        self._crop_cancel_btn = QPushButton("Cancel")
        self._crop_reset_btn = QPushButton("Reset")
        for btn, tip, slot in (
            (self._crop_apply_btn, "Commit crop insets", self.crop_apply_requested.emit),
            (self._crop_cancel_btn, "Leave crop mode without saving", self.crop_cancel_requested.emit),
            (self._crop_reset_btn, "Clear crop insets", self.crop_reset_requested.emit),
        ):
            btn.setObjectName("adjust_nr_btn")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            action_row.addWidget(btn, 1)
        self._crop_action_wrap.setVisible(False)
        layout.addWidget(self._crop_action_wrap)
        sect.add_widget(wrap)
        self._crop_active = False
        self._crop_insets = (0.0, 0.0, 0.0, 0.0)

    def _sync_crop_subcontrols_visible(self, active: bool) -> None:
        """Show ratio / Apply-Cancel-Reset only while crop mode is on."""
        on = bool(active)
        if hasattr(self, "_crop_ratio_wrap"):
            self._crop_ratio_wrap.setVisible(on)
        if hasattr(self, "_crop_action_wrap"):
            self._crop_action_wrap.setVisible(on)
        if not on:
            # autoExclusive blocks clearing the checked button; disable briefly.
            for btn in getattr(self, "_crop_ratio_btns", {}).values():
                btn.blockSignals(True)
                btn.setAutoExclusive(False)
                try:
                    btn.setChecked(False)
                finally:
                    btn.setAutoExclusive(True)
                    btn.blockSignals(False)

    def _on_crop_toggled(self, checked: bool) -> None:
        self._crop_active = bool(checked)
        self._sync_crop_subcontrols_visible(self._crop_active)
        if self._crop_active:
            self.disarm_dodge_burn()
        self.crop_mode_changed.emit(self._crop_active)

    def _on_crop_ratio(self, label: str) -> None:
        if not getattr(self, "_crop_active", False):
            return
        for name, btn in self._crop_ratio_btns.items():
            btn.setChecked(name == label)
        aspect = None
        for name, value in self._crop_ratio_defs:
            if name == label:
                aspect = value
                break
        self.crop_aspect_changed.emit(aspect)

    def is_crop_mode(self) -> bool:
        return bool(getattr(self, "_crop_active", False))

    def set_crop_mode_ui(self, active: bool) -> None:
        """Sync Crop button without re-emitting (host cancel / apply)."""
        btn = getattr(self, "_crop_btn", None)
        if btn is None:
            return
        btn.blockSignals(True)
        try:
            btn.setChecked(bool(active))
        finally:
            btn.blockSignals(False)
        self._crop_active = bool(active)
        self._sync_crop_subcontrols_visible(self._crop_active)

    def get_crop_insets(self) -> tuple[float, float, float, float]:
        return tuple(getattr(self, "_crop_insets", (0.0, 0.0, 0.0, 0.0)))

    def set_crop_insets(
        self, left: float, right: float, top: float, bottom: float, *, emit: bool = True
    ) -> None:
        self._crop_insets = (
            float(left),
            float(right),
            float(top),
            float(bottom),
        )
        if emit:
            self._emit_preview_and_save()

    def _build_wb_picker_button(self, row: QHBoxLayout) -> None:
        """Small icon-only dropper button, inline at the end of the Temperature row."""
        btn = QPushButton()
        btn.setObjectName("adjust_wb_picker_btn")
        btn.setCheckable(True)
        btn.setFixedSize(20, 20)
        btn.setIconSize(QSize(12, 12))
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(
            "Pick white balance — click, then click a neutral gray/white area"
            " in the image. Esc cancels."
        )
        self._wb_picker_icon_default = _qta_icon_safe("fa5s.eye-dropper", color=theme.INK_MUTED)
        self._wb_picker_icon_active = _qta_icon_safe("fa5s.eye-dropper", color=theme.EMBER)
        btn.setIcon(self._wb_picker_icon_default)
        btn.toggled.connect(self._on_wb_picker_toggled)
        row.addWidget(btn)
        self._wb_picker_btn = btn
        auto = QPushButton("AUTO")
        auto.setObjectName("adjust_auto_wb_btn")
        auto.setFixedHeight(20)
        auto.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        auto.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        auto.setToolTip("Auto white balance — estimate from the image (robust gray-world)")
        auto.clicked.connect(lambda: self.auto_wb_requested.emit())
        row.addWidget(auto)

    def _on_wb_picker_toggled(self, checked: bool) -> None:
        self._sync_wb_picker_icon(checked)
        self.wb_picker_toggled.emit(bool(checked))

    def set_wb_picker_active(self, active: bool) -> None:
        """Programmatically arm/disarm the dropper button (e.g. after a pick completes)."""
        btn = getattr(self, "_wb_picker_btn", None)
        if btn is None:
            return
        btn.blockSignals(True)
        try:
            btn.setChecked(bool(active))
        finally:
            btn.blockSignals(False)
        self._sync_wb_picker_icon(active)

    def _sync_wb_picker_icon(self, active: bool) -> None:
        btn = getattr(self, "_wb_picker_btn", None)
        if btn is None:
            return
        icon = (
            getattr(self, "_wb_picker_icon_active", None)
            if active
            else getattr(self, "_wb_picker_icon_default", None)
        )
        if icon is not None:
            btn.setIcon(icon)

    def apply_picked_white_balance(self, temperature: float, tint: float) -> None:
        """Set Temperature/Tint from an external sample (WB dropper) and save."""
        current = self.get_adjustments()
        current["Temperature"] = float(temperature)
        current["Tint"] = float(tint)
        self.set_adjustments(current)
        self._emit_preview_and_save()

    def _build_looks_section(self, sect: CollapsibleSection) -> None:
        """Shared library for Creative LUT (.cube) and XMP presets (.xmp)."""
        tip = QLabel(
            "Drop .cube / .xmp here to add. Click to apply; click again to clear. "
            "× removes from the library. .cube stays linked with Amount "
            "(default Rec.709 = after gamma; Linear = scene-linear); "
            ".xmp applies once like pasting a look."
        )
        tip.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 10px;")
        tip.setWordWrap(True)
        sect.add_widget(tip)

        drop = _FileDropFrame((".cube", ".xmp"))
        drop.setObjectName("looks_drop_frame")
        drop.setAcceptDrops(True)
        drop.setMinimumHeight(96)
        drop.setStyleSheet(
            f"""
            QFrame#looks_drop_frame {{
                background: {theme.SURFACE};
                border: 1px dashed {theme.LINE};
                border-radius: 6px;
            }}
            """
        )
        drop_layout = QVBoxLayout(drop)
        drop_layout.setContentsMargins(6, 6, 6, 6)
        drop_layout.setSpacing(4)

        self._looks_list = QListWidget()
        self._lut_list = self._looks_list  # legacy alias used by LUT helpers
        self._looks_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._looks_list.setMaximumHeight(130)
        self._looks_list.setSpacing(1)
        self._looks_list.setStyleSheet(
            f"""
            QListWidget {{
                background: transparent;
                border: none;
                color: {theme.INK};
                font-size: 11px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 0;
                margin: 0;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background: {theme.EMBER_DIM};
                color: {theme.INK};
            }}
            QListWidget::item:hover {{
                background: {theme.RAISED};
            }}
            """
        )
        self._looks_list.itemClicked.connect(self._on_looks_item_clicked)
        drop_layout.addWidget(self._looks_list)
        drop.files_dropped.connect(self._import_looks_paths)
        drop.unsupported_dropped.connect(self._on_looks_unsupported_dropped)
        sect.add_widget(drop)

        amt_row = QHBoxLayout()
        amt_row.setSpacing(6)
        amt_lbl = QLabel("LUT Amount")
        amt_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        amt_lbl.setMinimumWidth(78)
        amt_row.addWidget(amt_lbl)
        self._lut_amount_slider = AdjustSlider(Qt.Orientation.Horizontal)
        self._lut_amount_slider.setRange(0, 100)
        self._lut_amount_slider.setValue(100)
        self._lut_amount_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lut_amount_slider.setToolTip("Strength of the active .cube LUT (ignored for .xmp)")
        self._lut_amount_value = QLabel("100")
        self._lut_amount_value.setStyleSheet(f"color: {theme.EMBER}; font-size: 11px;")
        self._lut_amount_value.setMinimumWidth(28)
        self._lut_amount_slider.valueChanged.connect(self._on_lut_amount_changed)
        self._lut_amount_slider.sliderReleased.connect(self._emit_preview_and_save)
        amt_row.addWidget(self._lut_amount_slider, 1)
        amt_row.addWidget(self._lut_amount_value)
        sect.add_layout(amt_row)

        space_row = QHBoxLayout()
        space_row.setSpacing(6)
        space_lbl = QLabel("LUT Space")
        space_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        space_lbl.setMinimumWidth(78)
        space_row.addWidget(space_lbl)
        from raw_lut import LUT_SPACE_LINEAR, LUT_SPACE_REC709

        self._lut_space_combo = QComboBox()
        self._lut_space_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._lut_space_combo.addItem("Rec.709 (after gamma)", LUT_SPACE_REC709)
        self._lut_space_combo.addItem("Linear (scene)", LUT_SPACE_LINEAR)
        self._lut_space_combo.setToolTip(
            "Rec.709: apply .cube after BT.709 encode (typical creative LUTs). "
            "Linear: apply in display-linear before encode (scene-referred cubes)."
        )
        self._lut_space_combo.currentIndexChanged.connect(self._on_lut_space_changed)
        space_row.addWidget(self._lut_space_combo, 1)
        sect.add_layout(space_row)

        self._lut_active_name = ""
        self._xmp_highlight_name = ""
        self._looks_ignore_item_click = False
        self._refresh_looks_list(select_lut=None)

    def _looks_item_kind(self, item) -> str:
        if item is None:
            return ""
        kind = item.data(Qt.ItemDataRole.UserRole)
        return str(kind or "")

    def _looks_item_name(self, item) -> str:
        if item is None:
            return ""
        name = item.data(_LOOKS_NAME_ROLE)
        if name:
            return str(name)
        return str(item.text() or "")

    def _refresh_looks_list(self, select_lut: str | None = None) -> None:
        from raw_lut import list_managed_luts
        from raw_xmp_presets import list_managed_presets

        lw = getattr(self, "_looks_list", None)
        if lw is None:
            return
        wanted_lut = (
            select_lut
            if select_lut is not None
            else getattr(self, "_lut_active_name", "")
        )
        wanted_xmp = str(getattr(self, "_xmp_highlight_name", "") or "")
        lw.blockSignals(True)
        try:
            lw.clear()
            entries: list[tuple[str, str]] = []
            for name in list_managed_luts():
                entries.append((name, "cube"))
            for name in list_managed_presets():
                entries.append((name, "xmp"))
            entries.sort(key=lambda t: t[0].lower())
            select_item = None
            for name, kind in entries:
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, kind)
                item.setData(_LOOKS_NAME_ROLE, name)
                item.setText(name)  # for accessibility / findItems
                item.setToolTip(
                    "Creative LUT (.cube) — click to apply, click again to clear"
                    if kind == "cube"
                    else "XMP preset — click to paste onto this image; click again to deselect"
                )
                row = _LooksRowWidget(name, kind)
                row.remove_clicked.connect(
                    lambda checked=False, n=name, k=kind: self._remove_look(n, k)
                )
                item.setSizeHint(QSize(0, max(28, row.sizeHint().height())))
                lw.addItem(item)
                lw.setItemWidget(item, row)
                # Prefer the just-applied .xmp highlight over an embedded .cube
                # so click-to-apply feedback stays on the preset row.
                if kind == "xmp" and wanted_xmp and name == wanted_xmp:
                    select_item = item
                elif (
                    kind == "cube"
                    and wanted_lut
                    and name == wanted_lut
                    and select_item is None
                ):
                    select_item = item
            if select_item is not None:
                lw.setCurrentItem(select_item)
            else:
                lw.setCurrentRow(-1)
                lw.clearSelection()
        finally:
            lw.blockSignals(False)

    def _refresh_lut_list(self, select_name: str | None = None) -> None:
        """Back-compat for set_adjustments — refresh shared Looks list."""
        self._refresh_looks_list(select_lut=select_name)

    def _refresh_xmp_preset_list(self, select_name: str | None = None) -> None:
        """Back-compat — XMP highlight is ephemeral; just refresh the list."""
        if select_name:
            self._xmp_highlight_name = select_name
        self._refresh_looks_list()

    def _on_looks_unsupported_dropped(self, paths: list[str]) -> None:
        names = [os.path.basename(p) for p in paths if p]
        if not names:
            return
        sample = ", ".join(names[:3])
        extra = f" (+{len(names) - 3} more)" if len(names) > 3 else ""
        self.looks_drop_rejected.emit(
            f"Only .cube and .xmp files can be added here — ignored: {sample}{extra}"
        )

    def _import_looks_paths(self, paths: list[str]) -> None:
        """Drop / import only adds to the library — click a row to apply."""
        from raw_lut import import_cube_file
        from raw_xmp_presets import import_xmp_preset

        for p in paths:
            pl = p.lower()
            try:
                if pl.endswith(".cube"):
                    import_cube_file(p)
                elif pl.endswith(".xmp"):
                    import_xmp_preset(p)
                else:
                    self._on_looks_unsupported_dropped([p])
            except Exception as exc:
                print(f"[LOOKS] import failed: {p}: {exc}")
        self._refresh_looks_list(
            select_lut=getattr(self, "_lut_active_name", "") or None
        )

    def _remove_look(self, name: str, kind: str) -> None:
        from raw_lut import remove_managed_lut
        from raw_xmp_presets import remove_managed_preset

        self._looks_ignore_item_click = True
        try:
            if kind == "cube":
                remove_managed_lut(name)
                if getattr(self, "_lut_active_name", "") == name:
                    self._lut_active_name = ""
                    self._emit_preview_and_save()
            elif kind == "xmp":
                remove_managed_preset(name)
                if getattr(self, "_xmp_highlight_name", "") == name:
                    self._xmp_highlight_name = ""
        except Exception:
            pass
        self._refresh_looks_list(
            select_lut=getattr(self, "_lut_active_name", "") or None
        )

    def _remove_selected_look(self) -> None:
        """Back-compat helper — remove the current list selection."""
        lw = getattr(self, "_looks_list", None)
        if lw is None:
            return
        item = lw.currentItem()
        if item is None:
            return
        self._remove_look(self._looks_item_name(item), self._looks_item_kind(item))

    def _clear_active_lut(self) -> None:
        """Clear the active Creative LUT for this image (toggle-off / None)."""
        self._block_emit = True
        try:
            self._lut_active_name = ""
            self._xmp_highlight_name = ""
            lw = getattr(self, "_looks_list", None)
            if lw is not None:
                lw.setCurrentRow(-1)
                lw.clearSelection()
        finally:
            self._block_emit = False
        self._emit_preview_and_save()

    def _on_looks_item_clicked(self, item) -> None:
        if item is None or self._block_emit:
            return
        # The Looks list is rebuilt (items deleted) by set_adjustments /
        # _refresh_looks_list; a click queued against the old list can arrive
        # holding a dead QListWidgetItem — touching it raises RuntimeError
        # (observed crash in _on_looks_item_clicked, 2026-07-17).
        try:
            from PyQt6 import sip as _sip

            if _sip.isdeleted(item):
                return
        except Exception:
            pass
        lw = getattr(self, "_looks_list", None)
        if lw is not None:
            row = lw.itemWidget(item)
            if row is not None and getattr(row, "_suppress_item_click", False):
                row._suppress_item_click = False
                return
        if getattr(self, "_looks_ignore_item_click", False):
            self._looks_ignore_item_click = False
            return
        kind = self._looks_item_kind(item)
        name = self._looks_item_name(item)
        if not name:
            return
        if kind == "cube":
            if getattr(self, "_lut_active_name", "") == name:
                self._clear_active_lut()
                return
            self._xmp_highlight_name = ""
            self._lut_active_name = name
            if self._lut_amount_slider.value() < 1:
                self._lut_amount_slider.setValue(100)
            self._emit_preview_and_save()
        elif kind == "xmp":
            if getattr(self, "_xmp_highlight_name", "") == name:
                # Second click: deselect (preset pixels already baked in).
                self._xmp_highlight_name = ""
                if lw is not None:
                    lw.setCurrentRow(-1)
                    lw.clearSelection()
                return
            # Apply rebuilds the Looks list via set_adjustments — never reuse
            # this QListWidgetItem afterward (it is deleted on refresh).
            self._apply_xmp_preset_named(name)

    def _on_lut_amount_changed(self, value: int) -> None:
        lbl = getattr(self, "_lut_amount_value", None)
        if lbl is not None:
            lbl.setText(str(int(value)))
        if self._block_emit:
            return
        self._schedule_live_preview()

    def _on_lut_space_changed(self, _index: int = 0) -> None:
        if self._block_emit:
            return
        self._emit_preview_and_save()

    def _apply_xmp_preset_named(self, name: str) -> None:
        """One-shot: load managed XMP look onto this image and persist."""
        if not name:
            return
        try:
            from raw_dodge_burn import MASK_KEY
            from raw_xmp_presets import load_preset_adjustments

            preset = load_preset_adjustments(name)
        except Exception as exc:
            print(f"[XMP_PRESET] apply failed: {name}: {exc}")
            return
        current = self.get_adjustments()
        as_shot = current.get(AS_SHOT_TEMP_KEY)
        # Local paint stays per-image (same as copy/paste settings).
        preset.pop(MASK_KEY, None)
        try:
            from raw_spot_heal import MASK_KEY as HEAL_MASK_KEY

            preset.pop(HEAL_MASK_KEY, None)
        except Exception:
            pass
        preset.pop(AS_SHOT_TEMP_KEY, None)
        merged = dict(DEFAULT_ADJUSTMENTS)
        merged.update(preset)
        if as_shot is not None:
            merged[AS_SHOT_TEMP_KEY] = as_shot
        self.xmp_preset_applied.emit()
        self.set_adjustments(merged)
        # set_adjustments clears the ephemeral xmp highlight; restore selection
        # on the *new* list items after the rebuild.
        self._xmp_highlight_name = name
        self._refresh_looks_list(
            select_lut=getattr(self, "_lut_active_name", "") or None
        )
        self.editing_finished.emit(self.get_adjustments())

    def _build_hsl_section(self, layout: QVBoxLayout) -> None:
        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        color_lbl = QLabel("Color")
        color_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        color_lbl.setFixedWidth(82)
        color_row.addWidget(color_lbl)
        self._hsl_color_combo = QComboBox()
        self._hsl_color_combo.setStyleSheet(_adjust_combo_stylesheet())
        self._hsl_color_combo.addItems(list(HSL_COLOR_NAMES))
        self._hsl_color_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._hsl_color_combo.currentIndexChanged.connect(self._on_hsl_color_changed)
        color_row.addWidget(self._hsl_color_combo, 1)
        layout.addLayout(color_row)

        self._hsl_sliders: Dict[str, QSlider] = {}
        self._hsl_value_labels: Dict[str, QLabel] = {}
        for channel, label in (("Hue", "Hue"), ("Saturation", "Sat"), ("Luminance", "Lum")):
            row = QHBoxLayout()
            row.setSpacing(6)
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
            name_lbl.setFixedWidth(82)
            row.addWidget(name_lbl)
            slider = AdjustSlider(Qt.Orientation.Horizontal)
            slider.setTracking(True)
            slider.setRange(-100, 100)
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.sliderMoved.connect(
                lambda _v, ch=channel: self._update_hsl_label(ch)
            )
            slider.valueChanged.connect(
                lambda _v, ch=channel: self._on_hsl_slider_changed(ch)
            )
            slider.sliderReleased.connect(self._on_slider_released)
            row.addWidget(slider, 1)
            val_lbl = AdjustValueLabel()
            val_lbl.setProperty("class", "adjust_slider_value")
            val_lbl.setStyleSheet(f"color: {theme.EMBER}; font-size: 11px;")
            val_lbl.setFixedWidth(52)
            val_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setToolTip("Click to reset")
            val_lbl.clicked.connect(lambda ch=channel: self._reset_hsl_channel(ch))
            row.addWidget(val_lbl)
            layout.addLayout(row)
            self._hsl_sliders[channel] = slider
            self._hsl_value_labels[channel] = val_lbl

    def _hsl_color_name(self) -> str:
        combo = getattr(self, "_hsl_color_combo", None)
        if combo is None:
            return HSL_COLOR_NAMES[0]
        return combo.currentText()

    def _hsl_key(self, channel: str, color: str | None = None) -> str:
        c = color or self._hsl_color_name()
        return f"{channel}Adjustment{c}"

    def _load_hsl_sliders_from_adj(self, adj: Dict[str, float], color: str | None = None) -> None:
        c = color or self._hsl_color_name()
        for channel in ("Hue", "Saturation", "Luminance"):
            slider = self._hsl_sliders.get(channel)
            val_lbl = self._hsl_value_labels.get(channel)
            if slider is None:
                continue
            val = float(adj.get(self._hsl_key(channel, c), 0.0))
            slider.setValue(int(round(val)))
            if val_lbl is not None:
                val_lbl.setText(f"{int(round(val)):+.0f}")

    def _flush_hsl_cache_from_sliders(self, color: str | None = None) -> None:
        if not hasattr(self, "_hsl_sliders"):
            return
        c = color or getattr(self, "_current_hsl_color", self._hsl_color_name())
        for channel in ("Hue", "Saturation", "Luminance"):
            slider = self._hsl_sliders.get(channel)
            if slider is None:
                continue
            self._hsl_cache[self._hsl_key(channel, c)] = float(slider.value())

    def _on_hsl_color_changed(self, _index: int) -> None:
        if self._block_emit:
            return
        prev_color = getattr(self, "_current_hsl_color", HSL_COLOR_NAMES[0])
        self._flush_hsl_cache_from_sliders(prev_color)
        
        new_color = self._hsl_color_name()
        self._current_hsl_color = new_color
        
        self._block_emit = True
        try:
            self._load_hsl_sliders_from_adj(self._hsl_cache, new_color)
        finally:
            self._block_emit = False

    def _update_hsl_label(self, channel: str) -> None:
        slider = self._hsl_sliders.get(channel)
        val_lbl = self._hsl_value_labels.get(channel)
        if slider is None or val_lbl is None:
            return
        val_lbl.setText(f"{slider.value():+.0f}")

    def _on_hsl_slider_changed(self, channel: str) -> None:
        if self._block_emit:
            return
        self._update_hsl_label(channel)
        self._schedule_live_preview()

    def _reset_hsl_channel(self, channel: str) -> None:
        slider = self._hsl_sliders.get(channel)
        val_lbl = self._hsl_value_labels.get(channel)
        if slider is not None:
            slider.setValue(0)
        if val_lbl is not None:
            val_lbl.setText("+0")
        self._emit_preview_and_save()

    def set_adjustments(self, adj: Dict[str, float]) -> None:
        self.disarm_dodge_burn()
        from raw_dodge_burn import MASK_KEY as _db_mask_key

        self.set_dodge_burn_mask_present(bool(str((adj or {}).get(_db_mask_key, "") or "")))
        try:
            from raw_spot_heal import MASK_KEY as _heal_mask_key

            self.set_spot_heal_mask_present(
                bool(str((adj or {}).get(_heal_mask_key, "") or ""))
            )
        except Exception:
            pass
        self._block_emit = True
        try:
            merged = dict(DEFAULT_ADJUSTMENTS)
            merged.update(adj or {})
            self._as_shot_temperature = float(
                merged.get(AS_SHOT_TEMP_KEY, DEFAULT_ADJUSTMENTS["Temperature"])
            )
            temp_slider = getattr(self, "_temperature_slider", None)
            if temp_slider is not None:
                temp_slider.set_center_value(self._as_shot_temperature)
            for spec in SLIDER_SPECS:
                slider = self._sliders.get(spec.key)
                val_lbl = self._value_labels.get(spec.key)
                if slider is None:
                    continue
                value = float(merged.get(spec.key, spec.default_value))
                slider.setValue(spec.value_to_slider(value))
                if val_lbl is not None:
                    val_lbl.setText(spec.format_value(value))
            self._set_lens_correction_checked(
                float(merged.get("LensCorrectionEnabled", 0.0)) > 0.5
            )
            if self._tone_curve_row is not None:
                for name, key in _CHANNEL_CURVE_KEYS_BY_NAME.items():
                    self._channel_curve_cache[name] = str(merged.get(key, "") or "")
                self._current_curve_channel = "RGB"
                for chan, btn in self._channel_btns.items():
                    btn.setChecked(chan == "RGB")
                self._tone_curve_row.load_serial(self._channel_curve_cache["RGB"])
            for key in self._hsl_cache:
                self._hsl_cache[key] = float(merged.get(key, 0.0))
            if hasattr(self, "_hsl_sliders"):
                self._current_hsl_color = self._hsl_color_name()
                self._load_hsl_sliders_from_adj(self._hsl_cache)
            self.set_recovery_baseline(
                float(merged.get(RECOVERY_BASELINE_KEY, 0.0)) > 0.5
            )
            nr_amount = float(merged.get("ColorNoiseReduction", 0.0))
            if nr_amount > 0.0:
                method_idx = int(float(merged.get("DenoiseMethod", 0.0)))
                combo_idx = method_idx + 1
            else:
                combo_idx = 0
            if hasattr(self, "_denoise_method_combo"):
                self._denoise_method_combo.blockSignals(True)
                self._denoise_method_combo.setCurrentIndex(min(2, max(0, combo_idx)))
                self._denoise_method_combo.blockSignals(False)
            if hasattr(self, "_chroma_nr_amount_slider"):
                amt = int(round(nr_amount)) if nr_amount > 0.0 else int(CHROMA_NR_ON_VALUE)
                amt = max(1, min(100, amt))
                self._chroma_nr_amount_slider.blockSignals(True)
                self._chroma_nr_amount_slider.setValue(amt)
                self._chroma_nr_amount_slider.blockSignals(False)
                if hasattr(self, "_chroma_nr_amount_value"):
                    self._chroma_nr_amount_value.setText(str(amt))
                self._sync_chroma_nr_amount_row_visible()
            if hasattr(self, "_db_mask_strength_slider"):
                stops = float(merged.get(DB_STRENGTH_KEY, DB_DEFAULT_STRENGTH) or DB_DEFAULT_STRENGTH)
                stops = max(0.5, min(3.0, stops))
                self._db_mask_strength_slider.blockSignals(True)
                self._db_mask_strength_slider.setValue(int(round(stops * 20.0)))
                self._db_mask_strength_slider.blockSignals(False)
                if hasattr(self, "_db_mask_strength_value"):
                    self._db_mask_strength_value.setText(f"{stops:.2f}")
            self._sync_wb_preset_combo(float(merged.get("Temperature", self._as_shot_temperature)))
            # Refresh "As Shot" label with the file's Kelvin when known.
            combo = getattr(self, "_wb_preset_combo", None)
            if combo is not None:
                for i in range(combo.count()):
                    if combo.itemData(i) is None and not str(combo.itemText(i)).startswith("Custom"):
                        combo.setItemText(i, f"As Shot ({int(round(self._as_shot_temperature))}K)")
                        break
            self._crop_insets = (
                float(merged.get("CropLeft", 0.0) or 0.0),
                float(merged.get("CropRight", 0.0) or 0.0),
                float(merged.get("CropTop", 0.0) or 0.0),
                float(merged.get("CropBottom", 0.0) or 0.0),
            )
            from raw_lut import (
                LUT_AMOUNT_KEY,
                LUT_NAME_KEY,
                LUT_SPACE_KEY,
                LUT_SPACE_REC709,
                lut_working_space,
            )

            self._lut_active_name = str(merged.get(LUT_NAME_KEY, "") or "").strip()
            amt = int(round(float(merged.get(LUT_AMOUNT_KEY, 0.0) or 0.0)))
            if self._lut_active_name and amt < 1:
                amt = 100
            if hasattr(self, "_lut_amount_slider"):
                self._lut_amount_slider.setValue(max(0, min(100, amt)))
            space = lut_working_space(merged)
            combo = getattr(self, "_lut_space_combo", None)
            if combo is not None:
                idx = combo.findData(space)
                if idx < 0:
                    idx = combo.findData(LUT_SPACE_REC709)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
            # Loading adjustments from an image is not an "active xmp preset"
            # click — drop the ephemeral highlight so a prior Looks selection
            # does not stick across files. (Apply path restores highlight after.)
            self._xmp_highlight_name = ""
            self._refresh_looks_list(select_lut=self._lut_active_name or None)
        finally:
            self._block_emit = False

    def _set_nr_checked(self, on: bool) -> None:
        btn = getattr(self, "_nr_btn", None)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(bool(on))
        btn.setText("On" if on else "Off")
        btn.blockSignals(False)

    def _set_lens_correction_checked(self, on: bool) -> None:
        btn = getattr(self, "_lens_correction_btn", None)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(bool(on))
        btn.setText("On" if on else "Off")
        btn.blockSignals(False)
        
        lbl = getattr(self, "_lens_profile_lbl", None)
        if lbl is not None:
            name = lbl.text().replace(" (Applied)", "").strip()
            if name:
                if on:
                    lbl.setText(f"{name} (Applied)")
                    lbl.setStyleSheet(f"color: {theme.EMBER}; font-size: 10px; font-weight: 500;")
                else:
                    lbl.setText(name)
                    lbl.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")

    def set_lens_correction_available(self, available: bool, profile_name: str = "") -> None:
        """Show/hide the toggle and display the matched lens profile name."""
        row = getattr(self, "_lens_correction_row_widget", None)
        if row is not None:
            row.setVisible(bool(available))
        lbl = getattr(self, "_lens_profile_lbl", None)
        if lbl is not None:
            lbl.setText(profile_name if available else "")
        if not available:
            self._set_lens_correction_checked(False)
        else:
            btn = getattr(self, "_lens_correction_btn", None)
            self._set_lens_correction_checked(btn is not None and btn.isChecked())

    def _on_lens_correction_toggled(self, checked: bool) -> None:
        if self._block_emit:
            return
        self._set_lens_correction_checked(checked)
        self.lens_correction_toggled.emit(bool(checked))
        self.editing_finished.emit(self.get_adjustments())

    def _on_copy_settings_clicked(self) -> None:
        """Snapshot this image's edit settings into the session clipboard.

        Per-image reference values are excluded (AS_SHOT_TEMP_KEY is the
        camera's own WB reference for THIS file); dodge/burn masks never pass
        through get_adjustments at all, so local paint stays per-image the
        way Lightroom's default paste does.
        """
        global _EDIT_SETTINGS_CLIPBOARD
        adj = self.get_adjustments()
        _EDIT_SETTINGS_CLIPBOARD = {
            k: v for k, v in adj.items() if k != AS_SHOT_TEMP_KEY
        }
        self._paste_btn.setEnabled(True)

    def _on_paste_settings_clicked(self) -> None:
        """Apply the copied settings to the current image and persist them."""
        clip = _edit_settings_clipboard()
        if not clip:
            return
        merged = dict(self.get_adjustments())
        merged.update(clip)
        self.set_adjustments(merged)
        # Same path as a slider-release: full-quality preview + XMP write.
        self.editing_finished.emit(self.get_adjustments())

    def get_adjustments(self) -> Dict[str, float]:
        out = dict(DEFAULT_ADJUSTMENTS)
        out[AS_SHOT_TEMP_KEY] = float(
            getattr(self, "_as_shot_temperature", DEFAULT_ADJUSTMENTS["Temperature"])
        )
        for spec in SLIDER_SPECS:
            slider = self._sliders.get(spec.key)
            if slider is None:
                continue
            out[spec.key] = spec.slider_to_value(slider.value())
        if hasattr(self, "_denoise_method_combo"):
            idx = self._denoise_method_combo.currentIndex()
            out["DenoiseMethod"] = float(max(0, idx - 1))
            if idx > 0:
                if hasattr(self, "_chroma_nr_amount_slider"):
                    out["ColorNoiseReduction"] = float(self._chroma_nr_amount_slider.value())
                else:
                    out["ColorNoiseReduction"] = CHROMA_NR_ON_VALUE
            else:
                out["ColorNoiseReduction"] = 0.0
        else:
            out["ColorNoiseReduction"] = 0.0
        if hasattr(self, "_db_mask_strength_slider"):
            # Heal mode forces the Effect Strength slider to max for display
            # only — persist the pre-Heal dodge/burn value so Heal strokes
            # do not rewrite DodgeBurnStrength.
            saved = getattr(self, "_db_mask_strength_saved_for_heal", None)
            if self.dodge_burn_mode() == "heal" and saved is not None:
                out[DB_STRENGTH_KEY] = float(saved) / 20.0
            else:
                out[DB_STRENGTH_KEY] = (
                    float(self._db_mask_strength_slider.value()) / 20.0
                )
        else:
            out[DB_STRENGTH_KEY] = float(DB_DEFAULT_STRENGTH)
        lens_btn = getattr(self, "_lens_correction_btn", None)
        out["LensCorrectionEnabled"] = (
            1.0 if lens_btn is not None and lens_btn.isChecked() else 0.0
        )
        if self._tone_curve_row is not None:
            self._channel_curve_cache[self._current_curve_channel] = (
                self._tone_curve_row.serialized_points()
            )
            for name, key in _CHANNEL_CURVE_KEYS_BY_NAME.items():
                serial = str(self._channel_curve_cache.get(name, "") or "").strip()
                # serialize_tone_curve_points already returns "" for identity;
                # still skip any leftover "0,0;255,255" so Reset clears XMP.
                if not serial:
                    continue
                from raw_adjustments import tone_curve_serial_is_active

                if tone_curve_serial_is_active(serial):
                    out[key] = serial
        if hasattr(self, "_hsl_sliders"):
            self._flush_hsl_cache_from_sliders()
            for key, val in self._hsl_cache.items():
                out[key] = float(val)
        if self._recovery_baseline:
            out[RECOVERY_BASELINE_KEY] = 1.0
        l, r, t, b = self.get_crop_insets()
        out["CropLeft"] = float(l)
        out["CropRight"] = float(r)
        out["CropTop"] = float(t)
        out["CropBottom"] = float(b)
        from raw_lut import LUT_AMOUNT_KEY, LUT_NAME_KEY, LUT_SPACE_KEY, LUT_SPACE_REC709

        name = str(getattr(self, "_lut_active_name", "") or "").strip()
        amt = float(getattr(self, "_lut_amount_slider", None).value()) if hasattr(self, "_lut_amount_slider") else 0.0
        combo = getattr(self, "_lut_space_combo", None)
        space = LUT_SPACE_REC709
        if combo is not None:
            data = combo.currentData()
            if data:
                space = str(data)
        if name and amt > 0:
            out[LUT_NAME_KEY] = name
            out[LUT_AMOUNT_KEY] = amt
            out[LUT_SPACE_KEY] = space
        else:
            # Explicit empty name so XMP merge drops CreativeLUTName and
            # the preview stage key invalidates (toggle-off / amount 0).
            out[LUT_NAME_KEY] = ""
            out[LUT_AMOUNT_KEY] = 0.0
            out[LUT_SPACE_KEY] = space
        return out

    def _on_dodge_burn_toggled(self, btn: QPushButton, checked: bool) -> None:
        if self._block_emit:
            return
        others = [
            b
            for b in (
                self._dodge_btn,
                self._burn_btn,
                self._erase_btn,
                getattr(self, "_heal_btn", None),
            )
            if b is not None and b is not btn
        ]
        if checked:
            self._block_emit = True
            try:
                for other in others:
                    if other.isChecked():
                        other.setChecked(False)
            finally:
                self._block_emit = False
        mode = self.dodge_burn_mode()
        self._sync_dodge_burn_mask_button_enabled(mode is not None)
        self._sync_local_controls_for_mode(mode)
        self.dodge_burn_mode_changed.emit(mode)

    def _sync_dodge_burn_mask_button_enabled(self, armed: bool) -> None:
        """Mask overlay is available while a brush tool is selected.

        Default on when arming a tool so paint coverage is visible immediately
        (especially important for Heal). Turning the tool off clears Mask.
        """
        btn = getattr(self, "_db_show_mask_btn", None)
        if btn is None:
            return
        btn.setEnabled(bool(armed))
        if not armed:
            if btn.isChecked():
                self._block_emit = True
                try:
                    btn.setChecked(False)
                finally:
                    self._block_emit = False
                self.dodgeBurnMaskToggled.emit(False)
            return
        if not btn.isChecked():
            self._block_emit = True
            try:
                btn.setChecked(True)
            finally:
                self._block_emit = False
            self.dodgeBurnMaskToggled.emit(True)

    def _sync_local_controls_for_mode(self, mode: str | None) -> None:
        """Heal only uses Size/Flow — Effect Strength is dodge/burn stops.

        While Heal is armed the Effect Strength row is grayed out and forced
        to maximum (visual cue that heal always runs full-strength). The
        previous dodge/burn stops value is restored when leaving Heal so
        only Dodge/Burn honor that slider.
        """
        heal = mode == "heal"
        row = getattr(self, "_db_mask_strength_row", None)
        slider = getattr(self, "_db_mask_strength_slider", None)
        val_lbl = getattr(self, "_db_mask_strength_value", None)
        if row is not None:
            row.setEnabled(not heal)
        if slider is not None:
            if heal:
                if getattr(self, "_db_mask_strength_saved_for_heal", None) is None:
                    self._db_mask_strength_saved_for_heal = int(slider.value())
                max_v = int(slider.maximum())
                if slider.value() != max_v:
                    slider.blockSignals(True)
                    slider.setValue(max_v)
                    slider.blockSignals(False)
                stops = float(max_v) / 20.0
                if val_lbl is not None:
                    val_lbl.setText(f"{stops:.2f}")
                tip = (
                    "Heal always inpaints at full strength — only Brush Size "
                    "and Brush Flow apply. Effect Strength is for Dodge/Burn."
                )
            else:
                saved = getattr(self, "_db_mask_strength_saved_for_heal", None)
                if saved is not None:
                    self._db_mask_strength_saved_for_heal = None
                    slider.blockSignals(True)
                    slider.setValue(int(saved))
                    slider.blockSignals(False)
                    stops = float(saved) / 20.0
                    if val_lbl is not None:
                        val_lbl.setText(f"{stops:.2f}")
                tip = (
                    "How many stops of exposure the painted dodge/burn mask "
                    "applies at full strength.\n"
                    "Not used by Heal (inpaint always runs at full strength)."
                )
            slider.setToolTip(tip)
        edge = getattr(self, "_db_edge_btn", None)
        if edge is not None:
            edge.setEnabled(not heal)

    def dodge_burn_mode(self) -> str | None:
        if self._dodge_btn.isChecked():
            return "dodge"
        if self._burn_btn.isChecked():
            return "burn"
        if getattr(self, "_erase_btn", None) is not None and self._erase_btn.isChecked():
            return "erase"
        if getattr(self, "_heal_btn", None) is not None and self._heal_btn.isChecked():
            return "heal"
        return None

    def set_dodge_burn_mode(self, mode: str | None) -> None:
        """Arm Dodge/Burn/Eraser/Heal from a shortcut (D/B/X/H) or disarm."""
        want = (mode or "").strip().lower()
        if want in ("eraser", "erase"):
            want = "erase"
        if want not in ("dodge", "burn", "erase", "heal", ""):
            return
        dodge_on = want == "dodge"
        burn_on = want == "burn"
        erase_on = want == "erase"
        heal_on = want == "heal"
        erase_btn = getattr(self, "_erase_btn", None)
        heal_btn = getattr(self, "_heal_btn", None)
        cur_dodge = self._dodge_btn.isChecked()
        cur_burn = self._burn_btn.isChecked()
        cur_erase = bool(erase_btn is not None and erase_btn.isChecked())
        cur_heal = bool(heal_btn is not None and heal_btn.isChecked())
        if (
            cur_dodge == dodge_on
            and cur_burn == burn_on
            and cur_erase == erase_on
            and cur_heal == heal_on
        ):
            # Same mode again → toggle off.
            if want:
                dodge_on = burn_on = erase_on = heal_on = False
        self._block_emit = True
        try:
            self._dodge_btn.setChecked(dodge_on)
            self._burn_btn.setChecked(burn_on)
            if erase_btn is not None:
                erase_btn.setChecked(erase_on)
            if heal_btn is not None:
                heal_btn.setChecked(heal_on)
        finally:
            self._block_emit = False
        armed = dodge_on or burn_on or erase_on or heal_on
        self._sync_dodge_burn_mask_button_enabled(armed)
        if dodge_on:
            out = "dodge"
        elif burn_on:
            out = "burn"
        elif erase_on:
            out = "erase"
        elif heal_on:
            out = "heal"
        else:
            out = None
        self._sync_local_controls_for_mode(out)
        self.dodge_burn_mode_changed.emit(out)

    def disarm_dodge_burn(self) -> None:
        """Uncheck brush tool buttons (e.g. on file switch / crop)."""
        was_armed = self.dodge_burn_mode() is not None
        self._block_emit = True
        try:
            self._dodge_btn.setChecked(False)
            self._burn_btn.setChecked(False)
            erase_btn = getattr(self, "_erase_btn", None)
            if erase_btn is not None:
                erase_btn.setChecked(False)
            heal_btn = getattr(self, "_heal_btn", None)
            if heal_btn is not None:
                heal_btn.setChecked(False)
        finally:
            self._block_emit = False
        self._sync_dodge_burn_mask_button_enabled(False)
        self._sync_local_controls_for_mode(None)
        if was_armed:
            self.dodge_burn_mode_changed.emit(None)

    def dodge_burn_brush_radius(self) -> float:
        return float(self._db_size_slider.value())

    def nudge_dodge_burn_brush_size(self, wheel_delta: int) -> None:
        """Adjust Brush Size from a trackpad/wheel delta (positive = larger)."""
        slider = getattr(self, "_db_size_slider", None)
        if slider is None or not wheel_delta:
            return
        dy = int(wheel_delta)
        if abs(dy) >= 120:
            amount = (dy // 120) * 8
        else:
            # Trackpad pixel deltas: ~1 size unit per 4px of scroll.
            amount = int(round(dy / 4.0))
        if amount == 0:
            amount = 1 if dy > 0 else -1
        slider.setValue(
            max(slider.minimum(), min(slider.maximum(), slider.value() + amount))
        )

    def dodge_burn_brush_strength(self) -> float:
        """0..1: per-stamp delta at the brush center before pressure scaling."""
        return float(self._db_strength_slider.value()) / 100.0

    def dodge_burn_edge_assist(self) -> bool:
        btn = getattr(self, "_db_edge_btn", None)
        return True if btn is None else bool(btn.isChecked())

    def set_dodge_burn_mask_present(self, present: bool) -> None:
        self._db_clear_btn.setEnabled(bool(present))

    def set_spot_heal_mask_present(self, present: bool) -> None:
        btn = getattr(self, "_heal_clear_btn", None)
        if btn is not None:
            btn.setEnabled(bool(present))

    def dodge_burn_show_mask(self) -> bool:
        return bool(self._db_show_mask_btn.isChecked())

    def toggle_dodge_burn_show_mask(self) -> None:
        if not self._db_show_mask_btn.isEnabled():
            return
        self._db_show_mask_btn.toggle()

    def _update_slider_label(self, key: str, fmt: Callable[[float], str]) -> None:
        slider = self._sliders.get(key)
        val_lbl = self._value_labels.get(key)
        if slider is None:
            return
        spec = next(s for s in SLIDER_SPECS if s.key == key)
        text = fmt(spec.slider_to_value(slider.value()))
        if val_lbl is not None:
            val_lbl.setText(text)
        if key == "Temperature":
            self._sync_wb_preset_combo(float(spec.slider_to_value(slider.value())))

    def _emit_live_preview(self) -> None:
        if self._block_emit:
            return
        self.preview_changed.emit(self.get_adjustments())

    def _schedule_live_preview(self) -> None:
        if self._block_emit:
            return
        if not self._preview_timer.isActive():
            self._preview_timer.start(self._PREVIEW_THROTTLE_MS)

    def _on_tone_curve_changed(self) -> None:
        if self._block_emit:
            return
        self._clear_recovery_baseline()
        self._schedule_live_preview()

    def _on_tone_curve_finished(self) -> None:
        if self._block_emit:
            return
        self._emit_preview_and_save()

    def _on_curve_channel_selected(self, name: str) -> None:
        if self._tone_curve_row is None:
            return
        if name == self._current_curve_channel:
            for chan, btn in self._channel_btns.items():
                btn.setChecked(chan == name)
            return
        self._channel_curve_cache[self._current_curve_channel] = (
            self._tone_curve_row.serialized_points()
        )
        self._current_curve_channel = name
        for chan, btn in self._channel_btns.items():
            btn.setChecked(chan == name)
        was_blocked = self._block_emit
        self._block_emit = True
        try:
            self._tone_curve_row.load_serial(self._channel_curve_cache.get(name, ""))
        finally:
            self._block_emit = was_blocked

    def _clear_recovery_baseline(self) -> None:
        if not self._recovery_baseline:
            return
        self._recovery_baseline = False
        self._sync_recovery_button()

    def _sync_recovery_button(self) -> None:
        btn = getattr(self, "_recovery_btn", None)
        if btn is None:
            return
        if self._recovery_baseline:
            btn.setText("Recovery look · on")
            btn.setStyleSheet(
                "color: #E8F4FF; background: rgba(48, 96, 144, 90);"
                " border: 1px solid rgba(120, 180, 255, 120); border-radius: 4px;"
            )
        else:
            btn.setText("Use recovery look")
            btn.setStyleSheet("")

    def _on_recovery_baseline_clicked(self) -> None:
        if self._recovery_baseline:
            self._clear_recovery_baseline()
            self._emit_preview_and_save()
            return
        self.recovery_baseline_requested.emit()

    def set_recovery_baseline(self, enabled: bool) -> None:
        self._recovery_baseline = bool(enabled)
        if enabled:
            self._apply_recovery_slider_hints()
        self._sync_recovery_button()

    def _apply_recovery_slider_hints(self) -> None:
        """Move Highlights/Shadows readouts to match recovery look (hint only while recovery on)."""
        hints = recovery_baseline_slider_hints()
        self._block_emit = True
        try:
            for key, value in hints.items():
                spec = next((s for s in SLIDER_SPECS if s.key == key), None)
                slider = self._sliders.get(key)
                val_lbl = self._value_labels.get(key)
                if spec is None or slider is None:
                    continue
                slider.setValue(spec.value_to_slider(value))
                if val_lbl is not None:
                    val_lbl.setText(spec.format_value(value))
        finally:
            self._block_emit = False

    def _request_export(self, export_format: str) -> None:
        self.export_requested.emit(export_format, self.get_adjustments())

    def _on_slider_value_changed(self, key: str, fmt: Callable[[float], str]) -> None:
        if self._block_emit:
            return
        if is_pv2012_tone_slider(key):
            self._clear_recovery_baseline()
        self._update_slider_label(key, fmt)
        self._schedule_live_preview()

    def _emit_preview_and_save(self) -> None:
        if self._block_emit:
            return
        self._preview_timer.stop()
        self._emit_live_preview()
        self.editing_finished.emit(self.get_adjustments())

    def _on_slider_released(self) -> None:
        if self._block_emit:
            return
        self._emit_preview_and_save()

    def _on_denoise_method_changed(self, index: int) -> None:
        if self._block_emit:
            return
        self._sync_chroma_nr_amount_row_visible()
        # Turning on from Off: seed amount to the historical default (50) if
        # the slider was never touched for this file.
        if index > 0 and hasattr(self, "_chroma_nr_amount_slider"):
            if self._chroma_nr_amount_slider.value() < 1:
                self._chroma_nr_amount_slider.setValue(int(CHROMA_NR_ON_VALUE))
        self._emit_preview_and_save()

    def _sync_chroma_nr_amount_row_visible(self) -> None:
        row = getattr(self, "_chroma_nr_amount_row", None)
        combo = getattr(self, "_denoise_method_combo", None)
        if row is None or combo is None:
            return
        row.setVisible(combo.currentIndex() > 0)

    def _on_chroma_nr_amount_changed(self, value: int) -> None:
        if hasattr(self, "_chroma_nr_amount_value"):
            self._chroma_nr_amount_value.setText(str(int(value)))
        if self._block_emit:
            return
        self._schedule_live_preview()

    def _on_db_mask_strength_changed(self, value: int) -> None:
        stops = float(value) / 20.0
        if hasattr(self, "_db_mask_strength_value"):
            self._db_mask_strength_value.setText(f"{stops:.2f}")
        if self._block_emit:
            return
        self._schedule_live_preview()

    def _reset_slider(self, key: str) -> None:
        spec = next((s for s in SLIDER_SPECS if s.key == key), None)
        if spec is None:
            return
        self._block_emit = True
        try:
            slider = self._sliders.get(key)
            val_lbl = self._value_labels.get(key)
            if key == "Temperature":
                default = float(
                    getattr(self, "_as_shot_temperature", spec.default_value)
                )
            else:
                default = float(DEFAULT_ADJUSTMENTS.get(key, spec.default_value))
            if slider is not None:
                slider.setValue(spec.value_to_slider(default))
            if val_lbl is not None:
                val_lbl.setText(spec.format_value(default))
        finally:
            self._block_emit = False
        self._emit_preview_and_save()

    def _on_reset_clicked(self) -> None:
        reset = dict(DEFAULT_ADJUSTMENTS)
        reset[AS_SHOT_TEMP_KEY] = float(
            getattr(self, "_as_shot_temperature", DEFAULT_ADJUSTMENTS["Temperature"])
        )
        reset["Temperature"] = reset[AS_SHOT_TEMP_KEY]
        self.set_adjustments(reset)
        self._hsl_cache = {
            k: float(v)
            for k, v in DEFAULT_ADJUSTMENTS.items()
            if k.startswith("HueAdjustment")
            or k.startswith("SaturationAdjustment")
            or k.startswith("LuminanceAdjustment")
        }
        self._current_hsl_color = HSL_COLOR_NAMES[0]
        if hasattr(self, "_hsl_color_combo"):
            self._hsl_color_combo.blockSignals(True)
            self._hsl_color_combo.setCurrentIndex(0)
            self._hsl_color_combo.blockSignals(False)
        if self._tone_curve_row is not None:
            for name in list(self._channel_curve_cache.keys()):
                self._channel_curve_cache[name] = ""
            self._tone_curve_row.reset_linear()
        self._lut_active_name = ""
        self._xmp_highlight_name = ""
        if hasattr(self, "_lut_amount_slider"):
            self._lut_amount_slider.blockSignals(True)
            self._lut_amount_slider.setValue(100)
            self._lut_amount_slider.blockSignals(False)
        self._refresh_looks_list(select_lut=None)
        if hasattr(self, "_denoise_method_combo"):
            self._denoise_method_combo.blockSignals(True)
            self._denoise_method_combo.setCurrentIndex(0)
            self._denoise_method_combo.blockSignals(False)
        if hasattr(self, "_chroma_nr_amount_slider"):
            self._chroma_nr_amount_slider.blockSignals(True)
            self._chroma_nr_amount_slider.setValue(int(CHROMA_NR_ON_VALUE))
            self._chroma_nr_amount_slider.blockSignals(False)
            if hasattr(self, "_chroma_nr_amount_value"):
                self._chroma_nr_amount_value.setText(str(int(CHROMA_NR_ON_VALUE)))
            self._sync_chroma_nr_amount_row_visible()
        if hasattr(self, "_db_mask_strength_slider"):
            self._db_mask_strength_slider.blockSignals(True)
            self._db_mask_strength_slider.setValue(int(round(DB_DEFAULT_STRENGTH * 20.0)))
            self._db_mask_strength_slider.blockSignals(False)
            if hasattr(self, "_db_mask_strength_value"):
                self._db_mask_strength_value.setText(f"{DB_DEFAULT_STRENGTH:.2f}")
        self._clear_recovery_baseline()
        self.reset_requested.emit()
        self._emit_preview_and_save()

    def _on_export_clicked(self) -> None:
        self._request_export("tiff16")

    def set_export_enabled(self, enabled: bool) -> None:
        btn = getattr(self, "_export_btn", None)
        if btn is not None:
            btn.setEnabled(bool(enabled))



