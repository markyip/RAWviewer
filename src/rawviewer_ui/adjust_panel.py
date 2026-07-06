"""Draggable RAW adjustment panel for single-image view (E to toggle)."""

from __future__ import annotations

from typing import Callable, Dict

from PyQt6.QtCore import QRectF, Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QIcon, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
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

from raw_adjustments import (
    AS_SHOT_TEMP_KEY,
    CHROMA_NR_ON_VALUE,
    DEFAULT_ADJUSTMENTS,
    RECOVERY_BASELINE_KEY,
    SLIDER_SPECS,
    is_pv2012_tone_slider,
    recovery_baseline_slider_hints,
)
from raw_hsl import HSL_COLOR_NAMES
from raw_tone_curve import TONE_CURVE_SERIAL_KEY

# Point-curve + parametric PV rows.
_SHOW_TONE_CURVE_UI = True
# HSL section — hidden pending a saturation/vibrance review (see docs).
_SHOW_HSL_UI = True

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
    "Temperature": [(0.0, "#4A73B5"), (0.5, "#3A3A3A"), (1.0, "#C98A46")],
    "Tint": [(0.0, "#4A9B5E"), (0.5, "#3A3A3A"), (1.0, "#B457A0")],
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


class CollapsibleSection(QWidget):
    """A clean, Lightroom-style collapsible accordion section for PyQt6."""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._expanded = True
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # Header button/panel
        self.header = QWidget()
        self.header.setObjectName("accordion_header")
        self.header.setStyleSheet("""
            QWidget#accordion_header {
                background-color: #242424;
                border-top: 1px solid #2e2e2e;
                border-bottom: 1px solid #2e2e2e;
            }
        """)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(8)
        
        # Arrow label
        self.arrow = QLabel("▼")
        self.arrow.setStyleSheet("color: #888888; font-size: 10px; font-weight: bold;")
        header_layout.addWidget(self.arrow)
        
        # Title label
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("""
            color: #A0A0A0; 
            font-size: 10px; 
            font-weight: 700; 
            letter-spacing: 1px;
        """)
        header_layout.addWidget(self.title_lbl, 1)
        
        # Enable clicking on header
        self.header.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.header.mousePressEvent = self._on_header_pressed
        
        main_layout.addWidget(self.header)
        
        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 8, 12, 8)
        self.content_layout.setSpacing(10)
        main_layout.addWidget(self.content)
        
    def _on_header_pressed(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_expanded(not self._expanded)
            
    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.content.setVisible(self._expanded)
        self.arrow.setText("▼" if self._expanded else "▶")
        
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
        self._center_value: float | None = None  # None -> auto (0 if bipolar, else minimum)
        self._track_gradient: list[tuple[float, str]] | None = None
        self._accent = QColor("#90CAF9")

    def set_center_value(self, value: float | None) -> None:
        """Override the fill's zero/reference point (e.g. as-shot Temperature
        instead of a literal 0, which isn't in-range for an absolute-Kelvin
        slider). None restores the automatic bipolar/left-edge default."""
        self._center_value = value
        self.update()

    def set_track_gradient(self, stops: list[tuple[float, str]] | None) -> None:
        """stops: [(0.0, '#4080ff'), (0.5, '#3a3a3a'), (1.0, '#ffb040')] --
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
            painter.setBrush(QColor("#3A3A3A"))
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
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.setBrush(self._accent if self.underMouse() else QColor("#FFFFFF"))
        painter.drawRoundedRect(thumb_rect, 1.5, 1.5)
        painter.end()


class ImageAdjustPanelWidget(QWidget):
    """Floating adjustment card; drag anywhere except sliders / Reset.

  Slider tracking (UI vs preview):
  - ``valueChanged`` (tracking on): numeric readout updates immediately; preview is throttled.
  - Throttled ``preview_changed`` while dragging (image work runs off the GUI thread).
  - ``sliderReleased``: final preview + ``editing_finished`` (XMP write).
    """

    _PANEL_W = 280
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
    dodge_burn_mode_changed = pyqtSignal(object)
    dodge_burn_clear_requested = pyqtSignal()
    dodgeBurnMaskToggled = pyqtSignal(bool)

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
        self.setMaximumWidth(400)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        card = QWidget(self)
        card.setObjectName("adjust_panel_card")
        card.setStyleSheet(
            """
            QWidget#adjust_panel_card {
                background-color: #222222;
                border-left: 1px solid #333333;
            }
            QLabel#adjust_panel_title {
                color: #E0E0E0;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel.adjust_slider_label {
                color: #888888;
                font-size: 11px;
            }
            QLabel.adjust_slider_value {
                color: #E0E0E0;
                font-size: 11px;
                min-width: 44px;
            }
            QLabel.adjust_slider_value:hover {
                color: #3a8ac8;
            }
            QPushButton#adjust_reset_btn {
                color: #B0B0B0;
                font-size: 11px;
                border: none;
                background: transparent;
                padding: 2px 6px;
            }
            QPushButton#adjust_reset_btn:hover {
                color: #FFFFFF;
            }
            QPushButton#adjust_export_btn {
                color: #90CAF9;
                font-size: 11px;
                border: 1px solid rgba(144, 202, 249, 80);
                border-radius: 4px;
                background: rgba(144, 202, 249, 25);
                padding: 4px 10px;
            }
            QPushButton#adjust_export_btn:hover {
                background: rgba(144, 202, 249, 45);
                color: #FFFFFF;
            }
            QPushButton#adjust_export_btn:disabled {
                color: #606060;
                border-color: rgba(255, 255, 255, 20);
                background: transparent;
            }
            QPushButton#adjust_nr_btn {
                color: #B0B0B0;
                font-size: 11px;
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 4px;
                background: rgba(255, 255, 255, 12);
                padding: 4px 8px;
            }
            QPushButton#adjust_nr_btn:checked {
                color: #90CAF9;
                border-color: rgba(144, 202, 249, 90);
                background: rgba(144, 202, 249, 30);
            }
            QPushButton#adjust_nr_btn:hover {
                color: #FFFFFF;
            }
            QPushButton#adjust_wb_picker_btn {
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 4px;
                background: rgba(255, 255, 255, 12);
                padding: 0px;
            }
            QPushButton#adjust_wb_picker_btn:checked {
                border-color: rgba(144, 202, 249, 90);
                background: rgba(144, 202, 249, 30);
            }
            QPushButton#adjust_wb_picker_btn:hover {
                background: rgba(255, 255, 255, 24);
            }
            QPushButton#adjust_compare_btn {
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 4px;
                background: rgba(255, 255, 255, 12);
                padding: 0px;
            }
            QPushButton#adjust_compare_btn:checked {
                border-color: rgba(144, 202, 249, 90);
                background: rgba(144, 202, 249, 30);
            }
            QPushButton#adjust_compare_btn:hover {
                background: rgba(255, 255, 255, 24);
            }
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
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        card_layout.addWidget(scroll)

        inner = QWidget()
        inner.setMaximumWidth(self._PANEL_W)
        inner.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Adjust")
        title.setObjectName("adjust_panel_title")
        header.addWidget(title)
        header.addStretch(1)
        self._build_compare_button(header)
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("adjust_reset_btn")
        reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        reset_btn.clicked.connect(self._on_reset_clicked)
        header.addWidget(reset_btn)
        layout.addLayout(header)

        hint = QLabel("E — hide · drag sliders · click value to reset")
        hint.setStyleSheet("color: #808080; font-size: 10px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._tone_curve_row = None

        # Build Collapsible Sections
        self.histogram_widget = histogram_widget
        self.sect_histogram = CollapsibleSection("Histogram")
        if self.histogram_widget:
            # Add some styling wrapper or just add it directly
            # Set minimum height for the histogram
            self.histogram_widget.setMinimumHeight(120)
            self.sect_histogram.content_layout.addWidget(self.histogram_widget)
        else:
            self.sect_histogram.hide()

        self.sect_light = CollapsibleSection("Light")
        self.sect_color = CollapsibleSection("Color / WB")
        
        self.sect_curve = CollapsibleSection("Tone Curve")
        if not _SHOW_TONE_CURVE_UI:
            self.sect_curve.hide()
            
        self.sect_hsl = CollapsibleSection("HSL / Color Mixer")
        if not _SHOW_HSL_UI:
            self.sect_hsl.hide()
            
        self.sect_detail = CollapsibleSection("Detail / Correction")

        # Add Collapsible Sections to main scroll layout
        layout.addWidget(self.sect_histogram)
        layout.addWidget(self.sect_light)
        layout.addWidget(self.sect_color)
        layout.addWidget(self.sect_curve)
        layout.addWidget(self.sect_hsl)
        layout.addWidget(self.sect_detail)

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
                    background: #2A2A2A;
                    color: #B0B0B0;
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

            self._tone_curve_row = ToneCurveEditorRow()
            self._tone_curve_row.points_changed.connect(self._on_tone_curve_changed)
            self._tone_curve_row.editing_finished.connect(self._on_tone_curve_finished)
            self._tone_curve_point_layout.addWidget(self._tone_curve_row)

        # Build HSL mixer inside the HSL section
        if _SHOW_HSL_UI:
            self._build_hsl_section(self.sect_hsl.content_layout)

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
            elif spec.key in {"Sharpness", "Clarity2012", "Defringe", "LuminanceNoiseReduction"}:
                target_sect = self.sect_detail
            else:
                target_sect = None
                
            if target_sect is None and target_layout is None:
                continue

            row = QHBoxLayout()
            row.setSpacing(6)
            name_lbl = QLabel(spec.label)
            name_lbl.setProperty("class", "adjust_slider_label")
            name_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
            name_lbl.setMinimumWidth(72)
            row.addWidget(name_lbl)

            slider = AdjustSlider(Qt.Orientation.Horizontal)
            slider.setTracking(True)
            slider.setRange(spec.minimum, spec.maximum)
            slider.setSingleStep(spec.single_step)
            slider.setPageStep(max(spec.single_step, (spec.maximum - spec.minimum) // 20))
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            gradient = _SLIDER_TRACK_GRADIENTS.get(spec.key)
            if gradient is not None:
                slider.set_track_gradient(gradient)
            if spec.key == "Temperature":
                slider.set_center_value(self._as_shot_temperature)
                self._temperature_slider = slider
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
            row.addWidget(slider, 1)

            val_lbl = AdjustValueLabel()
            val_lbl.setProperty("class", "adjust_slider_value")
            val_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
            val_lbl.setMinimumWidth(32)
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

        # Chroma NR
        nr_row = QHBoxLayout()
        nr_row.setSpacing(6)
        nr_label = QLabel("Chroma NR")
        nr_label.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        nr_label.setMinimumWidth(72)
        nr_row.addWidget(nr_label)
        self._nr_btn = QPushButton("Off")
        self._nr_btn.setObjectName("adjust_nr_btn")
        self._nr_btn.setCheckable(True)
        self._nr_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._nr_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._nr_btn.setToolTip("Chroma-only bilateral denoise (luminance preserved)")
        self._nr_btn.toggled.connect(self._on_nr_toggled)
        nr_row.addWidget(self._nr_btn, 1)
        self.sect_detail.add_layout(nr_row)

        # Denoise Method Row
        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_lbl = QLabel("NR Method")
        method_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        method_lbl.setMinimumWidth(72)
        method_row.addWidget(method_lbl)
        
        self._denoise_method_combo = QComboBox()
        self._denoise_method_combo.addItems(["Bilateral", "Guided Filter"])
        self._denoise_method_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._denoise_method_combo.currentIndexChanged.connect(self._on_denoise_method_changed)
        self._denoise_method_combo.setStyleSheet("""
            QComboBox {
                background-color: #2D2D2D;
                border: 1px solid #555;
                border-radius: 3px;
                color: #DDD;
                font-size: 11px;
                padding: 2px 22px 2px 6px;
                selection-background-color: #4A4A4A;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #555;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background-color: #383838;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #AAA;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D2D;
                border: 1px solid #555;
                color: #DDD;
                selection-background-color: #4A6080;
                selection-color: #FFF;
                outline: none;
                font-size: 11px;
                padding: 2px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 22px;
                padding: 2px 6px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #3A4A5A;
                color: #FFF;
            }
        """)
        method_row.addWidget(self._denoise_method_combo, 1)
        self.sect_detail.add_layout(method_row)

        # Lens correction
        self._lens_correction_row = QHBoxLayout()
        self._lens_correction_row.setSpacing(6)
        
        lbl_vbox = QVBoxLayout()
        lbl_vbox.setSpacing(1)
        lbl_vbox.setContentsMargins(0, 0, 0, 0)
        
        lens_label = QLabel("Lens correction")
        lens_label.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        lbl_vbox.addWidget(lens_label)
        
        self._lens_profile_lbl = QLabel("")
        self._lens_profile_lbl.setStyleSheet("color: #808080; font-size: 10px;")
        self._lens_profile_lbl.setWordWrap(True)
        lbl_vbox.addWidget(self._lens_profile_lbl)
        
        lbl_container = QWidget()
        lbl_container.setLayout(lbl_vbox)
        lbl_container.setMinimumWidth(72)
        
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
        self.sect_detail.add_widget(self._lens_correction_row_widget)

        # Recovery-look button removed (2026-07): the editor's default
        # display transform now matches the browse render exactly (dcraw
        # BT.709 + highlight clip, see raw_edit_pipeline), which eliminated
        # the tonal gap the recovery baseline existed to bridge. The
        # pipeline still honors the flag for sessions/dicts that carry it
        # (set_recovery_baseline stays as a no-op-capable setter); nothing
        # in the UI can arm it anymore.
        self._recovery_btn = None

        # Dodge & burn brush: mutually-exclusive Dodge/Burn toggle + Size/
        # Strength sliders (transient tool settings, not persisted sliders --
        # see raw_dodge_burn.py; the mask itself is persisted separately).
        db_row = QHBoxLayout()
        db_row.setSpacing(6)
        db_label = QLabel("Dodge/Burn")
        db_label.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        db_label.setMinimumWidth(72)
        db_row.addWidget(db_label)
        self._dodge_btn = QPushButton("Dodge")
        self._burn_btn = QPushButton("Burn")
        for btn, tip in (
            (self._dodge_btn, "Brush to brighten (edge-snapped to the subject under the stroke)"),
            (self._burn_btn, "Brush to darken (edge-snapped to the subject under the stroke)"),
        ):
            btn.setObjectName("adjust_db_btn")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setToolTip(tip)
            db_row.addWidget(btn, 1)
        self._dodge_btn.toggled.connect(lambda on: self._on_dodge_burn_toggled(self._dodge_btn, on))
        self._burn_btn.toggled.connect(lambda on: self._on_dodge_burn_toggled(self._burn_btn, on))
        self._db_clear_btn = QPushButton("Clear")
        self._db_clear_btn.setObjectName("adjust_db_clear_btn")
        self._db_clear_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_clear_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._db_clear_btn.setToolTip("Erase the dodge/burn brush mask for this image")
        self._db_clear_btn.setEnabled(False)
        self._db_clear_btn.clicked.connect(self.dodge_burn_clear_requested.emit)
        db_row.addWidget(self._db_clear_btn)
        
        self._db_show_mask_btn = QPushButton("Show Mask (O)")
        self._db_show_mask_btn.setObjectName("adjust_db_show_mask_btn")
        self._db_show_mask_btn.setCheckable(True)
        self._db_show_mask_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_show_mask_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._db_show_mask_btn.setToolTip("Overlay mask in red")
        self._db_show_mask_btn.toggled.connect(self.dodgeBurnMaskToggled.emit)
        db_row.addWidget(self._db_show_mask_btn)
        
        self.sect_detail.add_layout(db_row)

        db_size_row = QHBoxLayout()
        db_size_row.setSpacing(6)
        db_size_lbl = QLabel("Brush Size")
        db_size_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        db_size_lbl.setMinimumWidth(72)
        db_size_row.addWidget(db_size_lbl)
        self._db_size_slider = AdjustSlider(Qt.Orientation.Horizontal)
        self._db_size_slider.setRange(8, 400)
        self._db_size_slider.setValue(60)
        self._db_size_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        db_size_row.addWidget(self._db_size_slider, 1)
        self.sect_detail.add_layout(db_size_row)

        db_strength_row = QHBoxLayout()
        db_strength_row.setSpacing(6)
        db_strength_lbl = QLabel("Brush Strength")
        db_strength_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        db_strength_lbl.setMinimumWidth(72)
        db_strength_row.addWidget(db_strength_lbl)
        self._db_strength_slider = AdjustSlider(Qt.Orientation.Horizontal)
        self._db_strength_slider.setRange(5, 100)
        self._db_strength_slider.setValue(35)
        self._db_strength_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        db_strength_row.addWidget(self._db_strength_slider, 1)
        self.sect_detail.add_layout(db_strength_row)

        export_btn = QPushButton("Export…")
        export_btn.setObjectName("adjust_export_btn")
        export_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        export_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        export_menu = QMenu(export_btn)
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
        self._compare_icon_default = _qta_icon_safe("fa5s.columns", color="#B0B0B0")
        self._compare_icon_active = _qta_icon_safe("fa5s.columns", color="#90CAF9")
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
        self._wb_picker_icon_default = _qta_icon_safe("fa5s.eye-dropper", color="#B0B0B0")
        self._wb_picker_icon_active = _qta_icon_safe("fa5s.eye-dropper", color="#90CAF9")
        btn.setIcon(self._wb_picker_icon_default)
        btn.toggled.connect(self._on_wb_picker_toggled)
        row.addWidget(btn)
        self._wb_picker_btn = btn

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

    def _build_hsl_section(self, layout: QVBoxLayout) -> None:
        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        color_lbl = QLabel("Color")
        color_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        color_lbl.setMinimumWidth(72)
        color_row.addWidget(color_lbl)
        self._hsl_color_combo = QComboBox()
        self._hsl_color_combo.setStyleSheet("""
            QComboBox {
                color: #E0E0E0;
                background-color: #2A2A2A;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 2px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
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
            name_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
            name_lbl.setMinimumWidth(72)
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
            self._set_nr_checked(float(merged.get("ColorNoiseReduction", 0.0)) > 0.0)
            self._set_lens_correction_checked(
                float(merged.get("LensCorrectionEnabled", 0.0)) > 0.5
            )
            if self._tone_curve_row is not None:
                self._tone_curve_row.load_serial(
                    str(merged.get(TONE_CURVE_SERIAL_KEY, "") or "")
                )
            for key in self._hsl_cache:
                self._hsl_cache[key] = float(merged.get(key, 0.0))
            if hasattr(self, "_hsl_sliders"):
                self._current_hsl_color = self._hsl_color_name()
                self._load_hsl_sliders_from_adj(self._hsl_cache)
            self.set_recovery_baseline(
                float(merged.get(RECOVERY_BASELINE_KEY, 0.0)) > 0.5
            )
            # Load denoise method combo index
            method_idx = int(float(merged.get("DenoiseMethod", 0.0)))
            if hasattr(self, "_denoise_method_combo"):
                self._denoise_method_combo.blockSignals(True)
                self._denoise_method_combo.setCurrentIndex(min(1, max(0, method_idx)))
                self._denoise_method_combo.blockSignals(False)
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
                    lbl.setStyleSheet("color: #90CAF9; font-size: 10px; font-weight: 500;")
                else:
                    lbl.setText(name)
                    lbl.setStyleSheet("color: #808080; font-size: 10px;")

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
        btn = getattr(self, "_nr_btn", None)
        out["ColorNoiseReduction"] = (
            CHROMA_NR_ON_VALUE if btn is not None and btn.isChecked() else 0.0
        )
        lens_btn = getattr(self, "_lens_correction_btn", None)
        out["LensCorrectionEnabled"] = (
            1.0 if lens_btn is not None and lens_btn.isChecked() else 0.0
        )
        if self._tone_curve_row is not None:
            serial = self._tone_curve_row.serialized_points()
            if serial:
                out[TONE_CURVE_SERIAL_KEY] = serial
        if hasattr(self, "_hsl_sliders"):
            self._flush_hsl_cache_from_sliders()
            for key, val in self._hsl_cache.items():
                out[key] = float(val)
        if self._recovery_baseline:
            out[RECOVERY_BASELINE_KEY] = 1.0
        if hasattr(self, "_denoise_method_combo"):
            out["DenoiseMethod"] = float(self._denoise_method_combo.currentIndex())
        return out

    def _on_nr_toggled(self, checked: bool) -> None:
        if self._block_emit:
            return
        btn = getattr(self, "_nr_btn", None)
        if btn is not None:
            btn.setText("On" if checked else "Off")
        self._emit_preview_and_save()

    def _on_dodge_burn_toggled(self, btn: QPushButton, checked: bool) -> None:
        if self._block_emit:
            return
        other = self._burn_btn if btn is self._dodge_btn else self._dodge_btn
        if checked and other.isChecked():
            self._block_emit = True
            try:
                other.setChecked(False)
            finally:
                self._block_emit = False
        mode = None
        if self._dodge_btn.isChecked():
            mode = "dodge"
        elif self._burn_btn.isChecked():
            mode = "burn"
        self.dodge_burn_mode_changed.emit(mode)

    def dodge_burn_mode(self) -> str | None:
        if self._dodge_btn.isChecked():
            return "dodge"
        if self._burn_btn.isChecked():
            return "burn"
        return None

    def disarm_dodge_burn(self) -> None:
        """Uncheck both tool buttons without emitting (e.g. on file switch)."""
        self._block_emit = True
        try:
            self._dodge_btn.setChecked(False)
            self._burn_btn.setChecked(False)
        finally:
            self._block_emit = False

    def dodge_burn_brush_radius(self) -> float:
        return float(self._db_size_slider.value())

    def dodge_burn_brush_strength(self) -> float:
        """0..1: per-stamp delta at the brush center before pressure scaling."""
        return float(self._db_strength_slider.value()) / 100.0

    def set_dodge_burn_mask_present(self, present: bool) -> None:
        self._db_clear_btn.setEnabled(bool(present))

    def _update_slider_label(self, key: str, fmt: Callable[[float], str]) -> None:
        slider = self._sliders.get(key)
        val_lbl = self._value_labels.get(key)
        if slider is None:
            return
        spec = next(s for s in SLIDER_SPECS if s.key == key)
        text = fmt(spec.slider_to_value(slider.value()))
        if val_lbl is not None:
            val_lbl.setText(text)

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
        self._emit_preview_and_save()

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
        self._set_nr_checked(False)
        self._current_hsl_color = HSL_COLOR_NAMES[0]
        if hasattr(self, "_hsl_color_combo"):
            self._hsl_color_combo.blockSignals(True)
            self._hsl_color_combo.setCurrentIndex(0)
            self._hsl_color_combo.blockSignals(False)
        if self._tone_curve_row is not None:
            self._tone_curve_row.reset_linear()
        if hasattr(self, "_denoise_method_combo"):
            self._denoise_method_combo.blockSignals(True)
            self._denoise_method_combo.setCurrentIndex(0)
            self._denoise_method_combo.blockSignals(False)
        self._clear_recovery_baseline()
        self.reset_requested.emit()
        self._emit_preview_and_save()

    def _on_export_clicked(self) -> None:
        self._request_export("tiff16")

    def set_export_enabled(self, enabled: bool) -> None:
        btn = getattr(self, "_export_btn", None)
        if btn is not None:
            btn.setEnabled(bool(enabled))



