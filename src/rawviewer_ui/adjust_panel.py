"""Draggable RAW adjustment panel for single-image view (E to toggle)."""

from __future__ import annotations

from typing import Callable, Dict

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor, QIcon
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

# Point-curve + parametric PV rows — hidden while core adjust is validated (see docs).
_SHOW_TONE_CURVE_UI = False
# HSL 分色 section — hidden pending a saturation/vibrance review (see docs).
_SHOW_HSL_UI = False

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


class AdjustSlider(QSlider):
    """Horizontal slider with a taller hit target (Qt handles track click-to-set)."""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setMinimumHeight(22)


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sliders: Dict[str, QSlider] = {}
        self._value_labels: Dict[str, QLabel] = {}
        self._drag_press_global = None
        self._drag_pos_at_press = None
        self._block_emit = False
        self._as_shot_temperature = float(DEFAULT_ADJUSTMENTS["Temperature"])
        self._recovery_baseline = False
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

        self.setFixedSize(self._PANEL_W, self._PANEL_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(0)

        card = QWidget(self)
        card.setObjectName("adjust_panel_card")
        card.setStyleSheet(
            """
            QWidget#adjust_panel_card {
                background-color: rgba(30, 30, 30, 215);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 8px;
            }
            QLabel#adjust_panel_title {
                color: #E0E0E0;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel.adjust_slider_label {
                color: #B0B0B0;
                font-size: 11px;
            }
            QLabel.adjust_slider_value {
                color: #90CAF9;
                font-size: 11px;
                min-width: 44px;
            }
            QLabel.adjust_slider_value:hover {
                color: #BBDEFB;
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
            QSlider::groove:horizontal {
                height: 4px;
                background: #3A3A3A;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin: -5px 0;
                background: #90CAF9;
                border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #5C9FD4;
                border-radius: 2px;
            }
            """
        )
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
        _SECTION_KEYS = {
            "ParametricShadows": "Tone curve",
            "LuminanceNoiseReduction": "Noise",
        }

        for spec in SLIDER_SPECS:
            if not _SHOW_TONE_CURVE_UI and spec.key in _PARAMETRIC_TONE_KEYS:
                continue
            if spec.key in _SECTION_KEYS:
                sec = QLabel(_SECTION_KEYS[spec.key])
                sec.setStyleSheet("color: #909090; font-size: 10px; font-weight: 600; margin-top: 4px;")
                layout.addWidget(sec)
            if _SHOW_TONE_CURVE_UI and spec.key == "ParametricShadows":
                self._tone_curve_row = ToneCurveEditorRow()
                self._tone_curve_row.points_changed.connect(self._on_tone_curve_changed)
                self._tone_curve_row.editing_finished.connect(self._on_tone_curve_finished)
                layout.addWidget(self._tone_curve_row)
            if _SHOW_HSL_UI and spec.key == "Sharpness":
                self._build_hsl_section(layout)
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
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setToolTip("Click to reset")
            val_lbl.clicked.connect(lambda k=spec.key: self._reset_slider(k))
            row.addWidget(val_lbl)

            if spec.key == "Temperature":
                self._build_wb_picker_button(row)

            layout.addLayout(row)
            self._sliders[spec.key] = slider
            self._value_labels[spec.key] = val_lbl

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
        layout.addLayout(nr_row)

        self._recovery_btn = QPushButton("Use recovery look")
        self._recovery_btn.setObjectName("adjust_recovery_btn")
        self._recovery_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._recovery_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._recovery_btn.setToolTip(
            "Use P-key recovery tone as the adjust starting point. "
            "Turns off P recovery preview if active."
        )
        self._recovery_btn.clicked.connect(self._on_recovery_baseline_clicked)
        layout.addWidget(self._recovery_btn)

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
        export_menu.addSeparator()
        export_menu.addAction(
            "DNG — copy RAW + XMP settings",
            lambda: self._request_export("dng_settings"),
        )
        export_menu.addAction(
            "DNG — baked 16-bit RGB",
            lambda: self._request_export("dng_rgb"),
        )
        export_btn.setMenu(export_menu)
        export_btn.setToolTip("Export baked image or non-destructive RAW + XMP for other editors")
        layout.addWidget(export_btn)
        self._export_btn = export_btn

        layout.addStretch(1)
        self.set_adjustments(dict(DEFAULT_ADJUSTMENTS))

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
        sec = QLabel("HSL")
        sec.setStyleSheet("color: #909090; font-size: 10px; font-weight: 600; margin-top: 4px;")
        layout.addWidget(sec)

        color_row = QHBoxLayout()
        color_row.setSpacing(6)
        color_lbl = QLabel("Color")
        color_lbl.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        color_lbl.setMinimumWidth(72)
        color_row.addWidget(color_lbl)
        self._hsl_color_combo = QComboBox()
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

    def _flush_hsl_cache_from_sliders(self) -> None:
        if not hasattr(self, "_hsl_sliders"):
            return
        color = self._hsl_color_name()
        for channel in ("Hue", "Saturation", "Luminance"):
            slider = self._hsl_sliders.get(channel)
            if slider is None:
                continue
            self._hsl_cache[self._hsl_key(channel, color)] = float(slider.value())

    def _on_hsl_color_changed(self, _index: int) -> None:
        if self._block_emit:
            return
        self._flush_hsl_cache_from_sliders()
        self._block_emit = True
        try:
            self._load_hsl_sliders_from_adj(self._hsl_cache)
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
        self._block_emit = True
        try:
            merged = dict(DEFAULT_ADJUSTMENTS)
            merged.update(adj or {})
            self._as_shot_temperature = float(
                merged.get(AS_SHOT_TEMP_KEY, DEFAULT_ADJUSTMENTS["Temperature"])
            )
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
            if self._tone_curve_row is not None:
                self._tone_curve_row.load_serial(
                    str(merged.get(TONE_CURVE_SERIAL_KEY, "") or "")
                )
            for key in self._hsl_cache:
                self._hsl_cache[key] = float(merged.get(key, 0.0))
            if hasattr(self, "_hsl_sliders"):
                self._load_hsl_sliders_from_adj(self._hsl_cache)
            self.set_recovery_baseline(
                float(merged.get(RECOVERY_BASELINE_KEY, 0.0)) > 0.5
            )
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
        return out

    def _on_nr_toggled(self, checked: bool) -> None:
        if self._block_emit:
            return
        btn = getattr(self, "_nr_btn", None)
        if btn is not None:
            btn.setText("On" if checked else "Off")
        self._emit_preview_and_save()

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
        if self._tone_curve_row is not None:
            self._tone_curve_row.reset_linear()
        self._clear_recovery_baseline()
        self.reset_requested.emit()
        self._emit_preview_and_save()

    def _on_export_clicked(self) -> None:
        self._request_export("tiff16")

    def set_export_enabled(self, enabled: bool) -> None:
        btn = getattr(self, "_export_btn", None)
        if btn is not None:
            btn.setEnabled(bool(enabled))

    def _pointer_on_interactive_child(self, global_pos) -> bool:
        """True when the cursor is over a slider, button, or value label."""
        w = QApplication.widgetAt(global_pos.toPoint())
        interactive_types: tuple = (
            QSlider,
            AdjustSlider,
            QPushButton,
            AdjustValueLabel,
            QComboBox,
        )
        if _SHOW_TONE_CURVE_UI:
            interactive_types = interactive_types + (ToneCurveEditorRow, ToneCurveWidget)
        while w is not None and w is not self:
            if isinstance(w, interactive_types):
                return True
            w = w.parentWidget()
        return False

    def enterEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._pointer_on_interactive_child(event.globalPosition()):
                super().mousePressEvent(event)
                return
            self._drag_press_global = event.globalPosition().toPoint()
            self._drag_pos_at_press = self.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_press_global is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            if self._pointer_on_interactive_child(event.globalPosition()):
                self._drag_press_global = None
                self._drag_pos_at_press = None
                super().mouseMoveEvent(event)
                return
            parent = self.parentWidget()
            if parent:
                d = event.globalPosition().toPoint() - self._drag_press_global
                nx = self._drag_pos_at_press.x() + d.x()
                ny = self._drag_pos_at_press.y() + d.y()
                nx = max(0, min(nx, parent.width() - self.width()))
                ny = max(0, min(ny, parent.height() - self.height()))
                self.move(nx, ny)
                if hasattr(parent, "mark_adjust_user_moved"):
                    parent.mark_adjust_user_moved()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_press_global = None
            self._drag_pos_at_press = None
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().mouseReleaseEvent(event)
