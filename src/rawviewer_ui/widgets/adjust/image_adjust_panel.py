"""ImageAdjustPanelWidget: image adjustment panel, assembled from mixin parts."""
from __future__ import annotations

from collections import OrderedDict

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QLabel, QSlider, QWidget

from core import theme
from raw_adjustments import DEFAULT_ADJUSTMENTS
from raw_adjustments.hsl import HSL_COLOR_NAMES

from .adjust_part1 import AdjustPart1
from .adjust_part2 import AdjustPart2
from .adjust_part3 import AdjustPart3
from .adjust_part4 import AdjustPart4
from .adjust_part5 import AdjustPart5
from .adjust_part6 import AdjustPart6
from .helpers import (
    ADJUST_DOCK_DEFAULT,
    _LutThumbSignals,
)


class ImageAdjustPanelWidget(AdjustPart1, AdjustPart2, AdjustPart3, AdjustPart4, AdjustPart5, AdjustPart6, QWidget):
    """Image adjustment panel widget."""

    def __init__(self, parent=None, histogram_widget=None):
        super().__init__(parent)
        self._sliders: dict[str, QSlider] = {}
        self._value_labels: dict[str, QLabel] = {}
        self._block_emit = False
        self._mask_user_hidden = False
        self._as_shot_temperature = float(DEFAULT_ADJUSTMENTS["Temperature"])
        self._recovery_baseline = False
        self._current_hsl_color = HSL_COLOR_NAMES[0]
        self._hsl_cache: dict[str, float] = {
            k: float(v)
            for k, v in DEFAULT_ADJUSTMENTS.items()
            if k.startswith("HueAdjustment")
            or k.startswith("SaturationAdjustment")
            or k.startswith("LuminanceAdjustment")
        }
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._emit_live_preview)
        self._dock_side = ADJUST_DOCK_DEFAULT
        self._anamorphic_ratio = 1.0
        self._camera_profile_identity = ("", "", None)
        self._lut_thumb_cache: OrderedDict = OrderedDict()
        self._lut_thumb_rows: dict = {}
        self._lut_thumb_token = 0
        self._lut_thumb_display_u8 = None
        self._lut_thumb_path = ""
        self._lut_thumb_source_gen = 0
        self._lut_thumb_signals = _LutThumbSignals()
        self._lut_thumb_signals.finished.connect(self._on_lut_thumbs_ready)
        self._lut_thumb_timer = QTimer(self)
        self._lut_thumb_timer.setSingleShot(True)
        self._lut_thumb_timer.timeout.connect(self._start_lut_thumb_worker)

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
        self._build_ui(histogram_widget)
