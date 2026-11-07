"""Adjust mixin part 6 of 6 for ImageAdjustPanelWidget."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from core import theme

from .helpers import (
    _SHOW_DODGE_BURN_UI,
    _SHOW_HSL_UI,
    _SHOW_TONE_CURVE_UI,
    CollapsibleSection,
    _edit_settings_clipboard,
)


class AdjustPart6:
    """Panel UI construction for the image adjustment panel."""

    def _build_ui(self, histogram_widget) -> None:
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
        self._build_header_dock_button(header)
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
        self._build_camera_profile_banner(layout)

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
        self._build_ui_rest(layout)
