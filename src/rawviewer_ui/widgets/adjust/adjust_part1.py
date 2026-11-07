"""Adjust mixin part 1 of 5 for ImageAdjustPanelWidget."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core import theme
from raw_adjustments import CHROMA_NR_ON_VALUE, DEFAULT_ADJUSTMENTS, SLIDER_SPECS
from raw_adjustments.dodge_burn import DEFAULT_STRENGTH as DB_DEFAULT_STRENGTH

from .helpers import (
    _PARAMETRIC_TONE_KEYS,
    _SHOW_HSL_UI,
    _SHOW_TONE_CURVE_UI,
    _SLIDER_TRACK_GRADIENTS,
    _TRANSFORM_SLIDER_KEYS,
    AdjustSlider,
    AdjustValueLabel,
    _adjust_combo_stylesheet,
    _sanitize_section_order,
)


class AdjustPart1:

    _PANEL_W = 440

    _PANEL_H = 520

    _PREVIEW_THROTTLE_MS = 80

    editing_finished = pyqtSignal(dict)

    preview_changed = pyqtSignal(dict)

    reset_requested = pyqtSignal()

    export_requested = pyqtSignal(str, dict, bool)  # format id, adjustments, use_ai_denoise

    ai_denoise_toggled = pyqtSignal(bool)

    ai_preview_requested = pyqtSignal()

    recovery_baseline_requested = pyqtSignal()

    wb_picker_toggled = pyqtSignal(bool)  # True: arm the WB dropper; False: cancel

    compare_toggled = pyqtSignal(bool)  # True: show compare-with-original split view

    lens_correction_toggled = pyqtSignal(bool)

    transform_interaction = pyqtSignal(bool)

    auto_wb_requested = pyqtSignal()

    auto_straighten_requested = pyqtSignal()

    auto_tone_requested = pyqtSignal()

    dock_side_requested = pyqtSignal(str)

    dodge_burn_mode_changed = pyqtSignal(object)

    dodge_burn_clear_requested = pyqtSignal()

    spot_heal_clear_requested = pyqtSignal()

    dodgeBurnMaskToggled = pyqtSignal(bool)

    dodge_burn_brush_changed = pyqtSignal()  # size/flow changed — refresh brush cursor

    xmp_preset_applied = pyqtSignal()

    looks_drop_rejected = pyqtSignal(str)

    crop_mode_changed = pyqtSignal(bool)

    crop_aspect_changed = pyqtSignal(object)  # float|None

    crop_apply_requested = pyqtSignal()

    crop_cancel_requested = pyqtSignal()

    crop_reset_requested = pyqtSignal()

    calibrate_requested = pyqtSignal()

    camera_profile_removed = pyqtSignal()

    def _build_ui_rest(self, layout) -> None:

        self._global_layout = layout
        self._global_sections = {
            "light": self.sect_light,
            "color": self.sect_color,
            "curve": self.sect_curve,
            "hsl": self.sect_hsl,
            "detail": self.sect_detail,
            "noise": self.sect_noise,
            "effects": self.sect_effects,
            "local": self.sect_local,
            "lut": self.sect_lut,
            "transform": self.sect_transform,
        }
        self._section_order = _sanitize_section_order(self._load_section_order())
        for key, sect in self._global_sections.items():
            sect.header.customContextMenuRequested.connect(
                lambda _pos, k=key: self._on_section_context_menu(k)
            )

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
            self._channel_curve_cache: dict[str, str] = {
                "RGB": "",
                "R": "",
                "G": "",
                "B": "",
            }
            self._current_curve_channel = "RGB"
            self._channel_btns: dict[str, QPushButton] = {}

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

            from rawviewer_ui.widgets.tone_curve_widget import ToneCurveEditorRow

            self._tone_curve_row = ToneCurveEditorRow()
            self._tone_curve_row.points_changed.connect(self._on_tone_curve_changed)
            self._tone_curve_row.editing_finished.connect(self._on_tone_curve_finished)
            self._tone_curve_point_layout.addWidget(self._tone_curve_row)

        # Build HSL mixer inside the HSL section
        if _SHOW_HSL_UI:
            self._build_hsl_section(self.sect_hsl.content_layout)

        # WB situation presets (Daylight/Cloudy/…) sit above Temp/Tint.
        self._build_wb_preset_row(self.sect_color)
        self._build_calibrate_button(self.sect_color)
        self._build_auto_tone_row(self.sect_light)

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
            elif spec.key in {"DenoiseDetail", "AIDenoiseAmount"}:
                try:
                    from core.profile import ai_export_enabled

                    if not ai_export_enabled():
                        continue
                except Exception:
                    continue
                target_sect = self.sect_noise
            elif spec.key in {
                "CropAngle", "PerspectiveVertical", "PerspectiveHorizontal",
                "Distortion",
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
            if spec.key == "DenoiseDetail":
                slider.setToolTip(
                    "How much original detail to keep after AI denoise.\n"
                    "0 = smoother, 100 = more detail (and more residual noise)."
                )
            if spec.key == "AIDenoiseAmount":
                slider.setToolTip(
                    "How strongly AI removes noise.\n"
                    "0 = no AI denoise, 100 = full; 10-30 for gentle cleanup. "
                    "Works with the 'AI denoise' toggle above."
                )
            if spec.key == "Distortion":
                slider.setToolTip(
                    "Manual lens distortion: + pincushion, − barrel "
                    "(use when no matched lens profile exists)"
                )
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

        # AI denoise: the checkbox turns it on for export; "Preview AI" renders
        # the AI look at full quality on demand (slow, so never on slider moves).
        ai_row = QHBoxLayout()
        ai_row.setSpacing(6)
        self._ai_denoise_check = QCheckBox("AI denoise")
        self._ai_denoise_check.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._ai_denoise_check.setStyleSheet(f"QCheckBox {{ color: {theme.INK}; font-size: 12px; }}")
        self._ai_denoise_check.toggled.connect(self._on_ai_denoise_toggled)
        ai_row.addWidget(self._ai_denoise_check, 1)
        self._ai_preview_btn = QPushButton("Preview AI")
        self._ai_preview_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._ai_preview_btn.setToolTip(
            "Show the AI denoise result at full quality. Takes a while on large files; "
            "slider changes show the normal preview again."
        )
        self._ai_preview_btn.clicked.connect(self.ai_preview_requested.emit)
        self._ai_preview_btn.setStyleSheet(
            f"QPushButton[aiBusy=\"true\"] {{ background: {theme.EMBER}; color: {theme.VOID}; "
            f"font-weight: 600; }}"
        )
        self._ai_busy = False
        ai_row.addWidget(self._ai_preview_btn)
        ai_widget = QWidget()
        ai_widget.setLayout(ai_row)
        self.sect_noise.add_widget(ai_widget)
        self._ai_denoise_row = ai_widget
        try:
            from core.profile import ai_export_enabled
            from raw_adjustments.pipeline import ai_denoise_model_present

            ai_ok = bool(ai_export_enabled() and ai_denoise_model_present())
        except Exception:
            ai_ok = False
        self._ai_denoise_check.setEnabled(ai_ok)
        ai_widget.setVisible(ai_ok)
        if not ai_ok:
            self._ai_denoise_check.setToolTip("AI denoise model is not installed.")
        self._sync_ai_preview_btn()

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
        self._build_anamorphic_row(self.sect_transform)

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

        self._db_show_mask_btn = QPushButton("Mask (M)")
        self._db_show_mask_btn.setObjectName("adjust_db_show_mask_btn")
        self._db_show_mask_btn.setCheckable(True)
        self._db_show_mask_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._db_show_mask_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._db_show_mask_btn.setEnabled(False)
        self._db_show_mask_btn.setToolTip(
            "Overlay the active brush mask (red/blue = dodge/burn, green = heal).\n"
            "Turns on automatically when a brush tool is armed. Shortcut: M"
        )
        self._db_show_mask_btn.toggled.connect(self._on_mask_btn_toggled)
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
        # Plain-language export options. AI denoise is controlled by the
        # "AI denoise" toggle in the Noise section (strength from AI Amount);
        # these buttons just pick the output file type.
        export_menu.addAction(
            "JPEG — smaller files, easy to share",
            lambda: self._request_export("jpeg"),
        )
        export_menu.addAction(
            "TIFF 16-bit — best quality, very large files",
            lambda: self._request_export("tiff16"),
        )
        export_menu.addAction(
            "WebP — small files, good for the web",
            lambda: self._request_export("webp"),
        )
        export_btn.setMenu(export_menu)
        tt = "Export the edited image."
        if self._ai_denoise_check.isEnabled():
            tt += (
                "\n\nAI denoise: use the 'AI denoise' toggle in the Noise "
                "section (strength from the AI Amount slider) to remove "
                "high-ISO grain before saving."
            )
        export_btn.setToolTip(tt)
        layout.addWidget(export_btn)
        self._export_btn = export_btn

        layout.addStretch(1)
        self._apply_section_order()
        self.set_adjustments(dict(DEFAULT_ADJUSTMENTS))


