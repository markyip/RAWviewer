"""Adjust mixin part 2 of 5 for ImageAdjustPanelWidget."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core import theme
from raw_adjustments import DEFAULT_ADJUSTMENTS, SLIDER_SPECS, WB_PRESETS

from .helpers import CollapsibleSection, _adjust_combo_stylesheet, _qta_icon_safe


class AdjustPart2:
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
        elif cur_data is not None and cur_data != -1.0 and abs(temperature - float(cur_data)) <= 25.0:
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
