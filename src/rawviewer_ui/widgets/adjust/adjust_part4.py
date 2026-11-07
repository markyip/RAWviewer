"""Adjust mixin part 4 of 5 for ImageAdjustPanelWidget."""
from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from core import theme
from raw_adjustments import (
    AS_SHOT_TEMP_KEY,
    CHROMA_NR_ON_VALUE,
    DEFAULT_ADJUSTMENTS,
    RECOVERY_BASELINE_KEY,
    SLIDER_SPECS,
    is_pv2012_tone_slider,
    recovery_baseline_slider_hints,
)
from raw_adjustments.dodge_burn import DEFAULT_STRENGTH as DB_DEFAULT_STRENGTH
from raw_adjustments.dodge_burn import STRENGTH_KEY as DB_STRENGTH_KEY
from raw_adjustments.hsl import HSL_COLOR_NAMES

from .helpers import _CHANNEL_CURVE_KEYS_BY_NAME, CollapsibleSection, _qta_icon_safe, normalize_adjust_dock


class AdjustPart4:

    def get_adjustments(self) -> dict[str, float]:
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
        left_ins, r, t, b = self.get_crop_insets()
        out["CropLeft"] = float(left_ins)
        out["CropRight"] = float(r)
        out["CropTop"] = float(t)
        out["CropBottom"] = float(b)
        out["AnamorphicRatio"] = float(getattr(self, "_anamorphic_ratio", 1.0) or 1.0)
        from raw_adjustments.lut import LUT_AMOUNT_KEY, LUT_NAME_KEY, LUT_SPACE_KEY, LUT_SPACE_REC709

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
        """Mask overlay is available while a brush tool is armed, and stays
        available afterward if a previously-painted dodge/burn/heal mask
        still has data (e.g. a loaded edit) so the user can inspect it
        without re-arming a tool.

        Default on when arming a tool so paint coverage is visible immediately
        (especially important for Heal).
        """
        btn = getattr(self, "_db_show_mask_btn", None)
        if btn is None:
            return
        has_data = bool(
            getattr(self, "_db_mask_has_data", False)
            or getattr(self, "_heal_mask_has_data", False)
        )
        available = bool(armed or has_data)
        btn.setEnabled(available)
        # setChecked below fires the button's own `toggled` signal, which
        # _on_mask_btn_toggled already forwards to dodgeBurnMaskToggled --
        # no need (and no longer any need) to emit it again here.
        if not available:
            if btn.isChecked():
                self._block_emit = True
                try:
                    btn.setChecked(False)
                finally:
                    self._block_emit = False
            return
        if armed and not btn.isChecked() and not self._mask_user_hidden:
            self._block_emit = True
            try:
                btn.setChecked(True)
            finally:
                self._block_emit = False

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
        # Same mode again → toggle off.
        if want and cur_dodge == dodge_on and cur_burn == burn_on and cur_erase == erase_on and cur_heal == heal_on:
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
        amount = (dy // 120) * 8 if abs(dy) >= 120 else int(round(dy / 4.0))
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
        self._db_mask_has_data = bool(present)
        self._sync_dodge_burn_mask_button_enabled(self.dodge_burn_mode() is not None)

    def set_spot_heal_mask_present(self, present: bool) -> None:
        btn = getattr(self, "_heal_clear_btn", None)
        if btn is not None:
            btn.setEnabled(bool(present))
        self._heal_mask_has_data = bool(present)
        self._sync_dodge_burn_mask_button_enabled(self.dodge_burn_mode() is not None)

    def _on_mask_btn_toggled(self, checked: bool) -> None:
        """Track explicit user clicks (not programmatic setChecked calls)
        so arming a brush tool doesn't force Mask back on after the user
        deliberately turned it off."""
        if not self._block_emit:
            self._mask_user_hidden = not checked
        self.dodgeBurnMaskToggled.emit(bool(checked))

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

    def ai_denoise_enabled(self) -> bool:
        """Used by main.py to decide whether AI denoise preview is active."""
        btn = getattr(self, "_ai_denoise_check", None)
        return bool(btn is not None and btn.isEnabled() and btn.isChecked())

    def set_ai_denoise_enabled(self, enabled: bool) -> None:
        """Mirror an external setting (e.g. after load / restore) to the toggle."""
        btn = getattr(self, "_ai_denoise_check", None)
        if btn is not None and btn.isChecked() != bool(enabled):
            btn.setChecked(bool(enabled))

    def _sync_ai_preview_btn(self) -> None:
        if self._ai_busy:
            return
        self._ai_preview_btn.setEnabled(self.ai_denoise_enabled())

    def set_ai_busy(self, busy: bool, progress: float | None = None) -> None:
        """Show or clear the AI denoise running state on the Preview AI button."""
        btn = self._ai_preview_btn
        self._ai_busy = bool(busy)
        if busy:
            pct = int(round(max(0.0, min(1.0, progress or 0.0)) * 100))
            btn.setText(f"Denoising… {pct}%")
            btn.setToolTip("AI denoise is running at full resolution. The button returns when it finishes.")
            btn.setEnabled(False)
        else:
            btn.setText("Preview AI")
            btn.setToolTip(
                "Show the AI denoise result at full quality. Takes a while on large files; "
                "slider changes show the normal preview again."
            )
            self._sync_ai_preview_btn()
        btn.setProperty("aiBusy", "true" if busy else "false")
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        # Export shares the AI model session with this preview: hold it until done.
        export_btn = getattr(self, "_export_btn", None)
        if export_btn is not None:
            export_btn.setEnabled(not busy)

    def _on_ai_denoise_toggled(self, checked: bool) -> None:
        self._sync_ai_preview_btn()
        self.ai_denoise_toggled.emit(bool(checked))
        # Show the regular look again; "Preview AI" renders the AI result.
        self._emit_preview_and_save()

    def _request_export(self, export_format: str) -> None:
        self.export_requested.emit(
            export_format, self.get_adjustments(), self.ai_denoise_enabled()
        )

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
        if index > 0 and hasattr(self, "_chroma_nr_amount_slider") and self._chroma_nr_amount_slider.value() < 1:
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

    def _build_header_dock_button(self, row: QHBoxLayout) -> None:
        """Move the panel to the other splitter edge (left ↔ right)."""
        btn = QPushButton()
        btn.setObjectName("adjust_compare_btn")
        btn.setFixedSize(22, 22)
        btn.setIconSize(QSize(13, 13))
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.clicked.connect(self._on_dock_button_clicked)
        row.addWidget(btn)
        self._header_dock_btn = btn
        self.set_dock_side(self._dock_side)

    def dock_side(self) -> str:
        return normalize_adjust_dock(self._dock_side)

    def set_dock_side(self, side: str) -> None:
        """Update the header affordance; does not persist or relayout."""
        self._dock_side = normalize_adjust_dock(side)
        btn = getattr(self, "_header_dock_btn", None)
        if btn is None:
            return
        if self._dock_side == "left":
            btn.setToolTip("Move Adjust panel to the right")
            icon = _qta_icon_safe("fa5s.arrow-right", color=theme.INK_MUTED)
            fallback = "→"
        else:
            btn.setToolTip("Move Adjust panel to the left")
            icon = _qta_icon_safe("fa5s.arrow-left", color=theme.INK_MUTED)
            fallback = "←"
        btn.setText("")
        if icon is not None and not icon.isNull():
            btn.setIcon(icon)
        else:
            btn.setText(fallback)

    def _on_dock_button_clicked(self) -> None:
        target = "left" if self.dock_side() == "right" else "right"
        self.dock_side_requested.emit(target)

    def _build_calibrate_button(self, sect: CollapsibleSection) -> None:
        btn = QPushButton("Calibrate from ColorChecker…")
        btn.setObjectName("adjust_nr_btn")
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(
            "Shoot or load a 24-patch ColorChecker and save a camera colour profile.\n"
            "Applied automatically to later photos from this body that have no XMP."
        )
        btn.clicked.connect(self.calibrate_requested.emit)
        sect.add_widget(btn)
        self._calibrate_btn = btn

    def _build_camera_profile_banner(self, parent_layout) -> None:
        row = QWidget()
        row.setVisible(False)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(6)
        lbl = QLabel("")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.EMBER}; font-size: 11px;")
        h.addWidget(lbl, 1)
        remove = QPushButton("Remove")
        remove.setObjectName("adjust_reset_btn")
        remove.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        remove.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        remove.setToolTip("Delete this camera's saved ColorChecker profile")
        remove.clicked.connect(self._on_camera_profile_remove_clicked)
        h.addWidget(remove)
        parent_layout.addWidget(row)
        self._camera_profile_row = row
        self._camera_profile_lbl = lbl

    def set_camera_profile_active(
        self, label: str = "", make: str = "", model: str = "", iso: int | None = None
    ) -> None:
        self._camera_profile_identity = (make, model, iso)
        row = getattr(self, "_camera_profile_row", None)
        lbl = getattr(self, "_camera_profile_lbl", None)
        if lbl is not None:
            lbl.setText(f"Camera profile applied — {label}" if label else "")
        if row is not None:
            row.setVisible(bool(label))

    def _on_camera_profile_remove_clicked(self) -> None:
        make, model, iso = getattr(self, "_camera_profile_identity", ("", "", None))
        if not (make or model):
            return
        from PyQt6.QtWidgets import QDialog

        try:
            from rawviewer_app.core.widgets import CustomConfirmDialog
        except ModuleNotFoundError:
            from widgets import CustomConfirmDialog

        iso_bit = f" (ISO {iso})" if iso else ""
        dlg = CustomConfirmDialog(
            self,
            title="Remove Camera Profile",
            message=f"Remove the saved calibration profile for {make} {model}{iso_bit}?",
            informative_text=(
                "Photos from this camera will return to the LibRaw baseline "
                "colour. Edits already saved in an XMP sidecar are unchanged."
            ),
            confirm_label="Remove",
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            from raw_adjustments.color_calibration import delete_camera_profile

            removed = delete_camera_profile(make, model, iso=iso)
            if not removed and iso is not None:
                removed = delete_camera_profile(make, model, iso=None)
        except Exception:
            removed = False
        if removed:
            self.set_camera_profile_active("")
            self.camera_profile_removed.emit()
