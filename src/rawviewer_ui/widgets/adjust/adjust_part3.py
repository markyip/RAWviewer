"""Adjust mixin part 3 of 5 for ImageAdjustPanelWidget."""
from __future__ import annotations

import os

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)

from core import theme
from raw_adjustments import (
    AS_SHOT_TEMP_KEY,
    CHROMA_NR_ON_VALUE,
    DEFAULT_ADJUSTMENTS,
    RECOVERY_BASELINE_KEY,
    SLIDER_SPECS,
)
from raw_adjustments.dodge_burn import DEFAULT_STRENGTH as DB_DEFAULT_STRENGTH
from raw_adjustments.dodge_burn import STRENGTH_KEY as DB_STRENGTH_KEY
from raw_adjustments.hsl import HSL_COLOR_NAMES

from .helpers import (
    _CHANNEL_CURVE_KEYS_BY_NAME,
    _LOOKS_NAME_ROLE,
    AdjustSlider,
    AdjustValueLabel,
    CollapsibleSection,
    _adjust_combo_stylesheet,
    _edit_settings_clipboard,
    _FileDropFrame,
    _LooksRowWidget,
)


class AdjustPart3:

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
        self._looks_list.setMaximumHeight(220)
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
        from raw_adjustments.lut import LUT_SPACE_LINEAR, LUT_SPACE_REC709

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
        from raw_adjustments.lut import list_managed_luts
        from raw_adjustments.xmp_presets import list_managed_presets

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
            self._lut_thumb_rows = {}
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
                if kind == "cube":
                    self._lut_thumb_rows[name] = row
                # Prefer the just-applied .xmp highlight over an embedded .cube
                # so click-to-apply feedback stays on the preset row.
                if kind == "xmp" and wanted_xmp and name == wanted_xmp or (
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
        self._apply_cached_lut_thumbs()
        self._schedule_lut_thumbs()

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
        from raw_adjustments.lut import import_cube_file
        from raw_adjustments.xmp_presets import import_xmp_preset

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
        from raw_adjustments.lut import remove_managed_lut
        from raw_adjustments.xmp_presets import remove_managed_preset

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
        self._schedule_lut_thumbs()
        self._emit_preview_and_save()

    def _apply_xmp_preset_named(self, name: str) -> None:
        """One-shot: load managed XMP look onto this image and persist."""
        if not name:
            return
        try:
            from raw_adjustments.dodge_burn import MASK_KEY
            from raw_adjustments.xmp_presets import load_preset_adjustments

            preset = load_preset_adjustments(name)
        except Exception as exc:
            print(f"[XMP_PRESET] apply failed: {name}: {exc}")
            return
        current = self.get_adjustments()
        as_shot = current.get(AS_SHOT_TEMP_KEY)
        # Local paint stays per-image (same as copy/paste settings).
        preset.pop(MASK_KEY, None)
        try:
            from raw_adjustments.spot_heal import MASK_KEY as HEAL_MASK_KEY

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

        self._hsl_sliders: dict[str, QSlider] = {}
        self._hsl_value_labels: dict[str, QLabel] = {}
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

    def _load_hsl_sliders_from_adj(self, adj: dict[str, float], color: str | None = None) -> None:
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

    def set_adjustments(self, adj: dict[str, float]) -> None:
        self._mask_user_hidden = False
        self.disarm_dodge_burn()
        from raw_adjustments.dodge_burn import MASK_KEY as _db_mask_key

        self.set_dodge_burn_mask_present(bool(str((adj or {}).get(_db_mask_key, "") or "")))
        try:
            from raw_adjustments.spot_heal import MASK_KEY as _heal_mask_key

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
            self._anamorphic_ratio = float(merged.get("AnamorphicRatio", 1.0) or 1.0)
            self._sync_anamorphic_buttons(self._anamorphic_ratio)
            from raw_adjustments.lut import (
                LUT_AMOUNT_KEY,
                LUT_NAME_KEY,
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
