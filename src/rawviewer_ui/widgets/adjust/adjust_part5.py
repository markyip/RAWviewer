"""Adjust mixin part 5 of 5 for ImageAdjustPanelWidget."""
from __future__ import annotations

import contextlib

from PyQt6.QtCore import QSettings, Qt, QThreadPool
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from core import theme

from .helpers import (
    _GLOBAL_SECTION_DEFAULT_ORDER,
    _GLOBAL_SECTION_ORDER_SETTING,
    _LUT_THUMB_CACHE_MAX,
    _LUT_THUMB_DEBOUNCE_MS,
    CollapsibleSection,
    _LutThumbWorker,
    _u8_rgb_to_pixmap,
)


class AdjustPart5:

    def _build_anamorphic_row(self, sect: CollapsibleSection) -> None:
        from raw_adjustments import ANAMORPHIC_DESQUEEZE_PRESETS

        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(4)
        lbl = QLabel("Anamorphic desqueeze")
        lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        col.addWidget(lbl)
        row = QHBoxLayout()
        row.setSpacing(4)
        self._anamorphic_btns: list[QPushButton] = []
        for label, value in ANAMORPHIC_DESQUEEZE_PRESETS:
            text = "1.0x" if abs(value - 1.0) < 1e-6 else label
            btn = QPushButton(text)
            btn.setObjectName("adjust_nr_btn")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setToolTip(
                "No desqueeze (native capture proportions)"
                if abs(value - 1.0) < 1e-6
                else f"Desqueeze {text} — stretch width to restore natural proportions"
            )
            btn.setProperty("anamorphic_ratio", float(value))
            btn.clicked.connect(
                lambda _checked=False, v=float(value): self._on_anamorphic_chosen(v)
            )
            row.addWidget(btn, 1)
            self._anamorphic_btns.append(btn)
        col.addLayout(row)
        hint = QLabel("Only for footage shot through an anamorphic adapter.")
        hint.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")
        hint.setWordWrap(True)
        col.addWidget(hint)
        sect.add_widget(wrap)
        self._sync_anamorphic_buttons(1.0)

    def _sync_anamorphic_buttons(self, ratio: float) -> None:
        buttons = getattr(self, "_anamorphic_btns", None)
        if not buttons:
            return
        for btn in buttons:
            value = float(btn.property("anamorphic_ratio") or 1.0)
            btn.setChecked(abs(value - float(ratio)) < 1e-3)

    def _on_anamorphic_chosen(self, ratio: float) -> None:
        self._sync_anamorphic_buttons(ratio)
        if self._block_emit:
            return
        self._anamorphic_ratio = float(ratio)
        self._emit_preview_and_save()

    def _build_auto_tone_row(self, sect: CollapsibleSection) -> None:
        """One-press Auto: white balance, Light sliders, and straighten."""
        row = QHBoxLayout()
        row.setSpacing(6)
        lbl = QLabel("Auto")
        lbl.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
        lbl.setFixedWidth(82)
        row.addWidget(lbl)
        btn = QPushButton("AUTO")
        btn.setObjectName("adjust_auto_btn")
        btn.setFixedHeight(20)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(
            "Auto — white balance, Exposure / Contrast / Highlights / Shadows "
            "/ Whites / Blacks, and straighten from the unedited photo.\n\n"
            "Does not move Saturation, Vibrance, or HSL — grade colour there "
            "after Auto, or with the WB dropper.\n\n"
            "Not the P-key recovery preview. Ctrl+Z undoes the lot."
        )
        btn.clicked.connect(lambda: self.auto_tone_requested.emit())
        row.addWidget(btn, 1)
        sect.add_layout(row)
        self._auto_tone_btn = btn

    def apply_auto_tone(self, values: dict) -> None:
        """Overwrite Auto keys (WB, Light sliders, CropAngle) and save."""
        current = self.get_adjustments()
        for key, value in values.items():
            current[key] = float(value)
        self.set_adjustments(current)
        self._emit_preview_and_save()

    def _settings(self) -> QSettings:
        return QSettings("RAWviewer", "RAWviewer")

    def _load_section_order(self) -> list:
        import json

        try:
            raw = self._settings().value(_GLOBAL_SECTION_ORDER_SETTING, "")
            if not raw:
                return []
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            if isinstance(raw, (list, tuple)):
                return list(raw)
            return json.loads(raw)
        except Exception:
            return []

    def _save_section_order(self) -> None:
        import json

        with contextlib.suppress(Exception):
            self._settings().setValue(
                _GLOBAL_SECTION_ORDER_SETTING, json.dumps(list(self._section_order))
            )

    def _apply_section_order(self) -> None:
        """Lay Global sections out in ``self._section_order``; Export stays last."""
        layout = getattr(self, "_global_layout", None)
        if layout is None:
            return
        stretch_item = None
        for i in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(i)
            if item is not None and item.spacerItem() is not None:
                stretch_item = layout.takeAt(i)
                break
        for key in self._section_order:
            sect = self._global_sections.get(key)
            if sect is None:
                continue
            layout.removeWidget(sect)
            layout.addWidget(sect)
        export_btn = getattr(self, "_export_btn", None)
        if export_btn is not None and layout.indexOf(export_btn) != -1:
            layout.removeWidget(export_btn)
            layout.addWidget(export_btn)
        if stretch_item is not None:
            layout.addItem(stretch_item)
        layout.activate()

    def _visible_section_order(self) -> list:
        return [
            k
            for k in self._section_order
            if (s := self._global_sections.get(k)) is not None and not s.isHidden()
        ]

    def _set_visible_section_order(self, visible: list) -> None:
        slots = [i for i, k in enumerate(self._section_order) if k in visible]
        for slot, k in zip(slots, visible, strict=False):
            self._section_order[slot] = k

    def _nudge_section(self, key: str, delta: int) -> None:
        visible = self._visible_section_order()
        if key not in visible:
            return
        pos = visible.index(key)
        target = pos + int(delta)
        if target < 0 or target >= len(visible):
            return
        visible.insert(target, visible.pop(pos))
        self._set_visible_section_order(visible)
        self._apply_section_order()
        self._save_section_order()

    def _reset_section_order(self) -> None:
        self._section_order = list(_GLOBAL_SECTION_DEFAULT_ORDER)
        self._apply_section_order()
        self._save_section_order()

    def _on_section_context_menu(self, key: str) -> None:
        visible = self._visible_section_order()
        if key not in visible:
            return
        pos = visible.index(key)
        menu = QMenu(self)
        up = menu.addAction("Move Up")
        up.setEnabled(pos > 0)
        up.triggered.connect(lambda: self._nudge_section(key, -1))
        down = menu.addAction("Move Down")
        down.setEnabled(pos < len(visible) - 1)
        down.triggered.connect(lambda: self._nudge_section(key, 1))
        menu.addSeparator()
        reset = menu.addAction("Reset Order")
        reset.setEnabled(list(self._section_order) != list(_GLOBAL_SECTION_DEFAULT_ORDER))
        reset.triggered.connect(self._reset_section_order)
        menu.exec(QCursor.pos())

    def _lut_thumb_image_token(self) -> str:
        path = str(getattr(self, "_lut_thumb_path", "") or "")
        return f"{path}#as_shot"

    def _lut_thumb_space(self) -> str:
        from raw_adjustments.lut import LUT_SPACE_DEFAULT

        combo = getattr(self, "_lut_space_combo", None)
        if combo is None:
            return LUT_SPACE_DEFAULT
        data = combo.currentData()
        return str(data or LUT_SPACE_DEFAULT)

    def clear_lut_preview_source(self) -> None:
        """Names-only until an as-shot LUT base for this photo arrives."""
        self._lut_thumb_token = int(getattr(self, "_lut_thumb_token", 0)) + 1
        self._lut_thumb_display_u8 = None
        self._lut_thumb_path = ""
        self._lut_thumb_source_gen = 0
        timer = getattr(self, "_lut_thumb_timer", None)
        if timer is not None:
            timer.stop()
        for row in (getattr(self, "_lut_thumb_rows", {}) or {}).values():
            row.set_preview(None)

    def set_lut_preview_source(
        self,
        display_u8,
        path: str,
        generation: int = 0,
        **_unused,
    ) -> None:
        """Debounced LUT thumbs from a tiny as-shot display of this photo."""
        self._lut_thumb_display_u8 = display_u8
        self._lut_thumb_path = str(path or "")
        self._lut_thumb_source_gen = int(generation)
        self._apply_cached_lut_thumbs()
        self._schedule_lut_thumbs()

    def _schedule_lut_thumbs(self) -> None:
        timer = getattr(self, "_lut_thumb_timer", None)
        if timer is None:
            return
        if getattr(self, "_lut_thumb_display_u8", None) is None:
            return
        timer.start(_LUT_THUMB_DEBOUNCE_MS)

    def _apply_cached_lut_thumbs(self) -> bool:
        from raw_adjustments.lut import LUT_PREVIEW_AMOUNT, lut_preview_cache_key

        cache = getattr(self, "_lut_thumb_cache", None)
        rows = getattr(self, "_lut_thumb_rows", None) or {}
        if not cache or not rows:
            return False
        if getattr(self, "_lut_thumb_display_u8", None) is None:
            for row in rows.values():
                row.set_preview(None)
            return False
        token = self._lut_thumb_image_token()
        amount = LUT_PREVIEW_AMOUNT
        space = self._lut_thumb_space()
        all_hit = True
        for lut_id, row in rows.items():
            key = lut_preview_cache_key(token, lut_id, amount, space)
            pm = cache.get(key)
            if pm is None:
                all_hit = False
                continue
            cache.move_to_end(key)
            row.set_preview(pm)
        return all_hit

    def _start_lut_thumb_worker(self) -> None:
        from raw_adjustments.lut import LUT_PREVIEW_AMOUNT, list_managed_luts

        display = getattr(self, "_lut_thumb_display_u8", None)
        if display is None:
            return
        if self._apply_cached_lut_thumbs():
            return
        self._lut_thumb_token = int(getattr(self, "_lut_thumb_token", 0)) + 1
        token = self._lut_thumb_token
        worker = _LutThumbWorker(
            token,
            display,
            list_managed_luts(),
            LUT_PREVIEW_AMOUNT,
            self._lut_thumb_space(),
            self._lut_thumb_signals,
        )
        QThreadPool.globalInstance().start(worker)

    def _on_lut_thumbs_ready(self, token: int, previews) -> None:
        from raw_adjustments.lut import LUT_PREVIEW_AMOUNT, lut_preview_cache_key

        if int(token) != int(getattr(self, "_lut_thumb_token", 0)):
            return
        if not previews:
            return
        token_s = self._lut_thumb_image_token()
        amount = LUT_PREVIEW_AMOUNT
        space = self._lut_thumb_space()
        cache = self._lut_thumb_cache
        rows = getattr(self, "_lut_thumb_rows", {}) or {}
        for lut_id, arr in previews.items():
            pm = _u8_rgb_to_pixmap(arr)
            if pm is None:
                continue
            key = lut_preview_cache_key(token_s, lut_id, amount, space)
            cache[key] = pm
            cache.move_to_end(key)
            row = rows.get(lut_id)
            if row is not None:
                row.set_preview(pm)
        while len(cache) > _LUT_THUMB_CACHE_MAX:
            cache.popitem(last=False)
