"""Interactive HDR, Panorama and Focus-Stack tuning dialog for RAWviewer.

The dialog tunes the merge on a downscaled proxy and then runs the real one at
full resolution with the settings the user landed on. See ``merge_tuning`` for
the preview engine; everything here is widgets and wiring.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import rawviewer_ui.theme as theme
except ModuleNotFoundError:
    import theme as theme

try:
    from rawviewer_ui.merge_tuning import (
        _SOURCE_COLORS,
        MergePreviewEngine,
        PreviewLabel,
        TuningSignals,
        load_proxies,
        run_async,
        safe_emit,
        source_map_rgb,
        to_pixmap,
    )
except ModuleNotFoundError:  # flat sys.path (tests, frozen app)
    from merge_tuning import (
        _SOURCE_COLORS,
        MergePreviewEngine,
        PreviewLabel,
        TuningSignals,
        load_proxies,
        run_async,
        safe_emit,
        source_map_rgb,
        to_pixmap,
    )


class _FrameRow(QWidget):
    """One source photo: include it, how much it counts, and what EV it was.

    The weight means what the mode needs it to mean -- for an exposure merge it
    scales that frame's contribution, for a focus stack it scales that frame's
    sharpness before the per-region winner is picked (0 drops the frame without
    re-aligning the set).
    """

    changed = pyqtSignal()

    def __init__(self, index: int, path: str, ev: Optional[float], color: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.path = path

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(8)

        self.include_cb = QCheckBox(self)
        self.include_cb.setChecked(True)
        self.include_cb.toggled.connect(lambda _v: self.changed.emit())
        row.addWidget(self.include_cb)

        swatch = QLabel(self)
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
        swatch.setToolTip("This photo's colour in the source map")
        row.addWidget(swatch)

        name = QLabel(os.path.basename(path), self)
        name.setStyleSheet(f"color: {theme.INK}; font-size: 12px;")
        name.setMinimumWidth(120)
        row.addWidget(name, 1)

        if ev is not None:
            badge = QLabel(f"{ev:+.1f} EV", self)
            badge.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 11px;")
            row.addWidget(badge)

        self.weight_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.weight_slider.setRange(0, 20)
        self.weight_slider.setValue(10)
        self.weight_slider.setFixedWidth(90)
        self.weight_slider.valueChanged.connect(lambda _v: self.changed.emit())
        row.addWidget(self.weight_slider)

        self.weight_lbl = QLabel("1.0x", self)
        self.weight_lbl.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 11px;")
        self.weight_lbl.setFixedWidth(34)
        self.weight_slider.valueChanged.connect(
            lambda v: self.weight_lbl.setText(f"{v / 10.0:.1f}x")
        )
        row.addWidget(self.weight_lbl)

    def is_active(self) -> bool:
        return self.include_cb.isChecked()

    def weight(self) -> float:
        return self.weight_slider.value() / 10.0


class HDRPanoramaDialog(QDialog):
    """Configure and preview a merge, then run it at full resolution."""

    def __init__(
        self,
        image_paths: List[str],
        mode: str = "hdr",  # "panorama", "hdr", "hdr_panorama", "focus_stack"
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        from raw_stitching import reorder_merge_paths

        self.mode = mode.lower()
        self.image_paths, self._order_note = reorder_merge_paths(list(image_paths), self.mode)
        self._block_signals = False

        # Preview state
        self._engine = MergePreviewEngine(self.mode)
        self._signals = TuningSignals()
        self._generation = 0
        self._rendering = False
        self._pending = False
        self._proxy_paths: List[str] = []
        self._proxy_of_row: Dict[int, int] = {}
        self._focus_points: List[Tuple[float, float, float, int]] = []
        self._last_included_rows: Tuple[int, ...] = ()
        self._result_rgb = None
        self._result_labels = None
        self._closing = False

        title_map = {
            "panorama": "Standard Panorama Stitching",
            "hdr": "HDR Exposure Fusion & Weight Tuning",
            "hdr_panorama": "Panorama HDR Merge",
            "focus_stack": "Focus Stacking (All-in-Focus Merge)",
        }
        self.setWindowTitle(title_map.get(self.mode, "Stitch & Merge"))
        self.setMinimumSize(980, 620)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {theme.VOID};
                color: {theme.INK};
            }}
            QLabel {{
                color: {theme.INK};
                font-size: 13px;
            }}
            QGroupBox {{
                font-weight: 600;
                font-size: 13px;
                color: {theme.EMBER};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {theme.SURFACE};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {theme.EMBER};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {theme.INK};
                border: 1px solid {theme.LINE};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            """
        )

        self._init_ui()
        self._last_included_rows = tuple(
            i for i, r in enumerate(self.frame_rows) if r.is_active()
        )

        # Debounced so dragging a slider queues one render, not sixty.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(220)
        self._debounce.timeout.connect(self._render_preview)

        self._signals.proxies_ready.connect(self._on_proxies_ready)
        self._signals.preview_ready.connect(self._on_preview_ready)
        self._signals.failed.connect(self._on_preview_failed)
        self._start_proxy_load()

    # ---------------------------------------------------------------- UI ---

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        header_lbl = QLabel(self.windowTitle())
        # Not QFont("Segoe UI", ...): naming one family gives Qt no list to
        # fall through, so on macOS it silently substitutes rather than
        # reaching the platform UI font.
        header_lbl.setFont(theme.ui_font(19, QFont.Weight.Bold))
        header_lbl.setStyleSheet(f"color: {theme.INK};")
        outer.addWidget(header_lbl)

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)
        body.addLayout(self._build_preview_column(), 3)
        body.addLayout(self._build_controls_column(), 2)

        outer.addLayout(self._build_button_row())

        if self.mode == "hdr_panorama":
            self._update_group_hint()

    def _build_preview_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)

        self.preview = PreviewLabel(self)
        self.preview.setStyleSheet(
            f"background-color: {theme.SURFACE}; border: 1px solid {theme.LINE}; "
            "border-radius: 6px;"
        )
        self.preview.clicked.connect(self._on_preview_clicked)
        col.addWidget(self.preview, 1)

        self.status_lbl = QLabel(
            (self._order_note + " Preparing preview…")
            if getattr(self, "_order_note", "")
            else "Preparing preview…",
            self,
        )
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 12px;")
        col.addWidget(self.status_lbl)

        if self.mode == "focus_stack":
            view_row = QHBoxLayout()
            self.view_result_rb = QRadioButton("Merged result", self)
            self.view_result_rb.setChecked(True)
            self.view_source_rb = QRadioButton("Source map", self)
            self.view_source_rb.setToolTip(
                "Which photo each region is being taken from"
            )
            for rb in (self.view_result_rb, self.view_source_rb):
                rb.setStyleSheet(f"color: {theme.INK}; font-size: 12px;")
                rb.toggled.connect(lambda _v: self._refresh_displayed())
                view_row.addWidget(rb)
            view_row.addStretch()
            self.clear_points_btn = QPushButton("Clear focus points", self)
            self.clear_points_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.clear_points_btn.setStyleSheet(
                f"QPushButton {{ background-color: {theme.SURFACE}; color: {theme.INK}; "
                f"border: 1px solid {theme.LINE}; border-radius: 5px; padding: 4px 10px; "
                "font-size: 12px; }"
            )
            self.clear_points_btn.clicked.connect(self._clear_focus_points)
            view_row.addWidget(self.clear_points_btn)
            col.addLayout(view_row)
        return col

    def _build_controls_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(10)
        n = len(self.image_paths)

        if self.mode in ("hdr", "hdr_panorama"):
            col.addWidget(self._build_exposure_box())

        col.addWidget(self._build_frames_box(), 1)

        if self.mode == "hdr_panorama":
            col.addWidget(self._build_bracket_box(n))

        col.addWidget(self._build_options_box())
        return col

    def _build_exposure_box(self) -> QGroupBox:
        box = QGroupBox("Exposure Weights", self)
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        def add_slider(label: str, attr: str) -> None:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            slider = QSlider(Qt.Orientation.Horizontal, self)
            slider.setRange(0, 20)
            slider.setValue(10)
            val = QLabel("1.0x")
            val.setFixedWidth(34)
            slider.valueChanged.connect(lambda v: val.setText(f"{v / 10.0:.1f}x"))
            slider.valueChanged.connect(lambda _v: self._schedule_render())
            row.addWidget(slider, 1)
            row.addWidget(val)
            lay.addLayout(row)
            setattr(self, attr, slider)
            setattr(self, attr.replace("slider", "val_lbl"), val)

        add_slider("Highlights (from the darker frames):", "hl_slider")
        add_slider("Shadows (from the brighter frames):", "sh_slider")
        add_slider("Midtones:", "mid_slider")
        return box

    def _build_frames_box(self) -> QGroupBox:
        box = QGroupBox(f"Source Photos ({len(self.image_paths)})", self)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 10, 6, 6)

        evs = self._exposure_offsets()
        holder = QWidget(self)
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(1)

        self.frame_rows: List[_FrameRow] = []
        for i, path in enumerate(self.image_paths):
            r, g, b = _SOURCE_COLORS[i % len(_SOURCE_COLORS)]
            row = _FrameRow(i, path, evs[i] if evs else None, f"rgb({r},{g},{b})", self)
            row.changed.connect(self._on_frame_row_changed)
            rows.addWidget(row)
            self.frame_rows.append(row)
        rows.addStretch()

        scroll = QScrollArea(self)
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {theme.SURFACE}; "
            f"border: 1px solid {theme.LINE}; border-radius: 4px; }}"
        )
        scroll.setMinimumHeight(120)
        lay.addWidget(scroll)
        return box

    def _build_bracket_box(self, n: int) -> QGroupBox:
        # The engine used to infer this as "2 if the count is even, else 3",
        # which silently fused exposures from DIFFERENT panels on any set that
        # did not fit -- 12 frames shot as 4 panels x 3 stops being the obvious
        # one. It is a fact about the shoot, so it is asked, seeded from the
        # EXIF exposure pattern.
        box = QGroupBox("Bracket Layout", self)
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Exposures per panel:"))
        self.group_spin = QSpinBox(self)
        self.group_spin.setRange(2, max(2, n // 2))
        self.group_spin.setValue(self._inferred_group_size())
        self.group_spin.setStyleSheet(
            f"QSpinBox {{ background-color: {theme.SURFACE}; color: {theme.INK}; "
            f"border: 1px solid {theme.LINE}; border-radius: 4px; padding: 3px; }}"
        )
        self.group_spin.valueChanged.connect(self._on_group_size_changed)
        row.addWidget(self.group_spin)

        row.addSpacing(12)
        row.addWidget(QLabel("Preview panel:"))
        self.panel_spin = QSpinBox(self)
        self.panel_spin.setRange(1, max(1, n // 2))
        self.panel_spin.setStyleSheet(self.group_spin.styleSheet())
        self.panel_spin.setToolTip(
            "The stitch itself is not previewed — the exposure controls act on "
            "each panel's fusion, so one panel is shown."
        )
        self.panel_spin.valueChanged.connect(lambda _v: self._schedule_render())
        row.addWidget(self.panel_spin)
        row.addStretch()
        lay.addLayout(row)

        self.group_hint = QLabel("", self)
        self.group_hint.setWordWrap(True)
        lay.addWidget(self.group_hint)
        return box

    def _build_options_box(self) -> QGroupBox:
        box = QGroupBox("Options", self)
        lay = QVBoxLayout(box)
        is_focus = self.mode == "focus_stack"

        crop_text = (
            "Auto-crop the alignment border"
            if is_focus
            else "Auto-crop warped black borders"
        )
        self.crop_cb = QCheckBox(crop_text, self)
        self.crop_cb.setChecked(True)
        self.crop_cb.setStyleSheet(f"color: {theme.INK}; font-size: 13px;")
        self.crop_cb.toggled.connect(lambda _v: self._schedule_render())
        lay.addWidget(self.crop_cb)

        # Not shown for a plain panorama: cv2.Stitcher does its own feature
        # registration, so there is no alignment step this could switch off.
        if self.mode != "panorama":
            align_text = (
                "Auto-align frames for focus breathing (ECC affine)"
                if is_focus
                else "Auto-align handheld exposures (MTB)"
            )
            self.align_cb = QCheckBox(align_text, self)
            self.align_cb.setChecked(True)
            self.align_cb.setStyleSheet(f"color: {theme.INK}; font-size: 13px;")
            self.align_cb.toggled.connect(self._on_alignment_option_changed)
            lay.addWidget(self.align_cb)

        if is_focus:
            self.parallax_cb = QCheckBox(
                "Correct parallax (local optical-flow align — slower, for handheld)", self
            )
            self.parallax_cb.setChecked(True)
            self.parallax_cb.setStyleSheet(f"color: {theme.INK}; font-size: 13px;")
            self.parallax_cb.toggled.connect(self._on_alignment_option_changed)
            lay.addWidget(self.parallax_cb)

            pick_row = QHBoxLayout()
            pick_row.addWidget(QLabel("Clicks assign the area to:"))
            self.focus_target_combo = QComboBox(self)
            self.focus_target_combo.setStyleSheet(
                f"QComboBox {{ background-color: {theme.SURFACE}; color: {theme.INK}; "
                f"border: 1px solid {theme.LINE}; border-radius: 4px; padding: 3px; "
                "font-size: 12px; }"
            )
            pick_row.addWidget(self.focus_target_combo, 1)
            lay.addLayout(pick_row)

            size_row = QHBoxLayout()
            size_row.addWidget(QLabel("Area size:"))
            self.focus_radius_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.focus_radius_slider.setRange(3, 40)  # 3%-40% of the long edge
            self.focus_radius_slider.setValue(12)
            self.focus_radius_slider.setToolTip(
                "How far a click's influence reaches. Applies to the next click."
            )
            size_row.addWidget(self.focus_radius_slider, 1)
            lay.addLayout(size_row)

            hint = QLabel(
                "Frames must be the same scene at different focal planes; order "
                "does not matter. Where the automatic choice picks the wrong "
                "photo, click that spot in the preview to hand the area to the "
                "photo chosen above."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 12px;")
            lay.addWidget(hint)
        return box

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch()

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setMinimumWidth(90)
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {theme.SURFACE};
                color: {theme.INK};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                font-weight: 500;
            }}
            """
        )
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(cancel_btn)

        # "&&" -- a single & is a Qt mnemonic and renders as "Merge _Stitch".
        self.merge_btn = QPushButton("Merge && Stitch", self)
        self.merge_btn.setMinimumHeight(36)
        self.merge_btn.setMinimumWidth(140)
        self.merge_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.merge_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {theme.EMBER};
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {theme.EMBER_DIM};
            }}
            QPushButton:disabled {{
                background-color: {theme.SURFACE};
                color: {theme.INK_FAINT};
            }}
            """
        )
        self.merge_btn.clicked.connect(self.accept)
        row.addWidget(self.merge_btn)
        return row

    # ----------------------------------------------------------- preview ---

    def _start_proxy_load(self) -> None:
        paths = list(self.image_paths)
        signals = self._signals

        def work():
            imgs, kept = load_proxies(paths)
            safe_emit(signals, "proxies_ready", imgs, kept)

        run_async(work)

    def _on_proxies_ready(self, images, paths) -> None:
        if self._closing:
            return
        self._proxy_paths = list(paths)
        # A source that failed to decode has no proxy, so the engine's list is
        # shorter than the dialog's rows. Map row -> proxy explicitly rather
        # than assuming they line up; assuming it silently previews the wrong
        # photographs the moment one file in the middle fails to decode.
        by_path = {os.path.normcase(os.path.abspath(p)): i for i, p in enumerate(paths)}
        self._proxy_of_row = {}
        for row_idx, row in enumerate(self.frame_rows):
            proxy_idx = by_path.get(os.path.normcase(os.path.abspath(row.path)))
            if proxy_idx is None:
                row.setEnabled(False)
                row.include_cb.setChecked(False)
            else:
                self._proxy_of_row[row_idx] = proxy_idx

        if len(images) < 2:
            self.status_lbl.setText(
                "Could not decode enough of these photos to preview the merge. "
                "The merge itself will still run."
            )
            return
        self._engine.set_images(images)
        self._refresh_focus_targets()
        self._render_preview()

    def _schedule_render(self) -> None:
        if not self._closing:
            self._debounce.start()

    def _on_alignment_option_changed(self, _checked=False) -> None:
        # Alignment is the cached stage; changing it invalidates the cache.
        self._engine.invalidate()
        self._schedule_render()

    def _on_frame_row_changed(self) -> None:
        if self.mode == "hdr_panorama":
            self._update_group_hint()
        included = tuple(i for i, r in enumerate(self.frame_rows) if r.is_active())
        # Unticking a photo renumbers the fused set, so stored focus points
        # would point at a different frame than the user picked. Drop them
        # rather than silently re-aiming them. Weight-only edits must not.
        if self._focus_points and included != self._last_included_rows:
            self._focus_points.clear()
            self.status_lbl.setText(
                "Focus points cleared — the set of photos changed."
            )
        self._last_included_rows = included
        self._refresh_focus_targets()
        self._schedule_render()

    def _on_group_size_changed(self, _v=0) -> None:
        self._update_group_hint()
        self._engine.invalidate()
        self._schedule_render()

    def _active_rows(self) -> List[int]:
        """Row indices that are ticked AND have a decoded proxy behind them."""
        return [
            i for i, r in enumerate(self.frame_rows)
            if r.is_active() and i in self._proxy_of_row
        ]

    def _render_opts(self) -> Dict[str, Any]:
        rows = self._active_rows()
        opts: Dict[str, Any] = {
            "active": [self._proxy_of_row[i] for i in rows],
            "frame_weights": [self.frame_rows[i].weight() for i in rows],
            "auto_crop": self.should_auto_crop(),
            "align": self.should_align(),
            "parallax": self.should_correct_parallax(),
            "focus_points": list(self._focus_points),
        }
        opts.update(self.get_hdr_weights())
        if self.mode == "hdr_panorama":
            opts["group_size"] = self.get_group_size()
            opts["preview_panel"] = max(0, self.panel_spin.value() - 1)
        return opts

    def _render_preview(self) -> None:
        if self._closing or not self._engine.images:
            return
        if self._rendering:
            self._pending = True
            return
        opts = self._render_opts()
        if len(opts["active"]) < 2:
            self.status_lbl.setText("Tick at least two photos to preview a merge.")
            self.preview.set_image(None)
            return

        self._rendering = True
        self._generation += 1
        gen = self._generation
        signals = self._signals
        engine = self._engine
        self.status_lbl.setText("Rendering preview…")

        def work():
            try:
                rgb, labels = engine.render(opts)
                safe_emit(signals, "preview_ready", gen, rgb, labels)
            except Exception as exc:  # engine limits, stitch failures, ...
                safe_emit(signals, "failed", gen, str(exc))

        run_async(work)

    def _on_preview_ready(self, gen: int, rgb, labels) -> None:
        self._rendering = False
        if self._closing:
            return
        if gen == self._generation:  # a stale render finished after a newer one
            self._result_rgb = rgb
            self._result_labels = labels
            self._refresh_displayed()
            self.status_lbl.setText(self._preview_hint())
        if self._pending:
            self._pending = False
            self._render_preview()

    def _on_preview_failed(self, gen: int, message: str) -> None:
        self._rendering = False
        if self._closing:
            return
        if gen == self._generation:
            self.preview.set_image(None)
            self.status_lbl.setText(f"Preview unavailable: {message}")
        if self._pending:
            self._pending = False
            self._render_preview()

    def _preview_hint(self) -> str:
        base = "Preview at reduced size; the merge runs at full resolution."
        if getattr(self, "_order_note", ""):
            base = self._order_note + " " + base
        if self.mode == "hdr_panorama":
            return "Showing one panel's exposure fusion. " + base
        if self.mode == "focus_stack" and self._focus_points:
            n = len(self._focus_points)
            return f"{n} focus point{'s' if n != 1 else ''} set. " + base
        return base

    def _refresh_displayed(self) -> None:
        if self._result_rgb is None:
            return
        show_source = (
            self.mode == "focus_stack"
            and getattr(self, "view_source_rb", None) is not None
            and self.view_source_rb.isChecked()
            and self._result_labels is not None
        )
        if show_source:
            # Colours in the order the engine fused, matching each row's swatch.
            colors = [
                _SOURCE_COLORS[i % len(_SOURCE_COLORS)] for i in self._active_rows()
            ]
            self.preview.set_image(to_pixmap(source_map_rgb(self._result_labels, colors)))
        else:
            self.preview.set_image(to_pixmap(self._result_rgb))

    def _refresh_focus_targets(self) -> None:
        """Keep the click-target picker in step with the ticked photos."""
        combo = getattr(self, "focus_target_combo", None)
        if combo is None:
            return
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for row_idx in self._active_rows():
            combo.addItem(os.path.basename(self.frame_rows[row_idx].path), row_idx)
        if previous is not None:
            found = combo.findData(previous)
            if found >= 0:
                combo.setCurrentIndex(found)
        combo.blockSignals(False)

    def _on_preview_clicked(self, nx: float, ny: float) -> None:
        if self.mode != "focus_stack" or not self._engine.images:
            return
        rows = self._active_rows()
        combo = getattr(self, "focus_target_combo", None)
        target_row = combo.currentData() if combo is not None else None
        if target_row not in rows:
            target_row = rows[0] if rows else None
        if target_row is None:
            return
        # Stored as a position in the FUSED set (what the engine indexes), not
        # as a row: rows the user unticked are not in the stack being fused.
        self._focus_points.append((nx, ny, self._focus_radius(), rows.index(target_row)))
        if getattr(self, "view_source_rb", None) is not None:
            self.view_source_rb.setChecked(True)
        self._schedule_render()

    def _focus_radius(self) -> float:
        slider = getattr(self, "focus_radius_slider", None)
        return (slider.value() / 100.0) if slider is not None else 0.12

    def _clear_focus_points(self) -> None:
        self._focus_points.clear()
        self._schedule_render()

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        self._closing = True
        super().closeEvent(event)

    # ------------------------------------------------------------ getters ---

    def _exposure_offsets(self) -> List[float]:
        """Per-frame EV offsets for the badges, best EXIF source available."""
        try:
            from raw_stitching import exposure_ev_offsets_for_paths

            evs = exposure_ev_offsets_for_paths(list(self.image_paths))
            return evs if len(evs) == len(self.image_paths) else []
        except Exception:
            return []

    def _inferred_group_size(self) -> int:
        """Best guess at exposures per panel, from the set's EXIF EV pattern."""
        try:
            from raw_stitching import infer_bracket_group_size

            return int(infer_bracket_group_size(self.image_paths))
        except Exception:
            return 3 if len(self.image_paths) % 2 else 2

    def _update_group_hint(self) -> None:
        """Show the resulting panel count, and refuse a layout that cannot fit."""
        n = len(self.get_active_image_paths()) or len(self.image_paths)
        g = self.group_spin.value()
        divides = g >= 2 and n % g == 0
        enough = n >= 4
        ok = divides and enough
        if ok:
            self.group_hint.setText(f"{n} photos = {n // g} panels of {g} exposures.")
            self.group_hint.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 12px;")
            self.panel_spin.setRange(1, max(1, n // g))
        elif not enough:
            self.group_hint.setText(
                "Panorama HDR needs at least two panels (4 photos). "
                "Use HDR Merge for a single bracket."
            )
            self.group_hint.setStyleSheet(f"color: {theme.EMBER}; font-size: 12px;")
        else:
            self.group_hint.setText(
                f"{n} photos do not divide into brackets of {g}. "
                "Adjust the count, or untick the extra frames."
            )
            self.group_hint.setStyleSheet(f"color: {theme.EMBER}; font-size: 12px;")
        btn = getattr(self, "merge_btn", None)
        if btn is not None:
            btn.setEnabled(ok)

    def should_auto_crop(self) -> bool:
        """Returns True if auto-cropping warped borders is enabled."""
        return self.crop_cb.isChecked()

    def should_align(self) -> bool:
        """Returns True if auto-alignment is enabled.

        Always True for a plain panorama, which has no checkbox: the Stitcher
        registers the frames itself.
        """
        cb = getattr(self, "align_cb", None)
        return bool(cb.isChecked()) if cb is not None else True

    def should_correct_parallax(self) -> bool:
        """Focus stacking only: True if local optical-flow refinement is on."""
        cb = getattr(self, "parallax_cb", None)
        return bool(cb.isChecked()) if cb is not None else False

    def get_hdr_weights(self) -> Dict[str, float]:
        """Returns dict of highlight, shadow, midtone weight multipliers."""
        if self.mode not in ("hdr", "hdr_panorama"):
            return {"highlight": 1.0, "shadow": 1.0, "midtone": 1.0}
        return {
            "highlight": float(self.hl_slider.value()) / 10.0,
            "shadow": float(self.sh_slider.value()) / 10.0,
            "midtone": float(self.mid_slider.value()) / 10.0,
        }

    def get_group_size(self) -> Optional[int]:
        """Exposures per panel for Panorama HDR; None for every other mode."""
        spin = getattr(self, "group_spin", None)
        return int(spin.value()) if spin is not None else None

    def get_active_image_paths(self) -> List[str]:
        """Returns list of ticked image paths."""
        return [r.path for r in self.frame_rows if r.is_active()]

    def get_image_weights(self) -> List[float]:
        """Per-frame weights, in the same order as get_active_image_paths()."""
        return [r.weight() for r in self.frame_rows if r.is_active()]

    def get_focus_points(self) -> List[Tuple[float, float, float, int]]:
        """Focus-stack overrides as (x, y, radius, active-frame index), 0-1."""
        return list(self._focus_points)
