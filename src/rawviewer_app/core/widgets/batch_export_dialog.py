"""BatchExportDialog: batch export confirmation dialog."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

from core import theme

from .confirm_dialog import CustomConfirmDialog
from .helpers import _norm_dir_key


class BatchExportDialog(CustomConfirmDialog):
    """Pre-export options dialog (workflow mirrors the delete confirmation).

    Image source (copy originals / export edited + format), destination folder
    with inline validation, and an optional reveal toggle. Accepting the dialog
    IS the confirmation — no second confirm step. The validation error appears
    inline below the body and the dialog stays open.
    """

    def __init__(
        self,
        parent=None,
        *,
        count: int = 0,
        start_dir: str = "",
        source_dir_keys: set | None = None,
        writable_check=None,
    ):
        noun = "photo" if count == 1 else "photos"
        super().__init__(
            parent=parent,
            title="Export",
            message=f"Export {count} {noun}",
            informative_text=(
                "Existing files are kept — new files get _1, _2 suffixes. "
                "Nothing is deleted or moved."
            ),
            confirm_action="Export",
        )

        from PyQt6.QtWidgets import QButtonGroup, QCheckBox, QLineEdit, QRadioButton

        self._source_dir_keys = set(source_dir_keys or {})
        self._writable_check = writable_check

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        _radio_style = f"""
            QRadioButton {{
                color: {theme.INK}; font-size: 14px; background: transparent;
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 14px; height: 14px; border-radius: 8px;
                border: 1px solid {theme.INK_MUTED}; background: transparent;
            }}
            QRadioButton::indicator:hover {{ border-color: {theme.INK}; }}
            QRadioButton::indicator:checked {{
                border: 1px solid {theme.EMBER}; background: {theme.EMBER};
            }}
            QRadioButton:disabled {{ color: {theme.INK_FAINT}; }}
            QRadioButton::indicator:disabled {{
                border-color: {theme.LINE}; background: transparent;
            }}
        """
        self.mode_copy_btn = QRadioButton("Copy originals (untouched + XMP sidecars)")
        self.mode_copy_btn.setStyleSheet(_radio_style)
        self.mode_edited_btn = QRadioButton("Export edited copies")
        self.mode_edited_btn.setStyleSheet(_radio_style)
        body_layout.addWidget(self.mode_copy_btn)
        body_layout.addWidget(self.mode_edited_btn)

        # QRadioButton auto-exclusive only groups siblings, and these buttons
        # all live under the same body widget -- without two explicit groups
        # picking a format would silently uncheck "Export edited copies"
        # (export then fell back to copy-originals mode).
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self.mode_copy_btn)
        self._mode_group.addButton(self.mode_edited_btn)

        format_row = QHBoxLayout()
        format_row.setContentsMargins(24, 0, 0, 0)
        format_row.setSpacing(14)
        self.format_jpeg_btn = QRadioButton("JPEG")
        self.format_tiff_btn = QRadioButton("TIFF 16-bit")
        self.format_webp_btn = QRadioButton("WebP")
        for btn in (self.format_jpeg_btn, self.format_tiff_btn, self.format_webp_btn):
            btn.setStyleSheet(_radio_style)
            format_row.addWidget(btn)
        body_layout.addLayout(format_row)

        self._format_group = QButtonGroup(self)
        self._format_group.setExclusive(True)
        for btn in (self.format_jpeg_btn, self.format_tiff_btn, self.format_webp_btn):
            self._format_group.addButton(btn)

        self.destination_edit = QLineEdit()
        self.destination_edit.setPlaceholderText("Choose a destination folder…")
        self.destination_edit.setStyleSheet(f"""
            QLineEdit {{
                color: {theme.INK}; background: {theme.SURFACE};
                border: 1px solid {theme.LINE}; border-radius: 6px;
                padding: 5px 8px; font-size: 13px;
            }}
            QLineEdit:focus {{ border-color: {theme.EMBER}; }}
        """)

        self.choose_btn = QToolButton()
        self.choose_btn.setText("Choose…")
        self.choose_btn.setStyleSheet(f"""
            QToolButton {{
                color: {theme.INK}; background: transparent;
                border: 1px solid {theme.LINE}; border-radius: 6px;
                padding: 5px 12px; font-size: 13px;
            }}
            QToolButton:hover {{ background: rgba(255, 255, 255, 0.05); }}
        """)
        dest_row = QHBoxLayout()
        dest_row.setSpacing(8)
        dest_row.addWidget(self.destination_edit, 1)
        dest_row.addWidget(self.choose_btn)
        body_layout.addLayout(dest_row)

        self.reveal_check = QCheckBox("Open destination when finished")
        self.reveal_check.setStyleSheet(f"""
            QCheckBox {{
                color: {theme.INK_MUTED}; font-size: 13px;
                spacing: 8px; background: transparent;
            }}
            QCheckBox::indicator {{
                width: 13px; height: 13px; border-radius: 4px;
                border: 1px solid {theme.INK_MUTED}; background: transparent;
            }}
            QCheckBox::indicator:hover {{ border-color: {theme.INK}; }}
            QCheckBox::indicator:checked {{
                border: 1px solid {theme.EMBER}; background: {theme.EMBER};
            }}
        """)
        body_layout.addWidget(self.reveal_check)

        self.ai_denoise_check = QCheckBox("Apply AI denoise (slower)")
        self.ai_denoise_check.setStyleSheet(f"""
            QCheckBox {{ color: {theme.INK_MUTED}; font-size: 13px; spacing: 8px; }}
            QCheckBox::indicator {{
                width: 13px; height: 13px; border-radius: 4px;
                border: 1px solid {theme.INK_MUTED}; background: transparent;
            }}
            QCheckBox::indicator:hover {{ border-color: {theme.INK}; }}
            QCheckBox::indicator:checked {{ border: 1px solid {theme.EMBER}; background: {theme.EMBER}; }}
        """)
        self.ai_denoise_check.hide()  # only meaningful for "Export edited copies"
        self.ai_denoise_check.toggled.connect(lambda *_: self._sync_ai_denoise_row())
        body_layout.addWidget(self.ai_denoise_check)

        # AI denoise strength slider (0–100). Visible only when the checkbox is
        # checked; overrides each photo's stored AI Amount for the batch.
        ai_row = QWidget()
        ai_row.setStyleSheet("background: transparent;")
        ai_row_layout = QHBoxLayout(ai_row)
        ai_row_layout.setContentsMargins(24, 0, 0, 0)
        ai_row_layout.setSpacing(8)
        self.ai_amount_label = QLabel("AI Amount: 100%")
        self.ai_amount_label.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 13px;")
        ai_row_layout.addWidget(self.ai_amount_label)
        self.ai_denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.ai_denoise_slider.setRange(0, 100)
        self.ai_denoise_slider.setValue(100)
        self.ai_denoise_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px; background: {theme.LINE}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 14px; height: 14px; margin: -5px 0;
                border-radius: 7px; background: {theme.EMBER};
            }}
            QSlider::sub-page:horizontal {{
                background: {theme.EMBER}; border-radius: 2px;
            }}
            QSlider::handle:horizontal:disabled {{ background: {theme.INK_FAINT}; }}
            QSlider::sub-page:horizontal:disabled {{ background: {theme.INK_FAINT}; }}
        """)
        self.ai_denoise_slider.valueChanged.connect(self._on_ai_amount_changed)
        ai_row_layout.addWidget(self.ai_denoise_slider, 1)
        self.ai_denoise_row = ai_row
        self.ai_denoise_row.hide()
        body_layout.addWidget(self.ai_denoise_row)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("QLabel { color: #FF8A80; font-size: 13px; }")
        self.error_label.hide()
        body_layout.addWidget(self.error_label)

        self.insert_body_widget(body)

        self.mode_copy_btn.toggled.connect(self._sync_mode_ui)
        self.destination_edit.textChanged.connect(lambda *_: self._show_error(None))
        self.choose_btn.clicked.connect(self._on_choose_clicked)
        self.result_config: dict | None = None

        # Restore previously remembered choices.
        try:
            settings = parent.get_settings() if parent is not None else None
        except Exception:
            settings = None
        if settings is not None:
            mode = (settings.value("batch_export_mode", "copy") or "copy").strip()
            fmt = (settings.value("batch_export_format", "jpeg") or "jpeg").strip()
            start = settings.value("last_batch_export_dir", "") or ""
            ai = settings.value("batch_export_ai_denoise", "0") in ("1", True, "true")
            ai_amount_raw = settings.value("batch_export_ai_amount", "100")
            try:
                ai_amount = max(0, min(100, int(ai_amount_raw)))
            except (TypeError, ValueError):
                ai_amount = 100
            self.reveal_check.setChecked(
                settings.value("batch_export_reveal", "0") in ("1", True, "true")
            )
        else:
            mode, fmt, start, ai, ai_amount = "copy", "jpeg", "", False, 100
        ai_ready = self._ai_denoise_available()
        self.ai_denoise_check.setEnabled(ai_ready)
        if not ai_ready:
            self.ai_denoise_check.setToolTip(
                "AI denoise is not available: the AI model is not installed."
            )
        self.ai_denoise_check.setChecked(bool(ai and ai_ready))
        self.ai_denoise_slider.setValue(ai_amount)
        self.ai_amount_label.setText(f"AI Amount: {ai_amount}%")
        start_dir = start_dir or start
        if start_dir and os.path.isdir(start_dir):
            self.destination_edit.setText(start_dir)
        (self.mode_edited_btn if mode == "edited" else self.mode_copy_btn).setChecked(True)
        {  # format restore; default JPEG
            "jpeg": self.format_jpeg_btn,
            "tiff16": self.format_tiff_btn,
            "webp": self.format_webp_btn,
        }.get(fmt, self.format_jpeg_btn).setChecked(True)
        # The restore may set the copy button while it is already checked, so
        # toggled() never fires and the format row would stay enabled; sync
        # explicitly (also sizes the dialog for the real body content).
        self._sync_mode_ui()

    @staticmethod
    def _ai_denoise_available() -> bool:
        """AI denoise needs both the feature flag and the installed model file."""
        try:
            from core.profile import ai_export_enabled
            from raw_adjustments.pipeline import ai_denoise_model_present

            return bool(ai_export_enabled() and ai_denoise_model_present())
        except Exception:
            return False

    def _sync_mode_ui(self, *_args) -> None:
        edited = self.mode_edited_btn.isChecked()
        for btn in (self.format_jpeg_btn, self.format_tiff_btn, self.format_webp_btn):
            btn.setEnabled(edited)
        self.ai_denoise_check.setVisible(edited)
        if not edited:
            self.ai_denoise_check.setChecked(False)
        self._sync_ai_denoise_row()
        self._show_error(None)

    def _sync_ai_denoise_row(self) -> None:
        """Show the AI Amount slider only when denoise is enabled."""
        show = self.ai_denoise_check.isVisible() and self.ai_denoise_check.isChecked()
        self.ai_denoise_row.setVisible(show)
        self.ai_denoise_slider.setEnabled(self.ai_denoise_check.isChecked())
        self._show_error(None)

    def _on_ai_amount_changed(self, value: int) -> None:
        self.ai_amount_label.setText(f"AI Amount: {value}%")

    def _on_choose_clicked(self) -> None:
        pick = getattr(self, "pick_directory", None)
        picked = pick() if callable(pick) else ""
        if picked and os.path.isdir(picked):
            self.destination_edit.setText(picked)

    def _show_error(self, text: str | None) -> None:
        if text:
            self.error_label.setText(text)
            self.error_label.show()
            # Re-fit so the inline error is never clipped by the fixed size.
            self._apply_dialog_size()
        else:
            self.error_label.hide()

    def validate(self) -> str | None:
        dest = self.destination_edit.text().strip()
        if not dest:
            return "Choose a destination folder."
        if not os.path.isdir(dest):
            return "Destination folder does not exist."
        if self._writable_check is not None and not self._writable_check(dest):
            return (
                "Destination folder is not writable (SD card / read-only volume). "
                "Choose Desktop, Pictures, or Downloads."
            )
        if (
            self.mode_copy_btn.isChecked()
            and _norm_dir_key(dest) in self._source_dir_keys
        ):
            return (
                "Destination is the same folder as the originals — choose a "
                "different folder (or use Export edited copies)."
            )
        return None

    def accept(self) -> None:
        err = self.validate()
        if err:
            self._show_error(err)
            return
        self.result_config = {
            "mode": "edited" if self.mode_edited_btn.isChecked() else "copy",
            "format": (
                "tiff16" if self.format_tiff_btn.isChecked()
                else "webp" if self.format_webp_btn.isChecked()
                else "jpeg"
            ),
            "destination": os.path.normpath(self.destination_edit.text().strip()),
            "reveal": self.reveal_check.isChecked(),
            "ai_denoise": self.mode_edited_btn.isChecked() and self.ai_denoise_check.isChecked(),
            "ai_amount": int(self.ai_denoise_slider.value()),
        }
        self.result_value = True
        super().accept()
