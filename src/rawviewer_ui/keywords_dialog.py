"""Themed keywords prompt (replaces stock QInputDialog)."""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QLabel, QLineEdit, QVBoxLayout

import theme
from rawviewer_ui.export_options_dialog import _form_row_buttons


class KeywordsDialog(QDialog):
    def __init__(self, parent=None, *, text: str = "", count: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Keywords")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet(theme.form_dialog_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        title = QLabel("Keywords")
        title.setStyleSheet(
            f"color: {theme.INK}; font-size: 17px; font-weight: 500;"
        )
        layout.addWidget(title)

        hint = QLabel(
            "Comma-separated keywords, saved to XMP dc:subject"
            + (f" on {count} photos." if count > 1 else ".")
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 12px;")
        layout.addWidget(hint)

        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setPlaceholderText("wedding, outdoor, 85mm")
        layout.addWidget(self.edit)

        layout.addLayout(_form_row_buttons(self, ok_text="Save"))

    def text(self) -> str:
        return self.edit.text()
