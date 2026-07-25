"""Endpoint configuration for generative editing.

Deliberately spartan: a URL, an optional key, and one checkbox that is
hard to tick by accident. There is no "recommended provider" list and no
default endpoint -- the app should never nudge a user into uploading
their photographs somewhere.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

import theme


class GenerativeSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generative Edit Setup")
        self.setModal(True)
        self.setMinimumWidth(460)

        from generative_settings import load_settings

        current = load_settings()

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        intro = QLabel(
            "RAWviewer ships no generative model. Point this at a server you "
            "run or trust — your own machine, or a service you have an "
            "account with."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 11px;")
        outer.addWidget(intro)

        form = QFormLayout()
        form.setSpacing(8)

        self._endpoint = QLineEdit(current.get("endpoint", ""))
        self._endpoint.setPlaceholderText("https://your-server.example/edit")
        form.addRow("Endpoint", self._endpoint)

        self._api_key = QLineEdit(current.get("api_key", ""))
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setPlaceholderText("optional")
        form.addRow("API key", self._api_key)

        self._model = QLineEdit(current.get("model_name", "remote"))
        self._model.setPlaceholderText("recorded in the file's provenance")
        form.addRow("Model name", self._model)

        outer.addLayout(form)

        self._insecure = QCheckBox("Allow plain http to non-local servers")
        self._insecure.setChecked(bool(current.get("allow_insecure", False)))
        self._insecure.setToolTip(
            "Photos sent over plain http can be read by anything on the "
            "network path. Local addresses (127.0.0.1) are always allowed."
        )
        self._insecure.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 11px;")
        outer.addWidget(self._insecure)

        contract = QLabel(
            "The endpoint receives JSON: {\"instruction\", \"image\" (base64 "
            "PNG), \"seed\", \"options\"} and must reply with "
            "{\"image\": \"<base64 PNG>\"} or {\"error\": \"...\"}."
        )
        contract.setWordWrap(True)
        contract.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")
        outer.addWidget(contract)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _on_save(self) -> None:
        from generative_settings import save_settings

        # save_settings revokes consent when the endpoint changes, so a
        # new destination always re-asks before anything is uploaded.
        save_settings(
            endpoint=self._endpoint.text().strip(),
            api_key=self._api_key.text(),
            model_name=self._model.text().strip() or "remote",
            allow_insecure=self._insecure.isChecked(),
        )
        self.accept()
