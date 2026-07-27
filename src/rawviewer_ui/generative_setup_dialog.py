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
    QComboBox,
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

        # Which kind of provider. "On this machine" is listed first and is the
        # only one that never asks for upload consent, because the photograph
        # never leaves the computer. That ordering is the recommendation.
        self._kind = QComboBox()
        self._kind.addItem("On this machine (local server)", "local_server")
        self._kind.addItem("Custom endpoint (JSON contract)", "http")
        current_kind = str(current.get("provider", "http") or "http")
        self._kind.setCurrentIndex(0 if current_kind in ("local_server", "mlx") else 1)
        self._kind.currentIndexChanged.connect(self._on_kind_changed)
        kind_form = QFormLayout()
        kind_form.setSpacing(8)
        kind_form.addRow("Run with", self._kind)
        outer.addLayout(kind_form)

        self._local_help = QLabel(
            "Talks to a model running on this Mac over localhost — the photo "
            "never leaves the machine, so no upload consent is needed.\n\n"
            "Requires a server exposing the OpenAI image-edit API. MLX Core "
            "does this and runs Mage-Flow-Edit-Turbo natively on Apple "
            "Silicon:\n"
            "    brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve\n"
            "    brew install --cask mlx-core"
        )
        self._local_help.setWordWrap(True)
        self._local_help.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")
        outer.addWidget(self._local_help)

        form = QFormLayout()
        form.setSpacing(8)

        from raw_generative_local_server import DEFAULT_ENDPOINT, DEFAULT_MODEL

        self._default_local_endpoint = DEFAULT_ENDPOINT
        self._default_local_model = DEFAULT_MODEL
        if self._kind.currentData() == "local_server":
            shown = current.get("server_endpoint", "") or DEFAULT_ENDPOINT
        else:
            shown = current.get("endpoint", "")
        self._endpoint = QLineEdit(shown)
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

        self._contract = QLabel(
            "The endpoint receives JSON: {\"instruction\", \"image\" (base64 "
            "PNG), \"seed\", \"options\"} and must reply with "
            "{\"image\": \"<base64 PNG>\"} or {\"error\": \"...\"}."
        )
        self._contract.setWordWrap(True)
        self._contract.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")
        outer.addWidget(self._contract)

        # Show the right half of the form for whichever kind is selected.
        self._on_kind_changed()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _on_kind_changed(self) -> None:
        """Swap the form between a local server and a custom endpoint."""
        local = self._kind.currentData() == "local_server"
        self._local_help.setVisible(local)
        self._contract.setVisible(not local)
        self._insecure.setVisible(not local)
        if local:
            self._endpoint.setPlaceholderText(self._default_local_endpoint)
            if not self._endpoint.text().strip():
                self._endpoint.setText(self._default_local_endpoint)
            self._model.setPlaceholderText(self._default_local_model)
        else:
            self._endpoint.setPlaceholderText("https://your-server.example/edit")
            if self._endpoint.text().strip() == self._default_local_endpoint:
                self._endpoint.clear()
            self._model.setPlaceholderText("recorded in the file's provenance")

    def _on_save(self) -> None:
        from generative_settings import save_server_settings, save_settings

        endpoint = self._endpoint.text().strip()
        if self._kind.currentData() == "local_server":
            # No consent bookkeeping: a loopback endpoint uploads nothing, and
            # LocalServerProvider computes requires_consent from the URL, so a
            # user who later points this at a remote host is still asked.
            save_server_settings(
                endpoint=endpoint or self._default_local_endpoint,
                model_name=self._model.text().strip() or self._default_local_model,
                api_key=self._api_key.text(),
            )
        else:
            # save_settings revokes consent when the endpoint changes, so a
            # new destination always re-asks before anything is uploaded.
            save_settings(
                endpoint=endpoint,
                api_key=self._api_key.text(),
                model_name=self._model.text().strip() or "remote",
                allow_insecure=self._insecure.isChecked(),
            )
        self.accept()
