"""Share-to-AI dialog: per-photo or group description, optional sidecar save."""

from __future__ import annotations

import os

from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

import theme
from rawviewer_share_agent import (
    DESCRIPTION_MAX,
    MODE_GENERATE,
    MODE_MCP_EDIT,
    clamp_description,
    description_template_from_store,
    generate_hook_status,
    member_description_lines,
    suggest_share_description,
)
from rawviewer_ui.export_options_dialog import _form_row_buttons


class DescriptionTemplateDialog(QDialog):
    """Optional starter text. Not applied to every share automatically."""

    def __init__(self, parent=None, *, template: str = "", style_brief: str = ""):
        super().__init__(parent)
        self.setWindowTitle("AI description template")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setStyleSheet(theme.form_dialog_stylesheet())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        hint = QLabel(
            "Optional starter text you can insert when sharing. "
            "Each photo or selection has its own description — this is not "
            "applied automatically."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 12px;")
        layout.addWidget(hint)

        self.brief_edit = QPlainTextEdit()
        self.brief_edit.setPlaceholderText(
            "e.g. keep skin natural, never crush blacks"
        )
        self.brief_edit.setPlainText(clamp_description(template or style_brief))
        self.brief_edit.setMinimumHeight(140)
        layout.addWidget(self.brief_edit)

        layout.addLayout(_form_row_buttons(self, ok_text="Save"))

    def template_text(self) -> str:
        return clamp_description(self.brief_edit.toPlainText())[:DESCRIPTION_MAX]

    def style_brief(self) -> str:
        return self.template_text()


StyleBriefDialog = DescriptionTemplateDialog


class ShareToAgentDialog(QDialog):
    """Per-photo or group description for this share."""

    def __init__(
        self,
        parent=None,
        *,
        paths: list[str] | None = None,
        photo_name: str = "",
        template: str = "",
        style_brief: str = "",
        extra_count: int = 0,
        settings_store=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Share to AI agent")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setStyleSheet(theme.form_dialog_stylesheet())
        self._paths = [p for p in (paths or []) if p]
        self._n = len(self._paths) or (1 if photo_name else 0)
        self._template = clamp_description(
            template or style_brief
        ) or description_template_from_store(settings_store)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        if self._n <= 1:
            name = photo_name or (
                os.path.basename(self._paths[0]) if self._paths else "Current photo"
            )
            if extra_count > 0:
                name = f"{name}  ·  plus {extra_count} more"
            summary = name
            field_label = "Description for this photo"
            save_label = "Save on this photo"
            placeholder = (
                "e.g. keep the neon signs  ·  wedding — don’t oversharpen"
            )
        else:
            summary = f"{self._n} photos selected"
            field_label = "Description for this group"
            save_label = "Save on these photos"
            placeholder = (
                "e.g. this wedding set: stay warm, don’t oversharpen"
            )
        title = QLabel(summary)
        title.setWordWrap(True)
        title.setStyleSheet(
            f"color: {theme.INK}; font-size: 17px; font-weight: 500;"
        )
        layout.addWidget(title)

        hint = QLabel(
            "Sends preview JPEG(s) plus paths. The agent edits via MCP (XMP), "
            "not by overwriting the RAW. Instructions are for this photo or "
            "selection — not an app-wide preset."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 12px;")
        layout.addWidget(hint)

        mode_lbl = QLabel("What should the agent do?")
        mode_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 12px; font-weight: 500;")
        layout.addWidget(mode_lbl)
        self.mcp_radio = QRadioButton("Adjust in RAWviewer (MCP)")
        self.mcp_radio.setToolTip(
            "Headless XMP edits: get_preview / set_adjustments / export_jpeg"
        )
        self.gen_radio = QRadioButton("Generate a new image")
        gen_ok, gen_reason = generate_hook_status()
        self.gen_radio.setEnabled(gen_ok)
        if gen_ok:
            self.gen_radio.setToolTip(
                "The agent may use its own image generation. "
                "Saves a new JPEG next to the source — never overwrites the RAW."
            )
        else:
            self.gen_radio.setToolTip(gen_reason)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mcp_radio)
        self._mode_group.addButton(self.gen_radio)
        self.mcp_radio.setChecked(True)
        layout.addWidget(self.mcp_radio)
        layout.addWidget(self.gen_radio)
        if not gen_ok:
            reason = QLabel(gen_reason)
            reason.setWordWrap(True)
            reason.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 12px;")
            layout.addWidget(reason)

        desc_lbl = QLabel(field_label)
        desc_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 12px; font-weight: 500;")
        layout.addWidget(desc_lbl)
        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setPlaceholderText(placeholder)
        self.desc_edit.setMinimumHeight(120)
        prefill = suggest_share_description(self._paths)
        self.desc_edit.setPlainText(prefill)
        layout.addWidget(self.desc_edit)

        insert_row = QHBoxLayout()
        insert_row.setContentsMargins(0, 0, 0, 0)
        insert_row.addStretch(1)
        insert_btn = QPushButton("Insert template")
        insert_btn.setMinimumHeight(32)
        insert_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        insert_btn.setStyleSheet(theme.form_secondary_button_stylesheet())
        insert_btn.setEnabled(bool(self._template))
        insert_btn.setToolTip(
            "Paste the optional saved template into this description"
            if self._template
            else "No template saved — File → AI description template…"
        )
        insert_btn.clicked.connect(self._insert_template)
        insert_row.addWidget(insert_btn)
        layout.addLayout(insert_row)

        if self._n > 1:
            existing = member_description_lines(self._paths)
            if existing:
                members = QLabel(
                    "Already saved on some photos:\n" + "\n".join(existing)
                )
                members.setWordWrap(True)
                members.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 12px;")
                layout.addWidget(members)

        self.save_radio = QRadioButton(save_label)
        self.share_only_radio = QRadioButton("Use for this share only")
        self._persist_group = QButtonGroup(self)
        self._persist_group.addButton(self.share_only_radio)
        self._persist_group.addButton(self.save_radio)
        if self._n <= 1:
            self.save_radio.setChecked(True)
        else:
            self.share_only_radio.setChecked(True)
        layout.addWidget(self.share_only_radio)
        layout.addWidget(self.save_radio)

        layout.addLayout(_form_row_buttons(self, ok_text="Share"))

    def _insert_template(self) -> None:
        if not self._template:
            return
        current = self.desc_edit.toPlainText().strip()
        if not current:
            self.desc_edit.setPlainText(self._template)
            return
        self.desc_edit.setPlainText(current + "\n" + self._template)

    def description(self) -> str:
        return clamp_description(self.desc_edit.toPlainText())[:DESCRIPTION_MAX]

    def save_on_photos(self) -> bool:
        return bool(self.save_radio.isChecked())

    def mode(self) -> str:
        if self.gen_radio.isEnabled() and self.gen_radio.isChecked():
            return MODE_GENERATE
        return MODE_MCP_EDIT

    def note(self) -> str:
        return self.description()

    def style_brief(self) -> str:
        return ""
