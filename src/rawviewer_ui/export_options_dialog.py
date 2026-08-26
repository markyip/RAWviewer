"""Shared export options: filename template, watermark, GPS, queue destination."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from export_options import (
    DEFAULT_FILENAME_TEMPLATE,
    DEFAULT_SIGNATURE_FONT,
    DEFAULT_WATERMARK_CORNER,
    MAX_SIGNATURE_NAME,
    MAX_WATERMARK_REMARK,
    format_export_filename,
    list_signature_fonts,
    list_watermark_corners,
    normalize_signature_font_id,
    normalize_watermark_corner,
    queue_ai_export_availability,
    template_context_for_path,
)
import theme


def _form_row_buttons(parent: QDialog, *, ok_text: str, cancel_text: str = "Cancel") -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 8, 0, 0)
    row.setSpacing(8)
    row.addStretch(1)

    cancel_btn = QPushButton(cancel_text, parent)
    cancel_btn.setMinimumHeight(36)
    cancel_btn.setMinimumWidth(96)
    cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    cancel_btn.setStyleSheet(theme.form_secondary_button_stylesheet())
    cancel_btn.clicked.connect(parent.reject)
    row.addWidget(cancel_btn)

    ok_btn = QPushButton(ok_text, parent)
    ok_btn.setMinimumHeight(36)
    ok_btn.setMinimumWidth(108)
    ok_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    ok_btn.setDefault(True)
    ok_btn.setStyleSheet(theme.form_primary_button_stylesheet())
    ok_btn.clicked.connect(parent.accept)
    row.addWidget(ok_btn)
    return row


class ExportOptionsForm(QWidget):
    def __init__(self, parent=None, *, show_format: bool = False, show_directory: bool = False):
        super().__init__(parent)
        self._show_format = show_format
        self._show_directory = show_directory
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.template_edit = QLineEdit()
        self.template_edit.setPlaceholderText(DEFAULT_FILENAME_TEMPLATE)
        self.template_edit.setToolTip(
            "{name} {ext} {seq} {yyyy} {mm} {dd} {hh} {min} {ss} {rating} {label}"
        )
        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 12px;")
        self.template_edit.textChanged.connect(self._refresh_preview)

        self.strip_gps = QCheckBox("Strip GPS from exported metadata")
        self.strip_gps.setChecked(True)
        self.copy_metadata = QCheckBox("Copy camera metadata (EXIF)")
        self.copy_metadata.setChecked(True)

        self.signature_name = QLineEdit()
        self.signature_name.setPlaceholderText("Your name")
        self.signature_name.setMaxLength(MAX_SIGNATURE_NAME)
        self.signature_name.setToolTip("Leave empty for no signature. Can be used with a remark.")
        self.signature_font = QComboBox()
        for font_id, label in list_signature_fonts():
            self.signature_font.addItem(label, font_id)

        self.watermark_text = QLineEdit()
        self.watermark_text.setPlaceholderText("© your name  ·  proof")
        self.watermark_text.setMaxLength(MAX_WATERMARK_REMARK)
        self.watermark_text.setToolTip("Optional one-line caption. Can be used with a signature.")

        self.watermark_corner = QComboBox()
        for corner_id, label in list_watermark_corners():
            self.watermark_corner.addItem(label, corner_id)
        br = self.watermark_corner.findData(DEFAULT_WATERMARK_CORNER)
        if br >= 0:
            self.watermark_corner.setCurrentIndex(br)
        self.watermark_corner.setToolTip("Signature and remark share this corner as one block.")

        layout.addRow("Filename", self.template_edit)
        layout.addRow("", self.preview_label)
        layout.addRow(self.copy_metadata)
        layout.addRow(self.strip_gps)
        layout.addRow("Signature", self.signature_name)
        layout.addRow("Font", self.signature_font)
        layout.addRow("Remark", self.watermark_text)
        layout.addRow("Position", self.watermark_corner)

        if show_format:
            self.format_combo = QComboBox()
            self.format_combo.addItem("JPEG", "jpeg")
            self.format_combo.addItem("WebP", "webp")
            self.format_combo.addItem("16-bit TIFF", "tiff16")
            layout.addRow("Format", self.format_combo)
            denoise_ready, upscale_ready = queue_ai_export_availability()
            self.ai_denoise = QCheckBox("AI denoise (SCUNet) — this queue only")
            self.ai_denoise.setToolTip(
                "Uses AI instead of the sidecar Chroma NR method. Same Plus gate as Adjust export."
            )
            self.ai_upscale = QCheckBox("AI upscale 2× (Real-ESRGAN) — this queue only")
            self.ai_upscale.setToolTip("Exports at double width and height.")
            self.ai_denoise.setEnabled(denoise_ready)
            self.ai_upscale.setEnabled(upscale_ready)
            if not denoise_ready:
                self.ai_denoise.setToolTip("AI denoise is unavailable (edition or model).")
            if not upscale_ready:
                self.ai_upscale.setToolTip("AI upscale is unavailable (edition or model).")
            if denoise_ready or upscale_ready:
                layout.addRow(self.ai_denoise)
                layout.addRow(self.ai_upscale)
        else:
            self.format_combo = None
            self.ai_denoise = None
            self.ai_upscale = None

        if show_directory:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            self.dir_edit = QLineEdit()
            browse = QPushButton("Browse…")
            browse.setMinimumHeight(32)
            browse.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            browse.setStyleSheet(theme.form_secondary_button_stylesheet())
            browse.clicked.connect(self._browse_dir)
            row.addWidget(self.dir_edit, 1)
            row.addWidget(browse)
            wrap = QWidget()
            wrap.setLayout(row)
            layout.addRow("Folder", wrap)
        else:
            self.dir_edit = None

        self._sample_path = "DSC_0001.CR3"
        self._refresh_preview()

    def _browse_dir(self) -> None:
        if self.dir_edit is None:
            return
        start = self.dir_edit.text().strip() or ""
        chosen = QFileDialog.getExistingDirectory(self, "Export folder", start)
        if chosen:
            self.dir_edit.setText(chosen)

    def _refresh_preview(self) -> None:
        fmt = "jpeg"
        if self.format_combo is not None:
            fmt = str(self.format_combo.currentData() or "jpeg")
        ctx = template_context_for_path(self._sample_path, export_format=fmt, seq=1)
        name = format_export_filename(self.template_edit.text(), ctx)
        self.preview_label.setText(f"Example: {name}")

    def set_values(self, prefs: dict) -> None:
        self.template_edit.setText(str(prefs.get("filename_template") or DEFAULT_FILENAME_TEMPLATE))
        self.strip_gps.setChecked(bool(prefs.get("strip_gps", True)))
        self.copy_metadata.setChecked(bool(prefs.get("copy_metadata", True)))
        self.watermark_text.setText(
            str(prefs.get("watermark_remark") or prefs.get("watermark_text") or "")
        )
        self.signature_name.setText(
            str(prefs.get("watermark_signature") or prefs.get("watermark_name") or "")
        )
        font_id = normalize_signature_font_id(
            str(prefs.get("watermark_font") or DEFAULT_SIGNATURE_FONT)
        )
        fidx = self.signature_font.findData(font_id)
        self.signature_font.setCurrentIndex(fidx if fidx >= 0 else 0)
        corner = normalize_watermark_corner(str(prefs.get("watermark_corner") or ""))
        cidx = self.watermark_corner.findData(corner)
        self.watermark_corner.setCurrentIndex(cidx if cidx >= 0 else 0)
        if self.format_combo is not None:
            fmt = str(prefs.get("queue_format") or "jpeg")
            idx = self.format_combo.findData(fmt)
            if idx >= 0:
                self.format_combo.setCurrentIndex(idx)
        if getattr(self, "ai_denoise", None) is not None:
            self.ai_denoise.setChecked(
                bool(prefs.get("queue_ai_denoise")) and self.ai_denoise.isEnabled()
            )
        if getattr(self, "ai_upscale", None) is not None:
            self.ai_upscale.setChecked(
                bool(prefs.get("queue_ai_upscale")) and self.ai_upscale.isEnabled()
            )
        if self.dir_edit is not None:
            self.dir_edit.setText(str(prefs.get("last_directory") or ""))

    def values(self) -> dict:
        fmt = "jpeg"
        if self.format_combo is not None:
            fmt = str(self.format_combo.currentData() or "jpeg")
        directory = ""
        if self.dir_edit is not None:
            directory = self.dir_edit.text().strip()
        remark = self.watermark_text.text().strip()[:MAX_WATERMARK_REMARK]
        name = self.signature_name.text().strip()[:MAX_SIGNATURE_NAME]
        font_id = normalize_signature_font_id(str(self.signature_font.currentData() or ""))
        corner = normalize_watermark_corner(str(self.watermark_corner.currentData() or ""))
        enabled = bool(remark or name)
        return {
            "filename_template": self.template_edit.text().strip() or DEFAULT_FILENAME_TEMPLATE,
            "strip_gps": self.strip_gps.isChecked(),
            "copy_metadata": self.copy_metadata.isChecked(),
            "watermark_kind": "on" if enabled else "off",
            "watermark_enabled": enabled,
            "watermark_text": remark,
            "watermark_signature": name,
            "watermark_name": name,
            "watermark_font": font_id,
            "watermark_corner": corner,
            "watermark_image": "",
            "queue_format": fmt,
            "queue_ai_denoise": bool(
                getattr(self, "ai_denoise", None) is not None
                and self.ai_denoise.isEnabled()
                and self.ai_denoise.isChecked()
            ),
            "queue_ai_upscale": bool(
                getattr(self, "ai_upscale", None) is not None
                and self.ai_upscale.isEnabled()
                and self.ai_upscale.isChecked()
            ),
            "last_directory": directory,
        }


class ExportOptionsDialog(QDialog):
    def __init__(self, parent=None, *, prefs: dict | None = None, queue_mode: bool = False, count: int = 1):
        super().__init__(parent)
        self.setWindowTitle("Export queue" if queue_mode else "Export options")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.queue_mode = queue_mode
        self.setStyleSheet(theme.form_dialog_stylesheet())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        hint = QLabel(
            f"Export {count} photo{'s' if count != 1 else ''} with the same naming and metadata rules."
            if queue_mode
            else "These options apply to this export. GPS is stripped by default."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.INK_MUTED}; font-size: 12px;")
        layout.addWidget(hint)
        self.form = ExportOptionsForm(self, show_format=queue_mode, show_directory=queue_mode)
        if prefs:
            self.form.set_values(prefs)
        layout.addWidget(self.form)
        ok_text = "Start export" if queue_mode else "Export"
        layout.addLayout(_form_row_buttons(self, ok_text=ok_text))

    def values(self) -> dict:
        return self.form.values()
