"""TopMetadataBar: metadata strip under the custom title bar."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel

from core import theme


class TopMetadataBar(QFrame):
    """Centered EXIF strip for macOS (no custom title bar)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.VOID};
                border-bottom: 1px solid {theme.LINE};
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(0)
        layout.addStretch(1)
        self.metadata_label = QLabel("")
        self.metadata_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.metadata_label.setStyleSheet(f"""
            color: {theme.INK_MUTED};
            font-size: 12px;
            font-weight: 500;
            padding: 0px 8px;
            letter-spacing: 0.15px;
            background: transparent;
            border: none;
        """)
        self.metadata_label.hide()
        layout.addWidget(self.metadata_label, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        self.hide()

    def set_metadata(self, text: str) -> None:
        """Set metadata centered in the bar with dynamic elision."""
        display = (text or "").strip()
        if not display:
            self.metadata_label.setText("")
            self.metadata_label.hide()
            self.hide()
            self.metadata_label.setToolTip("")
            return

        avail_w = max(200, self.width() - 50)
        from PyQt6.QtGui import QFontMetrics
        fm = QFontMetrics(self.metadata_label.font())
        elided = fm.elidedText(display, Qt.TextElideMode.ElideMiddle, avail_w)
        self.metadata_label.setText(elided)
        self.metadata_label.setVisible(True)
        self.setVisible(True)
        self.metadata_label.setToolTip(display)
