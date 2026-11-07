"""MD3 dialog for optional MobileCLIP download (prompt + in-dialog progress)."""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class MobileCLIPDownloadDialog(QDialog):
    """Prompt to download MobileCLIP, then show progress in the same dialog."""

    download_requested = pyqtSignal()
    _DIALOG_WIDTH = 460

    def __init__(self, parent=None):
        super().__init__(parent)
        self._download_started = False

        flags = Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        if sys.platform == "darwin":
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)

        self.container = QWidget(self)
        self.container.setObjectName("mobileclip_download_container")
        self.container.setStyleSheet("""
            #mobileclip_download_container {
                background-color: #14120F;
                border: 1px solid #3A332A;
                border-radius: 12px;
            }
        """)

        outer = QVBoxLayout(self.container)
        outer.setContentsMargins(24, 22, 24, 22)
        outer.setSpacing(12)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_prompt_page())
        self._stack.addWidget(self._build_progress_page())
        self._stack.addWidget(self._build_error_page())
        outer.addWidget(self._stack)

        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.addWidget(self.container)

        self._apply_size_for_page(0)
        self._center_on_parent()

    @property
    def download_started(self) -> bool:
        return self._download_started

    def _title_style(self) -> str:
        return """
            QLabel {
                color: #EDE7DD;
                font-size: 17px;
                font-weight: 600;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }
        """

    def _body_style(self) -> str:
        return """
            QLabel {
                color: #96897A;
                font-size: 13px;
                line-height: 1.45;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }
        """

    def _note_style(self) -> str:
        return """
            QLabel {
                color: #665D50;
                font-size: 12px;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }
        """

    def _secondary_button_style(self) -> str:
        return """
            QPushButton {
                background-color: transparent;
                color: #96897A;
                border: 1px solid #4A4A4A;
                border-radius: 18px;
                font-size: 13px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 20px;
            }
            QPushButton:hover {
                color: #EDE7DD;
                background-color: rgba(255, 255, 255, 0.05);
                border-color: #D9691E;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """

    def _primary_button_style(self) -> str:
        return """
            QPushButton {
                background-color: #3A332A;
                color: #EDE7DD;
                border: 1px solid #4A4A4A;
                border-radius: 18px;
                font-size: 13px;
                font-weight: 600;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 20px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                border-color: #D9691E;
            }
            QPushButton:pressed {
                background-color: #2F2F2F;
            }
        """

    def _build_prompt_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Enable Semantic Search")
        title.setStyleSheet(self._title_style())
        layout.addWidget(title)

        message = QLabel(
            "Semantic search needs a one-time download of MobileCLIP models "
            "(~150 MB, internet required). Files are saved under your home folder "
            "and used offline after that."
        )
        message.setWordWrap(True)
        message.setStyleSheet(self._body_style())
        layout.addWidget(message)

        note = QLabel("You can still use EXIF-only search without downloading.")
        note.setWordWrap(True)
        note.setStyleSheet(self._note_style())
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 10, 0, 0)
        buttons.setSpacing(12)
        buttons.addStretch()

        exif_only_btn = QPushButton("EXIF Only")
        exif_only_btn.setFixedHeight(36)
        exif_only_btn.setMinimumWidth(110)
        exif_only_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        exif_only_btn.setStyleSheet(self._secondary_button_style())
        exif_only_btn.clicked.connect(self.reject)
        buttons.addWidget(exif_only_btn)

        download_btn = QPushButton("Download")
        download_btn.setFixedHeight(36)
        download_btn.setMinimumWidth(110)
        download_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        download_btn.setStyleSheet(self._primary_button_style())
        download_btn.clicked.connect(self._on_download_clicked)
        download_btn.setDefault(True)
        download_btn.setFocus()
        buttons.addWidget(download_btn)
        layout.addLayout(buttons)
        return page

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Downloading Models")
        title.setStyleSheet(self._title_style())
        layout.addWidget(title)

        self._progress_message = QLabel("Downloading AI models…")
        self._progress_message.setWordWrap(False)
        self._progress_message.setStyleSheet(self._body_style())
        layout.addWidget(self._progress_message)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #272219;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #D9691E;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._progress_bar)

        note = QLabel("Keep RAWviewer open until the download finishes.")
        note.setWordWrap(True)
        note.setStyleSheet(self._note_style())
        layout.addWidget(note)
        layout.addStretch()
        return page

    def _build_error_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Download Failed")
        title.setStyleSheet(self._title_style())
        layout.addWidget(title)

        self._error_message = QLabel("")
        self._error_message.setWordWrap(True)
        self._error_message.setStyleSheet(self._body_style())
        layout.addWidget(self._error_message)

        note = QLabel("EXIF-only search remains available. Try again later from Search.")
        note.setWordWrap(True)
        note.setStyleSheet(self._note_style())
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 10, 0, 0)
        buttons.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(36)
        close_btn.setMinimumWidth(110)
        close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close_btn.setStyleSheet(self._primary_button_style())
        close_btn.clicked.connect(self.reject)
        close_btn.setDefault(True)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        return page

    def _apply_size_for_page(self, index: int) -> None:
        heights = {0: 220, 1: 200, 2: 230}
        height = heights.get(index, 220)
        self.container.setFixedSize(self._DIALOG_WIDTH, height)
        self.setFixedSize(self._DIALOG_WIDTH, height)

    def _center_on_parent(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        geo = parent.geometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 2
        self.move(x, y)

    def _on_download_clicked(self) -> None:
        self._download_started = True
        self.download_requested.emit()
        # Progress lives in the gallery search field (same as indexing), not this dialog.
        self.accept()

    def set_download_progress(self, pct: int, message: str = "") -> None:
        pct = max(0, min(100, int(pct)))
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(pct)
        self._progress_bar.setFormat("%p%")
        self._progress_message.setText("Downloading AI models…")

    def set_progress_message(self, message: str) -> None:
        """Legacy hook — maps to determinate progress when possible."""
        text = (message or "").strip()
        if text:
            self._progress_message.setText("Downloading AI models…")

    def show_download_complete(self) -> None:
        self.accept()

    def show_download_error(self, error: str) -> None:
        self._error_message.setText((error or "Download failed.").strip())
        self._stack.setCurrentIndex(2)
        self._apply_size_for_page(2)
        self._center_on_parent()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if getattr(self, "_dragging", False):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, "_dragging"):
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
