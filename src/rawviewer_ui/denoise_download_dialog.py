"""Dialog and worker for downloading the AI Denoise model."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
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


class DenoiseDownloadWorker(QThread):
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, dest_path: str, parent=None):
        super().__init__(parent)
        self.dest_path = dest_path
        self.url = "https://github.com/Phhofm/models/releases/download/1xDeNoise_realplksr_otf/1xDeNoise_realplksr_otf.safetensors"

    def run(self):
        dest = Path(self.dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest.with_suffix(".tmp")
        try:
            req = urllib.request.Request(self.url, headers={"User-Agent": "RAWviewer/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                chunk_size = 1024 * 64
                with open(temp_dest, "wb") as out:
                    while True:
                        if self.isInterruptionRequested():
                            temp_dest.unlink(missing_ok=True)
                            return
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total > 0:
                            frac = min(1.0, done / total)
                            self.progress.emit(int(frac * 100))
            if temp_dest.exists():
                if dest.exists():
                    dest.unlink()
                temp_dest.rename(dest)
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)
            self.error.emit(str(e))


class DenoiseModelDownloadDialog(QDialog):
    """Prompt to download AI Denoise model, showing progress in the dialog."""

    download_complete = pyqtSignal()
    _DIALOG_WIDTH = 460

    def __init__(self, dest_path: str, parent=None):
        super().__init__(parent)
        self.dest_path = dest_path
        self._worker = None

        flags = Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        if sys.platform == "darwin":
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)

        self.container = QWidget(self)
        self.container.setObjectName("denoise_download_container")
        self.container.setStyleSheet("""
            #denoise_download_container {
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

        title = QLabel("Enable AI Denoise")
        title.setStyleSheet(self._title_style())
        layout.addWidget(title)

        message = QLabel(
            "AI denoise requires a one-time download of the realPLKSR model "
            "(~1 MB, internet required). It will be saved locally for offline use."
        )
        message.setWordWrap(True)
        message.setStyleSheet(self._body_style())
        layout.addWidget(message)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 10, 0, 0)
        buttons.setSpacing(12)
        buttons.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setMinimumWidth(110)
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.setStyleSheet(self._secondary_button_style())
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        download_btn = QPushButton("Download")
        download_btn.setFixedHeight(36)
        download_btn.setMinimumWidth(110)
        download_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        download_btn.setStyleSheet(self._primary_button_style())
        download_btn.clicked.connect(self._start_download)
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

        title = QLabel("Downloading Model")
        title.setStyleSheet(self._title_style())
        layout.addWidget(title)

        self._progress_message = QLabel("Downloading 1xDeNoise_realplksr_otf…")
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

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 10, 0, 0)
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setMinimumWidth(110)
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.setStyleSheet(self._secondary_button_style())
        cancel_btn.clicked.connect(self._cancel_download)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

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
        heights = {0: 180, 1: 180, 2: 180}
        height = heights.get(index, 180)
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

    def _start_download(self) -> None:
        self._stack.setCurrentIndex(1)
        self._apply_size_for_page(1)
        self._center_on_parent()
        
        self._worker = DenoiseDownloadWorker(self.dest_path, parent=self)
        self._worker.progress.connect(self._progress_bar.setValue)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_download(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait()
        self.reject()

    def _on_error(self, err_msg: str) -> None:
        self._error_message.setText(f"Failed to download model:\\n{err_msg}")
        self._stack.setCurrentIndex(2)
        self._apply_size_for_page(2)
        self._center_on_parent()

    def _on_finished(self) -> None:
        self.download_complete.emit()
        self.accept()

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
