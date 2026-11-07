"""TaskProgressDialog: background task progress dialog."""
from __future__ import annotations

import contextlib
import sys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from core import theme

from .confirm_dialog import _ConfirmDialogButton
from .macos import _activate_macos_foreground_app


class TaskProgressDialog(QDialog):
    """Frameless MD3-styled modal progress dialog with a Cancel button, used for
    long batch operations (single export, batch copy/delete/export). Chrome
    matches CustomConfirmDialog (stock QProgressDialog had a native title bar
    and a marquee bar with no real percentage)."""

    _DIALOG_WIDTH = 420
    _CONTENT_MARGIN_H = 24
    _CONTENT_MARGIN_V = 20

    canceled = pyqtSignal()

    def __init__(self, parent=None, title: str = "Export", message: str = ""):
        super().__init__(parent)
        flags = Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        if sys.platform == "darwin":
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self._app_state_connected = False

        self.container = QWidget(self)
        self.container.setObjectName("exportProgressContainer")
        self.container.setStyleSheet(f"""
            #exportProgressContainer {{
                background-color: {theme.VOID};
                border-radius: 12px;
                border: 1px solid {theme.LINE};
            }}
        """)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(
            self._CONTENT_MARGIN_H, self._CONTENT_MARGIN_V,
            self._CONTENT_MARGIN_H, self._CONTENT_MARGIN_V,
        )
        content_layout.setSpacing(12)

        self._message_label = QLabel(message)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.INK};
                font-size: 14px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px;
                margin: 0px;
            }}
        """)
        content_layout.addWidget(self._message_label)

        bar_row = QHBoxLayout()
        bar_row.setContentsMargins(0, 0, 0, 0)
        bar_row.setSpacing(10)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {theme.EMBER};
                border-radius: 3px;
            }}
        """)
        bar_row.addWidget(self._bar, 1)

        self._percent_label = QLabel("0%")
        self._percent_label.setFixedWidth(36)
        self._percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._percent_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.INK_MUTED};
                font-size: 12px;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }}
        """)
        bar_row.addWidget(self._percent_label)
        content_layout.addLayout(bar_row)

        content_layout.addSpacing(2)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        self.cancel_btn = _ConfirmDialogButton("Cancel")
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setMinimumWidth(96)
        self.cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.cancel_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.cancel_btn.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {theme.INK};
                border: 1px solid {theme.LINE};
                border-radius: 18px;
                font-size: 13px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 20px;
                outline: none;
            }}
            QPushButton:hover {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.05);
                border-color: {theme.INK_FAINT};
            }}
            QPushButton:pressed {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.12);
            }}
        """)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        button_row.addWidget(self.cancel_btn)
        content_layout.addLayout(button_row)

        main_layout.addWidget(content)

        container_layout = QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.container)

        self.container.setFixedWidth(self._DIALOG_WIDTH)
        self.container.adjustSize()
        h = max(120, self.container.sizeHint().height())
        self.container.setFixedHeight(h)
        self.setFixedSize(self._DIALOG_WIDTH, h)

        if parent:
            center_global = parent.mapToGlobal(parent.rect().center())
            self.move(
                center_global.x() - self.width() // 2,
                center_global.y() - self.height() // 2,
            )

        self._canceled = False

    def _on_cancel_clicked(self) -> None:
        self._canceled = True
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling…")
        self.canceled.emit()

    def wasCanceled(self) -> bool:
        return self._canceled

    def set_message(self, text: str) -> None:
        self._message_label.setText(text)

    def set_progress(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        self._bar.setValue(percent)
        self._percent_label.setText(f"{percent}%")

    def showEvent(self, event):
        super().showEvent(event)
        _activate_macos_foreground_app()
        self.raise_()
        self.activateWindow()
        app = QApplication.instance()
        if app and not self._app_state_connected:
            app.applicationStateChanged.connect(self._on_application_state_changed)
            self._app_state_connected = True

    def hideEvent(self, event):
        app = QApplication.instance()
        if app and self._app_state_connected:
            with contextlib.suppress(Exception):
                app.applicationStateChanged.disconnect(self._on_application_state_changed)
            self._app_state_connected = False
        super().hideEvent(event)

    def _on_application_state_changed(self, state) -> None:
        # Long-running export: unlike CustomConfirmDialog, losing app focus
        # (alt-tab while an export churns) must not auto-cancel the job.
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._on_cancel_clicked()
            event.accept()
            return
        super().keyPressEvent(event)

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
