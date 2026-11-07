"""CustomWarningDialog: warning dialog."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from core import theme

from .confirm_dialog import _ConfirmDialogButton


class CustomWarningDialog(QDialog):
    """Custom warning dialog matching the CustomConfirmDialog MD3 styling."""

    _DIALOG_MIN_WIDTH = 420
    _DIALOG_MAX_WIDTH = 480
    _CONTENT_MARGIN_H = 24
    _CONTENT_MARGIN_V = 20

    def __init__(self, parent=None, title="Warning", message="", informative_text=""):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)

        self.container = QWidget(self)
        self.container.setObjectName("warningDialogContainer")
        self.container.setStyleSheet(f"""
            #warningDialogContainer {{
                background-color: {theme.VOID};
                border-radius: 12px;
                border: 1px solid {theme.LINE};
            }}
        """)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_widget = QWidget()
        content_widget.setObjectName("warningDialogContent")
        content_widget.setStyleSheet("#warningDialogContent { background-color: transparent; }")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(
            self._CONTENT_MARGIN_H,
            self._CONTENT_MARGIN_V,
            self._CONTENT_MARGIN_H,
            self._CONTENT_MARGIN_V,
        )
        content_layout.setSpacing(10)

        self._message_label = QLabel(message)
        self._message_label.setWordWrap(True)
        self._message_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._message_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._message_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.INK};
                font-size: 16px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px;
                margin: 0px;
            }}
        """)
        content_layout.addWidget(self._message_label)

        self._info_label = None
        if informative_text:
            self._info_label = QLabel(informative_text)
            self._info_label.setWordWrap(True)
            self._info_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            self._info_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
            self._info_label.setStyleSheet(f"""
                QLabel {{
                    color: {theme.INK_MUTED};
                    font-size: 14px;
                    font-family: 'Roboto', 'Segoe UI', sans-serif;
                    padding: 0px;
                    margin: 0px;
                }}
            """)
            content_layout.addWidget(self._info_label)

        content_layout.addSpacing(6)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)
        button_layout.addStretch(1)

        self.ok_btn = _ConfirmDialogButton("OK")
        self.ok_btn.setObjectName("warningOkBtn")
        self.ok_btn.setFixedHeight(40)
        self.ok_btn.setMinimumWidth(108)
        self.ok_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.ok_btn.setAutoDefault(True)
        self.ok_btn.setDefault(True)
        self.ok_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.ok_btn.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.ok_btn.setStyleSheet(f"""
            QPushButton#warningOkBtn {{
                background-color: transparent;
                color: {theme.INK};
                border: 1px solid {theme.LINE};
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 24px;
                outline: none;
            }}
            QPushButton#warningOkBtn:hover:!focus {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.05);
                border-color: {theme.INK_FAINT};
            }}
            QPushButton#warningOkBtn:focus {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.16);
                color: {theme.INK};
                border: 1px solid {theme.INK};
                outline: none;
            }}
            QPushButton#warningOkBtn:pressed {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.12);
            }}
        """)
        self.ok_btn.clicked.connect(self.accept)

        button_layout.addWidget(self.ok_btn)
        content_layout.addLayout(button_layout)
        main_layout.addWidget(content_widget)

        container_layout = QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.container)

        self._apply_dialog_size()

        if parent:
            center_global = parent.mapToGlobal(parent.rect().center())
            self.move(
                center_global.x() - self.width() // 2,
                center_global.y() - self.height() // 2,
            )

    def _apply_dialog_size(self) -> None:
        margin_h = self._CONTENT_MARGIN_H * 2
        inner_min = self._DIALOG_MIN_WIDTH - margin_h

        fm = self._message_label.fontMetrics()
        text_lines = [self._message_label.text() or ""]
        if self._info_label is not None:
            text_lines.extend((self._info_label.text() or "").splitlines())

        text_w = 0
        for line in text_lines:
            if line.strip():
                text_w = max(text_w, fm.horizontalAdvance(line))

        buttons_w = self.ok_btn.minimumWidth() + 12
        inner_w = max(inner_min, text_w, buttons_w)
        dialog_w = max(
            self._DIALOG_MIN_WIDTH,
            min(self._DIALOG_MAX_WIDTH, inner_w + margin_h),
        )
        inner_w = dialog_w - margin_h

        self._message_label.setMinimumWidth(inner_w)
        self._message_label.setMaximumWidth(inner_w)
        if self._info_label is not None:
            self._info_label.setMinimumWidth(inner_w)
            self._info_label.setMaximumWidth(inner_w)

        self.container.setMinimumWidth(dialog_w)
        self.container.setMaximumWidth(dialog_w)
        self.container.adjustSize()
        dialog_h = max(160, self.container.sizeHint().height())
        self.container.setFixedSize(dialog_w, dialog_h)
        self.setFixedSize(dialog_w, dialog_h)

    def showEvent(self, event):
        super().showEvent(event)
        self.ok_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.reject()
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
        if hasattr(self, '_dragging') and self._dragging:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, '_dragging'):
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
