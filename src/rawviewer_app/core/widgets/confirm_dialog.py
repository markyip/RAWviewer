"""_ConfirmDialogButton + CustomConfirmDialog: shared confirmation dialog."""
from __future__ import annotations

import contextlib
import sys
import time

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from core import theme

from .macos import _activate_macos_foreground_app


class _ConfirmDialogButton(QPushButton):
    """Pill button that uses stylesheet :focus for the whole control, not an inner focus rect."""
    pass


class CustomConfirmDialog(QDialog):
    """Material Design 3 style confirmation dialog with custom title bar."""

    _DIALOG_MIN_WIDTH = 420
    _DIALOG_MAX_WIDTH = 480
    _CONTENT_MARGIN_H = 24
    _CONTENT_MARGIN_V = 20

    def __init__(
        self,
        parent=None,
        title="Confirm Delete",
        message="",
        informative_text="",
        *,
        confirm_action: str = "Delete",
        confirm_label: str | None = None,
        cancel_label: str | None = None,
        body_widget=None,
    ):
        super().__init__(parent)
        flags = Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        if sys.platform == "darwin":
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self._app_state_connected = False
        self._ignore_app_inactive_until = 0.0
        if confirm_label:
            confirm_action = confirm_label

        self.container = QWidget(self)
        self.container.setObjectName("confirmDialogContainer")
        self.container.setStyleSheet(f"""
            #confirmDialogContainer {{
                background-color: {theme.VOID};
                border-radius: 12px;
                border: 1px solid {theme.LINE};
            }}
        """)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_widget = QWidget()
        content_widget.setObjectName("confirmDialogContent")
        content_widget.setStyleSheet("#confirmDialogContent { background-color: transparent; }")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(
            self._CONTENT_MARGIN_H,
            self._CONTENT_MARGIN_V,
            self._CONTENT_MARGIN_H,
            self._CONTENT_MARGIN_V,
        )
        content_layout.setSpacing(10)
        self._content_layout = content_layout

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

        # Optional extra UI (options, inputs) between the text and the buttons;
        # subclasses like BatchExportDialog build their whole body this way.
        self._body_widget = body_widget
        if body_widget is not None:
            content_layout.addWidget(body_widget)

        content_layout.addSpacing(6)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)
        button_layout.addStretch(1)
        self._buttons_layout = button_layout

        self.cancel_btn = _ConfirmDialogButton(cancel_label or "Cancel")
        self.cancel_btn.setObjectName("confirmCancelBtn")
        self.cancel_btn.setFixedHeight(40)
        self.cancel_btn.setMinimumWidth(108)
        self.cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.cancel_btn.setAutoDefault(True)
        self.cancel_btn.setDefault(True)
        self.cancel_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.cancel_btn.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton#confirmCancelBtn {{
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
            QPushButton#confirmCancelBtn:hover:!focus {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.05);
                border-color: {theme.INK_FAINT};
            }}
            QPushButton#confirmCancelBtn:focus {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.16);
                color: {theme.INK};
                border: 1px solid {theme.INK};
                outline: none;
            }}
            QPushButton#confirmCancelBtn:pressed {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.12);
            }}
        """)
        self.cancel_btn.clicked.connect(self.reject)

        action_label = (confirm_action or "Delete").strip()
        is_discard = action_label.lower() == "discard"
        self.delete_btn = _ConfirmDialogButton(action_label)
        if is_discard:
            self.delete_btn.setObjectName("confirmDiscardBtn")
            # Base rule uses chrome tokens; hover/focus/pressed keep the literal
            # amber warning colors as-is (destructive/warning colors are out of
            # scope for the darkroom palette migration).
            confirm_style = f"""
            QPushButton#confirmDiscardBtn {{
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
            QPushButton#confirmDiscardBtn:hover:!focus {{
                background-color: rgba(255, 183, 77, 0.12);
                border-color: #FFB74D;
                color: #FFE082;
            }}
            QPushButton#confirmDiscardBtn:focus {{
                background-color: #FFB74D;
                color: #1E1E1E;
                border: 1px solid #FFB74D;
                outline: none;
            }}
            QPushButton#confirmDiscardBtn:pressed {{
                background-color: #FFA726;
            }}
            """
        else:
            self.delete_btn.setObjectName("confirmDeleteBtn")
            # Base rule uses chrome tokens; hover/focus/pressed keep the literal
            # destructive-red colors as-is (out of scope for this migration).
            confirm_style = f"""
            QPushButton#confirmDeleteBtn {{
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
            QPushButton#confirmDeleteBtn:hover:!focus {{
                background-color: rgba(255, 82, 82, 0.12);
                border-color: #FF5252;
                color: #FF8A80;
            }}
            QPushButton#confirmDeleteBtn:focus {{
                background-color: #FF5252;
                color: #FFFFFF;
                border: 1px solid #FF5252;
                outline: none;
            }}
            QPushButton#confirmDeleteBtn:pressed {{
                background-color: #FF4444;
            }}
            """
        self.delete_btn.setFixedHeight(40)
        self.delete_btn.setMinimumWidth(108)
        self.delete_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.delete_btn.setAutoDefault(False)
        self.delete_btn.setDefault(False)
        self.delete_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.delete_btn.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.delete_btn.setStyleSheet(confirm_style)
        self.delete_btn.clicked.connect(self.accept)

        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.delete_btn)
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

        self.result_value = False

    def insert_body_widget(self, body_widget) -> None:
        """Insert an extra widget between the text and the button row.

        Used after construction (BatchExportDialog builds its body once the
        base dialog exists); the body_widget= init argument covers the
        pre-construction case.
        """
        lay = getattr(self, "_content_layout", None)
        holder = getattr(self, "_buttons_layout", None)
        if lay is not None:
            lay.insertWidget(max(0, lay.indexOf(holder)) if holder is not None else 0, body_widget)
            self._body_widget = body_widget
            self._apply_dialog_size()

    def _apply_dialog_size(self) -> None:
        """Size from the actual laid-out content.

        QLabel sizeHint alone is too narrow when wrapped, and reading the
        container hint before its layout ran returned a stale height -- which
        clipped everything a subclass inserted after construction (the export
        dialog's radio/format options ended up invisible).
        """
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

        buttons_w = self.cancel_btn.minimumWidth() + self.delete_btn.minimumWidth() + 12
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
        body = getattr(self, "_body_widget", None)
        if body is not None:
            body.setMinimumWidth(inner_w)
            body.setMaximumWidth(inner_w)

        self.container.setMinimumWidth(dialog_w)
        self.container.setMaximumWidth(dialog_w)

        # Activate every layout before measuring: sizeHint() stays stale until
        # the layout has actually run (that is what clipped the options body).
        outer = self.container.layout()
        content = None
        if outer is not None and outer.count():
            content = outer.itemAt(0).widget()
        if content is not None and content.layout() is not None:
            content.layout().activate()
        if outer is not None:
            outer.activate()
        self.container.adjustSize()
        dialog_h = max(160, self.container.sizeHint().height())
        self.container.setFixedSize(dialog_w, dialog_h)
        self.setFixedSize(dialog_w, dialog_h)

    def showEvent(self, event):
        super().showEvent(event)
        # Re-fit after the real font/layout pass; bodies added after
        # construction must never stay clipped inside a stale fixed size.
        self._apply_dialog_size()
        _activate_macos_foreground_app()
        self.raise_()
        self.activateWindow()
        self.cancel_btn.setFocus(Qt.FocusReason.OtherFocusReason)
        # macOS: parent deactivation briefly marks the app inactive when a modal opens.
        self._ignore_app_inactive_until = time.monotonic() + 0.75
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
        # Do not auto-close on ApplicationInactive — macOS fires that when the modal
        # opens and the main window loses activation (looks like Delete does nothing).
        if not self.isVisible():
            return
        if state == Qt.ApplicationState.ApplicationHidden:
            self.reject()
            return
        if state == Qt.ApplicationState.ApplicationInactive:
            if time.monotonic() < getattr(self, "_ignore_app_inactive_until", 0.0):
                return
            # Ignore transient inactive while this dialog is the active modal.
            if self.isActiveWindow() or self.isVisible():
                return
            self.reject()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.reject()
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            focused = self.focusWidget()
            if focused in (self.delete_btn, self.cancel_btn):
                focused.click()
                event.accept()
                return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            if self.cancel_btn.hasFocus() and key == Qt.Key.Key_Right:
                self.delete_btn.setFocus(Qt.FocusReason.TabFocusReason)
                event.accept()
                return
            if self.delete_btn.hasFocus() and key == Qt.Key.Key_Left:
                self.cancel_btn.setFocus(Qt.FocusReason.TabFocusReason)
                event.accept()
                return
        super().keyPressEvent(event)

    def accept(self):
        self.result_value = True
        super().accept()

    def reject(self):
        self.result_value = False
        super().reject()

    def mousePressEvent(self, event):
        """Allow dragging the dialog"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle dragging"""
        if hasattr(self, '_dragging') and self._dragging:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Stop dragging"""
        if hasattr(self, '_dragging'):
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
