"""Shared Qt widgets."""
import logging
import os
import sys
import time
from typing import List

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QPoint, QRect, QEvent, QAbstractAnimation
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QPen, QCursor
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
    QMainWindow, QScrollArea, QApplication, QGridLayout, QToolButton, QSizePolicy,
    QGraphicsOpacityEffect,
)
import theme
from rawviewer_app.env import safe_print, _env_true


def _activate_macos_foreground_app() -> None:
    """Opt-in NSApplication foreground (RAWVIEWER_MACOS_FORCE_FOREGROUND=1). No-op elsewhere."""
    if sys.platform != "darwin" or not _env_true("RAWVIEWER_MACOS_FORCE_FOREGROUND"):
        return
    try:
        import objc

        ns_app = objc.lookUpClass("NSApplication").sharedApplication()
        if ns_app is not None:
            ns_app.setActivationPolicy_(0)
            ns_app.activateIgnoringOtherApps_(True)
    except Exception:
        pass


class ThumbnailLabel(QLabel):
    """
    Thumbnail widget - keeps original pixmap and rescales cleanly.
    Based on reference implementation: simple and reliable.
    """
    clicked = pyqtSignal(str) # file_path
    
    def __init__(self, parent=None, pixmap=None, file_path=None):
        super().__init__(parent)
        self.file_path = file_path
        self.original_pixmap = pixmap
        if pixmap:
            self.setPixmap(pixmap)
        else:
            self.setText("Loading...")  # Consistent with check in JustifiedGallery
        # Use setScaledContents(False) - like reference code for JustifiedGallery
        self.setScaledContents(False)
        # Use Fixed size policy - prevents layout from resizing
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.file_path:
            self.clicked.emit(self.file_path)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def set_original_pixmap(self, pixmap):
        """Store the original pixmap for rescaling"""
        self.original_pixmap = pixmap
    
    def get_original_pixmap(self):
        """Get the original pixmap"""
        return self.original_pixmap


# -----------------------------
# Signal carrier (thread → UI)
class ResizeGripIndicator(QWidget):
    """Bottom-right resize hint for frameless windows (overlay, not app chrome)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self._hint_visible = False
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.hide()

    def set_hint_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._hint_visible == visible:
            return
        self._hint_visible = visible
        self.update()

    def _viewer(self):
        return self.parent()

    def mousePressEvent(self, event):
        viewer = self._viewer()
        if (
            viewer is not None
            and hasattr(viewer, "_begin_resize_at_edge")
            and viewer._begin_resize_at_edge(event, "bottom_right")
        ):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        viewer = self._viewer()
        if (
            viewer is not None
            and hasattr(viewer, "_continue_resize_drag")
            and viewer._continue_resize_drag(event)
        ):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        viewer = self._viewer()
        if viewer is not None and hasattr(viewer, "_finish_resize_drag"):
            if viewer._finish_resize_drag():
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self.set_hint_visible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.set_hint_visible(False)
        super().leaveEvent(event)

    def paintEvent(self, event):
        if not self._hint_visible:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*theme.INK_MUTED_RGB, 100))
        spacing = 3
        dot = 2
        origin_x = self.width() - dot
        origin_y = self.height() - dot
        for row in range(3):
            count = 3 - row
            for col in range(count):
                x = origin_x - (count - 1 - col) * spacing
                y = origin_y - row * spacing
                painter.drawEllipse(x, y, dot, dot)
        painter.end()


class CustomTitleBar(QFrame):
    """Material Design 3 style custom title bar for frameless window."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(40)  # Smaller height
        
        # Use the same background color as image viewing area (theme.VOID)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.VOID};
                border-bottom: 1px solid {theme.LINE};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(0)
        
        # Logo Icon (Favicon)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(24, 24)  # Smaller icon
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("background-color: transparent; border: none;")
        
        # Load favicon - try multiple paths
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        icon_paths = [
            os.path.join(base_path, "icons", "favicon.ico"),
            os.path.join(base_path, "icons", "appicon.ico"),
            os.path.join(base_path, "icons", "appicon.png"),
            os.path.join(base_path, "favicon.ico"),
            os.path.join(base_path, "appicon.ico"),
            os.path.join(os.getcwd(), "icons", "favicon.ico"),
            os.path.join(os.getcwd(), "favicon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "favicon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico"),
            "icons/favicon.ico",
            "icons/appicon.ico",
            "favicon.ico",
            "appicon.ico"
        ]
        
        icon_loaded = False
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    icon = QIcon(icon_path)
                    pixmap = icon.pixmap(24, 24)
                    if not pixmap.isNull():
                        self.icon_label.setPixmap(pixmap)
                        icon_loaded = True
                        break
                except Exception:
                    continue
        
        if not icon_loaded:
            # Fallback to 'R' if favicon not found
            self.icon_label.setText("R")
            self.icon_label.setStyleSheet(f"""
                background-color: {theme.RAISED};
                color: {theme.INK};
                border-radius: 12px;
                font-weight: bold;
                font-size: 14px;
            """)
        layout.addWidget(self.icon_label)

        # Reserved for API compatibility; title bar shows icon only (no app name/version).
        self.title_label = QLabel("")
        self.title_label.hide()
        
        layout.addStretch(1)

        self.metadata_label = QLabel("")
        self.metadata_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.metadata_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
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
        
        # Window Controls - Smaller buttons
        import qtawesome as qta
        
        control_btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                width: 46px;
                height: 40px;
                margin: 0px;
                padding: 0px;
            }
            QPushButton:hover { 
                background-color: rgba(255, 255, 255, 0.1); 
            }
        """
        
        self.min_btn = QPushButton()
        self.min_btn.setIcon(qta.icon('fa5s.minus', color=theme.INK))
        self.min_btn.setIconSize(QSize(12, 12))
        self.min_btn.setStyleSheet(control_btn_style)
        self.min_btn.clicked.connect(self.parent.showMinimized)
        layout.addWidget(self.min_btn)
        
        self.max_btn = QPushButton()
        self.max_btn.setIcon(qta.icon('fa5.square', color=theme.INK))
        self.max_btn.setIconSize(QSize(12, 12))
        self.max_btn.setStyleSheet(control_btn_style)
        self.max_btn.clicked.connect(self._toggle_maximize)
        layout.addWidget(self.max_btn)
        
        self.close_btn = QPushButton()
        self.close_btn.setIcon(qta.icon('fa5s.times', color=theme.INK))
        self.close_btn.setIconSize(QSize(12, 12))
        self.close_btn.setStyleSheet(control_btn_style + "QPushButton:hover { background-color: #f44336; }")
        self.close_btn.clicked.connect(self.parent.close)
        layout.addWidget(self.close_btn)
        
        self._is_maximized = False
        self._dragging = False
        self._drag_pos = None

    def _sync_maximize_state(self) -> None:
        """Keep title-bar button/icon aligned with the real window state."""
        import qtawesome as qta

        maximized = bool(self.parent.isMaximized())
        self._is_maximized = maximized
        if maximized:
            self.max_btn.setIcon(qta.icon("fa5s.clone", color=theme.INK))
        else:
            self.max_btn.setIcon(qta.icon("fa5.square", color=theme.INK))

    def _toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()
        self._sync_maximize_state()
        
        # Trigger gallery layout update after maximize/restore
        if hasattr(self.parent, 'view_mode') and self.parent.view_mode == 'gallery':
            if hasattr(self.parent, 'gallery_justified') and self.parent.gallery_justified:
                from PyQt6.QtCore import QTimer
                # Wait a bit for window size to settle
                QTimer.singleShot(200, self.parent.gallery_justified.force_layout_update)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            parent = self.parent
            if parent is not None and hasattr(parent, "_title_bar_corner_resize_at"):
                pos = parent.mapFromGlobal(event.globalPosition().toPoint())
                if parent._title_bar_corner_resize_at(pos):
                    super().mousePressEvent(event)
                    return
            # Top frame strip is reserved for window resize (handled by parent event filter).
            if event.position().y() <= 10:
                super().mousePressEvent(event)
                return
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.parent.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and not self._is_maximized:
            self.parent.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
            event.accept()
    
    def set_title(self, title):
        """No-op: custom title bar displays icon only (window title uses setWindowTitle)."""
        return

    def set_metadata(self, text: str) -> None:
        """Center EXIF / image metadata in the title bar (SkySpotter-style) with dynamic elision."""
        display = (text or "").strip()
        if not display:
            self.metadata_label.setText("")
            self.metadata_label.hide()
            self.metadata_label.setToolTip("")
            return

        avail_w = max(200, self.width() - 250)
        from PyQt6.QtGui import QFontMetrics
        fm = QFontMetrics(self.metadata_label.font())
        elided = fm.elidedText(display, Qt.TextElideMode.ElideMiddle, avail_w)
        self.metadata_label.setText(elided)
        self.metadata_label.setVisible(True)
        self.metadata_label.setToolTip(display)


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

    def _apply_dialog_size(self) -> None:
        """Size from text/button metrics — QLabel sizeHint alone is too narrow when wrapped."""
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

        self.container.setMinimumWidth(dialog_w)
        self.container.setMaximumWidth(dialog_w)
        self.container.adjustSize()
        dialog_h = max(160, self.container.sizeHint().height())
        self.container.setFixedSize(dialog_w, dialog_h)
        self.setFixedSize(dialog_w, dialog_h)

    def showEvent(self, event):
        super().showEvent(event)
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
            try:
                app.applicationStateChanged.disconnect(self._on_application_state_changed)
            except Exception:
                pass
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


class ExportProgressDialog(QDialog):
    """Frameless MD3-styled modal progress dialog, matching CustomConfirmDialog's
    chrome (see that class) instead of the stock QProgressDialog's native
    title bar / OS-themed marquee bar."""

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
            try:
                app.applicationStateChanged.disconnect(self._on_application_state_changed)
            except Exception:
                pass
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


# -----------------------------
# Single-image area: full-bleed scroll + draggable histogram overlay
# -----------------------------
def _filmstrip_ui_env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


class SingleImageViewOverlay(QWidget):
    """Scroll area fills the widget; histogram + film strip float on top."""

    _HIST_MARGIN = 8
    _FILMSTRIP_HEIGHT = 88
    # Hot zone extends above the strip so reveal starts before the cursor reaches thumbnails.
    _FILMSTRIP_HOTZONE_EXTRA = 40
    _BOTTOM_HOTZONE = _FILMSTRIP_HEIGHT + _FILMSTRIP_HOTZONE_EXTRA
    _FILMSTRIP_FADE_MS = 200
    _FILMSTRIP_HIDE_DELAY_MS = 90
    _FILMSTRIP_SHOW_DELAY_MS = _filmstrip_ui_env_int(
        "RAWVIEWER_FILMSTRIP_SHOW_DELAY_MS", 500
    )
    _FILMSTRIP_SHOW_DELAY_DIRECT_MS = _filmstrip_ui_env_int(
        "RAWVIEWER_FILMSTRIP_SHOW_DELAY_DIRECT_MS", 70
    )

    def __init__(self, scroll_area, histogram_widget, viewer=None, parent=None,
                 gpu_view=None, map_widget=None, adjust_widget=None):
        super().__init__(parent)
        self._viewer = viewer
        self._scroll = scroll_area
        self._gpu_view = gpu_view
        self._hist = histogram_widget
        self._map = map_widget
        self._adj = adjust_widget
        self._hist_user_placed = False
        self._map_user_placed = False
        self._adj_user_placed = False
        self._filmstrip_pointer_active = False
        self._filmstrip_reveal = False
        scroll_area.setParent(self)
        if gpu_view is not None:
            gpu_view.setParent(self)
            gpu_view.installEventFilter(self)
            gpu_view.viewport().installEventFilter(self)
            # Legacy scroll area sits under the OpenGL surface; on macOS it can steal
            # mouse/gesture input unless hidden (see docs/macos-sharing-v21-v22.md).
            self.set_gpu_view_active(True)
        histogram_widget.setParent(self)
        if map_widget is not None:
            map_widget.setParent(self)
            map_widget.hide()
        if adjust_widget is not None:
            adjust_widget.setParent(self)
            adjust_widget.hide()
        self.setObjectName("single_view_container")
        self.setStyleSheet(f"#single_view_container {{ background-color: {theme.VOID}; }}")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)

        from rawviewer_ui.filmstrip_view import FilmStripBar
        # Host on the main window (not beside QOpenGLWidget) so the strip overlays the image.
        film_host = viewer if viewer is not None else self
        self._filmstrip_layer = QWidget(film_host)
        self._filmstrip_layer.setObjectName("filmstrip_layer")
        self._filmstrip_layer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._filmstrip_layer.hide()
        self._filmstrip = FilmStripBar(self._filmstrip_layer, viewer=viewer)
        self._filmstrip_opacity = QGraphicsOpacityEffect(self._filmstrip_layer)
        self._filmstrip_layer.setGraphicsEffect(self._filmstrip_opacity)
        self._filmstrip_opacity.setOpacity(0.0)
        self._filmstrip_fade_anim = None
        self._filmstrip_fade_gen = 0
        self._filmstrip.hide()
        self._hide_filmstrip_layer()
        self._hide_filmstrip_timer = QTimer(self)
        self._hide_filmstrip_timer.setSingleShot(True)
        self._hide_filmstrip_timer.timeout.connect(self._hide_filmstrip_if_inactive)
        self._show_filmstrip_timer = QTimer(self)
        self._show_filmstrip_timer.setSingleShot(True)
        self._show_filmstrip_timer.timeout.connect(self._show_filmstrip_if_still_in_hotzone)

        from PyQt6.QtWidgets import QLabel

        self.recovery_badge = QLabel(self)
        self.recovery_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Recovery preview literally lifts shadows / holds back highlights --
        # the two darkroom operations BURN's cool slate-blue represents.
        self.recovery_badge.setStyleSheet(f"""
            QLabel {{
                background-color: {theme.rgba(theme.BURN_RGB, 200)};
                color: {theme.INK};
                font-size: 12px;
                font-weight: 600;
                border-radius: 4px;
                padding: 5px 10px;
                border: 1px solid {theme.rgba(theme.BURN_RGB, 140)};
            }}
        """)
        self.recovery_badge.hide()

    def set_recovery_preview_badge(self, visible: bool, *, detail: str = "", loading: bool = False) -> None:
        badge = getattr(self, "recovery_badge", None)
        if badge is None:
            return
        if visible:
            if loading:
                text = "Loading recovery preview…"
            else:
                text = "Recovery preview"
                if detail:
                    text = f"{text}{detail}"
                text = f"{text} · fit only · P to exit"
            badge.setText(text)
            badge.adjustSize()
            badge.show()
            self.relayout_recovery_badge()
            badge.raise_()
        else:
            badge.hide()

    def relayout_recovery_badge(self) -> None:
        badge = getattr(self, "recovery_badge", None)
        if badge is None or not badge.isVisible():
            return
        margin = 14
        badge.move(max(margin, self.width() - badge.width() - margin), margin)
        badge.raise_()

    def _filmstrip_layer_visible(self) -> bool:
        layer = getattr(self, "_filmstrip_layer", None)
        return layer is not None and layer.isVisible()

    def _show_filmstrip_layer(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is not None:
            layer.show()
        self._filmstrip.show()
        self._layout_filmstrip_inner()
        self._raise_filmstrip_chrome()

    def _hide_filmstrip_layer(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is not None:
            layer.hide()
        self._filmstrip.hide()

    def _layout_filmstrip_inner(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return
        self._filmstrip.setGeometry(0, 0, max(1, layer.width()), max(1, layer.height()))

    def _raise_filmstrip_chrome(self) -> None:
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return
        if not layer.isVisible() and self._filmstrip_opacity_value() <= 0.0:
            return
        layer.raise_()
        layer.update()

    def _stop_filmstrip_fade(self) -> None:
        self._filmstrip_fade_gen += 1
        anim = getattr(self, "_filmstrip_fade_anim", None)
        if anim is not None:
            # Disconnect all signals BEFORE stop() to prevent the synchronous
            # finished emission from triggering _finish_hide during cancellation.
            try:
                anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                anim.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            if anim.state() == QAbstractAnimation.State.Running:
                anim.stop()
        self._filmstrip_fade_anim = None

    def _filmstrip_opacity_value(self) -> float:
        try:
            return float(self._filmstrip_opacity.opacity())
        except Exception:
            return 0.0

    def _filmstrip_accepts_pointer(self) -> bool:
        return (
            self._filmstrip_layer_visible()
            and self._filmstrip_opacity_value() > 0.35
        )

    def _update_filmstrip_hit_testing(self) -> None:
        accept = self._filmstrip_accepts_pointer()
        self._filmstrip.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, not accept
        )
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is not None:
            layer.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, not accept
            )

    def _animate_filmstrip_opacity(self, target: float, on_finished=None) -> None:
        target = max(0.0, min(1.0, float(target)))
        self._stop_filmstrip_fade()
        
        self._filmstrip_opacity.setOpacity(target)
        self._update_filmstrip_hit_testing()
        self._raise_filmstrip_chrome()
        
        if on_finished is not None:
            on_finished()

    def _fade_in_filmstrip(self) -> None:
        if not self._filmstrip.isEnabled():
            return
        self._show_filmstrip_timer.stop()
        self._hide_filmstrip_timer.stop()

        self._stop_filmstrip_fade()
        self._filmstrip_reveal = True

        if self._filmstrip_opacity_value() >= 0.99:
            return

        needs_setup = (
            not self._filmstrip_layer_visible()
            or self._filmstrip_opacity_value() <= 0.01
        )
        if needs_setup:
            self._filmstrip_opacity.setOpacity(0.0)
            self._show_filmstrip_layer()
            self._layout_filmstrip()
            if hasattr(self._filmstrip, "center_on_current"):
                QTimer.singleShot(0, lambda: self._filmstrip.center_on_current())
            if hasattr(self._filmstrip, "refresh_visible_thumbnails"):
                QTimer.singleShot(0, lambda: self._filmstrip.refresh_visible_thumbnails(
                    refresh_cache=True
                ))
        self._raise_filmstrip_chrome()
        self._animate_filmstrip_opacity(1.0)

    def _fade_out_filmstrip(self) -> None:
        if not self._filmstrip_layer_visible():
            return

        def _finish_hide():
            self._hide_filmstrip_layer()
            self._filmstrip_opacity.setOpacity(0.0)
            self._filmstrip_reveal = False
            self._update_filmstrip_hit_testing()
            if self._viewer is not None:
                self._viewer._filmstrip_pointer_active = False

        self._animate_filmstrip_opacity(0.0, on_finished=_finish_hide)

    def filmstrip_widget(self):
        return getattr(self, "_filmstrip", None)

    def set_filmstrip_pointer_active(self, active: bool) -> None:
        if active:
            if self._filmstrip_pointer_active:
                return
            self._filmstrip_pointer_active = True
            self._hide_filmstrip_timer.stop()
            self._fade_in_filmstrip()
            if self._viewer is not None:
                self._viewer._filmstrip_pointer_active = True
            return
        self._filmstrip_pointer_active = False
        if self._viewer is not None:
            self._viewer._filmstrip_pointer_active = False
        self._schedule_hide_filmstrip()

    def _schedule_hide_filmstrip(self) -> None:
        self._hide_filmstrip_timer.start(self._FILMSTRIP_HIDE_DELAY_MS)

    def _cancel_show_filmstrip(self) -> None:
        self._show_filmstrip_timer.stop()
        if (
            self._filmstrip_layer_visible()
            and self._filmstrip_opacity_value() <= 0.01
            and not self._filmstrip_pointer_active
        ):
            self._hide_filmstrip_layer()
            self._filmstrip_reveal = False
            self._update_filmstrip_hit_testing()

    def _prepare_filmstrip_reveal(self) -> None:
        """Warm layout and thumbnails while the hover delay runs (opacity stays 0)."""
        if not self._filmstrip.isEnabled():
            return
        if self._filmstrip_layer_visible() and self._filmstrip_opacity_value() > 0.01:
            return
        self._filmstrip_reveal = True
        self._filmstrip_opacity.setOpacity(0.0)
        self._show_filmstrip_layer()
        self._layout_filmstrip()
        self._update_filmstrip_hit_testing()
        if hasattr(self._filmstrip, "center_on_current"):
            self._filmstrip.center_on_current()
        if hasattr(self._filmstrip, "refresh_visible_thumbnails"):
            self._filmstrip.refresh_visible_thumbnails(refresh_cache=True)
        viewer = self._viewer
        if viewer is not None and hasattr(viewer, "_prefetch_filmstrip_thumbnails"):
            viewer._prefetch_filmstrip_thumbnails()

    def _schedule_show_filmstrip(self, delay_ms: int | None = None) -> None:
        if self._filmstrip_opacity_value() >= 0.99:
            return
        delay = (
            int(self._FILMSTRIP_SHOW_DELAY_MS)
            if delay_ms is None
            else max(0, int(delay_ms))
        )
        if self._show_filmstrip_timer.isActive():
            remaining = self._show_filmstrip_timer.remainingTime()
            if remaining >= 0 and remaining <= delay:
                return
        self._prepare_filmstrip_reveal()
        self._show_filmstrip_timer.start(delay)

    def _show_filmstrip_if_still_in_hotzone(self) -> None:
        if not self._filmstrip.isEnabled():
            return
        if self._filmstrip.underMouse() or self._pointer_in_bottom_hotzone():
            self._fade_in_filmstrip()

    def _hide_filmstrip_if_inactive(self) -> None:
        if self._filmstrip_pointer_active:
            return
        if self._pointer_in_bottom_hotzone():
            return
        self._filmstrip_reveal = False
        self._fade_out_filmstrip()

    def _pointer_in_bottom_hotzone(self) -> bool:
        pos = self.mapFromGlobal(QCursor.pos())
        return self._pointer_in_bottom_hotzone_at(pos)

    def _pointer_in_bottom_hotzone_at(self, pos: QPoint) -> bool:
        return self.rect().contains(pos) and (
            pos.y() >= self.height() - self._BOTTOM_HOTZONE
        )

    def _pointer_in_filmstrip_band_at(self, pos: QPoint) -> bool:
        """Bottom strip band (where thumbnails appear), not the wider approach zone."""
        return self.rect().contains(pos) and (
            pos.y() >= self.height() - self._FILMSTRIP_HEIGHT
        )

    def _local_pos_from_global(self, global_pos) -> QPoint:
        if hasattr(global_pos, "toPoint"):
            global_pos = global_pos.toPoint()
        return self.mapFromGlobal(global_pos)

    def _filmstrip_rect_in_container(self) -> QRect:
        """Filmstrip geometry mapped into this overlay's coordinate system."""
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return QRect()
        if (
            not layer.isVisible()
            and self._filmstrip_opacity_value() <= 0.0
            and not self._filmstrip_reveal
        ):
            return QRect()
        if layer.parentWidget() is self:
            return layer.geometry()
        global_top_left = layer.mapToGlobal(QPoint(0, 0))
        local_top_left = self.mapFromGlobal(global_top_left)
        return QRect(local_top_left, layer.size())

    def _ensure_filmstrip_enabled(self) -> None:
        """Enable/sync filmstrip before hover handling (GPU path may skip main eventFilter)."""
        viewer = self._viewer
        if viewer is not None and hasattr(viewer, "_ensure_filmstrip_synced"):
            viewer._ensure_filmstrip_synced()

    def handle_pointer_for_filmstrip(self, global_pos) -> None:
        """Reveal/hide film strip when pointer moves over scroll area (child widgets)."""
        self._ensure_filmstrip_enabled()
        if not self._filmstrip.isEnabled():
            return
        pos = self._local_pos_from_global(global_pos)
        in_hot = self._pointer_in_bottom_hotzone_at(pos)
        in_strip_band = self._pointer_in_filmstrip_band_at(pos)
        strip_rect = self._filmstrip_rect_in_container()
        over_strip = (
            self._filmstrip_accepts_pointer()
            and strip_rect.isValid()
            and strip_rect.contains(pos)
        )
        strip_visible = self._filmstrip_opacity_value() > 0.35
        
        # If the filmstrip is already visible and we are hovering over it, keep it visible instantly.
        is_hovering_visible_strip = strip_visible and (
            over_strip or self._filmstrip.underMouse() or (
                getattr(self, "_filmstrip_layer", None) is not None
                and self._filmstrip_layer.underMouse()
            )
        )
        
        if is_hovering_visible_strip:
            self._cancel_show_filmstrip()
            self._hide_filmstrip_timer.stop()
            self._fade_in_filmstrip()
        elif in_hot:
            self._hide_filmstrip_timer.stop()
            if strip_visible:
                # Keep it visible without delay if we are in the hotzone
                self._cancel_show_filmstrip()
                self._fade_in_filmstrip()
            else:
                # Force the full 500ms delay when the filmstrip is not visible,
                # ignoring the 70ms in_strip_band shortcut so it doesn't accidentally trigger.
                self._schedule_show_filmstrip(self._FILMSTRIP_SHOW_DELAY_MS)
        else:
            self._cancel_show_filmstrip()
            if not self._filmstrip_pointer_active:
                self._schedule_hide_filmstrip()

    def eventFilter(self, watched, event):
        from PyQt6.QtCore import QEvent

        gpu = self._gpu_view
        if gpu is not None and watched in (gpu, gpu.viewport()):
            if event.type() in (
                QEvent.Type.MouseMove,
                QEvent.Type.HoverMove,
                QEvent.Type.Enter,
            ):
                self.handle_pointer_for_filmstrip(event.globalPosition())
        return False

    def _reveal_filmstrip(self) -> None:
        """Backward-compatible alias for fade-in."""
        self._fade_in_filmstrip()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Scroll area stays full size; film strip overlays the bottom (no image reflow).
        self._scroll.setGeometry(0, 0, self.width(), self.height())
        self._scroll.lower()
        if self._gpu_view is not None:
            self._gpu_view.setGeometry(0, 0, self.width(), self.height())
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "_try_apply_pending_overlay_positions"):
            viewer._try_apply_pending_overlay_positions()
        self._layout_filmstrip()
        self._layout_histogram()
        self._layout_map()
        self._layout_adjust()
        self.relayout_recovery_badge()
        self._raise_single_view_layers()

    def set_gpu_view_active(self, active: bool) -> None:
        """Toggle GPU vs legacy scroll input/display (GIF/WebP use scroll only)."""
        gpu = self._gpu_view
        if gpu is None:
            return
        if active:
            gpu.show()
            self._scroll.hide()
        else:
            gpu.hide()
            self._scroll.show()
        self._raise_single_view_layers()

    def _raise_single_view_layers(self) -> None:
        """Keep image view under filmstrip / histogram chrome."""
        self._scroll.lower()
        if self._gpu_view is not None and self._gpu_view.isVisible():
            self._gpu_view.raise_()
        if hasattr(self, "recovery_badge") and self.recovery_badge.isVisible():
            self.recovery_badge.raise_()
        if self._map is not None and self._map.isVisible():
            self._map.raise_()
        if self._hist.isVisible():
            self._hist.raise_()
        if self._adj is not None and self._adj.isVisible():
            self._adj.raise_()
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "_restore_keyboard_focus"):
            QTimer.singleShot(0, viewer._restore_keyboard_focus)

    def _layout_adjust(self):
        panel = self._adj
        if panel is None or not panel.isVisible():
            return
        pw, ph = self.width(), self.height()
        aw, ah = panel.width(), panel.height()
        if pw < 1 or ph < 1:
            return
        if not self._adj_user_placed:
            x = max(self._HIST_MARGIN, pw - aw - self._HIST_MARGIN)
            y = max(self._HIST_MARGIN, (ph - ah) // 2)
            x = min(max(0, x), max(0, pw - aw))
            y = min(max(0, y), max(0, ph - ah))
            panel.move(x, y)
        else:
            x = min(max(0, panel.x()), max(0, pw - aw))
            y = min(max(0, panel.y()), max(0, ph - ah))
            panel.move(x, y)
        panel.raise_()

    def mark_adjust_user_moved(self):
        self._adj_user_placed = True
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "schedule_save_session_state"):
            viewer.schedule_save_session_state()

    def relayout_adjust(self):
        self._layout_adjust()

    def _layout_map(self):
        m = self._map
        if m is None or not m.isVisible():
            return
        pw, ph = self.width(), self.height()
        mw, mh = m.width(), m.height()
        if pw < 1 or ph < 1:
            return
        if not self._map_user_placed:
            x = self._HIST_MARGIN
            y = self._HIST_MARGIN
            x = min(max(0, x), max(0, pw - mw))
            y = min(max(0, y), max(0, ph - mh))
            m.move(x, y)
        else:
            x = min(max(0, m.x()), max(0, pw - mw))
            y = min(max(0, m.y()), max(0, ph - mh))
            m.move(x, y)
        m.raise_()

    def relayout_map(self):
        self._layout_map()

    def mark_map_user_moved(self):
        self._map_user_placed = True
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "schedule_save_session_state"):
            viewer.schedule_save_session_state()

    def overlay_session_snapshot(self) -> dict:
        """Normalized overlay positions (0–1) for cross-session restore."""
        pw, ph = max(1, self.width()), max(1, self.height())
        out: dict = {}
        if self._hist_user_placed:
            out["histogram"] = {"x": self._hist.x() / pw, "y": self._hist.y() / ph}
        if self._map is not None and self._map.isVisible() and self._map_user_placed:
            out["map"] = {"x": self._map.x() / pw, "y": self._map.y() / ph}
        if self._adj is not None and self._adj.isVisible() and self._adj_user_placed:
            out["adjust"] = {"x": self._adj.x() / pw, "y": self._adj.y() / ph}
        return out

    def apply_overlay_session_positions(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        pw, ph = max(1, self.width()), max(1, self.height())
        hist = data.get("histogram")
        if isinstance(hist, dict):
            try:
                x = int(float(hist["x"]) * pw)
                y = int(float(hist["y"]) * ph)
                hw, hh = self._hist.width(), self._hist.height()
                self._hist.move(
                    min(max(0, x), max(0, pw - hw)),
                    min(max(0, y), max(0, ph - hh)),
                )
                self._hist_user_placed = True
            except (KeyError, TypeError, ValueError):
                pass
        map_pos = data.get("map")
        if map_pos and self._map is not None:
            try:
                x = int(float(map_pos["x"]) * pw)
                y = int(float(map_pos["y"]) * ph)
                mw, mh = self._map.width(), self._map.height()
                self._map.move(
                    min(max(0, x), max(0, pw - mw)),
                    min(max(0, y), max(0, ph - mh)),
                )
                self._map_user_placed = True
            except (KeyError, TypeError, ValueError):
                pass
        adjust_pos = data.get("adjust")
        if adjust_pos and self._adj is not None:
            try:
                x = int(float(adjust_pos["x"]) * pw)
                y = int(float(adjust_pos["y"]) * ph)
                aw, ah = self._adj.width(), self._adj.height()
                self._adj.move(
                    min(max(0, x), max(0, pw - aw)),
                    min(max(0, y), max(0, ph - ah)),
                )
                self._adj_user_placed = True
            except (KeyError, TypeError, ValueError):
                pass

    def _layout_filmstrip(self):
        layer = getattr(self, "_filmstrip_layer", None)
        if layer is None:
            return
        if (
            not layer.isVisible()
            and self._filmstrip_opacity_value() <= 0.0
            and not self._filmstrip_reveal
        ):
            return
        fh = self._FILMSTRIP_HEIGHT
        w = max(1, self.width())
        host = layer.parentWidget()
        if host is not None and host is not self:
            top_left = self.mapToGlobal(QPoint(0, max(0, self.height() - fh)))
            bottom_right = self.mapToGlobal(QPoint(w, self.height()))
            origin = host.mapFromGlobal(top_left)
            corner = host.mapFromGlobal(bottom_right)
            layer.setGeometry(
                QRect(origin, corner).normalized().intersected(host.rect())
            )
        else:
            layer.setGeometry(0, self.height() - fh, w, fh)
        self._layout_filmstrip_inner()
        self._raise_filmstrip_chrome()

    def mouseMoveEvent(self, event):
        self.handle_pointer_for_filmstrip(event.globalPosition())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._cancel_show_filmstrip()
        self._schedule_hide_filmstrip()
        super().leaveEvent(event)

    def _layout_histogram(self):
        h = self._hist
        if not h.isVisible():
            return
        pw, ph = self.width(), self.height()
        hw, hh = h.width(), h.height()
        if pw < 1 or ph < 1:
            return
        if not self._hist_user_placed:
            x = max(self._HIST_MARGIN, pw - hw - self._HIST_MARGIN)
            y = self._HIST_MARGIN
            x = min(max(0, x), max(0, pw - hw))
            y = min(max(0, y), max(0, ph - hh))
            h.move(x, y)
        else:
            x = min(max(0, h.x()), max(0, pw - hw))
            y = min(max(0, h.y()), max(0, ph - hh))
            h.move(x, y)
        h.raise_()

    def mark_histogram_user_moved(self):
        self._hist_user_placed = True
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "schedule_save_session_state"):
            viewer.schedule_save_session_state()

    def relayout_histogram(self):
        self._layout_histogram()


# -----------------------------
# Loading Overlay for Single View
# -----------------------------
class LoadingOverlay(QWidget):
    """Semi-transparent loading overlay with spinner-like message"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setVisible(False)
        self._message = "Loading Image..."
        
    def set_message(self, message):
        self._message = message
        self.update()
        
    def show_loading(self, message=None):
        if message:
            self._message = message
        self.setVisible(True)
        self.raise_()
        self.update()
        
    def hide_loading(self):
        self.setVisible(False)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw semi-transparent background
        painter.setBrush(QColor(*theme.VOID_RGB, 120))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())

        # Draw loading box
        box_width = 240
        box_height = 80
        x = (self.width() - box_width) // 2
        y = (self.height() - box_height) // 2

        painter.setBrush(QColor(*theme.RAISED_RGB, 230))
        painter.setPen(QPen(QColor(*theme.INK_FAINT_RGB), 1))
        painter.drawRoundedRect(x, y, box_width, box_height, 10, 10)

        # Draw text
        painter.setPen(QColor(theme.INK))
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x, y, box_width, box_height, Qt.AlignmentFlag.AlignCenter, self._message)

    def resizeEvent(self, event):
        # Already covers parent because we resize it in parent's resizeEvent
        super().resizeEvent(event)


def _windows_shell_verb_suggests_share(verb_name: object) -> bool:
    """Match Explorer context-menu verbs for 'Share' across English and common locales."""
    s = str(verb_name or "")
    low = s.replace("&", "").strip().lower()
    if "share" in low or "windows.share" in low:
        return True
    plain = s.replace("&", "")
    for token in (
        "\u5171\u7528",  # zh-TW: 共用
        "\u5206\u4eab",  # zh-CN: 分享
        "partage",
        "teilen",
        "condividi",
        "compartir",
        "delen",
    ):
        if token in plain or token in low:
            return True
    return False


def _open_windows_open_with_dialog(path: str, owner_hwnd: int = 0) -> bool:
    """Show the native Windows 'Open with' dialog for a file."""
    import subprocess
    import ctypes
    from ctypes import Structure, byref, c_long, wintypes

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False

    hwnd = int(owner_hwnd or 0)
    OAIF_EXEC = 0x0004  # Required on Win10+ or Windows shows "go to Settings" instead.

    # Classic Open With picker (Unicode). Prefer this over SHOpenWithDialog on Win11.
    try:
        shell32 = ctypes.windll.shell32
        shell32.OpenAs_RunDLLW.argtypes = [
            wintypes.HWND,
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            ctypes.c_int,
        ]
        shell32.OpenAs_RunDLLW.restype = None
        shell32.OpenAs_RunDLLW(hwnd, None, abs_path, 1)
        return True
    except Exception:
        pass

    # SHOpenWithDialog — must pass OAIF_EXEC or user only sees the Settings redirect.
    try:
        class _OPENASINFO(Structure):
            _fields_ = [
                ("pcszFile", wintypes.LPCWSTR),
                ("pcszClass", wintypes.LPCWSTR),
                ("oaifInFlags", wintypes.DWORD),
            ]

        shell32 = ctypes.windll.shell32
        shell32.SHOpenWithDialog.argtypes = [wintypes.HWND, ctypes.POINTER(_OPENASINFO)]
        shell32.SHOpenWithDialog.restype = c_long
        info = _OPENASINFO(abs_path, None, OAIF_EXEC)
        hr = int(shell32.SHOpenWithDialog(hwnd, byref(info)))
        if hr >= 0:
            return True
    except Exception:
        pass

    # rundll32 fallback — must use OpenAs_RunDLLW, not the ANSI OpenAs_RunDLL.
    try:
        subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLLW", abs_path])
        return True
    except Exception:
        pass

    try:
        flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Sta",
                "-Command",
                "Start-Process",
                "-FilePath",
                abs_path,
                "-Verb",
                "OpenAs",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=flags,
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass

    try:
        import win32com.client  # type: ignore

        folder = win32com.client.Dispatch("Shell.Application").Namespace(
            os.path.dirname(abs_path)
        )
        item = folder.ParseName(os.path.basename(abs_path))
        if item is None:
            return False
        open_with_tokens = (
            "openas",
            "open with",
            "openwith",
            "開啟方式",
            "選擇開啟方式",
            "打开方式",
            "选择打开方式",
        )
        for verb_key in open_with_tokens:
            try:
                item.InvokeVerb(verb_key)
                return True
            except Exception:
                continue
        for verb in item.Verbs():
            try:
                name = str(getattr(verb, "Name", "") or "").replace("&", "").strip().lower()
                if any(token in name for token in open_with_tokens):
                    verb.DoIt()
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _share_windows_clipboard_cf_hdrop(path: str) -> bool:
    """Place file(s) on clipboard as CF_HDROP (native Windows file clipboard)."""
    import struct

    try:
        import win32clipboard  # type: ignore
        import win32con  # type: ignore
    except ImportError:
        return False

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False
    paths = (abs_path + "\0").encode("utf-16le") + b"\x00\x00"
    dropfiles = struct.pack("<IIIII", 20, 0, 0, 0, 1)
    data = dropfiles + paths
    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_HDROP, data)
        finally:
            win32clipboard.CloseClipboard()
        return True
    except Exception:
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass
        return False


def _share_windows_clipboard_cf_hdrop_paths(file_paths: List[str]) -> bool:
    """Place multiple files on the clipboard as CF_HDROP."""
    import struct

    try:
        import win32clipboard  # type: ignore
        import win32con  # type: ignore
    except ImportError:
        return False

    abs_paths = [
        os.path.abspath(p) for p in file_paths if p and os.path.isfile(os.path.abspath(p))
    ]
    if not abs_paths:
        return False
    if len(abs_paths) == 1:
        return _share_windows_clipboard_cf_hdrop(abs_paths[0])
    paths_blob = "".join(p + "\0" for p in abs_paths).encode("utf-16le") + b"\x00\x00"
    dropfiles = struct.pack("<IIIII", 20, 0, 0, 0, len(abs_paths))
    data = dropfiles + paths_blob
    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_HDROP, data)
        finally:
            win32clipboard.CloseClipboard()
        return True
    except Exception:
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass
        return False


def _share_windows_clipboard_file_via_powershell(path: str) -> bool:
    """Put the file object on the clipboard (Windows) so the user can paste into Mail, Teams, Explorer, etc."""
    import subprocess

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False
    flags = 0
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Sta",
                "-Command",
                "Set-Clipboard",
                "-LiteralPath",
                abs_path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=flags,
        )
        return r.returncode == 0
    except Exception:
        return False


def _resolve_windows_share_helper_exe():
    """Locate built WindowsShareHelper.exe (dev tree or bundled install)."""
    rel = os.path.join(
        "windows_share_helper",
        "bin",
        "Release",
        "net8.0-windows10.0.19041.0",
        "WindowsShareHelper.exe",
    )
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, rel)]
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "WindowsShareHelper.exe"))
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "WindowsShareHelper.exe"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _share_windows_via_helper(path: str, owner_hwnd: int) -> bool:
    """Launch the .NET helper that wraps IDataTransferManagerInterop."""
    import subprocess
    import time

    helper = _resolve_windows_share_helper_exe()
    if not helper:
        return False
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False
    flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        proc = subprocess.Popen(
            [helper, str(int(owner_hwnd or 0)), abs_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        # Fail fast if the helper exits immediately, but never block the Qt UI
        # thread for hundreds of ms — that prevented the share flyout on first click.
        app = None
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
        except Exception:
            app = None
        deadline = time.time() + 0.15
        while time.time() < deadline:
            rc = proc.poll()
            if rc is not None:
                return rc == 0
            if app is not None:
                app.processEvents()
            time.sleep(0.01)
        return True
    except Exception:
        return False


def _share_windows_via_winrt(path: str, owner_hwnd: int) -> bool:
    """In-process WinRT share sheet (DataTransferManager + interop)."""
    if not owner_hwnd:
        return False
    try:
        from windows_share import share_file_windows

        return bool(share_file_windows(path, owner_hwnd))
    except Exception:
        return False


