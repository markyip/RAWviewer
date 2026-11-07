"""CustomTitleBar: custom title bar for frameless windows."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from core import theme


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

        # Load branded favicon / appicon (repo icons/ in dev; _MEIPASS when frozen).
        # widgets.py lives at src/rawviewer_app/ — do NOT use that as the icons root.
        if getattr(sys, "frozen", False):
            bases = [getattr(sys, "_MEIPASS", ""), os.path.dirname(sys.executable)]
        else:
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
            )
            bases = [repo_root, os.getcwd()]

        icon_paths: list[str] = []
        for base in bases:
            if not base:
                continue
            icon_paths.extend(
                [
                    os.path.join(base, "icons", "favicon.ico"),
                    os.path.join(base, "icons", "favicon.png"),
                    os.path.join(base, "icons", "appicon.ico"),
                    os.path.join(base, "icons", "appicon.png"),
                ]
            )

        icon_loaded = False
        for icon_path in icon_paths:
            if not os.path.isfile(icon_path):
                continue
            try:
                icon = QIcon(icon_path)
                pixmap = icon.pixmap(24, 24)
                if pixmap.isNull() and icon_path.lower().endswith((".png", ".ico")):
                    pixmap = QPixmap(icon_path).scaled(
                        24,
                        24,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
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
        if hasattr(self.parent, 'view_mode') and self.parent.view_mode == 'gallery' and hasattr(self.parent, 'gallery_justified') and self.parent.gallery_justified:
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
