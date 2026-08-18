"""RAWviewer-styled dialog for optional GitHub release updates."""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import theme
from release_update import APP_VERSION, RELEASE_PAGE_URL


class ReleaseUpdateDialog(QDialog):
    """Material Design 3 prompt matching CustomConfirmDialog / theme.py chrome."""

    _DIALOG_WIDTH = 480

    def __init__(
        self,
        parent=None,
        *,
        current: str | None = None,
        latest: str | None = None,
        release_url: str | None = None,
        release_notes: str | None = None,
    ):
        super().__init__(parent)
        self.release_url = (release_url or RELEASE_PAGE_URL).strip() or RELEASE_PAGE_URL
        current_v = (current or APP_VERSION).strip() or APP_VERSION
        latest_v = (latest or "").strip()

        flags = Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        if sys.platform == "darwin":
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)

        self.container = QWidget(self)
        self.container.setObjectName("releaseUpdateDialogContainer")
        self.container.setStyleSheet(f"""
            #releaseUpdateDialogContainer {{
                background-color: {theme.VOID};
                border: 1px solid {theme.LINE};
                border-radius: 12px;
            }}
        """)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(24, 22, 24, 22)
        main_layout.setSpacing(12)

        title_label = QLabel("Update Available")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.INK};
                font-size: 18px;
                font-weight: 600;
            }}
        """)
        main_layout.addWidget(title_label)

        message_label = QLabel(f"A new version of RAWviewer is available ({latest_v}).")
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.INK_MUTED};
                font-size: 14px;
                line-height: 1.45;
            }}
        """)
        main_layout.addWidget(message_label)

        notes = (release_notes or "").strip()
        if notes:
            whats_new_label = QLabel("What's new")
            whats_new_label.setObjectName("releaseUpdateWhatsNewHeading")
            whats_new_label.setStyleSheet(f"""
                QLabel {{
                    color: {theme.EMBER};
                    font-size: 13px;
                    font-weight: 600;
                }}
            """)
            main_layout.addWidget(whats_new_label)

            # Read-only, scroll-capped so a long release body can't grow the
            # dialog unbounded; the notes are already trimmed upstream.
            notes_label = QLabel(notes)
            notes_label.setObjectName("releaseUpdateWhatsNew")
            notes_label.setWordWrap(True)
            notes_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            notes_label.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
            )
            notes_label.setStyleSheet(f"""
                QLabel {{
                    color: {theme.INK_MUTED};
                    font-size: 13px;
                    line-height: 1.5;
                }}
            """)
            notes_scroll = QScrollArea()
            notes_scroll.setWidget(notes_label)
            notes_scroll.setWidgetResizable(True)
            notes_scroll.setFrameShape(QFrame.Shape.NoFrame)
            notes_scroll.setMaximumHeight(150)
            notes_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            notes_scroll.setStyleSheet("""
                QScrollArea { background: transparent; border: none; }
                QScrollArea > QWidget > QWidget { background: transparent; }
            """)
            main_layout.addWidget(notes_scroll)

        notes_link = QLabel(
            f'<a href="{self.release_url}" '
            f'style="color: {theme.EMBER}; text-decoration: none;">Release notes</a>'
        )
        notes_link.setObjectName("releaseUpdateNotesLink")
        notes_link.setTextFormat(Qt.TextFormat.RichText)
        notes_link.setOpenExternalLinks(True)
        notes_link.setStyleSheet(f"""
            QLabel {{
                color: {theme.EMBER};
                font-size: 13px;
            }}
        """)
        main_layout.addWidget(notes_link)

        note_label = QLabel(
            f"You are running {current_v}.\n"
            f"The latest release is {latest_v}.\n\n"
            "Open the GitHub releases page to download the update?"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.INK_FAINT};
                font-size: 13px;
            }}
        """)
        main_layout.addWidget(note_label)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        button_layout.setSpacing(12)
        button_layout.addStretch()

        not_now_btn = QPushButton("Not Now")
        not_now_btn.setFixedHeight(36)
        not_now_btn.setMinimumWidth(110)
        not_now_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        not_now_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {theme.INK_MUTED};
                border: 1px solid {theme.LINE};
                border-radius: 18px;
                font-size: 14px;
                font-weight: 500;
                padding: 0px 20px;
            }}
            QPushButton:hover {{
                color: {theme.INK};
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.05);
                border-color: {theme.EMBER};
            }}
            QPushButton:pressed {{
                background-color: rgba({theme.INK_RGB[0]}, {theme.INK_RGB[1]}, {theme.INK_RGB[2]}, 0.1);
            }}
        """)
        not_now_btn.clicked.connect(self.reject)
        button_layout.addWidget(not_now_btn)

        open_btn = QPushButton("Open Download Page")
        open_btn.setFixedHeight(36)
        open_btn.setMinimumWidth(160)
        open_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        open_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme.RAISED};
                color: {theme.INK};
                border: 1px solid {theme.LINE};
                border-radius: 18px;
                font-size: 14px;
                font-weight: 600;
                padding: 0px 20px;
            }}
            QPushButton:hover {{
                background-color: {theme.RAISED_HI};
                border-color: {theme.EMBER};
            }}
            QPushButton:pressed {{
                background-color: {theme.SURFACE};
            }}
        """)
        open_btn.clicked.connect(self.accept)
        open_btn.setDefault(True)
        open_btn.setFocus()
        button_layout.addWidget(open_btn)
        main_layout.addLayout(button_layout)

        container_layout = QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.container)

        self.container.adjustSize()
        dialog_h = max(230, self.container.sizeHint().height())
        self.container.setFixedSize(self._DIALOG_WIDTH, dialog_h)
        self.setFixedSize(self._DIALOG_WIDTH, dialog_h)

        if parent:
            parent_geometry = parent.geometry()
            dialog_x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            dialog_y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(dialog_x, dialog_y)

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


def show_release_update_dialog(
    parent=None,
    *,
    current: str | None = None,
    latest: str | None = None,
    release_url: str | None = None,
    release_notes: str | None = None,
) -> bool:
    """Return True when the user chose Not Now; False after Open Download Page."""
    dialog = ReleaseUpdateDialog(
        parent,
        current=current,
        latest=latest,
        release_url=release_url,
        release_notes=release_notes,
    )
    return dialog.exec() != QDialog.DialogCode.Accepted
