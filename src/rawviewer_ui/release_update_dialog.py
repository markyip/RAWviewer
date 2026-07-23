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
    QVBoxLayout,
    QWidget,
)

from release_update import APP_VERSION, RELEASE_PAGE_URL


class ReleaseUpdateDialog(QDialog):
    """Material Design 3 prompt matching MobileCLIPDownloadDialog styling."""

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
        self.container.setStyleSheet("""
            #releaseUpdateDialogContainer {
                background-color: #14120F;
                border: 1px solid #3A332A;
                border-radius: 12px;
            }
        """)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(24, 22, 24, 22)
        main_layout.setSpacing(12)

        title_label = QLabel("Update Available")
        title_label.setStyleSheet("""
            QLabel {
                color: #EDE7DD;
                font-size: 17px;
                font-weight: 600;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }
        """)
        main_layout.addWidget(title_label)

        message_label = QLabel(f"A new version of RAWviewer is available ({latest_v}).")
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            QLabel {
                color: #96897A;
                font-size: 13px;
                line-height: 1.45;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }
        """)
        main_layout.addWidget(message_label)

        notes = (release_notes or "").strip()
        if notes:
            whats_new_label = QLabel("What's new")
            whats_new_label.setStyleSheet("""
                QLabel {
                    color: #D9691E;
                    font-size: 12px;
                    font-weight: 600;
                    font-family: 'Roboto', 'Segoe UI', sans-serif;
                }
            """)
            main_layout.addWidget(whats_new_label)

            # Read-only, scroll-capped so a long release body can't grow the
            # dialog unbounded; the notes are already trimmed upstream.
            from PyQt6.QtWidgets import QScrollArea

            notes_label = QLabel(notes)
            notes_label.setWordWrap(True)
            notes_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            notes_label.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
            )
            notes_label.setStyleSheet("""
                QLabel {
                    color: #B7ADA0;
                    font-size: 12px;
                    line-height: 1.5;
                    font-family: 'Roboto', 'Segoe UI', sans-serif;
                }
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

        note_label = QLabel(
            f"You are running {current_v}.\n"
            f"The latest release is {latest_v}.\n\n"
            "Open the GitHub releases page to download the update?"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("""
            QLabel {
                color: #665D50;
                font-size: 12px;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }
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
        not_now_btn.setStyleSheet("""
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
        """)
        not_now_btn.clicked.connect(self.reject)
        button_layout.addWidget(not_now_btn)

        open_btn = QPushButton("Open Download Page")
        open_btn.setFixedHeight(36)
        open_btn.setMinimumWidth(160)
        open_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        open_btn.setStyleSheet("""
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
) -> bool:
    """Return True when the user chose Not Now; False after Open Download Page."""
    dialog = ReleaseUpdateDialog(
        parent,
        current=current,
        latest=latest,
        release_url=release_url,
    )
    return dialog.exec() != QDialog.DialogCode.Accepted
