"""Full-window keyboard cheat sheet.

A tooltip can only be read one line at a time and vanishes when you move to
try what it said. Someone learning the app needs to see the whole vocabulary
at once, grouped by what they are trying to DO -- cull, look, develop -- not
as one alphabetical list of keys.

So: grouped columns, keys drawn as keycaps, dismissed by Esc, click, or the
same key that opened it. Content is context-aware, because the same key means
different things with the editor open, and showing a photographer the
Dodge/Burn keys while they are culling is noise.

EMBER marks only the section for the mode you are actually in -- the theme
reserves it for "what is currently active", and a cheat sheet full of accent
colour would say nothing.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import theme

# (group title, [(keys, description), ...])
# Keys are split on "/" into separate caps; " + " keeps them joined.
_BROWSE = [
    (
        "Move around",
        [
            ("← / →", "Previous / next photo"),
            ("Space", "Fit to window ↔ 100%"),
            ("Pinch / Ctrl + Scroll", "Zoom in or out"),
            ("Esc", "Gallery: clear selection · Single: back to gallery"),
        ],
    ),
    (
        "Rate and cull",
        [
            ("0 – 5", "Star rating (0 clears it)"),
            ("↓", "Move to Discard (single view, not while editing)"),
            ("Shift + ↓", "Gallery: move the selection to Discard"),
            ("Delete", "Delete photo(s)"),
            ("C", "Compare selected photos side by side"),
        ],
    ),
    (
        "Look closer",
        [
            ("H", "Histogram"),
            ("J", "Clipping: blown highlights, crushed shadows"),
            ("G", "Composition guide (cycles)"),
            ("F", "Focus point"),
            ("M", "Location map"),
            ("P", "RAW recovery preview (fit only, this session)"),
        ],
    ),
    (
        "In the gallery",
        [
            ("Ctrl / ⌘ + click", "Add or remove one photo from the selection"),
            ("Shift + click", "Select everything between two photos"),
            ("↑ / ↓", "Scroll"),
            ("E", "Open the Adjust panel"),
        ],
    ),
]

_EDITOR = [
    (
        "The panel",
        [
            ("E / Esc", "Close the Adjust panel"),
            ("← / →", "Nudge the focused slider"),
            ("Space", "Fit to window ↔ 100%"),
        ],
    ),
    (
        "Retouch",
        [
            ("D / B", "Dodge / Burn brush"),
            ("X", "Eraser"),
            ("H", "Heal"),
            ("P", "Paint a mask (hold)"),
            ("M", "Show what is masked"),
            ("Two-finger scroll", "Brush size, while a brush is armed"),
        ],
    ),
    (
        "Masks",
        [
            ("Click the eye", "Hide a mask's overlay (the edit still applies)"),
            ("Click the value", "Reset that slider to 0"),
            ("M", "Show or hide the mask overlay"),
            ("Delete", "Delete the selected mask"),
            ("Drag a mask onto another", "Combine them into one mask"),
            ("Drag a part out", "Separate it again"),
            ("Ctrl / ⌘ + Z", "Undo"),
        ],
    ),
    (
        "Check your work",
        [
            ("J", "Clipping warnings"),
            ("G", "Composition guide"),
            ("F", "Focus point"),
        ],
    ),
]

_COMPARE = [
    (
        "Compare",
        [
            ("← / →", "Previous / next candidate"),
            ("↑", "Promote the candidate to selected"),
            ("↓", "Reject the candidate"),
            ("Shift + ↓", "Reject the selected one instead"),
            ("Space", "Synchronised zoom"),
            ("C / Esc", "Leave Compare"),
        ],
    ),
    (
        "While comparing",
        [
            ("F", "Focus overlays on both panes"),
            ("J", "Clipping on both panes"),
            ("G", "Composition guide on both panes"),
            ("Delete", "Delete the candidate (Shift: the selected one)"),
        ],
    ),
]


def _keycap(text: str) -> QLabel:
    """One key, drawn as a key."""
    cap = QLabel(text)
    cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
    cap.setFont(QFont(cap.font().family(), 11, QFont.Weight.DemiBold))
    cap.setStyleSheet(
        f"""
        QLabel {{
            background-color: {theme.RAISED_HI};
            border: 1px solid {theme.LINE};
            border-bottom: 2px solid {theme.LINE};
            border-radius: 4px;
            color: {theme.INK};
            padding: 2px 7px;
        }}
        """
    )
    return cap


def _key_row(keys: str) -> QWidget:
    """A key expression: caps for keys, plain text for the joiners."""
    wrap = QWidget()
    row = QHBoxLayout(wrap)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addStretch(1)  # right-aligned against the description
    # " / " and " + " are joiners between caps; anything else is one cap.
    parts: list[str] = []
    token = ""
    for chunk in keys.replace(" + ", "\x00+\x00").replace(" / ", "\x00/\x00").split("\x00"):
        parts.append(chunk)
    for part in parts:
        if part in ("+", "/"):
            joiner = QLabel(part)
            joiner.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 11px;")
            row.addWidget(joiner)
        elif part.strip():
            row.addWidget(_keycap(part.strip()))
    return wrap


class ShortcutCheatSheet(QWidget):
    """Frameless overlay listing every shortcut for the current context."""

    def __init__(self, parent=None, *, mode: str = "browse"):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setModal = None  # not modal: it is a reference, not a question

        groups = {"editor": _EDITOR, "compare": _COMPARE}.get(mode, _BROWSE)
        title = {
            "editor": "Editing",
            "compare": "Comparing",
        }.get(mode, "Browsing and culling")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("cheatsheet_card")
        card.setStyleSheet(
            f"""
            QFrame#cheatsheet_card {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 10px;
            }}
            """
        )
        outer.addWidget(card)

        card_col = QVBoxLayout(card)
        card_col.setContentsMargins(26, 20, 26, 18)
        card_col.setSpacing(14)

        head = QHBoxLayout()
        heading = QLabel("Keyboard shortcuts")
        heading.setStyleSheet(
            f"color: {theme.INK}; font-size: 15px; font-weight: 700; "
            "letter-spacing: 0.3px;"
        )
        head.addWidget(heading)
        # EMBER only here: the mode you are actually in.
        context = QLabel(title)
        context.setStyleSheet(
            f"color: {theme.EMBER}; font-size: 11px; font-weight: 700; "
            "letter-spacing: 1.2px;"
        )
        head.addSpacing(10)
        head.addWidget(context)
        head.addStretch(1)
        close = QPushButton("✕")
        close.setFixedSize(24, 24)
        close.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent; border: none; border-radius: 4px;
                color: {theme.INK_MUTED}; font-size: 13px;
            }}
            QPushButton:hover {{ background: {theme.RAISED_HI}; color: {theme.INK}; }}
            """
        )
        close.clicked.connect(self.close)
        head.addWidget(close)
        card_col.addLayout(head)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"background-color: {theme.LINE}; border: none;")
        card_col.addWidget(rule)

        # Columns, filled top to bottom so related things stay together.
        columns = QHBoxLayout()
        columns.setSpacing(34)
        per_column = max(1, (len(groups) + 1) // 2)
        for start in range(0, len(groups), per_column):
            col = QVBoxLayout()
            col.setSpacing(12)
            for group_title, entries in groups[start : start + per_column]:
                label = QLabel(group_title.upper())
                label.setStyleSheet(
                    f"color: {theme.INK_MUTED}; font-size: 9px; font-weight: 700; "
                    "letter-spacing: 1.4px;"
                )
                col.addWidget(label)

                grid = QGridLayout()
                grid.setHorizontalSpacing(12)
                grid.setVerticalSpacing(5)
                grid.setColumnStretch(1, 1)
                for row_i, (keys, description) in enumerate(entries):
                    grid.addWidget(_key_row(keys), row_i, 0, Qt.AlignmentFlag.AlignRight)
                    desc = QLabel(description)
                    desc.setStyleSheet(f"color: {theme.INK}; font-size: 11px;")
                    grid.addWidget(desc, row_i, 1)
                col.addLayout(grid)
            col.addStretch(1)
            columns.addLayout(col)
        card_col.addLayout(columns)

        footer = QLabel("Esc to close")
        footer.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 10px;")
        card_col.addWidget(footer, alignment=Qt.AlignmentFlag.AlignRight)

        for seq in ("Esc", "?"):
            QShortcut(QKeySequence(seq), self, activated=self.close)

    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        # Click anywhere to dismiss: this is a reference card, not a form.
        self.close()

    def show_centred_on(self, host) -> None:
        """Show centred over ``host`` (the main window), clamped to it."""
        self.adjustSize()
        try:
            geo = host.geometry()
            size = self.sizeHint()
            self.move(
                geo.x() + max(0, (geo.width() - size.width()) // 2),
                geo.y() + max(0, (geo.height() - size.height()) // 3),
            )
        except Exception:
            pass
        self.show()
        self.raise_()
        self.activateWindow()
