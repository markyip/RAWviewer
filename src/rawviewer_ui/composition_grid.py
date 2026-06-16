"""Composition guide overlays (rule of thirds, golden ratio) for single-image view."""

from __future__ import annotations

from typing import Tuple

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem

GRID_MODES: Tuple[str, ...] = (
    "off",
    "thirds",
    "diagonal",
    "thirds_diagonal",
    "golden",
)
_GRID_LINE = QColor(200, 200, 200, 175)
_GRID_LINE_WIDTH_COSMETIC = 2  # screen pixels (GPU overlay)
_GRID_LINE_WIDTH_IMAGE = 2  # image pixels (legacy QLabel path)
_INV_PHI = 0.618033988749895
_PHI_COMPLEMENT = 1.0 - _INV_PHI


def next_grid_mode(mode: str) -> str:
    try:
        idx = GRID_MODES.index(mode)
    except ValueError:
        idx = 0
    return GRID_MODES[(idx + 1) % len(GRID_MODES)]


def grid_mode_label(mode: str) -> str:
    return {
        "off": "Off",
        "thirds": "3×3 grid",
        "diagonal": "Diagonal",
        "thirds_diagonal": "3×3 + diagonal",
        "golden": "Golden ratio",
    }.get(mode, "Off")


def _grid_pen(*, cosmetic: bool) -> QPen:
    pen = QPen(_GRID_LINE)
    pen.setWidth(_GRID_LINE_WIDTH_COSMETIC if cosmetic else _GRID_LINE_WIDTH_IMAGE)
    pen.setStyle(Qt.PenStyle.SolidLine)
    if cosmetic:
        pen.setCosmetic(True)
    return pen


def draw_composition_grid(
    painter: QPainter,
    width: int,
    height: int,
    mode: str,
    *,
    cosmetic: bool = False,
) -> None:
    """Draw guide lines in image pixel coordinates (0,0)–(width,height)."""
    if mode == "off" or width <= 0 or height <= 0:
        return
    w, h = float(width), float(height)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(_grid_pen(cosmetic=cosmetic))
    if mode in ("thirds", "thirds_diagonal"):
        for i in (1, 2):
            x = w * i / 3.0
            y = h * i / 3.0
            painter.drawLine(int(round(x)), 0, int(round(x)), height)
            painter.drawLine(0, int(round(y)), width, int(round(y)))
    if mode in ("diagonal", "thirds_diagonal"):
        painter.drawLine(0, 0, width, height)
        painter.drawLine(width, 0, 0, height)
    if mode == "golden":
        for t in (_PHI_COMPLEMENT, _INV_PHI):
            x = w * t
            y = h * t
            painter.drawLine(int(round(x)), 0, int(round(x)), height)
            painter.drawLine(0, int(round(y)), width, int(round(y)))
    painter.restore()


class CompositionGridGraphicsItem(QGraphicsItem):
    """Non-interactive grid overlay in GPU image scene coordinates (image pixels)."""

    def __init__(self) -> None:
        super().__init__()
        self._mode = "off"
        self._w = 0
        self._h = 0
        self.setZValue(8)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, max(0.0, self._w), max(0.0, self._h))

    def paint(self, painter: QPainter, option, widget=None) -> None:
        draw_composition_grid(
            painter, int(self._w), int(self._h), self._mode, cosmetic=True
        )

    def set_grid(self, width: int, height: int, mode: str) -> None:
        if mode not in GRID_MODES:
            mode = "off"
        self.prepareGeometryChange()
        self._w = max(0, int(width))
        self._h = max(0, int(height))
        self._mode = mode
        visible = mode != "off" and self._w > 0 and self._h > 0
        self.setVisible(visible)
        self.update()
