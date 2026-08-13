"""Darkroom chrome glyphs — line icons that match theme.py, not Font Awesome.

qtawesome's filled FA/MDI shapes read as a different product sitting on the
status bar. These are 1.6px round-cap strokes in the same warm ink as the
rest of the chrome: a contact sheet, a tone curve, a film folder. Cached per
(name, color, size) so a hover recolor is one pixmap, not a redraw.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

import theme

_CACHE: Dict[Tuple[str, str, int], QIcon] = {}
_CANVAS = 20.0


def icon(name: str, *, color: str | None = None, size: int = 20) -> QIcon:
    """QIcon for a chrome glyph or a Font Awesome alias. Empty if unknown."""
    key_name = (name or "").strip()
    glyph = ALIASES.get(key_name, key_name)
    draw = _GLYPHS.get(glyph)
    if draw is None:
        return QIcon()
    tint = color or theme.INK_MUTED
    cache_key = (glyph, tint, int(size))
    hit = _CACHE.get(cache_key)
    if hit is not None:
        return hit
    px = max(12, int(size))
    dpr = 2.0
    pm = QPixmap(int(px * dpr), int(px * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.scale(px / _CANVAS, px / _CANVAS)
    pen = QPen(QColor(tint))
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    draw(p)
    p.end()
    ic = QIcon(pm)
    _CACHE[cache_key] = ic
    return ic


def _stroke_fill(p: QPainter) -> None:
    p.setBrush(QColor(p.pen().color()))


def _no_fill(p: QPainter) -> None:
    p.setBrush(Qt.BrushStyle.NoBrush)


def _folder(p: QPainter) -> None:
    tab = QPainterPath()
    tab.moveTo(3.5, 7.0)
    tab.lineTo(3.5, 6.2)
    tab.quadTo(3.5, 5.2, 4.5, 5.2)
    tab.lineTo(8.2, 5.2)
    tab.lineTo(9.6, 7.0)
    p.drawPath(tab)
    p.drawRoundedRect(QRectF(3.5, 7.0, 13.0, 9.0), 1.4, 1.4)
    p.drawLine(QPointF(5.5, 10.5), QPointF(14.5, 10.5))
    p.drawLine(QPointF(5.5, 12.7), QPointF(11.5, 12.7))


def _gallery(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3.2, 3.6, 6.2, 5.6), 0.9, 0.9)
    p.drawRoundedRect(QRectF(10.6, 3.6, 6.2, 5.6), 0.9, 0.9)
    p.drawRoundedRect(QRectF(3.2, 10.8, 6.2, 5.6), 0.9, 0.9)
    p.drawRoundedRect(QRectF(10.6, 10.8, 6.2, 5.6), 0.9, 0.9)


def _share(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4.2, 8.2, 11.6, 8.0), 1.4, 1.4)
    p.drawLine(QPointF(10.0, 3.6), QPointF(10.0, 11.0))
    p.drawLine(QPointF(10.0, 3.6), QPointF(7.2, 6.4))
    p.drawLine(QPointF(10.0, 3.6), QPointF(12.8, 6.4))


def _star_path(cx: float, cy: float, outer: float, inner: float) -> QPainterPath:
    path = QPainterPath()
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        r = outer if i % 2 == 0 else inner
        pt = QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang))
        if i == 0:
            path.moveTo(pt)
        else:
            path.lineTo(pt)
    path.closeSubpath()
    return path


def _star(p: QPainter) -> None:
    path = _star_path(10.0, 10.2, 6.4, 2.7)
    _stroke_fill(p)
    p.drawPath(path)
    _no_fill(p)


def _star_outline(p: QPainter) -> None:
    p.drawPath(_star_path(10.0, 10.2, 6.4, 2.7))


def _play(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(7.2, 4.6)
    path.lineTo(15.4, 10.0)
    path.lineTo(7.2, 15.4)
    path.closeSubpath()
    p.drawPath(path)


def _pause(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(6.0, 4.6, 2.6, 10.8), 0.7, 0.7)
    p.drawRoundedRect(QRectF(11.4, 4.6, 2.6, 10.8), 0.7, 0.7)


def _rotate(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4.6, 6.4, 8.4, 8.4), 1.1, 1.1)
    p.drawArc(QRectF(7.6, 4.0, 8.4, 8.4), 40 * 16, 230 * 16)
    p.drawLine(QPointF(15.8, 4.6), QPointF(15.8, 7.6))
    p.drawLine(QPointF(15.8, 4.6), QPointF(13.0, 4.6))


def _adjust(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3.6, 4.2, 12.8, 11.6), 1.4, 1.4)
    curve = QPainterPath()
    curve.moveTo(5.6, 13.2)
    curve.cubicTo(8.2, 13.0, 9.0, 6.8, 10.0, 8.4)
    curve.cubicTo(11.2, 10.4, 12.2, 6.6, 14.4, 6.8)
    p.drawPath(curve)


def _search(p: QPainter) -> None:
    p.drawEllipse(QRectF(4.0, 4.0, 9.2, 9.2))
    p.drawLine(QPointF(11.6, 11.6), QPointF(16.0, 16.0))


def _burst(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4.2, 4.0, 9.6, 9.6), 1.2, 1.2)
    p.drawRoundedRect(QRectF(6.2, 6.0, 9.6, 9.6), 1.2, 1.2)


def _compare(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3.2, 4.4, 6.2, 11.2), 1.1, 1.1)
    p.drawRoundedRect(QRectF(10.6, 4.4, 6.2, 11.2), 1.1, 1.1)


def _contrast(p: QPainter) -> None:
    p.drawEllipse(QRectF(4.2, 4.2, 11.6, 11.6))
    path = QPainterPath()
    path.moveTo(10.0, 4.2)
    path.arcTo(QRectF(4.2, 4.2, 11.6, 11.6), 90, 180)
    path.closeSubpath()
    _stroke_fill(p)
    p.drawPath(path)
    _no_fill(p)


def _eyedropper(p: QPainter) -> None:
    p.save()
    p.translate(10.0, 10.0)
    p.rotate(42)
    p.translate(-10.0, -10.0)
    p.drawLine(QPointF(10.0, 4.2), QPointF(10.0, 12.4))
    p.drawEllipse(QRectF(8.4, 3.2, 3.2, 3.2))
    p.drawRoundedRect(QRectF(8.6, 12.0, 2.8, 4.2), 0.8, 0.8)
    p.restore()


def _brush(p: QPainter) -> None:
    p.drawLine(QPointF(6.4, 14.8), QPointF(13.2, 5.4))
    p.drawRoundedRect(QRectF(11.6, 3.8, 3.4, 3.4), 0.8, 0.8)
    path = QPainterPath()
    path.moveTo(5.4, 15.8)
    path.quadTo(7.0, 13.6, 8.4, 14.2)
    path.quadTo(6.2, 16.6, 5.4, 15.8)
    p.drawPath(path)


def _eraser(p: QPainter) -> None:
    p.save()
    p.translate(10.0, 10.0)
    p.rotate(-28)
    p.translate(-10.0, -10.0)
    p.drawRoundedRect(QRectF(6.0, 6.2, 8.0, 7.6), 1.2, 1.2)
    p.drawLine(QPointF(10.0, 6.2), QPointF(10.0, 13.8))
    p.restore()


def _crosshair(p: QPainter) -> None:
    p.drawEllipse(QRectF(4.4, 4.4, 11.2, 11.2))
    p.drawLine(QPointF(10.0, 3.4), QPointF(10.0, 7.4))
    p.drawLine(QPointF(10.0, 12.6), QPointF(10.0, 16.6))
    p.drawLine(QPointF(3.4, 10.0), QPointF(7.4, 10.0))
    p.drawLine(QPointF(12.6, 10.0), QPointF(16.6, 10.0))


def _sparkle(p: QPainter) -> None:
    p.drawLine(QPointF(10.0, 3.6), QPointF(10.0, 16.4))
    p.drawLine(QPointF(4.2, 10.0), QPointF(15.8, 10.0))
    p.drawLine(QPointF(6.0, 6.0), QPointF(14.0, 14.0))
    p.drawLine(QPointF(14.0, 6.0), QPointF(6.0, 14.0))
    p.drawEllipse(QRectF(8.6, 8.6, 2.8, 2.8))


def _cloud(p: QPainter) -> None:
    p.drawEllipse(QRectF(4.0, 8.2, 6.0, 5.2))
    p.drawEllipse(QRectF(7.4, 6.2, 7.2, 6.4))
    p.drawEllipse(QRectF(12.0, 8.4, 4.4, 4.8))
    p.drawLine(QPointF(5.2, 13.2), QPointF(15.2, 13.2))


def _mountain(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(3.4, 15.2)
    path.lineTo(8.2, 6.4)
    path.lineTo(11.2, 11.0)
    path.lineTo(13.4, 8.2)
    path.lineTo(16.6, 15.2)
    path.closeSubpath()
    p.drawPath(path)


def _sun(p: QPainter) -> None:
    p.drawEllipse(QRectF(7.2, 7.2, 5.6, 5.6))
    for i in range(8):
        ang = i * math.pi / 4
        c, s = math.cos(ang), math.sin(ang)
        p.drawLine(
            QPointF(10.0 + 4.6 * c, 10.0 + 4.6 * s),
            QPointF(10.0 + 6.6 * c, 10.0 + 6.6 * s),
        )


def _burn(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(4.2, 15.2)
    path.quadTo(7.0, 6.4, 10.0, 4.8)
    path.quadTo(11.4, 8.6, 10.6, 10.4)
    path.quadTo(14.8, 7.2, 15.6, 15.2)
    path.closeSubpath()
    p.drawPath(path)


def _heal(p: QPainter) -> None:
    p.drawEllipse(QRectF(4.2, 4.2, 11.6, 11.6))
    p.drawLine(QPointF(10.0, 7.0), QPointF(10.0, 13.0))
    p.drawLine(QPointF(7.0, 10.0), QPointF(13.0, 10.0))


def _copy(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(5.8, 5.8, 9.4, 10.4), 1.1, 1.1)
    p.drawRoundedRect(QRectF(3.8, 3.8, 9.4, 10.4), 1.1, 1.1)


def _times(p: QPainter) -> None:
    p.drawLine(QPointF(6.0, 6.0), QPointF(14.0, 14.0))
    p.drawLine(QPointF(14.0, 6.0), QPointF(6.0, 14.0))


def _power(p: QPainter) -> None:
    p.drawArc(QRectF(4.6, 4.8, 10.8, 10.8), 50 * 16, 260 * 16)
    p.drawLine(QPointF(10.0, 3.8), QPointF(10.0, 10.0))


def _handles(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(5.2, 5.2, 9.6, 9.6), 1.0, 1.0)
    for x, y in ((4.4, 4.4), (14.0, 4.4), (4.4, 14.0), (14.0, 14.0)):
        p.drawRect(QRectF(x, y, 1.8, 1.8))


def _eye(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(3.4, 10.0)
    path.quadTo(10.0, 4.4, 16.6, 10.0)
    path.quadTo(10.0, 15.6, 3.4, 10.0)
    p.drawPath(path)
    p.drawEllipse(QRectF(8.2, 8.2, 3.6, 3.6))


def _eye_slash(p: QPainter) -> None:
    _eye(p)
    p.drawLine(QPointF(5.0, 15.2), QPointF(15.0, 4.8))


def _export(p: QPainter) -> None:
    p.drawLine(QPointF(4.2, 15.4), QPointF(15.8, 15.4))
    p.drawLine(QPointF(4.2, 15.4), QPointF(4.2, 12.6))
    p.drawLine(QPointF(15.8, 15.4), QPointF(15.8, 12.6))
    p.drawLine(QPointF(10.0, 4.0), QPointF(10.0, 12.6))
    p.drawLine(QPointF(10.0, 4.0), QPointF(7.2, 6.8))
    p.drawLine(QPointF(10.0, 4.0), QPointF(12.8, 6.8))


_GLYPHS: Dict[str, Callable[[QPainter], None]] = {
    "folder": _folder,
    "gallery": _gallery,
    "share": _share,
    "star": _star,
    "star_outline": _star_outline,
    "play": _play,
    "pause": _pause,
    "rotate": _rotate,
    "adjust": _adjust,
    "search": _search,
    "burst": _burst,
    "compare": _compare,
    "contrast": _contrast,
    "eyedropper": _eyedropper,
    "brush": _brush,
    "eraser": _eraser,
    "crosshair": _crosshair,
    "sparkle": _sparkle,
    "cloud": _cloud,
    "mountain": _mountain,
    "sun": _sun,
    "burn": _burn,
    "heal": _heal,
    "copy": _copy,
    "times": _times,
    "power": _power,
    "handles": _handles,
    "eye": _eye,
    "eye_slash": _eye_slash,
    "export": _export,
}

ALIASES = {
    "fa5s.folder-open": "folder",
    "fa5s.th": "gallery",
    "fa5s.share-alt": "share",
    "fa5s.star": "star",
    "mdi.star-outline": "star_outline",
    "fa5s.play": "play",
    "fa5s.pause": "pause",
    "fa5s.redo": "rotate",
    "fa5s.sliders-h": "adjust",
    "fa5s.search": "search",
    "fa5s.layer-group": "burst",
    "fa5s.columns": "compare",
    "fa5s.paint-brush": "brush",
    "fa5s.crosshairs": "crosshair",
    "fa5s.magic": "sparkle",
    "fa5s.cloud": "cloud",
    "fa5s.mountain": "mountain",
    "fa5s.adjust": "contrast",
    "fa5s.eye-dropper": "eyedropper",
    "fa5s.eraser": "eraser",
    "fa5s.copy": "copy",
    "fa5s.times": "times",
    "fa5s.sun": "sun",
    "fa5s.fire": "burn",
    "fa5s.band-aid": "heal",
    "fa5.eye": "eye",
    "fa5.eye-slash": "eye_slash",
    "fa5s.vector-square": "handles",
    "fa5s.file-export": "export",
    "fa5s.power-off": "power",
}


def known(name: str) -> bool:
    key = (name or "").strip()
    return ALIASES.get(key, key) in _GLYPHS
