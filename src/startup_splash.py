"""Startup splash card: icon, version, user-facing load status.

Kept out of main.py so tests can import it without constructing the app.
"""
from __future__ import annotations

import os
import sys

import theme

# Compact card. Full-res appicon.png is 2k+ and looks huge as QSplashScreen.
SPLASH_ICON_MAX_PX = 168
SPLASH_CARD_W = 340
SPLASH_CARD_H = 420
SPLASH_FOOTER_H = 56


def _resource_path(relative_path: str) -> str:
    try:
        base_path = getattr(sys, "_MEIPASS", None)
        if base_path is None:
            raise AttributeError
    except Exception:
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
    return os.path.join(base_path, relative_path)


def splash_icon_path() -> str | None:
    png = _resource_path(os.path.join("icons", "appicon.png"))
    if os.path.exists(png):
        return png
    for rel in (
        os.path.join("icons", "appicon.icns"),
        os.path.join("icons", "appicon.ico"),
    ):
        path = _resource_path(rel)
        if os.path.exists(path):
            return path
    return None


def app_version_for_splash() -> str:
    try:
        from app_version import APP_VERSION

        return str(APP_VERSION).strip()
    except Exception:
        return ""


def splash_subtitle() -> str:
    """Version · edition, e.g. ``3.1.0 · Plus``."""
    parts: list[str] = []
    ver = app_version_for_splash()
    if ver:
        parts.append(ver)
    try:
        from rawviewer_profile import edition_display_name

        edition = str(edition_display_name() or "").strip()
        if edition:
            parts.append(edition)
    except Exception:
        pass
    return " · ".join(parts)


def splash_show_status(splash, msg: str) -> None:
    """Paint a user-facing status line on the splash footer (no-op if hidden)."""
    if splash is None:
        return
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QApplication

        splash.showMessage(
            str(msg),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            QColor(*theme.INK_RGB),
        )
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass


def make_splash_pixmap():
    """Darkroom card: icon, app name, version; footer reserved for status."""
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap

    max_edge = SPLASH_ICON_MAX_PX
    icon_path = splash_icon_path()
    icon = QPixmap()
    if icon_path:
        if icon_path.lower().endswith(".png"):
            icon = QPixmap(icon_path)
        else:
            icon = QIcon(icon_path).pixmap(max_edge, max_edge)
            if icon.isNull():
                icon = QPixmap(icon_path)
    if not icon.isNull() and (icon.width() > max_edge or icon.height() > max_edge):
        icon = icon.scaled(
            max_edge,
            max_edge,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    card = QPixmap(SPLASH_CARD_W, SPLASH_CARD_H)
    card.fill(QColor(*theme.VOID_RGB))
    painter = QPainter(card)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    body_bottom = SPLASH_CARD_H - SPLASH_FOOTER_H
    if not icon.isNull():
        x = (SPLASH_CARD_W - icon.width()) // 2
        y = max(28, (body_bottom - 88 - icon.height()) // 2)
        painter.drawPixmap(x, y, icon)
        title_top = y + icon.height() + 16
    else:
        painter.setPen(QPen(QColor(*theme.INK_RGB), 3))
        font = painter.font()
        font.setPointSize(28)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QRect(0, 80, SPLASH_CARD_W, 64), Qt.AlignmentFlag.AlignCenter, "RAW"
        )
        title_top = 160

    title_font = QFont()
    title_font.setPointSize(16)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.setPen(QColor(*theme.INK_RGB))
    painter.drawText(
        QRect(16, title_top, SPLASH_CARD_W - 32, 28),
        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
        "RAWviewer",
    )
    subtitle = splash_subtitle()
    if subtitle:
        sub_font = QFont()
        sub_font.setPointSize(11)
        painter.setFont(sub_font)
        painter.setPen(QColor(*theme.INK_MUTED_RGB))
        painter.drawText(
            QRect(16, title_top + 28, SPLASH_CARD_W - 32, 22),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            subtitle,
        )

    painter.setPen(QColor(*theme.LINE_RGB))
    painter.drawLine(24, body_bottom, SPLASH_CARD_W - 24, body_bottom)
    painter.end()
    return card
