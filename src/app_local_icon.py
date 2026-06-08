"""Resolve QIcons from locally installed apps (.exe, .app) and macOS share services."""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

from PyQt6.QtCore import QFileInfo, QSize, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QFileIconProvider

_ICON_CACHE: Dict[str, QIcon] = {}
_PROVIDER: Optional[QFileIconProvider] = None
_DEFAULT_MENU_ICON_SIZE = 18


def _provider() -> QFileIconProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = QFileIconProvider()
    return _PROVIDER


def _scaled_icon(icon: QIcon, size: int) -> QIcon:
    if icon is None or icon.isNull():
        return QIcon()
    px = icon.pixmap(QSize(size, size))
    if px.isNull():
        return QIcon()
    return QIcon(px)


def _cache_key(kind: str, path: str, size: int) -> str:
    return f"{kind}:{size}:{os.path.normcase(os.path.normpath(path or ''))}"


def icon_for_local_app(path: str, *, size: int = _DEFAULT_MENU_ICON_SIZE) -> QIcon:
    """Return the OS file icon for an executable or .app bundle path."""
    if not path:
        return QIcon()
    abs_path = os.path.abspath(path)
    if sys.platform == "darwin" and abs_path.endswith(".app") and os.path.isdir(abs_path):
        lookup = abs_path
    elif os.path.isfile(abs_path):
        lookup = abs_path
    elif os.path.isdir(abs_path):
        lookup = abs_path
    else:
        return QIcon()

    key = _cache_key("local", lookup, size)
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached

    icon = _provider().icon(QFileInfo(lookup))
    scaled = _scaled_icon(icon, size)
    _ICON_CACHE[key] = scaled
    return scaled


def _nsimage_to_qicon(ns_image, *, size: int) -> QIcon:
    if ns_image is None:
        return QIcon()
    try:
        data = ns_image.TIFFRepresentation()
        if data is not None:
            qimg = QImage.fromData(bytes(data))
            if not qimg.isNull():
                return QIcon(
                    QPixmap.fromImage(qimg).scaled(
                        size,
                        size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
    except Exception:
        pass
    try:
        png = ns_image.representationUsingType_properties_(4, None)
        if png is not None:
            qimg = QImage.fromData(bytes(png))
            if not qimg.isNull():
                return QIcon(
                    QPixmap.fromImage(qimg).scaled(
                        size,
                        size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
    except Exception:
        pass
    return QIcon()


def icon_for_macos_sharing_service(service, *, size: int = _DEFAULT_MENU_ICON_SIZE) -> QIcon:
    """Return an icon for an NSSharingService instance (WhatsApp, Telegram, etc.)."""
    if service is None:
        return QIcon()
    try:
        title = str(service.title() or "")
    except Exception:
        title = ""
    try:
        ident = str(service.valueForKey_("NSSharingServiceName") or "")
    except Exception:
        ident = ""
    key = _cache_key("macsvc", f"{title}|{ident}", size)
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached

    icon = QIcon()
    try:
        ns_image = service.image()
        icon = _nsimage_to_qicon(ns_image, size=size)
    except Exception:
        icon = QIcon()

    if icon.isNull() and ident:
        try:
            from AppKit import NSSharingService

            ns_image = NSSharingService.iconForSharingService_(ident)
            icon = _nsimage_to_qicon(ns_image, size=size)
        except Exception:
            pass

    _ICON_CACHE[key] = icon
    return icon
