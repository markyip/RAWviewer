"""Small PyQt helper factories shared by the main window."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, cast

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QScrollArea, QScrollBar

from core import theme

from . import lazy
from .lazy import qta

lazy.register(sys.modules[__name__])

__all__ = [
    "_qta_icon",
    "_qt_scroll_bar",
    "_qt_app",
    "_get_bg_thread_pool",
]


def _qta_icon(name: str, *, color: str = theme.INK_MUTED) -> QIcon:
    """qtawesome icon with a stable PyQt6 QIcon type for the type checker."""
    if qta is None:
        return QIcon()
    return cast(QIcon, qta.icon(name, color=color))


def _qt_scroll_bar(scroll_area: QScrollArea, *, vertical: bool) -> QScrollBar:
    bar = scroll_area.verticalScrollBar() if vertical else scroll_area.horizontalScrollBar()
    if bar is None:
        raise RuntimeError("scroll bar unavailable")
    return bar


def _qt_app() -> Any:
    return QApplication.instance()


_shared_bg_pool = None


def _get_bg_thread_pool():
    global _shared_bg_pool
    n = max(2, (os.cpu_count() or 4) // 2)
    cap = os.environ.get("RAWVIEWER_BG_POOL_THREADS", "").strip()
    if cap:
        with contextlib.suppress(ValueError):
            n = max(1, min(int(cap), 16))
    if _shared_bg_pool is None:
        from PyQt6.QtCore import QThreadPool
        _shared_bg_pool = QThreadPool()
    _shared_bg_pool.setMaxThreadCount(n)
    return _shared_bg_pool
