"""Main-window package: bootstrap, helpers and the ``RAWImageViewer`` composite.

Import order note: ``main.py`` sets the Qt/OpenMP environment flags before
importing anything from this package.
"""

from __future__ import annotations

__all__ = [
    "bootstrap",
    "lazy",
    "logging_setup",
    "macos",
    "paths",
    "platform_win",
    "qt_helpers",
    "raw_image_viewer",
    "splash",
    "state",
    "windows_shell",
]
