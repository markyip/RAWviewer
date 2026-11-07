"""Mutable module-level state shared by the main-window modules."""

from __future__ import annotations

from typing import Any

# Set by the ultra-fast splash block in ``main`` and cleared by splash teardown.
_startup_splash: Any = None

# Filled in ``main`` once the custom QApplication subclass exists.
RAWApplication: Any = None

__all__ = ["_startup_splash", "RAWApplication"]
