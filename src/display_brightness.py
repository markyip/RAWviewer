"""Best-effort display brightness control (macOS built-in panels).

Uses private CoreDisplay APIs (Get/SetUserBrightness). External monitors
and non-Darwin platforms are no-ops. Callers must restore any saved value
when leaving the feature that raised brightness.
"""

from __future__ import annotations

import sys
from typing import Optional

_core_display = None
_cg = None
_ready = False


def _ensure_loaded() -> bool:
    global _core_display, _cg, _ready
    if _ready:
        return _core_display is not None and _cg is not None
    _ready = True
    if sys.platform != "darwin":
        return False
    try:
        from ctypes import CDLL, c_double, c_uint32

        _core_display = CDLL(
            "/System/Library/Frameworks/CoreDisplay.framework/CoreDisplay"
        )
        _cg = CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        _core_display.CoreDisplay_Display_GetUserBrightness.argtypes = [c_uint32]
        _core_display.CoreDisplay_Display_GetUserBrightness.restype = c_double
        _core_display.CoreDisplay_Display_SetUserBrightness.argtypes = [
            c_uint32,
            c_double,
        ]
        _cg.CGMainDisplayID.restype = c_uint32
        return True
    except Exception:
        _core_display = None
        _cg = None
        return False


def main_display_id() -> Optional[int]:
    if not _ensure_loaded():
        return None
    try:
        return int(_cg.CGMainDisplayID())
    except Exception:
        return None


def get_brightness(display_id: Optional[int] = None) -> Optional[float]:
    """Return current user brightness in 0..1, or None if unavailable."""
    if not _ensure_loaded():
        return None
    did = display_id if display_id is not None else main_display_id()
    if did is None:
        return None
    try:
        from ctypes import c_uint32

        val = float(
            _core_display.CoreDisplay_Display_GetUserBrightness(c_uint32(int(did)))
        )
        if val < 0.0 or val > 1.0:
            return None
        return val
    except Exception:
        return None


def set_brightness(value: float, display_id: Optional[int] = None) -> bool:
    """Set user brightness to ``value`` (clamped 0..1). Returns True on success."""
    if not _ensure_loaded():
        return False
    did = display_id if display_id is not None else main_display_id()
    if did is None:
        return False
    try:
        from ctypes import c_double, c_uint32

        v = max(0.0, min(1.0, float(value)))
        _core_display.CoreDisplay_Display_SetUserBrightness(
            c_uint32(int(did)), c_double(v)
        )
        return True
    except Exception:
        return False
