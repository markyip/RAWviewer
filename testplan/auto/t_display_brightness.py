#!/usr/bin/env python3
"""Display brightness helpers: clamp/scale math and Adjust raise/restore wiring.

Does not change the live panel in CI. Live get() may return None on a desktop
without DDC/WMI.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    extra = f"  {detail}" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{extra}")
    if not ok:
        FAILURES.append(name)


def test_range_math():
    from display_brightness import (
        DISPLAYPOLICY_BOTH,
        IOCTL_VIDEO_QUERY_DISPLAY_BRIGHTNESS,
        IOCTL_VIDEO_SET_DISPLAY_BRIGHTNESS,
        _clamp01,
        _from_min_max,
        _to_min_max,
    )

    check("clamp below", _clamp01(-1) == 0.0)
    check("clamp above", _clamp01(2) == 1.0)
    check("clamp mid", _clamp01(0.4) == 0.4)
    check("from min max 0-100", _from_min_max(40, 0, 100) == 0.4)
    check("from inverted range is none", _from_min_max(1, 10, 10) is None)
    check("to min max 0-100", _to_min_max(1.0, 0, 100) == 100)
    check("to min max half", _to_min_max(0.5, 0, 100) == 50)
    check("ioctl query", IOCTL_VIDEO_QUERY_DISPLAY_BRIGHTNESS == 0x230498)
    check("ioctl set", IOCTL_VIDEO_SET_DISPLAY_BRIGHTNESS == 0x23049C)
    check("display policy both", DISPLAYPOLICY_BOTH == 3)


def test_get_never_raises():
    from display_brightness import get_brightness

    try:
        val = get_brightness()
    except Exception as exc:
        check("get_brightness never raises", False, str(exc))
        return
    check(
        "get_brightness type",
        val is None or (isinstance(val, float) and 0.0 <= val <= 1.0),
        repr(val),
    )


def test_fake_backend_roundtrip():
    import display_brightness as db

    class _Fake:
        def __init__(self):
            self.v = 0.4

        def get(self):
            return self.v

        def set(self, value):
            self.v = float(value)
            return True

        def close(self):
            return

    prev = db._active_backend
    db._set_active_backend(_Fake())
    try:
        check("fake set to max", db.set_brightness(1.0) is True)
        check("fake stored max", abs(db._active_backend.v - 1.0) < 1e-9)
        check("fake restore", db.set_brightness(0.4) is True)
        check("fake stored restore", abs(db._active_backend.v - 0.4) < 1e-9)
    finally:
        db._set_active_backend(prev)


def test_adjust_wiring_in_main():
    src = os.path.join(os.path.dirname(__file__), "..", "..", "src", "main.py")
    text = open(src, encoding="utf-8").read()
    check(
        "enter Adjust raises brightness",
        "self._maybe_maximize_brightness_for_adjust(True)" in text,
    )
    check(
        "leave Adjust restores brightness",
        "self._maybe_maximize_brightness_for_adjust(False)" in text,
    )
    check(
        "close restores brightness",
        "self._restore_adjust_brightness()" in text,
    )
    check(
        "Windows hwnd passed to get",
        "get_brightness(hwnd=hwnd)" in text,
    )
    check(
        "default remains on",
        'value("adjust_maximize_brightness", True, type=bool)' in text,
    )


def main():
    test_range_math()
    test_get_never_raises()
    test_fake_backend_roundtrip()
    test_adjust_wiring_in_main()
    if FAILURES:
        print(f"\n{len(FAILURES)} failed")
        return 1
    print("\nAll display-brightness checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
