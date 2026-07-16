"""Shared gallery scroll step / gain helpers.

Keys, mouse-wheel notches, and trackpad pixel deltas used to apply independent
multipliers (the key path was ``singleStep * 4`` while wheel/trackpad felt
slower). All three now read from the same env-tunable base so future drift is
harder.
"""

from __future__ import annotations

import os
import time


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(minimum, int(float(raw)))
    except ValueError:
        return default


def key_scroll_step_multiplier() -> int:
    """Arrow-key scroll: ``singleStep * N`` (default 4)."""
    return _env_int("RAWVIEWER_GALLERY_KEY_SCROLL_STEPS", 4, minimum=1)


def key_scroll_delta_px(single_step: int) -> int:
    return max(int(single_step), 1) * key_scroll_step_multiplier()


def wheel_base_gain() -> float:
    return _env_float("RAWVIEWER_WHEEL_GAIN", 1.0, minimum=0.25)


def wheel_fast_gain() -> float:
    return _env_float(
        "RAWVIEWER_WHEEL_FAST_GAIN",
        max(wheel_base_gain(), 3.0),
        minimum=1.0,
    )


def trackpad_gain() -> float:
    return _env_float("RAWVIEWER_TRACKPAD_GAIN", 1.6, minimum=0.25)


def wheel_notch_gain(now: float | None = None, last_notch_t: float = 0.0) -> tuple[float, float]:
    """Return (gain, now_t) for a mouse-wheel notch; ramps toward fast gain on rapid spins."""
    now_t = time.perf_counter() if now is None else float(now)
    gain = wheel_base_gain()
    dt_notch = now_t - last_notch_t if last_notch_t > 0.0 else 1.0
    if dt_notch < 0.12:
        # 0 at 120ms → 1 at ~30ms
        speed = max(0.0, min(1.0, (0.12 - dt_notch) / 0.09))
        gain *= 1.0 + (wheel_fast_gain() - 1.0) * speed
    return gain, now_t
