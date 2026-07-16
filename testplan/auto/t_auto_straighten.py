#!/usr/bin/env python3
"""Auto-straighten estimator: synthetic tilted line → expected CropAngle."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _tilted_horizon(angle_deg: float, size: int = 800) -> np.ndarray:
    """White line on black at ``angle_deg`` (image coords, +CCW from +X)."""
    import cv2

    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = cy = size // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), np.sin(rad)
    length = size * 0.42
    p1 = (int(cx - dx * length), int(cy - dy * length))
    p2 = (int(cx + dx * length), int(cy + dy * length))
    cv2.line(img, p1, p2, (240, 240, 240), 3, cv2.LINE_AA)
    # A second parallel segment strengthens the Hough vote.
    off = 18
    nx, ny = -dy, dx
    q1 = (int(p1[0] + nx * off), int(p1[1] + ny * off))
    q2 = (int(p2[0] + nx * off), int(p2[1] + ny * off))
    cv2.line(img, q1, q2, (220, 220, 220), 2, cv2.LINE_AA)
    return img


def main() -> int:
    from raw_auto_adjust import estimate_straighten_angle

    # Already-level horizon → None or ~0 refused
    level = _tilted_horizon(0.0)
    a0 = estimate_straighten_angle(level)
    check(
        "level horizon refuses or near-zero",
        a0 is None or abs(a0) < 0.2,
        f"got {a0}",
    )

    # Line at +3° (tilts up-right in image y-down… arctan2): needs CropAngle ≈ -3
    tilted = _tilted_horizon(3.0)
    a3 = estimate_straighten_angle(tilted)
    check("detects +3° tilt", a3 is not None, f"got {a3}")
    if a3 is not None:
        check(
            "sign/magnitude near -3°",
            abs(a3 - (-3.0)) < 1.25,
            f"got {a3:.2f}",
        )

    tilted_n = _tilted_horizon(-4.0)
    an = estimate_straighten_angle(tilted_n)
    check("detects -4° tilt", an is not None, f"got {an}")
    if an is not None:
        check(
            "sign/magnitude near +4°",
            abs(an - 4.0) < 1.25,
            f"got {an:.2f}",
        )

    # Noise / organic — should refuse
    rng = np.random.RandomState(0)
    noise = rng.randint(0, 40, (600, 600, 3), dtype=np.uint8)
    check("noise returns None", estimate_straighten_angle(noise) is None)

    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS t_auto_straighten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
