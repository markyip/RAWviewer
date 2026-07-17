#!/usr/bin/env python3
"""Dodge/Burn brush: wheel changes size; preview uses flow; no hard ring."""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    from rawviewer_ui import gpu_image_view as giv
    from rawviewer_ui import adjust_panel as ap

    wheel_src = inspect.getsource(giv.GpuImageView.wheelEvent)
    check(
        "D&B mode consumes plain wheel for brush size",
        "_dodge_burn_mode" in wheel_src and "dodgeBurnBrushSizeWheel" in wheel_src,
    )
    check(
        "Ctrl+wheel still zooms while D&B is armed",
        "ControlModifier" in wheel_src,
    )

    ensure_src = inspect.getsource(giv.GpuImageView._ensure_brush_cursor_pixmap)
    check(
        "brush preview opacity uses flow",
        "_dodge_burn_brush_flow" in ensure_src and "peak_alpha" in ensure_src,
    )
    check(
        "brush preview has no hard size ring",
        "ring =" not in ensure_src and "np.abs(dist - r)" not in ensure_src,
    )

    nudge_src = inspect.getsource(ap.ImageAdjustPanelWidget.nudge_dodge_burn_brush_size)
    check(
        "panel exposes nudge_dodge_burn_brush_size",
        "_db_size_slider" in nudge_src,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
