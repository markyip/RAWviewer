#!/usr/bin/env python3
"""Dodge/Burn + Mask brush: wheel changes size/flow; preview uses flow; no hard ring."""
import inspect
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
    from raw_mask_layers import MaskLayer, MaskLayerStack
    import numpy as np

    wheel_src = inspect.getsource(giv.GpuImageView.wheelEvent)
    check(
        "brush mode consumes plain wheel for brush size",
        "_dodge_burn_mode" in wheel_src and "dodgeBurnBrushSizeWheel" in wheel_src,
    )
    check(
        "brush mode consumes horizontal wheel for flow",
        "dodgeBurnBrushStrengthWheel" in wheel_src
        and "angleDelta().x()" in wheel_src,
    )
    check(
        "Ctrl+wheel still zooms while a brush is armed",
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

    main_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py"
    )
    text = open(main_src, encoding="utf-8").read()
    start = text.index("def _adjust_brush_size_wheel_armed")
    end = text.index("\n    def ", start + 1)
    gate_src = text[start:end]
    check(
        "mask Paint/Erase also arms the brush-size wheel",
        '"paint"' in gate_src and '"erase"' in gate_src,
    )
    check(
        "size-wheel host uses the shared brush-armed gate",
        "_adjust_brush_size_wheel_armed" in text[
            text.index("def _on_dodge_burn_brush_size_wheel")
            : text.index("\n    def ", text.index("def _on_dodge_burn_brush_size_wheel") + 1)
        ],
    )

    # Behaviour: Masks Paint + wheel nudge moves the shared size slider.
    p = ap.ImageAdjustPanelWidget()
    p.show()
    p._panel_tabs.set_current(1)
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1.0
    p.set_mask_layer_stack(MaskLayerStack([MaskLayer(name="Brush 1", alpha=alpha)]))
    p.set_mask_layer_mode("paint")
    check("mask Paint arms", p.mask_layer_mode() == "paint")
    before = p.dodge_burn_brush_radius()
    p.nudge_dodge_burn_brush_size(120)
    check(
        "mask Paint can nudge brush size via the shared slider",
        p.dodge_burn_brush_radius() > before,
        f"{before} -> {p.dodge_burn_brush_radius()}",
    )
    flow_before = p.dodge_burn_brush_strength()
    p.nudge_dodge_burn_brush_strength(120)
    check(
        "mask Paint can nudge brush flow via the shared slider",
        p.dodge_burn_brush_strength() > flow_before,
        f"{flow_before} -> {p.dodge_burn_brush_strength()}",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
