"""Focus outline + Space: atomic GPU zoom wiring checks."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_focus_zoom_wiring() -> None:
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_path, encoding="utf-8") as f:
        src = f.read()

    zoom_fn = src.split("def _zoom_in_to_image_point_finish")[1].split(
        "def _single_view_is_fit_mode"
    )[0]
    assert "focus_jump" in zoom_fn
    assert "if not focus_jump:" in zoom_fn
    assert "self._pending_zoom = True" in zoom_fn

    anchor_fn = src.split("def _apply_gpu_pixmap_with_focus_anchor")[1].split(
        "def _clear_pending_point_zoom_state"
    )[0]
    assert "set_pixmap_zoomed_at" in anchor_fn
    assert "center_on_image_point" not in anchor_fn

    finish_fn = src.split("def _finish_pending_zoom_after_full_load")[1].split(
        "def _sync_displayed_half_size_flag"
    )[0]
    assert "set_pixmap_zoomed_at" in finish_fn
    assert "center_on_image_point" not in finish_fn


if __name__ == "__main__":
    test_focus_zoom_wiring()
    print("OK: focus Space zoom wiring checks passed")
