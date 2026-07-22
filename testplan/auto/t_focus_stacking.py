"""Focus stacking: fusion picks the sharpest source per region; dialog wiring."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "rawviewer_ui"))

import cv2  # noqa: E402
from focus_stacking import align_focus_stack, focus_stack  # noqa: E402


def _split_focus_pair(seed: int = 0):
    """Two frames of one textured scene: left sharp in A, right sharp in B."""
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    base = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    a = base.copy()
    a[:, w // 2:] = cv2.GaussianBlur(a[:, w // 2:], (0, 0), 6)
    b = base.copy()
    b[:, : w // 2] = cv2.GaussianBlur(b[:, : w // 2], (0, 0), 6)
    return base, a, b, w


def _sharp(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def test_fusion_recovers_sharp_regions() -> None:
    base, a, b, w = _split_focus_pair()
    res = focus_stack([a, b], align=False, auto_crop=False)
    assert res.success and res.image is not None
    out = res.image
    # Each source is blurred on one half; the fused frame must be sharp on both,
    # close to the fully-sharp base and far above the blurred inputs.
    for half in (slice(0, w // 2), slice(w // 2, None)):
        fused_s = _sharp(out[:, half])
        base_s = _sharp(base[:, half])
        assert fused_s > 0.5 * base_s, f"fused half too soft: {fused_s} vs base {base_s}"
    # The blurred half of each input is near-zero sharpness; fused must beat it.
    assert _sharp(out[:, w // 2:]) > 10 * _sharp(a[:, w // 2:])
    assert _sharp(out[:, : w // 2]) > 10 * _sharp(b[:, : w // 2])


def test_alignment_reduces_breathing_error() -> None:
    base, a, _b, w = _split_focus_pair(seed=2)
    # Frame B breathes: 3% scale + a few px shift, plus opposite-half blur.
    m = cv2.getRotationMatrix2D((w / 2, 120), 0, 1.03)
    m[0, 2] += 5
    m[1, 2] += 3
    warped = cv2.warpAffine(base, m, (w, 240), borderMode=cv2.BORDER_REFLECT)
    b = warped.copy()
    b[:, : w // 2] = cv2.GaussianBlur(b[:, : w // 2], (0, 0), 6)

    def err(x):
        return float(np.mean(np.abs(
            cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
            - cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
        )))

    aligned, _warn = align_focus_stack([a, warped])
    assert err(aligned[1]) < err(warped), "alignment did not reduce breathing error"


def test_rejects_mismatched_dimensions() -> None:
    a = np.zeros((100, 100, 3), np.uint8)
    b = np.zeros((100, 120, 3), np.uint8)
    res = focus_stack([a, b])
    assert not res.success and "dimension" in res.error_message.lower()


def test_rejects_single_image() -> None:
    res = focus_stack([np.zeros((50, 50, 3), np.uint8)])
    assert not res.success


def test_preserves_bit_depth() -> None:
    _base, a, b, _w = _split_focus_pair(seed=3)
    for cast, dt in ((lambda x: x, np.uint8),
                     (lambda x: (x.astype(np.float32) * 257), np.uint16),
                     (lambda x: x.astype(np.float32) / 255.0, np.float32)):
        res = focus_stack([cast(a).astype(dt), cast(b).astype(dt)], align=False)
        assert res.success and res.image.dtype == dt, f"{dt} not preserved"


def test_dialog_focus_mode_builds() -> None:
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui.hdr_panorama_dialog import HDRPanoramaDialog

    _app = QApplication.instance() or QApplication([])
    dlg = HDRPanoramaDialog(["/x/a.CR3", "/x/b.CR3", "/x/c.CR3"], mode="focus_stack")
    assert "Focus" in dlg.windowTitle()
    assert hasattr(dlg, "should_align")
    assert dlg.should_align() is True
    assert len(dlg.get_active_image_paths()) == 3
    # No HDR weight sliders in focus mode.
    assert dlg.get_hdr_weights() == {"highlight": 1.0, "shadow": 1.0, "midtone": 1.0}


def main() -> int:
    test_fusion_recovers_sharp_regions()
    test_alignment_reduces_breathing_error()
    test_rejects_mismatched_dimensions()
    test_rejects_single_image()
    test_preserves_bit_depth()
    test_dialog_focus_mode_builds()
    print("PASS t_focus_stacking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
