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
from focus_stacking import (  # noqa: E402
    _PARALLAX_WARN_PX,
    _seam_aware_weights,
    _sharpness_map,
    _to_gray_u8,
    align_focus_stack,
    focus_stack,
    refine_alignment_flow,
)


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


def test_flow_fixes_local_parallax_shift() -> None:
    """v2: dense flow must absorb a depth-dependent shift a global affine can't.

    Only the left third moves (a near object); a single affine can't satisfy
    that partial shift, so the flow refinement is what closes the residual.
    """
    rng = np.random.default_rng(5)
    h, w = 240, 360
    ref = cv2.GaussianBlur((rng.random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 1.2)
    mov = ref.copy()
    mov[:, : w // 3] = np.roll(ref[:, : w // 3], 8, axis=1)

    def err(a, b):
        sl = slice(0, w // 3)
        return float(np.mean(np.abs(
            cv2.cvtColor(a[:, sl], cv2.COLOR_RGB2GRAY).astype(np.float32)
            - cv2.cvtColor(b[:, sl], cv2.COLOR_RGB2GRAY).astype(np.float32)
        )))

    before = err(ref, mov)
    refined, parallax = refine_alignment_flow([ref, mov])
    after = err(ref, refined[1])
    # Confidence gating trades some parallax reduction for sharp-detail safety
    # (see refine_alignment_flow); a meaningful cut, not perfection, is the bar.
    assert after < before * 0.6, f"flow barely helped: {before:.1f} -> {after:.1f}"
    assert 4.0 < parallax < 10.0, f"parallax metric off: {parallax} (shift was 8px)"


def test_local_align_does_not_soften_clean_stack() -> None:
    """The parallax fix must not degrade a stack that has no parallax.

    Flow refinement resamples pixels; naively applied it halved sharpness in
    reference-blurred regions. Confidence gating + keeping original pixels where
    untrusted must hold a no-parallax merge close to the no-warp result.
    """
    base, a, b, w = _split_focus_pair(seed=0)
    v1 = focus_stack([a, b], align=False, local_align=False, seam_aware=True, auto_crop=False)
    v2 = focus_stack([a, b], align=False, local_align=True, seam_aware=True, auto_crop=False)
    for half in (slice(0, w // 2), slice(w // 2, None)):
        s1 = _sharp(v1.image[:, half])
        s2 = _sharp(v2.image[:, half])
        assert s2 > 0.65 * s1, f"local align softened a clean stack: {s2:.0f} vs {s1:.0f}"


def test_parallax_warning_fires_above_threshold() -> None:
    """A stack with real parallax must warn; a clean one must not."""
    _base, a, b, _w = _split_focus_pair(seed=9)
    clean = focus_stack([a, b], align=True, local_align=True)
    assert clean.success
    assert not any("parallax" in wmsg.lower() for wmsg in clean.alignment_warnings), \
        clean.alignment_warnings

    # Shift the whole second frame well past the warn threshold.
    shifted = np.roll(b, int(_PARALLAX_WARN_PX * 3) + 2, axis=1)
    noisy = focus_stack([a, shifted], align=False, local_align=True)
    assert noisy.success
    assert any("parallax" in wmsg.lower() for wmsg in noisy.alignment_warnings), \
        noisy.alignment_warnings


def test_seam_aware_weights_are_coherent() -> None:
    """Seam-aware weighting must assign a region wholesale, not per-pixel speckle."""
    _base, a, b, w = _split_focus_pair(seed=4)
    maps = [_sharpness_map(_to_gray_u8(a)), _sharpness_map(_to_gray_u8(b))]
    weights = _seam_aware_weights(maps, _to_gray_u8(a))
    # Frame A is sharp on the left, so its weight must dominate there and recede
    # on the right -- a coherent split, not a coin-flip per pixel.
    wa = weights[0]
    left_mean = float(wa[:, : w // 2].mean())
    right_mean = float(wa[:, w // 2:].mean())
    assert left_mean > 0.7 and right_mean < 0.3, f"incoherent split: L={left_mean:.2f} R={right_mean:.2f}"


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
    assert dlg.should_correct_parallax() is True  # v2 control present + on
    assert len(dlg.get_active_image_paths()) == 3
    # No HDR weight sliders in focus mode.
    assert dlg.get_hdr_weights() == {"highlight": 1.0, "shadow": 1.0, "midtone": 1.0}


def main() -> int:
    test_fusion_recovers_sharp_regions()
    test_alignment_reduces_breathing_error()
    test_flow_fixes_local_parallax_shift()
    test_local_align_does_not_soften_clean_stack()
    test_parallax_warning_fires_above_threshold()
    test_seam_aware_weights_are_coherent()
    test_rejects_mismatched_dimensions()
    test_rejects_single_image()
    test_preserves_bit_depth()
    test_dialog_focus_mode_builds()
    print("PASS t_focus_stacking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
