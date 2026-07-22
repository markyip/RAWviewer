"""Camera colour calibration: sampling space, band averaging, profile lifecycle.

The pixel-math tests use a real ColorChecker frame when one is available (set
RAWVIEWER_COLORCHECKER_CR3 to a RAW containing a 24-patch chart, with
RAWVIEWER_COLORCHECKER_CORNERS as "x0,y0,x1,y1,x2,y2,x3,y3" in full-res pixels,
TL/TR/BR/BL). Without it those tests fall back to a synthetic chart, which still
catches the space mismatch -- an ideal camera must calibrate to ~zero.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from color_calibration import (  # noqa: E402
    COLORCHECKER_24_REF,
    _srgb_decode_0_255,
    calibrate_camera_curves_and_hsl,
    delete_camera_profile,
    describe_camera_profile,
    extract_patch_colors,
    get_camera_profile,
    save_camera_profile,
    validate_and_detect_color_checker,
)


def _synthetic_linear_patches() -> np.ndarray:
    """A perfect camera: scene-linear samples of the reference chart itself."""
    return _srgb_decode_0_255(COLORCHECKER_24_REF)


def _real_patches() -> np.ndarray | None:
    path = os.environ.get("RAWVIEWER_COLORCHECKER_CR3")
    corners_env = os.environ.get("RAWVIEWER_COLORCHECKER_CORNERS")
    if not path or not os.path.isfile(path) or not corners_env:
        return None
    vals = [float(v) for v in corners_env.split(",")]
    if len(vals) != 8:
        return None
    corners = [(vals[i], vals[i + 1]) for i in range(0, 8, 2)]
    from unified_image_processor import UnifiedImageProcessor

    base = UnifiedImageProcessor().decode_raw_edit_base(path, use_full_resolution=False)
    if base is None:
        return None
    lin8 = (np.clip(base.astype(np.float32), 0, 1) * 255.0).astype(np.uint8)
    return np.array(extract_patch_colors(lin8, corners), dtype=np.float32)


def _patches() -> tuple[np.ndarray, str]:
    real = _real_patches()
    if real is not None and len(real) == 24:
        return real, "real"
    return _synthetic_linear_patches(), "synthetic"


def test_neutral_camera_calibrates_to_near_zero() -> None:
    """Scene-linear samples must be sRGB-encoded before HSL deltas are taken.

    Deriving HSL on linear values against the sRGB reference chart biased every
    band the same direction (lum +5.8..+18.4, sat -2.2..-9.8 on a neutral Canon
    CR3) -- a fixed error applied to every camera regardless of its real
    response. A well-behaved camera must ask for almost no correction.
    """
    patches, source = _patches()
    prof = calibrate_camera_curves_and_hsl([tuple(p) for p in patches])

    worst_lum = max(abs(v) for v in prof["hsl_lum"].values())
    worst_sat = max(abs(v) for v in prof["hsl_sat"].values())
    assert worst_lum <= 2.0, f"[{source}] lum bias {worst_lum}: {prof['hsl_lum']}"
    assert worst_sat <= 2.5, f"[{source}] sat bias {worst_sat}: {prof['hsl_sat']}"


def test_exposure_invariance() -> None:
    """Same chart, different exposure -> same profile.

    The edit base is a linear decode with no auto-brighten, so the chart's
    white patch lands wherever exposure put it. Without normalising against the
    reference white, a darker frame of the same chart yields a different
    "calibration" for the same camera.
    """
    patches, source = _patches()
    bright = calibrate_camera_curves_and_hsl([tuple(p) for p in patches])
    dark = calibrate_camera_curves_and_hsl([tuple(p) for p in patches * 0.45])

    for band in bright["hsl_lum"]:
        d = abs(bright["hsl_lum"][band] - dark["hsl_lum"][band])
        assert d <= 1.5, f"[{source}] {band} lum drifted {d} with exposure"


def test_band_shifts_are_averaged_not_summed() -> None:
    """Bands have unequal patch counts; summing scaled the correction by count.

    Red/Blue/Yellow appear on 3 patches, Magenta on 1. Summing meant Red got
    ~3x Magenta's weight purely from chart layout, so a uniform per-patch error
    must not produce a 3x larger Red correction than Magenta.
    """
    patches, _ = _patches()
    # Uniform desaturation across every colour patch.
    skewed = patches.copy()
    grey = skewed[:18].mean(axis=1, keepdims=True)
    skewed[:18] = grey + (skewed[:18] - grey) * 0.8

    prof = calibrate_camera_curves_and_hsl([tuple(p) for p in skewed])
    red = abs(prof["hsl_sat"]["Red"])
    magenta = abs(prof["hsl_sat"]["Magenta"])
    assert magenta > 0.0, "Magenta band produced no correction at all"
    assert red / magenta < 2.0, (
        f"Red ({red}) scaled with patch count vs Magenta ({magenta}) -- summed, not averaged"
    )


def test_dead_curve_keys_not_emitted() -> None:
    """curve_r/g/b were computed, saved, and read by nothing."""
    patches, _ = _patches()
    prof = calibrate_camera_curves_and_hsl([tuple(p) for p in patches])
    for key in ("curve_r", "curve_g", "curve_b"):
        assert key not in prof, f"{key} is still emitted but nothing consumes it"


def test_profile_save_describe_delete_roundtrip() -> None:
    """A bad calibration must be removable from the UI, not only by hand."""
    make, model, iso = "RAWviewerTest", "SelfTest900", 12800
    patches, _ = _patches()
    prof = calibrate_camera_curves_and_hsl([tuple(p) for p in patches])
    try:
        assert save_camera_profile(make, model, prof, iso=iso)
        assert get_camera_profile(make, model, iso=iso) is not None
        assert describe_camera_profile(make, model, iso=iso)

        assert delete_camera_profile(make, model, iso=iso) is True
        assert get_camera_profile(make, model, iso=iso) is None
        # Idempotent: removing an absent profile reports False, never raises.
        assert delete_camera_profile(make, model, iso=iso) is False
    finally:
        delete_camera_profile(make, model, iso=iso)


def test_mcc_autodetect_finds_a_synthetic_chart() -> None:
    """Guards the nc-vs-chart-type trap in CCheckerDetector.process().

    The binding is ``process(image[, nc])`` where nc is the NUMBER OF CHARTS to
    look for. The original code passed ``mcc.MCC24`` -- an enum whose value is
    0 -- so it asked for zero charts and could never detect anything, silently,
    because the caller swallows failures. Skips when cv2.mcc is unavailable
    (plain opencv-python-headless rather than the contrib build).
    """
    try:
        import cv2
        import cv2.mcc as mcc
    except Exception:
        print("SKIP mcc autodetect (cv2.mcc unavailable -- non-contrib OpenCV)")
        return

    ph = pw = 120
    gap, marg = 18, 90
    h = 4 * ph + 5 * gap + 2 * marg
    w = 6 * pw + 7 * gap + 2 * marg
    img = np.full((h, w, 3), 30, np.uint8)
    for r in range(4):
        for c in range(6):
            y = marg + gap + r * (ph + gap)
            x = marg + gap + c * (pw + gap)
            img[y : y + ph, x : x + pw] = COLORCHECKER_24_REF[r * 6 + c].astype(np.uint8)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    det = mcc.CCheckerDetector.create()
    assert det.process(bgr, 3), "MCC failed to detect an ideal synthetic chart"
    assert len(det.getListColorChecker()) >= 1

    # The original call shape must stay broken-by-construction, so this test
    # fails loudly if someone reintroduces it.
    stale = mcc.CCheckerDetector.create()
    assert not stale.process(bgr, mcc.MCC24), (
        "mcc.MCC24 no longer evaluates to nc=0 -- revisit auto_detect()'s call"
    )


def test_validation_rejects_non_chart_region() -> None:
    flat = np.full((400, 600, 3), 96, dtype=np.uint8)
    ok, msg, sampled = validate_and_detect_color_checker(
        flat, [(0, 0), (599, 0), (599, 399), (0, 399)]
    )
    assert not ok and not sampled and msg


def main() -> int:
    test_neutral_camera_calibrates_to_near_zero()
    test_exposure_invariance()
    test_band_shifts_are_averaged_not_summed()
    test_dead_curve_keys_not_emitted()
    test_profile_save_describe_delete_roundtrip()
    test_mcc_autodetect_finds_a_synthetic_chart()
    test_validation_rejects_non_chart_region()
    print("PASS t_color_calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
