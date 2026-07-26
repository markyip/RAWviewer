#!/usr/bin/env python3
"""Automatic ColorChecker detection (cv2.mcc) — headless.

Auto-detection already works; this suite exists because nothing guarded
it. It has silently broken at least once: ``detector.process(img,
mcc.MCC24)`` passed a chart-type enum where the API wants nc, the NUMBER
of charts to look for. MCC24 == 0, so the detector was asked to find zero
charts and could never succeed — and the failure path is a quiet
``return False`` behind a "please place the corners manually" message, so
it looks like an ordinary hard-to-detect photo rather than a bug.

Checks:
  1. A chart is detected on dialog open, with no user interaction.
  2. Detected corners pass validate_and_detect_color_checker.
  3. A PERFECT chart calibrates to ~zero correction. This is the strongest
     assertion available without a real camera: it proves corners, patch
     ordering, sampling geometry and the calibration math all agree. Any
     of them being wrong shows up as a large bogus correction.
  4. Rotation is handled — mcc returns corners in the CHART's orientation,
     so an upside-down chart must still calibrate to ~zero rather than
     mapping white onto black.
  5. Perspective (a handheld, off-square shot) is handled.
  6. An image with no chart returns False and does not raise.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FAILURES = []

# A perfect synthetic chart should need essentially no correction. Allow a
# little slack for 8-bit quantisation and the centre-50% patch sampling.
PERFECT_CHART_TOLERANCE = 2.0


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name} {detail}".rstrip())
    else:
        print(f"  FAIL  {name} {detail}".rstrip())
        FAILURES.append(name)


def _synth_srgb(patch=80, gap=10, margin=60):
    """A flat, evenly-lit 24-patch chart in display-referred sRGB."""
    from color_calibration import COLORCHECKER_24_REF

    w = 6 * patch + 5 * gap + 2 * margin
    h = 4 * patch + 3 * gap + 2 * margin
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (30, 30, 30)
    for r in range(4):
        for c in range(6):
            y = margin + r * (patch + gap)
            x = margin + c * (patch + gap)
            img[y : y + patch, x : x + patch] = COLORCHECKER_24_REF[r * 6 + c]
    return img


def _to_scene_linear(srgb_u8):
    """The dialog is handed a scene-linear edit base, not an sRGB image."""
    s = srgb_u8.astype(np.float32) / 255.0
    lin = np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
    return (lin * 255.0).astype(np.float32)


def _make_dialog(linear_img):
    from rawviewer_ui.color_calibration_dialog import ColorCalibrationDialog

    return ColorCalibrationDialog(linear_img, "TestMake", "TestModel", iso=100)


def _worst_correction(profile):
    return max(
        abs(v)
        for group in ("hsl_hue", "hsl_sat", "hsl_lum")
        for v in profile[group].values()
    )


def _detect_and_calibrate(label, srgb_img, tolerance=PERFECT_CHART_TOLERANCE):
    """Full chain: dialog open -> auto-detect -> validate -> calibrate."""
    from color_calibration import (
        calibrate_camera_curves_and_hsl,
        validate_and_detect_color_checker,
    )

    dlg = _make_dialog(_to_scene_linear(srgb_img))
    detected = dlg.canvas.auto_detect()
    check(f"{label}: chart auto-detected", detected is True)
    if not detected:
        return None

    corners = dlg.canvas.get_pixel_corners()
    check(f"{label}: four corners returned", len(corners) == 4, f"got {len(corners)}")

    valid, message, sampled = validate_and_detect_color_checker(dlg.canvas.image, corners)
    check(f"{label}: detected corners pass validation", valid, message[:80])
    if not valid:
        return None

    profile = calibrate_camera_curves_and_hsl(sampled)
    worst = _worst_correction(profile)
    # A perfect chart needs no correction. A large value here means the
    # corners, the patch order, or the sampling geometry is wrong -- the
    # failure this whole chain is prone to and cannot otherwise detect.
    check(
        f"{label}: perfect chart calibrates to ~zero",
        worst <= tolerance,
        f"worst |HSL delta| = {worst:.2f} (tolerance {tolerance})",
    )
    return profile


def test_upright():
    _detect_and_calibrate("upright", _synth_srgb())


def test_rotated_180():
    """mcc returns corners in the chart's own orientation, not the image's.

    If that were not true, an upside-down chart would map the white patch
    onto black and produce a wild correction instead of ~zero.
    """
    _detect_and_calibrate("rot180", cv2.rotate(_synth_srgb(), cv2.ROTATE_180))


def test_rotated_90():
    _detect_and_calibrate("rot90", cv2.rotate(_synth_srgb(), cv2.ROTATE_90_CLOCKWISE))


def test_perspective():
    """An off-square handheld shot must still resolve to the right patches."""
    base = _synth_srgb()
    h, w = base.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[40, 15], [w - 15, 45], [w - 45, h - 20], [15, h - 40]])
    warped = cv2.warpPerspective(
        base, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(30, 30, 30)
    )
    # Slightly looser: resampling softens patch edges.
    _detect_and_calibrate("perspective", warped, tolerance=4.0)


def test_noisy():
    rng = np.random.default_rng(0)
    base = _synth_srgb().astype(np.int16)
    noisy = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    _detect_and_calibrate("noisy", noisy, tolerance=4.0)


def test_no_chart_present():
    """No chart must fail cleanly, not raise and not invent corners."""
    rng = np.random.default_rng(1)
    noise = (rng.random((400, 600, 3)) * 255).astype(np.uint8)
    dlg = _make_dialog(_to_scene_linear(noise))
    try:
        detected = dlg.canvas.auto_detect()
        check("no chart: returns False", detected is False, f"got {detected}")
    except Exception as exc:  # noqa: BLE001
        check("no chart: returns False", False, f"raised {exc!r}")

    # The canvas must still hold four usable corners for manual placement.
    check("no chart: manual corners still available", len(dlg.canvas.corners) == 4)


def test_mcc_available():
    """cv2.mcc lives in opencv-contrib; a base opencv build would lose this."""
    check("cv2.mcc present", hasattr(cv2, "mcc"))
    check("CCheckerDetector.create works", cv2.mcc.CCheckerDetector.create() is not None)


def main():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    print("Automatic ColorChecker detection (cv2.mcc)")
    test_mcc_available()
    test_upright()
    test_rotated_180()
    test_rotated_90()
    test_perspective()
    test_noisy()
    test_no_chart_present()

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
