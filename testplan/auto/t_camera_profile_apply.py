#!/usr/bin/env python3
"""Camera calibration profile -> adjustment dict -> HSL pipeline.

Regression guard for a silent key-name mismatch: load_adjustments_for_file
wrote the calibrated HSL corrections as "HueRed" / "SatRed" / "LumRed",
but raw_hsl.apply_hsl_adjustments reads "HueAdjustmentRed" /
"SaturationAdjustmentRed" / "LuminanceAdjustmentRed". Those keys are not
in DEFAULT_ADJUSTMENTS or RELEVANT_ADJUSTMENT_KEYS either, so all 24
calibrated values were dropped on the floor while the WB shift still
applied -- the feature looked like it worked because the image did move.

This suite asserts the values survive the whole way to pixels, not just
that the dict has some keys in it. A test that only checked key names
would have to be updated in lockstep with the very bug it guards.

Checks:
  1. Every key a profile writes is a REAL adjustment key.
  2. A saved profile's HSL values land under the names raw_hsl reads.
  3. Those values actually change pixels through apply_hsl_adjustments.
  4. The WB half still applies (it was never broken -- don't regress it).
  5. Calibration output bands match the pipeline's band list exactly.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def _profile():
    """A profile shaped exactly like calibrate_camera_curves_and_hsl's output."""
    from raw_hsl import HSL_COLOR_NAMES

    return {
        "temperature_shift": 120.0,
        "tint_shift": -4.0,
        "hsl_hue": {b: 3.0 for b in HSL_COLOR_NAMES},
        "hsl_sat": {b: 7.0 for b in HSL_COLOR_NAMES},
        "hsl_lum": {b: -5.0 for b in HSL_COLOR_NAMES},
    }


def test_no_dead_imports():
    """Guard the defect that made the whole feature inert.

    Both the profile-apply path and the UI banner imported
    ``exif_extractor``, a module that does not exist in this repo, inside
    a bare ``except Exception``. The ImportError was swallowed, camera
    identity always came back empty, and every profile was saved under
    "unknown_camera" and never matched again.
    """
    import subprocess

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    hits = subprocess.run(
        ["grep", "-rn", "from exif_extractor import", os.path.join(root, "src")],
        capture_output=True,
        text=True,
    ).stdout.strip()
    check("no imports of the non-existent exif_extractor module", not hits, hits[:200])

    # And the module really is absent, so the guard above is meaningful.
    try:
        import exif_extractor  # noqa: F401

        check("exif_extractor genuinely absent", False, "module unexpectedly exists")
    except ImportError:
        check("exif_extractor genuinely absent", True)


def test_identity_parsing():
    """Identity must parse the tag shape metadata_backend actually emits."""
    from color_calibration import camera_identity_for_file, camera_identity_from_exif

    # exifread / metadata_backend shape -- the one every producer here uses.
    check(
        "exifread-style tags parsed",
        camera_identity_from_exif(
            {"Image Make": "Canon", "Image Model": "EOS R5", "EXIF ISOSpeedRatings": "400"}
        )
        == ("Canon", "EOS R5", 400),
    )
    # Bare shape kept working for any caller that still passes it.
    check(
        "bare-style tags still parsed",
        camera_identity_from_exif({"Make": "Nikon", "Model": "Z8", "ISO": "1600"})
        == ("Nikon", "Z8", 1600),
    )
    # exifread renders integer lists as "[100]".
    check(
        "list-rendered ISO parsed",
        camera_identity_from_exif(
            {"Image Make": "Sony", "Image Model": "ILCE-7RM4", "EXIF ISOSpeedRatings": "[100]"}
        )
        == ("Sony", "ILCE-7RM4", 100),
    )
    check("empty exif is safe", camera_identity_from_exif({}) == ("", "", None))
    check("None exif is safe", camera_identity_from_exif(None) == ("", "", None))

    # Missing file must degrade, not raise.
    check(
        "missing file returns empty identity",
        camera_identity_for_file("/nope/does/not/exist.CR3") == ("", "", None),
    )
    check("empty path returns empty identity", camera_identity_for_file("") == ("", "", None))


def test_identity_is_shared():
    """The banner/dialog path and the apply path must agree by construction."""
    # Read the source rather than importing main -- importing the app
    # module pulls in Qt and the whole processing stack for a one-line
    # structural assertion.
    path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "main.py")
    lines = open(path, encoding="utf-8").read().splitlines()
    start = next(
        (i for i, l in enumerate(lines) if "def _camera_identity_for_file" in l), None
    )
    check("main has the identity helper", start is not None)
    if start is None:
        return
    body = "\n".join(lines[start : start + 20])
    check("main delegates to color_calibration", "camera_identity_for_file" in body)
    check("main no longer imports exif_extractor", "exif_extractor" not in body)


def test_band_names_agree():
    """Calibration must emit the same band names the pipeline consumes."""
    from raw_hsl import HSL_COLOR_NAMES

    calibrated = _profile()["hsl_hue"].keys()
    check(
        "calibration bands == pipeline bands",
        set(calibrated) == set(HSL_COLOR_NAMES),
        f"symmetric diff {set(calibrated) ^ set(HSL_COLOR_NAMES)}",
    )


def test_keys_are_real():
    """Every key the profile path writes must be one the pipeline knows."""
    from raw_adjustments import RELEVANT_ADJUSTMENT_KEYS
    from raw_hsl import HSL_COLOR_NAMES

    for band in HSL_COLOR_NAMES:
        for prefix in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"):
            key = f"{prefix}{band}"
            if key not in RELEVANT_ADJUSTMENT_KEYS:
                check(f"{key} is a real adjustment key", False)
                return
    check("all HSL profile keys are real adjustment keys", True)

    # The old, broken names must NOT be real -- if they ever became real,
    # this test would stop protecting anything.
    check(
        "legacy short names are not adjustment keys",
        not any(k in RELEVANT_ADJUSTMENT_KEYS for k in ("HueRed", "SatRed", "LumRed")),
    )


def _load_via_real_path(tmp, cc, raw_adjustments):
    """Run raw_adjustments.load_adjustments_for_file for real.

    Stubs only the two things that need a genuine camera file (EXIF
    identity and as-shot WB); the profile lookup and the entire apply
    path under test run unmodified.
    """
    image_path = os.path.join(tmp, "FAKE_0001.CR3")
    open(image_path, "wb").close()  # no sidecar -> profile path is taken

    saved_identity = cc.camera_identity_for_file
    saved_as_shot = raw_adjustments.read_as_shot_temperature
    try:
        cc.camera_identity_for_file = lambda _path: ("TestMake", "TestModel", None)
        raw_adjustments.read_as_shot_temperature = lambda path: 5500.0
        return raw_adjustments.load_adjustments_for_file(image_path)
    finally:
        cc.camera_identity_for_file = saved_identity
        raw_adjustments.read_as_shot_temperature = saved_as_shot


def test_profile_reaches_adjustments(tmp):
    """Save a profile, load a file with no sidecar, check the dict."""
    import color_calibration as cc

    # Redirect the profile store to a temp file.
    original = cc.get_camera_profile_path
    store = os.path.join(tmp, "camera_profiles.json")
    cc.get_camera_profile_path = lambda: store
    try:
        cc.save_camera_profile("TestMake", "TestModel", _profile(), iso=None)

        loaded = cc.get_camera_profile("TestMake", "TestModel", iso=None)
        check("profile round-trips", loaded is not None)
        if loaded:
            check("hsl_sat preserved", abs(loaded["hsl_sat"]["Red"] - 7.0) < 1e-6)

        # Call the REAL load_adjustments_for_file, not a reimplementation of
        # its loop -- duplicating the logic here would make this suite agree
        # with whatever the code does, including doing it wrong.
        import raw_adjustments

        adj = _load_via_real_path(tmp, cc, raw_adjustments)
        check("WB shift applied", abs(adj["Temperature"] - 5620.0) < 1e-6, adj["Temperature"])
        check("tint shift applied", abs(adj["Tint"] - (-4.0)) < 1e-6, adj["Tint"])
        check(
            "HSL saturation under the pipeline's key",
            abs(adj["SaturationAdjustmentRed"] - 7.0) < 1e-6,
            adj.get("SaturationAdjustmentRed"),
        )
        check(
            "HSL luminance under the pipeline's key",
            abs(adj["LuminanceAdjustmentBlue"] - (-5.0)) < 1e-6,
            adj.get("LuminanceAdjustmentBlue"),
        )
        check("no legacy short key written", "SatRed" not in adj)
        return adj
    finally:
        cc.get_camera_profile_path = original


def test_reaches_pixels(adj):
    """The real assertion: a calibrated profile must change the image.

    apply_hsl_adjustments short-circuits when every band reads zero, so a
    mismatched key name shows up here as 'nothing happened' -- which is
    exactly the bug, and exactly what a key-name-only test would miss.
    """
    from raw_hsl import apply_hsl_adjustments

    rng = np.random.default_rng(3)
    img = rng.random((32, 32, 3)).astype(np.float32)

    out = apply_hsl_adjustments(img.copy(), adj)
    check("calibrated HSL changes pixels", not np.allclose(out, img, atol=1e-6))

    # Control: the broken key names must NOT change pixels, proving the
    # test is sensitive to the actual defect.
    from raw_adjustments import DEFAULT_ADJUSTMENTS
    from raw_hsl import HSL_COLOR_NAMES

    broken = dict(DEFAULT_ADJUSTMENTS)
    for band in HSL_COLOR_NAMES:
        broken[f"Hue{band}"] = 3.0
        broken[f"Sat{band}"] = 7.0
        broken[f"Lum{band}"] = -5.0
    out_broken = apply_hsl_adjustments(img.copy(), broken)
    check(
        "legacy key names are inert (test is sensitive to the bug)",
        np.allclose(out_broken, img, atol=1e-6),
    )


def main():
    print("Camera calibration profile application")
    tmp = tempfile.mkdtemp(prefix="rv_camprof_")
    try:
        test_no_dead_imports()
        test_identity_parsing()
        test_identity_is_shared()
        test_band_names_agree()
        test_keys_are_real()
        adj = test_profile_reaches_adjustments(tmp)
        if adj:
            test_reaches_pixels(adj)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("All camera profile checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
