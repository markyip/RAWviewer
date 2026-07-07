#!/usr/bin/env python3
"""XMP sidecar round-trip: every persisted control survives write -> load."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_adjustments import (
        AS_SHOT_TEMP_KEY,
        DEFAULT_ADJUSTMENTS,
        load_adjustments_for_file,
        read_as_shot_temperature,
        resolve_xmp_path,
        write_xmp_adjustments_for_file,
    )
    from raw_tone_curve import (
        TONE_CURVE_BLUE_KEY,
        TONE_CURVE_GREEN_KEY,
        TONE_CURVE_RED_KEY,
        TONE_CURVE_SERIAL_KEY,
    )

    tmpd = tempfile.mkdtemp()
    try:
        img = os.path.join(tmpd, "t.CR3")
        open(img, "wb").write(b"stub")

        adj = dict(DEFAULT_ADJUSTMENTS)
        adj[AS_SHOT_TEMP_KEY] = 5500.0
        edits = {
            "Exposure2012": 1.25, "Contrast2012": 10.0, "Highlights2012": -45.0,
            "Shadows2012": 60.0, "Whites2012": -20.0, "Blacks2012": 30.0,
            "Temperature": 4800.0, "Tint": 12.0, "Saturation": 5.0,
            "Vibrance": 15.0, "Sharpness": 40.0, "Clarity2012": 8.0,
            "ColorNoiseReduction": 50.0, "DenoiseMethod": 1.0,
            "LensCorrectionEnabled": 1.0,
            TONE_CURVE_SERIAL_KEY: "0,0;64,80;255,255",
            TONE_CURVE_RED_KEY: "0,0;128,150;255,255",
            TONE_CURVE_GREEN_KEY: "0,0;128,110;255,255",
            TONE_CURVE_BLUE_KEY: "0,10;128,140;255,245",
        }
        hsl = [k for k in DEFAULT_ADJUSTMENTS if k.startswith(("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"))][:3]
        for k in hsl:
            edits[k] = 25.0
        adj.update(edits)

        write_xmp_adjustments_for_file(img, adj)
        check("sidecar written", os.path.isfile(resolve_xmp_path(img)))

        back = load_adjustments_for_file(img)
        for k, v in edits.items():
            got = back.get(k, "<MISSING>")
            if isinstance(v, str):
                ok = str(got) == v
            else:
                ok = got != "<MISSING>" and abs(float(got) - float(v)) < 0.51
            check(f"round-trip {k}", ok, f"wrote {v!r} loaded {got!r}")

        # as-shot memo deterministic
        t1, t2 = read_as_shot_temperature(img), read_as_shot_temperature(img)
        check("as-shot temperature deterministic", t1 == t2)

        # resetting to defaults deletes the sidecar
        clean = dict(DEFAULT_ADJUSTMENTS)
        clean[AS_SHOT_TEMP_KEY] = 5500.0
        clean["Temperature"] = 5500.0
        write_xmp_adjustments_for_file(img, clean)
        check("default adjustments delete sidecar", not os.path.isfile(resolve_xmp_path(img)))
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
