#!/usr/bin/env python3
"""XMP sidecar round-trip: every persisted control survives write -> load."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"

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

        # resetting to defaults deletes the sidecar (when nothing foreign remains)
        clean = dict(DEFAULT_ADJUSTMENTS)
        clean[AS_SHOT_TEMP_KEY] = 5500.0
        clean["Temperature"] = 5500.0
        write_xmp_adjustments_for_file(img, clean)
        check("default adjustments delete sidecar", not os.path.isfile(resolve_xmp_path(img)))

        # Identity tone-curve serial must count as default (Reset / clear).
        from raw_adjustments import is_default_adjustments, tone_curve_serial_is_active

        ident = dict(DEFAULT_ADJUSTMENTS)
        ident[AS_SHOT_TEMP_KEY] = 5500.0
        ident["Temperature"] = 5500.0
        ident[TONE_CURVE_SERIAL_KEY] = "0,0;255,255"
        check("identity serial not active", not tone_curve_serial_is_active(ident[TONE_CURVE_SERIAL_KEY]))
        check("identity serial is_default", is_default_adjustments(ident))
        write_xmp_adjustments_for_file(img, ident)
        check("identity curve does not keep sidecar", not os.path.isfile(resolve_xmp_path(img)))

        # Merge write must preserve foreign Lightroom crs fields.
        xmp = resolve_xmp_path(img)
        with open(xmp, "w", encoding="utf-8") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
                '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
                '<rdf:Description rdf:about="" '
                'xmlns:crs="http://adobe.com/camera-raw-settings/1.0/" '
                'crs:GrainAmount="25" crs:LookTable="1"/>\n'
                "</rdf:RDF></x:xmpmeta>\n"
            )
        adj2 = dict(DEFAULT_ADJUSTMENTS)
        adj2[AS_SHOT_TEMP_KEY] = 5500.0
        adj2["Exposure2012"] = 0.5
        write_xmp_adjustments_for_file(img, adj2)
        text = open(xmp, encoding="utf-8").read()
        check("merge keeps GrainAmount", "GrainAmount" in text and "25" in text)
        check("merge keeps LookTable", "LookTable" in text)
        check("merge writes Exposure2012", "Exposure2012" in text)
        back2 = load_adjustments_for_file(img)
        check(
            "merge round-trip Exposure",
            abs(float(back2.get("Exposure2012", 0)) - 0.5) < 0.01,
        )
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
