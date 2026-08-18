#!/usr/bin/env python3
"""Export filename templates, watermark, and GPS XMP strip."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from export_options import (
        apply_export_watermark_rgb,
        apply_signature_watermark_rgb,
        apply_text_watermark_rgb,
        format_export_filename,
        load_signature_font,
        normalize_signature_font_id,
        normalize_watermark_corner,
        strip_gps_from_xmp_text,
        template_context_for_path,
        watermark_kwargs_from_prefs,
    )

    ctx = template_context_for_path(
        "/photos/DSC_0099.CR3",
        export_format="jpeg",
        seq=3,
        rating=5,
        label="Red",
        capture_time="2024:08:13 17:05:09",
    )
    name = format_export_filename("{name}_{yyyy}{mm}{dd}_{seq}.{ext}", ctx)
    check("template tokens", name == "DSC_0099_20240813_003.jpg", name)

    default = format_export_filename("", ctx)
    check("default template", default.endswith(".jpg") and "DSC_0099" in default, default)

    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[:] = 30
    out = apply_text_watermark_rgb(rgb, "© test", opacity=1.0)
    check("watermark changes pixels", out is not rgb and np.mean(out) > np.mean(rgb))

    canvas = np.full((200, 320, 3), 40, dtype=np.uint8)
    remark_only = apply_export_watermark_rgb(canvas, remark="© proof", font_id="dejavu")
    check("remark-only changes pixels", not np.array_equal(remark_only, canvas))
    sig_only = apply_export_watermark_rgb(
        canvas, signature="Jane Doe", font_id="dejavu"
    )
    check("signature-only changes pixels", not np.array_equal(sig_only, canvas))
    both = apply_export_watermark_rgb(
        canvas, remark="© proof", signature="Jane Doe", font_id="dejavu"
    )
    check("both differs from remark-only", not np.array_equal(both, remark_only))
    check("both differs from signature-only", not np.array_equal(both, sig_only))
    empty = apply_export_watermark_rgb(canvas, remark="", signature="")
    check("both empty leaves image", np.array_equal(empty, canvas))

    missing = apply_signature_watermark_rgb(canvas, "Jane Doe", font_id="not-a-real-font")
    check("unknown font still draws", not np.array_equal(missing, canvas))
    font = load_signature_font("dejavu", 32)
    check("dejavu fallback loads", font is not None)
    check("font id normalize", normalize_signature_font_id("nope") == "snell")
    check("corner normalize", normalize_watermark_corner("nope") == "br")

    tl = apply_export_watermark_rgb(canvas, remark="© test", corner="tl")
    br = apply_export_watermark_rgb(canvas, remark="© test", corner="br")
    check("tl marks top-left", not np.array_equal(tl[:50, :80], canvas[:50, :80]))
    check("br marks bottom-right", not np.array_equal(br[-50:, -80:], canvas[-50:, -80:]))
    check("tl leaves bottom-right", np.array_equal(tl[-40:, -60:], canvas[-40:, -60:]))
    check("br leaves top-left", np.array_equal(br[:40, :60], canvas[:40, :60]))
    check("corners differ", not np.array_equal(tl, br))

    kw = watermark_kwargs_from_prefs(
        {
            "watermark_text": "hello",
            "watermark_signature": "Jane Doe",
            "watermark_font": "brush",
            "watermark_corner": "tl",
            "watermark_image": "/tmp/old.png",
        }
    )
    check(
        "prefs both fields",
        kw["watermark_text"] == "hello"
        and kw["watermark_signature"] == "Jane Doe"
        and kw["watermark_font"] == "brush"
        and kw["watermark_corner"] == "tl"
        and "watermark_image" not in kw,
    )
    kw_old = watermark_kwargs_from_prefs(
        {"watermark_name": "Ada", "watermark_text": "© leftover", "watermark_image": "/x.png"}
    )
    check(
        "old name key still works",
        kw_old["watermark_signature"] == "Ada" and kw_old["watermark_text"] == "© leftover",
    )
    kw_img = watermark_kwargs_from_prefs(
        {"watermark_kind": "signature", "watermark_image": "/tmp/old.png"}
    )
    check(
        "old image-only is empty",
        not kw_img["watermark_text"] and not kw_img["watermark_signature"],
    )

    xml = '<rdf:Description exif:GPSLatitude="1" xmp:Label="Red"><exif:GPSLongitude>2</exif:GPSLongitude></rdf:Description>'
    stripped = strip_gps_from_xmp_text(xml)
    check("strip gps attr", "GPSLatitude" not in stripped)
    check("keep label", "Label" in stripped)

    if FAILURES:
        print(f"\n{len(FAILURES)} failed")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
