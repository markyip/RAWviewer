#!/usr/bin/env python3
"""Real JPEG files must not be treated as RAW-workflow camera-JPEG interim.

``_suppress_jpeg_interim`` returning True for non-RAW used to also trip the
<1024px paint gates in display_pixmap / display_numpy_image. A genuine small
JPEG (or a session-restore fast-open that cleared the view before full paint)
then stayed blank: the decoded pixels were refused as "interim", and folder
ready skipped reload because was_fast_open was enough on its own.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from common_image_loader import is_raw_file

    with tempfile.TemporaryDirectory() as tmp:
        jpeg = os.path.join(tmp, "launch.jpg")
        open(jpeg, "wb").write(b"\xff\xd8\xff\xd9")
        raw = os.path.join(tmp, "shot.CR3")
        open(raw, "wb").write(b"stub")
        check("is_raw_file rejects JPEG", not is_raw_file(jpeg))
        check("is_raw_file accepts CR3", is_raw_file(raw))

    main_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py"
    )
    text = open(main_src, encoding="utf-8").read()

    start = text.index("def _suppress_jpeg_interim")
    end = text.index("\n    def ", start + 1)
    body = text[start:end]
    # Non-RAW branch must return False (allow paint), not True (suppress).
    m = re.search(
        r"if not is_raw_file\(path\):\s*\n\s*return (True|False)",
        body,
    )
    check(
        "_suppress_jpeg_interim leaves real JPEG alone",
        m is not None and m.group(1) == "False",
        m.group(0).replace("\n", " ") if m else "branch missing",
    )

    for fn in ("def display_pixmap", "def display_numpy_image"):
        s = text.index(fn)
        e = text.index("\n    def ", s + 1)
        gate = text[s:e]
        check(
            f"{fn.split()[-1]} <1024 gate requires is_raw_file",
            "is_raw_file(cur)" in gate and "_suppress_jpeg_interim(cur)" in gate,
        )

    start = text.index("skip_single_reload = bool(")
    block = text[start : start + 900]
    check(
        "skip_single_reload requires pixels or first render (not was_fast_open alone)",
        "was_fast_open" not in block
        and "_single_view_pixels_on_screen" in block
        and "_single_view_first_render_logged" in block,
        block[:280].replace("\n", " "),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
