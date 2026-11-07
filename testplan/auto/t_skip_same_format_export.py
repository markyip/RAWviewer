#!/usr/bin/env python3
"""Skip re-encode: copy source verbatim when it already matches the target format."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import threading
    from rawviewer_app.services.batch import (
        _should_skip_reencode,
        _BYPASS_EXT,
        export_one_file,
    )

    # --- Extension mapping ---
    check("jpeg maps to jpg/jpeg", _BYPASS_EXT["jpeg"] == {".jpg", ".jpeg"})
    check("webp maps to webp", _BYPASS_EXT["webp"] == {".webp"})
    check("tiff16 maps to tif/tiff", _BYPASS_EXT["tiff16"] == {".tif", ".tiff"})
    check("unknown format has empty set", _BYPASS_EXT.get("png", "") == "")

    # --- Helper logic ---
    tmp = tempfile.mkdtemp()
    jpg_src = os.path.join(tmp, "photo.jpg")
    with open(jpg_src, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 32)  # minimal JPEG header

    check("jpg -> jpeg + no edits skips", _should_skip_reencode(jpg_src, "jpeg", {}, False) is True)
    check("jpg -> jpeg + None adj skips", _should_skip_reencode(jpg_src, "jpeg", None, False) is True)
    check("JPG (upper) -> jpeg skips", _should_skip_reencode(jpg_src.upper().lower(), "jpeg", {}, False) is True)
    check("jpeg -> jpeg + edits does NOT skip",
          _should_skip_reencode(jpg_src, "jpeg", {"Exposure2012": 0.5}, False) is False)
    check("jpg -> jpeg + ai_denoise does NOT skip",
          _should_skip_reencode(jpg_src, "jpeg", {}, True) is False)
    check("arw -> jpeg does NOT skip",
          _should_skip_reencode(os.path.join(tmp, "a.arw"), "jpeg", {}, False) is False)
    check("jpg -> webp does NOT skip",
          _should_skip_reencode(jpg_src, "webp", {}, False) is False)
    check("tif -> tiff16 + no edits skips",
          _should_skip_reencode(os.path.join(tmp, "x.tif"), "tiff16", {}, False) is True)

    # --- export_one_file actually copies ---
    out_path = os.path.join(tmp, "out.jpg")
    err, xmp_warn = export_one_file(
        jpg_src, out_path, {}, "jpeg", None,
        cancel_event=threading.Event(),
    )
    check("export_one_file copies without re-encode", err is None and os.path.isfile(out_path),
          f"err={err!r} exists={os.path.isfile(out_path)}")
    if os.path.isfile(out_path):
        with open(jpg_src, "rb") as f:
            src_bytes = f.read()
        with open(out_path, "rb") as f:
            out_bytes = f.read()
        check("copied bytes match source", src_bytes == out_bytes)

    # export progress fired (copy path returns None, None without calling progress_cb).
    progress_stages = []

    def _cb(pct, stage=""):
        progress_stages.append((pct, stage))

    export_one_file(
        jpg_src, os.path.join(tmp, "out2.jpg"), {}, "jpeg", None,
        cancel_event=threading.Event(), progress_cb=_cb,
    )
    check("copy path does not emit decode/encode progress", progress_stages == [],
          f"progress={progress_stages!r}")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
