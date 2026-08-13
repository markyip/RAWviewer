#!/usr/bin/env python3
"""Messaging apps need JPEG; RAW/TIFF should be proxied. AirDrop stays original."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    extra = f"  {detail}" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{extra}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from share_payload import needs_jpeg_proxy, write_jpeg_proxy

    check("JPEG is shareable as-is", not needs_jpeg_proxy("shot.jpg"))
    check("PNG is shareable as-is", not needs_jpeg_proxy("shot.png"))
    check("HEIC is shareable as-is", not needs_jpeg_proxy("shot.heic"))
    check("ARW needs a JPEG proxy", needs_jpeg_proxy("DSC01234.ARW"))
    check("CR3 needs a JPEG proxy", needs_jpeg_proxy("img.cr3"))
    check("NEF needs a JPEG proxy", needs_jpeg_proxy("/tmp/a.NEF"))
    check("TIFF needs a JPEG proxy", needs_jpeg_proxy("scan.tif"))
    check("DNG needs a JPEG proxy", needs_jpeg_proxy("frame.dng"))

    fd, src = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fd, dest = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        from PIL import Image

        Image.new("RGB", (32, 24), (12, 34, 56)).save(src, format="PNG")
        ok = write_jpeg_proxy(src, dest)
        check("PNG proxy writes a JPEG", ok and os.path.getsize(dest) > 0)
        with open(dest, "rb") as fh:
            head = fh.read(3)
        check("proxy starts with a JPEG SOI", head[:2] == b"\xff\xd8", head)
    finally:
        for p in (src, dest):
            try:
                os.remove(p)
            except OSError:
                pass

    main_py = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "main.py"
    )
    with open(main_py, encoding="utf-8") as fh:
        src_text = fh.read()
    check(
        "AirDrop is shown in the share menu by default",
        '_env_true("RAWVIEWER_SHARE_SHOW_AIRDROP", default=True)' in src_text,
    )
    check(
        "AirDrop is hosted outside the Qt process",
        "osascript" in src_text and "com.apple.share.AirDrop.send" in src_text,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
