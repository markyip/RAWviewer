#!/usr/bin/env python3
"""Headless MCP tools: path sandbox, slider clamp, catalog XMP, mocked preview."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _tiny_jpeg(path: str) -> None:
    from PIL import Image

    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8), mode="RGB").save(path, format="JPEG")


def fake_decode(path, *, use_full_resolution=False, apply_lens_correction=False):
    h, w = (48, 64) if not use_full_resolution else (96, 128)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 0.4
    rgb[..., 1] = 0.35
    rgb[..., 2] = 0.25
    return rgb


def main() -> int:
    from raw_adjustments import (
        DEFAULT_ADJUSTMENTS,
        load_keywords_for_file,
        load_label_for_file,
        load_rating_for_file,
        write_keywords_for_file,
        write_label_for_file,
        write_rating_for_file,
    )
    from rawviewer_mcp.adjustments import clamp_adjustment_value, clamp_patch
    from rawviewer_mcp.paths import PathSandboxError, list_photo_files, resolve_root, resolve_under_root
    from rawviewer_mcp.preview import render_preview_jpeg
    from rawviewer_mcp.service import RawviewerMcpService

    tmp = tempfile.mkdtemp(prefix="rawviewer_mcp_")
    inside = os.path.join(tmp, "photos")
    os.makedirs(inside)
    jpeg_path = os.path.join(inside, "a.jpg")
    _tiny_jpeg(jpeg_path)
    other = os.path.join(inside, "b.CR3")
    with open(other, "wb") as f:
        f.write(b"stub-raw")

    root = resolve_root(inside)
    check("resolve_root", os.path.isdir(root))

    try:
        resolve_under_root(root, os.path.join(tmp, "outside.jpg"))
        check("reject absolute outside", False)
    except PathSandboxError:
        check("reject absolute outside", True)

    try:
        resolve_under_root(root, "../../etc/passwd")
        check("reject traversal", False)
    except PathSandboxError:
        check("reject traversal", True)

    rel = resolve_under_root(root, "a.jpg", must_exist=True)
    check("accept relative inside", os.path.isfile(rel))

    listed = list_photo_files(root)
    names = {os.path.basename(p) for p in listed}
    check("list includes jpeg", "a.jpg" in names)
    check("list includes stub raw", "b.CR3" in names)
    globbed = list_photo_files(root, pattern="*.jpg")
    check("glob jpeg only", globbed and all(p.lower().endswith(".jpg") for p in globbed))
    limited = list_photo_files(root, limit=1)
    check("limit 1", len(limited) == 1)

    exp = clamp_adjustment_value("Exposure2012", 99.0)
    check("clamp exposure high", abs(exp - 5.0) < 1e-6, f"got {exp}")
    temp = clamp_adjustment_value("Temperature", 100.0)
    check("clamp temperature low", abs(temp - 2000.0) < 1e-6, f"got {temp}")
    tint = clamp_adjustment_value("Tint", -1000)
    check("clamp tint low", abs(tint - (-150.0)) < 1e-6, f"got {tint}")
    try:
        clamp_patch({"not_a_slider": 1})
        check("reject unknown key", False)
    except ValueError:
        check("reject unknown key", True)
    try:
        clamp_patch({"DodgeBurnMask": "x"})
        check("reject mask key", False)
    except ValueError:
        check("reject mask key", True)

    rgb = np.clip(np.linspace(0, 1, 32 * 40 * 3, dtype=np.float32).reshape(32, 40, 3), 0, 1)
    jpeg, w, h = render_preview_jpeg(rgb, dict(DEFAULT_ADJUSTMENTS), long_edge=20)
    check("preview jpeg magic", jpeg[:2] == b"\xff\xd8")
    check("preview downscale", max(w, h) <= 20, f"{w}x{h}")
    check("preview positive size", w >= 1 and h >= 1)

    svc = RawviewerMcpService(root, decode_fn=fake_decode)
    listing = svc.list_photos()
    check("service list count", listing["count"] >= 2)

    write_rating_for_file(jpeg_path, 3)
    write_label_for_file(jpeg_path, "Red")
    write_keywords_for_file(jpeg_path, ["mcp", "test"])
    info = svc.get_photo("a.jpg")
    check("get rating", info["rating"] == 3)
    check("get label", info["label"] == "Red")
    check("get keywords", "mcp" in info["keywords"])
    check("get adjustments dict", "Exposure2012" in info["adjustments"])

    patched = svc.set_adjustments("a.jpg", {"Exposure2012": 0.4, "Contrast2012": 12})
    check(
        "set exposure",
        abs(float(patched["adjustments"]["Exposure2012"]) - 0.4) < 1e-4,
        str(patched["adjustments"].get("Exposure2012")),
    )
    check(
        "set contrast",
        abs(float(patched["adjustments"]["Contrast2012"]) - 12.0) < 1e-4,
    )
    over = svc.set_adjustments("a.jpg", {"Exposure2012": 99})
    check(
        "set clamps exposure",
        abs(float(over["adjustments"]["Exposure2012"]) - 5.0) < 1e-4,
        str(over["adjustments"].get("Exposure2012")),
    )

    prev = svc.get_preview("a.jpg", long_edge=32)
    check("preview bytes", isinstance(prev["jpeg_bytes"], (bytes, bytearray)) and prev["jpeg_bytes"][:2] == b"\xff\xd8")
    check("preview meta size", prev["width"] >= 1 and prev["height"] >= 1)

    reset = svc.reset_adjustments("a.jpg")
    check(
        "reset exposure default",
        abs(float(reset["adjustments"]["Exposure2012"]) - 0.0) < 1e-4,
    )
    check("reset keeps rating", load_rating_for_file(jpeg_path) == 3)
    check("reset keeps label", load_label_for_file(jpeg_path) == "Red")
    check("reset keeps keywords", "mcp" in load_keywords_for_file(jpeg_path))

    cat = svc.set_catalog("a.jpg", rating=5, label="Blue", keywords=["alpha"])
    check("catalog rating", cat["rating"] == 5)
    check("catalog label", cat["label"] == "Blue")
    check("catalog keywords", cat["keywords"] == ["alpha"])

    try:
        svc.get_photo("../outside.jpg")
        check("service rejects traversal", False)
    except PathSandboxError:
        check("service rejects traversal", True)

    try:
        svc.export_jpeg("a.jpg", os.path.join(tmp, "leak.jpg"))
        check("export rejects outside dest", False)
    except PathSandboxError:
        check("export rejects outside dest", True)

    auto = svc.auto_adjust("a.jpg")
    check("auto returns adjustments", "adjustments" in auto)
    check("auto skipped or applied", isinstance(auto.get("applied"), list))

    import raw_xmp_presets

    def _fake_preset(name: str):
        return {**DEFAULT_ADJUSTMENTS, "Exposure2012": 0.25}

    raw_xmp_presets.load_preset_adjustments = _fake_preset  # type: ignore
    applied = svc.apply_preset("a.jpg", "look.xmp")
    check(
        "apply preset exposure",
        abs(float(applied["adjustments"]["Exposure2012"]) - 0.25) < 1e-4,
        str(applied["adjustments"].get("Exposure2012")),
    )

    from rawviewer_mcp.server import tool_call_result

    call = tool_call_result(svc, "get_preview", {"path": "a.jpg", "long_edge": 24})
    types = [c.get("type") for c in call.get("content") or []]
    check("mcp preview image content", "image" in types)
    check("mcp preview not error", call.get("isError") is False)

    err = tool_call_result(svc, "get_photo", {"path": "../../etc/passwd"})
    check("mcp sandbox isError", err.get("isError") is True)

    import base64

    jpeg_b64 = base64.b64encode(open(jpeg_path, "rb").read()).decode("ascii")
    saved = svc.save_derivative("a_gen.jpg", jpeg_b64, source="a.jpg")
    check("save_derivative writes under root", os.path.isfile(os.path.join(inside, "a_gen.jpg")))
    check("save_derivative dest relpath", saved.get("dest") == "a_gen.jpg")

    try:
        svc.save_derivative(os.path.join(tmp, "leak.jpg"), jpeg_b64, source="a.jpg")
        check("save_derivative rejects outside dest", False)
    except PathSandboxError:
        check("save_derivative rejects outside dest", True)

    try:
        svc.save_derivative("a.ARW", jpeg_b64, source="a.jpg")
        check("save_derivative rejects non-jpeg dest", False)
    except Exception:
        check("save_derivative rejects non-jpeg dest", True)

    try:
        svc.save_derivative("a.jpg", jpeg_b64, source="a.jpg")
        check("save_derivative refuses overwrite source", False)
    except Exception:
        check("save_derivative refuses overwrite source", True)

    if FAILURES:
        print(f"\n{len(FAILURES)} failed")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
