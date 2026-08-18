#!/usr/bin/env python3
"""Per-photo AI descriptions, group tickets, and mcp_edit vs generate modes."""
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


def _tiny_jpeg(path: str) -> None:
    from PIL import Image
    import numpy as np

    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8), mode="RGB").save(
        path, format="JPEG"
    )


def main() -> int:
    from rawviewer_share_agent import (
        MODE_GENERATE,
        MODE_MCP_EDIT,
        SETTINGS_TEMPLATE_KEY,
        SharePathError,
        ai_sidecar_path,
        build_batch_ticket,
        build_share_tickets,
        build_ticket,
        instructions_for_mode,
        load_photo_description,
        persist_share_descriptions,
        render_prompt_markdown,
        save_description_template_to_store,
        save_photo_description,
        suggested_derivative_path,
    )

    tmp = tempfile.mkdtemp(prefix="rawviewer_share_agent_")
    folder = os.path.join(tmp, "photos")
    os.makedirs(folder)
    a = os.path.join(folder, "a.ARW")
    b = os.path.join(folder, "b.ARW")
    with open(a, "wb") as fh:
        fh.write(b"stub-raw-a")
    with open(b, "wb") as fh:
        fh.write(b"stub-raw-b")
    prev_a = os.path.join(tmp, "preview_a.jpg")
    prev_b = os.path.join(tmp, "preview_b.jpg")
    _tiny_jpeg(prev_a)
    _tiny_jpeg(prev_b)

    save_photo_description(a, "keep the neon signs")
    save_photo_description(b, "wedding set — stay warm, don’t oversharpen")
    check("sidecar next to file", os.path.isfile(ai_sidecar_path(a)))
    check(
        "load a",
        load_photo_description(a) == "keep the neon signs",
        load_photo_description(a),
    )
    check(
        "two files differ",
        load_photo_description(a) != load_photo_description(b),
    )

    ticket_a = build_ticket(
        source_path=a,
        preview_path=prev_a,
        folder_root=folder,
        description="keep the neon signs",
        mode=MODE_MCP_EDIT,
    )
    check("ticket has mode mcp_edit", ticket_a.get("mode") == MODE_MCP_EDIT)
    check(
        "ticket description is per-image",
        ticket_a.get("description") == "keep the neon signs",
    )
    check(
        "ticket not using global template as description",
        "style_brief" not in ticket_a or ticket_a.get("description"),
    )
    check(
        "mcp instructions mention MCP",
        "MCP" in ticket_a["instructions"] and "generate pixels" in ticket_a["instructions"].lower()
        or "set_adjustments" in ticket_a["instructions"],
    )

    ticket_gen = build_ticket(
        source_path=a,
        preview_path=prev_a,
        folder_root=folder,
        description="keep the neon signs",
        mode=MODE_GENERATE,
    )
    check("ticket has mode generate", ticket_gen.get("mode") == MODE_GENERATE)
    mcp_i = instructions_for_mode(MODE_MCP_EDIT)
    gen_i = instructions_for_mode(MODE_GENERATE)
    check("mode instructions differ", mcp_i != gen_i)
    check("generate instructions mention derivative", "_gen.jpg" in gen_i or "save_derivative" in gen_i)
    check("generate ticket instructions differ from mcp", ticket_gen["instructions"] != ticket_a["instructions"])
    check(
        "derivative hint",
        ticket_gen.get("derivative_path") == suggested_derivative_path(ticket_gen["source_path"]),
        str(ticket_gen.get("derivative_path")),
    )

    store = {}
    save_description_template_to_store(store, "muted film, never crush blacks")
    check("template key", SETTINGS_TEMPLATE_KEY in store)
    ticket_no_template = build_ticket(
        source_path=a,
        preview_path=prev_a,
        folder_root=folder,
        description="keep neon",
        mode=MODE_MCP_EDIT,
    )
    check(
        "template is not auto-applied",
        ticket_no_template["description"] == "keep neon",
        ticket_no_template["description"],
    )

    group = build_share_tickets(
        [(a, prev_a), (b, prev_b)],
        description="this wedding set: stay warm, don’t oversharpen",
        folder_root=folder,
        mode=MODE_MCP_EDIT,
    )
    check("group has two tickets", len(group) == 2)
    check(
        "group text on ticket 0",
        group[0]["description"] == "this wedding set: stay warm, don’t oversharpen",
    )
    check(
        "group text on ticket 1",
        group[1]["description"] == group[0]["description"],
    )
    check("group modes match", group[0]["mode"] == group[1]["mode"] == MODE_MCP_EDIT)
    batch = build_batch_ticket(group)
    check("batch mode", batch.get("mode") == MODE_MCP_EDIT)
    check("batch description", batch.get("description") == group[0]["description"])
    md = render_prompt_markdown(batch)
    check("prompt has group text", "stay warm" in md)
    check("prompt has mode", "mcp_edit" in md)

    persist_share_descriptions([a, b], "share-only text", save_on_photos=False)
    check(
        "share-only does not write sidecar",
        load_photo_description(a) == "keep the neon signs",
    )
    persist_share_descriptions([a, b], "saved group grade", save_on_photos=True)
    check("save writes a", load_photo_description(a) == "saved group grade")
    check("save writes b", load_photo_description(b) == "saved group grade")

    outside = os.path.join(tmp, "outside.ARW")
    with open(outside, "wb") as fh:
        fh.write(b"nope")
    try:
        build_ticket(
            source_path=outside,
            preview_path=prev_a,
            folder_root=folder,
            description="x",
        )
        check("reject source outside folder", False)
    except SharePathError:
        check("reject source outside folder", True)

    try:
        build_ticket(
            source_path=a,
            preview_path=a,
            folder_root=folder,
            description="x",
        )
        check("reject RAW as preview", False)
    except SharePathError:
        check("reject RAW as preview", True)

    main_py = os.path.join(os.path.dirname(__file__), "..", "..", "src", "main.py")
    with open(main_py, encoding="utf-8") as fh:
        src = fh.read()
    check("share menu has AI agent", 'QAction("Share to AI agent"' in src)
    check("share calls mode", "dlg.mode()" in src or "share_mode = dlg.mode()" in src)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
