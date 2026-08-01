"""AI masks are Plus-only; model-free masks work in both editions.

No mask model is bundled in either installer -- they download on first use --
so this gate saves no install size. What it guarantees is that a Standard user
never triggers a 214 MB download by pressing a mask button.

Both directions matter. Gating too little means Standard downloads BiRefNet;
gating too much means Standard silently loses brush and gradients, which need
no model at all.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


# The profile is read at import time and cached by module state, so each
# edition is probed in its own interpreter rather than by mutating os.environ.
_PROBE = r"""
import sys, json
sys.path.insert(0, %r)
from PyQt6.QtWidgets import QApplication
app = QApplication([])
import rawviewer_profile as rp, raw_ai_masks as m
from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget as P
p = P(); p.show(); p._panel_tabs.set_current(1)
def shown(attr):
    b = getattr(p, attr, None)
    return bool(b is not None and b.parent() is not None and b.isVisible())
# The creation tools are the six buttons in the Create new mask section,
# so that is where the edition gate has to hold. The buttons behind it stay as
# the arming state machine and are deliberately unparented.
_items = [b.text() for b in p._mask_create_buttons.values()]
def offered(label):
    return label in _items
# The export menu's AI variants are Plus features too, and used to be gated
# on model presence alone -- which is not an edition check.
_export = [a.text() for a in p._export_btn.menu().actions() if a.text()]
print("@@" + json.dumps({
    "edition": rp.edition_display_name(),
    "enabled": m.ai_masks_enabled(),
    "ops": m.available_ops(),
    "ready_subject": m.op_is_ready("subject"),
    "ensure_subject": m.ensure_op_available("subject"),
    "brush": shown("_mask_paint_btn"),
    "linear": shown("_mask_linear_btn"),
    "radial": shown("_mask_radial_btn"),
    "erase": shown("_mask_erase_btn"),
    "invert": shown("_mask_invert_btn"),
    "subject": offered("Smart Object"),
    "sky": offered("Sky"),
    "click": offered("AI Selection"),
    "brush_item": offered("Brush"),
    "linear_item": offered("Linear Gradient"),
    "radial_item": offered("Radial Gradient"),
    "ai_btns_enabled": bool(p._mask_ai_subject_btn.isEnabled()),
    "export_ai": [x for x in _export if x.startswith("AI ")],
    "export_plain": [x for x in _export if not x.startswith("AI ")],
    "ai_export_gate": rp.ai_export_enabled(),
}))
"""


def probe(profile, extra_env=None):
    env = dict(os.environ)
    env["RAWVIEWER_BUILD_PROFILE"] = profile
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = SRC
    env.pop("RAWVIEWER_FORCE_AI_MASKS", None)
    if extra_env:
        env.update(extra_env)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE % os.path.abspath(SRC)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    for line in out.stdout.splitlines():
        if line.startswith("@@"):
            import json
            return json.loads(line[2:])
    raise AssertionError(f"probe failed for {profile}: {out.stderr[-400:]}")


def main() -> int:
    plus = probe("full")
    std = probe("lite")

    check("full resolves to Plus", plus["edition"] == "Plus", plus["edition"])
    check("lite resolves to Standard", std["edition"] == "Standard", std["edition"])

    # --- Plus keeps everything ---
    check("Plus enables AI masks", plus["enabled"] is True)
    check("Plus offers Smart Object", plus["subject"])
    check("Plus offers Sky", plus["sky"])
    check("Plus offers AI Selection", plus["click"])

    # --- Standard: no AI mask reaches the user, and none can download ---
    check("Standard disables AI masks", std["enabled"] is False)
    check("Standard hides Smart Object", not std["subject"])
    check("Standard hides Sky", not std["sky"])
    check("Standard hides AI Selection", not std["click"])
    check(
        "Standard reports every AI op unavailable",
        std["ops"] == {"subject": False, "sky": False, "click": False},
        str(std["ops"]),
    )
    check("Standard op_is_ready is False", std["ready_subject"] is False)
    check(
        "Standard ensure_op_available refuses (no 214 MB download)",
        std["ensure_subject"] is False,
    )

    check(
        "Standard also disables the buttons behind the menu",
        std["ai_btns_enabled"] is False,
        "omitting the menu entry is not enough on its own -- the arming "
        "buttons are still reachable from code paths like set_ai_tool_used",
    )

    # --- Standard keeps every mask that needs no model ---
    for name in ("brush_item", "linear_item", "radial_item"):
        check(f"Standard offers {name}", std[name], "model-free mask must survive")
    for name in ("brush", "erase", "invert"):
        check(f"Standard keeps {name}", std[name], "model-free mask must survive")

    # --- AI export variants follow the edition, not what is on disk ---
    check(
        "Plus offers the AI export variants",
        plus["export_ai"] == [
            "AI Denoise (SCUNet)",
            "AI Upscale 2\u00d7 (Real-ESRGAN)",
            "AI Denoise + Upscale 2\u00d7",
        ],
        str(plus["export_ai"]),
    )
    check("Plus enables the export gate", plus["ai_export_gate"] is True)
    check(
        "Standard offers none of them",
        std["export_ai"] == [],
        f"{std['export_ai']} -- these models are on this machine, so gating "
        "on presence alone showed Plus features in Standard, and neither "
        "onnx engine has an edition check to stop the export running",
    )
    check("Standard disables the export gate", std["ai_export_gate"] is False)
    check(
        "and Standard keeps every ordinary format",
        std["export_plain"] == plus["export_plain"] and std["export_plain"],
        str(std["export_plain"]),
    )

    # --- the override, for dev checkouts ---
    forced = probe("lite", {"RAWVIEWER_FORCE_AI_MASKS": "1"})
    check(
        "RAWVIEWER_FORCE_AI_MASKS re-enables them in a lite checkout",
        forced["enabled"] is True and forced["subject"],
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
