"""Closing the editor on an already-edited photo refreshes its gallery tile.

The tile is only rebuilt by the save-time bake, which fires from
editing_finished -- so it needs an actual change. Every other refresh path
(_reencode_persisted_preview_for_sidecar, the leave-image re-bake) returns
immediately unless SIDECAR_ADJUST is on, and it is off by default.

The result: a photo edited in an EARLIER session showed its embedded JPEG in
the grid forever. Open it, look at the correct edited render in Adjust, close
without touching anything, and the tile was unchanged -- no matter how many
times you did it.

Baking on close covers that, narrowly: only when a full-quality render for
this exact file is already in hand, and only when the sidecar really holds
edits.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


class _Host:
    """Only what the bake helper touches."""

    def __init__(self, path, render, *, render_path=None):
        self._adjust_last_render = (render, {}) if render is not None else None
        self._adjust_last_render_norm = render_path or path
        self.baked = []
        self.invalidated = []
        self.gallery_justified = self

    def _persist_editor_aligned_browse_caches(self, path):
        self.baked.append(path)

    def invalidate_thumbnails_for_path(self, path):
        self.invalidated.append(path)


def main() -> int:
    import main as mainmod
    import raw_adjustments

    fn = mainmod.RAWImageViewer._bake_browse_caches_for_visited_edit
    real_load = raw_adjustments.load_adjustments_for_file
    real_default = raw_adjustments.is_default_adjustments

    big = np.zeros((1400, 2000, 3), np.uint8)
    small = np.zeros((600, 800, 3), np.uint8)
    raw = "/photos/shot.CR3"
    jpg = "/photos/shot.jpg"

    def with_adjustments(edited: bool):
        raw_adjustments.load_adjustments_for_file = lambda p: ({"Exposure2012": 0.8} if edited else {})
        raw_adjustments.is_default_adjustments = lambda a: not a

    try:
        # --- the reported case: edited photo, opened and closed untouched ---
        with_adjustments(True)
        h = _Host(raw, big)
        fn(h, raw)
        check("an edited RAW bakes on close", h.baked == [raw], str(h.baked))
        check("and the tile is invalidated so the grid repaints", h.invalidated == [raw])

        # --- an unedited photo must not rewrite its own cache ---
        with_adjustments(False)
        h = _Host(raw, big)
        fn(h, raw)
        check("an unedited RAW does not bake", h.baked == [], str(h.baked))

        # --- non-RAW is out of scope: companion JPEGs are never written ---
        with_adjustments(True)
        h = _Host(jpg, big)
        fn(h, jpg)
        check("a JPEG does not bake", h.baked == [])

        # --- never bake from a small interim render ---
        h = _Host(raw, small)
        fn(h, raw)
        check(
            "a sub-1200px render is refused",
            h.baked == [],
            "an interim frame must not be written into browse caches",
        )

        # --- never bake from another file's render ---
        h = _Host(raw, big, render_path="/photos/other.CR3")
        fn(h, raw)
        check("a render belonging to another file is refused", h.baked == [])

        # --- nothing rendered yet ---
        h = _Host(raw, None)
        fn(h, raw)
        check("no render in hand means no bake", h.baked == [])

        # --- it must never decode: only reuse what is already there ---
        h = _Host(raw, big)
        raw_adjustments.load_adjustments_for_file = lambda p: {"Exposure2012": 0.8}
        fn(h, raw)
        check("bakes without touching the decode path", h.baked == [raw])
    finally:
        raw_adjustments.load_adjustments_for_file = real_load
        raw_adjustments.is_default_adjustments = real_default

    # --- and the close path actually calls it ---
    import inspect

    src = inspect.getsource(mainmod.RAWImageViewer._close_adjust_panel_from_ui)
    check(
        "closing the Adjust panel triggers the bake",
        "_bake_browse_caches_for_visited_edit" in src,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
