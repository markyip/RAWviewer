#!/usr/bin/env python3
"""Regression guard: clicking the search button must actually animate the
search bar open/close, not snap instantly.

Root cause (two-part): (1) _on_search_bottom_clicked -- the only real
trigger for the search bar -- called _set_search_panel_expanded(...,
animate=False) in both branches, even though the full QPropertyAnimation
infrastructure exists and works; the animation code was simply never
wired to the one place a user actually triggers it. (2) even after fixing
that, two secondary calls (_set_gallery_search_input_visible's
animate=False default, and a redundant _ensure_gallery_search_collapsed()
call on the collapse path) immediately re-snapped the container to its
target width in the same call stack, cancelling the just-started
animation before a single frame could render -- which is also what
produced the reported "flash to the right" of everything after the
search bar in the row (an instant width jump in one frame instead of a
smooth slide).
"""
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import main as mainmod

    src = inspect.getsource(mainmod.RAWImageViewer._on_search_bottom_clicked)

    check(
        "collapse path animates the panel closed",
        "_set_search_panel_expanded(False, animate=True)" in src,
    )
    check(
        "expand path animates the panel open",
        "_set_search_panel_expanded(True, animate=True)" in src,
    )
    check(
        "expand path does not re-snap the width via an unanimated input-visible call",
        "_set_gallery_search_input_visible()" not in src,
    )
    check(
        "expand path does not restart animation via input-visible(animate=True)",
        "_set_gallery_search_input_visible(animate=True)" not in src,
    )
    check(
        "collapse path no longer double-calls the unanimated collapse helper",
        not re.search(
            r"_set_search_panel_expanded\(False, animate=True\)\s*\n\s*self\._ensure_gallery_search_collapsed\(\)",
            src,
        ),
    )

    set_src = inspect.getsource(mainmod.RAWImageViewer._set_search_panel_expanded)
    check(
        "collapse defers idle width reset until animation finish",
        "_reset_gallery_search_to_idle()" in set_src
        and "def _finish" in set_src
        and set_src.index("def _finish") < set_src.rindex("_reset_gallery_search_to_idle()"),
    )
    sync_src = inspect.getsource(mainmod.RAWImageViewer._sync_search_expand_overlay)
    check(
        "overlay sync skips while width animation is running",
        "_search_panel_animation_running" in sync_src,
    )
    place_src = inspect.getsource(mainmod.RAWImageViewer._place_search_expand_overlay)
    check(
        "search strip is positioned as a floating overlay (setGeometry)",
        "setGeometry" in place_src,
    )
    check(
        "expand animates overlay width via QVariantAnimation",
        "QVariantAnimation" in set_src and "_place_search_expand_overlay" in set_src,
    )

    pin_src = inspect.getsource(mainmod.RAWImageViewer._apply_gallery_search_input_width)
    check(
        "input width pin skips while expand animation is running",
        "_search_panel_animation_running" in pin_src,
    )
    apply_src = inspect.getsource(
        mainmod.RAWImageViewer._apply_search_expand_container_width
    )
    check(
        "in-flight expand extends endValue instead of restarting",
        "setEndValue" in apply_src,
    )
    status_src = inspect.getsource(mainmod.RAWImageViewer._set_gallery_search_status)
    check(
        "status updates defer input pinning during animation",
        "not animating" in status_src,
    )
    click_src = inspect.getsource(mainmod.RAWImageViewer._on_search_bottom_clicked)
    check(
        "expand defers indexing until animation finish",
        "_pending_search_index_corpus" in click_src
        and "_start_user_semantic_indexing" not in click_src,
    )
    finish_helper = inspect.getsource(
        mainmod.RAWImageViewer._run_pending_search_index_after_expand
    )
    check(
        "pending index runner starts semantic indexing",
        "_start_user_semantic_indexing" in finish_helper,
    )
    place_src2 = inspect.getsource(mainmod.RAWImageViewer._place_search_expand_overlay)
    check(
        "overlay uses mask clip-reveal during width slide",
        "setMask" in place_src2 and "_search_panel_anim_content_width" in place_src2,
    )
    set_src2 = inspect.getsource(mainmod.RAWImageViewer._set_search_panel_expanded)
    check(
        "collapse defers child width release until animation finish",
        set_src2.index("def _finish") < set_src2.rindex("_release_gallery_search_panel_width()"),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
