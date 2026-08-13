#!/usr/bin/env python3
"""Gallery right-click: menu contents, and rating a multi-photo selection.

The gallery had no context menu at all -- the only menu reachable from it was
the share button's, which anchors under that button. These cover the new one:

  1. What it offers, and that Stitch & Merge appears only when more than one
     photo is selected (its whole purpose is multi-select HDR/panorama merges).
  2. Right-clicking a photo outside the current selection makes it the
     selection first, so the menu's targets and the actions' targets agree --
     otherwise Delete would act on something the menu never named.
  3. Rating a selection writes every photo's XMP, and repaints the gallery
     once rather than once per photo.

Built through the unbound methods against a stub viewer: constructing the real
RAWImageViewer needs a window, a folder and a cache, none of which these
behaviours depend on.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "rawviewer_ui"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _action_labels(menu):
    """Every action label in the menu, submenus included."""
    out = []
    for act in menu.actions():
        sub = act.menu()
        if sub is not None:
            out.append(act.text())
            out.extend(_action_labels(sub))
        elif not act.isSeparator():
            out.append(act.text())
    return out


class _StubCache:
    def __init__(self):
        self.exif = {}

    def get_exif(self, path):
        return self.exif.get(path)

    def put_exif(self, path, data):
        self.exif[path] = data


class _StubGallery:
    def __init__(self):
        self._metadata_cache = {}
        self.tile_refreshes = 0
        self.badge_refreshes = 0

    def refresh_visible_tile_for_path(self, path):
        self.tile_refreshes += 1

    def refresh_gallery_rating_visuals(self):
        self.badge_refreshes += 1


def _stub_viewer(selected=None):
    from PyQt6.QtWidgets import QWidget

    import main as main_mod

    v = QWidget()  # a real QWidget: QMenu(self) needs a QWidget parent
    v.__class__ = type("StubViewer", (QWidget,), {})
    v.image_cache = _StubCache()
    v._gallery_bulk_metadata = {}
    v.gallery_justified = _StubGallery()
    v._selected = list(selected or [])
    v.messages = []
    v.epoch_bumps = 0

    v.status_bar = types.SimpleNamespace(
        showMessage=lambda msg, *a, **k: v.messages.append(msg)
    )
    v._filmstrip_bar = lambda: None
    v._gallery_selected_canonical_paths = lambda: list(v._selected)
    v._bump_gallery_rating_presence_epoch = lambda: setattr(
        v, "epoch_bumps", v.epoch_bumps + 1
    )
    v.schedule_save_session_state = lambda: None
    v._open_hdr_panorama_dialog = lambda mode, ps: None
    v._share_paths_os = lambda ps: None
    v._copy_paths_to_clipboard = lambda ps: None
    v._open_paths_in_editing_app = lambda ps: None
    v._open_macos_choose_editing_app = lambda ps: None
    v._open_with_macos_app_path = lambda ps, app: None
    v._open_windows_choose_editing_app = lambda ps: None
    v._open_with_windows_exe_path = lambda ps, exe: None
    v._format_app_name = lambda p: "App"
    v._windows_editing_app_settings_key = lambda: "k"
    v._macos_editing_app_settings_key = lambda: "k"
    v.get_settings = lambda: types.SimpleNamespace(value=lambda *a, **k: "")
    v.delete_gallery_selection = lambda: v.messages.append("DELETE")
    v.discard_gallery_selection = lambda: v.messages.append("DISCARD")
    v._clear_gallery_selection = lambda: v._selected.clear()
    v._gallery_toggle_path_selection = lambda p: v._selected.append(p)

    # The real collaborators of the methods under test, bound to the stub.
    for name in (
        "_populate_share_open_actions",
        "_populate_stitch_actions",
        "_apply_rating_to_path",
        "_refresh_rating_visuals_for_paths",
        "_show_gallery_context_menu",
        "_build_gallery_context_menu",
    ):
        setattr(v, name, types.MethodType(getattr(main_mod.RAWImageViewer, name), v))
    return v, main_mod


def test_menu_offers_the_actions_for_one_photo():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    v, main_mod = _stub_viewer()
    menu = main_mod.RAWImageViewer._build_gallery_context_menu(v, ["/x/a.CR3"])
    labels = " | ".join(_action_labels(menu))

    check("single photo: rating is offered", "Rate" in labels)
    check("single photo: five stars down to clear", "⭐⭐⭐⭐⭐" in labels and "Clear rating" in labels)
    check("single photo: discard is offered", "Move to Discard" in labels, labels)
    check("single photo: delete is offered", "Delete…" in labels)
    check("single photo: sharing is offered", "Shar" in labels or "sharing" in labels, labels)
    check("single photo: no merge submenu", "Stitch & Merge" not in labels)
    check("single photo: the header names the file", "a.CR3" in labels)


def test_menu_offers_merges_for_a_multi_selection():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    v, main_mod = _stub_viewer()
    paths = [f"/x/{i}.CR3" for i in range(4)]
    menu = main_mod.RAWImageViewer._build_gallery_context_menu(v, paths)
    labels = " | ".join(_action_labels(menu))

    check("multi-select: Stitch & Merge appears", "Stitch & Merge" in labels)
    for wanted in ("Standard Panorama", "HDR Merge", "Panorama HDR Merge", "Focus Stack"):
        check(f"multi-select: {wanted} is reachable", wanted in labels)
    check("multi-select: the header states the count", "4 photos selected" in labels, labels)
    check("multi-select: delete names the count", "Delete 4…" in labels)


def test_empty_selection_builds_no_menu():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    v, main_mod = _stub_viewer()
    check("no paths, no menu",
          main_mod.RAWImageViewer._build_gallery_context_menu(v, []) is None)


def test_right_click_outside_the_selection_reselects():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QPoint

    _app = QApplication.instance() or QApplication([])
    v, main_mod = _stub_viewer(selected=["/x/a.CR3", "/x/b.CR3"])
    shown = {}
    v._show_gallery_context_menu = lambda paths, pos: shown.update(paths=list(paths))

    # Inside the selection: the selection is left alone.
    main_mod.RAWImageViewer._on_gallery_context_menu_requested(v, "/x/b.CR3", QPoint(0, 0))
    check("right-click inside the selection keeps it",
          shown["paths"] == ["/x/a.CR3", "/x/b.CR3"], str(shown["paths"]))

    # Outside it: that photo becomes the selection, so the menu and the
    # actions cannot disagree about what Delete would remove.
    main_mod.RAWImageViewer._on_gallery_context_menu_requested(v, "/x/c.CR3", QPoint(0, 0))
    check("right-click outside the selection replaces it",
          shown["paths"] == ["/x/c.CR3"], str(shown["paths"]))


def test_rating_a_selection_writes_every_sidecar():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as root:
        paths = []
        for i in range(3):
            p = os.path.join(root, f"img{i}.jpg")
            with open(p, "wb") as f:
                f.write(b"\xff\xd8" + b"x" * 512)
            paths.append(p)

        v, main_mod = _stub_viewer(selected=paths)
        main_mod.RAWImageViewer.rate_gallery_selection(v, 4)

        from raw_adjustments import resolve_xmp_path

        written = [os.path.isfile(resolve_xmp_path(p) or "") for p in paths]
        check("every selected photo gets an XMP sidecar", all(written), str(written))
        check("the rating is in the EXIF cache for each",
              [v.image_cache.exif[p]["rating"] for p in paths] == [4, 4, 4])
        check("the status message reports the batch",
              any("3 photos" in m for m in v.messages), str(v.messages))

        # The repaint is what makes a big selection affordable: one badge pass,
        # not one per photo.
        check("badges are refreshed once for the whole batch",
              v.gallery_justified.badge_refreshes == 1,
              f"{v.gallery_justified.badge_refreshes} passes")
        check("the rating-presence epoch bumps once", v.epoch_bumps == 1,
              f"{v.epoch_bumps} bumps")


def test_rating_an_empty_selection_says_so():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    v, main_mod = _stub_viewer(selected=[])
    main_mod.RAWImageViewer.rate_gallery_selection(v, 3)
    check("rating nothing reports it rather than failing silently",
          any("No images selected" in m for m in v.messages), str(v.messages))


def main() -> int:
    test_menu_offers_the_actions_for_one_photo()
    test_menu_offers_merges_for_a_multi_selection()
    test_empty_selection_builds_no_menu()
    test_right_click_outside_the_selection_reselects()
    test_rating_a_selection_writes_every_sidecar()
    test_rating_an_empty_selection_says_so()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
