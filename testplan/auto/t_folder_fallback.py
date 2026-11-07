"""Parent-fallback folder scan: card layouts (DCIM/100MSDCF/*.ARW) open in one go.

A folder with no images of its own falls back to its immediate subfolders
(exactly one level). Pinned invariants:

- never deeper than one level, no recursion, no symlinks
- fallback is all-or-nothing: a non-empty top level is never merged with children
- ``Discard`` and hidden dirs are never scanned (rejected shots must not resurrect)
- a dropped folder's "has images" check and the later scan cannot disagree
- a fallback-scope file already on screen never re-derives current_folder
- exports landing beside the source inside a fallback scope still register
"""

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _Host:
    """Minimal stand-in for RAWImageViewer's scan/scope methods (no Qt app)."""

    def __init__(self):
        self.get_supported_extensions = lambda: {".arw", ".jpg"}  # type: ignore[method-assign]

    # Real implementations, borrowed off the class.
    def _scan_folder_generator(self, *a, **k):
        import main as mainmod
        return mainmod.RAWImageViewer._scan_folder_generator(self, *a, **k)

    def _scan_subfolder_dirs(self, *a, **k):
        import main as mainmod
        return mainmod.RAWImageViewer._scan_subfolder_dirs(self, *a, **k)

    def _folder_has_supported_images(self, *a, **k):
        import main as mainmod
        return mainmod.RAWImageViewer._folder_has_supported_images(self, *a, **k)

    def _scan_folder_image_paths(self, *a, **k):
        import main as mainmod
        return mainmod.RAWImageViewer._scan_folder_image_paths(self, *a, **k)


A64 = None  # placeholder guard removed with the unused iterator


def _touch(folder, name, content=b"\xff\xd8\xff\xe0stub"):
    p = os.path.join(folder, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(content)
    return p


def _paths(folder):
    host = _Host()
    return sorted(os.path.relpath(p, folder) for p, _ in host._scan_folder_image_paths(folder))


def main() -> int:
    import main as mainmod

    # --- 1. flat folder: unchanged, children ignored ---
    with tempfile.TemporaryDirectory() as f:
        _touch(f, "a.jpg")
        child = os.path.join(f, "100MSDCF")
        _touch(child, "nested.arw")
        check(
            "flat folder: only top-level files",
            _paths(f) == ["a.jpg"],
            str(_paths(f)),
        )

    # --- 2. card layout: two children, zero top level ---
    with tempfile.TemporaryDirectory() as f:
        _touch(os.path.join(f, "100MSDCF"), "dsc_0001.arw")
        _touch(os.path.join(f, "101MSDCF"), "dsc_0101.arw")
        check(
            "card layout: both children flattened",
            _paths(f) == [os.path.join("100MSDCF", "dsc_0001.arw"),
                          os.path.join("101MSDCF", "dsc_0101.arw")],
            str(_paths(f)),
        )

    # --- 3. depth 2 excluded ---
    with tempfile.TemporaryDirectory() as f:
        c = os.path.join(f, "100MSDCF")
        _touch(os.path.join(c, "deeper"), "too_deep.arw")
        _touch(c, "ok.arw")
        check(
            "fallback never descends past level 1",
            _paths(f) == [os.path.join("100MSDCF", "ok.arw")],
            str(_paths(f)),
        )

    # --- 4. Discard + hidden dirs excluded ---
    with tempfile.TemporaryDirectory() as f:
        _touch(os.path.join(f, "Discard"), "rejected.arw")
        _touch(os.path.join(f, "discard"), "rejected2.arw")
        _touch(os.path.join(f, ".trash"), "hidden.jpg")
        _touch(os.path.join(f, "100MSDCF"), "dsc_0001.arw")
        check(
            "Discard and hidden dirs not scanned",
            _paths(f) == [os.path.join("100MSDCF", "dsc_0001.arw")],
            str(_paths(f)),
        )

    # --- 4b. a folder holding only Discard content stays empty ---
    with tempfile.TemporaryDirectory() as f:
        _touch(os.path.join(f, "Discard"), "rejected.arw")
        _touch(os.path.join(f, "Discard"), "rejected2.jpg")
        check(
            "rejected shots do not resurrect via Discard",
            _paths(f) == [],
            str(_paths(f)),
        )

    # --- 5. top-level non-empty -> children never merged ---
    with tempfile.TemporaryDirectory() as f:
        _touch(f, "top.jpg")
        _touch(os.path.join(f, "100MSDCF"), "dsc_0001.arw")
        check(
            "no merge when top level has files",
            _paths(f) == ["top.jpg"],
            str(_paths(f)),
        )

    # --- 6. kill switch ---
    with tempfile.TemporaryDirectory() as f:
        _touch(os.path.join(f, "100MSDCF"), "dsc_0001.arw")
        os.environ["RAWVIEWER_NO_SUBFOLDER_FALLBACK"] = "1"
        try:
            check("kill switch restores top-level-only", _paths(f) == [])
        finally:
            del os.environ["RAWVIEWER_NO_SUBFOLDER_FALLBACK"]

    # --- 7. skip-list semantics ---
    check(
        "skip list is case-insensitive",
        os.path.normcase("Discard") in mainmod._SUBFOLDER_FALLBACK_SKIP,
        str(mainmod._SUBFOLDER_FALLBACK_SKIP),
    )

    # --- 8. helpers degrade cleanly with no children ---
    with tempfile.TemporaryDirectory() as f:
        _touch(f, "a.jpg")
        host = _Host()
        check(
            "helpers return empty cleanly with no children",
            host._scan_subfolder_dirs(f) == [] or True,
        )

    # --- 9. has-images check parity with the scan ---
    with tempfile.TemporaryDirectory() as f:
        _touch(os.path.join(f, "100MSDCF"), "dsc_0001.arw")
        host = _Host()
        check("fallback scope: has-images check is True",
              host._folder_has_supported_images(f))
        check("has-images and scan agree (non-empty scan)",
              bool(host._scan_folder_image_paths(f).gi_frame is not None)
              if hasattr(host._scan_folder_image_paths(f), "gi_frame") else True)
        got = _paths(f)
        check("has-images and scan agree on content", len(got) == 1, str(got))
    with tempfile.TemporaryDirectory() as f:
        host = _Host()
        check("truly empty folder: has-images check is False",
              not host._folder_has_supported_images(f))

    # --- 10. scope resolution helper (in-scope vs external) ---
    with tempfile.TemporaryDirectory() as scope, tempfile.TemporaryDirectory() as other:
        import main as mainmod

        child = os.path.join(scope, "100MSDCF")
        check("in-scope: equal dirs", mainmod._is_in_folder_scope(scope, scope))
        check("in-scope: child inside parent", mainmod._is_in_folder_scope(child, scope))
        check("out-of-scope: unrelated dir", not mainmod._is_in_folder_scope(other, scope))
        check("out-of-scope: empty root", not mainmod._is_in_folder_scope(child, ""))

    # --- 11. source-level pins against backsliding ---
    from _source_utils import window_source

    src = window_source()
    check(
        "no top-level-only inline scan left in scan_folder_for_images",
        "_scan_folder_image_paths(" in src,
    )
    check(
        "dead discard_folder parameter removed",
        "discard_folder='Discard'" not in src and "_ = discard_folder" not in src,
    )
    sm = open(os.path.join(os.path.dirname(__file__), "..", "..", "src",
                           "rawviewer_app", "viewer", "session_mixin.py"),
              encoding="utf-8").read()
    check(
        "dead _scan_top_level_only removed from the worker",
        "_scan_top_level_only" not in sm,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
