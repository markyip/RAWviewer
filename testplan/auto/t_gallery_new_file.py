"""A file this app writes appears in the open folder without a reopen.

Exports, HDR merges, panoramas and focus stacks all land beside the
originals, and nothing told the gallery: there is no filesystem watcher
anywhere in the app, and image_files is only ever built by a folder scan.

Also pins the bug found while wiring this: _on_stitch_finished called
self._open_file(), which is defined nowhere, so every successful merge
raised AttributeError immediately after writing the file.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _Gallery:
    def __init__(self):
        self.received = None

    def set_images(self, images, force_rebuild=False):
        self.received = list(images)


class _Host:
    """Minimal stand-in carrying only what register_new_file touches."""

    def __init__(self, folder, files):
        self.current_folder = folder
        self.image_files = list(files)
        self.current_file_path = files[0] if files else None
        self.current_file_index = 0
        self.gallery_justified = _Gallery()
        self.sort_newest_first = False
        self.opened = []

    # Real helpers, borrowed off the class: the point is to exercise them,
    # not to reimplement the sort placement in the test.
    def _sorted_insert_index(self, files, path):
        import main as mainmod

        return mainmod.RAWImageViewer._sorted_insert_index(self, files, path)

    def _sync_filmstrip_files(self):
        import main as mainmod

        return mainmod.RAWImageViewer._sync_filmstrip_files(self)

    def _filmstrip_bar(self):
        return None

    def _show_created_file(self, path):
        self.opened.append(path)


def _touch(folder, name, mtime):
    p = os.path.join(folder, name)
    with open(p, "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0stub")
    os.utime(p, (mtime, mtime))
    return p


def main() -> int:
    import main as mainmod

    reg = mainmod.RAWImageViewer.register_new_file

    with tempfile.TemporaryDirectory() as folder:
        t0 = time.time() - 1000
        a = _touch(folder, "a.jpg", t0)
        b = _touch(folder, "b.jpg", t0 + 100)
        c = _touch(folder, "c.jpg", t0 + 300)
        host = _Host(folder, [a, b, c])

        # --- a new export lands mid-sequence by mtime ---
        new = _touch(folder, "edit.jpg", t0 + 200)
        added = reg(host, new)
        check("new file is added", added is True)
        check(
            "inserted in sort order, not appended",
            [os.path.basename(p) for p in host.image_files]
            == ["a.jpg", "b.jpg", "edit.jpg", "c.jpg"],
            str([os.path.basename(p) for p in host.image_files]),
        )
        check("gallery was handed the new list", host.gallery_justified.received is not None)
        check(
            "gallery list matches image_files",
            host.gallery_justified.received == host.image_files,
        )
        check("current file index still points at the same photo",
              host.image_files[host.current_file_index] == a)

        # --- idempotent ---
        before = len(host.image_files)
        check("adding the same file twice is a no-op", reg(host, new) is False)
        check("list unchanged", len(host.image_files) == before)

        # --- a file written elsewhere must not be injected ---
        with tempfile.TemporaryDirectory() as other:
            elsewhere = _touch(other, "far.jpg", t0 + 50)
            check("file in another folder is ignored", reg(host, elsewhere) is False)
            check("list still unchanged", len(host.image_files) == before)

        # --- missing file ---
        check("nonexistent path is refused", reg(host, os.path.join(folder, "nope.jpg")) is False)

        # --- select=True opens it ---
        sel = _touch(folder, "merged.tif", t0 + 400)
        reg(host, sel, select=True)
        check("select=True opens the created file", host.opened == [sel], str(host.opened))

    # --- the method the merge path used must actually exist now ---
    check(
        "_open_file is gone (it never existed)",
        not hasattr(mainmod.RAWImageViewer, "_open_file"),
    )
    check(
        "_show_created_file exists to replace it",
        hasattr(mainmod.RAWImageViewer, "_show_created_file"),
    )
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "src", "main.py")).read()
    check(
        "nothing calls the missing _open_file any more",
        "self._open_file(" not in src,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
