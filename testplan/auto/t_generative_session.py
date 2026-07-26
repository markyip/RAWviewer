"""Staging model for generative results (generative_session).

The point of staging is that a rejected experiment costs the user nothing:
nothing reaches their folder until Export. These tests pin that, plus the
chaining rule that makes "generate again" continue from the newest version
rather than restarting from the original.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from generative_session import GenerativeSession  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _img(value):
    return np.full((16, 24, 3), value, dtype=np.uint8)


def main() -> int:
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "020A1358.CR3")
        open(source, "wb").close()
        before = set(os.listdir(folder))

        session = GenerativeSession()
        try:
            check("no chain before generating", session.chain(source) == [])
            check("no latest before generating", session.latest(source) is None)
            check("nothing unexported yet", not session.has_unexported_work())

            # --- staging keeps the user's folder untouched ---
            s1 = session.stage(source, _img(40), "remove the bin")
            check("staged file exists", os.path.isfile(s1.path))
            check(
                "staging did not touch the source folder",
                set(os.listdir(folder)) == before,
                f"{sorted(set(os.listdir(folder)) - before)}",
            )
            check("staged outside the source folder", not s1.path.startswith(folder))
            check("version numbering starts at 1", s1.index == 1 and s1.label == "Version 1")
            check("instruction recorded", s1.instruction == "remove the bin")
            check("now has unexported work", session.has_unexported_work())

            # --- chaining ---
            s2 = session.stage(source, _img(90), "warm it up")
            check("chain grows in order", [s.index for s in session.chain(source)] == [1, 2])
            check("latest is the newest", session.latest(source).path == s2.path)
            check("versions do not overwrite each other", s1.path != s2.path)
            check("both staged files still exist",
                  os.path.isfile(s1.path) and os.path.isfile(s2.path))

            # --- per-image isolation ---
            other = os.path.join(folder, "683A1089.CR3")
            open(other, "wb").close()
            o1 = session.stage(other, _img(10), "different photo")
            check("a second image gets its own chain", len(session.chain(other)) == 1)
            check("first image's chain is unaffected", len(session.chain(source)) == 2)
            check("staged names do not collide", o1.path != s1.path)

            # --- undo drops only the newest ---
            now = session.undo_last(source)
            check("undo returns the new newest", now is not None and now.path == s1.path)
            check("undo removed the file", not os.path.isfile(s2.path))
            check("undo left the older version", os.path.isfile(s1.path))
            check("chain shrank", [s.index for s in session.chain(source)] == [1])

            # --- export is the only thing that writes a real file ---
            dest = os.path.join(folder, "020A1358-edit.tif")
            written = session.export(source, dest)
            check("export wrote the destination", os.path.isfile(written))
            check("export landed where asked", os.path.abspath(written) == os.path.abspath(dest))
            check(
                "exported pixels match the staged version",
                _read(written) is not None and int(_read(written)[0, 0, 0]) == 40,
                f"got {None if _read(written) is None else int(_read(written)[0, 0, 0])}",
            )
            check("no .part left behind", not os.path.exists(dest + ".part"))

            # --- discard clears the chain and its files ---
            remaining = session.chain(source)
            session.discard(source)
            check("discard empties the chain", session.chain(source) == [])
            check(
                "discard deleted the staged files",
                all(not os.path.isfile(s.path) for s in remaining),
            )
            check("exported file survives discard", os.path.isfile(dest))
            check("other image untouched by discard", len(session.chain(other)) == 1)

            # --- export with nothing staged is an error, not a silent no-op ---
            try:
                session.export(source, os.path.join(folder, "nope.tif"))
                check("export with nothing staged raises", False)
            except ValueError:
                check("export with nothing staged raises", True)

            root = session.root
        finally:
            session.cleanup()

        check("cleanup removed the staging directory", not os.path.isdir(root))
        check("cleanup left the exported file alone", os.path.isfile(dest))
        check("cleanup left the original alone", os.path.isfile(source))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


def _read(path):
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return None if bgr is None else bgr[:, :, ::-1]


if __name__ == "__main__":
    sys.exit(main())
