"""HE/HE* NEF must keep its embedded preview instead of showing a grey tile.

An undersized preview is normally discarded so a bigger one can be built by
demosaicing. For Nikon HE/HE* there is no demosaic, so the embedded JPEG is
the best that will ever exist -- discarding it leaves a grey tile.

The guard that used to protect this asked processor.is_libraw_unsupported(),
which only knows paths that have ALREADY failed a decode and been registered
in an in-memory, prunable set. On a cold gallery scan nothing has failed yet,
so the guard missed exactly the files it existed for and three HE NEFs
rendered as grey blocks.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _FakeProcessor:
    """Stands in for UnifiedImageProcessor's registry lookup."""

    def __init__(self, registered=()):
        self._registered = {os.path.normcase(os.path.abspath(p)) for p in registered}

    def is_libraw_unsupported(self, file_path):
        return os.path.normcase(os.path.abspath(file_path)) in self._registered


def main() -> int:
    import image_load_manager as ilm

    # --- the registry route still works ---
    known = "/tmp/already-failed.CR3"
    check(
        "a path already registered unsupported is spared",
        ilm._preview_is_best_possible(_FakeProcessor([known]), known),
    )
    check(
        "an ordinary RAW is not spared (a better preview may exist)",
        not ilm._preview_is_best_possible(_FakeProcessor(), "/tmp/ordinary.CR3"),
    )

    # --- HE detection route: the cold-scan case that regressed ---
    ilm._HE_PREVIEW_ONLY_CACHE.clear()
    calls = {"n": 0}
    import enhanced_raw_processor as erp

    real = erp._detect_nef_he_compression

    def fake(path):
        calls["n"] += 1
        return path.endswith("he.NEF")

    erp._detect_nef_he_compression = fake
    try:
        proc = _FakeProcessor()  # nothing registered: a cold scan
        check(
            "HE NEF is spared even with an empty unsupported registry",
            ilm._preview_is_best_possible(proc, "/tmp/shot-he.NEF"),
        )
        check(
            "non-HE NEF is not spared",
            not ilm._preview_is_best_possible(proc, "/tmp/shot-normal.NEF"),
        )

        # Detection reads EXIF, and a folder scan asks repeatedly.
        before = calls["n"]
        for _ in range(5):
            ilm._preview_is_best_possible(proc, "/tmp/shot-he.NEF")
        check(
            "repeat lookups are memoised (EXIF read once)",
            calls["n"] == before,
            f"{calls['n'] - before} extra read(s)",
        )

        # Non-NEF must never pay for HE detection at all.
        before = calls["n"]
        ilm._preview_is_best_possible(proc, "/tmp/shot.CR3")
        check(
            "non-NEF does not trigger HE detection",
            calls["n"] == before,
            f"{calls['n'] - before} read(s)",
        )

        # A detector that throws must not take the gallery down with it.
        def boom(path):
            raise OSError("unreadable")

        erp._detect_nef_he_compression = boom
        ilm._HE_PREVIEW_ONLY_CACHE.clear()
        try:
            result = ilm._preview_is_best_possible(proc, "/tmp/broken.NEF")
            check("a failing detector degrades to False, does not raise", result is False)
        except Exception as exc:  # noqa: BLE001
            check("a failing detector degrades to False, does not raise", False, str(exc))
    finally:
        erp._detect_nef_he_compression = real
        ilm._HE_PREVIEW_ONLY_CACHE.clear()

    # --- the threshold that made this bite ---
    # Nikon's embedded preview is 1620x1080 against a 1632 floor: 12 pixels.
    check(
        "the floor is close enough to Nikon's preview to matter",
        ilm._display_preview_min_dim() > 1024,
        f"floor={ilm._display_preview_min_dim()}",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
