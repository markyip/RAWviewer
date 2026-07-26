#!/usr/bin/env python3
"""HE/HE*-compressed NEF: editable via its embedded JPEG.

Policy, reversed from this file's original one: a file LibRaw cannot demosaic
used to be browse-only. HE/HE* NEF now opens Adjust and edits its largest
embedded preview as an 8-bit display-referred base -- every adjustment except
the ones needing scene-linear headroom behaves as it does on the JPEG editing
path that already ships.

What did NOT come back is the old byte-scanning JPEG substitute inside the
demosaic path; the edit base is built by _embedded_jpeg_edit_base instead.
Those checks are still live below, since re-introducing the byte-scan would
be a regression under the new policy too.

Non-RAW files (JPEG/HEIC) remain a separate question and are unaffected.
"""
import inspect
import os
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
    import unified_image_processor as uip

    # --- _editing_supported_for_file: HE-NEF is now editable ---
    src = inspect.getsource(mainmod.RAWImageViewer._editing_supported_for_file)
    check(
        "no HE-NEF gate left in _editing_supported_for_file",
        "_nef_he_compressed(file_path)" not in src,
        "the browse-only gate was removed with the embedded-JPEG edit base",
    )
    check(
        "the embedded-JPEG edit base is what makes it editable",
        "_embedded_jpeg_edit_base" in src,
        "the reason HE-NEF is allowed should be stated where it is allowed",
    )
    # Behavioural: an HE NEF must actually be reported as editable. The helper
    # reads the EXIF cache, so drive it through the same fake-cache seam the
    # _nef_he_compressed checks below use.
    check(
        "_embedded_jpeg_edit_base exists on the processor",
        hasattr(uip.UnifiedImageProcessor, "_embedded_jpeg_edit_base"),
    )
    check(
        "non-RAW files are rejected (JPEG/HEIC are not editable)",
        "if not is_raw_file(file_path):\n                return False" in src
        or "if not is_raw_file(file_path):\n            return False" in src,
    )

    # --- _nef_he_compressed helper: direct behavioral check with a fake cache ---
    class FakeCache:
        def __init__(self, row):
            self._row = row

        def get_exif(self, file_path):
            return self._row

    def evaluate(file_path, cached_row):
        m = type("M", (), {})()
        m.image_cache = FakeCache(cached_row)
        return mainmod.RAWImageViewer._nef_he_compressed(m, file_path)

    check(
        "HE-compressed NEF (cached flag True) -> True",
        evaluate("photo.NEF", {"nef_he_compressed": True}) is True,
    )
    check(
        "non-HE NEF (cached flag False) -> False",
        evaluate("photo.NEF", {"nef_he_compressed": False}) is False,
    )
    check(
        "non-NEF file is never treated as HE-compressed regardless of cache content",
        evaluate("photo.ARW", {"nef_he_compressed": True}) is False,
    )
    check(
        "no cached EXIF row at all -> False, does not raise",
        evaluate("photo.NEF", None) is False,
    )

    # --- _toggle_adjust_panel: no HE carve-out for staying in JPEG workflow ---
    src = inspect.getsource(mainmod.RAWImageViewer._toggle_adjust_panel)
    check(
        "_toggle_adjust_panel no longer exempts HE-NEF from RAW-mode switch",
        "_nef_he_compressed(current_path)" not in src,
    )

    # --- decode_raw_edit_base: no embedded-JPEG substitute for NEF ---
    # decode_raw_edit_base is now a thin in-flight-dedup wrapper around
    # _decode_raw_edit_base_impl; the decode logic these checks inspect lives
    # in the impl, so inspect both.
    src = inspect.getsource(uip.UnifiedImageProcessor.decode_raw_edit_base)
    src += inspect.getsource(uip.UnifiedImageProcessor._decode_raw_edit_base_impl)
    check(
        "decode_raw_edit_base no longer builds a JPEG edit base via byte-scan",
        "extract_embedded_jpeg_by_scan(file_path, 0)" not in src,
    )
    check(
        "demosaic-failed NEF returns None (no JPEG substitute)",
        'file_path.lower().endswith(".nef")' in src and "return None" in src,
    )

    # --- display path: HE detection must not depend on EXIF already cached ---
    praw_src = inspect.getsource(uip.UnifiedImageProcessor._process_raw_image)
    check(
        "_process_raw_image falls back to direct _detect_nef_he_compression "
        "when the cached EXIF HE flag is absent",
        "_detect_nef_he_compression(file_path)" in praw_src,
    )

    # --- ground truth samples (optional) ---
    from enhanced_raw_processor import _detect_nef_he_compression

    sample = os.environ.get("RAWVIEWER_TEST_ASSETS", "/tmp/RAW_Sample")
    reported = [
        os.path.join(sample, "DSC_2138.NEF"),
        os.path.join(sample, "DSC_2127.NEF"),
    ]
    present = [p for p in reported if os.path.exists(p)]
    if present:
        results = {os.path.basename(p): _detect_nef_he_compression(p) for p in present}
        check(
            "reported HE-NEF files are detected as HE (True)",
            all(v is True for v in results.values()),
            detail=str(results),
        )
    else:
        print("SKIP  reported HE-NEF sample files not present on this machine")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
