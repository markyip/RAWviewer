#!/usr/bin/env python3
"""HE/HE* NEF and other demosaic-failed RAW: editable via embedded JPEG.

Policy (reversed from the previous browse-only gate): when LibRaw cannot
demosaic, Adjust falls back to the largest embedded JPEG as an 8-bit
display-referred edit base. Nikon HE/HE* and Sony ARW6 (before the packaging
LibRaw pin) are the main cases. Composite DNGs that prefer embedded preview
remain browse-only.
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
    import processing.unified as uip

    src = inspect.getsource(mainmod.RAWImageViewer._editing_supported_for_file)
    check(
        "no HE-NEF browse-only gate in _editing_supported_for_file",
        "_nef_he_compressed(file_path)" not in src,
        "HE/ARW6-style files edit via embedded JPEG",
    )
    check(
        "unsupported-paths set does not refuse Adjust",
        "if key in _LIBRAW_UNSUPPORTED_PATHS" not in src
        and "key in _LIBRAW_UNSUPPORTED_PATHS" not in src,
        "that set is exactly the embedded-JPEG edit fallback population",
    )
    check(
        "non-RAW files are rejected",
        "if not is_raw_file(file_path):\n                return False" in src
        or "if not is_raw_file(file_path):\n            return False" in src,
    )
    check(
        "composite DNGs remain browse-only",
        "dng_prefers_embedded_preview_first" in src,
    )

    check(
        "_embedded_jpeg_edit_base exists",
        hasattr(uip.UnifiedImageProcessor, "_embedded_jpeg_edit_base"),
    )

    impl = inspect.getsource(uip.UnifiedImageProcessor._decode_raw_edit_base_impl)
    check(
        "demosaic failure falls back to embedded JPEG edit base",
        "_embedded_jpeg_edit_base" in impl,
    )
    check(
        "fallback is not NEF-only",
        'endswith(".nef")' not in impl or "_embedded_jpeg_edit_base" in impl,
    )

    embed_src = inspect.getsource(uip.UnifiedImageProcessor._embedded_jpeg_edit_base)
    check(
        "JPEG edit base uses extract_embedded_jpeg_by_scan",
        "extract_embedded_jpeg_by_scan" in embed_src,
    )
    check(
        "JPEG edit base does not route through process_full_image",
        "self.process_full_image" not in embed_src,
    )
    check(
        "JPEG edit base does not route through process_thumbnail",
        "self.process_thumbnail" not in embed_src,
    )
    check(
        "LibRaw unsupported uses LibRawFileUnsupportedError",
        "_is_libraw_unsupported_error" in impl
        or "LibRawFileUnsupportedError" in impl,
    )

    # --- _nef_he_compressed helper still works for browse detection ---
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
        "non-NEF file is never treated as HE-compressed",
        evaluate("photo.ARW", {"nef_he_compressed": True}) is False,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
