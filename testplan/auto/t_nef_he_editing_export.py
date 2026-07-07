#!/usr/bin/env python3
"""HE/HE*-compressed NEF: editing and export must actually work via the
embedded-JPEG edit base instead of silently failing.

Before this fix: opening the Adjust panel on a never-before-tried HE-NEF
forced RAW mode, failed, and left the panel open with nothing usable (only
a 5s status message); export always attempted a full RAW decode and
hard-raised RuntimeError("Full-resolution RAW decode failed") regardless of
the embedded-JPEG workflow toggle.

This is a source-inspection + fake-object behavioral suite (mirrors the
t_gallery_closes_editor.py convention) since exercising the real Adjust
panel/export worker needs a live QMainWindow and real RAW files neither of
which are practical here -- it verifies the exact conditions/branches the
fix depends on, not just that "some code changed."
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

    # --- _editing_supported_for_file: HE-NEF is editable, not blocked ---
    src = inspect.getsource(mainmod.RAWImageViewer._editing_supported_for_file)
    check(
        "_editing_supported_for_file carves out an HE-NEF exception before "
        "the general _LIBRAW_UNSUPPORTED_PATHS block",
        "_nef_he_compressed(file_path)" in src,
    )
    he_idx = src.find("_nef_he_compressed(file_path)")
    blocked_idx = src.find("if key in _LIBRAW_UNSUPPORTED_PATHS")
    check(
        "HE-NEF check runs before the generic unsupported-paths block "
        "(so it isn't shadowed by an early return)",
        -1 not in (he_idx, blocked_idx) and he_idx < blocked_idx,
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
        "never-classified NEF (flag absent) -> False (falls through to normal handling)",
        evaluate("photo.NEF", {}) is False,
    )
    check(
        "non-NEF file is never treated as HE-compressed regardless of cache content",
        evaluate("photo.ARW", {"nef_he_compressed": True}) is False,
    )
    check(
        "no cached EXIF row at all -> False, does not raise",
        evaluate("photo.NEF", None) is False,
    )

    # --- _toggle_adjust_panel: does not force RAW mode for HE-NEF ---
    src = inspect.getsource(mainmod.RAWImageViewer._toggle_adjust_panel)
    check(
        "_toggle_adjust_panel's RAW-mode auto-switch is gated on _nef_he_compressed",
        "not (current_path and self._nef_he_compressed(current_path))" in src,
    )

    # --- decode_raw_edit_base: NEF fallback uses the app's own byte-scan
    # extractor, not the DNG branch's PIL multi-frame trick ---
    src = inspect.getsource(uip.UnifiedImageProcessor.decode_raw_edit_base)
    check(
        "decode_raw_edit_base has a dedicated .nef branch",
        'file_path.lower().endswith(".nef")' in src,
    )
    check(
        "NEF branch uses extract_embedded_jpeg_by_scan (verified byte-scan "
        "path), not PIL's generic multi-frame reader",
        "extract_embedded_jpeg_by_scan(file_path, 0)" in src,
    )
    nef_branch_idx = src.find('file_path.lower().endswith(".nef")')
    dng_branch_idx = src.find("dng_prefers_embedded_preview_first(file_path)")
    check(
        "NEF branch is a sibling of (not nested inside) the DNG branch",
        -1 not in (nef_branch_idx, dng_branch_idx) and dng_branch_idx < nef_branch_idx,
    )

    # --- _on_adjust_edit_base_ready: message reflects embedded-JPEG substitutes ---
    src = inspect.getsource(mainmod.RAWImageViewer._on_adjust_edit_base_ready)
    check(
        "edit-base-ready handler distinguishes an embedded-JPEG substitute "
        "(uint8) from a true RAW decode by dtype",
        "_np.uint8" in src,
    )

    # --- export: decode_raw_edit_base is called unconditionally, and its
    # NEF fallback means export no longer depends on a successful RAW decode ---
    export_src = inspect.getsource(mainmod._AdjustExportWorker.run)
    check(
        "export always calls decode_raw_edit_base (now HE-NEF-aware) rather "
        "than a separate/duplicated decode path",
        "processor.decode_raw_edit_base(" in export_src,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
