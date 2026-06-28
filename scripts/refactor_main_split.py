#!/usr/bin/env python3
"""Split main.py: drop legacy gallery + extracted classes; wire rawviewer_app imports."""
from __future__ import annotations

import pathlib
import re

MAIN = pathlib.Path(__file__).resolve().parents[1] / "src" / "main.py"

IMPORT_BLOCK = """
# --- rawviewer_app modules (Phase 4 split) ---
from rawviewer_app.env import _env_int, _env_true, _norm_path, safe_print, safe_print_err
from rawviewer_app.processing import RAWProcessor, PixmapConverter
from rawviewer_app.signals import (
    FolderLoadSignals,
    FolderSortRefineSignals,
    GalleryMetadataSignals,
    QuickFolderIndexSignals,
    ReleaseUpdateCheckSignals,
    SemanticIndexPrepSignals,
    SemanticIndexSignals,
)
from rawviewer_app.widgets import (
    CustomConfirmDialog,
    CustomTitleBar,
    CustomWarningDialog,
    LoadingOverlay,
    ResizeGripIndicator,
    SingleImageViewOverlay,
    ThumbnailLabel,
    TopMetadataBar,
)
from rawviewer_app.workers import (
    _ReleaseUpdateCheckWorker,
    _SemanticIndexPrepWorker,
    SemanticAssetDownloadSignals,
)

from rawviewer_app.viewer.session_mixin import SessionMixin

"""

CONFIGURE_GUI = """
try:
    from rawviewer_app.env import configure_gui_process
    configure_gui_process(_IS_GUI_MAIN_PROCESS)
except Exception:
    pass

"""


def find_line(lines: list[str], pattern: str, start: int = 0) -> int:
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            return i
    raise ValueError(f"Pattern not found: {pattern!r} (from {start})")


def main() -> None:
    lines = MAIN.read_text(encoding="utf-8").splitlines(keepends=True)

    if "configure_gui_process" not in "".join(lines):
        i = find_line(lines, r"^_IS_GUI_MAIN_PROCESS = ")
        lines[i + 1 : i + 1] = ["\n", *CONFIGURE_GUI.splitlines(keepends=True)]

    raw_processor = find_line(lines, r"^class RAWProcessor\(QThread\):")
    image_loaded = find_line(lines, r"^class ImageLoaded\(QObject\):")
    resize_grip = find_line(lines, r"^class ResizeGripIndicator\(QWidget\):")
    filmstrip_fn = find_line(lines, r"^def _filmstrip_ui_env_int\(")
    single_overlay = find_line(lines, r"^class SingleImageViewOverlay\(QWidget\):")
    share_fn = find_line(lines, r"^def _windows_shell_verb_suggests_share\(")
    viewer_class = find_line(lines, r"^class RAWImageViewer\(QMainWindow\):")

    # Legacy gallery + gallery-only ImageLoadTask/ImageLoaded
    del lines[image_loaded:resize_grip]

    # Re-index
    raw_processor = find_line(lines, r"^class RAWProcessor\(QThread\):")
    resize_grip = find_line(lines, r"^class ResizeGripIndicator\(QWidget\):")
    filmstrip_fn = find_line(lines, r"^def _filmstrip_ui_env_int\(")
    single_overlay = find_line(lines, r"^class SingleImageViewOverlay\(QWidget\):")
    share_fn = find_line(lines, r"^def _windows_shell_verb_suggests_share\(")
    viewer_class = find_line(lines, r"^class RAWImageViewer\(QMainWindow\):")

    # processing + signals + workers (now in rawviewer_app)
    del lines[raw_processor:resize_grip]

    resize_grip = find_line(lines, r"^class ResizeGripIndicator\(QWidget\):")
    filmstrip_fn = find_line(lines, r"^def _filmstrip_ui_env_int\(")
    single_overlay = find_line(lines, r"^class SingleImageViewOverlay\(QWidget\):")
    share_fn = find_line(lines, r"^def _windows_shell_verb_suggests_share\(")
    viewer_class = find_line(lines, r"^class RAWImageViewer\(QMainWindow\):")

    # widgets (keep _filmstrip_ui_env_int helper)
    del lines[resize_grip:filmstrip_fn]
    single_overlay = find_line(lines, r"^class SingleImageViewOverlay\(QWidget\):")
    share_fn = find_line(lines, r"^def _windows_shell_verb_suggests_share\(")
    del lines[single_overlay:share_fn]

    viewer_class = find_line(lines, r"^class RAWImageViewer\(QMainWindow\):")
    lines[viewer_class] = "class RAWImageViewer(SessionMixin, QMainWindow):\n"

    load_folder = find_line(lines, r"^    def load_folder_images\(")
    change_event = find_line(lines, r"^    def changeEvent\(", load_folder)
    del lines[load_folder:change_event]

    viewer_class = find_line(lines, r"^class RAWImageViewer\(SessionMixin")
    lines[viewer_class:viewer_class] = IMPORT_BLOCK.splitlines(keepends=True)

    MAIN.write_text("".join(lines), encoding="utf-8")
    print(f"Refactored {MAIN}: {len(lines)} lines")


if __name__ == "__main__":
    main()
