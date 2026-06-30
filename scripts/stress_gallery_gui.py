#!/usr/bin/env python3
"""GUI gallery scroll stress — GalleryView + widgets + concurrent indexing.

Closer to the real crash path than stress_gallery_thumbs.py (headless ILM only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class _MockViewer:
    def __init__(self, image_manager, folder: str):
        self.image_manager = image_manager
        self.current_folder = folder
        self._folder_load_generation = 1
        self.image_cache = image_manager._cache


def _list_images(folder: str) -> list[str]:
    exts = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".jpg", ".jpeg", ".heic", ".png"}
    out: list[str] = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
                out.append(entry.path)
    out.sort()
    return out


def _start_indexing(folder: str, paths: list[str]) -> None:
    if os.environ.get("RAWVIEWER_ENABLE_SEMANTIC_SEARCH", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return

    def _worker() -> None:
        try:
            from semantic_search import SemanticImageIndex

            index = SemanticImageIndex(folder)
            index.build_index(paths)
        except Exception as exc:
            print(f"[GUI-STRESS] indexing error: {exc}", flush=True)

    import threading

    threading.Thread(target=_worker, name="GuiStressIndex", daemon=True).start()


def run_gui_stress(
    folder: str,
    *,
    label: str,
    duration_s: float,
    viewport_w: int,
    viewport_h: int,
) -> int:
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QWidget

    from image_load_manager import get_image_load_manager
    from rawviewer_ui.gallery_view import JustifiedGallery

    paths = _list_images(folder)
    if not paths:
        print(f"[GUI-STRESS] no images in {folder}", flush=True)
        return 2

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)

    mgr = get_image_load_manager()
    mgr.update_volume_throttling(folder)
    mgr.prime_volume_speed_async(folder, sample_path=paths[0])

    viewer = _MockViewer(mgr, folder)
    container = QWidget()
    container.resize(viewport_w, viewport_h)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(False)
    scroll.resize(viewport_w, viewport_h)
    gallery = JustifiedGallery([], None)
    gallery.parent_viewer = viewer
    gallery.images = paths
    gallery._rebuild_normalized_map()
    gallery.resize(viewport_w - 24, 8000)
    scroll.setWidget(gallery)
    layout.addWidget(scroll)
    container.show()

    gallery.begin_gallery_warmup(file_count=len(paths))
    gallery.set_images(paths, force_rebuild=True)

    t0 = time.monotonic()
    tick = 0
    max_scroll = 0

    def simulate_scroll() -> None:
        nonlocal tick, max_scroll
        elapsed = time.monotonic() - t0
        if elapsed >= duration_s:
            print(
                f"[GUI-STRESS] {label} DONE {elapsed:.1f}s ticks={tick} "
                f"active={len(gallery._active_tasks)} widgets={len(gallery._visible_widgets)}",
                flush=True,
            )
            mgr.shutdown()
            app.quit()
            return
        tick += 1
        sb = scroll.verticalScrollBar()
        max_scroll = max(0, sb.maximum())
        # Sweep up/down like user scrolling through 1136-image folder.
        phase = tick % 200
        if phase < 100:
            sb.setValue(min(max_scroll, phase * max(1, max_scroll // 100)))
        else:
            sb.setValue(max(0, max_scroll - (phase - 100) * max(1, max_scroll // 100)))
        gallery._is_scrolling_fast = phase % 17 < 3
        gallery._request_load_visible_images(0)
        if tick % 25 == 0:
            print(
                f"[GUI-STRESS] {label} t={elapsed:.0f}s tick={tick} "
                f"scroll={sb.value()}/{max_scroll} active={len(gallery._active_tasks)} "
                f"visible_widgets={len(gallery._visible_widgets)}",
                flush=True,
            )
        QTimer.singleShot(180, simulate_scroll)

    print(
        f"[GUI-STRESS] {label} folder={folder} images={len(paths)} "
        f"duration={duration_s}s viewport={viewport_w}x{viewport_h}",
        flush=True,
    )
    _start_indexing(folder, paths)
    QTimer.singleShot(500, simulate_scroll)
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=r"I:\Photos\London")
    parser.add_argument("--label", default="gui_baseline")
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    try:
        return run_gui_stress(
            args.folder,
            label=args.label,
            duration_s=args.duration,
            viewport_w=args.width,
            viewport_h=args.height,
        )
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
