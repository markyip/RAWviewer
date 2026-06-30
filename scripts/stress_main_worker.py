#!/usr/bin/env python3
"""Worker: load main.py as a module, patch gallery automation, call main()."""
from __future__ import annotations

import importlib.util
import os
import sys
import time


def run(folder: str, start_file: str, duration_s: float, label: str) -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.join(root, "src")
    os.chdir(root)
    if src not in sys.path:
        sys.path.insert(0, src)

    spec = importlib.util.spec_from_file_location("rawviewer_main", os.path.join(src, "main.py"))
    if spec is None or spec.loader is None:
        print("[MAIN-STRESS] failed to load main.py", flush=True)
        return 2
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rawviewer_main"] = mod
    spec.loader.exec_module(mod)

    from PyQt6.QtCore import QTimer

    RAWImageViewer = mod.RAWImageViewer
    viewer_ref: list = []
    t0_holder: list = []
    armed = {"done": False}

    def _scroll_gallery() -> None:
        viewer = viewer_ref[0] if viewer_ref else None
        if viewer is None:
            return
        elapsed = time.monotonic() - t0_holder[0]
        if elapsed >= duration_s:
            print(f"[MAIN-STRESS] {label} done {elapsed:.1f}s", flush=True)
            app = mod.QApplication.instance()
            if app is not None:
                app.quit()
            return
        scroll = getattr(viewer, "gallery_scroll", None)
        gallery = getattr(viewer, "gallery_justified", None)
        if scroll is not None and gallery is not None:
            sb = scroll.verticalScrollBar()
            mx = max(0, sb.maximum())
            tick = int(elapsed * 5)
            phase = tick % 200
            if phase < 100:
                sb.setValue(min(mx, phase * max(1, mx // 100)))
            else:
                sb.setValue(max(0, mx - (phase - 100) * max(1, mx // 100)))
            gallery._is_scrolling_fast = phase % 17 < 3
            gallery._request_load_visible_images(0)
            if tick % 25 == 0:
                print(
                    f"[MAIN-STRESS] {label} t={elapsed:.0f}s active="
                    f"{len(getattr(gallery, '_active_tasks', {}) or {})} "
                    f"mode={getattr(viewer, 'view_mode', '?')}",
                    flush=True,
                )
        QTimer.singleShot(200, _scroll_gallery)

    def _begin_stress() -> None:
        viewer = viewer_ref[0]
        t0_holder.append(time.monotonic())
        print(f"[MAIN-STRESS] {label} -> gallery", flush=True)
        if getattr(viewer, "view_mode", "single") != "gallery":
            viewer.toggle_view_mode()
        QTimer.singleShot(500, _scroll_gallery)

    _orig_show = RAWImageViewer.show

    def _patched_show(self) -> None:
        _orig_show(self)
        if armed["done"]:
            return
        armed["done"] = True
        viewer_ref[:] = [self]
        print(f"[MAIN-STRESS] {label} fast-open {start_file}", flush=True)
        self.load_folder_images(folder, start_file=start_file)
        QTimer.singleShot(3500, _begin_stress)

    RAWImageViewer.show = _patched_show  # type: ignore[method-assign]

    sys.argv = ["main.py", os.path.join(folder, start_file)]
    # Force main.main()'s "logging already configured" branch (sets platform_system).
    import logging

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    mod.main()
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--folder", default=r"I:\Photos\London")
    p.add_argument("--start-file", default="DSC00734.ARW")
    p.add_argument("--duration", type=float, default=210.0)
    p.add_argument("--label", default="main_baseline")
    args = p.parse_args()
    raise SystemExit(run(args.folder, args.start_file, args.duration, args.label))
