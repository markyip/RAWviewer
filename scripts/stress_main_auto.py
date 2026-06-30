#!/usr/bin/env python3
"""Full-app gallery crash reproduction (real RAWImageViewer + GPU view).

Runs main.py with automated: fast-open file -> wait full-res -> gallery -> scroll.
Execute in subprocess to capture native exit codes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _worker_main(folder: str, start_file: str, duration_s: float, label: str) -> int:
    root = _repo_root()
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    os.environ.setdefault("RAWVIEWER_FILE_LOG", "1")
    os.environ.setdefault("RAWVIEWER_FATAL_DUMP", "1")

    from PyQt6.QtCore import QTimer

    import logging

    # main.main() takes an `if not logger.handlers: ...` branch that never
    # assigns `platform_system` before referencing it later in that function
    # (real startup avoids this because `if __name__ == '__main__'` calls
    # setup_logging() before main() runs). Clear handlers, then call
    # setup_logging() ourselves so this worker matches the real entrypoint.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.WARNING)

    import main as main_mod
    main_mod.setup_logging()
    from main import RAWImageViewer

    viewer_ref: list = []
    t0_holder: list = []
    stress_armed = {"done": False}

    def _scroll_gallery() -> None:
        viewer = viewer_ref[0] if viewer_ref else None
        if viewer is None:
            return
        import time

        elapsed = time.monotonic() - t0_holder[0]
        if elapsed >= duration_s:
            print(
                f"[MAIN-STRESS] {label} finished {elapsed:.1f}s — quitting",
                flush=True,
            )
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
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
                active = len(getattr(gallery, "_active_tasks", {}) or {})
                print(
                    f"[MAIN-STRESS] {label} t={elapsed:.0f}s scroll={sb.value()}/{mx} "
                    f"active={active} mode={getattr(viewer, 'view_mode', '?')}",
                    flush=True,
                )
        QTimer.singleShot(200, _scroll_gallery)

    def _begin_stress() -> None:
        viewer = viewer_ref[0]
        import time

        t0_holder.append(time.monotonic())
        print(f"[MAIN-STRESS] {label} toggling to gallery", flush=True)
        if getattr(viewer, "view_mode", "single") != "gallery":
            viewer.toggle_view_mode()
        QTimer.singleShot(500, _scroll_gallery)

    _orig_show = RAWImageViewer.show

    def _patched_show(self) -> None:
        _orig_show(self)
        if stress_armed["done"]:
            return
        stress_armed["done"] = True
        viewer_ref.clear()
        viewer_ref.append(self)
        folder_path = folder
        start = start_file
        print(
            f"[MAIN-STRESS] {label} loading {start} in {folder_path}",
            flush=True,
        )
        self.load_folder_images(folder_path, start_file=start)
        # Match user log: full-res had time to load before gallery toggle (~3s).
        QTimer.singleShot(3500, _begin_stress)

    RAWImageViewer.show = _patched_show  # type: ignore[method-assign]

    sys.argv = ["main.py", os.path.join(folder, start_file)]
    try:
        main_mod.main()
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else 0
        return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=r"I:\Photos\London")
    parser.add_argument("--start-file", default="DSC00734.ARW")
    parser.add_argument("--duration", type=float, default=210.0)
    parser.add_argument("--label", default="main_baseline")
    parser.add_argument("--in-process", action="store_true")
    args = parser.parse_args()

    if args.in_process:
        return _worker_main(args.folder, args.start_file, args.duration, args.label)

    env = os.environ.copy()
    env.update(
        {
            "RAWVIEWER_USE_PROCESS_POOL": env.get("RAWVIEWER_USE_PROCESS_POOL", "1"),
            "RAWVIEWER_ENABLE_SEMANTIC_SEARCH": env.get(
                "RAWVIEWER_ENABLE_SEMANTIC_SEARCH", "1"
            ),
            "RAWVIEWER_AUTO_METADATA_INDEX": "1",
            "RAWVIEWER_GPU_VIEW": env.get("RAWVIEWER_GPU_VIEW", "1"),
            "RAWVIEWER_FILE_LOG": "1",
            "RAWVIEWER_FATAL_DUMP": "1",
            "RAWVIEWER_VERBOSE_INFO_LOGS": "1",
        }
    )
    cmd = [
        "pixi",
        "run",
        "python",
        "scripts/stress_main_worker.py",
        "--folder",
        args.folder,
        "--start-file",
        args.start_file,
        "--duration",
        str(args.duration),
        "--label",
        args.label,
    ]
    print(f"[MAIN-STRESS] launching subprocess: {args.label}", flush=True)
    proc = subprocess.run(cmd, cwd=_repo_root(), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
