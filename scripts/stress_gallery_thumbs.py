#!/usr/bin/env python3
"""Headless gallery thumbnail stress test (Windows crash isolation).

Simulates gallery load_visible_images bursts: many concurrent RAW thumbnail
decodes on an external folder, repeated in waves like scrolling.

Run via scripts/run_gallery_stress_tests.ps1 or directly:
  pixi run python scripts/stress_gallery_thumbs.py --label baseline
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# Repo src on path when launched from repo root.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _collect_arw(folder: str, limit: int) -> list[str]:
    out: list[str] = []
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                if entry.name.lower().endswith(".arw"):
                    out.append(entry.path)
                    if len(out) >= limit:
                        break
    except OSError as exc:
        print(f"[STRESS] scandir failed: {exc}", flush=True)
    return out


def run_stress(
    folder: str,
    *,
    label: str,
    file_limit: int,
    waves: int,
    batch_size: int,
    wave_pause_s: float,
) -> int:
    from PyQt6.QtCore import QCoreApplication, QTimer
    from PyQt6.QtWidgets import QApplication

    from image_load_manager import ImageLoadManager, Priority

    paths = _collect_arw(folder, file_limit)
    if not paths:
        print(f"[STRESS] No ARW files in {folder}", flush=True)
        return 2

    # Headless Qt (no window).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv)

    mgr = ImageLoadManager()
    mgr.update_volume_throttling(folder)
    mgr.prime_volume_speed_async(folder, sample_path=paths[0])

    done: set[str] = set()
    errors: list[str] = []
    wave_idx = 0
    scheduled_total = 0
    t0 = time.monotonic()

    def on_thumb(path: str, _thumb) -> None:
        done.add(path)

    def on_err(path: str, msg: str) -> None:
        errors.append(f"{os.path.basename(path)}: {msg}")

    mgr.thumbnail_ready.connect(on_thumb)
    mgr.error_occurred.connect(on_err)

    def schedule_wave() -> None:
        nonlocal wave_idx, scheduled_total
        if wave_idx >= waves:
            finish()
            return
        wave_idx += 1
        start = (wave_idx - 1) * batch_size
        batch = paths[start : start + batch_size]
        if not batch:
            finish()
            return
        print(
            f"[STRESS] wave {wave_idx}/{waves} scheduling {len(batch)} thumbs "
            f"(active_threads={mgr._thread_pool.activeThreadCount()}, "
            f"raw_limit={getattr(mgr, '_raw_load_limit', '?')})",
            flush=True,
        )
        for path in batch:
            mgr.load_image(
                path,
                priority=Priority.CURRENT,
                cancel_existing=False,
                stages={"thumbnail"},
                gallery_thumbnail=True,
                bypass_cache=True,
            )
            scheduled_total += 1
        QTimer.singleShot(int(wave_pause_s * 1000), schedule_wave)

    def finish() -> None:
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            app.processEvents()
            pending = scheduled_total - len(done) - len(errors)
            active = mgr._thread_pool.activeThreadCount()
            if pending <= 0 and active == 0:
                break
            time.sleep(0.05)
        elapsed = time.monotonic() - t0
        print(
            f"[STRESS] {label} DONE in {elapsed:.1f}s "
            f"scheduled={scheduled_total} ok={len(done)} err={len(errors)} "
            f"pending={scheduled_total - len(done) - len(errors)}",
            flush=True,
        )
        if errors[:5]:
            print(f"[STRESS] sample errors: {errors[:5]}", flush=True)
        mgr.shutdown()
        app.quit()

    print(
        f"[STRESS] {label} folder={folder} files={len(paths)} "
        f"waves={waves} batch={batch_size} pause={wave_pause_s}s",
        flush=True,
    )
    QTimer.singleShot(0, schedule_wave)
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=r"I:\Photos\London")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--file-limit", type=int, default=320)
    parser.add_argument("--waves", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--wave-pause", type=float, default=2.0)
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[STRESS] Folder missing: {args.folder}", flush=True)
        return 2

    try:
        return run_stress(
            args.folder,
            label=args.label,
            file_limit=args.file_limit,
            waves=args.waves,
            batch_size=args.batch_size,
            wave_pause_s=args.wave_pause,
        )
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
