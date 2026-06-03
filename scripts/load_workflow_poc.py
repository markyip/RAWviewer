#!/usr/bin/env python3
"""
Headless POC: compare RAWviewer load workflows for TTFR (thumbnail-ready proxy).

Scenarios (per workflow):
  - cold: memory + disk preview caches cleared for the file
  - warm: immediate re-request (production repeat-open)
  - interrupt: start file A, switch to file B after --interrupt-ms

Workflows mirror production paths:
  - preview_first: stages thumbnail only (fast-open / deferred full+exif)
  - thumb_then_exif: thumbnail then exif in one worker pass
  - full_pipeline: thumbnail + exif + full in one task
  - processor_direct: UnifiedImageProcessor display-tier path only (no Qt manager)

Usage (repo root):
  python scripts/load_workflow_poc.py "K:\\Photos\\Croatia"
  python scripts/load_workflow_poc.py "K:\\Photos\\Croatia" --file DSC05381.ARW
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SUPPORTED = {
    ".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2", ".pef",
    ".srw", ".raf", ".jpg", ".jpeg", ".png", ".tif", ".tiff",
}


@dataclass
class LoadMetrics:
    workflow: str
    scenario: str
    file_path: str
    ttfr_ms: Optional[float] = None
    exif_ms: Optional[float] = None
    full_ms: Optional[float] = None
    done_ms: Optional[float] = None
    thumb_max_dim: int = 0
    cache_hit: bool = False
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _buffer_max_dim(buf) -> int:
    try:
        if hasattr(buf, "shape"):
            return max(int(buf.shape[0]), int(buf.shape[1]))
        if hasattr(buf, "width"):
            return max(int(buf.width()), int(buf.height()))
    except Exception:
        pass
    return 0


def pick_random_arw(
    folder: str, exclude: Optional[str | List[str]] = None
) -> str:
    excluded = set()
    if exclude is None:
        pass
    elif isinstance(exclude, str):
        excluded.add(os.path.normcase(os.path.abspath(exclude)))
    else:
        for item in exclude:
            excluded.add(os.path.normcase(os.path.abspath(item)))
    paths: List[str] = []
    with os.scandir(folder) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in SUPPORTED:
                continue
            ap = os.path.abspath(entry.path)
            if os.path.normcase(ap) in excluded:
                continue
            paths.append(ap)
    if not paths:
        raise SystemExit(f"No images under {folder}")
    return random.choice(paths)


def cold_invalidate(cache, path: str) -> None:
    cache.invalidate_file(path)
    try:
        cache.disk_preview_cache.remove(path)
        cache.disk_thumbnail_cache.remove(path)
        cache.disk_grid_cache.remove(path)
    except Exception:
        pass


def run_manager_workflow(
    manager,
    *,
    workflow: str,
    scenario: str,
    path: str,
    interrupt_path: Optional[str] = None,
    interrupt_ms: int = 120,
    timeout_s: float = 120.0,
) -> LoadMetrics:
    from PyQt6.QtCore import QEventLoop, QTimer
    from image_load_manager import Priority

    m = LoadMetrics(workflow=workflow, scenario=scenario, file_path=path)
    t0 = time.perf_counter()
    thumb_at: List[float] = []
    exif_at: List[float] = []
    full_at: List[float] = []
    done_at: List[float] = []
    errors: List[str] = []

    def _elapsed() -> float:
        return (time.perf_counter() - t0) * 1000.0

    def on_thumb(fp, thumb):
        if os.path.normcase(fp) != os.path.normcase(path):
            return
        thumb_at.append(_elapsed())
        m.thumb_max_dim = max(m.thumb_max_dim, _buffer_max_dim(thumb))

    def on_exif(fp, _data):
        if os.path.normcase(fp) != os.path.normcase(path):
            return
        exif_at.append(_elapsed())

    def on_full(fp, _img):
        if os.path.normcase(fp) != os.path.normcase(path):
            return
        full_at.append(_elapsed())

    def on_err(fp, msg):
        if os.path.normcase(fp) in {
            os.path.normcase(path),
            os.path.normcase(interrupt_path or ""),
        }:
            errors.append(str(msg))

    def on_done(fp):
        if os.path.normcase(fp) == os.path.normcase(path):
            done_at.append(_elapsed())

    manager.thumbnail_ready.connect(on_thumb)
    manager.exif_data_ready.connect(on_exif)
    manager.image_ready.connect(on_full)
    manager.error_occurred.connect(on_err)
    manager.task_completed.connect(on_done)

    loop = QEventLoop()
    finished = {"ok": False}

    def _finish():
        finished["ok"] = True
        loop.quit()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(_finish)
    watchdog.start(int(timeout_s * 1000))

    def _maybe_finish():
        if workflow == "preview_first":
            if thumb_at and done_at:
                watchdog.stop()
                _finish()
        elif workflow == "full_pipeline":
            if done_at and (full_at or errors):
                watchdog.stop()
                _finish()
        elif done_at:
            watchdog.stop()
            _finish()

    manager.task_completed.connect(lambda _fp: _maybe_finish())
    manager.thumbnail_ready.connect(lambda _fp, _t: _maybe_finish())
    manager.image_ready.connect(lambda _fp, _i: _maybe_finish())

    stages_map = {
        "preview_first": {"thumbnail"},
        "thumb_then_exif": {"thumbnail", "exif"},
        "full_pipeline": {"thumbnail", "exif", "full"},
    }
    stages = stages_map.get(workflow)
    if stages is None:
        m.error = f"unknown workflow {workflow}"
        return m

    use_full = False
    if scenario == "interrupt" and interrupt_path:
        m.extra["interrupt_from"] = os.path.basename(interrupt_path)

        manager.load_image(
            interrupt_path,
            priority=Priority.CURRENT,
            cancel_existing=False,
            use_full_resolution=use_full,
            stages=stages,
        )

        def _switch():
            manager.cancel_task(interrupt_path)
            manager.load_image(
                path,
                priority=Priority.CURRENT,
                cancel_existing=True,
                use_full_resolution=use_full,
                stages=stages,
            )

        QTimer.singleShot(interrupt_ms, _switch)
    else:
        manager.load_image(
            path,
            priority=Priority.CURRENT,
            cancel_existing=True,
            use_full_resolution=use_full,
            stages=stages,
        )

    loop.exec()
    manager.thumbnail_ready.disconnect(on_thumb)
    manager.exif_data_ready.disconnect(on_exif)
    manager.image_ready.disconnect(on_full)
    manager.error_occurred.disconnect(on_err)
    manager.task_completed.disconnect(on_done)

    if thumb_at:
        m.ttfr_ms = thumb_at[0]
    if exif_at:
        m.exif_ms = exif_at[0]
    if full_at:
        m.full_ms = full_at[0]
    if done_at:
        m.done_ms = done_at[-1]
    if errors and not thumb_at and not full_at:
        m.error = errors[0]
    if scenario == "warm" and m.ttfr_ms is not None and m.ttfr_ms < 80:
        m.cache_hit = True
    return m


def run_processor_direct(path: str, *, cold: bool) -> LoadMetrics:
    from image_cache import get_image_cache
    from unified_image_processor import UnifiedImageProcessor

    m = LoadMetrics(
        workflow="processor_direct",
        scenario="cold" if cold else "warm",
        file_path=path,
    )
    cache = get_image_cache()
    if cold:
        cold_invalidate(cache, path)
    proc = UnifiedImageProcessor()
    t0 = time.perf_counter()
    try:
        exif, thumb = proc._extract_raw_preview_before_full_exiftool(path, None, None)
        if thumb is None:
            thumb = proc.process_thumbnail(path, allow_heavy_fallback=True)
        if thumb is not None:
            thumb = proc.ensure_display_tier_preview(path, thumb)
        elapsed = (time.perf_counter() - t0) * 1000.0
        m.ttfr_ms = elapsed
        m.thumb_max_dim = _buffer_max_dim(thumb) if thumb is not None else 0
        m.extra["has_minimal_exif"] = bool(exif and exif.get("minimal_preview_exif"))
    except Exception as e:
        m.error = str(e)
    return m


def format_table(rows: List[LoadMetrics]) -> str:
    headers = [
        "workflow",
        "scenario",
        "file",
        "TTFR_ms",
        "EXIF_ms",
        "FULL_ms",
        "DONE_ms",
        "thumb_px",
        "cache_hit",
        "error",
    ]
    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    for r in rows:
        base = os.path.basename(r.file_path)
        lines.append(
            " | ".join(
                [
                    r.workflow,
                    r.scenario,
                    base,
                    f"{r.ttfr_ms:.1f}" if r.ttfr_ms is not None else "-",
                    f"{r.exif_ms:.1f}" if r.exif_ms is not None else "-",
                    f"{r.full_ms:.1f}" if r.full_ms is not None else "-",
                    f"{r.done_ms:.1f}" if r.done_ms is not None else "-",
                    str(r.thumb_max_dim or "-"),
                    "Y" if r.cache_hit else "",
                    (r.error or "")[:40],
                ]
            )
        )
    return "\n".join(lines)


def recommend(rows: List[LoadMetrics]) -> str:
    def _one(wf: str, sc: str) -> Optional[LoadMetrics]:
        for r in rows:
            if r.workflow == wf and r.scenario == sc:
                return r
        return None

    preview_c = _one("preview_first", "cold")
    thumb_c = _one("thumb_then_exif", "cold")
    full_c = _one("full_pipeline", "cold")
    warm = _one("preview_first", "warm")
    interrupt = _one("preview_first", "interrupt")
    direct_c = _one("processor_direct", "cold")

    lines = ["## Recommendation (POC)"]
    if preview_c:
        lines.append(
            f"- **Cold preview_first (production path):** {preview_c.ttfr_ms:.0f} ms "
            f"@ {preview_c.thumb_max_dim}px — {os.path.basename(preview_c.file_path)}"
        )
    if thumb_c:
        lines.append(
            f"- **Cold thumb+exif combined:** {thumb_c.ttfr_ms:.0f} ms — "
            f"{os.path.basename(thumb_c.file_path)}"
        )
    if warm:
        lines.append(
            f"- **Warm repeat (preview_first, same session):** {warm.ttfr_ms:.0f} ms"
            + (" (likely memory cache)" if warm.cache_hit else "")
        )
    if interrupt:
        lines.append(
            f"- **Interrupt switch TTFR:** {interrupt.ttfr_ms:.0f} ms "
            f"(from {interrupt.extra.get('interrupt_from', '?')} → "
            f"{os.path.basename(interrupt.file_path)})"
        )
    if full_c and preview_c and full_c.ttfr_ms and preview_c.ttfr_ms:
        lines.append(
            f"- **Cold full_pipeline TTFR:** {full_c.ttfr_ms:.0f} ms; "
            f"full decode at {full_c.full_ms:.0f} ms"
        )
    if direct_c and preview_c and direct_c.ttfr_ms and preview_c.ttfr_ms:
        lines.append(
            f"- **Processor-only vs manager preview_first (cold):** "
            f"{direct_c.ttfr_ms:.0f} ms vs {preview_c.ttfr_ms:.0f} ms"
        )
    lines.extend(
        [
            "- **Optimal workflow:** `preview_first` — `stages={'thumbnail'}` only, "
            "`cancel_existing=True` on navigation, defer sensor-full + full exiftool until "
            "after first paint; avoid `full_pipeline` on cold open (adds full decode latency).",
            "- **Warm target:** memory preview hit <50 ms TTFR; repeat-open should not call "
            "`load_raw_image` when pixels already on screen (app-level guard).",
            "- **Interrupt:** prioritize CURRENT + cancel prior path; expect TTFR ~ cold-open "
            "minus overlap if first file decode was cancelled early.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless load workflow POC")
    parser.add_argument("folder", help="Folder with images (e.g. K:\\Photos\\Croatia)")
    parser.add_argument("--file", help="Specific file name or path (default: random ARW)")
    parser.add_argument(
        "--interrupt-file",
        help="File to load before interrupt switch (default: second random)",
    )
    parser.add_argument("--interrupt-ms", type=int, default=120)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json-out", help="Write results JSON to path")
    parser.add_argument(
        "--clear-all-between-workflows",
        action="store_true",
        help="clear_all() between workflow families (harsher cold)",
    )
    parser.add_argument(
        "--harsh-cold",
        action="store_true",
        help="clear_all + per-file invalidate before each cold run (true cold)",
    )
    args = parser.parse_args()
    if args.harsh_cold:
        args.clear_all_between_workflows = True

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 1

    if args.seed is not None:
        random.seed(args.seed)

    if args.file:
        path = args.file if os.path.isabs(args.file) else os.path.join(folder, args.file)
        path = os.path.abspath(path)
    else:
        path = pick_random_arw(folder)

    interrupt_path = None
    if args.interrupt_file:
        interrupt_path = (
            args.interrupt_file
            if os.path.isabs(args.interrupt_file)
            else os.path.join(folder, args.interrupt_file)
        )
        interrupt_path = os.path.abspath(interrupt_path)
    else:
        interrupt_path = pick_random_arw(folder, exclude=path)

    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("RAWVIEWER_VERBOSE_ORIENTATION_LOGS", "0")

    from PyQt6.QtCore import QCoreApplication
    from image_cache import get_image_cache
    from image_load_manager import ImageLoadManager

    app = QCoreApplication(sys.argv)
    cache = get_image_cache()
    manager = ImageLoadManager()

    print(f"POC folder: {folder}")
    print(f"Primary:   {os.path.basename(path)} ({path})")
    print(f"Interrupt: {os.path.basename(interrupt_path)} ({interrupt_path})")
    print(f"Platform:  {os.environ.get('QT_QPA_PLATFORM', 'default')}")
    print()

    rows: List[LoadMetrics] = []

    # Fair cold runs: separate files so OS read cache does not bias later workflows.
    used: List[str] = [path]
    cold_files = {"preview_first": path}
    cold_files["thumb_then_exif"] = pick_random_arw(folder, exclude=used)
    used.append(cold_files["thumb_then_exif"])
    cold_files["full_pipeline"] = pick_random_arw(folder, exclude=used)

    print("Cold-run files (isolated per workflow):")
    for wf, fp in cold_files.items():
        print(f"  {wf}: {os.path.basename(fp)}")
    print()

    for workflow in ("preview_first", "thumb_then_exif", "full_pipeline"):
        cold_path = cold_files[workflow]
        if args.clear_all_between_workflows or args.harsh_cold:
            cache.clear_all()
        cold_invalidate(cache, cold_path)
        rows.append(
            run_manager_workflow(
                manager,
                workflow=workflow,
                scenario="cold",
                path=cold_path,
            )
        )
        app.processEvents()

        # Warm repeat always uses primary path (user-facing repeat open).
        rows.append(
            run_manager_workflow(
                manager,
                workflow=workflow,
                scenario="warm",
                path=path,
            )
        )
        app.processEvents()

    if args.clear_all_between_workflows:
        cache.clear_all()
    cold_invalidate(cache, path)
    cold_invalidate(cache, interrupt_path)
    rows.append(
        run_manager_workflow(
            manager,
            workflow="preview_first",
            scenario="interrupt",
            path=path,
            interrupt_path=interrupt_path,
            interrupt_ms=args.interrupt_ms,
        )
    )
    app.processEvents()

    cold_invalidate(cache, path)
    rows.append(run_processor_direct(path, cold=True))
    rows.append(run_processor_direct(path, cold=False))

    manager.shutdown()
    app.processEvents()

    print(format_table(rows))
    print()
    print(recommend(rows))

    if args.json_out:
        payload = [
            {
                "workflow": r.workflow,
                "scenario": r.scenario,
                "file": r.file_path,
                "ttfr_ms": r.ttfr_ms,
                "exif_ms": r.exif_ms,
                "full_ms": r.full_ms,
                "done_ms": r.done_ms,
                "thumb_max_dim": r.thumb_max_dim,
                "cache_hit": r.cache_hit,
                "error": r.error,
                "extra": r.extra,
            }
            for r in rows
        ]
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
