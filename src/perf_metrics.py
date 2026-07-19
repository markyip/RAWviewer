"""
Uniform, machine-parseable performance metrics.

Every hot-path stage emits one line through this module in a single stable
format so runs can be quantified and COMPARED (before/after a change, machine
A vs machine B) instead of eyeballing ad-hoc log strings:

    [PERF] metric=decode_full_cpu ms=412.3 file=IMG_0042.CR3 mp=32.7

Aggregate a log with ``scripts/bench/perf_report.py``:

    RAWVIEWER_PERF=1 pixi run python src/main.py 2>&1 | tee /tmp/run.log
    pixi run python scripts/bench/perf_report.py /tmp/run.log
    pixi run python scripts/bench/perf_report.py --compare baseline.log new.log

Metrics are ON by default (they are one logging call each); set
RAWVIEWER_PERF=0 to silence. The old scattered tags ([NAVTIME], [DECODE_T],
[PIPE_T], [FAST_RAW] timings) remain for human reading; this is the
machine-readable channel.

Canonical metric names (keep stable -- comparisons depend on them):
    unpack            LibRaw open+unpack (fast_raw_decode.unpack_raw)
    wb_sanity         embedded-JPEG WB check (first file of a model)
    decode_half       fit-view half tier from an unpack
    decode_full_cpu   sensor-res pixel math, CPU path
    decode_full_gpu   sensor-res pixel math, GPU path
    gpu_isp_stages    fused GPU develop stage breakdown (RAWVIEWER_GPU_ISP_TIMING=1)
    decode_rawpy      rawpy.postprocess fallback
    sidecar_apply     XMP adjustments applied to a full buffer (cache miss)
    nav_to_display    navigation keypress -> pixels on screen (main.py)
    adjust_preview    Adjust panel live/settle render (worker thread)
    db_stamp          dodge/burn single brush stamp (slow outliers only)
    db_live_patch     dodge/burn instant display-patch blit (slow outliers)
    db_edge_snap      dodge/burn guided-filter snap on stroke release
    db_stroke         dodge/burn per-stroke summary (stamp/patch totals)
    db_apply          dodge/burn gain-map build inside apply_dodge_burn (cache miss)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

_logger = logging.getLogger("rawviewer.perf")


def perf_enabled() -> bool:
    return str(os.environ.get("RAWVIEWER_PERF", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def perf_mark(metric: str, ms: float, file_path: Optional[str] = None, **kv) -> None:
    """Emit one uniform [PERF] line. Cheap; safe to call from any thread."""
    if not perf_enabled():
        return
    parts = [f"[PERF] metric={metric}", f"ms={ms:.1f}"]
    if file_path:
        parts.append(f"file={os.path.basename(file_path)}")
    for k, v in kv.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    _logger.info(" ".join(parts))


@contextmanager
def perf_timer(metric: str, file_path: Optional[str] = None, **kv):
    """Time a block and emit its [PERF] line on exit (also on exception)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        perf_mark(metric, (time.perf_counter() - t0) * 1000.0, file_path, **kv)
