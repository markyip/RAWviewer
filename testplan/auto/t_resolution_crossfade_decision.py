#!/usr/bin/env python3
"""Regression guard: a same-file re-paint at the same resolution must NOT
crossfade -- crossfading identical content is a visible pulse/flash.

Real bug (observed live in rawviewer_dev.log, navigating a folder): after the
legitimate low->high crossfade (e.g. 512x341 -> 8256x5504 fade=True), a SECOND
display_pixmap for the same file fired at the SAME resolution and still logged
fade=True (8256x5504 -> 8256x5504). Root cause: the viewport crossfade DEFERS
_commit_display_metadata, so _manager_displayed_max_dim / _displayed_content_path
stay stale during the fade; duplicate full-res deliveries then slip past the
redundancy guards and each re-marks _pending_resolution_crossfade, so the
redundant second paint crossfades a full-res buffer over an identical one --
the "low-res -> high-res flashing" the user reported.

Fix: _resolution_upgrade_needs_crossfade() returns False when the new buffer's
max dimension is not actually larger than what's on screen (<= prev * 1.06),
even if _pending_resolution_crossfade is set. There is no resolution change to
dissolve. Genuine upgrades (new_max well above prev) still fade.

Uses a lightweight fake object (mirrors the other t_*.py suites) with a QPixmap
stand-in exposing width()/height().
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


class FakePm:
    def __init__(self, w, h):
        self._w, self._h = w, h

    def isNull(self):
        return self._w <= 0 or self._h <= 0

    def width(self):
        return self._w

    def height(self):
        return self._h


def make_stub(*, pending_crossfade=False, gallery_pending=False, crossfade_enabled=True):
    import main as m

    v = type("V", (), {})()
    v._crossfade_enabled = crossfade_enabled
    v._pending_resolution_crossfade = pending_crossfade
    v._gallery_preview_pending_full = gallery_pending
    v._resolution_upgrade_needs_crossfade = (
        m.RAWImageViewer._resolution_upgrade_needs_crossfade.__get__(v)
    )
    return v


def needs(v, prev_wh, new_wh, *, prev_path="/x/A.NEF", cur_path="/x/A.NEF"):
    return v._resolution_upgrade_needs_crossfade(
        FakePm(*prev_wh), FakePm(*new_wh), prev_path=prev_path, cur_path=cur_path
    )


def main() -> int:
    # --- the flash case: same file, same resolution, pending flag set ---
    v = make_stub(pending_crossfade=True)
    check(
        "same-file same-res re-paint does NOT crossfade even with pending flag (no pulse)",
        needs(v, (8256, 5504), (8256, 5504)) is False,
    )
    v = make_stub(pending_crossfade=True)
    check(
        "same-file re-paint within 6% tolerance does NOT crossfade",
        needs(v, (8256, 5504), (8300, 5504)) is False,
    )
    v = make_stub(pending_crossfade=True)
    check(
        "same-file DOWNGRADE does NOT crossfade",
        needs(v, (8256, 5504), (1024, 768)) is False,
    )

    # --- genuine upgrades still fade ---
    v = make_stub(pending_crossfade=False)
    check(
        "genuine low->high upgrade crossfades (512 -> 8256)",
        needs(v, (512, 341), (8256, 5504)) is True,
    )
    v = make_stub(pending_crossfade=True)
    check(
        "pending flag still forces a fade on a real upgrade",
        needs(v, (1536, 1024), (8256, 5504)) is True,
    )

    # --- unchanged guards ---
    v = make_stub(pending_crossfade=True)
    check(
        "different file never crossfades (navigation)",
        needs(v, (8256, 5504), (512, 341), cur_path="/x/B.NEF") is False,
    )
    v = make_stub(pending_crossfade=True, crossfade_enabled=False)
    check(
        "crossfade globally disabled -> never fades",
        needs(v, (512, 341), (8256, 5504)) is False,
    )
    v = make_stub(gallery_pending=True)
    check(
        "gallery blur->sharp upgrade still crossfades",
        needs(v, (512, 341), (8256, 5504)) is True,
    )

    # --- the no-actual-upgrade guard runs BEFORE the pending-flag shortcut ---
    src = inspect.getsource(
        __import__("main").RAWImageViewer._resolution_upgrade_needs_crossfade
    )
    noupgrade = src.find("new_max <= int(prev_max * 1.06)")
    pending = src.rfind("_pending_resolution_crossfade")
    check(
        "no-upgrade early-return precedes the pending-crossfade shortcut",
        noupgrade != -1 and pending != -1 and noupgrade < pending,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
