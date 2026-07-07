#!/usr/bin/env python3
"""Regression guard: don't decode the same file's full-resolution image twice
concurrently just because two callers asked for it under different stage sets.

Real bug (from a real running log): navigating to a file queues a combined
{'exif', 'full', 'thumbnail'} task; ~1.2s later (once the user "pauses" on
it), _schedule_idle_full_decode_after_nav_preview separately queues a bare
{'full'} task via _maybe_queue_background_full_decode. ImageLoadManager's
task-key dedup (_make_task_key) keys on the *exact* stage set, so these two
tasks -- both wanting 'full' for the same file -- never matched each other,
and both ran a full LibRaw/embedded-JPEG decode of the same multi-megapixel
image concurrently (confirmed in a real log: "[PREVIEW] Using full-resolution
embedded JPEG for DSC03236.ARW (6656x9984)" logged twice, ~1s apart, for one
navigation to one file).

Fix: load_image() now checks _full_stage_already_in_flight() before queuing
-- if another active task for the same file already owns the 'full' stage
(regardless of its other stages), 'full' is stripped from the new request
(or the whole request is skipped if 'full' was all it wanted).

This uses a lightweight fake object (mirrors the t_gallery_closes_editor.py
convention) rather than constructing a real ImageLoadManager, since that
spins up real thread pools/watchdogs -- _full_stage_already_in_flight only
reads self._queue_lock/_active_tasks/_task_keys_by_path, so a fake with just
those three attributes exercises the real bound method faithfully.
"""
import inspect
import os
import sys
import threading
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class FakeTask:
    def __init__(self, cancelled=False, priority=None):
        self._cancelled = cancelled
        # Default owners to BACKGROUND (the common prefetch/idle full-decode
        # priority); tests that care set it explicitly.
        import image_load_manager as ilm

        self.priority = priority if priority is not None else ilm.Priority.BACKGROUND

    def is_cancelled(self):
        return self._cancelled


def make_manager_stub():
    import image_load_manager as ilm

    m = type("M", (), {})()
    m._queue_lock = threading.Lock()
    m._active_tasks = {}
    m._task_keys_by_path = defaultdict(set)
    m._full_stage_already_in_flight = ilm.ImageLoadManager._full_stage_already_in_flight.__get__(m)
    return m


def register(m, file_path, use_full_resolution, stages, *, cancelled=False, priority=None):
    key = (file_path, use_full_resolution, tuple(sorted(stages)), None)
    m._active_tasks[key] = FakeTask(cancelled=cancelled, priority=priority)
    m._task_keys_by_path[file_path].add(key)
    return key


def main() -> int:
    import image_load_manager as ilm

    P = ilm.Priority

    # --- _full_stage_already_in_flight direct behavioral checks ---
    # A BACKGROUND requester never triggers the supersede path, so it sees any
    # surviving full owner; use it for the base "is it detected at all" checks.
    m = make_manager_stub()
    check(
        "no active tasks -> not in flight",
        m._full_stage_already_in_flight("a.NEF", True, P.BACKGROUND) is False,
    )

    register(m, "a.NEF", True, {"full"})
    check(
        "bare {'full'} task registered -> in flight for same file+use_full_resolution",
        m._full_stage_already_in_flight("a.NEF", True, P.BACKGROUND) is True,
    )
    check(
        "different use_full_resolution value -> not considered the same work",
        m._full_stage_already_in_flight("a.NEF", False, P.BACKGROUND) is False,
    )
    check(
        "different file -> not in flight",
        m._full_stage_already_in_flight("b.NEF", True, P.BACKGROUND) is False,
    )

    m2 = make_manager_stub()
    register(m2, "c.CR3", True, {"exif", "full", "thumbnail"})
    check(
        "'full' embedded in a combined stage set is still detected",
        m2._full_stage_already_in_flight("c.CR3", True, P.BACKGROUND) is True,
    )

    m3 = make_manager_stub()
    register(m3, "d.ARW", True, {"exif", "thumbnail"})
    check(
        "task without 'full' in its stages does not count as in flight",
        m3._full_stage_already_in_flight("d.ARW", True, P.BACKGROUND) is False,
    )

    m4 = make_manager_stub()
    register(m4, "e.ARW", True, {"full"}, cancelled=True)
    check(
        "cancelled task is not considered in flight",
        m4._full_stage_already_in_flight("e.ARW", True, P.BACKGROUND) is False,
    )

    # --- regression: a CURRENT request must NOT defer to an owner it will
    # supersede (cancel). Pressing "next" onto a prefetched neighbor issues a
    # CURRENT {'exif','thumbnail','full'} load while a PRELOAD_NEXT {'full'}
    # prefetch is in flight; if the dedup strips 'full' and then the supersede
    # block cancels that prefetch, the full decode is lost entirely. ---
    for owner_prio, label in (
        (P.PRELOAD_NEXT, "PRELOAD_NEXT"),
        (P.PRELOAD_PREV, "PRELOAD_PREV"),
        (P.BACKGROUND, "BACKGROUND"),
    ):
        m5 = make_manager_stub()
        register(m5, "f.NEF", False, {"full"}, priority=owner_prio)
        check(
            f"CURRENT requester does NOT defer to a lower-priority ({label}) full owner it will cancel",
            m5._full_stage_already_in_flight("f.NEF", False, P.CURRENT) is False,
        )
        check(
            f"a non-CURRENT (PRELOAD_NEXT) requester still dedups against a {label} full owner",
            m5._full_stage_already_in_flight("f.NEF", False, P.PRELOAD_NEXT) is True,
        )

    # A CURRENT requester DOES defer to another CURRENT full owner (equal
    # priority is not superseded -- strict '>' -- so the owner survives).
    m6 = make_manager_stub()
    register(m6, "g.NEF", False, {"full"}, priority=P.CURRENT)
    check(
        "CURRENT requester defers to another CURRENT full owner (survives supersede)",
        m6._full_stage_already_in_flight("g.NEF", False, P.CURRENT) is True,
    )

    # --- integration: load_image() actually consults this before queuing ---
    import image_load_manager as ilm

    src = inspect.getsource(ilm.ImageLoadManager.load_image)
    check(
        "load_image() checks _full_stage_already_in_flight before creating a task",
        "_full_stage_already_in_flight(" in src,
    )
    check(
        "load_image() strips 'full' from the request rather than silently keeping it",
        "effective_stages.discard('full')" in src,
    )
    check(
        "load_image() skips queuing entirely when nothing is left to do",
        "if not effective_stages:" in src,
    )
    dedup_idx = src.find("_full_stage_already_in_flight(")
    task_create_idx = src.find("task = ImageLoadTask(")
    check(
        "the in-flight check runs before the task is actually created",
        -1 not in (dedup_idx, task_create_idx) and dedup_idx < task_create_idx,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
