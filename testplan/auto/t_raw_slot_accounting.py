"""Heavy RAW slots survive cancellation without double-freeing.

A task cancelled mid-demosaic keeps running: the GPU/LibRaw call is not
interruptible. Slots used to be reconciled by counting _active_tasks, which
that task has already been removed from, so its slot was dropped on the
floor -- then its _task_finished decremented again, taking the counter
below the real concurrency and admitting more heavy decodes than the limit
allows. On heavy_limit=1 that is exactly the thrash the limit exists to
prevent.

The holder set is now the source of truth, so the count cannot disagree
with reality regardless of which cancel path ran.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []
_QAPP = None


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _Task:
    def __init__(self, path, priority):
        self.file_path = path
        self.priority = priority
        self.stages = {"full"}
        self._counted_raw_slot = False
        self._cancelled = False
        self.task_key = f"{path}:full"

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled


def _manager():
    """A manager with only the slot-accounting state wired up."""
    import threading

    import image_load_manager as ilm

    m = ilm.ImageLoadManager.__new__(ilm.ImageLoadManager)
    m._queue_lock = threading.RLock()
    m._active_tasks = {}
    m._task_keys_by_path = {}
    m._raw_slot_holders = set()
    m._active_raw_tasks = 0
    m._running_tasks = set()
    m._running_tasks_lock = threading.Lock()
    m._compact_work_queue = lambda: None
    return m


def _admit(m, task):
    with m._queue_lock:
        m._active_tasks[task.task_key] = task
        m._claim_raw_slot_locked(task)


def _finish(m, task):
    """What _task_finished does to the slot."""
    with m._queue_lock:
        m._active_tasks.pop(task.task_key, None)
        m._release_raw_slot_locked(task)


def main() -> int:
    import image_load_manager as ilm
    from image_load_manager import Priority

    # Construct the REAL object first. Every other case here uses __new__ to
    # isolate the slot accounting, which meant __init__ was never exercised --
    # and a lock aliased one line before it was created shipped green, with
    # the app unable to start at all. Cheap check, caught nothing else.
    from PyQt6.QtWidgets import QApplication

    # Bound to a name: an unreferenced QApplication is collected, and
    # ImageLoadManager refuses to construct without one.
    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    try:
        real = ilm.ImageLoadManager()
        check("ImageLoadManager() constructs", True)
        check(
            "throttle state uses one lock",
            real._throttle_lock is real._indexing_throttle_lock,
        )
        check(
            "and it is reentrant, so the existing nesting is safe",
            type(real._throttle_lock).__name__ == "RLock",
            type(real._throttle_lock).__name__,
        )
        real.shutdown()
    except Exception as exc:  # noqa: BLE001
        check("ImageLoadManager() constructs", False, repr(exc))

    # --- the exact interleaving from the report ---
    m = _manager()
    a = _Task("/a.CR3", Priority.CURRENT)
    b = _Task("/b.CR3", Priority.CURRENT)
    _admit(m, a)
    _admit(m, b)
    check("two heavies claim two slots", m._active_raw_tasks == 2, str(m._active_raw_tasks))

    # Folder change: cancel_all_tasks, while A and B are still demosaicing.
    ilm.ImageLoadManager.cancel_all_tasks(m)
    check(
        "cancel_all_tasks releases the slots it cancels",
        m._active_raw_tasks == 0,
        str(m._active_raw_tasks),
    )
    check("no stale holders left", len(m._raw_slot_holders) == 0)
    check("flags cleared so finish cannot double-free", not a._counted_raw_slot and not b._counted_raw_slot)

    # New folder admits C and D.
    c = _Task("/c.CR3", Priority.CURRENT)
    d = _Task("/d.CR3", Priority.CURRENT)
    _admit(m, c)
    _admit(m, d)
    check("new folder's heavies count 2", m._active_raw_tasks == 2, str(m._active_raw_tasks))

    # The cancelled-but-running A and B now finish. This is the double free.
    _finish(m, a)
    _finish(m, b)
    check(
        "A and B finishing does NOT decrement C and D's slots",
        m._active_raw_tasks == 2,
        f"{m._active_raw_tasks} (was 0 before the fix)",
    )
    _finish(m, c)
    _finish(m, d)
    check("counter reaches zero when the real holders finish", m._active_raw_tasks == 0)

    # --- cancel_tasks_by_priority released nothing at all ---
    m = _manager()
    e = _Task("/e.CR3", Priority.PRELOAD_NEXT)
    _admit(m, e)
    ilm.ImageLoadManager.cancel_tasks_by_priority(m, Priority.PRELOAD_NEXT)
    check("cancel_tasks_by_priority releases the slot", m._active_raw_tasks == 0, str(m._active_raw_tasks))
    _finish(m, e)
    check("its later finish does not go negative", m._active_raw_tasks == 0)

    # --- a cancel that keeps some tasks must keep their slots ---
    m = _manager()
    keep = _Task("/keep.CR3", Priority.CURRENT)
    drop = _Task("/drop.CR3", Priority.PRELOAD_NEXT)
    _admit(m, keep)
    _admit(m, drop)
    ilm.ImageLoadManager.cancel_tasks_by_priority(m, Priority.PRELOAD_NEXT)
    check("kept task keeps its slot", m._active_raw_tasks == 1, str(m._active_raw_tasks))
    check("kept task still flagged", keep._counted_raw_slot is True)

    # --- releasing twice is harmless ---
    m = _manager()
    f = _Task("/f.CR3", Priority.CURRENT)
    _admit(m, f)
    _finish(m, f)
    _finish(m, f)
    check("double release cannot go negative", m._active_raw_tasks == 0, str(m._active_raw_tasks))

    # --- claiming twice does not double-count ---
    m = _manager()
    g = _Task("/g.CR3", Priority.CURRENT)
    _admit(m, g)
    _admit(m, g)
    check("double claim counts once", m._active_raw_tasks == 1, str(m._active_raw_tasks))

    # --- cancelled after claim, before tryStart: the slot must come back ---
    m = _manager()
    m._stopped = False
    late = _Task("/late.CR3", Priority.CURRENT)
    _admit(m, late)
    late.cancel()
    # What the start loop does for a task cancelled between admit and tryStart.
    with m._queue_lock:
        m._release_raw_slot_locked(late)
        m._active_tasks.pop(late.task_key, None)
    check(
        "slot released when the task never runs",
        m._active_raw_tasks == 0,
        f"{m._active_raw_tasks} -- a leak here pins the limit forever",
    )

    # --- cancel_all_tasks_except must release the ones it drops ---
    m = _manager()
    keep2 = _Task("/keep2.CR3", Priority.CURRENT)
    drop2 = _Task("/drop2.CR3", Priority.CURRENT)
    _admit(m, keep2)
    _admit(m, drop2)
    m._task_keys_by_path = {"/keep2.CR3": {keep2.task_key}}
    m._work_queue = None
    ilm.ImageLoadManager.cancel_all_tasks_except(m, "/keep2.CR3")
    check(
        "cancel_all_tasks_except releases the cancelled heavy",
        m._active_raw_tasks == 1,
        f"{m._active_raw_tasks} (2 means the neighbour still holds the limit)",
    )
    check("the kept task still holds its slot", keep2._counted_raw_slot is True)

    # --- reconcile agrees with the holder set, not the task map ---
    m = _manager()
    h = _Task("/h.CR3", Priority.CURRENT)
    _admit(m, h)
    m._active_tasks.clear()  # cancelled-but-running: gone from the map
    with m._queue_lock:
        m._reconcile_raw_slots_locked()
    check(
        "reconcile keeps a slot whose task left the map",
        m._active_raw_tasks == 1,
        f"{m._active_raw_tasks} (counting _active_tasks would give 0)",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
