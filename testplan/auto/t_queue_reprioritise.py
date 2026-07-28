"""Queued work follows the viewport instead of the enqueue order.

Task priority was fixed when the task was created, so after a scroll -- and
especially after a jump, where EVERY queued item becomes irrelevant at once
-- the queue was ordered by where the user used to be. Measured on a real
session: thumbnail tasks waited a median of 509 ms and a p99 of 3.8 s before
their work even started.

Two halves. reprioritise() re-scores the queue against the current viewport;
cancel_distant_queued_tasks() drops queued work the user has scrolled well
past. The second matters because the gallery's own cancel diff only runs once
scrolling settles, which is exactly when the queue is no longer flooded.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _manager():
    import queue as _q
    import threading

    import image_load_manager as ilm

    m = ilm.ImageLoadManager.__new__(ilm.ImageLoadManager)
    m._queue_lock = threading.RLock()
    m._work_queue = _q.PriorityQueue()
    m._active_tasks = {}
    m._task_keys_by_path = {}
    m._raw_slot_holders = set()
    m._active_raw_tasks = 0
    m._stopped = False
    m._schedule_next_task = lambda: None
    return m


def main() -> int:
    import image_load_manager as ilm
    from image_load_manager import ImageLoadTask, Priority

    def task(path, priority=Priority.BACKGROUND):
        t = ImageLoadTask(path, priority=priority, stages={"thumbnail"})
        t.task_key = (path, False, frozenset({"thumbnail"}))
        return t

    def drain(m):
        out = []
        while not m._work_queue.empty():
            out.append(m._work_queue.get_nowait().file_path)
        return out

    # --- default ordering does not depend on the viewport ---
    m = _manager()
    for i in range(6):
        t = task(f"/f/{i}.ARW")
        m._active_tasks[t.task_key] = t
        m._task_keys_by_path.setdefault(t.file_path, set()).add(t.task_key)
        m._work_queue.put(t)
    check("queue starts with six tasks", m._work_queue.qsize() == 6)

    # --- the jump case: the user is now at 4-5, everything else is behind ---
    distances = {"/f/0.ARW": 40, "/f/1.ARW": 30, "/f/2.ARW": 20,
                 "/f/3.ARW": 10, "/f/4.ARW": 0, "/f/5.ARW": 0}
    n = ilm.ImageLoadManager.reprioritise(m, lambda p: distances[p])
    check("every queued task is re-scored", n == 6, str(n))
    order = drain(m)
    # 4 and 5 are both distance 0; their order relative to each other is a
    # heap tie and not something worth pinning.
    check(
        "what is on screen comes out first",
        set(order[:2]) == {"/f/4.ARW", "/f/5.ARW"},
        str(order),
    )
    check(
        "and the rest follow by distance",
        order[2:] == ["/f/3.ARW", "/f/2.ARW", "/f/1.ARW", "/f/0.ARW"],
        str(order[2:]),
    )

    # --- CURRENT still outranks a nearer background tile ---
    m = _manager()
    near_bg = task("/f/near.ARW", Priority.BACKGROUND)
    far_current = task("/f/far.ARW", Priority.CURRENT)
    for t in (near_bg, far_current):
        m._active_tasks[t.task_key] = t
        m._task_keys_by_path.setdefault(t.file_path, set()).add(t.task_key)
        m._work_queue.put(t)
    ilm.ImageLoadManager.reprioritise(
        m, lambda p: 0 if p == "/f/near.ARW" else 50
    )
    check(
        "priority band still wins over distance",
        drain(m)[0] == "/f/far.ARW",
        "the photo being viewed must not queue behind a thumbnail",
    )

    # --- dropping work the user scrolled past ---
    m = _manager()
    for i in range(6):
        t = task(f"/f/{i}.ARW")
        m._active_tasks[t.task_key] = t
        m._task_keys_by_path.setdefault(t.file_path, set()).add(t.task_key)
        m._work_queue.put(t)
    far = {f"/f/{i}.ARW": (0 if i >= 4 else 100) for i in range(6)}
    dropped = ilm.ImageLoadManager.cancel_distant_queued_tasks(m, lambda p: far[p], 24)
    check("distant queued work is cancelled", dropped == 4, str(dropped))
    remaining = drain(m)
    check("near work survives", sorted(remaining) == ["/f/4.ARW", "/f/5.ARW"], str(remaining))
    check(
        "cancelled tasks leave the active map",
        len(m._active_tasks) == 2,
        f"{len(m._active_tasks)} left -- a stale entry would block re-requests",
    )

    # --- CURRENT is never dropped, however far away ---
    m = _manager()
    t = task("/f/current.ARW", Priority.CURRENT)
    m._active_tasks[t.task_key] = t
    m._task_keys_by_path.setdefault(t.file_path, set()).add(t.task_key)
    m._work_queue.put(t)
    dropped = ilm.ImageLoadManager.cancel_distant_queued_tasks(m, lambda p: 999, 24)
    check("the viewed photo is never cancelled by distance", dropped == 0)
    check("and stays queued", drain(m) == ["/f/current.ARW"])

    # --- a distance function that raises must not lose the queue ---
    m = _manager()
    for i in range(3):
        t = task(f"/f/{i}.ARW")
        m._work_queue.put(t)

    def boom(_p):
        raise RuntimeError("no layout yet")

    ilm.ImageLoadManager.reprioritise(m, boom)
    check("a failing distance function keeps every task", m._work_queue.qsize() == 3)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
