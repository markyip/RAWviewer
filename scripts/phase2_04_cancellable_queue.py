"""Phase 2 item 4: ImageLoadManager queue compaction after cancel."""
from __future__ import annotations

import os
import queue
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_load_manager import ImageLoadManager, ImageLoadTask, Priority


class _CompactQueueManager(ImageLoadManager):
    """Minimal manager stub for queue compaction tests (no Qt app required)."""

    def __init__(self):
        self._work_queue = queue.PriorityQueue()
        self._active_tasks = {}
        self._task_keys_by_path = {}
        self._queue_lock = __import__("threading").Lock()
        self._active_raw_tasks = 0
        self._stopped = False


def _queue_tasks(mgr: _CompactQueueManager) -> list[ImageLoadTask]:
    kept = []
    while not mgr._work_queue.empty():
        try:
            kept.append(mgr._work_queue.get_nowait())
        except queue.Empty:
            break
    return kept


def test_compact_drops_cancelled_tasks() -> None:
    mgr = _CompactQueueManager()
    t1 = ImageLoadTask("/a.jpg", Priority.CURRENT)
    t2 = ImageLoadTask("/b.jpg", Priority.PRELOAD_NEXT)
    t3 = ImageLoadTask("/c.jpg", Priority.BACKGROUND)
    mgr._work_queue.put(t1)
    mgr._work_queue.put(t2)
    mgr._work_queue.put(t3)

    t1.cancel()
    t3.cancel()

    with mgr._queue_lock:
        mgr._compact_work_queue()

    remaining = _queue_tasks(mgr)
    assert len(remaining) == 1
    assert remaining[0].file_path == "/b.jpg"
    assert not remaining[0].is_cancelled()


def test_cancel_task_compacts_queue() -> None:
    mgr = _CompactQueueManager()
    task = ImageLoadTask("/x.cr3", Priority.CURRENT)
    task.task_key = ("/x.cr3", False, ("thumbnail",), None)
    mgr._active_tasks[task.task_key] = task
    mgr._task_keys_by_path["/x.cr3"] = {task.task_key}
    mgr._work_queue.put(task)

    mgr.cancel_task("/x.cr3")

    remaining = _queue_tasks(mgr)
    assert remaining == []


def test_cancel_all_compacts_queue() -> None:
    mgr = _CompactQueueManager()
    tasks = [
        ImageLoadTask("/1.jpg", Priority.CURRENT),
        ImageLoadTask("/2.jpg", Priority.PRELOAD_NEXT),
    ]
    for t in tasks:
        mgr._work_queue.put(t)
        key = (t.file_path, False, ("thumbnail",), None)
        t.task_key = key
        mgr._active_tasks[key] = t
        mgr._task_keys_by_path.setdefault(t.file_path, set()).add(key)

    mgr.cancel_all_tasks()

    remaining = _queue_tasks(mgr)
    assert remaining == []


if __name__ == "__main__":
    test_compact_drops_cancelled_tasks()
    test_cancel_task_compacts_queue()
    test_cancel_all_compacts_queue()
    print("OK: Phase 2 item 4 (cancellable queue compaction) tests passed")
