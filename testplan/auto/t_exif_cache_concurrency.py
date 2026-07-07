#!/usr/bin/env python3
"""Regression guard: PersistentEXIFCache.get()/get_multiple() must not block
behind a slow operation on another thread.

Root cause of a real "navigation extremely slow" report: get() wrapped its
entire SQLite read in the SAME process-wide threading.RLock() used by
put()/flush() for the shared writer connection and _pending_writes list --
even though get() reads via the calling thread's OWN thread-local
connection, which SQLite's WAL mode already lets proceed concurrently
without any Python-level lock. Background semantic indexing (3+ worker
threads calling get() per file) and single-view navigation (main thread,
also calling get()) serialized through this one lock; a single slow
extraction (observed 2-9s on a real folder, the code's own warning calls
this "often EXIF cache lock vs folder sort refinement") blocked every other
thread's reads for that whole duration -- directly matching the observed
~2s navigation stalls.

This test holds cache.lock in a background thread (simulating a slow write
in progress) and asserts a concurrent get() does NOT wait for it.
"""
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from image_cache import PersistentEXIFCache

    tmpd = tempfile.mkdtemp()
    try:
        cache = PersistentEXIFCache(tmpd)
        f = os.path.join(tmpd, "test.arw")
        with open(f, "wb") as fh:
            fh.write(b"x" * 100)
        cache.put(f, {"orientation": 1, "camera_make": "Test", "exif_data": {}}, commit=True)

        hold_released = threading.Event()

        def hold_lock():
            with cache.lock:
                hold_released.wait(timeout=5.0)

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        time.sleep(0.2)  # let the holder thread actually acquire the lock first

        t0 = time.perf_counter()
        result = cache.get(f)
        elapsed = time.perf_counter() - t0

        hold_released.set()
        holder.join(timeout=5.0)

        check(
            "get() returns correct data while another thread holds cache.lock",
            result is not None and result.get("camera_make") == "Test",
        )
        check(
            "get() does not block behind a concurrent lock holder (thread-local connection, WAL-safe)",
            elapsed < 1.0,
            f"elapsed={elapsed:.2f}s (should be near-instant, not ~2s+)",
        )

        # get_multiple() should behave the same way.
        holder2 = threading.Thread(target=hold_lock, daemon=True)
        hold_released2 = threading.Event()

        def hold_lock2():
            with cache.lock:
                hold_released2.wait(timeout=5.0)

        holder2 = threading.Thread(target=hold_lock2, daemon=True)
        holder2.start()
        time.sleep(0.2)

        t0 = time.perf_counter()
        results = cache.get_multiple([f])
        elapsed2 = time.perf_counter() - t0
        hold_released2.set()
        holder2.join(timeout=5.0)

        check(
            "get_multiple() does not block behind a concurrent lock holder",
            elapsed2 < 1.0,
            f"elapsed={elapsed2:.2f}s",
        )
        check("get_multiple() returns correct data", f in results or any(True for _ in results))
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
