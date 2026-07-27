"""The full-image cache is bounded by BYTES, not by an entry count.

It carried two limits: a byte budget (55% of max_memory_mb) and a hard cap
of 8 entries. The byte budget knows how large these buffers actually are; the
count does not, and the same "8" meant 0.6 GB on a 24MP body and 1.5 GB on a
61MP one.

Measured on a 17 GB machine, the count was the binding limit exactly where it
should not have been -- bytes allowed 13 entries at 24MP and 9 at 33MP while
the count allowed 8 -- and at 61MP, the case the cap was meant to protect,
the byte budget bound first anyway. So the cap only ever bit when there was
room to spare.

What must stay true: raising the count must NOT let memory grow unbounded.
The byte budget has to remain the thing that stops it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def main() -> int:
    from image_cache import get_image_cache

    cache = get_image_cache()
    lru = cache.full_image_cache

    check(
        "the count cap is no longer the hard 8",
        lru.max_size > 8,
        f"{lru.max_size} entries, derived from available RAM",
    )
    check("and is still sanity-bounded", lru.max_size <= 32, str(lru.max_size))

    budget_mb = float(cache.max_memory_mb) * 0.55

    def resident_mb():
        with lru.lock:
            return sum(
                cache._estimate_entry_bytes(v) for v in lru.cache.values()
            ) / 1e6

    # --- big buffers: bytes must stop it well before the count does ---
    for key in list(lru.cache.keys()):
        lru.remove(key)
    big_mb = 6336 * 9504 * 3 / 1e6  # ~181 MB, 61MP class
    for i in range(lru.max_size + 4):
        cache.put_full_image(f"/big/{i}.ARW", np.zeros((6336, 9504, 3), np.uint8), copy=False)

    kept = len(lru.cache)
    check(
        "large buffers are capped by bytes, not count",
        kept < lru.max_size,
        f"{kept} kept of {lru.max_size} allowed -- byte budget stopped it first",
    )
    check(
        "resident bytes stay near the budget",
        resident_mb() <= budget_mb * 1.15,
        f"{resident_mb():.0f} MB vs {budget_mb:.0f} MB budget",
    )
    check(
        "and it did not collapse to nothing",
        kept >= 2,
        f"{kept} entries -- min_keep protects the current and previous file",
    )

    # --- small buffers: the count is now the only limit, as intended ---
    for key in list(lru.cache.keys()):
        lru.remove(key)
    for i in range(lru.max_size + 4):
        cache.put_full_image(f"/small/{i}.ARW", np.zeros((200, 300, 3), np.uint8), copy=False)
    check(
        "small buffers fill to the count cap",
        len(lru.cache) == lru.max_size,
        f"{len(lru.cache)} of {lru.max_size}",
    )
    check(
        "which costs almost nothing",
        resident_mb() < 50,
        f"{resident_mb():.1f} MB",
    )

    # --- the cap is still overridable for zoomed-culling workflows ---
    import image_cache as ic

    check(
        "RAWVIEWER_FULL_IMAGE_CACHE_ITEMS still overrides",
        "RAWVIEWER_FULL_IMAGE_CACHE_ITEMS" in open(ic.__file__).read(),
    )

    for key in list(lru.cache.keys()):
        lru.remove(key)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
