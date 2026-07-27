"""Gallery cache pruning is cheap and correct; failed tiles can recover.

Two bugs, both user-visible as "the gallery got slower" and "a missing tile
never comes back in this session".

Pruning: the stale check was `any(str(key).startswith(path) for path in keep)`
over every cache entry. Keys are (path, size_bucket) tuples, so str() gives
"('/p', 512)" and the prefix never matched -- the WHOLE thumbnail cache was
dropped on every folder change, forcing every tile to decode again. Being
O(keys x paths) it also took ~27 seconds on the GUI thread for a 3000-file
folder.

Recovery: after its retries a tile got a grey square written into the normal
thumbnail cache, which then satisfied every later lookup. A transient failure
became a permanent one for the session.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


class _Cache:
    def __init__(self, n):
        self.lock = threading.Lock()
        self.cache = {(f"/f/IMG_{i:05d}.CR3", 512): object() for i in range(n)}


class _Host:
    def __init__(self, n):
        self._metadata_cache = {f"/f/IMG_{i:05d}.CR3": {"k": i} for i in range(n)}
        self._thumbnail_cache = _Cache(n)
        self._placeholder_thumb_paths = set()
        self._thumb_fail_counts = {}
        self._requested_thumbnail_paths = set()


def main() -> int:
    from rawviewer_ui.gallery_view import JustifiedGallery as G

    # --- pruning keeps what is still on show, drops the rest ---
    host = _Host(10000)
    keep = {f"/f/IMG_{i:05d}.CR3" for i in range(3000)}
    t = time.perf_counter()
    G._prune_path_caches(host, keep)
    elapsed = (time.perf_counter() - t) * 1000

    check(
        "prune keeps entries for files still present",
        len(host._thumbnail_cache.cache) == 3000,
        f"{len(host._thumbnail_cache.cache)} of 10000 -- 0 means the whole cache was wiped",
    )
    check(
        "prune keeps their metadata too",
        len(host._metadata_cache) == 3000,
        str(len(host._metadata_cache)),
    )
    check(
        "surviving keys are the kept ones",
        all(k[0] in keep for k in host._thumbnail_cache.cache),
    )
    # Generous bound: the point is that it is not seconds.
    check(
        "prune is not O(keys x paths)",
        elapsed < 250,
        f"{elapsed:.1f} ms for 10k keys x 3k paths (was ~27000 ms)",
    )

    # --- a placeholder can be cleared so the tile decodes for real ---
    host = _Host(4)
    path = "/f/IMG_00001.CR3"
    host._placeholder_thumb_paths.add(path)
    host._thumb_fail_counts[path] = 3
    host._requested_thumbnail_paths.add(path)

    n = G.retry_placeholder_thumbnails(host, [path])
    check("retry resets the failed tile", n == 1, str(n))
    check("placeholder forgotten", path not in host._placeholder_thumb_paths)
    check(
        "fail count cleared so retries are available again",
        path not in host._thumb_fail_counts,
    )
    check(
        "request record cleared so the next pass re-requests it",
        path not in host._requested_thumbnail_paths,
    )
    check(
        "the grey square is evicted, not left to satisfy lookups",
        not any(k[0] == path for k in host._thumbnail_cache.cache),
    )
    check(
        "other files are untouched",
        len(host._thumbnail_cache.cache) == 3,
        str(len(host._thumbnail_cache.cache)),
    )

    # --- only placeholders are reset; healthy tiles are never thrown away ---
    host = _Host(4)
    healthy = "/f/IMG_00002.CR3"
    check(
        "a tile that never failed is not reset",
        G.retry_placeholder_thumbnails(host, [healthy]) == 0,
    )
    check("its cache entry survives", any(k[0] == healthy for k in host._thumbnail_cache.cache))

    # --- with no argument, every placeholder is retried ---
    host = _Host(4)
    for i in (0, 1):
        p = f"/f/IMG_{i:05d}.CR3"
        host._placeholder_thumb_paths.add(p)
        host._thumb_fail_counts[p] = 3
    check("retry-all resets every placeholder", G.retry_placeholder_thumbnails(host) == 2)
    check("none left", not host._placeholder_thumb_paths)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
