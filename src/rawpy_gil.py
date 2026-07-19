"""GIL-friendly rawpy open.

``rawpy.imread(path)`` performs its file reads inside LibRaw C code while
holding the GIL — measured 40–160ms per open on external volumes. Gallery
worker threads streaming those opens starve the Qt main thread (event loop
stalls; scrolling reads as choppy even though the work is off-thread).

Full in-memory buffering (rawpy.imread(BytesIO(data))) fixed the GIL hold
(max 3ms measured) but regressed hard: it reads the whole ~60MB file where
LibRaw reads sparsely (~2MB), and keeps the bytes resident per open —
measured 1.4GB RSS + system memory-pressure cache shrink + multi-second
thumbnail queue waits on a 2000-file gallery.

``rawpy_imread_warm`` instead pre-reads a bounded head of the file with
Python I/O (which releases the GIL) so LibRaw's metadata/preview reads hit
the OS page cache; the C-side GIL-held work then becomes fast memcpys. No
buffers are retained, cold-I/O amplification is capped at WARM_BYTES.
"""

import os
import sys


# Embedded previews + metadata almost always live in the first few MB of a
# RAW container. Tune with RAWVIEWER_RAWPY_WARM_MB (0 disables warming).
#
# Default ON only for macOS: the GIL starvation this mitigates was measured
# there (trackpad momentum keeps the event loop visibly busy, external-volume
# opens hold the GIL 40-450ms). On Windows the extra 8MB read per open
# amplified folder-scan I/O enough to stall gallery loading (reported on
# 2026-07-20), so it stays off unless explicitly enabled for testing.
def _warm_bytes() -> int:
    default_mb = "8" if sys.platform == "darwin" else "0"
    try:
        mb = float(os.environ.get("RAWVIEWER_RAWPY_WARM_MB", default_mb) or default_mb)
    except ValueError:
        mb = float(default_mb)
    return max(0, int(mb * 1024 * 1024))


def rawpy_imread_warm(file_path):
    """rawpy.imread with a GIL-releasing page-cache warm of the file head."""
    import rawpy

    limit = _warm_bytes()
    if limit > 0:
        try:
            with open(file_path, "rb", buffering=0) as f:
                chunk = 4 * 1024 * 1024
                remaining = limit
                while remaining > 0:
                    got = f.read(min(chunk, remaining))
                    if not got:
                        break
                    remaining -= len(got)
        except OSError:
            pass
    return rawpy.imread(file_path)
