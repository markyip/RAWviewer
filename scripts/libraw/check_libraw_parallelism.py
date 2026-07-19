"""
One-command probe: is this environment's LibRaw (via rawpy) decoding with
multiple threads?

Usage (any platform):
    python scripts/libraw/check_libraw_parallelism.py <some.CR3 or other RAW> [more files...]

Interpretation:
- parallelism ~1.0x  -> single-threaded LibRaw (stock macOS/Linux PyPI wheel).
  On macOS run scripts/libraw/build_libraw_openmp.sh to fix (release packages do
  this automatically via build_macos.sh with standalone libomp).
- parallelism >1.5x  -> OpenMP active (Windows PyPI wheels ship this out of
  the box: raw_r.dll links VCOMP140.DLL; verified for rawpy 0.27.0).
  Note: Sony ARW / Nikon NEF unpack itself has little OpenMP work in LibRaw
  0.22 — look at the postprocess ratio for those formats.

Notes: Canon CR3, Fuji RAF, Panasonic show the largest unpack gains; Sony ARW
and Nikon NEF unpack has no OpenMP path in LibRaw 0.22 (only the demosaic and
postprocessing stages parallelize for those). Run on a warm file (second run)
to exclude disk I/O.
"""

import os
import sys
import time

import rawpy


def probe(path: str) -> None:
    with open(path, "rb") as fh:  # warm the OS file cache
        fh.read()
    # unpack stage
    raw = rawpy.imread(path)
    w0, c0 = time.perf_counter(), time.process_time()
    _ = raw.raw_image
    w1, c1 = time.perf_counter(), time.process_time()
    raw.close()
    # full postprocess (AHD demosaic parallelizes too)
    with rawpy.imread(path) as raw:
        w2, c2 = time.perf_counter(), time.process_time()
        raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8, user_flip=0)
        w3, c3 = time.perf_counter(), time.process_time()
    print(
        f"{os.path.basename(path)}: "
        f"unpack {1000*(w1-w0):.0f}ms ({(c1-c0)/max(1e-9, w1-w0):.2f}x cores)  "
        f"postprocess {1000*(w3-w2):.0f}ms ({(c3-c2)/max(1e-9, w3-w2):.2f}x cores)"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(f"rawpy {rawpy.libraw_version=} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}")
    for p in sys.argv[1:]:
        probe(p)
