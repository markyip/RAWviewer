#!/bin/bash
# Build LibRaw with OpenMP and swap it into the pixi env's rawpy (macOS arm64).
#
# Why: the PyPI rawpy wheel bundles a single-threaded LibRaw. LibRaw's CR3
# (CRX), Fuji RAF, Panasonic pana8 decoders, AHD/DHT/X-Trans demosaic, and
# raw2image/postprocessing aux stages are all OpenMP-parallel when built with
# LIBRAW_USE_OPENMP. Measured on M1 (warm cache, min of 3):
#   half-size decode: ARW 33MP 301->196ms, CR3 24MP 426->195ms
#   full AHD decode:  ARW 1121->697ms, CR3 24MP 1010->531ms, CR3 45MP 1845->1039ms
# Output verified byte-identical to the stock wheel on ARW+CR3 (half+full),
# deterministic under concurrent decodes; adjust-pipeline + parity suites pass.
#
# The produced dylib is self-contained (@loader_path libomp + libjpeg copied
# beside it) so PyInstaller bundles pick everything up by dependency tracing.
#
# Requires: pixi env with llvm-openmp + libjpeg-turbo (pixi add llvm-openmp
# libjpeg-turbo), Xcode CLT. Re-run after any `pixi install` that recreates
# the env (the swap lives inside .pixi and is not tracked by the lockfile).
#
# Note: built with --disable-lcms (LibRaw's LCMS use is output-profile
# application in postprocess, which this app never enables).
set -euo pipefail

LIBRAW_VERSION="0.22.1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$REPO_ROOT/.pixi/envs/default"
RAWPY_DIR=$(find "$ENV/lib" -path "*/site-packages/rawpy" -maxdepth 5 | head -1)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

[ -f "$ENV/lib/libomp.dylib" ] || { echo "missing libomp: pixi add llvm-openmp"; exit 1; }
[ -f "$ENV/lib/libjpeg.8.dylib" ] || { echo "missing libjpeg: pixi add libjpeg-turbo"; exit 1; }
[ -n "$RAWPY_DIR" ] || { echo "rawpy not found in env"; exit 1; }

# Sanity: only swap into the LibRaw version rawpy expects.
"$ENV/bin/python3" - << EOF
import rawpy
v = rawpy.libraw_version
assert v[:2] == (0, 22), f"rawpy links LibRaw {v}; update LIBRAW_VERSION + soname in this script"
EOF

cd "$WORK"
curl -sLO "https://www.libraw.org/data/LibRaw-${LIBRAW_VERSION}.tar.gz"
tar xf "LibRaw-${LIBRAW_VERSION}.tar.gz"
cd "LibRaw-${LIBRAW_VERSION}"

export CXX=clang++ CC=clang
export CXXFLAGS="-O3 -arch arm64 -Xpreprocessor -fopenmp -DLIBRAW_USE_OPENMP -I$ENV/include"
export CFLAGS="-O3 -arch arm64 -I$ENV/include"
export LDFLAGS="-L$ENV/lib -lomp -Wl,-rpath,$ENV/lib"
./configure --disable-examples --disable-static --disable-lcms --enable-jpeg > configure.log

# libtool strips -Xpreprocessor and Apple clang rejects bare -fopenmp at link,
# so `make` fails at the final link step -- compile objects, then link by hand.
make -j"$(sysctl -n hw.ncpu)" > build.log 2>&1 || true
find src -path '*/.libs/*.o' > objs.txt
[ "$(wc -l < objs.txt)" -ge 70 ] || { echo "compile incomplete:"; tail -20 build.log; exit 1; }

# shellcheck disable=SC2046
clang++ -dynamiclib -o libraw_r.25.dylib $(tr '\n' ' ' < objs.txt) \
  -L"$ENV/lib" -lomp -ljpeg -lz -lm -O3 -arch arm64 \
  -install_name @loader_path/libraw_r.25.dylib \
  -compatibility_version 25 -current_version 25.0

# Self-contain: runtime deps live beside the dylib via @loader_path.
cp "$ENV/lib/libjpeg.8.dylib" "$RAWPY_DIR/libjpeg.8.dylib"
install_name_tool -id @loader_path/libjpeg.8.dylib "$RAWPY_DIR/libjpeg.8.dylib"
JPEG_DEP=$(otool -L libraw_r.25.dylib | awk '/libjpeg/ {print $1}')
install_name_tool -change "$JPEG_DEP" @loader_path/libjpeg.8.dylib libraw_r.25.dylib

# OpenMP runtime: exactly ONE libomp may load per process. When torch is in
# the env it force-loads torch/lib/libomp.dylib at startup (main.py splash
# pre-import), so point LibRaw at torch's copy -- two loaded LLVM OpenMP
# runtimes abort ("OMP: Error #15"), and KMP_DUPLICATE_LIB_OK=TRUE papers
# over it at the documented risk of crashes/silently wrong results (both
# observed: segfaults in dev runs). Without torch, fall back to a bundled
# copy beside the dylib as before.
TORCH_OMP=$("$ENV/bin/python3" -c "import os,importlib.util as u; s=u.find_spec('torch'); print(os.path.join(os.path.dirname(s.origin),'lib','libomp.dylib') if s else '')" 2>/dev/null)
if [ -n "$TORCH_OMP" ] && [ -f "$TORCH_OMP" ]; then
  echo "Unifying OpenMP runtime with torch: $TORCH_OMP"
  install_name_tool -change @rpath/libomp.dylib @loader_path/../torch/lib/libomp.dylib libraw_r.25.dylib
  rm -f "$RAWPY_DIR/libomp.dylib"
else
  cp "$ENV/lib/libomp.dylib" "$RAWPY_DIR/libomp.dylib"
  install_name_tool -id @loader_path/libomp.dylib "$RAWPY_DIR/libomp.dylib"
  install_name_tool -change @rpath/libomp.dylib @loader_path/libomp.dylib libraw_r.25.dylib
fi

if [ ! -f "$RAWPY_DIR/libraw_r.25.dylib.orig" ]; then
  cp "$RAWPY_DIR/libraw_r.25.dylib" "$RAWPY_DIR/libraw_r.25.dylib.orig"
fi
cp libraw_r.25.dylib "$RAWPY_DIR/libraw_r.25.dylib"
codesign -f -s - "$RAWPY_DIR/libraw_r.25.dylib" "$RAWPY_DIR/libjpeg.8.dylib"
[ -f "$RAWPY_DIR/libomp.dylib" ] && codesign -f -s - "$RAWPY_DIR/libomp.dylib"

"$ENV/bin/python3" - << 'EOF'
import time, rawpy
print("rawpy loads OK, libraw", rawpy.libraw_version)
EOF
echo "Done. Restore stock wheel: cp $RAWPY_DIR/libraw_r.25.dylib.orig $RAWPY_DIR/libraw_r.25.dylib"
