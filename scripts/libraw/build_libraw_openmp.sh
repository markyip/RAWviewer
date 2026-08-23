#!/bin/bash
# Build LibRaw with OpenMP and swap it into the pixi env's rawpy (macOS arm64).
#
# Why (two jobs, one dylib):
# 1) OpenMP — the PyPI rawpy wheel bundles a single-threaded LibRaw. CR3/CRX,
#    Fuji RAF, Panasonic pana8, AHD/DHT/X-Trans, and postprocess aux stages
#    all parallelise with LIBRAW_USE_OPENMP. Measured on M1 (warm cache, min
#    of 3): half ARW 301->196ms, half CR3 426->195ms; full AHD ARW 1121->697ms.
# 2) New Sony bodies — stock LibRaw 0.22.1 cannot open some ARWs (e.g. A7 V /
#    ILCE-7M5 ARW6). Sources come from third_party/LibRaw (git submodule,
#    master pin with sony_arw6). See scripts/libraw/PIN.txt + README.md.
#
# The dylib keeps the stock rawpy soname libraw_r.25.dylib even when upstream
# SHLIB_CURRENT is 26, so we can swap without rebuilding the Cython extension.
# It is self-contained (@loader_path libomp + libjpeg) for PyInstaller.
#
# Official macOS .app builds EXCLUDE torch (--exclude-module torch in build.py).
# For those packages you MUST use standalone libomp:
#   RAWVIEWER_LIBRAW_OPENMP_STANDALONE=1 bash scripts/libraw/build_libraw_openmp.sh
# (build_macos.sh sets this automatically before packaging.)
#
# Local Full pixi envs that keep torch loaded may unify onto torch's libomp
# (default when STANDALONE is unset) so only one OpenMP runtime loads.
#
# Requires: pixi env with llvm-openmp + libjpeg-turbo, Xcode CLT, autoconf
# (brew install autoconf automake libtool). Re-run after any `pixi install`
# that recreates the env.
#
# Env knobs:
#   RAWVIEWER_LIBRAW_OPENMP_STANDALONE=1  — always ship libomp beside rawpy
#   RAWVIEWER_LIBRAW_OPENMP_SKIP_COMPILE=1 — relink/install only (reuse cache)
#   RAWVIEWER_LIBRAW_FORCE=1              — rebuild even if cache matches pin
#   RAWVIEWER_LIBRAW_SKIP=1               — leave stock LibRaw untouched
#
# Note: built with --disable-lcms (LibRaw's LCMS use is output-profile
# application in postprocess, which this app never enables).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LIBRAW_SRC="$REPO_ROOT/third_party/LibRaw"
PIN_FILE="$REPO_ROOT/scripts/libraw/PIN.txt"
ENV="$REPO_ROOT/.pixi/envs/default"
CACHE_DIR="$REPO_ROOT/.cache/libraw_openmp"
RAWPY_DIR=$(find "$ENV/lib" -path "*/site-packages/rawpy" -maxdepth 5 | head -1)
STANDALONE="${RAWVIEWER_LIBRAW_OPENMP_STANDALONE:-0}"
SKIP_COMPILE="${RAWVIEWER_LIBRAW_OPENMP_SKIP_COMPILE:-0}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

if [ "${RAWVIEWER_LIBRAW_SKIP:-0}" = "1" ]; then
  echo "[libraw] RAWVIEWER_LIBRAW_SKIP=1 — leaving stock LibRaw in place."
  exit 0
fi

[ -f "$ENV/lib/libomp.dylib" ] || { echo "missing libomp: pixi add llvm-openmp"; exit 1; }
[ -f "$ENV/lib/libjpeg.8.dylib" ] || { echo "missing libjpeg: pixi add libjpeg-turbo"; exit 1; }
[ -n "$RAWPY_DIR" ] || { echo "rawpy not found in env"; exit 1; }

_pinned_libraw() {
  if [ -f "$PIN_FILE" ]; then
    sed -n 's/^LibRaw=//p' "$PIN_FILE" | head -1
  fi
}

_ensure_libraw_submodule() {
  if [ ! -f "$LIBRAW_SRC/src/decoders/sony_arw6.cpp" ]; then
    echo "[libraw] Initializing third_party/LibRaw submodule..."
    git -C "$REPO_ROOT" submodule update --init --recursive third_party/LibRaw third_party/LibRaw-cmake
  fi
  if [ ! -f "$LIBRAW_SRC/src/decoders/sony_arw6.cpp" ]; then
    echo "[ERROR] third_party/LibRaw missing sony_arw6.cpp"
    echo "  Run: git submodule update --init --recursive"
    exit 1
  fi
}

# macOS kills (exit 137 / Killed: 9) processes that load dylibs whose ad-hoc
# signature was invalidated by install_name_tool. Always re-sign rawpy's
# native libs before any `import rawpy`, and restore the stock LibRaw if a
# previous OpenMP swap left the env unusable.
_resign_rawpy_dylibs() {
  local f
  shopt -s nullglob
  for f in "$RAWPY_DIR"/*.dylib; do
    if ! codesign -f -s - "$f" >/dev/null 2>&1; then
      echo "[WARNING] codesign failed for $f"
    fi
  done
  shopt -u nullglob
}

_rawpy_import_ok() {
  # Swallow stdout; surface kill/crash via exit status only.
  "$ENV/bin/python3" -c "import rawpy" >/dev/null 2>&1
}

_ensure_rawpy_importable() {
  _resign_rawpy_dylibs
  if _rawpy_import_ok; then
    return 0
  fi
  if [ -f "$RAWPY_DIR/libraw_r.25.dylib.orig" ]; then
    echo "[WARNING] rawpy cannot load (often Killed: 9 after unsigned dylib edits)."
    echo "          Restoring stock LibRaw from libraw_r.25.dylib.orig ..."
    cp -f "$RAWPY_DIR/libraw_r.25.dylib.orig" "$RAWPY_DIR/libraw_r.25.dylib"
    _resign_rawpy_dylibs
  fi
  if ! _rawpy_import_ok; then
    echo "[ERROR] rawpy still cannot import after re-sign/restore."
    echo "  macOS 'Killed: 9' here usually means an invalid code signature on a"
    echo "  dylib under: $RAWPY_DIR"
    echo "  Try: codesign -f -s - $RAWPY_DIR/*.dylib"
    echo "  Or:  pixi install   # recreate env, then re-run this script"
    exit 1
  fi
  echo "[OK] rawpy import restored."
}

_ensure_rawpy_importable

_ensure_libraw_submodule

mkdir -p "$CACHE_DIR"
CACHED_DYLIB="$CACHE_DIR/libraw_r.25.dylib"
CACHED_COMMIT_FILE="$CACHE_DIR/SOURCE_COMMIT.txt"
PINNED="$(_pinned_libraw)"
CACHED_COMMIT=""
[ -f "$CACHED_COMMIT_FILE" ] && CACHED_COMMIT="$(tr -d '[:space:]' < "$CACHED_COMMIT_FILE")"

# Fast path for repeat packaging: standalone OpenMP LibRaw already installed
# AND cache matches the pinned submodule commit.
if [ "$STANDALONE" = "1" ] && [ -f "$CACHED_DYLIB" ] && [ -f "$RAWPY_DIR/libomp.dylib" ] \
   && [ "${RAWVIEWER_LIBRAW_FORCE:-0}" != "1" ] \
   && { [ -z "$PINNED" ] || [ "$PINNED" = "$CACHED_COMMIT" ]; }; then
  if otool -L "$RAWPY_DIR/libraw_r.25.dylib" 2>/dev/null | grep -q '@loader_path/libomp.dylib'; then
    if _rawpy_import_ok; then
      echo "[OK] OpenMP LibRaw already installed (standalone); skipping rebuild."
      exit 0
    fi
  fi
fi

# Sanity: only swap into the LibRaw major.minor rawpy expects.
"$ENV/bin/python3" - << EOF
import rawpy
v = rawpy.libraw_version
assert v[:2] == (0, 22), f"rawpy links LibRaw {v}; update soname / pin in scripts/libraw/"
EOF

build_libraw() {
  # Git checkout has configure.ac but not a generated ./configure.
  command -v autoreconf >/dev/null || {
    echo "[ERROR] autoreconf not found. Install: brew install autoconf automake libtool"
    exit 1
  }

  SRC_COPY="$WORK/LibRaw-src"
  # Copy so autoreconf/configure do not dirty the submodule working tree.
  rsync -a --delete --exclude .git "$LIBRAW_SRC/" "$SRC_COPY/"
  cd "$SRC_COPY"

  echo "[libraw] autoreconf (submodule → $(git -C "$LIBRAW_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown))..."
  autoreconf -fi > "$WORK/autoreconf.log" 2>&1 || {
    echo "autoreconf failed:"; tail -40 "$WORK/autoreconf.log"; exit 1
  }

  export CXX=clang++ CC=clang
  export CXXFLAGS="-O3 -arch arm64 -Xpreprocessor -fopenmp -DLIBRAW_USE_OPENMP -I$ENV/include"
  export CFLAGS="-O3 -arch arm64 -I$ENV/include"
  export LDFLAGS="-L$ENV/lib -lomp -Wl,-rpath,$ENV/lib"
  ./configure --disable-examples --disable-static --disable-lcms --enable-jpeg > configure.log

  # libtool strips -Xpreprocessor and Apple clang rejects bare -fopenmp at link,
  # so `make` fails at the final link step -- compile objects, then link by hand.
  make -j"$(sysctl -n hw.ncpu)" > build.log 2>&1 || true
  find src -path '*/.libs/*.o' > objs.txt
  # Master has more decoder objects than 0.22.1; require a sane floor.
  [ "$(wc -l < objs.txt)" -ge 70 ] || { echo "compile incomplete:"; tail -40 build.log; exit 1; }
  # Must include the ARW6 decoder object.
  grep -q 'sony_arw6' objs.txt || {
    echo "[ERROR] sony_arw6.o missing from build — submodule pin may be wrong."
    cat objs.txt
    exit 1
  }

  # Keep stock rawpy's libraw_r.25.dylib soname (upstream SHLIB_CURRENT may be 26).
  # shellcheck disable=SC2046
  clang++ -dynamiclib -o libraw_r.25.dylib $(tr '\n' ' ' < objs.txt) \
    -L"$ENV/lib" -lomp -ljpeg -lz -lm -O3 -arch arm64 \
    -install_name @loader_path/libraw_r.25.dylib \
    -compatibility_version 25 -current_version 25.0

  cp -f libraw_r.25.dylib "$CACHED_DYLIB"
  if [ -n "$PINNED" ]; then
    printf '%s\n' "$PINNED" > "$CACHED_COMMIT_FILE"
  fi
  echo "Cached OpenMP LibRaw (ARW6) → $CACHED_DYLIB"
}

NEED_COMPILE=1
if [ "${RAWVIEWER_LIBRAW_FORCE:-0}" = "1" ]; then
  NEED_COMPILE=1
elif [ "$SKIP_COMPILE" = "1" ] && [ -f "$CACHED_DYLIB" ]; then
  echo "Reusing cached OpenMP LibRaw: $CACHED_DYLIB"
  NEED_COMPILE=0
elif [ "$SKIP_COMPILE" = "1" ] && [ -f "$RAWPY_DIR/libraw_r.25.dylib.omp" ]; then
  echo "Reusing previous OpenMP build: $RAWPY_DIR/libraw_r.25.dylib.omp"
  cp -f "$RAWPY_DIR/libraw_r.25.dylib.omp" "$CACHED_DYLIB"
  NEED_COMPILE=0
elif [ -f "$CACHED_DYLIB" ] && { [ -z "$PINNED" ] || [ "$PINNED" = "$CACHED_COMMIT" ]; }; then
  echo "Using cached OpenMP LibRaw: $CACHED_DYLIB (pin=$CACHED_COMMIT)"
  NEED_COMPILE=0
fi

if [ "$NEED_COMPILE" = "1" ]; then
  build_libraw
fi

STAGE="$WORK/stage"
mkdir -p "$STAGE"
cp -f "$CACHED_DYLIB" "$STAGE/libraw_r.25.dylib"

# Self-contain: runtime deps live beside the dylib via @loader_path.
cp -f "$ENV/lib/libjpeg.8.dylib" "$RAWPY_DIR/libjpeg.8.dylib"
install_name_tool -id @loader_path/libjpeg.8.dylib "$RAWPY_DIR/libjpeg.8.dylib"
JPEG_DEP=$(otool -L "$STAGE/libraw_r.25.dylib" | awk '/libjpeg/ {print $1; exit}')
if [ -n "$JPEG_DEP" ]; then
  install_name_tool -change "$JPEG_DEP" @loader_path/libjpeg.8.dylib "$STAGE/libraw_r.25.dylib"
fi

# OpenMP runtime: exactly ONE libomp may load per process. Official macOS
# packages exclude torch, so packaging MUST use standalone @loader_path.
# Local Full envs may unify onto torch's copy when STANDALONE is off.
TORCH_OMP=$("$ENV/bin/python3" -c "import os,importlib.util as u; s=u.find_spec('torch'); print(os.path.join(os.path.dirname(s.origin),'lib','libomp.dylib') if s else '')" 2>/dev/null || true)
USE_TORCH_OMP=0
if [ "$STANDALONE" != "1" ] && [ -n "$TORCH_OMP" ] && [ -f "$TORCH_OMP" ]; then
  USE_TORCH_OMP=1
fi

# Clear any previous OpenMP install-name before rewriting.
for dep in $(otool -L "$STAGE/libraw_r.25.dylib" | awk '/libomp/ {print $1}'); do
  install_name_tool -change "$dep" @rpath/libomp.dylib "$STAGE/libraw_r.25.dylib" 2>/dev/null || true
done

if [ "$USE_TORCH_OMP" = "1" ]; then
  echo "Unifying OpenMP runtime with torch: $TORCH_OMP"
  install_name_tool -change @rpath/libomp.dylib @loader_path/../torch/lib/libomp.dylib "$STAGE/libraw_r.25.dylib"
  rm -f "$RAWPY_DIR/libomp.dylib"
else
  echo "Bundling standalone OpenMP runtime beside rawpy (release-safe)"
  cp -f "$ENV/lib/libomp.dylib" "$RAWPY_DIR/libomp.dylib"
  install_name_tool -id @loader_path/libomp.dylib "$RAWPY_DIR/libomp.dylib"
  install_name_tool -change @rpath/libomp.dylib @loader_path/libomp.dylib "$STAGE/libraw_r.25.dylib"
fi

if [ ! -f "$RAWPY_DIR/libraw_r.25.dylib.orig" ]; then
  cp -f "$RAWPY_DIR/libraw_r.25.dylib" "$RAWPY_DIR/libraw_r.25.dylib.orig"
fi
# Keep a copy of the OpenMP-built dylib (pre-install-name finalization) for
# SKIP_COMPILE / packaging retargets.
cp -f "$CACHED_DYLIB" "$RAWPY_DIR/libraw_r.25.dylib.omp"
cp -f "$STAGE/libraw_r.25.dylib" "$RAWPY_DIR/libraw_r.25.dylib"

codesign -f -s - "$RAWPY_DIR/libraw_r.25.dylib" "$RAWPY_DIR/libjpeg.8.dylib" >/dev/null 2>&1 || true
[ -f "$RAWPY_DIR/libomp.dylib" ] && codesign -f -s - "$RAWPY_DIR/libomp.dylib" >/dev/null 2>&1 || true
# Cover sibling sonames macOS may load (unsigned → Killed: 9 on next import).
_resign_rawpy_dylibs

if ! "$ENV/bin/python3" - << 'EOF'
import rawpy
print("rawpy loads OK, libraw", rawpy.libraw_version)
EOF
then
  echo "[ERROR] rawpy failed to import after OpenMP LibRaw install (exit $?)."
  echo "  Re-sign: codesign -f -s - $RAWPY_DIR/*.dylib"
  exit 1
fi

echo "LibRaw OpenMP install names:"
otool -L "$RAWPY_DIR/libraw_r.25.dylib" | sed 's/^/  /'
echo "Done. Restore stock wheel: cp $RAWPY_DIR/libraw_r.25.dylib.orig $RAWPY_DIR/libraw_r.25.dylib"
