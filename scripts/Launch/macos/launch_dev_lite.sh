#!/bin/bash
# Launch RAWviewer from source in lite profile (no semantic/face AI).
# Repo root: scripts/Launch/macos -> ../../..

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export REPO_ROOT
cd "$REPO_ROOT"

export RAWVIEWER_BUILD_PROFILE=lite
export RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0
export RAWVIEWER_ENABLE_FACE_SCAN=0
export RAWVIEWER_AUTO_METADATA_INDEX=1
export RAWVIEWER_TEST_SEMANTIC=0
export RAWVIEWER_PREVIEW_CACHE_ADAPTIVE="${RAWVIEWER_PREVIEW_CACHE_ADAPTIVE:-1}"
export RAWVIEWER_MEMORY_PREVIEW_MAX="${RAWVIEWER_MEMORY_PREVIEW_MAX:-2304}"
export RAWVIEWER_GPU_VIEW="${RAWVIEWER_GPU_VIEW:-1}"
# Lite product default: CPU Fast RAW (no torch/kornia in Windows Lite packs).
export RAWVIEWER_PREFER_GPU_DECODE="${RAWVIEWER_PREFER_GPU_DECODE:-0}"
export RAWVIEWER_GPU_CUDA_GL="${RAWVIEWER_GPU_CUDA_GL:-0}"
export RAWVIEWER_DEBUG="${RAWVIEWER_DEBUG:-1}"
export RAWVIEWER_TEST_PYEXIV2="${RAWVIEWER_TEST_PYEXIV2:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Same pixi-first policy as launch_dev.sh (lite profile; no semantic preflight).
PIXI_PYTHON="${REPO_ROOT}/.pixi/envs/default/bin/python"
if command -v pixi >/dev/null 2>&1 && [ -f "${REPO_ROOT}/pixi.toml" ] && [ -x "$PIXI_PYTHON" ]; then
    export PATH="${REPO_ROOT}/.pixi/envs/default/bin:${PATH}"
    echo "[launch_dev_lite] Using pixi env (.pixi/envs/default)"
elif [ -f "${REPO_ROOT}/rawviewer_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/rawviewer_env/bin/activate"
    echo "[launch_dev_lite] Using rawviewer_env (pixi unavailable)"
fi

if [ "${RAWVIEWER_TEST_PYEXIV2}" = "1" ]; then
    echo "Testing pyexiv2 import..."
    if ! python3 - <<'PY'
import pyexiv2
print("pyexiv2 OK:", pyexiv2.__file__)
PY
    then
        echo "[ERROR] pyexiv2 import failed. Run with RAWVIEWER_TEST_PYEXIV2=0 to skip."
        exit 1
    fi
fi

echo "Launching RAWviewer LITE from ${REPO_ROOT} (metadata/GPS search only, semantic off)..."
exec python3 src/main.py "$@"
