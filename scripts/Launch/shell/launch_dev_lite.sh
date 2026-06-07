#!/bin/bash
# Launch RAWviewer from source in lite profile (no semantic/face AI; higher prefetch).
# Repo root: scripts/Launch/shell -> ../../..

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export REPO_ROOT
cd "$REPO_ROOT"

LAUNCH_PY_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == *=* ]] && [[ "$arg" != /* ]] && [[ "$arg" != ./* ]]; then
        export "$arg"
        echo "[launch_dev_lite] Applied from argument: $arg"
    else
        LAUNCH_PY_ARGS+=("$arg")
    fi
done
if ((${#LAUNCH_PY_ARGS[@]} > 0)); then
    set -- "${LAUNCH_PY_ARGS[@]}"
else
    set --
fi

export RAWVIEWER_BUILD_PROFILE=lite
export RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0
export RAWVIEWER_ENABLE_FACE_SCAN=0
export RAWVIEWER_AUTO_METADATA_INDEX=1
export RAWVIEWER_TEST_SEMANTIC=0
export RAWVIEWER_GPU_VIEW="${RAWVIEWER_GPU_VIEW:-1}"
export RAWVIEWER_DEBUG="${RAWVIEWER_DEBUG:-1}"
export RAWVIEWER_TEST_PYEXIV2="${RAWVIEWER_TEST_PYEXIV2:-1}"
export RAWVIEWER_SHARE_MENU="${RAWVIEWER_SHARE_MENU:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Lite prefetch (NAV radius is RAM-adaptive unless RAWVIEWER_NAV_PRELOAD_RADIUS is set)
export RAWVIEWER_NAV_PRELOAD_ADAPTIVE="${RAWVIEWER_NAV_PRELOAD_ADAPTIVE:-1}"
export RAWVIEWER_NAV_PRELOAD_NEAR="${RAWVIEWER_NAV_PRELOAD_NEAR:-4}"
export RAWVIEWER_IDLE_DISPLAY_PREFETCH_RADIUS="${RAWVIEWER_IDLE_DISPLAY_PREFETCH_RADIUS:-0}"
export RAWVIEWER_IDLE_DISPLAY_PREFETCH_BATCH="${RAWVIEWER_IDLE_DISPLAY_PREFETCH_BATCH:-4}"
export RAWVIEWER_IDLE_DISPLAY_PREFETCH_QUEUE_CAP="${RAWVIEWER_IDLE_DISPLAY_PREFETCH_QUEUE_CAP:-16}"
export RAWVIEWER_IDLE_DISPLAY_PREFETCH_DELAY_MS="${RAWVIEWER_IDLE_DISPLAY_PREFETCH_DELAY_MS:-400}"
export RAWVIEWER_IDLE_DISPLAY_PREFETCH_INTERVAL_MS="${RAWVIEWER_IDLE_DISPLAY_PREFETCH_INTERVAL_MS:-300}"
export RAWVIEWER_FILMSTRIP_PREFETCH_RADIUS="${RAWVIEWER_FILMSTRIP_PREFETCH_RADIUS:-36}"
export RAWVIEWER_PRELOAD_THREADS="${RAWVIEWER_PRELOAD_THREADS:-12}"
export RAWVIEWER_ADJACENT_PRELOAD_NEXT="${RAWVIEWER_ADJACENT_PRELOAD_NEXT:-8}"
export RAWVIEWER_ADJACENT_PRELOAD_PREV="${RAWVIEWER_ADJACENT_PRELOAD_PREV:-6}"
export RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH="${RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH:-120}"
export RAWVIEWER_GALLERY_IDLE_PRELOAD_MS="${RAWVIEWER_GALLERY_IDLE_PRELOAD_MS:-180}"
export RAWVIEWER_GALLERY_IDLE_PRELOAD_PRIORITY="${RAWVIEWER_GALLERY_IDLE_PRELOAD_PRIORITY:-preload_next}"

if [ -f "${REPO_ROOT}/rawviewer_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/rawviewer_env/bin/activate"
fi

if [ "${RAWVIEWER_TEST_PYEXIV2}" = "1" ]; then
    echo "Testing pyexiv2 import..."
    python3 - <<'PY'
import pyexiv2
print("pyexiv2 OK:", pyexiv2.__file__)
PY
fi

echo "Launching RAWviewer LITE from ${REPO_ROOT} (metadata/GPS search only, semantic off)..."
exec python3 src/main.py "$@"
