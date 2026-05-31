#!/bin/bash
# Launch RAWviewer from source (debug / development).
# Repo root: scripts/Launch/shell -> ../../..

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

export RAWVIEWER_VERBOSE_ORIENTATION_LOGS=1
export RAWVIEWER_DEBUG=1
# GPU-accelerated single-image view (QGraphicsView + OpenGL). Override: RAWVIEWER_GPU_VIEW=0
export RAWVIEWER_GPU_VIEW=1
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [ -f "${REPO_ROOT}/rawviewer_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/rawviewer_env/bin/activate"
fi

echo "Launching RAWviewer from ${REPO_ROOT} (GPU view enabled)..."
exec python3 src/main.py "$@"
