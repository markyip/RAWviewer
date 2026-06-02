#!/bin/bash
# Launch RAWviewer from source (debug / development).
# Repo root: scripts/Launch/shell -> ../../..

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export REPO_ROOT
cd "$REPO_ROOT"

COLD_START_FLAG="${REPO_ROOT}/.rawviewer_cold_start"
if [ -f "$COLD_START_FLAG" ]; then
    export RAWVIEWER_DISABLE_SESSION_RESTORE=1
    rm -f "$COLD_START_FLAG"
    echo "[launch_dev] Cold start: session restore disabled for this launch (after clear_cache.sh)."
fi

pause_if_interactive() {
    if [ -t 0 ]; then
        read -r -p "Press Enter to close this session..."
    fi
}

export RAWVIEWER_VERBOSE_ORIENTATION_LOGS=1
export RAWVIEWER_DEBUG=1
# File picker: macOS uses Finder via AppleScript/NSOpenPanel. Qt UI: RAWVIEWER_QT_FILE_DIALOG=1
# Semantic search (CLIP embeddings). Off by default in app; enable for dev like run_debug.bat.
export RAWVIEWER_ENABLE_SEMANTIC_SEARCH="${RAWVIEWER_ENABLE_SEMANTIC_SEARCH:-1}"
export RAWVIEWER_AUTO_METADATA_INDEX="${RAWVIEWER_AUTO_METADATA_INDEX:-1}"
# Semantic index speed (macOS Core ML): parallel thumb warm + encode chunks of 8
# export RAWVIEWER_SEMANTIC_WARM_THUMBS=1
# export RAWVIEWER_SEMANTIC_COREML_CHUNK=16
# export RAWVIEWER_SEMANTIC_PREP_WORKERS=8
# export RAWVIEWER_COREML_COMPUTE_UNITS=all
# GPU-accelerated single-image view (QGraphicsView + OpenGL). Override: RAWVIEWER_GPU_VIEW=0
export RAWVIEWER_GPU_VIEW=1
# pyexiv2 check before app launch. Override: RAWVIEWER_TEST_PYEXIV2=0
export RAWVIEWER_TEST_PYEXIV2="${RAWVIEWER_TEST_PYEXIV2:-1}"
# Semantic backend check before app launch. Override: RAWVIEWER_TEST_SEMANTIC=0
export RAWVIEWER_TEST_SEMANTIC="${RAWVIEWER_TEST_SEMANTIC:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [ -f "${REPO_ROOT}/rawviewer_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/rawviewer_env/bin/activate"
fi

if [ "${RAWVIEWER_TEST_PYEXIV2}" = "1" ]; then
    echo "Testing pyexiv2 import..."
    if ! python3 - <<'PY'
import pyexiv2
print("pyexiv2 OK:", pyexiv2.__file__)
PY
    then
        echo "[ERROR] pyexiv2 import failed. Run with RAWVIEWER_TEST_PYEXIV2=0 to skip this check."
        pause_if_interactive
        exit 1
    fi
fi

if [ "${RAWVIEWER_TEST_SEMANTIC}" = "1" ]; then
    echo "Testing semantic search backend..."
    if ! python3 - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["REPO_ROOT"] + "/src")
from semantic_search import resolve_mobileclip_backend, semantic_embeddings_enabled

if not semantic_embeddings_enabled():
    raise SystemExit(
        "RAWVIEWER_ENABLE_SEMANTIC_SEARCH is off (set to 1 for semantic search)"
    )

backend = resolve_mobileclip_backend()
err = backend.availability_error()
if err:
    raise SystemExit(err)

model_id = getattr(backend, "MODEL_ID", type(backend).__name__)
print(f"Semantic backend OK: {type(backend).__name__} ({model_id})")
print(f"Model dir: {getattr(backend, 'model_dir', '(n/a)')}")
PY
    then
        echo "[ERROR] Semantic search backend is not ready."
        echo "  macOS: ensure models/mobileclip2_coreml has image/text .mlpackage + bpe_simple_vocab_16e6.txt.gz"
        echo "  Or download via the in-app MobileCLIP prompt / huggingface_hub."
        echo "  Skip this check: RAWVIEWER_TEST_SEMANTIC=0"
        pause_if_interactive
        exit 1
    fi
fi

echo "Launching RAWviewer from ${REPO_ROOT} (GPU view enabled, semantic search on)..."
if ! python3 src/main.py "$@"; then
    echo "[ERROR] RAWviewer exited with an error."
    pause_if_interactive
    exit 1
fi
