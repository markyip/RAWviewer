#!/bin/bash
# Launch RAWviewer from source (debug / development).
# Repo root: scripts/Launch/macos -> ../../..

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export REPO_ROOT
cd "$REPO_ROOT"

# Allow `launch_dev.sh RAWVIEWER_GPU_VIEW=0` — env vars must be *before* bash, not argv.
LAUNCH_PY_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == *=* ]] && [[ "$arg" != /* ]] && [[ "$arg" != ./* ]]; then
        export "$arg"
        echo "[launch_dev] Applied from argument: $arg"
    else
        LAUNCH_PY_ARGS+=("$arg")
    fi
done
if ((${#LAUNCH_PY_ARGS[@]} > 0)); then
    set -- "${LAUNCH_PY_ARGS[@]}"
else
    set --
fi

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
# Core ML compute units (optional):
#   RAWVIEWER_COREML_COMPUTE_UNITS       — both index + search (unless split defaults apply)
#   RAWVIEWER_COREML_COMPUTE_UNITS_INDEX — image encoder / indexing only
#   RAWVIEWER_COREML_COMPUTE_UNITS_SEARCH — text encoder / semantic search only
# On macOS 26+ beta the app defaults to index=all, search=cpu (MPSGraph text crash).
export RAWVIEWER_DEBUG=1
# macOS NSSharingServicePicker is AppKit; pointer on the sheet has no Qt window (harmless).
# Default share UI is a Qt menu of NSSharingService targets (v2.2 Qt6). Native popover is opt-in.
export QT_LOGGING_RULES="${QT_LOGGING_RULES:-qt.pointer.dispatch=false}"
# File picker: macOS uses Finder via AppleScript/NSOpenPanel. Qt UI: RAWVIEWER_QT_FILE_DIALOG=1
# Semantic search (CLIP embeddings). Off by default in app; enable for dev like run_debug.bat.
export RAWVIEWER_ENABLE_SEMANTIC_SEARCH="${RAWVIEWER_ENABLE_SEMANTIC_SEARCH:-1}"
export RAWVIEWER_ENABLE_FACE_SCAN="${RAWVIEWER_ENABLE_FACE_SCAN:-1}"
export RAWVIEWER_AUTO_METADATA_INDEX="${RAWVIEWER_AUTO_METADATA_INDEX:-1}"
# Match packaged Plus (pyi_rth_release_defaults) and launch_dev_full.sh.
export RAWVIEWER_PERF_V2="${RAWVIEWER_PERF_V2:-1}"
# Plus vs Standard: use launch_dev_full.sh when comparing to a Plus .app
# (sets RAWVIEWER_BUILD_PROFILE=full). Gallery scroll prefetch is shared.
# Semantic index: auto-tune batch/chunk (Core ML + ONNX). RAM tier auto: RAWVIEWER_MEMORY_TIER_AUTO=1 (default)
# export RAWVIEWER_MEMORY_TIER_AUTO=0
# export RAWVIEWER_SEMANTIC_BATCH_AUTO=0
# export RAWVIEWER_SEMANTIC_BATCH_CANDIDATES=8,16,32,64
# export RAWVIEWER_SEMANTIC_PREP_WORKERS=8
# GPU-accelerated single-image view (QGraphicsView + OpenGL). Default on for dev.
# Opt out: RAWVIEWER_GPU_VIEW=0
export RAWVIEWER_GPU_VIEW="${RAWVIEWER_GPU_VIEW:-1}"
# Prefer PyTorch CUDA demosaic + CUDA↔GL display for debug launches.
# Opt out: RAWVIEWER_PREFER_GPU_DECODE=0 / RAWVIEWER_GPU_CUDA_GL=0
# macOS default off: MPS demosaic serializes heavy RAW decodes (raw_limit=1),
# which starves gallery tile fill through the shared load pool.
export RAWVIEWER_PREFER_GPU_DECODE="${RAWVIEWER_PREFER_GPU_DECODE:-0}"
export RAWVIEWER_GPU_CUDA_GL="${RAWVIEWER_GPU_CUDA_GL:-1}"
# macOS share: Qt menu with NSSharingService targets (works in v2.2 Qt6). Popover often spins empty.
export RAWVIEWER_SHARE_MENU="${RAWVIEWER_SHARE_MENU:-1}"
# AirDrop hidden from share menu by default (RAWVIEWER_SHARE_SHOW_AIRDROP=1 to show; PERFORM=1 for in-app test)
# Try native popover (+ auto menu ~900ms): RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1
# Heavy picker (filter pause + delegate): RAWVIEWER_SHARE_HEAVY_PICKER=1
# Brute-force NSApplication foreground (file dialogs from Terminal). Default off.
# export RAWVIEWER_MACOS_FORCE_FOREGROUND=1
# pyexiv2 check before app launch. Override: RAWVIEWER_TEST_PYEXIV2=0
export RAWVIEWER_TEST_PYEXIV2="${RAWVIEWER_TEST_PYEXIV2:-1}"
# AppKit (pyobjc-framework-Cocoa) required for macOS Share + NSOpenPanel. Skip: RAWVIEWER_TEST_APPKIT=0
export RAWVIEWER_TEST_APPKIT="${RAWVIEWER_TEST_APPKIT:-1}"
# Semantic backend check before app launch. Override: RAWVIEWER_TEST_SEMANTIC=0
export RAWVIEWER_TEST_SEMANTIC="${RAWVIEWER_TEST_SEMANTIC:-1}"
# GPU Raw Processor check before app launch. Override: RAWVIEWER_TEST_GPU_RAW=0
export RAWVIEWER_TEST_GPU_RAW="${RAWVIEWER_TEST_GPU_RAW:-0}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Prefer pixi (.pixi/envs/default): includes PyObjC CoreML needed for full-profile semantic search on macOS.
# rawviewer_env from build_macos.sh may lack CoreML; only fall back when pixi is unavailable.
PIXI_PYTHON="${REPO_ROOT}/.pixi/envs/default/bin/python"
if command -v pixi >/dev/null 2>&1 && [ -f "${REPO_ROOT}/pixi.toml" ] && [ -x "$PIXI_PYTHON" ]; then
    export PATH="${REPO_ROOT}/.pixi/envs/default/bin:${PATH}"
    echo "[launch_dev] Using pixi env (.pixi/envs/default)"
elif [ -f "${REPO_ROOT}/rawviewer_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/rawviewer_env/bin/activate"
    echo "[launch_dev] Using rawviewer_env (pixi unavailable)"
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

if [ "$(uname -s)" = "Darwin" ] && [ "${RAWVIEWER_TEST_APPKIT}" = "1" ]; then
    echo "Testing AppKit (pyobjc-framework-Cocoa) for macOS Share..."
    if ! python3 - <<'PY'
from AppKit import NSSharingService, NSURL
print("AppKit OK:", NSSharingService)
PY
    then
        echo "[ERROR] AppKit import failed — Share and macOS file dialogs will not work."
        echo "  pixi:  cd \"${REPO_ROOT}\" && pixi install"
        echo "  venv:  pip install pyobjc-framework-Cocoa pyobjc-framework-Quartz"
        echo "  Skip:  RAWVIEWER_TEST_APPKIT=0"
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

if [ "${RAWVIEWER_TEST_GPU_RAW}" = "1" ]; then
    echo "Testing GPU RAW processor dependencies..."
    if ! python3 - <<'PY'
import sys
try:
    import torch
    import kornia
    print(f"GPU RAW backend OK: torch {torch.__version__}, kornia {kornia.__version__}")
except ImportError:
    sys.exit(1)
PY
    then
        echo "[WARNING] GPU RAW processor dependencies (torch, kornia) are not installed."
        echo "  The app will fall back to CPU decode seamlessly."
        echo "  To test GPU raw decode:"
        echo "  pixi:  pixi add --pypi torch kornia"
        echo "  venv:  pip install torch kornia"
        echo "  Skip this check: RAWVIEWER_TEST_GPU_RAW=0"
        echo "  Continuing in 3 seconds..."
        sleep 3
    fi
fi

_gpu_label="off"
if [ "${RAWVIEWER_GPU_VIEW}" = "1" ]; then
    _gpu_label="on"
fi
_gpu_decode_label="off"
if [ "${RAWVIEWER_PREFER_GPU_DECODE}" = "1" ]; then
    _gpu_decode_label="on"
fi
_cuda_gl_label="off"
if [ "${RAWVIEWER_GPU_CUDA_GL}" = "1" ]; then
    _cuda_gl_label="on"
fi
_share_label="menu"
if [ "${RAWVIEWER_SHARE_TRY_NATIVE_PICKER:-}" = "1" ] || [ "${RAWVIEWER_SHARE_NATIVE_PICKER:-}" = "1" ]; then
    _share_label="picker+menu-fallback"
fi
echo "Launching RAWviewer from ${REPO_ROOT} (GPU view ${_gpu_label}, demosaic ${_gpu_decode_label}, CUDA-GL ${_cuda_gl_label}, share ${_share_label}, semantic ${RAWVIEWER_ENABLE_SEMANTIC_SEARCH:-1})..."
set +e
python3 src/main.py "$@"
_launch_ec=$?
set -e
if [ "$_launch_ec" -ne 0 ]; then
    if [ "$_launch_ec" -eq 137 ] || [ "$_launch_ec" -eq 9 ]; then
        echo "[ERROR] RAWviewer was killed by macOS (SIGKILL, exit ${_launch_ec})."
        echo "  This usually means the system ran out of memory — common when:"
        echo "    • Restoring a large folder + semantic / metadata indexing"
        echo "    • Core ML embedding + gallery prefetch on a 16 GB Mac"
        echo "  Try one or more of these, then relaunch:"
        echo "    RAWVIEWER_DISABLE_SESSION_RESTORE=1 ./scripts/Launch/macos/launch_dev.sh"
        echo "    RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0 ./scripts/Launch/macos/launch_dev.sh"
        echo "    RAWVIEWER_AUTO_METADATA_INDEX=0 ./scripts/Launch/macos/launch_dev.sh"
        echo "    RAWVIEWER_GPU_VIEW=0 ./scripts/Launch/macos/launch_dev.sh"
        echo "  Or clear caches: ./scripts/Launch/macos/clear_cache.sh"
    else
        echo "[ERROR] RAWviewer exited with an error (code ${_launch_ec})."
    fi
    pause_if_interactive
    exit "$_launch_ec"
fi
