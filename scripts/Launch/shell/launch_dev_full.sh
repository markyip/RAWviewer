#!/bin/bash
# Launch RAWviewer from source in full profile (semantic search + face scan).
# Repo root: scripts/Launch/shell -> ../../..

export RAWVIEWER_BUILD_PROFILE=full
export RAWVIEWER_ENABLE_SEMANTIC_SEARCH="${RAWVIEWER_ENABLE_SEMANTIC_SEARCH:-1}"
export RAWVIEWER_ENABLE_FACE_SCAN="${RAWVIEWER_ENABLE_FACE_SCAN:-1}"
export RAWVIEWER_TEST_SEMANTIC="${RAWVIEWER_TEST_SEMANTIC:-1}"
# Dev full profile: opt into perf-v2 (async search metadata, parallel prep).
# Core ML stays serial (chunk=1) unless RAWVIEWER_SEMANTIC_COREML_BATCH=1.
export RAWVIEWER_PERF_V2="${RAWVIEWER_PERF_V2:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/launch_dev.sh" "$@"
