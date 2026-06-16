#!/bin/bash
# Build both Full and Lite macOS release zips in a single run.
#
# Usage:
#   bash scripts/Launch/shell/build_macos_all.sh
#
# Outputs (in dist/):
#   RAWviewer-vX.Y.Z-macOS.zip       — Full edition (AI semantic search + face scan)
#   RAWviewer-vX.Y.Z-macOS-Lite.zip  — Lite edition (EXIF/GPS search only, no AI)
#
# Options (passed straight through to build_macos.sh for each profile):
#   RAWVIEWER_USE_SYSTEM_PYTHON_BUILD=1  — skip project venv, use system python3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

BUILD_SCRIPT="$REPO_ROOT/scripts/Launch/shell/build_macos.sh"

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "[ERROR] This script is designed for macOS only."
    exit 1
fi

if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "[ERROR] build_macos.sh not found at: $BUILD_SCRIPT"
    exit 1
fi

VERSION="$(grep -E '^VERSION = ' "$REPO_ROOT/build.py" | sed -E 's/.*"([^"]+)".*/\1/')"
VERSION="${VERSION:-unknown}"

OVERALL_START=$SECONDS

echo "╔══════════════════════════════════════════════════════╗"
echo "║   RAWviewer macOS — Build All (Full + Lite)  v${VERSION}   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Build Full ────────────────────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────┐"
echo "│  Step 1/2 — Full edition (AI semantic search + face)    │"
echo "└─────────────────────────────────────────────────────────┘"

FULL_START=$SECONDS
if bash "$BUILD_SCRIPT" full "$@"; then
    FULL_ZIP="dist/RAWviewer-v${VERSION}-macOS.zip"
    FULL_ELAPSED=$(( SECONDS - FULL_START ))
    echo ""
    if [ -f "$FULL_ZIP" ]; then
        FULL_SIZE=$(du -sh "$FULL_ZIP" 2>/dev/null | cut -f1)
        echo "[✓] Full build succeeded in ${FULL_ELAPSED}s → $FULL_ZIP ($FULL_SIZE)"
    else
        echo "[✓] Full build succeeded in ${FULL_ELAPSED}s (zip not found — check dist/)"
    fi
else
    echo ""
    echo "[✗] Full build FAILED. Aborting."
    exit 1
fi

echo ""

# ── Build Lite ────────────────────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────┐"
echo "│  Step 2/2 — Lite edition (EXIF/GPS search only, no AI)  │"
echo "└─────────────────────────────────────────────────────────┘"

LITE_START=$SECONDS
if bash "$BUILD_SCRIPT" lite "$@"; then
    LITE_ZIP="dist/RAWviewer-v${VERSION}-macOS-Lite.zip"
    LITE_ELAPSED=$(( SECONDS - LITE_START ))
    echo ""
    if [ -f "$LITE_ZIP" ]; then
        LITE_SIZE=$(du -sh "$LITE_ZIP" 2>/dev/null | cut -f1)
        echo "[✓] Lite build succeeded in ${LITE_ELAPSED}s → $LITE_ZIP ($LITE_SIZE)"
    else
        echo "[✓] Lite build succeeded in ${LITE_ELAPSED}s (zip not found — check dist/)"
    fi
else
    echo ""
    echo "[✗] Lite build FAILED."
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - OVERALL_START ))
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Build complete in ${TOTAL_ELAPSED}s"
echo ""
echo "  Release artifacts:"
for zip in \
    "dist/RAWviewer-v${VERSION}-macOS.zip" \
    "dist/RAWviewer-v${VERSION}-macOS-Lite.zip"
do
    if [ -f "$zip" ]; then
        SIZE=$(du -sh "$zip" 2>/dev/null | cut -f1)
        echo "    ✓  $zip  ($SIZE)"
    else
        echo "    ?  $zip  (not found)"
    fi
done
echo ""
echo "  To install locally:"
echo "    open dist/RAWviewer.app"
echo "    open dist/RAWviewer_Lite.app"
echo "══════════════════════════════════════════════════════════"
