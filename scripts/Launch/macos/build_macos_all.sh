#!/bin/bash
# Build both Full and Lite macOS release zips in a single run.
#
# Usage:
#   bash scripts/Launch/macos/build_macos_all.sh
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

BUILD_SCRIPT="$REPO_ROOT/scripts/Launch/macos/build_macos.sh"

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

# Temp stash — build_macos.sh always does `rm -rf dist` at the start, so we
# must rescue the full-build artifacts before invoking the lite build.
STASH_DIR="$(mktemp -d /tmp/rawviewer_build_stash.XXXXXX)"
cleanup_stash() { rm -rf "$STASH_DIR"; }
trap cleanup_stash EXIT

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
    FULL_APP="dist/RAWviewer.app"
    FULL_ELAPSED=$(( SECONDS - FULL_START ))
    echo ""
    if [ -f "$FULL_ZIP" ]; then
        FULL_SIZE=$(du -sh "$FULL_ZIP" 2>/dev/null | cut -f1)
        echo "[✓] Full build succeeded in ${FULL_ELAPSED}s → $FULL_ZIP ($FULL_SIZE)"

        # Stash the full zip (and app bundle) — the lite build will wipe dist/
        echo "[→] Stashing full artifacts to $STASH_DIR ..."
        cp "$FULL_ZIP" "$STASH_DIR/"
        if [ -d "$FULL_APP" ]; then
            ditto "$FULL_APP" "$STASH_DIR/RAWviewer.app"
        fi
        echo "[→] Stash complete."
    else
        echo "[✓] Full build succeeded in ${FULL_ELAPSED}s (zip not found — check dist/)"
    fi
else
    full_rc=$?
    echo ""
    echo "[✗] Full build FAILED (exit $full_rc). Aborting."
    echo "    Tip: if you saw 'Killed: 9' during OpenMP LibRaw, re-sign rawpy dylibs:"
    echo "      codesign -f -s - .pixi/envs/default/lib/python3.12/site-packages/rawpy/*.dylib"
    echo "    Then re-run: bash scripts/Launch/macos/build_macos_all.sh"
    exit 1
fi

echo ""

# ── Build Lite ────────────────────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────────────────┐"
echo "│  Step 2/2 — Lite edition (EXIF/GPS search only, no AI)  │"
echo "└─────────────────────────────────────────────────────────┘"
echo "[i] Note: build_macos.sh will clean dist/ — full artifacts are safely stashed."
echo ""

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

# ── Restore full artifacts into dist/ ─────────────────────────────────────────
STASHED_ZIP="$STASH_DIR/RAWviewer-v${VERSION}-macOS.zip"
STASHED_APP="$STASH_DIR/RAWviewer.app"

if [ -f "$STASHED_ZIP" ]; then
    echo ""
    echo "[→] Restoring full artifacts from stash into dist/ ..."
    mkdir -p dist
    cp "$STASHED_ZIP" "dist/RAWviewer-v${VERSION}-macOS.zip"
    if [ -d "$STASHED_APP" ]; then
        rm -rf "dist/RAWviewer.app"
        ditto "$STASHED_APP" "dist/RAWviewer.app"
    fi
    echo "[→] Restore complete."
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
