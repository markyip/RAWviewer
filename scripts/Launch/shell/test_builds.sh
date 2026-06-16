#!/bin/bash
# Smoke-test full vs lite profiles and report dist artifacts (macOS).
# Repo root: scripts/Launch/shell -> ../../..

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "RAWviewer build profile smoke test"
echo "=================================="
echo ""

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [ -f "${REPO_ROOT}/rawviewer_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/rawviewer_env/bin/activate"
fi

PYTHON="${PYTHON:-python3}"

echo "[1/4] py_compile changed Python modules..."
"$PYTHON" -m py_compile build.py src/build_profile.py src/rawviewer_profile.py src/bootstrap.py
echo "[OK] py_compile"

echo ""
echo "[2/4] Profile resolution (full / lite)..."
"$PYTHON" - <<'PY'
import importlib
import os
import sys

sys.path.insert(0, "src")
import rawviewer_profile as rp

os.environ["RAWVIEWER_BUILD_PROFILE"] = "full"
importlib.reload(rp)
assert rp.resolved_profile() == "full", rp.resolved_profile()

os.environ["RAWVIEWER_BUILD_PROFILE"] = "lite"
importlib.reload(rp)
assert rp.resolved_profile() == "lite", rp.resolved_profile()

print("[OK] rawviewer_profile.resolved_profile()")
PY

echo ""
echo "[3/4] Dist artifacts (optional — run build scripts first)..."
FOUND=0
for artifact in \
    "dist/RAWviewer.app" \
    "dist/RAWviewer_Lite.app" \
    "dist/RAWviewer-v"*-macOS.zip \
    "dist/RAWviewer-v"*-macOS-Lite.zip
do
    # shellcheck disable=SC2086
    for path in $artifact; do
        if [ -e "$path" ]; then
            echo "  [found] $path"
            FOUND=1
        fi
    done
done
if [ "$FOUND" -eq 0 ]; then
    echo "  (none — build with build_macos_full.sh / build_macos_lite.sh)"
fi

echo ""
echo "[4/4] Dev launch commands:"
echo "  ./scripts/Launch/shell/launch_dev_full.sh"
echo "  ./scripts/Launch/shell/launch_dev_lite.sh"
echo ""
echo "Smoke test passed."
