#!/bin/bash
# Build RAWviewer Lite on macOS (no semantic/face AI; EXIF+GPS search; higher prefetch).
# Repo root: scripts/Launch/shell -> ../../..

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(grep -E '^VERSION = ' "$REPO_ROOT/build.py" | sed -E 's/.*"([^"]+)".*/\1/')"
VERSION="${VERSION:-2.3.2}"
echo "RAWviewer macOS Lite Build Script (v${VERSION})"
echo "=============================================="
echo ""
echo "Profile: lite — viewing + metadata/GPS gallery search only (no MobileCLIP / face scan)."
echo ""

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "[ERROR] This script is designed for macOS only."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH"
    exit 1
fi

VENV_DIR="$REPO_ROOT/rawviewer_env"
PYTHON_BIN="$VENV_DIR/bin/python3"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Missing venv interpreter: $PYTHON_BIN"
    exit 1
fi

echo "Using virtual environment: $VENV_DIR"

if command -v brew >/dev/null 2>&1; then
    brew list inih &>/dev/null || brew install inih
    brew list gettext &>/dev/null || brew install gettext
fi

"$PYTHON_BIN" -m pip install --upgrade pip

echo "Installing lite core dependencies (GPS/scipy kept; no CoreML/Vision/huggingface)..."
"$PYTHON_BIN" -m pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow psutil numpy scipy qtawesome pyqtgraph reverse-geocoder pycountry requests certifi pyobjc-framework-Cocoa pyobjc-framework-Quartz

if ! "$PYTHON_BIN" -m pip install --upgrade pyexiv2; then
    echo "[ERROR] pyexiv2 install failed."
    echo "  brew install inih gettext"
    exit 1
fi

"$PYTHON_BIN" -m pip uninstall -y sentence-transformers torch torchvision transformers scikit-learn tokenizers safetensors coremltools huggingface-hub pyobjc-framework-CoreML pyobjc-framework-Vision >/dev/null 2>&1 || true

chmod -R u+w build dist 2>/dev/null || true
rm -rf build dist
rm -f *.spec

echo "Building RAWviewer lite..."
export RAWVIEWER_BUILD_PROFILE=lite
if "$PYTHON_BIN" build.py --profile lite; then
    RELEASE_NAME="RAWviewer-v${VERSION}-macOS-lite"
    RELEASE_DIR="dist/${RELEASE_NAME}"
    rm -rf "${RELEASE_DIR}"
    mkdir -p "${RELEASE_DIR}"
    ditto "dist/RAWviewer.app" "${RELEASE_DIR}/RAWviewer.app"
    cp "${REPO_ROOT}/scripts/Launch/shell/install_macos_app.sh" "${RELEASE_DIR}/"
    cp "${REPO_ROOT}/scripts/Launch/shell/remove_macos_quarantine.sh" "${RELEASE_DIR}/"
    cp "${REPO_ROOT}/scripts/Launch/shell/macos_release_readme.txt" "${RELEASE_DIR}/Start Here.txt"
    chmod +x "${RELEASE_DIR}/install_macos_app.sh" "${RELEASE_DIR}/remove_macos_quarantine.sh"
    rm -f "dist/${RELEASE_NAME}.zip"
    ditto -c -k --sequesterRsrc --keepParent "${RELEASE_DIR}" "dist/${RELEASE_NAME}.zip"
    echo ""
    echo "[SUCCESS] Lite build completed!"
    echo "  App:  dist/RAWviewer.app"
    echo "  Zip:  dist/${RELEASE_NAME}.zip"
    echo ""
    echo "Dev run (source):  ./scripts/Launch/shell/launch_dev_lite.sh"
    echo "Install zip:       cd dist/${RELEASE_NAME} && bash install_macos_app.sh"
else
    echo "[ERROR] Lite build failed."
    exit 1
fi
