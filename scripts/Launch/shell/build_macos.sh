#!/bin/bash
# Build RAWviewer on macOS.
# Repo root: scripts/Launch/shell -> ../../..

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(grep -E '^VERSION = ' "$REPO_ROOT/build.py" | sed -E 's/.*"([^"]+)".*/\1/')"
VERSION="${VERSION:-2.3.0}"
echo "RAWviewer macOS Build Script (v${VERSION})"
echo "======================================"
echo ""

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "[ERROR] This script is designed for macOS only."
    echo "Current OS: $OSTYPE"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH"
    echo "Please install Python 3.10 or higher from https://www.python.org/"
    exit 1
fi

if [ ! -f "icons/appicon.icns" ]; then
    echo "[WARNING] Icon file not found: icons/appicon.icns"
    echo "The app will be built without a custom icon."
fi

VENV_DIR="$REPO_ROOT/rawviewer_env"
PYTHON_BIN="$VENV_DIR/bin/python3"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Missing venv interpreter: $PYTHON_BIN"
    echo "Remove the broken folder and re-run: rm -rf rawviewer_env"
    exit 1
fi

echo "Using virtual environment: $VENV_DIR"

if command -v brew >/dev/null 2>&1; then
    echo "Checking Homebrew dependencies for pyexiv2 (inih, gettext)..."
    brew list inih &>/dev/null || brew install inih
    brew list gettext &>/dev/null || brew install gettext
else
    echo "[INFO] brew not on PATH. If the build fails on pyexiv2: install Homebrew, then: brew install inih gettext"
fi

echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip

echo "Installing core dependencies..."
"$PYTHON_BIN" -m pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow psutil numpy scipy qtawesome pyqtgraph reverse-geocoder pycountry huggingface-hub requests pyobjc-framework-Cocoa pyobjc-framework-CoreML pyobjc-framework-Quartz pyobjc-framework-Vision

echo "Installing required dependency: pyexiv2..."
if ! "$PYTHON_BIN" -m pip install --upgrade pyexiv2; then
    echo "[ERROR] pyexiv2 install failed (required for macOS release builds)."
    echo "  Install native libraries, then re-run:"
    echo "    brew install inih gettext"
    exit 1
fi
echo "[INFO] pyexiv2 installed (Exiv2 / focus-point path enabled)."

"$PYTHON_BIN" -m pip uninstall -y sentence-transformers torch torchvision transformers scikit-learn tokenizers safetensors coremltools >/dev/null 2>&1 || true

echo "Cleaning previous builds..."
chmod -R u+w build dist 2>/dev/null || true
rm -rf build || true
chmod -R u+w dist 2>/dev/null || true
rm -rf dist || true
rm -f *.spec

echo "Building RAWviewer (GPU single-image viewport enabled by default)..."
if "$PYTHON_BIN" build.py; then
    echo ""
    echo "[SUCCESS] Build completed!"
    echo ""

    if [ -d "dist/RAWviewer.app" ]; then
        echo "macOS App Bundle created: dist/RAWviewer.app (v${VERSION})"
        echo ""
        echo "Packaging release zip (app + one-click installer)..."
        RELEASE_NAME="RAWviewer-v${VERSION}-macOS"
        RELEASE_DIR="dist/${RELEASE_NAME}"
        rm -rf "${RELEASE_DIR}"
        mkdir -p "${RELEASE_DIR}"
        ditto "dist/RAWviewer.app" "${RELEASE_DIR}/RAWviewer.app"
        cp "${REPO_ROOT}/scripts/Launch/shell/install_macos_app.sh" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/Install RAWviewer.command" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/remove_macos_quarantine.sh" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/Remove Quarantine.command" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/macos_release_readme.txt" "${RELEASE_DIR}/Start Here.txt"
        chmod +x "${RELEASE_DIR}/install_macos_app.sh" "${RELEASE_DIR}/Install RAWviewer.command"
        chmod +x "${RELEASE_DIR}/remove_macos_quarantine.sh" "${RELEASE_DIR}/Remove Quarantine.command"
        rm -f "dist/${RELEASE_NAME}.zip"
        ditto -c -k --sequesterRsrc --keepParent "${RELEASE_DIR}" "dist/${RELEASE_NAME}.zip"
        echo "Release zip: dist/${RELEASE_NAME}.zip"
        echo ""
        echo "Smoke test (release checklist):"
        echo "  unzip -q dist/${RELEASE_NAME}.zip -d /tmp && open \"/tmp/${RELEASE_NAME}\""
        echo "  # Double-click \"Install RAWviewer.command\" (right-click → Open if blocked once)"
        echo "  # Or: bash install_macos_app.sh"
        echo "  # Single-image view: share icon -> Qt menu -> Mail (or Messages)"
        echo "  # See scripts/Launch/README.md and docs/macos-sharing-v21-v22.md"
        echo ""
        echo "To run without installing:"
        echo "  xattr -cr dist/RAWviewer.app && open dist/RAWviewer.app"
    elif [ -f "dist/RAWviewer" ]; then
        echo "Executable created: dist/RAWviewer"
        echo ""
        echo "To run the app:"
        echo "  ./dist/RAWviewer"
    else
        echo "[WARNING] Build completed but output files not found in expected location"
        echo "Check the dist/ directory for output files"
    fi
else
    echo ""
    echo "[ERROR] Build failed. Check the error messages above."
    exit 1
fi
