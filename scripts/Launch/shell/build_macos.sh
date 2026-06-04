#!/bin/bash
# Build RAWviewer on macOS.
# Repo root: scripts/Launch/shell -> ../../..

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "RAWviewer macOS Build Script"
echo "==========================="
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
"$PYTHON_BIN" -m pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow psutil numpy qtawesome pyqtgraph reverse-geocoder pycountry huggingface-hub requests pyobjc-framework-Cocoa pyobjc-framework-CoreML pyobjc-framework-Quartz pyobjc-framework-Vision

echo "Installing optional dependency: pyexiv2 (best-effort)..."
if "$PYTHON_BIN" -m pip install --upgrade pyexiv2; then
    echo "[INFO] pyexiv2 installed (full focus-point / Exiv2 path enabled)."
else
    echo "[WARNING] pyexiv2 install failed on this macOS/Python combo."
    echo "[WARNING] Continuing build with exifread fallback (focus-point extraction may be less complete)."
fi

"$PYTHON_BIN" -m pip uninstall -y sentence-transformers torch torchvision transformers scikit-learn scipy tokenizers safetensors coremltools >/dev/null 2>&1 || true

echo "Cleaning previous builds..."
chmod -R u+w build dist 2>/dev/null || true
rm -rf build || true
chmod -R u+w dist 2>/dev/null || true
rm -rf dist || true
rm -f *.spec

echo "Building RAWviewer..."
if "$PYTHON_BIN" build.py; then
    echo ""
    echo "[SUCCESS] Build completed!"
    echo ""

    if [ -d "dist/RAWviewer.app" ]; then
        echo "macOS App Bundle created: dist/RAWviewer.app"
        echo ""
        echo "To run the app, you can:"
        echo "  1. Double-click RAWviewer.app in Finder"
        echo "  2. Run: open dist/RAWviewer.app"
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
