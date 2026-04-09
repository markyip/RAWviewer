#!/bin/bash

# Exit on any error
set -e

# Repo root = directory containing this script (works if you cd elsewhere or project was moved)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "RAWviewer macOS Build Script"
echo "==========================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "[ERROR] This script is designed for macOS only."
    echo "Current OS: $OSTYPE"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

# Check if icon file exists
if [ ! -f "icons/appicon.icns" ]; then
    echo "[WARNING] Icon file not found: icons/appicon.icns"
    echo "The app will be built without a custom icon."
fi

VENV_DIR="$SCRIPT_DIR/rawviewer_env"
PYTHON_BIN="$VENV_DIR/bin/python3"

# Check if virtual environment exists
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

# Always call the venv interpreter by path. "source activate" embeds the old absolute
# path when the repo is copied or moved, so PATH may not include this venv and plain
# "pip" can install into Homebrew or another Python instead.
echo "Using virtual environment: $VENV_DIR"

# Upgrade pip first
echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip

# Install/upgrade dependencies
echo "Installing dependencies..."
"$PYTHON_BIN" -m pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow psutil numpy qtawesome pyqtgraph

# Clean previous builds
echo "Cleaning previous builds..."
# macOS/Finder can leave metadata files (e.g. .DS_Store) with odd flags; be defensive.
rm -rf build || true
if [ -d "dist" ]; then
    chmod -R u+w dist 2>/dev/null || true
fi
rm -rf dist || true
# Only remove spec files if they exist (don't fail if they don't)
rm -f *.spec

# Build the application using build.py for consistency
echo "Building RAWviewer..."
if "$PYTHON_BIN" build.py; then
    echo ""
    echo "[SUCCESS] Build completed!"
    echo ""
    
    # Check what was actually created
    if [ -d "dist/RAWviewer.app" ]; then
        echo "✅ macOS App Bundle created: dist/RAWviewer.app"
        echo ""
        echo "To run the app, you can:"
        echo "  1. Double-click RAWviewer.app in Finder"
        echo "  2. Run: open dist/RAWviewer.app"
    elif [ -f "dist/RAWviewer" ]; then
        echo "✅ Executable created: dist/RAWviewer"
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