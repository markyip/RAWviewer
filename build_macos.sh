#!/bin/bash

# Exit on any error
set -e

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

# Check if virtual environment exists
if [ ! -d "rawviewer_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv rawviewer_env
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source rawviewer_env/bin/activate

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow psutil numpy

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist
# Only remove spec files if they exist (don't fail if they don't)
rm -f *.spec

# Build the application using build.py for consistency
echo "Building RAWviewer..."
if python3 build.py; then
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