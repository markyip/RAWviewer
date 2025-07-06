#!/bin/bash

echo "RAWviewer macOS Build Script"
echo "==========================="
echo ""

# Check if virtual environment exists
if [ ! -d "rawviewer_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv rawviewer_env
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source rawviewer_env/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist *.spec

# Build the application
echo "Building RAWviewer..."
python build.py

echo ""
echo "Build completed!"
echo ""
echo "You can find the executable at:"
echo "  - Command line: dist/RAWviewer"
echo "  - macOS App: dist/RAWviewer.app"
echo ""
echo "To run the app, you can:"
echo "  1. Double-click RAWviewer.app in Finder"
echo "  2. Run: open dist/RAWviewer.app"
echo "  3. Run: ./dist/RAWviewer" 