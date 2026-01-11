#!/bin/bash

# RAWviewer Launcher (Source Mode)

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "Checking dependencies..."

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install Python 3."
    exit 1
fi

# Install essential dependencies (skipping rawpy for special version if needed)
# Using --no-warn-script-location to avoid path warnings
pip3 install --upgrade pip
pip3 install PyQt6 numpy Pillow exifread send2trash psutil qtawesome natsort typing_extensions

# Run the application
echo "Starting RAWviewer..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Fix for "Could not find the Qt platform plugin 'cocoa'" error
# Dynamically locate PyQt6 plugins directory
PLUGINS_DIR=$(python3 -c "import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), 'Qt6', 'plugins'))")
if [ -d "$PLUGINS_DIR" ]; then
    echo "Setting QT_PLUGIN_PATH to: $PLUGINS_DIR"
    export QT_PLUGIN_PATH="$PLUGINS_DIR"
else
    echo "WARNING: Could not find PyQt6 plugins directory at $PLUGINS_DIR"
fi

python3 src/main.py

echo ""
echo "Application exited. Press Enter to close..."
read
