#!/bin/bash
# Launch script for development
# This runs the app directly from source so you can see terminal output and trace logs

# Enable verbose logs for troubleshooting
export RAWVIEWER_VERBOSE_ORIENTATION_LOGS=1
export RAWVIEWER_DEBUG=1

# Add src to PYTHONPATH just in case
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Run the app
echo "Launching RAWviewer..."
python3 src/main.py "$@"
