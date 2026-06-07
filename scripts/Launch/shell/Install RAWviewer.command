#!/bin/bash
# Double-clickable wrapper for install_macos_app.sh (shipped in release zips).
# macOS blocks downloaded scripts on double-click — right-click → Open → Open once.
cd "$(dirname "$0")"
exec bash "./install_macos_app.sh"
