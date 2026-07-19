#!/bin/bash
# Double-clickable wrapper for uninstall_macos_app.sh (shipped in release zips).
# Right-click → Open → Open if macOS blocks a double-click.
cd "$(dirname "$0")"
exec bash "./uninstall_macos_app.sh"
