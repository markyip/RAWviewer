#!/bin/bash
# Double-clickable wrapper for remove_macos_quarantine.sh (shipped in release zips).
# Right-click → Open → Open if macOS blocks a double-click.
cd "$(dirname "$0")"
exec bash "./remove_macos_quarantine.sh"
