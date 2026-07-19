#!/bin/bash
# Double-clickable wrapper for clear_macos_cache.sh (shipped in release zips).
# Right-click → Open → Open if macOS blocks a double-click.
cd "$(dirname "$0")"
exec bash "./clear_macos_cache.sh"
