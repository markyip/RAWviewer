#!/bin/bash
# Double-click after extracting the macOS release zip to remove Gatekeeper quarantine.
# If macOS blocks this script once: right-click → Open → Open.
cd "$(dirname "$0")"
exec bash "./remove_macos_quarantine.sh"
