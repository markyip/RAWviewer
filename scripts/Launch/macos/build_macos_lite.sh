#!/bin/bash
# Build RAWviewer lite for macOS (viewing + EXIF/GPS metadata search; no AI).
cd "$(dirname "$0")/../../.." || exit 1
exec ./scripts/Launch/macos/build_macos.sh lite "$@"
