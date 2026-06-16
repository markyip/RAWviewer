#!/bin/bash
# Build RAWviewer lite for macOS (viewing + EXIF/GPS search + map; no AI).
cd "$(dirname "$0")/../../.." || exit 1
exec ./scripts/Launch/shell/build_macos.sh lite "$@"
