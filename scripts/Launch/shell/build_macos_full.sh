#!/bin/bash
# Build RAWviewer full for macOS (semantic search + face scan).
cd "$(dirname "$0")/../../.." || exit 1
exec ./scripts/Launch/shell/build_macos.sh full "$@"
