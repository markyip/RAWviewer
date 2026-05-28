#!/bin/bash
# Forwarder — see scripts/Launch/shell/build_macos.sh
exec "$(cd "$(dirname "$0")" && pwd)/scripts/Launch/shell/build_macos.sh" "$@"
