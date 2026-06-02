#!/bin/bash
# Forwarder — see scripts/Launch/shell/clear_cache.sh
exec "$(cd "$(dirname "$0")" && pwd)/scripts/Launch/shell/clear_cache.sh" "$@"
