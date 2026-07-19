#!/bin/bash
# Forward to scripts/Launch/macos/clear_cache.sh
exec "$(cd "$(dirname "$0")" && pwd)/scripts/Launch/macos/clear_cache.sh" "$@"
