#!/bin/bash
# Forwarder — see scripts/Launch/shell/launch_dev.sh
exec "$(cd "$(dirname "$0")" && pwd)/scripts/Launch/shell/launch_dev.sh" "$@"
