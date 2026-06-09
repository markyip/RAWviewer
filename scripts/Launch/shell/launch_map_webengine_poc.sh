#!/usr/bin/env bash
# Leaflet GPS map POC — uses project venv (conda base PyQt6 lacks libglib for WebEngine).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
exec "${ROOT}/rawviewer_env/bin/python" "${ROOT}/scripts/poc/location_map_webengine_poc.py" "$@"
