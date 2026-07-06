#!/bin/bash
# Run the RAWviewer automated test plan. See testplan/TEST_PLAN.md.
#   ./testplan/run_all.sh          full run (invariants + golden parity + perf)
#   ./testplan/run_all.sh --fast   invariant suites only (< 30s, no golden files)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

PY="pixi run python3"
FAILED=()

run() {
  local name="$1"; shift
  echo ""
  echo "=== $name ==="
  if PYTHONPATH=src $PY "$@"; then
    echo "--- $name: OK"
  else
    echo "--- $name: FAILED"
    FAILED+=("$name")
  fi
}

run "compile check" -m py_compile \
  src/main.py src/fast_raw_decode.py src/unified_image_processor.py \
  src/raw_pv2012.py src/raw_adjustments.py src/raw_edit_pipeline.py \
  src/image_cache.py src/image_load_manager.py src/gpu_raw_processor.py \
  src/perf_metrics.py

run "tone engine invariants" testplan/auto/t_tone_engine.py
run "xmp round-trip" testplan/auto/t_xmp_roundtrip.py
run "slider specs" testplan/auto/t_slider_specs.py
run "cache semantics" testplan/auto/t_cache_semantics.py
run "dodge & burn" testplan/auto/t_dodge_burn.py
run "gallery closes editor" testplan/auto/t_gallery_closes_editor.py
run "search bar animation" testplan/auto/t_search_bar_animation.py
run "gpu return_linear" testplan/auto/t_gpu_return_linear.py

if [ "$FAST" = "0" ]; then
  # Perf first: it must measure on a quiet machine, not one still hot from
  # the parity gate's 24-file decode workload (measured +15-30% noise when
  # run immediately after).
  run "perf baseline" testplan/auto/t_perf_baseline.py
  run "wb sanity (golden)" testplan/auto/t_wb_sanity.py
  run "color parity gate (golden)" scripts/fast_raw_decode_parity_gate.py
fi

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL SUITES PASSED"
  exit 0
fi
echo "FAILED SUITES: ${FAILED[*]}"
exit 1
