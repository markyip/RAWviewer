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
  src/perf_metrics.py src/enhanced_raw_processor.py src/common_image_loader.py src/torch_bootstrap.py \
  src/raw_detail_enhance.py src/raw_tone_recovery.py

run "tone engine invariants" testplan/auto/t_tone_engine.py
run "xmp round-trip" testplan/auto/t_xmp_roundtrip.py
run "slider specs" testplan/auto/t_slider_specs.py
run "cache semantics" testplan/auto/t_cache_semantics.py
run "dodge & burn" testplan/auto/t_dodge_burn.py
run "gallery closes editor" testplan/auto/t_gallery_closes_editor.py
run "gallery hover prefetch" testplan/auto/t_gallery_hover_prefetch.py
run "editor skip unsupported" testplan/auto/t_editor_skip_unsupported.py
run "adjust loading overlay" testplan/auto/t_adjust_loading_overlay.py
run "adjust copy paste shortcuts" testplan/auto/t_adjust_copy_paste_shortcuts.py
run "adjust workflow restore" testplan/auto/t_adjust_workflow_restore.py
run "dodge & burn reset" testplan/auto/t_dodge_burn_reset.py
run "dodge & burn undo" testplan/auto/t_dodge_burn_undo.py
run "spot heal" testplan/auto/t_spot_heal.py
run "dodge & burn brush wheel" testplan/auto/t_dodge_burn_brush_wheel.py
run "filmstrip hidden in editor" testplan/auto/t_filmstrip_editor_hidden.py
run "search bar animation" testplan/auto/t_search_bar_animation.py
run "gpu return_linear" testplan/auto/t_gpu_return_linear.py
run "half decode return_linear" testplan/auto/t_half_decode_return_linear.py
run "shadow edge-aware damp" testplan/auto/t_shadow_edge_aware_damp.py
run "shadow smoothed ratio" testplan/auto/t_shadow_smoothed_ratio.py
run "channel tone curve (RGB Standard mode)" testplan/auto/t_channel_tone_curve.py
run "gallery thumbnail cold-decode perf" testplan/auto/t_gallery_thumb_perf.py
run "detail enhance (sharpness/clarity/defringe) perf" testplan/auto/t_detail_enhance_perf.py
run "CR3 preview extraction perf" testplan/auto/t_cr3_preview_perf.py
run "EXIF cache read concurrency" testplan/auto/t_exif_cache_concurrency.py
run "NEF HE/HE* detection" testplan/auto/t_nef_he_detection.py
run "NEF HE/HE* editing & export" testplan/auto/t_nef_he_editing_export.py
run "full-res decode dedup" testplan/auto/t_full_decode_dedup.py
run "redundant full re-display" testplan/auto/t_redundant_full_redisplay.py
run "capture-time-only sort" testplan/auto/t_capture_time_only_sort.py
run "resolution crossfade decision" testplan/auto/t_resolution_crossfade_decision.py

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
