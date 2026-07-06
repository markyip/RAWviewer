# RAWviewer Test Plan

Automated + manual verification for the RAW decode/edit pipeline. All
automation is dependency-free (no pytest) and runs inside the pixi env.

## Quick start

```bash
./testplan/run_all.sh              # everything automated
./testplan/run_all.sh --fast      # skip golden-file + perf suites
pixi run python testplan/auto/t_tone_engine.py   # one suite
```

Every suite is a standalone script exiting 0 (pass) / 1 (fail); suites that
need golden RAW files **skip** (exit 0 with a SKIP line) when the files are
absent, so the plan runs on any machine.

## Layers

### 1. Invariant suites (`testplan/auto/t_*.py`, hermetic, < 30s)

| Suite | Covers | Key assertions |
|---|---|---|
| `t_tone_engine.py` | PV2012 tone engine (`raw_pv2012.py`, `raw_edit_pipeline.py`) | default adjustments are an exact no-op; Shadows/Blacks/Whites/Highlights monotonic at extremes (no banding); absolute black pinned at 0 under max lift; black-anchor toe shape; Shadows+Blacks ≥ either alone; recovery strength floor (deep-shadow lift ≥ 2× the pre-rework 3.0-cap engine) |
| `t_xmp_roundtrip.py` | Sidecar persistence (`raw_adjustments.py`) | write→load round-trip exact for tone sliders, WB, HSL, NR method, lens toggle, tone curve; default adjustments delete the sidecar; as-shot temperature memoized & deterministic |
| `t_slider_specs.py` | Adjust panel specs | `value_to_slider`/`slider_to_value` are inverses at min/default/max for every spec; ranges sane |
| `t_cache_semantics.py` | `UnifiedImageProcessor` caches | sidecar-adjusted memo: hit is identical object, sidecar edit invalidates, half-size result never served for a full-res request (`image_covers_sensor_resolution` gate); unpack stash consumed exactly once |

### 2. Golden-file suites (need sample RAWs; skip otherwise)

| Suite | Covers |
|---|---|
| `scripts/fast_raw_decode_parity_gate.py` | fast decode color parity vs rawpy (±1 8-bit LSB, half + full), split-unpack byte identity, GPU-vs-CPU color agreement, WB embedded-JPEG sanity trigger/residual on known-misparsed bodies |
| `t_wb_sanity.py` | WB model-verdict cache: clean model measured once then skipped (0ms), misparsed model corrected per file, model-key disambiguation (matrix+dims+black+white) |

Golden set: `/Volumes/Development/Manchester/DSC01089.ARW` +
`/Volumes/Development/Development/Canon_Sample/*.CR3|NEF` (includes the
EOS R6 Mark III misparse regression files `683A*.CR3`).

### 3. Performance regression (`t_perf_baseline.py`)

Benchmarks `unpack` / `decode_half` / `decode_full_cpu` / `sidecar_apply`
on a golden file (min-of-3, warm cache) and compares against
`testplan/baselines/perf_baseline.json`. **Fails if any metric regresses
> 25%.** First run (or `--rebaseline`) writes the baseline for this machine;
baselines are per-machine and gitignored by design.

For app-level numbers, use the uniform `[PERF]` log channel instead:

```bash
RAWVIEWER_PERF=1 pixi run python src/main.py 2>&1 | tee /tmp/run.log
pixi run python scripts/perf_report.py /tmp/run.log
pixi run python scripts/perf_report.py --compare baseline.log new.log
```

Metrics emitted (see `src/perf_metrics.py` for the canonical list): `unpack`,
`decode_half`, `decode_full_cpu`, `decode_full_gpu`, `decode_rawpy`,
`sidecar_apply`, `nav_to_display`.

### 4. Manual UI checklist (not automatable headlessly)

Run after significant display/editing changes:

- [ ] Rapid arrow-key browse 20+ RAWs: first paint < ~1.5s fresh, instant on revisit, no blurry-stuck frames
- [ ] Zoom to 100%, edit a slider: framing/zoom preserved through live drag and release
- [ ] Edit, save, browse 10+ files, return: edited look paints (no pre-edit flash regression check)
- [ ] RAW↔JPEG workflow toggle on a 1000+ file folder: UI responsive during toggle
- [ ] EDR button: off = fast browse; on = EDR upgrade only after pausing on an image
- [ ] R6 Mark III (or other misparsed-WB body) files: no pink/red cast in browse, editor, export
- [ ] Shadows/Blacks +100 on an underexposed RAW: strong recovery, blacks stay black, no grey veil, colors survive
- [ ] Export TIFF/JPEG of an edited file matches the on-screen preview

## When to run

- `run_all.sh --fast` — before every commit touching `src/`
- `run_all.sh` (full) — before pushing decode/edit-pipeline changes; after
  `pixi install` / LibRaw rebuild (`scripts/build_libraw_openmp.sh`)
- Manual checklist — before tagging a release
