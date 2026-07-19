# Scripts

Helper scripts for packaging, models, LibRaw, benches, and tests.  
Day-to-day launch / installer wrappers live under [`Launch/`](Launch/README.md).

```
scripts/
  Launch/       Windows + macOS launch, build, release-zip assets
  libraw/       OpenMP LibRaw build + parallelism check (macOS packaging)
  models/       MobileCLIP ONNX / Core ML download & export
  packaging/    App icon / social-preview asset generators
  bench/        Perf benches and [PERF] log report
  test/         Correctness gates (Fast RAW parity, Adjust pipeline)
  tools/        One-off tuning helpers (not part of CI)
```

## By folder

| Folder | Keep? | Notes |
|--------|-------|--------|
| **libraw/** | **Active** | `build_libraw_openmp.sh` is required before macOS packaging |
| **models/** | **Active** | `download_mobileclip_onnx.py` runs from Windows installer / `build_windows.bat` |
| **test/** | **Active** | `fast_raw_decode_parity_gate.py` is in `testplan/run_all.sh` |
| **bench/** | Dev tool | Local machine benches; some PS1 scripts hard-code dataset paths |
| **packaging/** | Dev tool | Regenerate icons / GitHub social preview |
| **tools/** | Weak / optional | `shadow_tuning_*` — hard-coded sample paths, not referenced by product or CI |

## Obsolete / not here

- **SCUNet** download scripts — removed with export AI denoise (3.0.1). Leftover weights under `models/scunet_*.pth` can be deleted locally.
- **BYO torch** installer helpers — removed; Plus CUDA uses CuPy.

## Common commands

```bash
# Windows installer model fetch (also invoked by Setup)
pixi run python -u scripts/models/download_mobileclip_onnx.py

# Fast RAW color gate
PYTHONPATH=src pixi run python scripts/test/fast_raw_decode_parity_gate.py

# Adjust pipeline smoke
PYTHONPATH=src pixi run python scripts/test/phase_develop_adjust_linear.py

# Aggregate [PERF] logs
pixi run python scripts/bench/perf_report.py /tmp/run.log

# macOS OpenMP LibRaw (before packaging)
bash scripts/libraw/build_libraw_openmp.sh
```
