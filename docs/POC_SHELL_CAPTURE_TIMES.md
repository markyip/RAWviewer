# POC: App EXIF vs Windows Shell Capture Times

**Date:** June 2026  
**Target folder:** `K:\Photos\Japan Trip` (6,889 top-level images, mostly Sony `.ARW`)  
**Platform:** Windows  
**Decision:** **Do not use Shell as a capture-time source for folder sort.** Keep EXIF cache + `metadata_backend` probe.

---

## Goal

Evaluate whether reading capture times via **Windows Shell** (`System.Photo.DateTaken` through `IPropertyStore`) improves folder/gallery sort compared to RAWviewer’s current path:

- **`resolve_folder_sort_timestamp()`** in `src/common_image_loader.py`
- Production: bulk **`get_multiple_exif`** + resolve (warm cache)
- Cold start: **`probe_capture_timestamp_from_file()`** via `metadata_backend` (no cache read)

Compare **accuracy**, **sort order**, and **wall-clock time** against Shell.

---

## Methodology

### Folder scan

- Top-level only (same as RAWviewer), supported image extensions.
- **No pre-sort by filesystem birth/mtime** — selection and ordering use **capture time only**.

### App paths measured

| Mode | Flag | Behaviour |
|------|------|-----------|
| **Probe (default for POC)** | *(none)* | `metadata=None`, `probe_file=True` — no EXIF cache read |
| **Cached (production warm)** | `--use-cache` | `get_multiple_exif` + resolve |
| **Shell** | always | `bulk_shell_date_taken_timestamps()` in `src/windows_shell_meta.py` |

### Report sets

| Run | `--limit` | Meaning |
|-----|-----------|---------|
| Subset A | `1000` | 1,000 chronologically **oldest** files by **app** capture time |
| Subset B | `3000` | 3,000 oldest by app capture time |
| Full | `0` | Entire folder (6,889 files) |

Timing for App and Shell is always measured over the **full folder** extraction pass; value/sort comparison applies to the report set.

### Sort comparison

- Sort key: capture timestamp only (newest-first, matching RAWviewer default), with filename tie-break.
- “Identical order” = same file sequence from App vs Shell sorts.
- Value match: `<1 s` (display-equivalent), `<1 ms` (exact), `≥1 s` (material difference).

### Tooling

```batch
REM Default: probe, no cache read
python scripts\compare_shell_capture_times.py "K:\Photos\Japan Trip" --limit 1000

REM Full folder
python scripts\compare_shell_capture_times.py "K:\Photos\Japan Trip" --limit 0

REM Warm cache comparison
python scripts\compare_shell_capture_times.py "K:\Photos\Japan Trip" --use-cache --limit 0
```

CSV outputs: `scripts/output/japan_trip_*.csv`

### Shell implementation note

Initial POC returned **0 Shell dates** due to a `PyPROPVARIANT` parsing bug (must use `GetValue()`, not `.value`). Fixed in `src/windows_shell_meta.py` before final runs. Module remains **POC-only** — not wired into `main.py`.

---

## Results summary

### Timing (full folder, 6,889 files)

| App mode | Wall time | Per file (avg) |
|----------|-----------|----------------|
| **Probe (no cache read)** | **~9.8–10.0 s** | **~1.4 ms** |
| **Cached (warm EXIF DB)** | **~0.18 s** | **~0.03 ms** |
| **Shell (DateTaken)** | **~23–25 s** | **~3.4 ms** |

Cold-start probe is **~2.4× faster** than Shell. Warm cache is **~130× faster** than Shell.

### Capture-time values (App probe vs Shell)

| Report set | Files | Both have capture time | Within 1 s | Exact (<1 ms) | ≥1 s difference |
|------------|-------|------------------------|------------|---------------|-----------------|
| Oldest 1,000 | 1,000 | 1,000 | 1,000 | 10 | **0** |
| Oldest 3,000 | 3,000 | 3,000 | 3,000 | 22 | **0** |
| **Full folder** | **6,889** | **6,889** | **6,889** | 63 | **0** |

App source for probe runs: **100% `probe`** (no `cache`).

Capture range (full folder, app probe): **2023-01-31 12:52:55** → **2023-02-16 18:46:03**  
(`DSC03534.ARW` … `DSC00733.ARW`)

### Sort order (capture time only, newest first)

| Report set | Order mismatches (full precision) | Order mismatches (rounded to second) |
|------------|-----------------------------------|--------------------------------------|
| Oldest 1,000 | 414 | 473 |
| Oldest 3,000 | 1,556 | 1,555 |
| Full folder | 3,084 | 2,956 |

Mismatches are **not** from wrong dates (zero files differ by ≥1 s). They come from **sub-second precision**: App probe/cache often stores **whole seconds**; Shell exposes **fractional seconds** on burst sequences, so tie-breaking and pairwise order can differ slightly.

### Cached vs probe (reference, full folder warm cache)

When `--use-cache` was used earlier on the same folder:

- App total **~0.18 s**, all sources **`cache`**
- Values still within 1 s of Shell for compared subsets
- Same sub-second sort-order divergence pattern

---

## Cold-cache full-folder run

Attempted to delete `%USERPROFILE%\.rawviewer_cache` before a cold benchmark. SQLite files were **locked** by another process (RAWviewer / Python), so the on-disk cache was not fully removed.

Mitigation for the reported cold run:

- App path: **probe only** (no cache read)
- `RAWVIEWER_PERSISTENT_CACHE=0`

Result unchanged in practice: **~9.8 s** probe vs **~23.4 s** Shell for 6,889 files.

---

## Conclusions

1. **Shell is slower with no accuracy benefit** for this dataset: all capture times agree within one second; none differ by a second or more.
2. **Sort order** is effectively the same for practical use; differences are limited to **burst / sub-second** ordering.
3. **Production path remains EXIF-based:**
   - Warm: `get_multiple_exif` + `capture_timestamp_for_sort`
   - Cold: `probe_capture_timestamp_from_file` via `metadata_backend`
4. **`windows_shell_meta.py` is not integrated** into the app; retained only for dev comparison (`scripts/compare_shell_capture_times.py`).
5. **Do not enable** `RAWVIEWER_USE_SHELL_SORT_DATES` for folder sort (removed from user-facing README / release notes).

---

## Output artifacts

| File | Description |
|------|-------------|
| `scripts/compare_shell_capture_times.py` | POC runner |
| `scripts/output/japan_trip_probe_vs_shell.csv` | Oldest 1,000 by capture (probe) |
| `scripts/output/japan_trip_probe_vs_shell_3000.csv` | Oldest 3,000 by capture (probe) |
| `scripts/output/japan_trip_full_folder_cold_probe.csv` | Full folder 6,889 (probe) |
| `src/windows_shell_meta.py` | Shell reader (POC-only, not in production path) |

CSV columns include per-file app/shell timestamps, delta seconds, and rank positions for newest-first sort.

---

## Related code

- `src/common_image_loader.py` — `resolve_folder_sort_timestamp`, `probe_capture_timestamp_from_file`
- `src/main.py` — `sort_image_files_with_metadata` / folder load uses `capture_timestamp_for_sort`
- `scripts/Launch/bat/clear_cache.bat` — full cache wipe for repeat cold tests (close RAWviewer first)
