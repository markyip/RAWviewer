# Parallelization tuning & platform notes

## Why CPU usage can stay low at a bottleneck

Folder sort and indexing spend most time on **disk I/O** (reading EXIF headers from thousands of RAW/JPEG files), not pure CPU. Too many concurrent readers on **network shares or USB HDDs** increases seek latency and can **slow** the whole app — an earlier build capped sort probes at **3 threads** for that reason.

On **local SSD/NVMe**, the same cap under-utilizes the CPU. Defaults now scale with core count (typically `min(12, cpu−1)`), while UNC paths and `RAWVIEWER_SLOW_STORAGE_PREFIXES` stay at 3.

Other limits that cap visible CPU:

| Subsystem | Default limit | Purpose |
|-----------|---------------|---------|
| EXIF sort probe | 4–12 workers (local) / 3 (slow storage) | Header reads for capture time |
| Semantic index metadata | 2–6 workers (1–3 on huge folders) | SQLite EXIF cache lock contention |
| LibRaw decode | `RAWVIEWER_RAW_LOAD_LIMIT=4` | Avoid RAM/IO thrash during view |
| Thumbnail QThreadPool | `max(16, 2×cores)` | Many light JPEG embed reads |
| Process pool (LibRaw post) | `cores÷2` | Optional; `RAWVIEWER_USE_PROCESS_POOL` |

## Environment variables

See README **Optional dev toggles** for:

- `RAWVIEWER_SORT_PROBE_WORKERS`
- `RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE` (fast-open window; default 3)
- `RAWVIEWER_SLOW_STORAGE_PREFIXES` (e.g. `K:\Photos,\\nas\share`)
- `RAWVIEWER_INDEX_METADATA_WORKERS`
- `RAWVIEWER_RAW_LOAD_LIMIT`
- `RAWVIEWER_PROCESS_POOL_WORKERS`

## macOS compatibility (recent `main` changes)

The following **7 commits** on `main` (gallery scroll, EXIF sort refinement, fast-open, GPU navigation) were reviewed for macOS:

| Area | macOS status |
|------|----------------|
| Gallery scroll / prefetch (`gallery_view.py`) | **OK** — Qt only |
| EXIF sort / parallel probe (`main.py`, `common_image_loader.py`) | **OK** — `metadata_backend` + exifread/pyexiv2; path heuristics use `normpath`, not Win32 APIs |
| Fast-open + delayed sort refinement | **OK** — `QThreadPool` / `QTimer` |
| GPU viewport (`gpu_image_view.py`) | **OK** — opt-in `RAWVIEWER_GPU_VIEW=1`; Metal/OpenGL via Qt |
| `windows_shell_meta.py` | **N/A** — POC-only, not imported by the app on any OS |
| `clear_cache.bat` | **Windows only** — macOS: delete `~/.rawviewer_cache` or use Launch scripts |
| Windows share / Open With / frameless chrome | **Guarded** — `sys.platform == "win32"`; macOS uses native share sheet and standard window chrome |

**No macOS-specific regressions** were found in the sort/GPU/gallery paths. Validate on Ventura+ with:

1. Large folder open (cold cache) — capture-time order completes in background.
2. Fast single-file open — first image appears before full-folder EXIF sort.
3. Gallery scroll on 1000+ images — no multi-minute UI freeze.
4. Optional: `RAWVIEWER_GPU_VIEW=1` — Space / pinch zoom on RAW.

**Caveats (all platforms):**

- Instant EXIF sort when the semantic DB is 100% cached runs on the **main thread** — can hitch UI briefly on very large folders.
- Semantic indexing + folder sort can contend on the **same SQLite EXIF cache**; metadata worker count is intentionally lower on folders with &gt;2000 files.
