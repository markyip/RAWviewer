# Performance & Memory Review — Cross-Validation Summary

**Repository:** RAWviewer (`markyip/RAWviewer`)  
**Branch:** `development`  
**HEAD commit:** `a902ebd`  
**Review snapshot:** 2026-08-19 — **HEAD plus a dirty working tree**  
**Dirty files in this snapshot:** `src/main.py`, `src/raw_ai_masks.py`, `src/rawviewer_mcp/paths.py`, `src/rawviewer_mcp/preview.py`, `src/rawviewer_mcp/service.py`, plus this document (untracked)  
**Scope:** Export pipeline, export queue, MCP headless cache, gallery scroll UX, XMP/adjustment loading, LUT preview worker  
**Purpose:** Independent re-check of findings before 3.1.0 stable patch ship

**Do not `git checkout a902ebd` and expect F-01 to match this document.** At HEAD, `LinearBaseCache` is still an unbounded `dict`. The LRU (max 3) lives only in the uncommitted `src/rawviewer_mcp/preview.py` working copy. Prefer function names over line numbers: `main.py` has a large uncommitted diff.

---

## How to use this document

Each item has:

| Column | Meaning |
|--------|---------|
| **ID** | Stable reference for discussion |
| **Severity** | P0 blocker · P1 ship-risk · P2 improvement |
| **Status** | `FIXED` · `OPEN` · `MITIGATED` · `BY_DESIGN` |
| **Where** | `HEAD` = in `a902ebd`; `WT` = working tree only |
| **Verify** | Concrete grep / read / manual step for a second reviewer |

Mark each row **Confirmed** / **Disputed** / **N/A** after independent check.

---

## Executive summary

Counts below are against the **working tree**, not a clean `a902ebd`.

| Category | Fixed | Open | Mitigated | By design |
|----------|-------|------|-----------|-----------|
| P1 (ship-risk) | 1 (F-01, WT only) | 1 (O-01) | 1 (O-02) | — |
| P2 (improvement) | 2 (F-04, F-05) | 2 (O-03, O-05) | — | 2 (O-04, O-06) |
| Also fixed (was P1) | F-02, F-03 | — | — | — |

**Bottom line:** Export-queue bookkeeping (path-only add, success-only pop) is in `a902ebd`. MCP linear LRU is **not** in HEAD — only in the working tree. Remaining P1: no RAM purge before full-res export; queue still loads adjustments on the UI thread (parse cost mitigated by XMP root cache). Queue export does not forward AI denoise/upscale flags (P2).

---

## Confirmed fixes (re-verify still recommended)

### F-01 — MCP `LinearBaseCache` bounded LRU

| Field | Value |
|-------|-------|
| **Severity** | P1 (was) |
| **Status** | `FIXED` in **working tree only** — **not** in `a902ebd` |
| **File** | `src/rawviewer_mcp/preview.py` |
| **Evidence (WT)** | `_LINEAR_BASE_CACHE_MAX = 3`; `OrderedDict` + `popitem(last=False)` on insert |
| **Evidence (HEAD)** | Unbounded `dict`; no maxsize / eviction |
| **Verify** | `rg "_LINEAR_BASE_CACHE_MAX|popitem" src/rawviewer_mcp/preview.py` — empty on clean `a902ebd` |
| **Test gap** | `testplan/auto/t_mcp_tools.py` does **not** assert LRU eviction |

### F-02 — Export queue stores paths only (no adj preload on add)

| Field | Value |
|-------|-------|
| **Severity** | P1 (was) |
| **Status** | `FIXED` (`HEAD`) |
| **File** | `src/main.py` → `_add_paths_to_export_queue` |
| **Evidence** | `queue.append({"path": path})`; comment that XMP is loaded in `_pump_export_queue` |
| **Verify** | Read `_add_paths_to_export_queue` — must not call `load_adjustments_for_file` |

### F-03 — Export queue pops head only on success

| Field | Value |
|-------|-------|
| **Severity** | P1 (was) |
| **Status** | `FIXED` (`HEAD`) |
| **File** | `src/main.py` → `_finish_export_queue_item`, `_pump_export_queue` |
| **Evidence** | Pop deferred to `_finish_export_queue_item(success=True)`; failure sets `_export_queue_running = False` |
| **Verify** | Read `_pump_export_queue` — must **not** `pop(0)` before `_launch_export_job` returns |

### F-04 — XMP root parse cache (LRU 8)

| Field | Value |
|-------|-------|
| **Severity** | P2 (was) |
| **Status** | `FIXED` (`HEAD`) |
| **File** | `src/raw_adjustments.py` → `_cached_xmp_root`, `_XMP_ROOT_CACHE_MAX` |
| **Evidence** | Shared `OrderedDict` keyed by `(mtime_ns, size)` |
| **Verify** | `PYTHONPATH=src python3 testplan/auto/t_xmp_root_cache.py` — **not** listed in `testplan/run_all.sh` |

### F-05 — Gallery scroll date bubble: cache-hit only

| Field | Value |
|-------|-------|
| **Severity** | P2 (was) |
| **Status** | `FIXED` (`HEAD`) |
| **File** | `src/rawviewer_ui/gallery_view.py` → `_scroll_preview_pixmap` |
| **Evidence** | Docstring: “Never decodes, never hits disk”; uses `_thumbnail_cache` + memory-tier peek only |
| **Verify** | `PYTHONPATH=src python3 testplan/auto/t_gallery_scroll_date_bubble.py` — **not** listed in `testplan/run_all.sh` |

---

## Open issues (primary cross-validation targets)

### O-01 — No heavy-tier purge before full-res export

| Field | Value |
|-------|-------|
| **ID** | O-01 |
| **Severity** | **P1** |
| **Status** | `OPEN` (`HEAD` and WT) |
| **Risk** | Peak RAM = browse full cache + export decode + AI pipeline → macOS jetsam on 8–16 GB machines |
| **Files** | `src/main.py` (`_AdjustExportWorker.run`, `_launch_export_job`, `_on_adjust_panel_export_requested`), `src/image_cache.py` (`clear_heavy_memory_tiers`) |
| **Evidence** | `clear_heavy_memory_tiers` is only called from `on_folder_changed`. No call on any export start path. |
| **Verify** | `rg "clear_heavy_memory" src/` — expect **only** `image_cache.py` |
| **Suggested fix** | On the main thread, immediately before spawning the export worker, call `image_cache.clear_heavy_memory_tiers()` with **`keep_path=None`** (or otherwise drop the current file’s full/preview/pixmap). Do **not** pass `keep_path=export_source`: that retains the browse full buffer of the file being exported, which is the duplicate the purge is meant to free. |
| **Manual test** | Open 24 MP RAW in Adjust (full browse cache warm) → Export JPEG with AI denoise → watch Activity Monitor peak vs same export after folder change. |

### O-02 — Export queue loads adjustments on UI thread

| Field | Value |
|-------|-------|
| **ID** | O-02 |
| **Severity** | **P1** |
| **Status** | `MITIGATED` (XMP parse cache) / **still OPEN** (thread + large masks) |
| **Risk** | UI freeze when pumping queue over files with fat mask sidecars |
| **File** | `src/main.py` → `_pump_export_queue` |
| **Evidence** | `adj = load_adjustments_for_file(path)` on main thread before `_launch_export_job` |
| **Verify** | `rg -n -A 30 "def _pump_export_queue" src/main.py` — confirm `load_adjustments_for_file` is inside that function. Confirm callers are `QTimer.singleShot(0, self._pump_export_queue)` and `_run_export_queue_dialog` (GUI thread). Do **not** pipe the load-line into `rg pump_export`; that match is always empty. |
| **Suggested fix** | Load adj in a background runnable; launch the export job only when that load finishes. |
| **Manual test** | Queue 10+ edited files with AI masks → Start export → scroll gallery during pump; profile main-thread stalls. |

### O-03 — Queue export omits AI denoise / upscale flags

| Field | Value |
|-------|-------|
| **ID** | O-03 |
| **Severity** | **P2** |
| **Status** | `OPEN` (unless intentional) |
| **Risk** | Batch queue never applies SCUNet / upscale even when Adjust single-export would |
| **Files** | `src/main.py` — `_on_adjust_panel_export_requested` vs `_launch_export_job` |
| **Evidence** | Single path passes `use_ai_denoise=`, `use_ai_upscale=` into `_AdjustExportWorker`. `_launch_export_job` (queue path) omits both; worker defaults `False`. |
| **Verify** | `rg -n "use_ai_denoise=" src/main.py` — expect the kwarg on the Adjust-panel worker construction, not on `_launch_export_job`. `ExportOptionsDialog` queue mode has no AI toggles. |
| **Product question** | Is queue-without-AI intentional for 3.1.0? If yes → document; if no → wire flags from prefs or dialog. |

### O-04 — Export always re-decodes full-res (no browse cache reuse)

| Field | Value |
|-------|-------|
| **ID** | O-04 |
| **Severity** | **P2** |
| **Status** | `BY_DESIGN` (likely) |
| **Risk** | Extra CPU + transient RAM vs reusing `full_image_cache` when same file already decoded |
| **File** | `src/main.py` → `_AdjustExportWorker.run` — `decode_raw_edit_base(use_full_resolution=True)` |
| **Verify** | No `image_cache.get` / full-tier reuse in the export worker |
| **Note** | Reuse is non-trivial (lens correction flag, embedded-JPEG vs true RAW provenance). Accept as trade-off unless profiling shows export decode dominates. If O-01 purges heavies first, reuse is unavailable anyway. |

### O-05 — LUT chooser renders all LUTs in one worker batch

| Field | Value |
|-------|-------|
| **ID** | O-05 |
| **Severity** | **P2** |
| **Status** | `OPEN` |
| **Risk** | Spike CPU/RAM when many LUTs installed; blocks one thread pool slot |
| **Files** | `src/rawviewer_ui/adjust_panel.py` → `_LutThumbWorker.run`; `src/raw_lut.py` → `render_lut_chooser_previews` |
| **Evidence** | Single `render_lut_chooser_previews(tiny, self.lut_names, …)` over full list |
| **Verify** | Per-LUT incremental worker or chunked batches would be the fix pattern |
| **Manual test** | Install 30+ LUTs → open Looks tab → measure time-to-all-thumbs vs LUT count |

### O-06 — `PreviewStageCache` holds multiple stage ndarrays per file

| Field | Value |
|-------|-------|
| **ID** | O-06 |
| **Severity** | **P2** |
| **Status** | `BY_DESIGN` |
| **Risk** | Memory proportional to stage count × buffer size during live Adjust drag |
| **File** | `src/raw_edit_pipeline.py` → `PreviewStageCache` |
| **Mitigation** | `reset()` on file change; separate fast/full caches in `main.py` |
| **Verify** | Class docstring: export calls `process_linear_edit_buffer` / `_apply_display_stage` with no stage cache |

---

## Known-good patterns (sanity check — should remain true)

| Pattern | Location | Expected behaviour |
|---------|----------|-------------------|
| Gallery thumb **item** cap | `ImageCache.get_recommended_cache_sizes` | `max_thumbnails = max(50, cache_budget_gb * 1024 * 0.2 / 1MB)`, then `min(..., 1000)`. Not a flat 768 MB cap. |
| Pixmap **byte** cap | `ImageCache.put_pixmap` | Separate from thumbs: `_enforce_numpy_cache_byte_budget(..., fraction=0.20)` of `max_memory_mb` (the cache budget, not system RAM). |
| Full image byte cap | `ImageCache.put` / `_enforce_full_image_byte_budget` | `fraction=0.55` of **`max_memory_mb`**, which is ~25% of *available* RAM capped at 4 GB — **not** 55% of system RAM. Peak full-tier ≈ 0.55 × 4 GB ≈ 2.2 GB. |
| Scroll bubble no disk decode | `gallery_view.py` `_scroll_preview_pixmap` | Miss → date text only |
| Watermark corner-patch | `export_options.py` | Not full-frame RGBA composite |
| Export error mentions OOM | `_AdjustExportWorker.run` | User-visible jetsam hint after full-res decode failure |
| MCP path sandbox | `rawviewer_mcp/paths.py` | Traversal rejected — `t_mcp_tools.py` **is** in `run_all.sh` |

---

## Test coverage matrix

| Finding | Automated test | In `run_all.sh`? | Notes |
|---------|----------------|------------------|-------|
| F-01 MCP LRU | ❌ | n/a | Add: insert 4 paths → `len(_store) == 3` |
| F-02 Queue path-only | ❌ | n/a | Could inspect stored dict keys after add |
| F-03 Success-only pop | ❌ | n/a | Integration / UI test |
| F-04 XMP root cache | ✅ `t_xmp_root_cache.py` | **No** | Run the file directly |
| F-05 Scroll bubble | ✅ `t_gallery_scroll_date_bubble.py` | **No** | Run the file directly |
| O-01 Export purge | ❌ | n/a | Manual / instrumentation |
| O-02 Queue adj on UI | ❌ | n/a | Manual profiling |
| O-03 Queue AI flags | ❌ | n/a | Assert worker kwargs |

---

## Suggested verification commands

Run these against the **same tree you intend to ship** (commit F-01 first if that tree is supposed to include MCP LRU).

```bash
git rev-parse --short HEAD
git status --short

# F-01: empty on clean a902ebd; non-empty on WT with LRU
rg "_LINEAR_BASE_CACHE_MAX|popitem" src/rawviewer_mcp/preview.py

# O-01: expect only image_cache.py
rg "clear_heavy_memory" src/

# O-02: load must appear inside _pump_export_queue
rg -n -A 30 "def _pump_export_queue" src/main.py

# O-03: kwarg on Adjust-panel export, not on _launch_export_job
rg -n "use_ai_denoise=" src/main.py

# Tests that exist (F-04 / F-05 are not in run_all.sh)
PYTHONPATH=src python3 testplan/auto/t_xmp_root_cache.py
PYTHONPATH=src python3 testplan/auto/t_gallery_scroll_date_bubble.py
PYTHONPATH=src python3 testplan/auto/t_mcp_tools.py
```

`testplan/run_all.sh` covers `t_mcp_tools.py` and many other suites; it does **not** cover F-04 / F-05.

---

## Recommended 3.1.0 patch minimum (if shipping stable)

| Priority | Item | Effort |
|----------|------|--------|
| **Must** | Commit F-01 (MCP LRU) if it is meant to ship | Already in WT |
| **Must** | O-01 — purge heavy tiers **without** `keep_path=export_source` before export worker | Small |
| **Should** | O-02 — move queue `load_adjustments_for_file` off UI thread | Medium |
| **Decide** | O-03 — document queue-no-AI or wire flags | Small / product |
| **Defer** | O-04–O-06 | 3.2+ unless profiling demands |

---

## Cross-validation sign-off

| Reviewer | Date | Snapshot | F-items | O-items | Notes |
|----------|------|----------|---------|---------|-------|
| _(initial)_ | 2026-08-19 | claimed `a902ebd` | F-01–F-05 confirmed | O-01–O-06 as listed | Did not distinguish HEAD vs dirty WT. Claimed ~768 MB thumb byte cap (incorrect). |
| Claude (independent re-check) | 2026-08-19 | claimed `a902ebd` | F-01–F-05 **Confirmed** | O-01–O-06 **Confirmed** | Corrected thumb item-count formula. Still treated F-01 as in `a902ebd` (it is WT-only). Still suggested `keep_path=export_source` for O-01 (wrong). Still claimed F-04/F-05 run via `run_all.sh`. Full-image “55% RAM” row left inaccurate. |
| Cursor (doc correction) | 2026-08-19 | `a902ebd` + dirty WT listed above | F-01 = WT only; F-02–F-05 in HEAD | O-01 OPEN; O-02 MITIGATED/OPEN; O-03/O-05 OPEN; O-04/O-06 BY_DESIGN | Fixed baseline, O-01 purge advice, verify commands, cache-budget wording, test-harness claims, and counts. |

---

## Related docs

- `testplan/TEST_PLAN_3.1.md` — functional QA for 3.1.0 stable patch scope  
- `docs/MCP.md` — MCP server usage (headless; no GUI reload of XMP)  
- `RELEASE_NOTES.md` — 3.1.0 = stability/perf patch narrative
