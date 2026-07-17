# Developing RAWviewer

Build scripts, tuning flags, and architecture notes. End-user documentation lives in the [README](../README.md).

Scripts and build matrix: [`scripts/Launch/README.md`](../scripts/Launch/README.md)

## Quick start

```bash
pixi install
pixi run start          # full profile (default)
```

**Windows**

| Task | Script |
|------|--------|
| Run (full) | `scripts\Launch\bat\launch_dev_full.bat` |
| Run (lite) | `scripts\Launch\bat\launch_dev_lite.bat` |
| Build Full installers | `scripts\Launch\bat\build_windows_full.bat` (CUDA) or `build_windows_full.bat directml` |
| Build Lite installer | `scripts\Launch\bat\build_windows_lite.bat` |
| Build both Full backends | `scripts\Launch\bat\build_windows_all.bat` |

**macOS**

| Task | Script |
|------|--------|
| Run (full) | `./scripts/Launch/shell/launch_dev_full.sh` |
| Run (lite) | `./scripts/Launch/shell/launch_dev_lite.sh` |
| Build Full | `./scripts/Launch/shell/build_macos_full.sh` → `dist/RAWviewer.app` |
| Build Lite | `./scripts/Launch/shell/build_macos_lite.sh` → `dist/RAWviewer_Lite.app` |

Build outputs:

| Profile | Windows | macOS |
|---------|---------|-------|
| **Full / Unified** | `dist/RAWviewer_Setup.exe` (includes Full & Lite options) | `dist/RAWviewer-v3.0-macOS.zip` |
| **Lite** | (Select Lite option in `RAWviewer_Setup.exe`) | `dist/RAWviewer-v3.0-macOS-Lite.zip` |

Dependencies are in `pixi.toml`. Packaging scripts use a local `rawviewer_env/` venv when building release artifacts.

### Build from source (commands)

**Windows**
```batch
scripts\Launch\bat\build_windows_full.bat
scripts\Launch\bat\build_windows_lite.bat
```

**macOS**
```bash
./scripts/Launch/shell/build_macos_full.sh
./scripts/Launch/shell/build_macos_lite.sh
# or: pixi install && pixi run python build.py --profile full
```

### macOS version support

| Your Mac | Official `.zip` | Build from source |
|----------|-----------------|-------------------|
| macOS 13 Ventura (Intel) | ✅ | `build_macos_full.sh` or Pixi |
| macOS 13 Ventura (Apple Silicon) | ✅ | Use **`build_macos_full.sh`** (Pixi needs 14+) |
| macOS 14 Sonoma+ | ✅ | Pixi or `build_macos.sh` |
| macOS 12 Monterey or older | ❌ | ❌ |

## Automatic memory tuning

On every launch, RAWviewer reads **installed system RAM** (not free RAM at that moment) and applies conservative defaults for load concurrency, preview cache, prefetch, and indexing — **only when you have not already set the same environment variables yourself**.

| Tier | Installed RAM | Typical Mac | What changes (summary) |
|------|---------------|-------------|-------------------------|
| **low** | &lt; 10 GB | 8 GB MacBook Air | Face scan off during indexing; fewer parallel workers; smaller preview cache; less idle prefetch |
| **medium** | 10–14 GB | 12 GB unified | Moderate limits on workers and cache |
| **balanced** | 14–20 GB | 16 GB | Light tuning (default on many laptops) |
| **high** | 20–28 GB | 24 GB | Slightly higher cache / worker caps |
| **ultra** | ≥ 28 GB | 32 GB+ studio machines | Stock app defaults (no overrides) |

**What you might notice**

- Startup log (dev / Terminal): `[PROFILE] memory tier=balanced (16.0 GB RAM)`
- A note file: `~/.rawviewer_cache/memory_tier.json` (tier, RAM, how many defaults were applied)
- **Lite** builds still use Lite profile defaults first; RAM tier only fills in gaps
- **Full** on an 8 GB Mac: semantic AI search can still run, but face indexing is disabled automatically to reduce memory pressure
- **Relaunch (v2.4.1+):** Session restore staggers full decode and prefetch so reopening the last folder is less likely to OOM; see release notes if fit view stays soft for a few seconds

**Disable auto-tuning** (use only your own env vars or scripts):

```bash
export RAWVIEWER_MEMORY_TIER_AUTO=0
```

**Force a specific override** (wins over auto-tuning — examples for low-RAM machines):

```bash
export RAWVIEWER_ENABLE_FACE_SCAN=0
export RAWVIEWER_SEMANTIC_PREP_WORKERS=2
export RAWVIEWER_MEMORY_PREVIEW_MAX=1280
export RAWVIEWER_IDLE_DISPLAY_PREFETCH=0
```

Semantic batch/chunk size for AI indexing is **auto-tuned separately** on first index pass (Core ML on macOS, ONNX on Windows); results are cached under `~/.rawviewer_cache/semantic_batch_tuning.json`.

## Environment variables

| Variable | Effect |
|----------|--------|
| `RAWVIEWER_MEMORY_TIER_AUTO=1` | **Default.** Tune workers, cache, and prefetch from installed RAM at startup |
| `RAWVIEWER_MEMORY_TIER_AUTO=0` | Disable RAM-tier defaults; only explicit env vars apply |
| `RAWVIEWER_MOBILECLIP_VARIANT` | Windows ONNX model: `b` (default), `s0`, `s2`, `l14` |
| `RAWVIEWER_GPU_VIEW=1` | GPU single-image viewport (OpenGL zoom/pan; on by default in release builds) |
| `RAWVIEWER_GPU_VIEW=0` | Force legacy scroll-area single-image view |
| `RAWVIEWER_FAST_RAW_DECODE=1` | **Default.** Shared unpack for half/full Fast RAW; `0` falls back toward rawpy |
| `RAWVIEWER_SIDECAR_ADJUST=0` | **Default.** Browse shows original pixels; edits render in Adjust. Set `1` to apply saved XMP in browse |
| `RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1` | Same color pipeline for fit vs 100% zoom on RAW (default on) |
| `RAWVIEWER_EXIF_BACKEND=auto` | `auto`, `pyexiv2`, or `exifread` |
| `RAWVIEWER_SHARE_MENU=1` | macOS: Qt share menu (recommended) |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | macOS: try native share sheet first |
| `RAWVIEWER_INDEX_DEFER_FACE_SCAN=1` | Defer face scan until after semantic index (default) |
| `RAWVIEWER_SEMANTIC_PREP_WORKERS` | Parallel CPU workers before AI encode (RAM tier may set this) |
| `RAWVIEWER_SEMANTIC_BATCH_AUTO=1` | Auto-tune AI batch/chunk size on index (default) |
| `RAWVIEWER_SEMANTIC_BATCH_CANDIDATES` | Candidate batch sizes for auto-tune (default `8,16,32,64,128`) |
| `RAWVIEWER_PREVIEW_CACHE_ITEMS` | Cap in-memory preview LRU count |
| `RAWVIEWER_MEMORY_PREVIEW_MAX` | Max long edge for in-memory RAW/JPEG preview (pixels) |
| `RAWVIEWER_IDLE_DISPLAY_PREFETCH=0` | Disable idle neighbor prefetch in single view |
| `RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD=1` | **Default.** After relaunch, delay full decode and neighbor prefetch (see v2.4.1 release notes) |
| `RAWVIEWER_SESSION_RESTORE_FULL_DECODE_DELAY_MS` | Milliseconds to wait after first paint before full decode on session restore (default `2500`) |
| `RAWVIEWER_DISABLE_SESSION_RESTORE=1` | Do not reopen the last folder/file on launch |

Full list and dev defaults: [`scripts/Launch/README.md`](../scripts/Launch/README.md), [`docs/macos-sharing-v21-v22.md`](macos-sharing-v21-v22.md).

## MobileCLIP models (Full — AI search)

| Platform | When downloaded | Change variant (Windows) |
|----------|-----------------|--------------------------|
| **Windows Full** | During setup (~600 MB) | Set `RAWVIEWER_MOBILECLIP_VARIANT` to `s0`, `s2`, `b`, or `l14` |
| **macOS Full** | First gallery search (~150 MB) | Dev helper: `python scripts/download_mobileclip_coreml.py --out-dir models/mobileclip2_coreml` |

**Lite builds** do not use MobileCLIP models.

## Upcoming / remaining work

Tracked in [`RELEASE_NOTES.md`](../RELEASE_NOTES.md) (v3.0 Known Issues & Remaining Work).

**Windows HDR / restore Mac EDR safely** — v2.5 shipped macOS EDR; **3.0 removed it** so Fast RAW stays fast. A future path would use HDR-capable displays without regressing browse speed.

**Remaining work (feasibility high → low):**
1. **Cold-folder edited tile regen** — save-from-Adjust already bakes editor-aligned thumbs; optional full `SIDECAR_ADJUST` for never-opened edits.
2. **Broader local masks** — gradients / clone stamp (Dodge/Burn + Heal + crop already ship).
3. **DNG export** — real writer.
4. **ML subject masks** — Full only.
5. **Windows HDR / restore Mac EDR safely**.
6. **VLM-assisted adjust** — large product/model scope.
7. **HE-NEF RAW edit** — blocked until a decoder exists (browse-only today).

**Shipped in 3.0:** Fast RAW vs 2.5, Adjust panel (HSL, Creative LUT, WB presets, crop, D&B + Heal, vignette/dehaze, XMP presets), OpenMP LibRaw in macOS packages, star ratings, burst / Compare (**C**), Lite without torch, gallery loading overhaul. GPU **viewport** on by default (`RAWVIEWER_GPU_VIEW=0` to disable).

## Architecture (brief)

- **ImageLoadManager** — threaded load queue; folder changes cancel in-flight tasks (**v2.5.0**)
- **UnifiedImageProcessor** — RAW/JPEG/TIFF via one path; **Fast RAW decode** shared unpack half/full (**v3.0.0**)
- **Star ratings** — 1–5 + XMP; gallery min-rating filter (**v3.0.0**)
- **Adjust / Develop** — tone, WB, crop, Dodge/Burn, Heal (`cv2.inpaint`), vignette/dehaze, LUT/XMP presets; edits in Adjust by default (**v3.0.0**)
- **Cache** — memory-first; optional disk cache via env; **RAM-tier defaults** at startup (`rawviewer_profile.py`)
- **Semantic index** — SQLite + local embeddings (Core ML on macOS, ONNX on Windows; **Full only**); background passes abort when folder scope changes (**v2.5.0**)
- **Gallery (JustifiedGallery)** — justified grid with zoom slider; capture-time order after EXIF sort (**v2.5.0**+ gallery fill overhaul in **v3.0.0**)
- **RAW recovery preview** — **P** key, half-res linear decode + local tone recovery (**v2.5.0**)
- **Clipping overlay** — **J** key (`exposure_clipping.py`; **v2.5.0**)
- **Lite profile** — no torch/kornia; semantic/face off; CPU Fast RAW + Adjust (**v3.0.0**)
