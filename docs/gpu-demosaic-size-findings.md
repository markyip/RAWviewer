# GPU demosaic: size vs speed findings & torch footprint review

**Date:** 2026-07-19  
**Hardware:** NVIDIA GeForce RTX 4070 · `torch 2.6.0+cu124`  
**Method:** RAW HQ (`RAWVIEWER_AUTOTEST_RAW_MODE=1`); clear `~/.rawviewer_cache` + `HKCU\Software\RAWviewer` between every run.  
**Harness:** `scripts/run_gpu_vs_cpu_demosaic_justification_bench.ps1`  
**Raw report:** `bench/logs/gpu_vs_cpu_demosaic_report_20260719_022103.txt`

---

## 1. Bench findings

### Suites

| Suite | Dataset | n | CPU median | GPU median | Speedup | TIMEOUTs |
|-------|---------|---|------------|------------|---------|----------|
| **A Variation** | `I:\RAW_Sample` (multi-manufacturer) | 259 | 0.803 s | 0.641 s | **1.25×** | 0 / 0 |
| **B Stress** | `I:\Photos\Japan Trip` (Sony ARW ~26 MP) | 1000 | 0.784 s | 0.570 s | **1.38×** | 0 / 0 |

Stress wall-clock: CPU **16.6 min** → GPU **11.4 min** (~**1.46×**).  
Stress GPU path logged **1000** `pytorch_cuda` demosaics.

### Variation by format (cold paint median)

| Ext | Maker (typical) | CPU | GPU | Ratio |
|-----|-----------------|-----|-----|-------|
| CR3 | Canon | 0.577 s | 0.378 s | **1.53×** |
| RW2 | Panasonic | 0.653 s | 0.460 s | **1.42×** |
| NEF | Nikon | 0.703 s | 0.532 s | **1.32×** |
| ARW | Sony | 1.072 s | 0.827 s | **1.30×** |
| ORF | Olympus | 0.839 s | 0.792 s | 1.06× |
| RAF | Fujifilm | 1.162 s | 1.100 s | 1.06× |
| DNG | mixed | 1.832 s | 1.721 s | 1.06× |
| 3FR | Hasselblad | 2.039 s | 2.025 s | 1.01× |

### Install size context (measured on this machine)

| Component | Approx. size |
|-----------|--------------|
| CUDA Full install total | **~5.9 GB** |
| `torch` package alone | **~4.45 GB** (`torch/lib` ~4.36 GB) |
| MobileCLIP + SCUNet models | **~642 MB** |
| `RAWviewer_Setup.exe` left in install dir | **~119 MB** |
| Lite install | **~0.9 GB** |
| Ratio CUDA Full / Lite | **~6.5×** |

### Verdict

Observed speedup **1.25×–1.38×** (avg ~**1.31×**) does **not** justify shipping CUDA Full as the default for everyone. Keep:

- **DirectML Full** as the recommended Full path (AI search, smaller install).
- **CUDA Full** as an **opt-in** for heavy RAW HQ culling (best gains on CR3 / RW2 / NEF / ARW).

---

## 2. Do we need the whole `torch/lib`? Can we cherry-pick?

### What RAWviewer actually uses

| Feature | Stack | Torch surface |
|---------|--------|---------------|
| GPU demosaic | `gpu_raw_processor.py` + **kornia** | CUDA tensors, streams, pin_memory, `matmul` / clamp / dtype casts, `kornia.color.raw_to_rgb` |
| SCUNet export denoise | `raw_nn_denoise.py` + **spandrel** | Load `.pth`, fp16 inference on CUDA/MPS |
| AI search | **onnxruntime-directml** | No torch |

Demosaic is a **general tensor ISP**, not a single frozen ONNX graph. We do not call a tiny subset of “one op”; we need a working PyTorch CUDA runtime.

### Measured `torch/lib` (CUDA Full install)

| Kind | Size | Notes |
|------|------|--------|
| `*.dll` | **~3.6 GB** | Runtime — required for a stock cu124 wheel |
| `*.lib` | **~770 MB** | **Link-time only** (e.g. `dnnl.lib` ~623 MB) — **not needed to run** the app |
| Largest DLLs | `torch_cuda.dll` ~913 MB, cuDNN engines ~562 MB, `cublasLt` ~451 MB, `cufft` ~278 MB, `cusparse` ~263 MB, `torch_cpu.dll` ~240 MB, … | Monolithic upstream binaries |

### Cherry-pick options (realistic)

| Approach | Feasible? | Saving | Risk |
|----------|-----------|--------|------|
| **Delete `torch/lib/*.lib` after install** | **Yes** (safe prune) | **~0.7–0.8 GB** | None for runtime |
| Delete unused CUDA DLLs (`cufft`, `cusparse`, `cusolver`, parts of cuDNN, …) | **Risky / fragile** | Possibly 0.5–1.5 GB | Hard to prove unused; upgrades / kornia / inductor / SCUNet can pull symbols; breakage is opaque |
| Strip symbols inside `torch_cuda.dll` / ship a custom torch build | **No** for this product | Large in theory | Requires maintaining a custom PyTorch build matrix |
| “Cherry-pick only the Python functions we call” | **No** | — | Python API ≠ disk; one `import torch` + CUDA op loads the shared libs |
| Replace demosaic with non-torch GPU (OpenCL / shaders / LibRaw-only) | Long-term rewrite | Could drop torch from CUDA profile | Large engineering; separate from install prune |
| SCUNet → ONNX + DirectML | Medium project | Keeps AI denoise without torch on DirectML builds | Does not shrink CUDA demosaic path |

**Conclusion:** We cannot meaningfully “cherry-pick torch functions” from the wheel. The only **low-risk** near-term win on `torch/lib` is **post-install removal of `*.lib`**. Deeper cuts mean either a custom torch build or rewriting GPU demosaic off PyTorch.

---

## 3. Reuse existing user CUDA / torch instead of downloading again?

### What users often already have

On a typical ML workstation (this machine as example):

- `CUDA_PATH` / Toolkit **12.2** and **12.6** on `PATH`
- System / Miniconda `python` that may already have (or can install) torch
- Optional `TORCH_HOME` (usually **model cache**, not the library)

PyTorch’s Windows **cu124 wheel does not use the CUDA Toolkit install** as its primary dependency: it **vendors** CUDA runtime / cuDNN / cublas DLLs under `torch/lib` (or `nvidia-*` packages). Pointing at `CUDA_PATH` alone does **not** remove the ~4 GB wheel.

### Reuse strategies

| Strategy | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Probe another env’s site-packages** (conda/venv/`AppData\Roaming\Python`) and set `PYTHONPATH` / symlink into Pixi | Skip re-download when versions match | ABI / CUDA tag mismatch (`cu118` vs `cu124`); broken upgrades; hard for support; conflicts with `PYTHONNOUSERSITE=1` isolation | **Advanced / opt-in only**, with strict version checks |
| **Use system CUDA Toolkit libs** and install a “slim” torch | Smaller if it worked | Official Windows wheels are not packaged this way; version skew with Toolkit 12.2 vs wheel 12.4 | **Not recommended** |
| **“Bring your own torch” installer mode** | Power users with a known-good `torch==2.x+cu124` skip Pixi’s torch dep | Must validate `torch.cuda.is_available()`, device, and import `kornia`/`spandrel`; document unsupported configs | **Worth prototyping** as optional CUDA Full path |
| **Shared cache / offline wheel** (`PIP_CACHE_DIR`, Pixi cache, copy wheel from another machine) | Avoids re-download bandwidth; still uses disk once per install root | Does not reduce **on-disk install size** unless installs share one env | Good for **bandwidth**; separate from size |
| **Single shared Pixi/conda env** used by multiple apps | One torch on disk | Couples RAWviewer to foreign env lifecycle | Only for advanced users |

### Practical recommendation

1. **Default:** keep bundling an isolated Pixi env (predictable support).  
2. **Near-term size:** prune `*.lib` after CUDA install; don’t leave `RAWviewer_Setup.exe` in the install folder; prefer DirectML Full by default.  
3. **Optional later:** “Use existing PyTorch” checkbox that:
   - discovers candidates (`py -0p`, `%LOCALAPPDATA%\…\site-packages`, conda envs),
   - requires `torch` with CUDA, matching major.minor policy (e.g. `2.5–2.6` + cu12x),
   - verifies `import kornia` / demosaic smoke test,
   - records the chosen prefix in a local config (no silent fallback to a broken env).

**Implemented (2026-07-19):** CUDA Full installer now **probes first** via
`torch_provider.discover_external_torch()`. If a torch 2.x + CUDA 12.x install
is found, it installs `pixi-cuda-byo` (no torch wheel), writes
`torch_provider.json`, and binds that site-packages at launch. If the external
provider later disappears, `ensure_torch_for_gpu()` notifies the user and
downloads cu124 into the local Pixi env. Bundled install still runs only when
no external provider passes validation.

Do **not** blindly honor a generic “CUDA on PATH” as a substitute for the torch wheel.

---

## 4. Product implications (from this review)

1. **CUDA Full stays opt-in** — speedup is real but modest vs ~6× disk.  
2. **Safe prune:** drop `torch/lib/*.lib` (~770 MB) in the installer post-step (**implemented** in `bootstrap._prune_torch_link_libs`).  
3. **Do not** chase per-DLL cherry-picks without a dedicated compatibility test matrix.  
4. **Reuse existing torch** is a possible advanced feature; **reuse CUDA Toolkit alone** is not a substitute for the wheel.  
5. Longer-term size wins: SCUNet→ONNX/DirectML; optional non-torch demosaic rewrite — not “trim torch/lib to only our Python calls.”

---

## 5. CPU Fast RAW concurrency (follow-up)

Bottleneck was not only worker-pool size: in-process LibRaw unpack was gated to **1** concurrent context (`RAWVIEWER_UNPACK_CONCURRENCY`), while process pool was often forced off (`RAWVIEWER_USE_PROCESS_POOL=0`) even when GPU demosaic was unused.

**Changes (2026-07-19):**

| Lever | Before | After |
|-------|--------|--------|
| Unpack gate | default **1** | adaptive **1–3** (cores + RAM); Lite setdefault **2**; low-RAM tier forces **1** |
| Process pool (Windows release) | forced `0` in `pyi_rth_release_defaults` | auto: on when GPU demosaic **not** in use and CPU ≥ 4 |
| Pixi day-to-day | `USE_PROCESS_POOL=0` | unset (same auto rule) |

Override anytime: `RAWVIEWER_UNPACK_CONCURRENCY=1..4`, `RAWVIEWER_USE_PROCESS_POOL=0|1`.

### A/B on this machine (RAW_Sample 259, CPU Fast RAW, cache cleared)

| Setting | Cold median | Cold mean | Wall | Notes |
|---------|-------------|-----------|------|-------|
| `UNPACK=1` (old default) | 0.840 s | 1.335 s | 6.4 min | one 60 s TIMEOUT |
| adaptive / `=3` (new default here) | **0.749 s** | **1.013 s** | **4.9 min** | 0 TIMEOUT |
| `UNPACK=3` explicit | 0.799 s | 1.049 s | 5.2 min | ≈ adaptive |

Wall-clock improved ~**1.3×** vs serialized unpack (prefetch benefits more than single-image median). Live install also pruned **22 × `.lib` → −769.5 MB** (`torch/lib` 4358 → 3589 MB).

---

## 6. Evaluation: rewrite demosaic vs custom torch

### Option A — Rewrite GPU demosaic off PyTorch

| Aspect | Assessment |
|--------|------------|
| Goal | CUDA Full without ~4.5 GB torch; keep GPU demosaic |
| Approach | OpenCL / CUDA kernels / compute shaders for CFA → RGB + colour matrix; keep CPU path as cv2 EA |
| Effort | **Large** (weeks–months): parity tests across Bayer patterns, bit depths, orientation, half-size, cancel, VRAM budgeting |
| Risk | Quality/perf regressions; no kornia; SCUNet still needs a NN runtime unless also ported |
| Fit | Best **long-term** if CUDA Full must stay under ~2 GB |

**Verdict:** Viable product direction, **not** a quick size win. Do only with a dedicated milestone and golden-image suite.

### Option B — Custom / slim PyTorch build

| Aspect | Assessment |
|--------|------------|
| Goal | Ship a torch that omits unused CUDA libs / ops |
| Approach | Build from source with `USE_CUDNN=0` / selective CUDA libs, or community “lite” wheels |
| Effort | **High ongoing**: own CI for Windows cu12x wheels, track every torch upgrade |
| Risk | Breakage when kornia/spandrel need an op you stripped; support burden |
| Saving | Uncertain; `torch_cuda.dll` alone is ~900 MB and hard to carve |

**Verdict:** **Poor ROI** for an app installer vs Option A or “CUDA opt-in + DirectML default.” Prefer prune `*.lib` + optional BYO-torch later.

### Option C — Keep stock torch, shrink elsewhere (chosen near-term)

1. Prune `*.lib` (~0.7–0.8 GB) — **done**.  
2. DirectML Full default; CUDA Full opt-in.  
3. Improve CPU Fast RAW concurrency so Lite/DirectML stay competitive (above).  
4. Optional later: SCUNet→ONNX/DirectML; BYO existing `torch+cu12x` env.
