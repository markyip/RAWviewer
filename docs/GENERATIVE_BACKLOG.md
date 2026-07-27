# Generative editing — parked

**Status:** backend complete and tested, **no UI**. Nothing in the app reaches
this code; it is reachable only from tests. Parked on `generative-editing`.

**Why parked:** no generative model that is actually good is small enough to
embed, and the ones small enough to embed are not good. Details below — the
conclusion is the interesting part, not the code.

---

## What exists and works

| module | what it does |
|---|---|
| `raw_generative_edit.py` | Provider interface, `HttpEndpointProvider` (custom JSON), `StubProvider`, `make_provider()` |
| `raw_generative_local_server.py` | `LocalServerProvider` — OpenAI `/v1/images/edits` client, for a model running on the user's own machine |
| `raw_generative_local.py` | `LocalInstructPix2PixProvider` — InstructPix2Pix on ONNX Runtime, fully in-process |
| `clip_bpe_tokenizer.py` | CLIP byte-pair tokenizer in pure Python (no `transformers`) |
| `onnx_ip2p_scheduler.py` | Euler-ancestral scheduler in numpy (no `diffusers`) |
| `generative_session.py` | Staging: results live in a temp dir and become files only on export |
| `generative_derived_file.py` | Derived-file naming, writing, provenance sidecars, lineage |
| `generative_settings.py` | QSettings keys, and the consent gate |
| `rawviewer_ui/generative_setup_dialog.py` | Setup dialog — **orphaned**, nothing opens it |

Tests: `t_generative_edit`, `t_generative_derived`, `t_generative_session`,
`t_generative_local`, `t_generative_local_server`. All green.

## What was removed

The Generate tab, its page and section, the host wiring in `main.py` (worker,
handlers, navigation hooks, quit warning), and the two UI tests
(`t_generative_tab`, `t_generative_panel`). Recover from git history if the
feature is revived; the backend contracts they exercised are unchanged.

---

## The finding that stopped it

Every current instruction-edit model pairs a diffusion transformer with a
**multi-billion-parameter LLM or VLM as its text encoder**. That is what lets
it act on "remove the bin on the left" rather than keyword-match, and it is
roughly half the weight:

| model | text encoder | total |
|---|---|---|
| `microsoft/Mage-Flow-Edit-Turbo` | Qwen3-VL, 36L | 17.5 GB (8.9 GB encoder) |
| `microsoft/Mage-Flow-Edit-Base` | same | 17.5 GB |
| `black-forest-labs/FLUX.2-klein-4B` | Qwen3ForCausalLM, 36L | 23.7 GB (8.0 GB encoder) |
| `ddalcu/…-MLX-Serve-8bit` (Mage-Flow, 8-bit) | quantised | 9.7 GB |

Quantising the transformer works well — `FLUX.2-klein-4b-fp8` is 4.07 GB,
Apache-2.0, and GGUF reaches 3.07 GB at Q5 — but the encoder does not go away.
Realistic floor for a *good* edit model is ~3 GB transformer + ~2-3 GB encoder
+ VAE ≈ **6-7 GB**, resident.

InstructPix2Pix is 2.2 GB *only* because its text encoder is CLIP (250 MB,
2021). There is no middle ground: small means 2022-era, modern means 7 GB+.

## Measured, not estimated

**Embedded InstructPix2Pix** (ONNX, CoreML EP, M-series, 16 GB):
one UNet step of the three-branch batch = **10.7 s at 384×512**, **18.6 s at
512×512**. The reference 20 steps is a **3.6-6.2 minute** wait. Verified
working end to end ("make it winter" moved the blue channel mean 137.9 → 250.3)
— it is correct, just slow, and weak at object removal.

**Mage-Flow-Edit-Turbo via mlx-serve** on the same machine: model loaded
(9.07 GB resident), matched the source size, began its 4-step turbo schedule,
then `MLX error: [METAL] Command buffer execution failed` and the server died.
Free RAM at that point: **2.7 GB**. This needs ~12 GB free, i.e. a 32 GB
machine.

Also found: `mlx-serve pull` fetched only `model_index.json` for that repo
("1 files, 0 MB") and never walked the subdirectories — weights had to be
downloaded directly.

---

## If this is revived

**Ship the endpoint, not the model.** `LocalServerProvider` already covers
both positions that matter: `localhost` (mlx-serve, ComfyUI, Draw Things —
the photo never leaves the machine) and a remote API (best quality).
`requires_consent` is computed from the URL, so the two are distinguished
automatically. It costs zero bytes and never goes stale — Mage-Flow shipped
2026-07-21, mid-investigation.

**Do not ship the embedded InstructPix2Pix.** It is the only part that costs
real complexity — a hand-written tokenizer, a scheduler, a sampling loop, and
2.2 GB of weights from a third-party conversion repo with 2 downloads — to
deliver something slower and worse than the endpoint path. Keep it as a
reference implementation.

**Weights provenance:** the IP2P ONNX set is pinned by SHA-256 in
`raw_generative_local.py`, hashes taken from files that were downloaded and
verified end to end. The upstream conversion is unaudited against
`timbrooks/instruct-pix2pix`; confirming that needs torch to re-export and
compare.

**Unfinished when parked:** mask/subtract-aware generation, multi-reference
composition (mlx-serve supports repeated `image[]`), and a spawn-the-server
mode so the user never sees a second app.
