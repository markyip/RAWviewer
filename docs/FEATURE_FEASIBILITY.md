# Feature Feasibility Research

**Status:** Research notes (dev branch, targeting 3.1.x+)
**Last updated:** 2026-07-24

Forward-looking feasibility and effort evaluations for candidate editing
features. Each entry records the conclusion, the effort sizing, the constraints
that shape it, and the existing infrastructure it can lean on. Effort figures are
T-shirt sizes / rough week-ranges for one focused developer, not commitments.

Cross-cutting leverage and constraints that recur below are collected in
[Shared infrastructure & constraints](#shared-infrastructure--constraints).

---

## Focus Stacking — prototype landed

**Status:** Prototype on branch (`src/focus_stacking.py`, Stitch & Merge menu,
`t_focus_stacking.py`). v1 + v2 implemented.

All-in-focus merge of frames shot at different focal planes. Reuses the existing
HDR/panorama scaffold (dialog, background worker, TIFF output) — the only new
engine is `focus_stacking.py`.

- **Alignment:** ECC affine, **not** the HDR path's MTB. Focus stacks share
  exposure but not magnification (focus breathing drifts scale/perspective),
  which MTB (translation-only) cannot absorb.
- **Fusion:** per-image sharpness (smoothed |Laplacian|) → Laplacian-pyramid
  blend weighted by sharpness (seam-free), bit-depth preserved.
- **v2 — parallax (option 2):** dense DIS optical-flow refinement after the
  global affine, estimated on heavily-blurred luma so focus differences don't
  dominate. Confidence-gated on reference structure (contrast^1.5) + keep-
  original-where-untrusted, because naive flow warping *halved* sharpness in
  reference-blurred regions. Measured: cut a local depth-shift error ~45–50%
  (residual ratio 0.45–0.48 at the chosen tuning) while retaining ~75% of
  clean-stack sharpness. Reports a parallax metric → user warning above ~2px.
- **v2 — seam-aware fusion (option 3):** argmax-sharpest source, regularized
  with an edge-aware guided filter so a contested (parallax) region is assigned
  wholesale to one frame instead of composited from two.

**Open items:** the confidence-gating knob (parallax reduction vs sharpness
retention) is tuned on synthetic stacks — needs real handheld/macro frames.
Memory: N frames + working copies resident; capped at `_MAX_FRAMES=30`
(env-overridable), sources released before the pyramid, intermediates freed as
consumed. Panorama parallax is unchanged **by choice** (`cv2.Stitcher`
homography + seam finding is adequate for rotational panoramas); porting the
flow refinement there was explicitly deferred. The concurrency fixes (worker →
GUI-thread signal marshaling, N cap, intermediate freeing) landed after the
prototype commit, in the gallery/editing concurrency pass.

---

## Mask Editing / Local Adjustments — evaluated

**Status:** Not started. Docs currently list Lightroom-style masks as
unsupported.

**The critical fact:** today's masking is single-purpose. `DodgeBurnMask.data`
is a signed float32 exposure map applied only as `img * 2**(mask * stops)` —
local *brightness only*. Spot-heal adds an inpaint mask. Every other adjustment
(WB, tone, HSL, saturation, sharpness) is applied **globally**. There is no
general "apply arbitrary adjustments through a mask" system to wire into. The
dominant cost is generalizing the engine from one global adjustment set to
**layered masked adjustments**.

### Two tracks

**Track A — general mask editing (the expensive one)**

| Piece | Notes | Size |
|---|---|---|
| Pipeline generalization | Each masked layer renders its own adj set + alpha-blends. **The critical path** — perf vs the 80ms preview throttle (a single full-frame gain map already costs ~56ms). Needs per-mask stage caching. | Large |
| Data model + XMP | List of `{mask, adjustments}` layers. Decide Lightroom `MaskGroupBasedCorrections` round-trip vs own sidecar schema — **early product decision**. | Medium–Large |
| Mask UI | Mask list, per-mask scoped sliders (reuse slider components), overlay viz (`toggle_dodge_burn_show_mask` partly exists). | Medium–Large |
| Mask primitives | Brush exists (refactor stamping to emit 0–1 alpha, not signed exposure); linear/radial gradients small; add/subtract/intersect + luminance/color range masks medium. | Small–Medium |

**Track B — rides on Track A's data model** (an AI mask is just another alpha
layer). See [Object-Detection Masking](#object-detection-masking--evaluated).

### Sizing & recommendation

| Phase | Scope | Size |
|---|---|---|
| 0 | Data model + one brush mask driving a small adj subset (exp/contrast/temp) — the perf spike | ~1–1.5 wk |
| 1 | Full per-mask adjustments + pipeline perf + mask-list UI | ~2–4 wk |
| 2 | Linear/radial gradients, range/luminance masks, combination | ~1 wk |
| 3 | AI masks: subject/person/sky (Vision on mac, one ONNX model on win) + refinement | ~1.5–2 wk |
| 4 | Click-to-segment (MobileSAM) | ~1–2 wk |

~1.5–2.5 months total for a Lightroom-comparable suite; the shape matters more
than the number. **Phases 0–1 (data model + pipeline perf) are the risk and the
gate.** Recommendation: build **Phase 0 as a throwaway-friendly perf spike** —
one brush-alpha mask carrying 3 sliders (exposure/contrast/temperature) through
`raw_edit_pipeline`, benchmarked against the 80ms throttle on a real RAW. That
single question decides whether this is a ~6-week or a ~4-month project. Don't
write UI until the spike passes.

**Watch:** preview perf (N layers × pipeline vs 80ms); Windows bundle size;
cross-platform seg parity; XMP compatibility (painful to retrofit).

---

## Object-Detection Masking — evaluated

**Status:** Not started; depends on Mask Editing Track A's data model.

Auto-generate masks (subject / person / sky / arbitrary object). Well-supported
and additive once masks exist — the infra is already proven in-repo.

- **macOS:** `pyobjc-framework-Vision` + `CoreML` are already dependencies.
  Vision gives person segmentation (`VNGeneratePersonSegmentationRequest`),
  saliency-based subject selection, horizon/sky hints. Small–Medium per type.
- **Windows/cross:** `onnxruntime` / `onnxruntime-directml` already ship (SCUNet,
  Restormer, MobileCLIP). Adding a segmentation model (U²-Net salient object,
  sky/person, or MobileSAM click-to-segment) is a solved pattern. Medium per
  model.
- **Refinement:** reuse `cv2.ximgproc.guidedFilter` (focus stacking) and
  `edge_snap_region` (dodge/burn). `raw_dodge_burn.py` already has a
  `MASK_OBJ_KEY` slot anticipating an object mask.

Sizing: subject/person/sky + one ONNX model ~1.5–2 wk; click-to-segment
(MobileSAM) ~1–2 wk (bundle/download tradeoff). **Watch:** Vision (mac) vs ONNX
(win) produce different masks → UX-consistency work; each seg model adds MB.

---

## Content-Aware AI Removal / Recovery — evaluated

**Status:** Not started. Current heal = `cv2.inpaint` (Telea), small blemishes
only.

**Reframe:** "removal/recovery" and diffusion inpainting pull in different
directions. Removal/recovery fills from *what the scene plausibly contained*
(the surroundings are the answer); diffusion *invents* content. For pure
removal, diffusion is the expensive, wrong-shaped tool — the features people call
"magic eraser" are **LaMa-family feed-forward inpainting**, not diffusion.

| Tier | What | Fit | Size |
|---|---|---|---|
| 1 | Classical exemplar/PatchMatch (upgrade `cv2.inpaint`) | Perfect fit, weak on large/structured holes | Small–Medium |
| 2 | **LaMa via ONNX** — feed-forward AI removal | **Strong — recommended** | ~1.5–2.5 wk |
| 3 | Diffusion inpaint | Fights the architecture | Large + conflict |

**Recommend Tier 2 (LaMa).** ~200MB, fully-convolutional (high-res RAW native,
unlike diffusion's fixed 512/1024 latent), single forward pass on
DirectML/CoreML/CUDA, fills from real surrounding structure (recovery, not
fabrication). It's the idea minus the diffusion baggage and matches
"removal/recovery" better than diffusion does.

**Why diffusion specifically fights this app:**
1. **Torch-free + bundle-conscious** — SD-inpaint is a multi-GB UNet+VAE+text-
   encoder stack needing a GPU; head-on collision with the packaging philosophy.
2. **Color fidelity** — a ColorChecker-calibrated scene-linear RAW tool inventing
   8-bit sRGB pixels contradicts its identity.
3. **Resolution & speed** — fixed latent size → tiling/seams on 40MP; 10–40s CPU.

**Leverage (Tier 2 is mostly swapping the inpaint kernel):**
- `scripts/models/export_*.py` already export SCUNet/Restormer/NAFNet to ONNX —
  LaMa export is the same workflow.
- `onnx_scunet.py` is a ready ONNX-inference template (DirectML/CUDA/CoreML).
- Download-on-demand exists (`hf_hub_download`, MobileCLIP/YuNet progress UI) —
  so ~200MB is a download, not bundle weight.
- `raw_spot_heal.py` already does crop-to-ROI → 8-bit decode → inpaint →
  soft-composite back into the linear pipeline. LaMa drops into that exact seam.
- Brush masking for the removal stroke.

Tier 3 (diffusion) only realistic as an optional GPU-only huge download or a
**cloud API** (breaks offline/privacy, ongoing cost). Reserve for a future,
clearly-separated *generative fill (addition)* feature — honestly labeled as
invention, not recovery.

**Recommended spike:** a LaMa ONNX module modeled on `onnx_scunet.py`, wired into
`raw_spot_heal.py`'s ROI/composite path with a download-on-demand hook — proves
removal quality and linear-composite handling before UI work.

---

## Appendix: HE-NEF decode — not feasible (constraint, not a feature)

Nikon High Efficiency / HE\* (Z8/Z9/Z6III/Z50II) cannot be decoded, and this is a
**legal wall, not an engineering backlog item**:

- The pixel data uses **intoPIX TicoRAW** (a JPEG-XS-family wavelet codec Nikon
  licenses), not classic Nikon lossless/compressed NEF. LibRaw/dcraw are open
  source and cannot legally ship a licensed decoder — hence "no upstream ETA".
- The algorithm (wavelet + rate control) is also far harder to reverse-engineer
  than the legacy Huffman-predictor NEF.
- The app detects HE via `_detect_nef_he_compression` (MakerNotes 0x93 / 0x51)
  and refuses cleanly, still showing the embedded JPEG preview (browsable, not
  editable — the sensor demosaic is the licensed part).

**User path:** convert via Adobe DNG Converter or Nikon NX Studio (both
licensed) → DNG decodes everywhere.

---

## Shared infrastructure & constraints

**Leverage (reasons the above is cheaper than it looks):**
- **ONNX runtime** with DirectML/CUDA/CoreML providers, proven on SCUNet,
  Restormer, MobileCLIP; `onnx_scunet.py` is a reusable inference template and
  `scripts/models/export_*.py` a reusable export workflow.
- **Download-on-demand** for large models (`hf_hub_download`, progress UI) —
  keeps optional models out of the installer.
- **macOS Vision + CoreML** already dependencies (segmentation, saliency).
- **cv2 5.0 contrib** (guided filter, DIS flow, MCC, multiband blend, seam
  finders).
- **`raw_spot_heal.py`** ROI-crop → decode → process → linear-composite seam,
  reusable by any region-based ML op.
- Brush masking, `edge_snap_region`, mask overlay.

**Constraints (reasons some of the above is bounded):**
- **Torch-free by design** — CuPy + ONNX; the ~2.4GB torch wheel is deliberately
  avoided. Rules out heavy diffusion stacks.
- **Windows bundle size** sensitivity (cv2 already the largest component) — each
  bundled model adds MB; prefer download-on-demand. Precedent: the switch to
  `opencv-contrib-python-headless` for ColorChecker Auto-Detect (`cv2.mcc`) was
  accepted at ~+10MB compressed / ~+26MB installed — cv2 is one monolithic
  binary, so a contrib module cannot be taken alone. Future size asks can cite
  that bar. (Related trap, for anyone touching MCC: `CCheckerDetector.process()`
  takes `nc` — a chart *count* — not a chart type; `mcc.MCC24` is 0.)
- **Scene-linear / color-accurate identity** — ColorChecker-calibrated; any ML
  op that fabricates or tints pixels must justify itself against this, and must
  composite correctly between display space and the linear edit pipeline.
- **80ms live-preview throttle** — the hard budget any per-tick editing work
  (especially masked layers) must fit inside.
