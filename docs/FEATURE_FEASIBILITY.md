# Feature Feasibility Research

**Status:** Research notes (dev branch, targeting 3.1.x+)
**Last updated:** 2026-07-25

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

### Masks — types & edit categories (evaluated)

Concrete scope, evaluated against commercial masking (Lightroom Classic,
Capture One Layers) and against what's actually reusable in this codebase.

**Mask types** — all five proposed are feasible; two notes:

| Type | Feasibility | Notes |
|---|---|---|
| Linear Gradient (+reverse) | Trivial | Pure alpha ramp by angle/position, no ML. |
| Brush (+reverse) | Small | ~90% exists. `DodgeBurnMask.data` is a **signed** float32 exposure map in `[-1.5, 1.5]` (`raw_dodge_burn.py`); generalizing means re-encoding to a plain 0–1 alpha buffer. Stamping, edge-assist flood-fill, brush cursor, size/flow gestures all carry over. |
| Radial mask (+reverse) | Small | Same alpha-ramp math as Linear, elliptical distance. **Recommend merging** the originally-separate "circular mask" into this as one primitive with a **Feather** parameter (0 = hard edge, 100 = soft falloff) — every commercial tool (Lightroom's Radial Gradient, Capture One's Radial Mask) models it this way; a separate hard-circle-only tool would just duplicate what Brush-at-a-fixed-size already covers. |
| Sky Detection (+reverse) | Medium | Needs a real segmentation model (see correction above) — not free from a platform API. |
| Focus/Subject detection (+reverse) | Medium | Vision (mac) / ONNX (win); see the AF-point-seeding idea above. |

Reverse/invert is the same one-line op (`1 - alpha`) for all five — build it
once in the shared mask chrome, not per-type.

**Edit categories per mask** — Light, WB/Saturation/Vibrance, Tone Curve,
HSL/Color Mixer, Sharpness/Clarity, **plus Dehaze** (added below). One scope
call worth naming explicitly: **Tone Curve and full HSL/Color Mixer per mask
exceed Lightroom Classic's own masking panel** (LR's masks get Light + WB +
Saturation + Clarity/Dehaze/Texture + Sharpness/Noise, but no full parametric
curve and no full 8-channel mixer inside a mask, even in current versions).
This scope matches **Capture One Layers** instead, whose layer architecture is
closer to true image layers. Worth deciding that target deliberately rather
than discovering it mid-build.

- **Light, WB/Sat/Vibrance:** direct reuse of existing global math
  (`_apply_saturation_vibrance` in `raw_edit_pipeline.py`), only per-mask
  compositing is new. **Include Temperature alongside Tint** — the original
  list had Tint without Temp, an asymmetry: local white balance (both
  together) is one of the most common real edits (warm a shaded face while a
  sunlit background stays neutral), and every commercial tool ships both.
- **Tone Curve is the single most expensive item, cost-wise.** Confirmed in
  `raw_pv2012.py`: the global curve builds a **65,536-entry LUT** per
  evaluation. Doing that N times per tick (once per mask) stacks directly on
  top of Track A's per-mask compositing cost — the item to stress-test first
  in the Phase 0 spike.
- **HSL/Color Mixer** reuses the existing float32 HSV-scale math directly — a
  per-pixel remap, not a rebuilt LUT, so meaningfully cheaper than Tone Curve
  despite looking similarly "big."
- **Sharpness/Clarity** has a concrete implementation trap: sharpening is a
  neighborhood op, so filtering tightly cropped to a mask's bounding box
  fringes at the edge (the kernel sees replicated/wrong border pixels right at
  the boundary). `raw_spot_heal.py` already solved this exact problem — it
  dilates its ROI before running Telea "so \[the filter\] has a clean border
  of source pixels" (`raw_spot_heal.py:191`). Same pad-before-filter,
  crop-after pattern applies here.
- **Dehaze, added:** `raw_effects.py` already has global `apply_dehaze` — the
  math exists, only regional compositing is new. Pairs narratively with Sky
  Detection specifically ("dehaze the sky" is probably the most common reason
  anyone masks a sky at all).

**Not recommended for this scope:** per-mask Noise Reduction (deferred by
request, despite the existing AI-denoise infra making it cheap) and per-body-
part face/skin/hair masks (a materially bigger model + UI investment, out of
scope here).

**Other gaps found and now in scope:**
- **Range masks (Color Range, Luminance Range)** — the biggest miss from the
  original five/five list. Present in both Lightroom and Capture One, and
  categorically different from every type above: they select by pixel *value*
  (luminance/hue), not geometry or object identity ("everything above 80%
  luminance" to grab a blown sky). Commonly combined with a brush or gradient
  as a refinement rather than used standalone.
- **Mask composability (Add/Subtract blend against the mask below).** This is
  how Lightroom's own masking panel actually works internally — masks are
  composed of sub-shapes, not siloed types. The common expert workflow
  ("Select Subject, then Subtract the hand with a brush") needs this. Full
  nested sub-component composability is a bigger UI lift; a cheaper interim
  that gets most of the value: a simple **Add / Subtract blend mode per stack
  entry** against the entry below it.
- **Duplicate mask** (copy a mask + its settings) — minor, near-zero cost once
  the data model exists.

### Mask count / memory cap

**Yes — cap it, on the same shape as focus stacking's `_MAX_FRAMES`.** Each
masked layer isn't just an alpha buffer; per-mask *adjustments* (not just
dodge/burn's single exposure gain) need their own full working RGB render to
composite, plus a cache buffer to avoid recompute on unrelated slider ticks
(mirroring the existing single-mask `_gain_cache` pattern). Estimated
per-mask footprint at the mask's working resolution (which matches the
half-res edit base, per `_dodge_burn_mask_shape`'s docstring — not full
sensor res): **alpha buffer (4 B/px) + one float32 RGB working render
(12 B/px) + cache overhead ≈ 16–24 B/px per mask.**

| Sensor | Half-res working px | Memory per mask (16–24 B/px) | 10 masks | 20 masks |
|---|---|---|---|---|
| 24 MP | ~6.0 M px | ~96–144 MB | ~1.0–1.4 GB | ~2.0–2.9 GB |
| 45 MP | ~7.7 M px | ~123–185 MB | ~1.2–1.8 GB | ~2.5–3.7 GB |
| 61 MP | ~15 M px | ~240–360 MB | ~2.4–3.6 GB | ~4.8–7.2 GB |

That's for the mask working set alone — on top of the base image, undo
history, and the gallery/preview caches already resident. Unbounded N breaks
**both** memory and the 80ms live-preview budget (more masks = more
per-tick pipeline passes), so the cap is doing double duty, same as
`_MAX_FRAMES` in focus stacking protects both RAM and correctness there.

There's no hard precedent to match from a competitor — neither Lightroom nor
Capture One enforce a UI-level cap — but community guidance in both
consistently points at noticeable slowdown past a couple dozen complex masks
on large files, which is roughly where the table above also turns
uncomfortable.

**Recommendation — two-part cap, mirroring `RAWVIEWER_FOCUS_STACK_MAX_FRAMES`:**
1. **Byte-budget primary.** Compute `N × per-mask-bytes-at-current-resolution`
   against a configurable budget (default ~1.5–2 GB, env-overridable). This
   scales the *effective* limit with the image's actual working resolution —
   more masks allowed on a 24 MP file than a 61 MP one — rather than a flat
   count that's wrong at either extreme. (This also avoids repeating a mistake
   already found and fixed elsewhere this session: the gallery thumbnail
   cache was a **count-capped** `LRUCache(10000)` on a byte-varying resource,
   which is exactly the wrong shape for this same reason.)
2. **Hard count backstop** (default 24, env-overridable) regardless of bytes —
   so a low-res image can't let someone create hundreds of masks and choke the
   per-tick compositing loop even though bytes stay cheap.
3. Reject with a clear message (same UX as `focus_stack`'s frame-count
   rejection) rather than silently degrading — "Add Mask" disables past the
   limit with a tooltip naming the reason, not a slowdown the user has to
   diagnose themselves.

### Two tracks

**Track A — general mask editing (the expensive one)**

| Piece | Notes | Size |
|---|---|---|
| Pipeline generalization | Each masked layer renders its own adj set + alpha-blends. **The critical path** — perf vs the 80ms preview throttle (a single full-frame gain map already costs ~56ms). Needs per-mask stage caching. | Large |
| Data model + XMP | List of `{mask, adjustments}` layers. Decide Lightroom `MaskGroupBasedCorrections` round-trip vs own sidecar schema — **early product decision**. | Medium–Large |
| Mask UI | Mask list, per-mask scoped sliders (reuse slider components), overlay viz (`toggle_dodge_burn_show_mask` partly exists). | Medium–Large |
| Mask primitives | Brush exists (refactor stamping to emit 0–1 alpha, not signed exposure); linear/radial gradients small (radial subsumes "circular" via Feather=0); Add/Subtract blend mode small; luminance/color range masks medium; per-mask Dehaze small (math already global). | Small–Medium |

**Track B — rides on Track A's data model** (an AI mask is just another alpha
layer). See [Object-Detection Masking](#object-detection-masking--evaluated).

### Sizing & recommendation

| Phase | Scope | Size |
|---|---|---|
| 0 | Data model + one brush mask driving a small adj subset (exp/contrast/temp) — the perf spike | ~1–1.5 wk |
| 1 | Full per-mask adjustments + pipeline perf + mask-list UI | ~2–4 wk |
| 2 | Linear/radial gradients (feather-merged), color/luminance range masks, Add/Subtract blend, per-mask Dehaze, duplicate mask, mask-count cap | ~1.5 wk |
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
  Vision gives person segmentation (`VNGeneratePersonSegmentationRequest`) and
  saliency-based subject selection. Small–Medium per type.
- **Correction:** an earlier draft of this doc claimed Vision's horizon API
  covers Sky detection — it doesn't. Vision's horizon request returns a **line
  angle**, not a pixel mask (RAWviewer's own auto-straighten already gets this
  classically, via Hough lines in `raw_auto_adjust.py`, no ML involved). A real
  Sky mask needs a genuine small segmentation model, same as Subject/Person —
  ship one cross-platform ONNX model rather than leaning on a platform API
  that doesn't actually produce what's needed. Sizes below in [Masks — types &
  edit categories](#masks--types--edit-categories-evaluated) already reflect
  this.
- **RAWviewer-specific opportunity:** the app already extracts the camera's AF
  point (`exif_subject_area.py`) — a strong prior no generic photo tool has at
  edit time (a JPEG viewer never sees the original capture metadata). Seeding
  subject segmentation with the real focus point, rather than guessing from
  pixels alone, is a plausible differentiator worth prototyping alongside the
  base segmentation model.
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
