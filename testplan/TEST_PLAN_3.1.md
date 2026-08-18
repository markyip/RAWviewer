# RAWviewer 3.1.0 vs 3.0.2 — stability / performance test plan

Manual walkthrough for testers. Automated suites stay in [`TEST_PLAN.md`](TEST_PLAN.md) (`./testplan/run_all.sh`).

**3.1.0 is a stability + performance patch, not a feature tour.** Do **not** spend this pass on masks, pairing, MCP, Share to AI, ColorChecker, merges, labels, or export-queue UX. Feature checklists wait for **3.2+**.

**Baseline:** tag **3.0.2**. **This version:** `development` plus optional dirty-tree cache/bubble hardening. Record OS, edition, and whether the tree was dirty.

**Locked product rules (fail if violated):**

- No memory-tier picker in the UI.
- Generative editing UI must stay gone.
- Export of an edited RAW still matches on-screen preview and is **full resolution** (not the 640 px live-drag).

---

## 0. Setup

**Need:** a 3.0.2 mental baseline (gallery cull, Adjust global tone / D&B / heal / crop, JPEG/WebP/TIFF export, stars 0–5, Plus AI Denoise export). Folders: mixed RAW on a large folder; ≤16 GB machine if you have one; a Nikon HE/HE\* NEF if you have one; geotagged files for Map.

**Launch:** `pixi run start` or the packaged app. Splash / pairing / MCP menus may exist in the dirty tree — **ignore them** for this pass.

---

## 1. Gallery scroll and load (regression + 3.1 fixes)

| | |
|---|---|
| **Setup** | 1000+ RAW folder. Slow disk or ≤16 GB if possible. |
| **Steps** | Fast trackpad / wheel scroll. Jump the scrollbar. Open a folder, switch to another, switch back. Rapid arrow-key browse 20+ files. Scroll until a tile is grey, wait. Drag the scrollbar thumb (date bubble if present). |
| **Expected** | No freeze; no permanent grey holes; jump does not wait behind off-screen thumbs. Folder change does **not** rebuild every thumbnail from scratch (~27 s on 3000 files was the bug). Cancelled loads do not stall the viewport. Date bubble (if shown) does **not** flash or hitch every tick *(dirty tree)*. |
| **Fail** | Stuck grey tiles; must reopen folder to load; UI freeze; bubble flickers / hits disk. |

---

## 2. Memory / RAM-tier Adjust cap

| | |
|---|---|
| **Setup** | ≤16 GB machine if possible. RAW in Adjust (**E**). |
| **Steps** | Drag Exposure (live). Release (settle). Confirm **no** Low/Medium/High RAM control anywhere. Export JPEG/TIFF. |
| **Expected** | Live-drag stays light; settle does not feel like a 12 MP re-render on 8–16 GB. Export is sharp / full-res, not the preview size. `[PROFILE] memory tier=…` in a Terminal launch is enough; users never pick a tier. |
| **Fail** | Settings page for memory; export matches the 640 px preview; 8 GB machine behaves like 24 GB prefetch (fans, swap, jetsam). |

---

## 3. Cache / sidecar / EXIF

| | |
|---|---|
| **Setup** | Photo with existing XMP (stars + tone from 3.0.2). |
| **Steps** | Open Adjust, change nothing, close. Revisit: should not feel like a full re-decode. Rate 1–5, restart app, confirm stars on tiles. Edit Exposure, save, browse away, return: edited look paints. |
| **Expected** | No-edit visit does not drop every cache tier. Sidecar still round-trips. Stars survive restart (EXIF cache). *(Dirty tree:)* fields must not live only in RAM then vanish. |
| **Fail** | Stars gone after restart; opening Adjust always re-decodes; default sliders wipe a 3.0.2 sidecar. |

---

## 4. Export crash / band / AI denoise cancel (Plus)

| | |
|---|---|
| **Setup** | Edited RAW. Plus with denoise model if testing AI path. |
| **Steps** | Export JPEG, WebP, 16-bit TIFF — match preview, no band seams. Plus: **AI Denoise** export; **Cancel** mid-run; retry. First-use download (if needed) must not be treated as a timed-out failure. Standard: no AI Denoise / Upscale entries even if model files sit on disk. |
| **Expected** | Cancel stops; no partial file left behind; progress updates; banded path still runs denoise (3.0.2 skip bug). Timeout/cancel must not hang the UI. |
| **Fail** | Hang; silent skip of denoise; leftover `.jpg` half-file; Standard shows Upscale. |

Do **not** treat export **queue**, filename template, GPS strip, or watermark as in-scope for 3.1.0. If those UI bits appear (dirty tree), skip them.

---

## 5. HE/HE\* NEF and undecodable RAW

| | |
|---|---|
| **Setup** | Nikon HE or HE\* NEF; optionally a RAW LibRaw cannot demosaic. |
| **Steps** | Browse. **E** on HE-NEF. Export. Browse an undecodable RAW. |
| **Expected** | HE-NEF opens Adjust via embedded JPEG (8-bit; WB/highlights have little headroom) — **no refuse / error dialog**. Undecodable RAW shows embedded preview, **not** a permanent grey tile. Prefetch does not spam errors. |
| **Fail** | “Cannot edit” / corrupt dialog; grey tile forever; log storm. |

---

## 6. Crash / thread / cancel smoke

- [ ] Rapid gallery scroll + folder change + Adjust open/close: no crash
- [ ] Cancel a load by jumping away immediately after requesting a full-res
- [ ] Plus CUDA: browse + Adjust + export without GL/CUDA teardown crash
- [ ] Idle on a folder with a few broken files: no error-storm freeze

---

## 7. Regression vs 3.0.2 (the product you already shipped)

Walk **3.0.2** so 3.1 did not break it. Skip 3.2+ surfaces.

- [ ] Gallery justified grid, size slider, capture-time order
- [ ] Cull: **1–5**, **0**, **↓** Discard (single view only), Delete
- [ ] Compare (**C**): sync zoom
- [ ] Map (**M**) on geotagged files
- [ ] Plus search: `sunset on beach`, `has:face`
- [ ] Adjust global: tone, WB presets, crop, D&B **D/B/X**, Heal **H**, LUTs, presets
- [ ] **P** / **J** in browse still RAW/DNG-only; **P** is Reconstruct preview, not baked by a visit to Adjust
- [ ] XMP round-trip with Lightroom: stars + Exposure
- [ ] No generative editing controls
- [ ] UI still English (an `i18n/` folder on disk is not a language picker)

---

## Out of scope (wait for 3.2+)

Masks tab, AI masks, Auto Tone, HDR/pano/focus stack, ColorChecker, anamorphic, raster develop, RAW+JPEG pairing, labels/keywords, MCP, Share to AI, splash, LUT thumbs, DPReview button, Adjust dock, export queue/template/watermark.

Those plans are **not** this file. Do not expand this pass into a 15-area tour.
