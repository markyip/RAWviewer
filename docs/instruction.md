# Masked local adjustments — the full workflow

Every step, as the code actually implements it, from opening a photo to
having an edited file on disk. Written against `development` at the time of
writing; file/line references are pointers to the authority, not decoration.

A short vocabulary first, because three different things in this app are
called a "mask" in casual speech:

| Term | What it is | Where it lives |
|---|---|---|
| **Mask** | One `MaskLayer`: a coverage map plus its own adjustment set | `raw_mask_layers.py` |
| **Group** | A mask whose coverage comes from several components combined | `MaskLayer.components` |
| **Overlay** | The coloured tint drawn over the photo showing coverage | `gpu_image_view.py` |
| **Dodge/Burn mask** | A separate, older brush system on the Global page | `raw_dodge_burn.py` |

The Masks page and Dodge/Burn are independent. They share one brush pipeline
(only one tool can be armed at a time — `adjust_panel.py:_on_mask_tool_toggled`),
but they store and apply their edits separately.

---

## 1. Open the editor

1. In the gallery, click a photo to open it in single-image view.
2. Press **E** to open the Adjust panel. **E** or **Esc** closes it again.
   Esc first disarms any armed brush (`_disarm_adjust_brushes_if_any`), then
   closes Adjust.
3. The panel opens on the **Global** page. Click **Masks** to switch pages.

Switching pages is not cosmetic — it changes what several keys mean, and it
hides the mask overlay (`adjust_panel.py:_on_panel_tab_changed`). Leaving or
entering a page also disarms mask / dodge brushes so a held tool does not
keep painting into the wrong page.

The header hint line under "Adjust" follows the page:

```
Global:  E closes · hold D / B / X / H = Dodge / Burn / Erase / Heal · scroll = size
Masks:   E closes · P = Add · X = Erase · again to put away · M shows mask · scroll = size
```

On Global, **D / B / X / H** are hold-to-paint: release the key to end the
stroke and put the tool away. On Masks, **P / X** are tap-to-toggle latches
(see §3). While Adjust is open on Masks, **P** is routed to mask Add rather
than RAW-recovery preview.

---

## 2. Create a mask

The **Create new mask** section holds every way to start one, as five buttons
— one press each.

| Button | What happens | Needs a model |
|---|---|---|
| **Brush** | Makes an empty mask. Does **not** arm Add — press **P** (or Add) to paint | no |
| **Smart Object** | One press; finds whatever stands out and masks it | yes (~214 MB) |
| **Sky** | One press; masks the sky | yes |
| **Depth** | One press; full-strength near, fading to nothing far | yes |
| **AI Selection** | Arms click-to-select; click a thing to mask that thing | yes |

Only **Brush** creates a layer up front. The AI tools each create their own
on completion, so arming one and changing your mind leaves nothing behind
(`adjust_panel.py:_on_mask_create`).

**Brush creates and persists immediately** (`main.py:_on_mask_layer_add` →
`_on_adjust_panel_editing_finished`). Empty layers are kept in the sidecar
(`mask_layers_xmp.serialize_stack`), so the first paint's Ctrl+Z steps back
the stroke instead of restoring "no masks at all".

Pressing **Brush** again while Add is already armed disarms Add rather than
spawning another empty mask.

The AI buttons are **Plus only**. Standard omits them entirely rather
than greying them out, because each downloads a model on first press
(`t_mask_edition_gating.py`).

**AI Selection stays lit while armed** — it waits for you to click on the
photo. Clicking a lit one puts it away. A line under the buttons also says
what the armed tool is waiting for, which a lit button cannot say.

**AI Selection is not a brush.** It takes a point, so the pointer is a
crosshair rather than a brush circle, and the Brush sliders are not offered
for it. A click that finds nothing (or returns an empty / speck selection)
leaves it armed, drops that prompt point, and says so — trying again is the
obvious next move. The floor is an absolute pixel count
(`_SAM_MIN_SELECTED_PX`), not a fraction of the frame: AI Selection is for
small specific things, and a coverage-% gate would refuse them.

**Smart Object / Sky / Depth / AI Selection** put up a progress dialog with
Cancel and a 90-second inference timeout (`_AI_MASK_TIMEOUT_MS`). The timeout
applies to inference only — on first use the model downloads and the wait is
left open, because no sane timeout covers that.

Two refusals worth knowing:
- If Smart Object / Sky return near-empty coverage, **no layer is added** and
  the app says so (`_AI_MASK_MIN_COVERAGE`). Depth is exempt: it is a fade
  across the whole frame, and a mostly-distant scene with one near subject
  can legitimately have tiny high-alpha coverage; "flat" is reported by the
  depth estimator returning `None`, not by this gate.
- Pressing **Smart Object** / **Sky** / **Depth** twice selects the existing
  mask of that kind instead of stacking a duplicate. Smart Object and Sky
  also nudge toward AI Selection for picking a specific object; Depth does
  not, because distance grading is not object selection.
- A used one-shot tool disables with an explanation
  (`set_ai_tool_used`). Only subject / sky / depth are one-shot; brush and
  AI Selection are ignored harmlessly.

---

## 3. Edit the mask's coverage

The **This mask** row acts on the selected mask only:

| Control | Key | Effect |
|---|---|---|
| **Add** | **P** (again to put away) | Brush more coverage into the selected mask |
| **Erase** | **X** (again to put away) | Remove coverage from it |
| **Invert** | — | The mask applies everywhere *except* its coverage |
| **⧉** | — | Duplicate: same coverage and adjustments, as a separate mask |
| **✕** | **Delete** | Delete the selected mask |

**P / X are tap-to-toggle latches on Masks**, not hold-to-paint. Press once
to arm; sweep or drag to paint while armed; press the same key again (or Esc,
or the button) to put the tool away. Key-up must not disarm — that is what
made "put the brush away" feel unreliable when the mode was hold-based.
Clicking Add / Erase arms or disarms without a key.

**P with no mask yet** creates one via `ensure_mask_layer_for_painting` (same
as Create → Brush, including the immediate persist). **X with no mask**
explains instead of arming nothing.

**Brush Size / Flow / Feather / Edge Assist** appear under **Brush** while a
brush is in hand. They mirror the Local section on the Global page — the same
values, kept in step both ways, so you do not have to leave the Masks page to
resize. **Two-finger scroll** also changes size (and horizontal scroll Flow)
while a brush is armed, including mid-stroke.

An **inverted** mask says so in its row (`Sky (inverted)`), because the row
thumbnail still shows the painted region — which is precisely the part an
inverted mask does *not* cover. Invert covers the **whole frame**, not only
the painted region.

Local mask coverage (and dodge / burn / heal) is sized to the **post-geometry**
frame — including anamorphic desqueeze — so preview, settle and Compare stay
aligned (`_dodge_burn_mask_shape`, `_ai_mask_source_rgb`, lite path order).

### Grouping

Drag one mask onto another to combine them into a single mask. The drop
target keeps its adjustments; the dragged mask's are discarded, because a
group holds exactly one adjustment set. Drag a component out to separate it
again. Both are undoable with **Ctrl/⌘+Z**.

---

## 4. Change visibility

Three rules, and everything defers to them:

1. **At most one mask's overlay is drawn.** Two coloured regions at once
   cannot be told apart, which is the opposite of what the overlay is for.
   Showing one mask hides the rest.
2. **Editing coverage shows it.** Arming Add or Erase, or starting a stroke,
   shows the mask you are about to change — and only that one.
3. **Editing the adjustments hides it.** Moving Exposure, Sharpness or any
   other mask slider puts the tint away, because you are judging the effect
   on the photo and the tint is the one thing in the way.

**The eye, per mask row.** Shows that mask and hides the others; clicking a
shown mask hides it, leaving nothing shown. It is view state only: not
written to the sidecar, not part of the render cache key, and toggling it
does not re-render or save. **The adjustment still applies either way** —
this controls the tint, not the edit.

Only top-level masks carry an eye. A group's parts do not: the overlay is
built from the top-level layers and `_combined_alpha_at` folds a group's
components together without consulting `overlay_hidden`, so a per-component
eye could not change anything on screen.

**M / the Mask button** switches the overlay off and on wholesale.
**Switching between Global and Masks** turns it off — the tint would
otherwise sit over the photo you switched pages to judge.

Exclusivity lives in `overlay_hidden` itself rather than in a separate
"solo" flag. That matters: an earlier attempt kept the two apart, and the
solo latched and stopped the eye from showing anything but the selected
mask.

> There is no UI control that disables a mask's *effect* while keeping it in
> the list. `MaskLayer.enabled` implements exactly that and is fully wired
> (compositor, overlay, fingerprint, sidecar) but has no switch attached.

---

## 5. Apply the adjustments

With a mask selected, the **Adjust** section drives that mask only. It is
grouped the way the Global page groups the same controls:

- **Light** — Exposure, Contrast
- **Color** — Temp (rel), Tint, Saturation, Vibrance
- **Detail** — Sharpness, Clarity, Dehaze, Defringe

Each group folds, using the same `CollapsibleSection` as the Global page, and
remembers its state for the rest of the run. All three start open — the Masks
page has nothing else to scroll past, and adjusting the selected mask is the
reason to be on it.

Click a slider's **value readout** to reset it to 0, as on the Global page.

This is every adjustment the compositor can apply — the panel and
`SUPPORTED_ADJUSTMENT_KEYS` are held equal by a test. Notably **absent**:
Highlights, Shadows, Whites and Blacks live in the global PV2012 tone LUT and
have no per-region implementation; Tone Curve is excluded on cost.

Temperature and Tint here are **relative** (±100), unlike the Global page's
absolute Kelvin slider.

Masks composite back to front — "top paints last". Each mask's adjustment is
computed against the image as the masks below it left it, so overlapping
masks stack the way a real local-adjustment tool behaves.

### What renders when

| You do | What happens |
|---|---|
| Drag a mask slider | Throttled preview re-render |
| Release the slider | Full render, and the sidecar is written |
| Paint mid-stroke | **Overlay only** — the photo is not re-rendered |
| Release the brush | One full render, and the sidecar is written |
| Toggle the eye | Overlay rebuild only — no render, no save |

Mid-stroke rendering was removed because masks composite before denoise
inside one cache stage, so each stamp invalidated WB, exposure, dodge/burn,
heal and denoise together — ~1360 ms at a 2200×3300 base, requested eight
times a second.

---

## 6. Review the edit

- **Compare** (the split icon in the panel header) — split view against the
  original; drag the divider. Available on both pages. Geometry (including
  anamorphic) is part of the compare fingerprint, so changing transform while
  Compare is on refreshes rather than showing a stale frame.
- **H** histogram · **J** clipping warnings · **G** composition guide ·
  **F** focus point.
- **M** to drop the overlay and look at the photo itself.
- **Copy** / **Paste** are hidden on the Masks page: they carry the global
  edit only (they work off `get_adjustments()`, which never contains a mask),
  so on that page they would look like they copied the mask and silently
  would not.
- **?** opens the shortcut cheat sheet, which is context-aware.

---

## 7. Save

There are two distinct things called saving, and only one of them produces a
new image file.

### The sidecar (automatic)

Edits are written to an **XMP sidecar** beside the original whenever
`editing_finished` fires — on slider release, stroke end, brush-mask create,
and on closing the panel. The original file is never modified.

A no-change guard means simply opening and closing the editor does not
rewrite the sidecar or invalidate caches (`main.py:_adjustments_match`).

Mask coverage is stored in that sidecar as base64 PNG at half resolution
(`mask_layers_xmp.py`), including empty brush layers so undo has a real
pre-paint snapshot.

### Export (explicit)

**Export** bakes the edit into a new file. It is on the **Global page only**.

- **16-bit TIFF (baked)** — RAW only
- **JPEG (baked)**
- **WebP (baked)**
- **AI Denoise / AI Upscale 2×** — **Plus only**, and only when those models
  are on disk (`rawviewer_profile.ai_export_enabled`). Standard never offers
  them, even if the files happen to be present.

Export runs the pipeline with no preview cache, so it cannot inherit a
preview-path bug.

### Back to the gallery

Closing the editor re-bakes the browse-cache thumbnail for an edited RAW, so
the gallery tile shows the edited render rather than the embedded JPEG
(`main.py:_bake_browse_caches_for_visited_edit`).

---

# Workflow review

Walking the above as a user, seven things do not hold up. Ordered by how much
they cost. #4 has since been fixed and is kept here with its reasoning.

## 1. Export is on the wrong page — the workflow has no ending

You mask, you adjust, you review, and then to produce a file you must **switch
back to Global**, because that is the only page with an Export button.

The current placement was deliberate: a large Export button "followed the user
into mask editing where finishing the export is not the next thing they want"
(`adjust_panel.py`). That reasoning holds for the button's *visual
weight*, not for its *availability*. The result is that the one page where a
user does the most work is the one page they cannot finish from, and nothing
on the Masks page hints where to go.

**Worst part:** a user who never returns to Global has an edit saved only as a
sidecar. That is recoverable, but it is not what "I finished editing this
photo" usually means to them.

**Suggested:** keep Export off the Masks page as a primary button, but make it
reachable — a small text action in the panel header, present on both pages.

## 2. Nothing tells you the edit is already saved

Sidecar writes are silent and automatic. There is no "saved" indicator, no
dirty marker, and no explicit Save. A user coming from Lightroom will assume
this; a user coming from Photoshop will look for ⌘S and find nothing.

**Suggested:** a quiet "Saved" state in the header after a write. Cheap, and
it removes the single most likely reason for someone to lose confidence in the
app.

## 3. Two visibility controls that read as the same control

The per-mask eye hides **one mask's overlay**. **M** hides **the whole
overlay**. Both are "visibility", neither says which scope it has, and the eye
looks like the standard layer-visibility control from every other editor —
where it disables the layer's *effect*.

The tooltip says "the adjustment still applies", but a tooltip is not where
someone forms that expectation.

**Suggested:** the eye is fine as-is, but the empty state or section heading
should say once that hiding affects the overlay only.

## 4. Hidden overlay + painting = no feedback at all — **fixed**

Since mid-stroke rendering was removed, the overlay is the *only* mid-stroke
feedback. Arming Add or Erase turned it on, which covered the common path, but
**M** overruled that — so a user who hid the overlay and then painted saw
nothing at all until release, which felt *worse* than the lag the change
removed.

A stroke now turns the overlay on regardless
(`adjust_panel.py:force_mask_overlay_visible`). It stays on afterwards rather
than snapping back, because the tint is what you just painted. The user's
**M** setting is not overwritten, so it still governs the arming and
page-switch paths.

## 5. "Create new mask" and "Add" are the same gesture with different meanings

Both end in painting. **Create new mask → Brush** starts a *new* empty mask
and leaves Add unarmed; **P / Add** grows the *selected* one. The distinction
is real and correctly implemented, but the two controls sit in different
sections with no visual relationship, and the failure mode is silent: paint
with the wrong one and you have either one mask you wanted as two, or two you
wanted as one.

There is no undo-friendly way back — splitting a mask after the fact is not
possible; you repaint.

**Suggested:** name it "Add to this mask" rather than "Add".

## 6. Duplicate copies the adjustments, which is rarely what you want

The two real uses are *same region, different adjustment* and *same
adjustment, different region*. The copy gives you both at once, so one of them
is always wrong and must be cleared by hand. This was the original reason the
button was removed.

**Suggested:** keep it, but consider a second entry — "Duplicate coverage
only" — which is the more common of the two.

## 7. Smaller things

- **The Masks page has no empty-state route to Global.** With no masks you get
  a hint pointing at Create new mask, which is right, but a user who opened
  Masks by mistake has no signposting back.
- **Invert has no keyboard shortcut** while Add, Erase, Delete and the overlay
  all do.
- **Group discards the dragged mask's adjustments** with only a status message
  to say so. It is undoable, but the message is easy to miss.
- **Temperature means different things on the two pages** — absolute Kelvin on
  Global, relative ±100 on Masks — under the same word.
- **`MaskLayer.enabled` has no UI.** Hiding a mask's *effect* to judge its
  contribution is a normal thing to want, the model supports it completely, and
  there is no way to ask for it.
- **P with masks present but none selected** still creates a new empty mask
  (`ensure_mask_layer_for_painting`), while the Add button is disabled in that
  state — the shortcut and the button disagree.

## What holds up well

- One control creates masks, and it names the tool — no "make a mask, then
  pick how" two-step.
- Brush does not auto-arm Add, so Create → Brush then **P** again to put away
  behaves as a latch rather than canceling an unintended arm.
- Empty brush masks persist, so Ctrl+Z after the first stroke is usable.
- The adjustment groups match the Global page's names and order, so a slider
  is in the same place on both.
- Reset-by-clicking-the-value works identically on both pages.
- Coverage tools that make their own mask do not pre-create an empty one, so
  arming and changing your mind leaves nothing behind.
- The overlay follows the tool, and a stroke can never paint into nothing.
- Painting no longer renders the photo per stamp.
- Local masks stay aligned under anamorphic / geometry with the live preview.
