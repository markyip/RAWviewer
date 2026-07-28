# Masked local adjustments — the full workflow

Every step, as the code actually implements it, from opening a photo to
having an edited file on disk. Written against `development` at the time of
writing; file/line references are pointers to the authority, not decoration.

A short vocabulary first, because three different things in this app are
called a "mask" in casual speech:

| Term | What it is | Where it lives |
|---|---|---|
| **Mask** | One `MaskLayer`: a coverage map plus its own adjustment set | `raw_mask_layers.py:189` |
| **Group** | A mask whose coverage comes from several components combined | `MaskLayer.components` |
| **Overlay** | The coloured tint drawn over the photo showing coverage | `gpu_image_view.py:926` |
| **Dodge/Burn mask** | A separate, older brush system on the Global page | `raw_dodge_burn.py` |

The Masks page and Dodge/Burn are independent. They share one brush pipeline
(only one tool can be armed at a time — `adjust_panel.py:_on_mask_tool_toggled`),
but they store and apply their edits separately.

---

## 1. Open the editor

1. In the gallery, click a photo to open it in single-image view.
2. Press **E** (`main.py:9967`) to open the Adjust panel. **E** or **Esc**
   closes it again.
3. The panel opens on the **Global** page. Click **Masks** to switch pages.

Switching pages is not cosmetic — it changes what several keys mean, and it
hides the mask overlay (`adjust_panel.py:_on_panel_tab_changed`).

The header hint line under "Adjust" follows the page:

```
Global:  E closes · hold D B X H to paint · scroll = size
Masks:   E closes · hold P to paint, X to erase · M shows the mask · scroll = size
```

---

## 2. Create a mask

One control starts a mask: **Create new mask ▾**. It names the tool up front.

| Menu entry | What happens | Needs a model |
|---|---|---|
| **Brush** | Makes an empty mask, arms Add. You paint the coverage | no |
| **Linear Gradient** | Arms the tool; drag across the photo to place it | no |
| **Radial Gradient** | Arms the tool; drag a box, the ellipse is inscribed | no |
| **Smart Object** | One press; finds whatever stands out and masks it | yes (214 MB) |
| **Sky** | One press; masks the sky | yes |
| **AI Selection** | Arms click-to-select; click a thing to mask that thing | yes |

Only **Brush** creates a layer up front. The gradients and the AI tools each
create their own on completion, so arming one and changing your mind leaves
nothing behind (`adjust_panel.py:_on_mask_create`).

The three AI entries are **Plus only**. Standard omits them from the menu
entirely rather than greying them out, because each downloads a model on
first press (`t_mask_edition_gating.py`).

**While a tool is armed**, a line under the button says what it is waiting
for — "Drag across the photo to place the gradient." It clears when the tool
disarms. This exists because a menu item cannot stay lit the way an armed
button could.

**Smart Object / Sky / AI Selection** put up a progress dialog with a Cancel
and a 15-second timeout (`main.py:_begin_ai_mask_busy`). The timeout applies
to inference only — on first use the model downloads (~214 MB) and the wait
is left open, because no sane timeout covers that.

Two refusals worth knowing:
- If the model returns near-empty coverage, **no layer is added** and the app
  says so, rather than leaving you an "AI Sky" mask covering nothing
  (`_AI_MASK_MIN_COVERAGE`).
- Pressing **Smart Object** twice selects the existing mask instead of
  stacking a duplicate — the model is a saliency segmenter with no notion of
  instances and returns an identical matte every time.

---

## 3. Edit the mask's coverage

The **This mask** row acts on the selected mask only:

| Control | Key | Effect |
|---|---|---|
| **Add** | hold **P** | Brush more coverage into the selected mask |
| **Erase** | hold **X** | Remove coverage from it |
| **Invert** | — | The mask applies everywhere *except* its coverage |
| **⧉** | — | Duplicate: same coverage and adjustments, as a separate mask |
| **✕** | **Delete** | Delete the selected mask |

Holding **P** or **X** paints on pointer movement with no click, and releasing
the key puts the tool away. Clicking the button instead keeps it armed.

**Brush Size / Flow / Feather** appear under **Brush** while a brush is in
hand. They mirror the Local section on the Global page — the same values, kept
in step both ways, so you do not have to leave the Masks page to resize.
**Two-finger scroll** also changes size while a brush is armed.

An **inverted** mask says so in its row (`Sky (inverted)`), because the row
thumbnail still shows the painted region — which is precisely the part an
inverted mask does *not* cover.

### Grouping

Drag one mask onto another to combine them into a single mask. The drop
target keeps its adjustments; the dragged mask's are discarded, because a
group holds exactly one adjustment set. Drag a component out to separate it
again. Both are undoable with **Ctrl/⌘+Z**.

---

## 4. Change visibility

Two different controls, easily confused:

**The eye, per mask row.** Hides that mask's coloured overlay. **The
adjustment still applies.** It is view state only: not written to the sidecar,
not part of the render cache key, and toggling it does not re-render or save
(`raw_mask_layers.py` — `MaskLayer.overlay_hidden`).

**M / the Mask button.** Shows or hides the overlay for everything at once.

The overlay follows what you are doing:
- Arming **Add** or **Erase** turns it on — painting coverage you cannot see
  is guesswork.
- **Starting a stroke** turns it on even if **M** switched it off, because
  since mid-stroke rendering was removed the overlay is the only feedback a
  stroke has. It stays on afterwards: the tint is what you just painted.
  When the stroke is what switched it on, **only the mask being edited is
  drawn** — bringing back every mask you had put away is not what asking to
  paint one of them meant. Pressing **M** ends that and shows the stack
  normally.
- Switching between **Global** and **Masks** turns it off — the tint would
  otherwise sit over the photo you switched pages to judge.
- **M** overrules the first and third of those, and your choice is remembered
  across them.

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
  original; drag the divider. Available on both pages.
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
`editing_finished` fires — on slider release, stroke end, and on closing the
panel. The original file is never modified.

A no-change guard means simply opening and closing the editor does not
rewrite the sidecar or invalidate caches (`main.py:_adjustments_match`).

Mask coverage is stored in that sidecar as base64 PNG at half resolution
(`mask_layers_xmp.py`), with gradients stored as geometry rather than pixels
so they stay re-draggable and resolution-independent.

### Export (explicit)

**Export** bakes the edit into a new file. It is on the **Global page only**.

- **16-bit TIFF (baked)** — RAW only
- **JPEG (baked)**
- **WebP (baked)**
- **AI Denoise / AI Upscale 2×** variants appear only if those models are on
  disk

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
(`adjust_panel.py:2130`). That reasoning holds for the button's *visual
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

Both make you paint. **Create new mask → Brush** starts a *new* mask; **Add**
grows the *selected* one. The distinction is real and correctly implemented,
but the two controls sit in different sections with no visual relationship, and
the failure mode is silent: paint with the wrong one and you have either one
mask you wanted as two, or two you wanted as one.

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

## What holds up well

- One control creates masks, and it names the tool — no "make a mask, then
  pick how" two-step.
- The adjustment groups match the Global page's names and order, so a slider
  is in the same place on both.
- Reset-by-clicking-the-value works identically on both pages.
- Coverage tools that make their own mask do not pre-create an empty one, so
  arming and changing your mind leaves nothing behind.
- The overlay follows the tool, and a stroke can never paint into nothing.
- Painting no longer renders the photo per stamp.
