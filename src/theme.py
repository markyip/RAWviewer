"""Darkroom color palette: warm-neutral chrome for judging photographs.

A cool-tinted surround measurably biases how a viewer judges white balance in
the image next to it, so every background/border/text token here is warm-
neutral rather than the blue-black most dev tools default to. Beyond that:

- EMBER is reserved for exactly one meaning: what's currently active
  (selection ring, dragged slider, armed tool). Nowhere else.
- DODGE marks status already decided about a photo (star rating, bookmark,
  edited badge) — distinct from EMBER so a rated *and* selected photo doesn't
  collapse both meanings into one color.
- HIST_R / HIST_G / HIST_B double as the histogram's real channel colors and
  the highlight/shadow clipping indicators — the same instrument reading the
  same thing twice, not a separate palette for each.

Existing destructive/warning colors (delete confirmations, disk-space and
update alerts) are intentionally left out of this module; they're functional
alert colors, not chrome, and out of scope for this palette.
"""

VOID = "#14120f"
SURFACE = "#1d1a16"
RAISED = "#272219"
RAISED_HI = "#302a1f"
LINE = "#3a332a"
LINE_SOFT = "#2a251d"
INK = "#ede7dd"
INK_MUTED = "#96897a"
INK_FAINT = "#665d50"

EMBER = "#d9691e"
EMBER_DIM = "rgba(217, 105, 30, 0.28)"
EMBER_GLOW = "rgba(217, 105, 30, 0.45)"

DODGE = "#d9a441"
BURN = "#5b7a8c"

HIST_R = "#e5484d"
HIST_G = "#3dd68c"
HIST_B = "#4a9eff"

# Integer-tuple equivalents for QColor(r, g, b[, a]) call sites that don't
# use hex strings. Alpha is supplied separately at the call site.
VOID_RGB = (20, 18, 15)
SURFACE_RGB = (29, 26, 22)
RAISED_RGB = (39, 34, 25)
RAISED_HI_RGB = (48, 42, 31)
LINE_RGB = (58, 51, 42)
LINE_SOFT_RGB = (42, 37, 29)
INK_RGB = (237, 231, 221)
INK_MUTED_RGB = (150, 137, 122)
INK_FAINT_RGB = (102, 93, 80)
EMBER_RGB = (217, 105, 30)
DODGE_RGB = (217, 164, 65)
BURN_RGB = (91, 122, 140)
HIST_R_RGB = (229, 72, 77)
HIST_G_RGB = (61, 214, 140)
HIST_B_RGB = (74, 158, 255)


def rgba(rgb: tuple[int, int, int], alpha: int) -> str:
    """QSS/QColor-style rgba() string from one of the *_RGB tuples above."""
    r, g, b = rgb
    return f"rgba({r}, {g}, {b}, {alpha})"
