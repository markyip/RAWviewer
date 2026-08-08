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


# --- Type ------------------------------------------------------------------
#
# ONE font stack for the whole app, for the same reason there is one palette:
# a second one is a second answer to a question that has one right answer.
#
# The app used to declare three, and the majority stack put **Roboto** first.
# Roboto ships with neither macOS nor Windows, so on a default install of
# either it fell through -- to the system UI font on macOS, to Segoe UI on
# Windows -- and everything happened to match. That match was luck, not
# design: a user who has Roboto installed (Android Studio and several design
# tools install it) got Roboto in the twelve places that named it first and
# the platform font in the three that did not, splitting the UI across two
# typefaces on one screen.
#
# The fix is NOT a better stack in every stylesheet -- it is no family in any
# stylesheet. The app font is set once (main.py, at startup) from the families
# below, and widgets inherit it. A QSS `font-family` rule OVERRIDES that app
# font, so every such rule is a place the single typeface can silently come
# apart again; there are now none outside bootstrap.py.
#
# Order matters. The platform's own UI font comes first, and no third-party
# family is named at all -- naming one is what created a preference for a font
# the app does not ship and cannot rely on. Qt resolves this list itself, which
# a hardcoded QFont("Segoe UI", ...) cannot do: one family is not a list, so it
# substitutes silently on macOS rather than falling through.
#
# `bootstrap.py` deliberately does NOT use this: it imports winreg at module
# scope, so it only ever runs on Windows, and naming Segoe UI directly there
# is correct rather than inconsistent.
FONT_FAMILIES = (".AppleSystemUIFont", "Segoe UI", "Helvetica Neue", "Arial")

# How much bigger than the platform default the base UI font runs, in points.
# A delta rather than an absolute size: macOS defaults to 13pt and Windows to
# 9pt, and pinning one number makes the app correct on one platform and wrong
# on the other. Stylesheet `font-size: Npx` rules do NOT scale with this --
# they are absolute by definition -- so they were bumped to match, and any new
# one should be sized against its neighbours rather than against this.
FONT_BASE_BUMP_PT = 1.0


def shortcut(*keys: str) -> str:
    """Render a shortcut for a tooltip, in the platform's own vocabulary.

    ``shortcut("Ctrl", "C")`` reads "⌘C" on macOS and "Ctrl+C" on Windows.
    Writing "Ctrl+C" on a Mac is not a small inaccuracy -- Macs have a Control
    key that does something else, so it names the wrong key rather than the
    right key awkwardly.

    Pass the modifier as the literal "Ctrl" regardless of platform; this is
    where it becomes ⌘.
    """
    import sys

    mac = sys.platform == "darwin"
    glyphs = {"Ctrl": "⌘", "Shift": "⇧", "Alt": "⌥", "Meta": "⌃"} if mac else {}
    parts = [glyphs.get(k, k) for k in keys]
    return "".join(parts) if mac else "+".join(parts)


def with_shortcut(tip: str, *keys: str) -> str:
    """A tooltip with its shortcut appended in a consistent place and form.

    One helper rather than each call site formatting its own, so that a
    control's key is always in the same position -- at the end, parenthesised
    -- instead of sometimes mid-sentence and sometimes missing.
    """
    return f"{tip} ({shortcut(*keys)})"


def ui_font(size_px: int, weight=None):
    """A QFont on the canonical stack, for non-stylesheet call sites."""
    from PyQt6.QtGui import QFont

    f = QFont()
    f.setFamilies(list(FONT_FAMILIES))
    f.setPixelSize(int(size_px))
    if weight is not None:
        f.setWeight(weight)
    return f
