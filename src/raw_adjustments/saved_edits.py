"""sRGB decode, already-adjusted tracking, and apply_saved_edits_for_display."""

from __future__ import annotations

import os
import threading
import weakref

import numpy as np

from .checks import is_default_adjustments
from .xmp_read import edited_previews_enabled, load_adjustments_for_file, resolve_xmp_path

_SRGB_DECODE_LUT = None


def _decode_srgb_uint8_to_linear(arr: np.ndarray) -> np.ndarray:
    """Standard sRGB EOTF, uint8 [0,255] -> linear [0,1] float32.

    Gallery/thumbnail buffers for RAW files are the camera's embedded JPEG
    (extract_embedded_jpeg_by_scan / rawpy.extract_thumb) or, absent that,
    LibRaw's own raw.postprocess() output -- both approximately sRGB gamma
    (~2.2 with a linear toe), NOT the app's own dcraw-style BT.709
    linear16->uint8 encode from fast_raw_decode._gamma_lut8. This function
    used to invert THAT LUT (see git history), which assumes an input this
    code path never actually receives: the flatter BT.709 shadow curve,
    inverted, overestimates linear values versus the steeper true sRGB toe,
    so the reconstructed "linear" buffer was too bright even before edits --
    then process_linear_edit_buffer's exposure/tone stage (built for a
    genuinely linear-RAW input, e.g. decode_raw_edit_base) amplified that
    overestimate further, making edited gallery thumbnails visibly brighter
    than the same edit applied in single view (which decodes true
    scene-linear RAW, never through an 8-bit round trip at all). Report:
    "thumbnail preview brightness significantly higher than edited image."
    """
    global _SRGB_DECODE_LUT
    if _SRGB_DECODE_LUT is None:
        c = np.arange(256, dtype=np.float64) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        _SRGB_DECODE_LUT = lin.astype(np.float32)
    return _SRGB_DECODE_LUT[arr]



# id(arr) -> weakref to that exact array.
#
# This was a bare set of id()s, which was wrong in BOTH directions. numpy
# hands the same address back for the next same-shaped allocation, so a fresh
# preview at a recycled address was read as "already adjusted" and skipped its
# XMP edits. And the overflow guard called set.pop(), which removes an
# ARBITRARY element -- often a live, still-in-flight buffer -- so that one got
# adjusted a second time.
#
# The weakref settles identity (a recycled address cannot match, because the
# referent is dead or different) and the finalizer keeps the map from growing,
# which removes the need to evict live entries at all.
_already_adjusted_ids: dict[int, weakref.ref] = {}
_already_adjusted_lock = threading.Lock()


def mark_array_as_already_adjusted(arr):
    if arr is None:
        return
    key = id(arr)

    def _drop(_ref, _key=key):
        with _already_adjusted_lock:
            _already_adjusted_ids.pop(_key, None)

    try:
        ref = weakref.ref(arr, _drop)
    except TypeError:
        # Not weak-referenceable: better to re-apply than to record a mark
        # that can never be validated.
        return
    with _already_adjusted_lock:
        _already_adjusted_ids[key] = ref


def is_array_already_adjusted(arr) -> bool:
    """Whether THIS array object was marked -- not merely its address."""
    if arr is None:
        return False
    with _already_adjusted_lock:
        ref = _already_adjusted_ids.get(id(arr))
    return ref is not None and ref() is arr

def apply_saved_edits_for_display(file_path: str, arr):
    """Apply saved XMP edits to a display-bound uint8 RGB ndarray."""
    if is_array_already_adjusted(arr):
        return arr
    """Apply saved XMP edits to a display-bound uint8 RGB ndarray.

    Returns the input unchanged when previews-with-edits is disabled, the
    file is not RAW, the file has no (non-default) saved adjustments, or the
    buffer is not an ndarray (QImage branches skip). Never raises. Callers
    must NOT persist the result into pixel caches -- caches hold the
    unadjusted base.

    RAW-only by contract: sidecars are resolved by BASENAME (Lightroom
    convention), so a JPEG sharing its basename with an edited RAW resolves
    to the RAW's sidecar -- without this gate the JPEG's gallery tile showed
    the RAW's edits applied to it ("also flagged as edited" report). Edits
    can only ever be created for RAW files (the Adjust panel requires a
    demosaiced RAW base), so non-RAW is always pass-through.

    Renders through the REAL scene-linear pipeline (sRGB decode ->
    process_linear_edit_buffer -> linear_to_display_uint8), not the legacy
    gamma-space approximation: the gamma path's tone math diverges visibly
    from the Adjust panel's render (reported as the gallery/fit preview
    being "way off" from the edited image). Working from an already
    tone-curved camera JPEG is still an approximation of the true RAW edit
    base, but the tone/color processing applied on top is now the same code
    the editor runs. The uint8->linear decode uses the standard sRGB EOTF,
    matching the actual gamma of the embedded-JPEG/LibRaw-postprocess
    thumbnail buffers this function receives (see
    _decode_srgb_uint8_to_linear's docstring for why the BT.709 LUT this
    used previously was the wrong inverse and made edited thumbnails read
    brighter than the true edited image).
    """
    try:
        if arr is None or not hasattr(arr, "shape") or not edited_previews_enabled():
            return arr
        from common_image_loader import is_raw_file

        if not is_raw_file(file_path):
            return arr
        # Sidecar existence FIRST (two isfile checks, ~0.05ms). Going straight
        # to load_adjustments_for_file ran read_as_shot_temperature -- a full
        # EXIF parse of the RAW file -- for EVERY delivered gallery tile,
        # edited or not, before the is_default early-out could save it. On a
        # cold session that extra per-tile file-open competed with thumbnail
        # extraction on the same disk: "gallery scrolling slower than before".
        xmp_path = resolve_xmp_path(file_path)
        if not (xmp_path and os.path.isfile(xmp_path)):
            return arr
        adj = load_adjustments_for_file(file_path)
        if is_default_adjustments(adj):
            return arr
        from raw_adjustments.pipeline import linear_to_display_uint8, process_linear_edit_buffer

        lin = _decode_srgb_uint8_to_linear(np.ascontiguousarray(arr))
        out = process_linear_edit_buffer(lin, adj, preview=True)
        return linear_to_display_uint8(out, adj)
    except Exception:
        return arr
