"""
Derived-file lifecycle for generative edits.

A generative edit does not modify anything. It produces a NEW image file
next to the original -- the "Edit in Photoshop" model photographers
already understand:

    IMG_1234.CR3                  the RAW, never touched
    IMG_1234-edit1.tif            what the model returned
    IMG_1234-edit1.tif.xmp        neutral sliders + provenance

The derived file is an ordinary image from the app's point of view: it
appears in the folder scan, it gets its own parametric stack, and it can
be edited non-destructively like anything else. So a generative edit and
a parametric edit COMPOSE, in that order, instead of competing -- which
is the whole reason this shape works and an in-pipeline generative filter
would not.

Two deliberate decisions:

  * **Next to the original**, not in an ``edits/`` subfolder. It keeps
    derived files inside the existing folder scan with no special-casing,
    and lets burst_grouping stack them under the parent. An exports
    subfolder would hide them from the gallery until someone wrote code
    to un-hide them.
  * **Chaining allowed.** Editing a generated file again yields
    ``-edit2`` rather than ``-edit1-edit1``, and the provenance chain
    records the full lineage back to the RAW. Forbidding it would be an
    arbitrary limit; the data model handles it, so it is permitted.

The derived file's sliders start NEUTRAL on purpose: the parent's edits
are already baked into the pixels the model was given, so replaying them
on top would double-apply. This is exactly why PROVENANCE_KEY has to
force is_default_adjustments False -- a neutral-slider sidecar would
otherwise be cleared as "no edit", destroying the AI-generated record.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DERIVED_SUFFIX = "edit"
DERIVED_EXT = ".tif"

# IMG_1234-edit3.tif -> ("IMG_1234", 3)
_DERIVED_RE = re.compile(rf"^(?P<stem>.+)-{DERIVED_SUFFIX}(?P<n>\d+)$")


def parse_derived_name(path: str):
    """(base_stem, index) if this path is itself a derived file, else None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _DERIVED_RE.match(stem)
    if not match:
        return None
    return match.group("stem"), int(match.group("n"))


def root_stem_for(path: str) -> str:
    """Stem of the ORIGINAL, collapsing any -editN suffix.

    Chaining off IMG_1234-edit1 must produce IMG_1234-edit2, not
    IMG_1234-edit1-edit1 -- names stay flat however deep the lineage goes.
    """
    parsed = parse_derived_name(path)
    if parsed:
        return parsed[0]
    return os.path.splitext(os.path.basename(path))[0]


def derived_path_for(source_path: str, *, ext: str = DERIVED_EXT) -> str:
    """First free ``<root>-editN<ext>`` beside ``source_path``.

    Scans for the lowest unused N rather than counting existing files, so
    deleting -edit1 does not cause the next edit to collide with -edit2.
    """
    directory = os.path.dirname(os.path.abspath(source_path))
    root = root_stem_for(source_path)
    n = 1
    while True:
        candidate = os.path.join(directory, f"{root}-{DERIVED_SUFFIX}{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1
        if n > 9999:  # pathological directory; fail loudly rather than spin
            raise OSError(f"Could not find a free derived filename for {source_path}")


def write_derived_image(image: np.ndarray, output_path: str) -> None:
    """Write the model's output as an 8-bit sRGB TIFF.

    8-bit because that is genuinely all the model returns -- writing 16
    would imply precision that does not exist. TIFF because it is
    lossless and the format photographers expect from a round-trip; the
    parametric stack that gets layered on top afterwards then has a clean
    source rather than JPEG artifacts to amplify.
    """
    import cv2

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5
        arr = arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Derived image must be RGB.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Encode to bytes rather than cv2.imwrite(tmp_path): imwrite picks its
    # codec from the file extension, and the ".part" temp suffix leaves it
    # unable to find a writer at all.
    ext = os.path.splitext(output_path)[1] or DERIVED_EXT
    ok, buf = cv2.imencode(ext, arr[:, :, ::-1])  # cv2 wants BGR
    if not ok:
        raise OSError(f"Could not encode {output_path}")

    # Write-then-rename: a folder scan racing this must never see a
    # half-written TIFF and cache it as a corrupt thumbnail.
    partial = output_path + ".part"
    with open(partial, "wb") as fh:
        fh.write(buf.tobytes())
    os.replace(partial, output_path)


def write_derived_sidecar(image_path: str, provenance: dict) -> str:
    """Write the derived file's sidecar: neutral sliders + provenance.

    Returns the sidecar path (empty string if editing features are off).
    """
    from raw_adjustments import (
        DEFAULT_ADJUSTMENTS,
        editing_features_enabled,
        resolve_xmp_path,
        write_xmp_adjustments,
    )
    from raw_generative_edit import PROVENANCE_KEY

    if not editing_features_enabled():
        return ""
    xmp_path = resolve_xmp_path(image_path)
    if not xmp_path:
        return ""
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj[PROVENANCE_KEY] = json.dumps(provenance, separators=(",", ":"))
    write_xmp_adjustments(xmp_path, adj)
    return xmp_path


def read_provenance(image_path: str) -> Optional[dict]:
    """Provenance record for a file, or None if it is not AI-generated."""
    from raw_adjustments import parse_generative_provenance_from_xmp, resolve_xmp_path

    xmp_path = resolve_xmp_path(image_path)
    if not xmp_path or not os.path.isfile(xmp_path):
        return None
    serial = parse_generative_provenance_from_xmp(xmp_path)
    if not serial:
        return None
    try:
        data = json.loads(serial)
    except Exception:
        logger.warning("[GENEDIT] Malformed provenance on %s", image_path)
        return None
    return data if isinstance(data, dict) else None


def is_generated(image_path: str) -> bool:
    """True if this file came out of a generative edit (for the UI badge)."""
    return read_provenance(image_path) is not None


def lineage(image_path: str) -> list:
    """Flat oldest-first list of provenance entries, for a details panel."""
    provenance = read_provenance(image_path)
    if not provenance:
        return []
    entries = list(provenance.get("chain") or [])
    head = dict(provenance)
    head.pop("chain", None)
    entries.append(head)
    return entries


def create_derived_file(source_path: str, result, *, ext: str = DERIVED_EXT) -> str:
    """Write a GenerativeResult beside its source. Returns the new path.

    The image lands before the sidecar so a crash in between leaves a
    valid (if unlabelled) image rather than a sidecar pointing at nothing.
    """
    output_path = derived_path_for(source_path, ext=ext)
    write_derived_image(result.image, output_path)
    try:
        write_derived_sidecar(output_path, result.provenance)
    except Exception:
        # An unlabelled derived image is recoverable; losing the image is
        # not. Surface the failure but keep what we wrote.
        logger.warning("[GENEDIT] Could not write provenance for %s", output_path, exc_info=True)
    return output_path


def parent_provenance_for(source_path: str) -> Optional[dict]:
    """Provenance to chain from when re-editing ``source_path``."""
    return read_provenance(source_path)
