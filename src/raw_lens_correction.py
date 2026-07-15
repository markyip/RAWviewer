"""
Automatic lens-profile geometry correction (barrel/pincushion distortion),
driven by a lensfun camera+lens+focal-length+aperture lookup.

This is applied once to the decoded RAW edit base -- not a per-tick pipeline
adjustment -- because it has no continuous "amount" the way tone/color
sliders do: the correction is either the exact profile match for this
camera+lens+focal+aperture, or nothing. See docs/EDIT_PIPELINE.md Roadmap
investigation F for the feasibility writeup this implements.

Scope: geometric distortion only (the "aspect correction" originally asked
about). Vignetting/TCA correction (lensfunpy also supports both) are not
wired up here -- a reasonable future extension, not implied by "aspect
correction".
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

_DB = None  # lazy singleton; the bundled database has ~950 cameras / ~1300 lenses


def _get_database():
    global _DB
    if _DB is None:
        import lensfunpy

        _DB = lensfunpy.Database()
    return _DB


def _tag_text(tags: dict, *names: str) -> str:
    for name in names:
        value = tags.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _parse_leading_number(text: str, *, strip_prefix: str = "") -> Optional[float]:
    if not text:
        return None
    t = text.strip()
    if strip_prefix and t.lower().startswith(strip_prefix.lower()):
        t = t[len(strip_prefix):]
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)", t.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def lens_profile_key_from_exif(exif_data: Optional[dict]) -> Optional[dict]:
    """
    Extract a lensfun lookup key from the dict returned by
    EXIFExtractor.extract_exif_data() (camera_make/camera_model/focal_length/
    aperture already parsed there; lens make/model pulled from the raw
    exif_data tag dict using the same tag-name fallback chain already used
    for search indexing, see semantic_search.py's lens lookup).

    Returns None if any required field is missing -- a profile lookup can't
    be attempted without all five (camera make/model, lens model, focal
    length, aperture).
    """
    if not exif_data:
        return None
    camera_make = str(exif_data.get("camera_make") or "").strip()
    camera_model = str(exif_data.get("camera_model") or "").strip()
    tags = exif_data.get("exif_data") or {}
    lens_model = _tag_text(
        tags,
        "EXIF LensModel",
        "MakerNote LensType",
        "MakerNote Lens",
        "Image LensModel",
        "Composite LensID",
    )
    lens_make = _tag_text(tags, "EXIF LensMake") or camera_make
    focal_length = _parse_leading_number(str(exif_data.get("focal_length") or ""))
    aperture = _parse_leading_number(str(exif_data.get("aperture") or ""), strip_prefix="f/")
    if not (camera_make and camera_model and lens_model and focal_length and aperture):
        return None
    return {
        "camera_make": camera_make,
        "camera_model": camera_model,
        "lens_make": lens_make,
        "lens_model": lens_model,
        "focal_length": focal_length,
        "aperture": aperture,
    }


def _find_camera_and_lens(camera_make: str, camera_model: str, lens_make: str, lens_model: str):
    """
    Strict matching only (loose_search=False, lensfun's default). lensfun's
    own strict matcher already normalizes case and whitespace (confirmed:
    "canon"/"Canon" and "EF16-35mm..."/"EF 16-35mm..." both match). Its
    loose_search=True mode is *not* a tighter fuzzy match -- for a lens with
    no real match it falls back to returning every lens on the camera's
    mount, which would silently apply a wrong lens's distortion correction
    to the image (actively corrupting geometry rather than fixing it), so
    it must never be used for this "does a genuine profile exist" gate.
    """
    db = _get_database()
    cams = db.find_cameras(camera_make, camera_model)
    if not cams:
        return None, None
    cam = cams[0]
    lenses = db.find_lenses(cam, lens_make, lens_model)
    if not lenses:
        return cam, None
    return cam, lenses[0]


def has_lens_profile(exif_data: Optional[dict]) -> bool:
    """True if a lensfun profile matches this file's camera+lens -- gates
    whether the Adjust panel's lens-correction toggle should even be shown."""
    try:
        key = lens_profile_key_from_exif(exif_data)
        if key is None:
            return False
        cam, lens = _find_camera_and_lens(
            key["camera_make"], key["camera_model"], key["lens_make"], key["lens_model"]
        )
        return cam is not None and lens is not None
    except Exception:
        return False


def get_lens_profile_name(exif_data: Optional[dict]) -> str:
    """Return the matched lens model name if a lensfun profile exists, otherwise empty string."""
    try:
        key = lens_profile_key_from_exif(exif_data)
        if key is None:
            return ""
        cam, lens = _find_camera_and_lens(
            key["camera_make"], key["camera_model"], key["lens_make"], key["lens_model"]
        )
        if cam is not None and lens is not None:
            return str(lens.model)
    except Exception:
        pass
    return ""


def _pixel_format_for(dtype) -> type:
    if dtype == np.uint16:
        return np.uint16
    if dtype == np.float32 or dtype == np.float64:
        return np.float32
    return np.uint8


def apply_lens_correction(
    rgb_image: Optional[np.ndarray], exif_data: Optional[dict]
) -> Optional[np.ndarray]:
    """
    Undistort a decoded RAW buffer (any dtype cv2.remap accepts -- uint16,
    float32, uint8) using the matched lensfun profile's geometry distortion
    model. Returns the input unchanged (same object, no copy) if no profile
    matches, required EXIF fields are missing, or anything goes wrong --
    this must never be the reason a RAW fails to open.
    """
    if rgb_image is None:
        return rgb_image
    try:
        key = lens_profile_key_from_exif(exif_data)
        if key is None:
            return rgb_image
        cam, lens = _find_camera_and_lens(
            key["camera_make"], key["camera_model"], key["lens_make"], key["lens_model"]
        )
        if cam is None or lens is None:
            return rgb_image

        import cv2
        import lensfunpy

        h, w = rgb_image.shape[0], rgb_image.shape[1]
        mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
        mod.initialize(
            key["focal_length"],
            key["aperture"],
            pixel_format=_pixel_format_for(rgb_image.dtype),
        )
        coords = mod.apply_geometry_distortion()
        if coords is None:
            return rgb_image
        return cv2.remap(rgb_image, coords, None, cv2.INTER_LINEAR)
    except Exception:
        return rgb_image
