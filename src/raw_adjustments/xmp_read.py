
"""XMP sidecar path resolution, parse helpers, and adjustment load logic."""

from __future__ import annotations

import contextlib
import os
import xml.etree.ElementTree as ET

from .checks import read_as_shot_temperature
from .defaults import (
    _PROFILE_STALE_WARNED,
    AS_SHOT_TEMP_KEY,
    CRS_NS,
    DEFAULT_ADJUSTMENTS,
    RDF_NS,
    RELEVANT_ADJUSTMENT_KEYS,
)


# --- path resolution & feature toggles
def resolve_xmp_path(image_path: str) -> str:
    """Sidecar path next to the image (Lightroom-style basename.xmp)."""
    if not image_path:
        return ""
    from common_image_loader import is_raw_file
    if not is_raw_file(image_path):
        return ""
    companion = image_path + ".xmp"
    if os.path.isfile(companion):
        return companion
    return os.path.splitext(image_path)[0] + ".xmp"


def editing_features_enabled() -> bool:
    """Whether the Adjust panel, XMP writes, and edit export are available.

    On by default on the development branch
    (``RAWVIEWER_ENABLE_EDITING=0`` to disable). Rating read/write and plain
    browse/export-without-adjustments stay available either way.
    """
    return os.environ.get("RAWVIEWER_ENABLE_EDITING", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def sidecar_adjustments_enabled() -> bool:
    """Whether browse/full-res display applies saved XMP edit sliders to pixels.

    Off by default (RAWVIEWER_SIDECAR_ADJUST=1 to enable): saved XMP edits
    render ONLY inside the Adjust panel (explicit
    `apply_sidecar_adjustments=True` callers). Browse surfaces show the
    original pixels ΓÇö gallery tiles and single-view non-RAW show the embedded
    JPEG as shot, the RAW tier shows the unadjusted raw decode. Rendering
    edits per-tier produced visible mismatches between the thumbnail and the
    single-image edited version (tone pipeline differs at thumbnail scale),
    which was worse than not previewing edits at all. Requires
    `editing_features_enabled()` even when the env var opts in.
    """
    if not editing_features_enabled():
        return False
    return os.environ.get("RAWVIEWER_SIDECAR_ADJUST", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def edited_previews_enabled() -> bool:
    """Whether gallery tiles and the fit preview render saved XMP edits.

    Off by default (RAWVIEWER_EDITED_PREVIEWS=1 to enable): edits render only
    in the Adjust panel; browse tiers show original pixels (see
    sidecar_adjustments_enabled for why ΓÇö tier-to-tier edit rendering
    mismatched). Applied at display/delivery time only -- adjusted pixels are
    never written back to any pixel cache, so there is no stale-thumbnail
    invalidation problem: re-editing simply changes what the next delivery
    applies.
    """
    if not editing_features_enabled():
        return False
    return os.environ.get("RAWVIEWER_EDITED_PREVIEWS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

# --- XMP parse helpers
def _parse_point_curve_from_xmp(xmp_path: str, tag_name: str) -> str:
    """Read a crs:<tag_name> point list ΓåÆ 'x,y;x,y' serialized string.

    Shared by the main luminance curve (ToneCurvePV2012) and the three
    Standard-mode channel curves (ToneCurvePV2012Red/Green/Blue) -- same
    XMP shape (an rdf:Seq of "x, y" li entries), different tag name.
    """
    if not xmp_path or not os.path.isfile(xmp_path):
        return ""
    from raw_adjustments.tone_curve import serialize_tone_curve_points

    points: list[tuple[float, float]] = []
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {
            "rdf": RDF_NS,
            "crs": CRS_NS,
        }
        for desc in root.findall(".//rdf:Description", ns):
            for child in desc:
                tag = child.tag
                local_key = tag.split("}")[-1] if "}" in tag else tag.split(":")[-1]
                if local_key != tag_name:
                    continue
                for li in child.findall(".//rdf:li", ns):
                    if not li.text:
                        continue
                    parts = [p.strip() for p in li.text.replace(" ", "").split(",")]
                    if len(parts) < 2:
                        continue
                    try:
                        points.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
    except Exception:
        return ""
    return serialize_tone_curve_points(points)


def parse_tone_curve_pv2012_from_xmp(xmp_path: str) -> str:
    """Read crs:ToneCurvePV2012 point list ΓåÆ 'x,y;x,y' serialized string."""
    return _parse_point_curve_from_xmp(xmp_path, "ToneCurvePV2012")


def parse_dodge_burn_mask_from_xmp(xmp_path: str) -> str:
    """Read the custom DodgeBurnMask child element -> base64 PNG string."""
    return _parse_crs_child_text_from_xmp(xmp_path, "DodgeBurnMask")


def parse_spot_heal_mask_from_xmp(xmp_path: str) -> str:
    """Read the custom SpotHealMask child element -> base64 PNG string."""
    return _parse_crs_child_text_from_xmp(xmp_path, "SpotHealMask")


def _parse_crs_child_text_from_xmp(xmp_path: str, local_name: str) -> str:
    if not xmp_path or not os.path.isfile(xmp_path) or not local_name:
        return ""
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {"rdf": RDF_NS, "crs": CRS_NS}
        for desc in root.findall(".//rdf:Description", ns):
            for child in desc:
                tag = child.tag
                local_key = tag.split("}")[-1] if "}" in tag else tag.split(":")[-1]
                if local_key == local_name and child.text:
                    return child.text.strip()
    except Exception:
        pass
    return ""

def parse_creative_lut_name_from_xmp(xmp_path: str) -> str:
    """Read crs:CreativeLUTName (string basename of a managed .cube)."""
    return _parse_crs_string_from_xmp(xmp_path, "CreativeLUTName")


def parse_creative_lut_space_from_xmp(xmp_path: str) -> str:
    """Read crs:CreativeLUTWorkingSpace (``Rec709`` / ``Linear``)."""
    return _parse_crs_string_from_xmp(xmp_path, "CreativeLUTWorkingSpace")


def _parse_crs_string_from_xmp(xmp_path: str, local_name: str) -> str:
    if not xmp_path or not os.path.isfile(xmp_path) or not local_name:
        return ""
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {"rdf": RDF_NS, "crs": CRS_NS}
        for desc in root.findall(".//rdf:Description", ns):
            for key, val in desc.attrib.items():
                local_key = key.split("}")[-1] if "}" in key else key.split(":")[-1]
                if local_key == local_name and val:
                    return str(val).strip()
            for child in desc:
                tag = child.tag
                local_key = tag.split("}")[-1] if "}" in tag else tag.split(":")[-1]
                if local_key == local_name and child.text:
                    return child.text.strip()
    except Exception:
        pass
    return ""

# --- load adjustments
def load_adjustments_from_xmp(xmp_path: str) -> dict[str, float]:
    """Load edit settings from an arbitrary XMP file (sidecar or managed preset).

    Does not invent AsShotTemperature ΓÇö callers that apply onto an image should
    keep the current file's as-shot reference. Omits nothing the sidecar path
    would load (sliders, tone curves, LUT name, dodge/burn mask).
    """
    adj: dict[str, float] = dict(DEFAULT_ADJUSTMENTS)
    if not xmp_path or not os.path.isfile(xmp_path):
        return adj
    parsed = parse_xmp_adjustments(xmp_path)
    adj.update(parsed)
    from raw_adjustments.tone_curve import (
        TONE_CURVE_BLUE_KEY,
        TONE_CURVE_GREEN_KEY,
        TONE_CURVE_RED_KEY,
        TONE_CURVE_SERIAL_KEY,
    )

    serial = parse_tone_curve_pv2012_from_xmp(xmp_path)
    if serial:
        adj[TONE_CURVE_SERIAL_KEY] = serial
    for key, tag in (
        (TONE_CURVE_RED_KEY, "ToneCurvePV2012Red"),
        (TONE_CURVE_GREEN_KEY, "ToneCurvePV2012Green"),
        (TONE_CURVE_BLUE_KEY, "ToneCurvePV2012Blue"),
    ):
        ch = _parse_point_curve_from_xmp(xmp_path, tag)
        if ch:
            adj[key] = ch
    from raw_adjustments.dodge_burn import MASK_KEY

    mask_serial = parse_dodge_burn_mask_from_xmp(xmp_path)
    if mask_serial:
        adj[MASK_KEY] = mask_serial
    from raw_adjustments.spot_heal import MASK_KEY as HEAL_MASK_KEY

    heal_serial = parse_spot_heal_mask_from_xmp(xmp_path)
    if heal_serial:
        adj[HEAL_MASK_KEY] = heal_serial
    from raw_adjustments.lut import LUT_NAME_KEY, LUT_SPACE_DEFAULT, LUT_SPACE_KEY

    lut_name = parse_creative_lut_name_from_xmp(xmp_path)
    if lut_name:
        adj[LUT_NAME_KEY] = lut_name
    lut_space = parse_creative_lut_space_from_xmp(xmp_path)
    if lut_space:
        adj[LUT_SPACE_KEY] = lut_space
    elif lut_name:
        adj[LUT_SPACE_KEY] = LUT_SPACE_DEFAULT
    return adj


def load_adjustments_for_file(image_path: str) -> dict[str, float]:
    adj = dict(DEFAULT_ADJUSTMENTS)
    as_shot = read_as_shot_temperature(image_path)
    adj[AS_SHOT_TEMP_KEY] = as_shot
    xmp_path = resolve_xmp_path(image_path)
    has_xmp_temp = False
    if xmp_path and os.path.isfile(xmp_path):
        # Only crs:Temperature present in the file counts ΓÇö defaults from
        # load_adjustments_from_xmp must not fake an authored WB.
        has_xmp_temp = "Temperature" in parse_xmp_adjustments(xmp_path)
        parsed = load_adjustments_from_xmp(xmp_path)
        parsed.pop(AS_SHOT_TEMP_KEY, None)
        adj.update(parsed)
    if not has_xmp_temp:
        adj["Temperature"] = as_shot
    if not (xmp_path and os.path.isfile(xmp_path)):
        try:
            from raw_adjustments.color_calibration import camera_identity_for_file, get_camera_profile

            make, model, iso_val = camera_identity_for_file(image_path)
            if make or model:
                prof = get_camera_profile(make, model, iso=iso_val)
                if prof:
                    _warn_if_profile_decode_stale(
                        prof, make, model, iso_val, image_path
                    )
                    if "temperature_shift" in prof:
                        adj["Temperature"] = float(
                            adj.get("Temperature", as_shot)
                        ) + float(prof["temperature_shift"])
                    if "tint_shift" in prof:
                        adj["Tint"] = float(adj.get("Tint", 0.0)) + float(
                            prof["tint_shift"]
                        )
                    from raw_adjustments.hsl import HSL_COLOR_NAMES

                    for band in HSL_COLOR_NAMES:
                        for prof_key, adj_prefix in (
                            ("hsl_hue", "HueAdjustment"),
                            ("hsl_sat", "SaturationAdjustment"),
                            ("hsl_lum", "LuminanceAdjustment"),
                        ):
                            values = prof.get(prof_key)
                            if isinstance(values, dict) and band in values:
                                adj[f"{adj_prefix}{band}"] = float(values[band])
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "[CALIB] Could not apply camera profile to %s",
                image_path,
                exc_info=True,
            )
    return adj

# --- profile warning
def _warn_if_profile_decode_stale(
    prof: dict,
    make: str,
    model: str,
    iso_val,
    image_path: str,
) -> None:
    try:
        from raw_adjustments.color_calibration import normalize_camera_key, profile_decode_mismatch

        reason = profile_decode_mismatch(prof, image_path)
        if not reason:
            return
        key = normalize_camera_key(make, model, iso=iso_val)
        if key in _PROFILE_STALE_WARNED:
            return
        _PROFILE_STALE_WARNED.add(key)
        import logging

        logging.getLogger(__name__).warning(
            "[CALIB] Camera profile for %s may no longer match this decode: %s "
            "(still applied)",
            key,
            reason,
        )
    except Exception:
        pass

# --- parse XMP adjustments
def parse_xmp_adjustments(xmp_path: str) -> dict[str, float]:
    """Parse Lightroom-compatible crs adjustment sliders from an XMP sidecar file."""
    adjustments = {}
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'x': 'adobe:ns:meta/',
            'crs': 'http://adobe.com/camera-raw-settings/1.0/'
        }

        # 1. Search attributes on rdf:Description elements
        for desc in root.findall('.//rdf:Description', ns):
            for key, val in desc.attrib.items():
                local_key = key
                if '}' in key:
                    ns_uri, local_key = key.split('}', 1)
                    if ns_uri != '{' + ns['crs']:
                        continue
                elif ':' in key:
                    prefix, local_key = key.split(':', 1)
                    if prefix != 'crs':
                        continue

                with contextlib.suppress(ValueError):
                    adjustments[local_key] = float(val)

        # 2. Search child elements in crs namespace under rdf:Description
        for desc in root.findall('.//rdf:Description', ns):
            for child in desc:
                tag = child.tag
                local_key = tag
                if '}' in tag:
                    ns_uri, local_key = tag.split('}', 1)
                    if ns_uri != '{' + ns['crs']:
                        continue
                elif ':' in tag:
                    prefix, local_key = tag.split(':', 1)
                    if prefix != 'crs':
                        continue
                if child.text:
                    with contextlib.suppress(ValueError):
                        adjustments[local_key] = float(child.text.strip())
    except Exception:
        pass

    relevant_keys = RELEVANT_ADJUSTMENT_KEYS
    out = {k: v for k, v in adjustments.items() if k in relevant_keys}
    if "Defringe" not in out:
        purple = float(adjustments.get("DefringePurpleAmount", 0.0))
        green = float(adjustments.get("DefringeGreenAmount", 0.0))
        if purple > 0.0 or green > 0.0:
            out["Defringe"] = max(purple, green)
    return out
