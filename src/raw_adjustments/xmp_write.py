"""XMP sidecar write: star ratings and adjustment sliders."""

from __future__ import annotations

import contextlib
import os
import tempfile
import xml.etree.ElementTree as ET

from .apply import _format_xmp_value
from .checks import is_default_adjustments_for_sidecar, tone_curve_serial_is_active, wb_reference_temperature
from .defaults import (
    CRS_NS,
    DEFAULT_ADJUSTMENTS,
    RDF_NS,
    RELEVANT_ADJUSTMENT_KEYS,
    X_NS,
    XMP_NS,
    _xmp_write_lock,
)
from .xmp_read import editing_features_enabled, resolve_xmp_path


def _parse_rating_value(raw: object) -> int:
    if raw is None:
        return 0
    try:
        text = str(raw).strip()
        if not text:
            return 0
        return max(0, min(5, int(float(text.split()[0]))))
    except (TypeError, ValueError):
        return 0


def parse_xmp_rating(xmp_path: str) -> int:
    """Parse xmp:Rating from a sidecar without loading adjustment sliders."""
    if not xmp_path or not os.path.isfile(xmp_path):
        return 0
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xmp": "http://ns.adobe.com/xap/1.0/",
        }
        for desc in root.findall(".//rdf:Description", ns):
            for key, val in desc.attrib.items():
                local_key = key.split("}", 1)[-1] if "}" in key else key.split(":")[-1]
                if local_key == "Rating":
                    rating = _parse_rating_value(val)
                    if rating:
                        return rating
            rating_el = desc.find("{http://ns.adobe.com/xap/1.0/}Rating")
            if rating_el is not None and rating_el.text:
                rating = _parse_rating_value(rating_el.text)
                if rating:
                    return rating
    except Exception:
        pass
    return 0


def load_rating_for_file(image_path: str) -> int:
    """Lightweight star rating from the image's XMP sidecar (no edit sliders)."""
    return parse_xmp_rating(resolve_xmp_path(image_path))


def write_xmp_rating(xmp_path: str, rating: int) -> None:
    """Write (or clear) xmp:Rating in a sidecar, preserving everything else.

    Independent of editing_features_enabled() -- ratings must persist even
    in browse-only builds that never touch the crs adjustment sliders.
    Edits the existing tree in place rather than rebuilding it (unlike
    write_xmp_adjustments) so a rating set here survives untouched even if
    the sidecar also holds crs sliders / tone curve / dodge-burn data from
    a build with editing enabled.

    Holds this sidecar's write lock for the whole read-modify-write so a
    concurrent write_xmp_adjustments() call (runs on the export worker
    thread) can't interleave with this one and lose either change -- see
    the _xmp_write_lock module comment.
    """
    if not xmp_path:
        raise ValueError("Invalid xmp path")
    with _xmp_write_lock(xmp_path):
        _write_xmp_rating_locked(xmp_path, rating)


def _write_xmp_rating_locked(xmp_path: str, rating: int) -> None:
    rating = max(0, min(5, int(rating)))

    ET.register_namespace("x", X_NS)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("crs", CRS_NS)
    ET.register_namespace("xmp", XMP_NS)

    root = None
    if os.path.isfile(xmp_path):
        try:
            root = ET.parse(xmp_path).getroot()
        except Exception:
            root = None

    if root is None:
        if rating == 0:
            return
        root = ET.Element(f"{{{X_NS}}}xmpmeta")
        rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
        desc.set(f"{{{RDF_NS}}}about", "")
    else:
        rdf = root.find(f"{{{RDF_NS}}}RDF")
        if rdf is None:
            rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        desc = rdf.find(f"{{{RDF_NS}}}Description")
        if desc is None:
            desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
            desc.set(f"{{{RDF_NS}}}about", "")

    rating_attr = f"{{{XMP_NS}}}Rating"
    if rating > 0:
        desc.set(rating_attr, str(rating))
    else:
        desc.attrib.pop(rating_attr, None)
        for child in list(desc):
            tag = child.tag
            local = tag.split("}", 1)[-1] if "}" in tag else tag
            if local == "Rating":
                desc.remove(child)

    # Nothing left worth keeping (no rating, no crs sliders, no child
    # elements like tone curve / dodge-burn mask) -- remove the sidecar
    # instead of littering the folder with an empty .xmp per photo.
    has_rating = rating_attr in desc.attrib
    has_crs = any(
        key.startswith(f"{{{CRS_NS}}}") for key in desc.attrib
    )
    has_children = len(list(desc)) > 0
    if not has_rating and not has_crs and not has_children:
        if os.path.isfile(xmp_path):
            os.remove(xmp_path)
        return

    tree = ET.ElementTree(root)
    parent = os.path.dirname(os.path.abspath(xmp_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Atomic write -- see write_xmp_adjustments for why (crash/race safety).
    fd, tmp_path = tempfile.mkstemp(
        dir=parent or None, prefix=os.path.basename(xmp_path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, xmp_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def write_rating_for_file(image_path: str, rating: int) -> None:
    """Persist a star rating (0-5) to the image's XMP sidecar."""
    xmp_path = resolve_xmp_path(image_path)
    if not xmp_path:
        raise ValueError("Invalid image path")
    write_xmp_rating(xmp_path, rating)

def write_xmp_adjustments(xmp_path: str, adj: dict[str, float]) -> None:
    """Merge Lightroom-compatible crs sliders into an XMP sidecar.

    Updates only RAWviewer-owned ``crs:`` attributes / child elements and
    preserves foreign Lightroom fields (Look, masks, Grain, Profile, ΓÇª).
    Rating is preserved (also written independently via write_xmp_rating).

    Holds this sidecar's write lock for the whole read-modify-write ΓÇö
    this runs on the export worker thread (see write_xmp_adjustments_for_file
    callers), so without it a same-moment write_xmp_rating() call from the
    main thread could interleave and lose either change.
    """
    if not editing_features_enabled():
        return
    with _xmp_write_lock(xmp_path):
        _write_xmp_adjustments_locked(xmp_path, adj)


# crs: attributes / children that RAWviewer authors. Everything else on the
# rdf:Description is left untouched on merge write.
_OUR_CRS_ATTR_LOCALS = frozenset(RELEVANT_ADJUSTMENT_KEYS) | frozenset(
    {
        "DefringePurpleAmount",
        "DefringeGreenAmount",
        "ProcessVersion",
        "ToneCurveName2012",
        "CreativeLUTName",
        "CreativeLUTWorkingSpace",
    }
)
_OUR_CRS_CHILD_LOCALS = frozenset(
    {
        "ToneCurvePV2012",
        "ToneCurvePV2012Red",
        "ToneCurvePV2012Green",
        "ToneCurvePV2012Blue",
        "DodgeBurnMask",
        "SpotHealMask",
        "CreativeLUTName",
        "CreativeLUTWorkingSpace",
    }
)


def _crs_local_name(tag_or_attr: str) -> str:
    if "}" in tag_or_attr:
        return tag_or_attr.split("}", 1)[-1]
    if ":" in tag_or_attr:
        return tag_or_attr.split(":")[-1]
    return tag_or_attr


def _find_or_create_rdf_description(root: ET.Element) -> ET.Element:
    rdf = root.find(f"{{{RDF_NS}}}RDF")
    if rdf is None:
        # Some sidecars use a default/prefixed form; fall back to any RDF.
        for el in root.iter():
            if _crs_local_name(el.tag) == "RDF":
                rdf = el
                break
    if rdf is None:
        rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
    desc = rdf.find(f"{{{RDF_NS}}}Description")
    if desc is None:
        for el in rdf:
            if _crs_local_name(el.tag) == "Description":
                desc = el
                break
    if desc is None:
        desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
        desc.set(f"{{{RDF_NS}}}about", "")
    return desc


def _strip_our_crs_fields(desc: ET.Element) -> None:
    """Remove only RAWviewer-owned crs attributes and child elements."""
    for attr in list(desc.attrib.keys()):
        local = _crs_local_name(attr)
        if local in _OUR_CRS_ATTR_LOCALS and (  # noqa: SIM102  # Intentional: nested structure is clearer for complex XML filtering
            attr.startswith(f"{{{CRS_NS}}}")
            or attr.startswith("crs:")
            or (not attr.startswith("{") and local in _OUR_CRS_ATTR_LOCALS)
        ):
            # Only strip crs-scoped attrs (never dc:/xmp: with same local name).
            if attr.startswith(f"{{{CRS_NS}}}") or attr.startswith("crs:"):
                desc.attrib.pop(attr, None)
    for child in list(desc):
        local = _crs_local_name(child.tag)
        if local not in _OUR_CRS_CHILD_LOCALS:
            continue
        if child.tag.startswith(f"{{{CRS_NS}}}") or (
            not child.tag.startswith("{") and local in _OUR_CRS_CHILD_LOCALS
        ):
            desc.remove(child)


def _sidecar_has_foreign_content(desc: ET.Element) -> bool:
    """True if Description still holds non-RAWviewer / non-empty content."""
    rating_attr = f"{{{XMP_NS}}}Rating"
    if rating_attr in desc.attrib:
        try:
            if int(desc.attrib.get(rating_attr) or 0) > 0:
                return True
        except ValueError:
            return True
    for attr, _val in desc.attrib.items():
        if attr in (f"{{{RDF_NS}}}about", rating_attr):
            continue
        local = _crs_local_name(attr)
        if attr.startswith(f"{{{CRS_NS}}}"):
            if local not in _OUR_CRS_ATTR_LOCALS:
                return True
        elif not attr.startswith("{"):
            # Unprefixed / other ΓÇö treat as foreign to be safe.
            if local not in _OUR_CRS_ATTR_LOCALS and local != "about":
                return True
        else:
            # Other namespaces (dc, photoshop, ΓÇª)
            if not attr.startswith(f"{{{CRS_NS}}}") and not attr.startswith(f"{{{RDF_NS}}}"):
                return True
    for child in desc:
        local = _crs_local_name(child.tag)
        if child.tag.startswith(f"{{{CRS_NS}}}"):
            if local not in _OUR_CRS_CHILD_LOCALS:
                return True
        else:
            return True
    return False


def _write_xmp_tree_atomic(root: ET.Element, xmp_path: str) -> None:
    tree = ET.ElementTree(root)
    parent = os.path.dirname(os.path.abspath(xmp_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=parent or None, prefix=os.path.basename(xmp_path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, xmp_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def _write_xmp_adjustments_locked(xmp_path: str, adj: dict[str, float]) -> None:
    existing_rating = parse_xmp_rating(xmp_path)
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})

    ET.register_namespace("x", X_NS)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("crs", CRS_NS)
    ET.register_namespace("xmp", XMP_NS)

    root = None
    if os.path.isfile(xmp_path):
        try:
            root = ET.parse(xmp_path).getroot()
        except Exception:
            root = None

    if is_default_adjustments_for_sidecar(merged):
        if root is None:
            if existing_rating > 0:
                _write_xmp_rating_locked(xmp_path, existing_rating)
            elif os.path.isfile(xmp_path):
                os.remove(xmp_path)
            return
        desc = _find_or_create_rdf_description(root)
        _strip_our_crs_fields(desc)
        rating_attr = f"{{{XMP_NS}}}Rating"
        if existing_rating > 0:
            desc.set(rating_attr, str(existing_rating))
        else:
            desc.attrib.pop(rating_attr, None)
        if not _sidecar_has_foreign_content(desc) and existing_rating <= 0:
            if os.path.isfile(xmp_path):
                os.remove(xmp_path)
            return
        _write_xmp_tree_atomic(root, xmp_path)
        return

    if root is None:
        root = ET.Element(f"{{{X_NS}}}xmpmeta")
        rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
        desc.set(f"{{{RDF_NS}}}about", "")
    else:
        desc = _find_or_create_rdf_description(root)

    # Replace our fields in place; leave Look / masks / Grain / etc. alone.
    _strip_our_crs_fields(desc)

    from raw_adjustments.pv2012 import PROCESS_VERSION, TONE_CURVE_NAME_2012

    desc.set(f"{{{CRS_NS}}}ProcessVersion", PROCESS_VERSION)
    desc.set(f"{{{CRS_NS}}}ToneCurveName2012", TONE_CURVE_NAME_2012)
    if existing_rating > 0:
        desc.set(f"{{{XMP_NS}}}Rating", str(existing_rating))

    for key in sorted(RELEVANT_ADJUSTMENT_KEYS):
        if key == "Defringe":
            continue
        value = float(merged.get(key, DEFAULT_ADJUSTMENTS[key]))
        default = DEFAULT_ADJUSTMENTS[key]
        if key == "Temperature":
            ref = wb_reference_temperature(merged)
            if abs(value - ref) <= 0.5:
                continue
        elif abs(value - default) <= 1e-4:
            continue
        if key == "Temperature" and value <= 0:
            value = default
        desc.set(f"{{{CRS_NS}}}{key}", _format_xmp_value(key, value))

    defringe = float(merged.get("Defringe", 0.0))
    if defringe > 1e-4:
        amt = _format_xmp_value("Defringe", defringe)
        desc.set(f"{{{CRS_NS}}}DefringePurpleAmount", amt)
        desc.set(f"{{{CRS_NS}}}DefringeGreenAmount", amt)

    from raw_adjustments.tone_curve import (
        TONE_CURVE_BLUE_KEY,
        TONE_CURVE_GREEN_KEY,
        TONE_CURVE_RED_KEY,
        TONE_CURVE_SERIAL_KEY,
        deserialize_tone_curve_points,
    )

    def _write_point_curve(key: str, tag: str) -> None:
        serial = str(merged.get(key, "") or "")
        points = deserialize_tone_curve_points(serial)
        if len(points) < 2 or not tone_curve_serial_is_active(serial):
            return
        tc = ET.SubElement(desc, f"{{{CRS_NS}}}{tag}")
        seq = ET.SubElement(tc, f"{{{RDF_NS}}}Seq")
        for x, y in points:
            li = ET.SubElement(seq, f"{{{RDF_NS}}}li")
            li.text = f"{int(round(x))}, {int(round(y))}"

    _write_point_curve(TONE_CURVE_SERIAL_KEY, "ToneCurvePV2012")
    _write_point_curve(TONE_CURVE_RED_KEY, "ToneCurvePV2012Red")
    _write_point_curve(TONE_CURVE_GREEN_KEY, "ToneCurvePV2012Green")
    _write_point_curve(TONE_CURVE_BLUE_KEY, "ToneCurvePV2012Blue")

    from raw_adjustments.dodge_burn import MASK_KEY

    mask_serial = str(merged.get(MASK_KEY, "") or "")
    if mask_serial:
        db = ET.SubElement(desc, f"{{{CRS_NS}}}DodgeBurnMask")
        db.text = mask_serial

    from raw_adjustments.spot_heal import MASK_KEY as HEAL_MASK_KEY

    heal_serial = str(merged.get(HEAL_MASK_KEY, "") or "")
    if heal_serial:
        heal = ET.SubElement(desc, f"{{{CRS_NS}}}SpotHealMask")
        heal.text = heal_serial

    from raw_adjustments.lut import (
        LUT_AMOUNT_KEY,
        LUT_NAME_KEY,
        LUT_SPACE_DEFAULT,
        lut_working_space,
    )

    lut_name = str(merged.get(LUT_NAME_KEY, "") or "").strip()
    lut_amt = float(merged.get(LUT_AMOUNT_KEY, 0.0) or 0.0)
    if lut_name and abs(lut_amt) > 1e-4:
        desc.set(f"{{{CRS_NS}}}CreativeLUTName", lut_name)
        space = lut_working_space(merged)
        # Persist explicitly so older builds that assumed Linear can detect Rec709.
        desc.set(f"{{{CRS_NS}}}CreativeLUTWorkingSpace", space or LUT_SPACE_DEFAULT)

    _write_xmp_tree_atomic(root, xmp_path)


def write_xmp_adjustments_for_file(image_path: str, adj: dict[str, float]) -> None:
    if not editing_features_enabled():
        return
    xmp_path = resolve_xmp_path(image_path)
    if not xmp_path:
        raise ValueError("Invalid image path")
    write_xmp_adjustments(xmp_path, adj)
