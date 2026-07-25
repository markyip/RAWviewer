#!/usr/bin/env python3
"""Derived-file lifecycle for generative edits (generative_derived_file).

Everything runs against a temp directory with the StubProvider -- no
model, no GPU, no network. The full round-trip (bake -> provider ->
derived file -> sidecar -> re-read -> chain) is exercised end to end.

Checks:
  1. Naming: -edit1, -edit2, ... beside the original; chaining off a
     derived file stays FLAT (-edit2, never -edit1-edit1); a freed slot
     is reused rather than colliding.
  2. The image is written atomically and reads back identically.
  3. The sidecar survives: a derived file has NEUTRAL sliders, and
     without the PROVENANCE_KEY guard write_xmp_adjustments would clear
     it as "no edit" -- this is the regression this suite exists for.
  4. Provenance round-trips through XMP, and lineage() returns the full
     chain oldest-first.
  5. is_generated distinguishes generated from ordinary files.
  6. A sidecar write failure still leaves the image on disk.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

import generative_derived_file as gdf  # noqa: E402
import raw_generative_edit as gen  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def _img(h=12, w=16):
    rng = np.random.default_rng(11)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_naming(tmp):
    src = os.path.join(tmp, "IMG_1234.CR3")
    open(src, "wb").close()

    p1 = gdf.derived_path_for(src)
    check("first derived is -edit1", os.path.basename(p1) == "IMG_1234-edit1.tif", p1)
    check("lands beside the original", os.path.dirname(p1) == os.path.abspath(tmp))

    open(p1, "wb").close()
    p2 = gdf.derived_path_for(src)
    check("second derived is -edit2", os.path.basename(p2) == "IMG_1234-edit2.tif", p2)

    # Chaining off a derived file must not stack suffixes.
    p3 = gdf.derived_path_for(p1)
    check("chaining stays flat", os.path.basename(p3) == "IMG_1234-edit2.tif", p3)
    check("root stem collapses -editN", gdf.root_stem_for(p1) == "IMG_1234")

    parsed = gdf.parse_derived_name(p1)
    check("parses derived name", parsed == ("IMG_1234", 1), f"got {parsed}")
    check("plain file is not derived", gdf.parse_derived_name(src) is None)

    # A freed slot is reused rather than skipped into a collision.
    open(p2, "wb").close()
    os.remove(p1)
    check("lowest free slot reused", os.path.basename(gdf.derived_path_for(src)) == "IMG_1234-edit1.tif")

    # Names with dots and dashes must survive.
    odd = os.path.join(tmp, "2026-07-25_shoot.v2.ARW")
    open(odd, "wb").close()
    po = gdf.derived_path_for(odd)
    check(
        "dotted/dashed stems handled",
        os.path.basename(po) == "2026-07-25_shoot.v2-edit1.tif",
        po,
    )


def test_image_write(tmp):
    out = os.path.join(tmp, "written.tif")
    src = _img(9, 7)
    gdf.write_derived_image(src, out)
    check("image file created", os.path.isfile(out))
    check("no .part left behind", not os.path.exists(out + ".part"))

    import cv2

    back = cv2.imread(out, cv2.IMREAD_COLOR)[:, :, ::-1]
    check("image round-trips losslessly", np.array_equal(back, src))

    # Float input is baked, not written raw.
    outf = os.path.join(tmp, "float.tif")
    gdf.write_derived_image(np.ones((4, 4, 3), dtype=np.float32), outf)
    back2 = cv2.imread(outf, cv2.IMREAD_COLOR)
    check("float baked to 8-bit", back2.dtype == np.uint8 and back2.max() == 255)

    try:
        gdf.write_derived_image(np.zeros((4, 4), dtype=np.uint8), os.path.join(tmp, "bad.tif"))
        check("non-RGB rejected", False, "no raise")
    except ValueError:
        check("non-RGB rejected", True)


def test_full_roundtrip(tmp):
    src = os.path.join(tmp, "SHOT_9.CR3")
    open(src, "wb").close()

    provider = gen.StubProvider()
    request = gen.GenerativeRequest(_img(), "remove the bin", seed=5, source_path=src)
    result = provider.edit(request)

    derived = gdf.create_derived_file(src, result)
    check("derived file written", os.path.isfile(derived))
    check("derived is -edit1", os.path.basename(derived) == "SHOT_9-edit1.tif")

    sidecar = derived + ".xmp"
    check("sidecar written", os.path.isfile(sidecar), f"looked for {sidecar}")

    # THE regression this suite exists for: a derived file has neutral
    # sliders, so without the PROVENANCE_KEY guard in
    # is_default_adjustments the sidecar would be cleared as "no edit".
    if os.path.isfile(sidecar):
        text = open(sidecar, encoding="utf-8").read()
        check("provenance survived the neutral-slider clear", "RVGenerativeProvenance" in text)

    prov = gdf.read_provenance(derived)
    check("provenance reads back", prov is not None)
    if prov:
        check("instruction preserved", prov["instruction"] == "remove the bin")
        check("seed preserved", prov["seed"] == 5)
        check("model preserved", prov["model"] == "stub-v1")
        check("flagged generated", prov["generated"] is True)

    check("is_generated True for derived", gdf.is_generated(derived) is True)
    check("is_generated False for original", gdf.is_generated(src) is False)
    return derived


def test_chaining(tmp, first_derived):
    """Editing a generated file appends to the lineage."""
    parent = gdf.parent_provenance_for(first_derived)
    check("parent provenance found", parent is not None)

    provider = gen.StubProvider()
    request = gen.GenerativeRequest(
        _img(),
        "now make it dusk",
        seed=6,
        source_path=first_derived,
        options={"parent_provenance": parent},
    )
    result = provider.edit(request)
    second = gdf.create_derived_file(first_derived, result)
    check("chained file is -edit2", os.path.basename(second) == "SHOT_9-edit2.tif", second)

    chain = gdf.lineage(second)
    check("lineage has both edits", len(chain) == 2, f"got {len(chain)}")
    if len(chain) == 2:
        check("lineage is oldest-first", chain[0]["instruction"] == "remove the bin")
        check("newest last", chain[1]["instruction"] == "now make it dusk")

    check("original has no lineage", gdf.lineage(os.path.join(tmp, "SHOT_9.CR3")) == [])


def test_sidecar_failure_keeps_image(tmp):
    """A provenance write failure must not cost the user the image."""
    src = os.path.join(tmp, "FRAGILE.CR3")
    open(src, "wb").close()
    result = gen.StubProvider().edit(gen.GenerativeRequest(_img(), "x", source_path=src))

    original = gdf.write_derived_sidecar

    def boom(*_a, **_k):
        raise OSError("disk full")

    gdf.write_derived_sidecar = boom
    try:
        derived = gdf.create_derived_file(src, result)
        check("image survives sidecar failure", os.path.isfile(derived))
        check("unlabelled file reads as not generated", gdf.is_generated(derived) is False)
    except Exception as exc:  # noqa: BLE001
        check("image survives sidecar failure", False, f"raised {exc!r}")
    finally:
        gdf.write_derived_sidecar = original


def test_malformed_provenance(tmp):
    src = os.path.join(tmp, "BAD.tif")
    gdf.write_derived_image(_img(4, 4), src)
    with open(src + ".xmp", "w", encoding="utf-8") as fh:
        fh.write(
            '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
            'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
            "<crs:RVGenerativeProvenance>{not json</crs:RVGenerativeProvenance>"
            "</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
        )
    check("malformed provenance returns None", gdf.read_provenance(src) is None)
    check("malformed provenance is not generated", gdf.is_generated(src) is False)


def main():
    print("Generative derived-file lifecycle")
    tmp = tempfile.mkdtemp(prefix="rv_genedit_")
    try:
        test_naming(tmp)
        test_image_write(tmp)
        derived = test_full_roundtrip(tmp)
        test_chaining(tmp, derived)
        test_sidecar_failure_keeps_image(tmp)
        test_malformed_provenance(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("All derived-file checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
