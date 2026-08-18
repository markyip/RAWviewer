#!/usr/bin/env python3
"""Color labels and keywords round-trip on XMP sidecars; rating still survives."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_adjustments import (
        DEFAULT_ADJUSTMENTS,
        load_keywords_for_file,
        load_label_for_file,
        load_rating_for_file,
        resolve_xmp_path,
        write_keywords_for_file,
        write_label_for_file,
        write_rating_for_file,
        write_xmp_adjustments_for_file,
    )

    tmpd = tempfile.mkdtemp()
    img = os.path.join(tmpd, "a.CR3")
    open(img, "wb").write(b"stub")

    write_label_for_file(img, "Red")
    write_keywords_for_file(img, ["wedding", "flash"])
    write_rating_for_file(img, 4)
    check("label round-trip", load_label_for_file(img) == "Red")
    check("keywords round-trip", load_keywords_for_file(img) == ["wedding", "flash"])
    check("rating with label", load_rating_for_file(img) == 4)
    xmp = resolve_xmp_path(img)
    check("sidecar exists", os.path.isfile(xmp))

    write_xmp_adjustments_for_file(img, {**DEFAULT_ADJUSTMENTS, "Exposure2012": 0.5})
    check("label survives develop write", load_label_for_file(img) == "Red")
    check("keywords survive develop write", "wedding" in load_keywords_for_file(img))

    write_xmp_adjustments_for_file(img, dict(DEFAULT_ADJUSTMENTS))
    check("label survives reset sliders", load_label_for_file(img) == "Red")
    check("sidecar kept for catalog-only", os.path.isfile(resolve_xmp_path(img)))

    write_label_for_file(img, "")
    write_keywords_for_file(img, [])
    write_rating_for_file(img, 0)
    check("empty catalog deletes sidecar", not os.path.isfile(resolve_xmp_path(img)))

    from raw_adjustments import load_catalog_for_file, parse_xmp_catalog

    write_label_for_file(img, "Blue")
    write_keywords_for_file(img, ["studio"])
    write_rating_for_file(img, 3)
    cat = load_catalog_for_file(img)
    check("catalog helper one-shot", cat == (3, "Blue", ["studio"]), str(cat))
    missing = os.path.join(tmpd, "no-sidecar.CR3")
    open(missing, "wb").write(b"stub")
    check("catalog skip missing sidecar", load_catalog_for_file(missing) == (0, "", []))
    check("parse skip missing xmp path", parse_xmp_catalog(missing + ".xmp") == (0, "", []))

    from image_cache import PersistentEXIFCache

    cache_dir = os.path.join(tmpd, "exif-cache")
    cache = PersistentEXIFCache(cache_dir)
    cache.put(
        img,
        {
            "orientation": 1,
            "camera_make": "Test",
            "camera_model": "Cam",
            "color_label": "Red",
            "keywords": ["wedding", "flash"],
            "rating": 4,
            "exif_data": {"EXIF ISOSpeedRatings": "100"},
        },
        commit=True,
    )
    got = cache.get(img) or {}
    check("cache get promotes color_label", got.get("color_label") == "Red", str(got.get("color_label")))
    check("cache get promotes keywords", got.get("keywords") == ["wedding", "flash"], str(got.get("keywords")))
    check("cache get promotes rating", got.get("rating") == 4, str(got.get("rating")))
    inner = got.get("exif_data") or {}
    check("blob stored color_label", inner.get("color_label") == "Red")
    check("blob stored keywords", inner.get("keywords") == ["wedding", "flash"])
    multi = cache.get_multiple([img], fast_mode=False) or {}
    row = multi.get(img) or {}
    check("get_multiple promotes color_label", row.get("color_label") == "Red", str(row.get("color_label")))
    check("get_multiple promotes keywords", row.get("keywords") == ["wedding", "flash"], str(row.get("keywords")))

    if FAILURES:
        print(f"\n{len(FAILURES)} failed")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
