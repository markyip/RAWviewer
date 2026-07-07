#!/usr/bin/env python3
"""Proactive Nikon HE/HE* NEF detection: LibRaw can't decode High Efficiency
compressed NEF (Z8/Z9/Z6III/Z50II generation, no upstream ETA). Rather than
paying for one failed LibRaw decode attempt per file, _detect_nef_he_compression
reads the NEFCompression value directly -- a hand-rolled targeted TIFF-IFD
walk chosen over the `exifread` library's full maker-note parse because that
was benchmarked at ~136ms/file vs ~0.1ms here, too slow to add to any hot path.

Two on-disk locations are checked, per exiftool's Nikon.pm:
  - Tag 0x0093 (NEFCompression), a plain top-level int16u -- older/non-Z9-gen bodies.
  - Tag 0x0051 ("MakerNotes0x51"), byte offset 10 within that sub-block's own
    data -- exiftool's source notes NEFCompression was "relocated to
    MakerNotes_0x51 at offset x'0a (Z9)" for the Z8/Z9/Z6III generation.

Both paths are covered here with hand-built synthetic TIFF fixtures for
deterministic byte-level verification, PLUS (when available) a real-file
ground-truth sweep against /Volumes/T5 EVO/RAW_Sample: every result is cross-
checked directly against rawpy's actual decode success/failure, not just
against what the detector claims. When last run against that sample set (61
real NEFs spanning Z5/Z6/Z6II/Z6III/Z7/Z7II/Z8/Z9/Z50/Z50II/Zf/Zfc/Z30/D780/
D850), this matched LibRaw's ground truth on all 61 files (20 true positives,
41 true negatives, 0 mismatches).
"""
import inspect
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

SAMPLE_DIR = "/Volumes/T5 EVO/RAW_Sample"

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def build_nef_fixture(
    nef_compression_value=None,
    *,
    nikon_makernote=True,
    via_tag51=False,
):
    """Minimal little-endian TIFF: IFD0 -> Exif IFD (0x8769) -> MakerNote
    (0x927C) -> Nikon inner TIFF -> compression value, either directly at
    tag 0x0093 or (``via_tag51=True``) nested inside tag 0x0051's own data
    at byte offset 10, matching the Z8/Z9/Z6III relocation.
    ``nikon_makernote=False`` builds a MakerNote blob that isn't
    Nikon-prefixed, to exercise the "not Nikon" early-out.
    """
    e = "<"

    def pack(fmt, *args):
        return struct.pack(e + fmt, *args)

    header = b"II" + pack("H", 42) + pack("I", 8)  # IFD0 at offset 8

    ifd0_offset = 8
    ifd0_num_entries = 1
    ifd0_size = 2 + ifd0_num_entries * 12 + 4
    exif_ifd_offset = ifd0_offset + ifd0_size

    ifd0 = pack("H", ifd0_num_entries)
    ifd0 += pack("H", 0x8769) + pack("H", 4) + pack("I", 1) + pack("I", exif_ifd_offset)
    ifd0 += pack("I", 0)

    exif_num_entries = 1
    exif_size = 2 + exif_num_entries * 12 + 4
    mn_offset = exif_ifd_offset + exif_size

    if not nikon_makernote:
        mn_blob = b"NotNikonMakerNoteData000"
    elif nef_compression_value is None:
        # Nikon-prefixed but no compression tag present anywhere (older format).
        mn_blob = b"Nikon\x00" + b"\x02\x10" + b"\x00\x00"
        mn_blob += b"II" + pack("H", 42) + pack("I", 8)
        mn_blob += pack("H", 0)
        mn_blob += pack("I", 0)
    elif via_tag51:
        # Inner IFD has one entry: tag 0x0051, an offset-addressed blob whose
        # own data (relative to inner_base) holds an 8-byte firmware string
        # at offset 0 and the compression int16u at offset 10.
        inner_ifd_offset = 8
        inner_ifd_size = 2 + 1 * 12 + 4
        tag51_blob_offset = inner_ifd_offset + inner_ifd_size
        tag51_blob = b"01020501" + b"\x00" * 2 + pack("H", nef_compression_value)
        inner_ifd = pack("H", 1)
        inner_ifd += (
            pack("H", 0x0051) + pack("H", 7) + pack("I", len(tag51_blob))
            + pack("I", tag51_blob_offset)
        )
        inner_ifd += pack("I", 0)
        mn_blob = b"Nikon\x00" + b"\x02\x11" + b"\x00\x00"
        mn_blob += b"II" + pack("H", 42) + pack("I", inner_ifd_offset)
        mn_blob += inner_ifd + tag51_blob
    else:
        mn_blob = b"Nikon\x00" + b"\x02\x10" + b"\x00\x00"
        mn_blob += b"II" + pack("H", 42) + pack("I", 8)
        mn_blob += pack("H", 1)
        mn_blob += (
            pack("H", 0x0093) + pack("H", 3) + pack("I", 1)
            + pack("H", nef_compression_value) + b"\x00\x00"
        )
        mn_blob += pack("I", 0)

    mn_len = len(mn_blob)

    exif_ifd = pack("H", exif_num_entries)
    exif_ifd += pack("H", 0x927C) + pack("H", 7) + pack("I", mn_len) + pack("I", mn_offset)
    exif_ifd += pack("I", 0)

    return header + ifd0 + exif_ifd + mn_blob


def main() -> int:
    from enhanced_raw_processor import _detect_nef_he_compression

    cases = [
        ("tag 0x0093: High Efficiency (13)", dict(nef_compression_value=13), True),
        ("tag 0x0093: High Efficiency* (14)", dict(nef_compression_value=14), True),
        ("tag 0x0093: Lossless (3, not HE)", dict(nef_compression_value=3), False),
        ("tag 0x0093: Uncompressed (2, not HE)", dict(nef_compression_value=2), False),
        ("tag 0x0051 (Z9-gen relocation): High Efficiency (13)",
         dict(nef_compression_value=13, via_tag51=True), True),
        ("tag 0x0051 (Z9-gen relocation): High Efficiency* (14)",
         dict(nef_compression_value=14, via_tag51=True), True),
        ("tag 0x0051 (Z9-gen relocation): Lossless (3, not HE)",
         dict(nef_compression_value=3, via_tag51=True), False),
        ("Nikon MakerNote, no compression tag anywhere (older format)",
         dict(nef_compression_value=None), None),
        ("non-Nikon MakerNote", dict(nef_compression_value=None, nikon_makernote=False), None),
    ]

    tmpd = tempfile.mkdtemp()
    try:
        for label, kwargs, expected in cases:
            data = build_nef_fixture(**kwargs)
            path = os.path.join(tmpd, "fixture.NEF")
            with open(path, "wb") as f:
                f.write(data)
            result = _detect_nef_he_compression(path)
            check(f"{label}: expected {expected!r}, got {result!r}", result == expected)

        # Non-TIFF / garbage input never raises, just returns None.
        garbage_path = os.path.join(tmpd, "garbage.NEF")
        with open(garbage_path, "wb") as f:
            f.write(b"not a tiff file at all")
        check(
            "garbage input returns None without raising",
            _detect_nef_he_compression(garbage_path) is None,
        )

        # Nonexistent file never raises.
        check(
            "nonexistent file returns None without raising",
            _detect_nef_he_compression(os.path.join(tmpd, "does_not_exist.NEF")) is None,
        )
    finally:
        import shutil

        shutil.rmtree(tmpd, ignore_errors=True)

    # --- integration: EXIFExtractor wires the flag into the cached EXIF row ---
    import enhanced_raw_processor as erp

    src = inspect.getsource(erp.EXIFExtractor.extract_exif_data)
    check(
        "extract_exif_data() stores nef_he_compressed for .nef files",
        "result['nef_he_compressed'] = _detect_nef_he_compression(file_path)" in src,
    )
    check(
        "detection is gated to .nef files only",
        'file_path.lower().endswith(".nef")' in src,
    )

    # --- real-file ground truth sweep (skipped if the sample volume isn't mounted) ---
    if not os.path.isdir(SAMPLE_DIR):
        print(f"SKIP  real-file ground-truth sweep: sample folder not present on this machine: {SAMPLE_DIR}")
    else:
        import glob

        nef_paths = sorted(
            p for p in glob.glob(os.path.join(SAMPLE_DIR, "*"))
            if p.lower().endswith(".nef")
        )
        if not nef_paths:
            print(f"SKIP  real-file ground-truth sweep: no NEF files found in {SAMPLE_DIR}")
        else:
            import rawpy

            mismatches = []
            true_positives = false_positives = true_negatives = false_negatives = 0
            for p in nef_paths:
                detected = _detect_nef_he_compression(p)
                try:
                    with rawpy.imread(p) as raw:
                        raw.postprocess(half_size=True)
                    actually_fails = False
                except (rawpy.LibRawFileUnsupportedError, rawpy.LibRawDataError):
                    actually_fails = True
                except Exception:
                    continue  # unrelated decode error, not this detector's concern
                if detected is True and actually_fails:
                    true_positives += 1
                elif detected is True and not actually_fails:
                    false_positives += 1
                    mismatches.append(p)
                elif detected in (False, None) and actually_fails:
                    false_negatives += 1
                    mismatches.append(p)
                else:
                    true_negatives += 1
            check(
                f"real-file ground truth: {len(nef_paths)} NEFs, "
                f"{true_positives} true positives, {true_negatives} true negatives, "
                f"0 mismatches expected",
                not mismatches,
                f"got {len(mismatches)} mismatch(es): {mismatches}" if mismatches else "",
            )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
