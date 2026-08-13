#!/usr/bin/env python3
"""Regression cover for the stitch / focus-stack verification findings.

Each case pins a defect that shipped silently -- none of them raised, and
several reported success while doing nothing:

  1. align_hdr_images returned an EMPTY list (the cv2 binding does not fill a
     Python list passed as ``dst``), so every HDR merge and Panorama HDR merge
     died in merge_hdr_exposure_fusion with "No images provided".
  1b. ...and MTB only aligns 8-bit data: on the uint16/float32 buffers the app
     actually passes, process() returned unaligned frames without raising.
  2. Focus-stack auto-crop was a no-op: frames are warped with BORDER_REFLECT,
     so the old `luma > 0` validity test saw mirrored photograph in the
     alignment border and kept it.
  3. crop_panorama_borders used contours[0] (not the largest) and a luminance
     row test, so a dark-background panorama cropped to a few pixels and an
     ordinary one lost a third of its width to one black pixel per row.
  4. The Highlight/Shadow sliders never reached the fusion (cv2 Mertens
     ignores external weight maps).
  5. Bracket grouping guessed `2 if n % 2 == 0 else 3`, mis-pairing 4 panels
     x 3 stops.
  6. to_display_referred: merges ran and were saved scene-linear.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "rawviewer_ui"))

import cv2  # noqa: E402

from focus_stacking import focus_stack  # noqa: E402
from raw_stitching import (  # noqa: E402
    align_hdr_images,
    crop_panorama_borders,
    infer_bracket_group_size,
    merge_hdr_exposure_fusion,
    reorder_merge_paths,
    stitch_panorama,
    to_display_referred,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _textured(h=300, w=400, seed=0, bg=140):
    """A frame with real structure, so features and MTB have something to hold."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), bg, np.uint8)
    for _ in range(220):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = tuple(int(v) for v in rng.integers(20, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), int(rng.integers(6, 28)), c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 50)),
                                        y + int(rng.integers(10, 50))), c, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def _shift(img, dx, dy):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (img.shape[1], img.shape[0]))


# --- 1 / 1b. MTB alignment actually aligns, at every depth ------------------

def test_align_hdr_returns_frames_and_aligns():
    base = _textured(seed=1)
    for label, conv in (
        ("uint8", lambda x: x),
        ("uint16", lambda x: (x.astype(np.uint16) * 257)),
        ("float32", lambda x: x.astype(np.float32) / 255.0),
    ):
        frames = [conv(base), conv(_shift(base, 6, 4))]
        out, ok = align_hdr_images(frames)
        check(f"align_hdr_images returns every frame ({label})",
              ok and len(out) == len(frames), f"len={len(out)} ok={ok}")
        if len(out) != len(frames):
            continue

        def err(a, b):
            m = slice(20, -20)
            return float(np.mean(np.abs(a[m, m].astype(np.float64) - b[m, m].astype(np.float64))))

        before, after = err(frames[1], frames[0]), err(out[1], out[0])
        check(f"align_hdr_images actually removes the shift ({label})",
              after < before * 0.05, f"{before:.4f} -> {after:.4f}")


def test_align_hdr_output_feeds_the_merge():
    """The exact path that used to raise: align -> fuse."""
    base = _textured(seed=2)
    frames = [np.clip(base.astype(np.float32) * g, 0, 255).astype(np.uint8)
              for g in (0.4, 1.0, 2.2)]
    frames[1] = _shift(frames[1], 5, 3)
    aligned, _ok = align_hdr_images(frames)
    merged = merge_hdr_exposure_fusion(aligned)
    check("aligned frames fuse into an image (was: ValueError 'No images provided')",
          merged is not None and merged.size > 0 and merged.ndim == 3)


# --- 2. Focus-stack auto-crop trims the alignment border --------------------

def test_focus_stack_auto_crop_trims_border():
    # Photo-like texture, not white noise: ECC cannot register noise whose
    # halves are differently blurred (measured cc 0.04), and the alignment it
    # returns there is rejected by design -- which would leave nothing to crop.
    h, w = 240, 320
    base = _textured(h=h, w=w, seed=4)
    a = base.copy()
    a[:, w // 2:] = cv2.GaussianBlur(a[:, w // 2:], (0, 0), 6)
    b = _shift(base, 20, 12)
    b[:, : w // 2] = cv2.GaussianBlur(b[:, : w // 2], (0, 0), 6)

    cropped = focus_stack([a, b], align=True, local_align=False, auto_crop=True)
    kept = focus_stack([a, b], align=True, local_align=False, auto_crop=False)
    check("auto_crop=True trims the alignment border", cropped.success
          and cropped.image.shape[:2] != (h, w), f"shape {cropped.image.shape[:2]}")
    check("auto_crop=False leaves the frame alone", kept.success
          and kept.image.shape[:2] == (h, w), f"shape {kept.image.shape[:2]}")
    # The trim must be about the size of the shift, not a wild over-crop.
    ch, cw = cropped.image.shape[:2]
    check("the trim is proportional to the shift, not an over-crop",
          h - 40 <= ch < h and w - 60 <= cw < w, f"{cw}x{ch} from {w}x{h}")


def test_focus_stack_crop_survives_local_align():
    """Flow refinement must not re-contaminate the coverage mask."""
    h, w = 240, 320
    base = _textured(h=h, w=w, seed=6)
    a = base.copy()
    a[:, w // 2:] = cv2.GaussianBlur(a[:, w // 2:], (0, 0), 6)
    b = _shift(base, 16, 10)
    b[:, : w // 2] = cv2.GaussianBlur(b[:, : w // 2], (0, 0), 6)
    res = focus_stack([a, b], align=True, local_align=True, auto_crop=True)
    check("auto_crop still trims with local align on",
          res.success and res.image.shape[:2] != (h, w), f"shape {res.image.shape[:2]}")


# --- 3. Panorama border crop ------------------------------------------------

def test_crop_keeps_dark_background_panorama():
    """contours[0] on a dark-background stitch collapsed 600x1200 to 1x2 px.

    A night-sky panorama, near-black but exposed: after the display encode its
    darkest tones sit a few code values above zero (the BT.709 toe has a 4.5x
    slope), so true zero means "no data" and nothing else.
    """
    img = _textured(h=600, w=1200, seed=7, bg=3)
    img[:, :40] = 0  # the warp's black edge, the only real no-data region
    out = crop_panorama_borders(img)
    check("dark-background panorama survives the crop",
          out.shape[0] > img.shape[0] * 0.9 and out.shape[1] > img.shape[1] * 0.9,
          f"{img.shape[:2]} -> {out.shape[:2]}")
    check("...and the black edge is gone", out.shape[1] <= img.shape[1] - 40,
          f"width {out.shape[1]} of {img.shape[1]}")


def test_crop_ignores_isolated_black_pixels():
    """One black scene pixel per row used to disqualify the row."""
    img = _textured(h=400, w=900, seed=8, bg=120)
    rng = np.random.default_rng(9)
    for y in range(img.shape[0]):
        img[y, int(rng.integers(100, 800))] = 0
    out = crop_panorama_borders(img)
    check("isolated black pixels do not shrink the crop",
          out.shape[1] > img.shape[1] * 0.95, f"{img.shape[:2]} -> {out.shape[:2]}")


def test_crop_still_removes_a_real_border():
    img = _textured(h=400, w=600, seed=10, bg=130)
    framed = np.zeros((460, 700, 3), np.uint8)
    framed[30:430, 50:650] = img
    out = crop_panorama_borders(framed)
    check("a real black border is still removed",
          out.shape[0] <= 400 and out.shape[1] <= 600, f"-> {out.shape[:2]}")


# --- 4. Every exposure-fusion control reaches the output --------------------

def test_hdr_sliders_move_the_output():
    rng = np.random.default_rng(11)
    scene = (rng.random((160, 240, 3)) * 4.0).astype(np.float32)
    frames = [to_display_referred(np.clip(scene * g, 0, None)) for g in (0.2, 1.0, 3.0)]

    def maxdiff(a, b):
        return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())

    same = maxdiff(merge_hdr_exposure_fusion(frames), merge_hdr_exposure_fusion(frames))
    check("fusion is deterministic (cv2 Mertens was not)", same == 0, f"maxdiff {same}")
    for name, kw_lo, kw_hi in (
        ("highlight", {"highlight_weight": 0.0}, {"highlight_weight": 2.0}),
        ("shadow", {"shadow_weight": 0.0}, {"shadow_weight": 2.0}),
        ("midtone", {"midtone_weight": 0.2}, {"midtone_weight": 2.0}),
    ):
        d = maxdiff(merge_hdr_exposure_fusion(frames, **kw_lo),
                    merge_hdr_exposure_fusion(frames, **kw_hi))
        check(f"the {name} slider changes the merge", d > 64, f"maxdiff {d}")

    # Neutral settings must not be perturbed by the recovery controls at all.
    neutral = merge_hdr_exposure_fusion(frames)
    explicit = merge_hdr_exposure_fusion(frames, highlight_weight=1.0, shadow_weight=1.0)
    check("neutral highlight/shadow is the identity", maxdiff(neutral, explicit) == 0)

    zeroed = merge_hdr_exposure_fusion(frames, image_weights=[0.0, 0.0, 1.0])
    check("a zeroed per-image weight drops that frame",
          maxdiff(zeroed, neutral) > 64, f"maxdiff {maxdiff(zeroed, neutral)}")


# --- 5. Bracket grouping ----------------------------------------------------

def test_group_size_inference():
    def frames_for(pattern):
        # The paths do not exist, so infer_bracket_group_size falls back to the
        # pixel-luminance EV estimate -- drive it with flat frames whose
        # brightness ratio IS the EV step. Base 0.06 keeps +2 EV off the clip
        # point; clipping would collapse two exposures onto one EV and destroy
        # the very pattern under test.
        return [np.full((16, 16, 3), float(0.06 * (2.0 ** ev)), np.float32)
                for ev in pattern]

    cases = [
        ([-2, 0, 2] * 4, 3, "12 frames, 4 panels x 3 stops (old guess: 2)"),
        ([-1, 1] * 5, 2, "10 frames, 5 panels x 2 stops"),
        ([-2, -1, 1, 2] * 2, 4, "8 frames, 2 panels x 4 stops (old guess: 2)"),
        ([-2, 0, 2] * 3, 3, "9 frames, 3 panels x 3 stops"),
    ]
    for pattern, expect, label in cases:
        imgs = frames_for(pattern)
        paths = [f"/nonexistent/{i}.arw" for i in range(len(pattern))]
        got = infer_bracket_group_size(paths, images=imgs)
        check(f"group size from the EV pattern: {label}", got == expect, f"got {got}")


def test_panorama_hdr_end_to_end():
    """The assembled Panorama HDR path: bracket -> fuse -> stitch -> crop.

    This was one of the two menu entries that failed 100% of the time, so the
    whole chain needs to run, not just its parts. It also exercises the panel
    size normalisation: aligning each bracket crops it to that bracket's own
    common rectangle, so panels arrive at the stitcher a few pixels apart.
    """
    from raw_stitching import merge_hdr_panorama

    # A landscape's dynamic range: bright at the top, dark at the bottom, even
    # across the pan. Two gains of an already well-exposed frame would not do
    # -- fusion would average them back toward the midtones and the result is
    # too flat to match on (verified; cv2 Mertens flattens it the same way).
    # The range has to run VERTICALLY: a horizontal light ramp would tone the
    # two panels differently inside their own overlap, which is not what a
    # panorama looks like and defeats feature matching on synthetic texture.
    tex = _textured(h=400, w=1000, seed=21).astype(np.float32) / 255.0
    light = np.linspace(2.0, 0.12, 400, dtype=np.float32)[:, None, None]
    scene = tex * light
    panels = [scene[:, :600], scene[:, 400:]]
    frames, paths = [], []
    for pi, panel in enumerate(panels):
        for ei, (gain, dx, dy) in enumerate(((0.5, 0, 0), (3.0, 3, 2))):
            f = (np.clip(panel * gain, 0, 1) * 255).astype(np.uint8)
            frames.append(_shift(f, dx, dy) if (dx or dy) else f)
            paths.append(f"/x/panel{pi}_exp{ei}.arw")

    res = merge_hdr_panorama(frames, paths=paths, group_size=2, align=True)
    check("Panorama HDR completes end to end", res.success,
          "" if res.success else res.error_message)
    if res.success:
        h, w = res.image.shape[:2]
        check("Panorama HDR output is wider than one panel", w > 620, f"{w}x{h}")
        check("Panorama HDR output is a real image, not a sliver",
              h > 300 and int(res.image.max()) > 32, f"{w}x{h} max={int(res.image.max())}")


def test_group_size_mismatch_is_refused():
    from raw_stitching import merge_hdr_panorama

    imgs = [np.full((32, 32, 3), 100, np.uint8) for _ in range(7)]
    res = merge_hdr_panorama(imgs, paths=[f"/x/{i}.arw" for i in range(7)], group_size=3)
    check("a bracket layout that cannot divide the set is refused, not guessed",
          not res.success and "do not divide" in res.error_message, res.error_message[:60])


# --- 6. Output domain -------------------------------------------------------

def test_display_referred_encode():
    lin = np.full((8, 8, 3), 0.18, np.float32)  # 18% scene grey
    enc = to_display_referred(lin)
    val = int(enc[0, 0, 0]) >> 8
    check("scene-linear 18% grey encodes to a mid tone, not 46/255",
          100 <= val <= 130, f"{val}/255")
    u8 = np.full((8, 8, 3), 118, np.uint8)
    check("uint8 (already display-referred) passes through untouched",
          to_display_referred(u8) is u8)
    check("encode preserves shape and gives 16-bit",
          enc.shape == lin.shape and enc.dtype == np.uint16)


def test_panorama_depth_and_float64():
    big = _textured(h=400, w=1000, seed=12)
    left, right = big[:, :600], big[:, 400:]
    res = stitch_panorama([left, right])
    check("panorama stitches display-referred frames", res.success,
          "" if res.success else res.error_message)
    if res.success:
        check("panorama output is uint8 (cv2.Stitcher's only depth)",
              res.image.dtype == np.uint8)
    r64 = stitch_panorama([left.astype(np.float64) / 255.0, right.astype(np.float64) / 255.0])
    # The point is that float64 is scaled, not truncated to black by .astype.
    check("float64 input is scaled, not blackened",
          (not r64.success) or int(np.asarray(r64.image).max()) > 32,
          f"max {int(np.asarray(r64.image).max()) if r64.success else 'n/a'}")


# --- 5b. Merge selection is reordered from EXIF, not click order ------------

def test_reorder_merge_paths():
    none_t = [None] * 3
    hdr_in = ["/x/bright.CR3", "/x/dark.CR3", "/x/mid.CR3"]
    hdr_out, note = reorder_merge_paths(
        hdr_in, "hdr", evs=[2.0, -2.0, 0.0], capture_times=none_t,
    )
    check("HDR sorts darkest → brightest",
          hdr_out == ["/x/dark.CR3", "/x/mid.CR3", "/x/bright.CR3"], str(hdr_out))
    check("HDR reports that it reordered", bool(note))

    same, note2 = reorder_merge_paths(
        hdr_out, "hdr", evs=[-2.0, 0.0, 2.0], capture_times=none_t,
    )
    check("already-sorted HDR leaves a blank note", note2 == "" and same == hdr_out)

    pano_in = ["/x/c.CR3", "/x/a.CR3", "/x/b.CR3"]
    pano_out, _ = reorder_merge_paths(
        pano_in, "panorama", evs=[0, 0, 0], capture_times=[30.0, 10.0, 20.0],
    )
    check("panorama sorts by capture time, not click order",
          pano_out == ["/x/a.CR3", "/x/b.CR3", "/x/c.CR3"], str(pano_out))

    # Two panels of -2/0/+2, selected brightest-first and panel 2 before panel 1.
    ph_in = [
        "/x/p2_+2.CR3", "/x/p2_0.CR3", "/x/p2_-2.CR3",
        "/x/p1_+2.CR3", "/x/p1_0.CR3", "/x/p1_-2.CR3",
    ]
    ph_out, ph_note = reorder_merge_paths(
        ph_in,
        "hdr_panorama",
        evs=[2.0, 0.0, -2.0, 2.0, 0.0, -2.0],
        capture_times=[14.0, 13.0, 12.0, 4.0, 3.0, 2.0],
    )
    check(
        "Panorama HDR is panel order, then dark→bright inside each bracket",
        ph_out == [
            "/x/p1_-2.CR3", "/x/p1_0.CR3", "/x/p1_+2.CR3",
            "/x/p2_-2.CR3", "/x/p2_0.CR3", "/x/p2_+2.CR3",
        ],
        str(ph_out),
    )
    check("Panorama HDR mentions both time and EV", "bracket" in ph_note.lower())

    named = ["/x/IMG_003.CR3", "/x/IMG_001.CR3", "/x/IMG_002.CR3"]
    named_out, _ = reorder_merge_paths(
        named, "focus_stack", evs=[0, 0, 0], capture_times=[None, None, None],
    )
    check("missing capture time falls back to filename, not mtime",
          named_out == ["/x/IMG_001.CR3", "/x/IMG_002.CR3", "/x/IMG_003.CR3"],
          str(named_out))


# --- Dialog wiring ----------------------------------------------------------

def test_dialog_controls():
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui.hdr_panorama_dialog import HDRPanoramaDialog

    _app = QApplication.instance() or QApplication([])

    pano = HDRPanoramaDialog(["/x/a.CR3", "/x/b.CR3"], mode="panorama")
    check("no align checkbox in plain panorama mode (Stitcher registers itself)",
          not hasattr(pano, "align_cb") and pano.should_align() is True)
    check("no bracket-layout control outside Panorama HDR",
          pano.get_group_size() is None)

    ph = HDRPanoramaDialog([f"/x/{i}.CR3" for i in range(8)], mode="hdr_panorama")
    check("Panorama HDR exposes exposures-per-panel", ph.get_group_size() >= 2)
    # 3 is inside the spinner's range for 8 photos but does not divide them.
    ph.group_spin.setValue(3)
    check("an impossible bracket layout disables the merge button",
          not ph.merge_btn.isEnabled(), ph.group_hint.text())
    ph.group_spin.setValue(4)
    check("a valid layout re-enables it", ph.merge_btn.isEnabled(), ph.group_hint.text())

    fs = HDRPanoramaDialog(["/x/a.CR3", "/x/b.CR3"], mode="focus_stack")
    check("focus stacking keeps its align + parallax controls",
          fs.should_align() is True and fs.should_correct_parallax() is True)


def main() -> int:
    test_align_hdr_returns_frames_and_aligns()
    test_align_hdr_output_feeds_the_merge()
    test_focus_stack_auto_crop_trims_border()
    test_focus_stack_crop_survives_local_align()
    test_crop_keeps_dark_background_panorama()
    test_crop_ignores_isolated_black_pixels()
    test_crop_still_removes_a_real_border()
    test_hdr_sliders_move_the_output()
    test_group_size_inference()
    test_reorder_merge_paths()
    test_panorama_hdr_end_to_end()
    test_group_size_mismatch_is_refused()
    test_display_referred_encode()
    test_panorama_depth_and_float64()
    test_dialog_controls()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
