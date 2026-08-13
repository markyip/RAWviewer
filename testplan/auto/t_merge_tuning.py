#!/usr/bin/env python3
"""Merge tuning panel: live preview engine, focus points, and settings hand-off.

The panel tunes on a downscaled proxy and then runs the real merge at full
size, so the things worth pinning are the ones where those two can disagree:

  1. The expensive stage is prepared once and reused, and the cheap stage
     responds to every control.
  2. Focus points are stored NORMALISED, so a point placed on a 900px proxy
     lands in the same place on a full-resolution stack.
  3. Unticking a photo renumbers the fused set; indices must follow.
  4. Whatever the dialog ends up holding is what reaches the engines.
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

from focus_stacking import (  # noqa: E402
    apply_focus_points,
    fuse_prepared_stack,
    prepare_focus_stack,
    source_label_map,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _split_pair(h=240, w=320, seed=0):
    """Left sharp in A, right sharp in B."""
    rng = np.random.default_rng(seed)
    base = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    a = base.copy()
    a[:, w // 2:] = cv2.GaussianBlur(a[:, w // 2:], (0, 0), 6)
    b = base.copy()
    b[:, : w // 2] = cv2.GaussianBlur(b[:, : w // 2], (0, 0), 6)
    return a, b


def test_prepare_then_fuse_matches_one_shot():
    """The split stages must produce what the one-shot call produces."""
    from focus_stacking import focus_stack

    a, b = _split_pair(seed=1)
    prep = prepare_focus_stack([a, b], align=False, local_align=False, auto_crop=False)
    split = fuse_prepared_stack(prep, auto_crop=False)
    whole = focus_stack([a, b], align=False, local_align=False, auto_crop=False)
    check("prepare + fuse == focus_stack", np.array_equal(split, whole.image))


def test_fusion_responds_to_frame_weights():
    a, b = _split_pair(seed=2)
    prep = prepare_focus_stack([a, b], align=False, local_align=False, auto_crop=False)
    even = fuse_prepared_stack(prep, auto_crop=False)
    drop_b = fuse_prepared_stack(prep, auto_crop=False, frame_weights=[1.0, 0.0])
    check("dropping a frame's weight changes the fusion",
          not np.array_equal(even, drop_b))
    # With B's weight at zero, the right half must come from A -- which is
    # blurred there, so the result gets softer on that side.
    def sharp(img, sl):
        return float(cv2.Laplacian(cv2.cvtColor(img[:, sl], cv2.COLOR_RGB2GRAY),
                                   cv2.CV_64F).var())
    right = slice(160, None)
    check("a zeroed frame stops contributing its sharp region",
          sharp(drop_b, right) < 0.5 * sharp(even, right),
          f"{sharp(drop_b, right):.0f} vs {sharp(even, right):.0f}")


def test_focus_point_overrides_the_automatic_choice():
    """A point must hand its area to the chosen frame, and only its area."""
    a, b = _split_pair(seed=3)
    prep = prepare_focus_stack([a, b], align=False, local_align=False, auto_crop=False)
    auto = source_label_map(prep)
    # Left half is A's (index 0) by sharpness. Force a spot there to frame B.
    check("left region is automatically frame A",
          float((auto[:, :150] == 0).mean()) > 0.9,
          f"{float((auto[:, :150] == 0).mean()):.2f}")

    forced = source_label_map(prep, focus_points=[(0.2, 0.5, 0.1, 1)])
    yy, xx = np.mgrid[0:240, 0:320]
    spot = ((xx - 0.2 * 320) ** 2 + (yy - 0.5 * 240) ** 2) < 20 ** 2
    check("the clicked spot is handed to the chosen frame",
          float((forced[spot] == 1).mean()) > 0.9,
          f"{float((forced[spot] == 1).mean()):.2f}")
    far = xx > 280
    check("the rest of the frame is untouched",
          np.array_equal(forced[far], auto[far]))


def test_focus_points_are_resolution_independent():
    """The whole point of normalised coordinates: same spot at any size."""
    a, b = _split_pair(h=240, w=320, seed=4)
    big = [cv2.resize(x, (640, 480), interpolation=cv2.INTER_LINEAR) for x in (a, b)]
    point = [(0.25, 0.5, 0.1, 1)]

    small_prep = prepare_focus_stack([a, b], align=False, local_align=False, auto_crop=False)
    big_prep = prepare_focus_stack(big, align=False, local_align=False, auto_crop=False)
    small = source_label_map(small_prep, focus_points=point)
    large = source_label_map(big_prep, focus_points=point)

    # The forced region must sit at the same *relative* place in both.
    def centroid(labels):
        ys, xs = np.where(labels == 1)
        if xs.size == 0:
            return None
        return float(xs.mean()) / labels.shape[1], float(ys.mean()) / labels.shape[0]

    cs, cl = centroid(small), centroid(large)
    check("a normalised point lands in the same relative place at 2x the size",
          cs is not None and cl is not None
          and abs(cs[0] - cl[0]) < 0.06 and abs(cs[1] - cl[1]) < 0.06,
          f"{cs} vs {cl}")


def test_apply_focus_points_ignores_bad_input():
    maps = [np.ones((20, 30), np.float32), np.ones((20, 30), np.float32)]
    check("no points is a no-op", apply_focus_points(maps, None) is maps)
    out = apply_focus_points(maps, [(0.5, 0.5, 0.1, 9)])  # frame index out of range
    check("an out-of-range frame index is skipped, not crashed",
          len(out) == 2 and np.isfinite(out[0]).all())


def test_preview_engine_reuses_alignment():
    """The engine must prepare once and re-fuse, or the panel cannot be live."""
    from rawviewer_ui import merge_tuning

    calls = {"n": 0}
    real = merge_tuning.MergePreviewEngine._render_focus

    a, b = _split_pair(seed=5)
    engine = merge_tuning.MergePreviewEngine("focus_stack")
    engine.set_images([a, b])
    opts = {
        "active": [0, 1], "align": False, "parallax": False, "auto_crop": False,
        "frame_weights": [1.0, 1.0], "focus_points": [],
    }
    import focus_stacking

    original_prepare = focus_stacking.prepare_focus_stack

    def counting_prepare(*args, **kwargs):
        calls["n"] += 1
        return original_prepare(*args, **kwargs)

    focus_stacking.prepare_focus_stack = counting_prepare
    try:
        engine.render(opts)
        engine.render(dict(opts, frame_weights=[1.0, 0.4]))
        engine.render(dict(opts, focus_points=[(0.5, 0.5, 0.1, 1)]))
        check("alignment is prepared once across three renders", calls["n"] == 1,
              f"prepared {calls['n']}x")
        engine.render(dict(opts, align=True))
        check("changing an alignment option re-prepares", calls["n"] == 2,
              f"prepared {calls['n']}x")
    finally:
        focus_stacking.prepare_focus_stack = original_prepare
        merge_tuning.MergePreviewEngine._render_focus = real


def test_dialog_exposes_tuned_settings():
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui.hdr_panorama_dialog import HDRPanoramaDialog

    _app = QApplication.instance() or QApplication([])
    dlg = HDRPanoramaDialog([f"/x/{i}.CR3" for i in range(4)], mode="focus_stack")

    check("one control row per photo", len(dlg.frame_rows) == 4)
    check("weights default to neutral", dlg.get_image_weights() == [1.0] * 4)
    dlg.frame_rows[1].weight_slider.setValue(4)
    check("a weight slider reaches get_image_weights",
          dlg.get_image_weights() == [1.0, 0.4, 1.0, 1.0], str(dlg.get_image_weights()))

    dlg.frame_rows[2].include_cb.setChecked(False)
    check("unticking drops that photo from the active set",
          dlg.get_active_image_paths() == ["/x/0.CR3", "/x/1.CR3", "/x/3.CR3"])
    check("and from the weights, in the same order",
          dlg.get_image_weights() == [1.0, 0.4, 1.0], str(dlg.get_image_weights()))

    check("focus points start empty", dlg.get_focus_points() == [])


def test_panorama_hdr_refuses_a_single_bracket():
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui.hdr_panorama_dialog import HDRPanoramaDialog

    _app = QApplication.instance() or QApplication([])
    too_few = HDRPanoramaDialog([f"/x/{i}.CR3" for i in range(2)], mode="hdr_panorama")
    check("merge is disabled with only one bracket",
          not too_few.merge_btn.isEnabled())
    enough = HDRPanoramaDialog([f"/x/{i}.CR3" for i in range(4)], mode="hdr_panorama")
    check("merge is enabled with two panels of 2",
          enough.merge_btn.isEnabled())
    eight = HDRPanoramaDialog([f"/x/{i}.CR3" for i in range(8)], mode="hdr_panorama")
    eight.group_spin.setValue(3)
    check("merge is disabled when the count does not divide",
          not eight.merge_btn.isEnabled())


def test_dialog_focus_points_track_the_active_set():
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui.hdr_panorama_dialog import HDRPanoramaDialog

    _app = QApplication.instance() or QApplication([])
    dlg = HDRPanoramaDialog([f"/x/{i}.CR3" for i in range(3)], mode="focus_stack")
    # Pretend the proxies arrived so clicks are accepted.
    dlg._proxy_of_row = {0: 0, 1: 1, 2: 2}
    dlg._engine.images = [np.zeros((8, 8, 3), np.uint8)] * 3
    dlg._refresh_focus_targets()
    dlg.focus_target_combo.setCurrentIndex(2)  # third photo
    dlg._on_preview_clicked(0.4, 0.6)
    pts = dlg.get_focus_points()
    check("a click records a normalised point for the chosen photo",
          len(pts) == 1 and pts[0][3] == 2 and 0 < pts[0][0] < 1 and 0 < pts[0][1] < 1,
          str(pts))

    dlg.frame_rows[1].weight_slider.setValue(4)
    check("changing a frame weight does not drop focus points",
          dlg.get_focus_points() == pts)

    dlg.frame_rows[0].include_cb.setChecked(False)
    check("changing the photo set clears points rather than re-aiming them",
          dlg.get_focus_points() == [])


def test_source_map_is_cropped_with_the_fusion():
    """Auto-crop must shrink the source map to the fused frame, not leave it full-size."""
    a, b = _split_pair(h=80, w=100, seed=6)
    # Fake coverage that trims a 10px border so crop is deterministic without ECC.
    from focus_stacking import FocusStackPrep, fuse_prepared_stack, source_label_map

    prep = prepare_focus_stack([a, b], align=False, local_align=False, auto_crop=False)
    cover = np.zeros((80, 100), np.uint8)
    cover[10:70, 10:90] = 1
    prep = FocusStackPrep(
        floats=prep.floats,
        sharp_maps=prep.sharp_maps,
        guide=prep.guide,
        coverage=[cover, cover],
        dtype=prep.dtype,
        warnings=[],
    )
    fused = fuse_prepared_stack(prep, auto_crop=True, seam_aware=False)
    labels = source_label_map(prep, auto_crop=True)
    check(
        "source map matches the cropped fusion's shape",
        labels.shape == fused.shape[:2],
        f"labels {labels.shape} vs fused {fused.shape[:2]}",
    )


def test_realign_tuning_drops_failed_decodes():
    from rawviewer_ui.merge_tuning import realign_tuning_to_decoded

    paths = ["/a.CR3", "/b.CR3", "/c.CR3"]
    valid = ["/a.CR3", "/c.CR3"]
    weights, points = realign_tuning_to_decoded(
        paths, valid, [1.0, 0.4, 2.0], [(0.5, 0.5, 0.1, 2)]
    )
    check("weights follow surviving frames", weights == [1.0, 2.0], str(weights))
    check("focus point index remaps onto the surviving set",
          points == [(0.5, 0.5, 0.1, 1)], str(points))
    weights2, points2 = realign_tuning_to_decoded(
        paths, valid, [1.0, 0.4, 2.0], [(0.5, 0.5, 0.1, 1)]
    )
    check("a point on a failed frame is dropped", points2 == [], str(points2))
    same_w, same_p = realign_tuning_to_decoded(paths, paths, [1.0, 1.0, 1.0], [])
    check("a complete decode is a no-op", same_w == [1.0, 1.0, 1.0] and same_p == [])


def test_hdr_panorama_preview_slices_frame_weights():
    from rawviewer_ui.merge_tuning import MergePreviewEngine

    captured = {}

    def fake_merge(frames, **kwargs):
        captured["n"] = len(frames)
        captured["weights"] = kwargs.get("image_weights")
        return np.zeros((8, 8, 3), np.uint8)

    engine = MergePreviewEngine("hdr_panorama")
    engine.set_images([np.zeros((8, 8, 3), np.uint8) for _ in range(4)])
    import raw_stitching

    original = raw_stitching.merge_hdr_exposure_fusion
    raw_stitching.merge_hdr_exposure_fusion = fake_merge
    try:
        engine._render_hdr({
            "active": [0, 1, 2, 3],
            "align": False,
            "group_size": 2,
            "preview_panel": 1,
            "highlight": 1.0,
            "shadow": 1.0,
            "midtone": 1.0,
            "frame_weights": [0.1, 0.2, 0.3, 0.4],
        })
    finally:
        raw_stitching.merge_hdr_exposure_fusion = original
    check("preview fuses one bracket, not the whole set", captured.get("n") == 2,
          str(captured.get("n")))
    check("preview weights match that bracket, not the full set",
          captured.get("weights") == [0.3, 0.4], str(captured.get("weights")))


def test_source_map_colors_follow_the_rows():
    from rawviewer_ui.merge_tuning import _SOURCE_COLORS, source_map_rgb

    labels = np.array([[0, 1], [1, 0]], dtype=np.int32)
    colors = [_SOURCE_COLORS[1], _SOURCE_COLORS[3]]  # rows 1 and 3 still ticked
    rgb = source_map_rgb(labels, colors)
    check("source map paints each fused frame in its row's colour",
          tuple(rgb[0, 0]) == _SOURCE_COLORS[1] and tuple(rgb[0, 1]) == _SOURCE_COLORS[3])


def main() -> int:
    test_prepare_then_fuse_matches_one_shot()
    test_fusion_responds_to_frame_weights()
    test_focus_point_overrides_the_automatic_choice()
    test_focus_points_are_resolution_independent()
    test_apply_focus_points_ignores_bad_input()
    test_preview_engine_reuses_alignment()
    test_dialog_exposes_tuned_settings()
    test_panorama_hdr_refuses_a_single_bracket()
    test_dialog_focus_points_track_the_active_set()
    test_source_map_is_cropped_with_the_fusion()
    test_realign_tuning_drops_failed_decodes()
    test_hdr_panorama_preview_slices_frame_weights()
    test_source_map_colors_follow_the_rows()
    # The dialogs above kicked off background proxy decodes for paths that do
    # not exist. Let them finish before the interpreter tears Qt down, or the
    # teardown races them and prints RuntimeErrors that are not failures.
    try:
        from PyQt6.QtCore import QThreadPool

        QThreadPool.globalInstance().waitForDone(5000)
    except Exception:
        pass
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
