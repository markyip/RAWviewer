#!/usr/bin/env python3
"""Manual lens distortion + anamorphic desqueeze wiring -- headless.

Distortion is a single-k1 radial correction for lenses with no matched
profile. Positive corrects PINCUSHION (telephoto bowing lines inward),
negative corrects BARREL. The signs are asserted against synthetically
distorted grids rather than reasoned about, because getting them backwards
looks plausible in code and obviously wrong on a photo.

Also covers the desqueeze bug this work uncovered: the pipeline's cache
invalidation list was a hand-maintained copy of raw_transform.TRANSFORM_KEYS
that had drifted, omitting AnamorphicRatio. Changing the ratio therefore did
not invalidate the cached pre-tone stage, so the live preview kept showing
the previous ratio -- the control looked dead while exporting correctly.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import raw_transform as rt  # noqa: E402
from raw_transform import apply_geometry, has_geometry  # noqa: E402

GRID = 400


def _grid():
    g = np.zeros((GRID, GRID, 3), np.float32)
    for x in range(20, GRID, 40):
        g[:, x - 1 : x + 2] = 1.0
    for y in range(20, GRID, 40):
        g[y - 1 : y + 2, :] = 1.0
    return g


def _distort(img, k):
    """Apply the same radial model the corrector inverts."""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    norm = float(np.hypot(cx, cy))
    ys, xs = np.indices((h, w), dtype=np.float32)
    dx = (xs - cx) / norm
    dy = (ys - cy) / norm
    f = 1.0 + k * (dx * dx + dy * dy)
    return cv2.remap(
        img,
        (cx + dx * f * norm).astype(np.float32),
        (cy + dy * f * norm).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _bow(img):
    """RMS bow of horizontal grid lines as a fraction of height.

    Normalised, so the crop a positive correction applies does not change the
    score -- an absolute pixel measure would reward cropping.
    """
    c = img[:, :, 0]
    h, w = c.shape
    scores = []
    for frac in (0.18, 0.30, 0.70, 0.82):
        target = h * frac
        ys, xs = [], []
        for x in range(int(w * 0.10), int(w * 0.90), 4):
            idx = np.where(c[:, x] > 0.35)[0]
            if len(idx) == 0:
                continue
            ys.append(idx[np.argmin(np.abs(idx - target))])
            xs.append(x)
        if len(ys) < 10:
            continue
        ys = np.asarray(ys, float)
        xs = np.asarray(xs, float)
        fit = np.polyval(np.polyfit(xs, ys, 1), xs)
        scores.append(float(np.sqrt(np.mean((ys - fit) ** 2)) / h))
    return float(np.mean(scores)) if scores else float("nan")


def _slider_for(k):
    """Slider value that inverts a source distorted by ``k``."""
    return -k / rt._DISTORTION_MAX * 100.0


def test_positive_slider_corrects_pincushion():
    src = _distort(_grid(), -0.22)  # pincushion: lines bow inward
    raw = _bow(src)
    fixed = _bow(apply_geometry(src.copy(), {"Distortion": _slider_for(-0.22)}))
    wrong = _bow(apply_geometry(src.copy(), {"Distortion": -_slider_for(-0.22)}))
    assert fixed < raw / 2.0, f"pincushion not corrected: {raw:.5f} -> {fixed:.5f}"
    assert wrong > fixed, f"wrong sign scored better ({wrong:.5f} vs {fixed:.5f})"
    print(f"  OK   +slider corrects pincushion ({raw:.5f} -> {fixed:.5f})")


def test_negative_slider_corrects_barrel():
    """Scored by similarity to the reference: barrel correction does not crop,
    so the framing is comparable and correlation is the cleaner measure."""
    ref = _grid()
    src = _distort(ref, 0.22)  # barrel: lines bow outward

    def centre(img, frac=0.55):
        h, w = img.shape[:2]
        ch, cw = int(h * frac), int(w * frac)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(img[y0 : y0 + ch, x0 : x0 + cw, 0], (200, 200),
                          interpolation=cv2.INTER_AREA)

    target = centre(ref)

    def score(img):
        return float(np.corrcoef(centre(img).ravel(), target.ravel())[0, 1])

    fixed = score(apply_geometry(src.copy(), {"Distortion": _slider_for(0.22)}))
    assert fixed > 0.9, f"barrel correction did not restore the grid (r={fixed:.3f})"
    assert fixed > score(src), "correction made it worse than the distorted source"
    print(f"  OK   -slider corrects barrel (r={fixed:.3f})")


def test_zero_is_a_no_op():
    g = _grid()
    assert apply_geometry(g, {"Distortion": 0.0}) is g, "zero distortion copied the buffer"
    assert not has_geometry({"Distortion": 0.0})
    assert has_geometry({"Distortion": 5.0}), "Distortion not seen as geometry"
    print("  OK   zero is identity; non-zero registers as geometry")


def test_output_never_grows_and_has_no_border_fill():
    """The module's hard rule: no pixels the source did not have."""
    g = _grid()
    for slider in (-100.0, -50.0, 50.0, 100.0):
        out = apply_geometry(g.copy(), {"Distortion": slider})
        assert out.shape[0] <= GRID and out.shape[1] <= GRID, (
            f"slider {slider}: output grew to {out.shape}"
        )
        # A black border would show as an all-zero edge row/column; the source
        # grid has bright lines throughout, so any fully-dark edge is fill.
        for name, edge in (
            ("top", out[0]), ("bottom", out[-1]), ("left", out[:, 0]), ("right", out[:, -1]),
        ):
            assert float(np.abs(edge).max()) > 0.0, f"slider {slider}: {name} edge is border fill"
    print("  OK   output never grows and shows no border fill")


def test_valid_scale_solves_its_equation():
    for k in (0.05, 0.15, 0.30):
        s = rt._distortion_valid_scale(k)
        assert 0.0 < s < 1.0
        assert abs(s * (1.0 + k * s * s) - 1.0) < 1e-3, f"k={k}: s={s} is not the root"
    assert rt._distortion_valid_scale(0.0) == 1.0
    assert rt._distortion_valid_scale(-0.2) == 1.0, "negative k needs no crop"
    print("  OK   valid-area scale solves s(1+k s^2)=1")


def test_distortion_composes_with_straighten():
    """Distortion runs first; a combined edit must still produce a valid frame."""
    g = _grid()
    out = apply_geometry(g.copy(), {"Distortion": 60.0, "CropAngle": 4.0})
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape[0] <= GRID and out.shape[1] <= GRID
    assert float(out.max()) > 0.3, "combined transform produced an empty frame"
    print(f"  OK   composes with straighten ({out.shape[1]}x{out.shape[0]})")


# ------------------------------------------------------- cache-key drift


def test_pipeline_cache_covers_every_transform_key():
    """The bug behind the dead-looking desqueeze control."""
    import raw_edit_pipeline as rep

    missing = set(rt.TRANSFORM_KEYS) - set(rep._TRANSFORM_KEYS)
    assert not missing, f"transform keys missing from the preview cache: {sorted(missing)}"
    assert "AnamorphicRatio" in rep._TRANSFORM_KEYS
    assert "Distortion" in rep._TRANSFORM_KEYS
    print(f"  OK   preview cache covers all {len(rt.TRANSFORM_KEYS)} transform keys")


def test_transform_keys_invalidate_pre_tone():
    import raw_edit_pipeline as rep

    for key in ("AnamorphicRatio", "Distortion"):
        assert key in rep._PRE_TONE_KEYS, f"{key} does not invalidate the pre-tone stage"
    print("  OK   geometry keys invalidate the pre-tone stage")


def test_anamorphic_actually_resizes():
    g = _grid()
    out = apply_geometry(g.copy(), {"AnamorphicRatio": 1.5})
    assert out.shape[1] > g.shape[1] * 1.4, f"desqueeze did not widen: {out.shape}"
    assert out.shape[0] == g.shape[0], "desqueeze changed height"
    print(f"  OK   1.5x desqueeze widens {g.shape[1]} -> {out.shape[1]}")


# ------------------------------------------------------------------ panel UI


def test_panel_exposes_distortion_and_anamorphic_buttons():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.show()
    assert getattr(p, "sect_anamorphic", None) is not None, "no Anamorphic section"
    labels = [b.text() for b in p._anamorphic_btns]
    assert labels == ["1.0x", "1.33x", "1.5x", "1.6x", "2.0x"], labels
    assert "Distortion" in p.get_adjustments(), "Distortion missing from adjustments"

    for btn in p._anamorphic_btns:
        if btn.text() == "1.5x":
            btn.click()
    assert p.get_adjustments()["AnamorphicRatio"] == 1.5
    checked = [b.text() for b in p._anamorphic_btns if b.isChecked()]
    assert checked == ["1.5x"], f"selection not exclusive: {checked}"

    # Re-clicking the active ratio must not leave nothing selected.
    for btn in p._anamorphic_btns:
        if btn.text() == "1.5x":
            btn.click()
    assert p.get_adjustments()["AnamorphicRatio"] == 1.5
    assert [b.text() for b in p._anamorphic_btns if b.isChecked()] == ["1.5x"]

    # set_adjustments must drive the buttons, not just the stored value.
    adj = p.get_adjustments()
    adj["AnamorphicRatio"] = 2.0
    adj["Distortion"] = -42.0
    p.set_adjustments(adj)
    out = p.get_adjustments()
    assert out["AnamorphicRatio"] == 2.0
    assert out["Distortion"] == -42.0
    assert [b.text() for b in p._anamorphic_btns if b.isChecked()] == ["2.0x"]
    print("  OK   panel exposes both, exclusively and round-trippably")


def test_distortion_excluded_from_burst_apply():
    """Per-lens geometry must not be sprayed across a burst group."""
    from raw_adjustments import EXCLUDED_BURST_GROUP_KEYS, fundamental_adjustments_for_burst

    assert "Distortion" in EXCLUDED_BURST_GROUP_KEYS
    fund = fundamental_adjustments_for_burst({"Distortion": 50.0, "Exposure2012": 1.0})
    assert "Distortion" not in fund
    assert "Exposure2012" in fund
    print("  OK   Distortion is excluded from burst-group apply")


def main() -> int:
    test_positive_slider_corrects_pincushion()
    test_negative_slider_corrects_barrel()
    test_zero_is_a_no_op()
    test_output_never_grows_and_has_no_border_fill()
    test_valid_scale_solves_its_equation()
    test_distortion_composes_with_straighten()
    test_pipeline_cache_covers_every_transform_key()
    test_transform_keys_invalidate_pre_tone()
    test_anamorphic_actually_resizes()
    test_panel_exposes_distortion_and_anamorphic_buttons()
    test_distortion_excluded_from_burst_apply()
    print("\nPASS t_transform_distortion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
