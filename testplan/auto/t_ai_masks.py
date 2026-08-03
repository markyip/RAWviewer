#!/usr/bin/env python3
"""AI mask generation (raw_ai_masks) -- contract + plumbing suite.

The ONNX weights are downloaded on first use and are NOT a test
dependency: everything here exercises the pure pre/post-processing and
the session plumbing against stub sessions, so the suite runs green on a
machine that has never fetched a model. That is deliberate -- the parts
most likely to break silently are the coordinate/normalization/shape
contracts around the model, not the model itself.

Checks:
  1. _to_float_rgb accepts uint8 and float, clamps scene-linear overshoot.
  2. _preprocess emits NCHW float32 at the model's input size.
  3. _to_alpha detects logits vs probabilities, squeezes any leading
     batch/channel axes, and resizes to the requested mask resolution.
  4. Output is a valid MaskLayer.alpha -- float32, (H, W), within [0, 1] --
     which is the whole integration contract with raw_mask_layers.
  5. Missing models degrade to None rather than raising.
  6. SamPredictor caches the encoder embedding per image_key (the property
     that makes click-to-mask interactive) and maps click coords through
     the longest-side resize.
  7. The SAM decoder picks the highest-IoU candidate mask, not index 0,
     and thresholds its logits rather than sigmoiding them.
  8. Depth feeds patch-aligned dimensions and min-max normalizes its
     output instead of sigmoiding it, and reports a flat map as nothing
     found rather than as a half-strength full-frame mask.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

import raw_ai_masks as aim  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


class _StubSession:
    """Minimal onnxruntime.InferenceSession stand-in."""

    def __init__(self, input_names, outputs, input_shape=None):
        self._input_names = list(input_names)
        self._outputs = outputs
        self._input_shape = input_shape or [1, 3, 64, 64]
        self.last_feeds = None

    class _IO:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    def get_inputs(self):
        return [self._IO(n, self._input_shape) for n in self._input_names]

    def run(self, _output_names, feeds):
        self.last_feeds = feeds
        return self._outputs


def test_float_rgb():
    u8 = np.full((4, 5, 3), 128, dtype=np.uint8)
    out = aim._to_float_rgb(u8)
    check("uint8 -> float32 [0,1]", out.dtype == np.float32 and abs(out.max() - 128 / 255.0) < 1e-6)

    # Scene-linear buffers overshoot 1.0 at speculars; unclamped values
    # would skew the normalization and wash out the mask.
    linear = np.full((4, 5, 3), 3.5, dtype=np.float32)
    check("float overshoot clamped", aim._to_float_rgb(linear).max() <= 1.0)

    gray = np.zeros((4, 5), dtype=np.uint8)
    check("grayscale promoted to 3ch", aim._to_float_rgb(gray).shape == (4, 5, 3))

    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    check("alpha channel dropped", aim._to_float_rgb(rgba).shape == (4, 5, 3))


def test_preprocess():
    rgb = np.random.rand(97, 133, 3).astype(np.float32)
    t = aim._preprocess(rgb, 64, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    check("preprocess NCHW shape", t.shape == (1, 3, 64, 64), f"got {t.shape}")
    check("preprocess dtype", t.dtype == np.float32)
    check("preprocess contiguous", t.flags["C_CONTIGUOUS"])


def test_to_alpha():
    # Logits (outside [0,1]) must get a sigmoid.
    logits = np.array([[[[-10.0, 10.0], [10.0, -10.0]]]], dtype=np.float32)
    a = aim._to_alpha(logits, (2, 2))
    check("logits sigmoided", a[0, 0] < 0.01 and a[0, 1] > 0.99, f"got {a}")

    # Already-probabilities pass through unchanged.
    probs = np.array([[[[0.25, 0.75], [0.5, 0.5]]]], dtype=np.float32)
    a = aim._to_alpha(probs, (2, 2))
    check("probabilities preserved", abs(a[0, 0] - 0.25) < 1e-5, f"got {a}")

    # Resize to the mask's working resolution.
    a = aim._to_alpha(probs, (16, 24))
    check("resized to target", a.shape == (16, 24), f"got {a.shape}")

    # This is the raw_mask_layers.MaskLayer.alpha contract.
    check("alpha dtype float32", a.dtype == np.float32)
    check("alpha in range", a.min() >= 0.0 and a.max() <= 1.0)


def test_matte_model_contract():
    """A stub 'model' must flow through to a valid MaskLayer alpha."""
    size = 64
    out = np.random.randn(1, 1, size, size).astype(np.float32)
    stub = _StubSession(["input"], [out], input_shape=[1, 3, size, size])
    saved = dict(aim._SESSIONS)
    try:
        aim._SESSIONS["subject"] = stub
        rgb = np.random.rand(200, 300, 3).astype(np.float32)
        alpha = aim.segment_subject(rgb, (200, 300))
        check("subject returns alpha", alpha is not None)
        if alpha is not None:
            check("subject alpha shape", alpha.shape == (200, 300), f"got {alpha.shape}")
            check("subject alpha dtype", alpha.dtype == np.float32)
            check("subject alpha range", alpha.min() >= 0.0 and alpha.max() <= 1.0)
            # Confirm it is a drop-in MaskLayer alpha.
            from raw_mask_layers import MaskLayer

            layer = MaskLayer(alpha, adjustments={"Exposure2012": 0.5}, name="AI Subject")
            check("alpha builds a MaskLayer", not layer.is_empty and layer.bbox() is not None)
    finally:
        aim._SESSIONS.clear()
        aim._SESSIONS.update(saved)


def test_graph_input_size_override():
    """Input size comes from the graph so a different export needs no code change."""
    stub = _StubSession(["input"], [], input_shape=[1, 3, 512, 512])
    check("graph size wins", aim._graph_input_size(stub, 1024) == 512)
    dynamic = _StubSession(["input"], [], input_shape=[1, 3, "h", "w"])
    check("dynamic axis falls back", aim._graph_input_size(dynamic, 1024) == 1024)


def test_missing_model_degrades():
    """No weights on disk must return None, never raise."""
    saved_models = {k: dict(v) for k, v in aim._MODELS.items()}
    saved_sessions = dict(aim._SESSIONS)
    try:
        aim._SESSIONS.clear()
        # Point at a file that cannot exist and a URL that cannot resolve.
        aim._MODELS["subject"]["filename"] = "definitely_not_a_model_xyz.onnx"
        aim._MODELS["subject"]["url"] = "https://127.0.0.1:1/nope.onnx"
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        try:
            result = aim.segment_subject(rgb, (32, 32))
            check("missing model returns None", result is None, f"got {type(result)}")
        except Exception as exc:  # noqa: BLE001
            check("missing model returns None", False, f"raised {exc!r}")
        check("op_is_ready False when absent", aim.op_is_ready("subject") is False)
    finally:
        for k, v in saved_models.items():
            aim._MODELS[k] = v
        aim._SESSIONS.clear()
        aim._SESSIONS.update(saved_sessions)


def test_sam_embedding_cache():
    """The encoder must run once per image -- this is what makes clicks fast."""
    calls = {"n": 0}
    embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)

    class _CountingEncoder(_StubSession):
        def run(self, _names, feeds):
            calls["n"] += 1
            return [embedding]

    enc = _CountingEncoder(["images"], [embedding], input_shape=[1, 3, 1024, 1024])
    saved = dict(aim._SESSIONS)
    try:
        aim._SESSIONS["sam_encoder"] = enc
        pred = aim.SamPredictor()
        rgb = np.random.rand(120, 200, 3).astype(np.float32)
        pred.set_image(rgb, image_key="img-a")
        pred.set_image(rgb, image_key="img-a")
        pred.set_image(rgb, image_key="img-a")
        check("encoder runs once per image", calls["n"] == 1, f"ran {calls['n']}x")
        check("predictor ready", pred.is_ready())

        pred.set_image(rgb, image_key="img-b")
        check("new image re-encodes", calls["n"] == 2, f"ran {calls['n']}x")

        # Longest side (200) maps to the encoder size (1024).
        check("scale from longest side", abs(pred._scale - 1024.0 / 200.0) < 1e-6)
        check("unpadded extent tracked", pred._unpadded_hw == (614, 1024), f"got {pred._unpadded_hw}")
    finally:
        aim._SESSIONS.clear()
        aim._SESSIONS.update(saved)


def test_sam_decoder_picks_best_iou():
    """SAM's slot 0 is not the best slot -- highest IoU must win."""
    embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)
    enc = _StubSession(["images"], [embedding], input_shape=[1, 3, 1024, 1024])

    # Three candidates; only index 2 is 'on', and it has the top IoU.
    masks = np.full((1, 3, 8, 8), -10.0, dtype=np.float32)
    masks[0, 2] = 10.0
    iou = np.array([[0.1, 0.2, 0.9]], dtype=np.float32)
    dec = _StubSession(
        [
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
        ],
        [masks, iou],
    )

    saved = dict(aim._SESSIONS)
    try:
        aim._SESSIONS["sam_encoder"] = enc
        aim._SESSIONS["sam_decoder"] = dec
        pred = aim.SamPredictor()
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        pred.set_image(rgb, image_key="k")
        alpha = pred.predict([(50.0, 50.0)], [1], (100, 100))
        check("decoder returns alpha", alpha is not None)
        if alpha is not None:
            check("best-IoU mask chosen", alpha.mean() > 0.9, f"mean={alpha.mean():.3f}")
            check("decoder alpha shape", alpha.shape == (100, 100))

        feeds = dec.last_feeds
        check("padding point appended", feeds["point_coords"].shape == (1, 2, 2))
        check("padding label is -1", feeds["point_labels"][0, -1] == -1.0)
        # Click at (50,50) on a 100px image scales by 1024/100.
        check(
            "click scaled to model frame",
            abs(float(feeds["point_coords"][0, 0, 0]) - 50.0 * (1024.0 / 100.0)) < 1e-3,
            f"got {feeds['point_coords'][0, 0]}",
        )
    finally:
        aim._SESSIONS.clear()
        aim._SESSIONS.update(saved)


def test_sam_thresholds_logits_not_sigmoids_them():
    """SAM's output is cut at 0, not squashed through a sigmoid.

    Regression test for a real shipped bug. SAM emits a signed field
    trained to be thresholded at zero -- measured [-12.3, +14.3] on a real
    frame -- but _to_alpha sigmoids anything outside [0, 1] because that is
    right for the matte models. The mask therefore never reached 1.0 even
    at the centre of the subject, and 8.8% of the frame sat in a visibly
    streaked half-transparent band.

    The tell is a wide logit range: sigmoid(2.0) is 0.88, so a mask built
    that way has interior pixels well below 1.0. Thresholding gives a
    solid interior and confines soft pixels to the resize's antialiasing.
    """
    embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)
    enc = _StubSession(["images"], [embedding], input_shape=[1, 3, 1024, 1024])

    # A logit field on SAM's real scale: strongly negative outside, mildly
    # positive inside. Under a sigmoid the "inside" would read ~0.88.
    masks = np.full((1, 1, 64, 64), -8.0, dtype=np.float32)
    masks[0, 0, 16:48, 16:48] = 2.0
    dec = _StubSession(
        [
            "image_embeddings", "point_coords", "point_labels",
            "mask_input", "has_mask_input", "orig_im_size",
        ],
        [masks],
    )

    saved = dict(aim._SESSIONS)
    try:
        aim._SESSIONS["sam_encoder"] = enc
        aim._SESSIONS["sam_decoder"] = dec
        pred = aim.SamPredictor()
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        pred.set_image(rgb, image_key="k")
        alpha = pred.predict([(32.0, 32.0)], [1], (64, 64))
        check("sam thresholded returns alpha", alpha is not None)
        if alpha is not None:
            check(
                "interior is fully opaque, not sigmoid(2.0)=0.88",
                alpha[32, 32] > 0.99,
                f"centre={alpha[32, 32]:.4f}",
            )
            check(
                "exterior is fully clear",
                alpha[2, 2] < 0.01,
                f"corner={alpha[2, 2]:.4f}",
            )
            # Soft pixels are the resize's antialiasing only, not a ramp
            # across the whole subject.
            mush = float(np.mean((alpha > 0.05) & (alpha < 0.95)))
            check("no half-transparent band", mush < 0.02, f"mush={mush:.4f}")
    finally:
        aim._SESSIONS.clear()
        aim._SESSIONS.update(saved)


def test_depth_feed_size():
    """Both fed dims must be multiples of 14, longest side at the target.

    The graph's own output shape is "14*floor(height/14)", so a feed that
    is not a multiple silently comes back smaller than asked for and the
    depth map lands misaligned against the frame.
    """
    for h, w in ((1100, 1650), (4484, 10228), (3000, 3000), (10, 4000)):
        fh, fw = aim._depth_feed_size(h, w, 518)
        check(
            f"feed {h}x{w} is patch-aligned",
            fh % 14 == 0 and fw % 14 == 0,
            f"got {fh}x{fw}",
        )
        check(f"feed {h}x{w} longest side <= 518", max(fh, fw) <= 518, f"got {fh}x{fw}")
        check(f"feed {h}x{w} is non-degenerate", min(fh, fw) >= 14, f"got {fh}x{fw}")

    # Aspect ratio survives the rounding, within one patch of slack.
    fh, fw = aim._depth_feed_size(1000, 2000, 518)
    check("aspect preserved", abs((fw / fh) - 2.0) < 0.1, f"got {fw}/{fh}")


def test_depth_normalization():
    """Depth is min-max normalized, NOT sigmoided.

    This is the one place the depth path must diverge from the matte
    models: the graph emits relative inverse depth on an arbitrary
    positive scale, and _to_alpha's sigmoid would crush the whole useful
    range onto the flat top of the curve.
    """
    # A linear ramp on a 0..4.5 scale, the range real frames produce.
    raw = np.linspace(0.0, 4.5, 32 * 32, dtype=np.float32).reshape(1, 32, 32)
    stub = _StubSession(["pixel_values"], [raw], input_shape=[1, 3, "h", "w"])
    saved = dict(aim._SESSIONS)
    try:
        aim._SESSIONS["depth"] = stub
        rgb = np.random.rand(64, 96, 3).astype(np.float32)
        alpha = aim.estimate_depth(rgb, (64, 96))
        check("depth returns alpha", alpha is not None)
        if alpha is not None:
            check("depth alpha shape", alpha.shape == (64, 96), f"got {alpha.shape}")
            check("depth alpha dtype", alpha.dtype == np.float32)
            check(
                "depth spans the full 0..1 range",
                alpha.min() < 1e-4 and alpha.max() > 1.0 - 1e-4,
                f"got [{alpha.min():.4f}, {alpha.max():.4f}]",
            )
            # The giveaway for a wrongly-applied sigmoid: sigmoid(4.5)=0.989
            # and sigmoid(2.25)=0.905, so the midpoint would sit near 0.9
            # instead of near 0.5.
            mid = float(np.median(alpha))
            check("midpoint is linear, not sigmoid", 0.4 < mid < 0.6, f"median={mid:.3f}")

        # A flat depth map carries no gradient to grade by -- that must be
        # reported as "nothing found", not returned as a constant 0.5 mask
        # covering the whole frame.
        flat = np.full((1, 32, 32), 2.0, dtype=np.float32)
        aim._SESSIONS["depth"] = _StubSession(
            ["pixel_values"], [flat], input_shape=[1, 3, "h", "w"]
        )
        check("flat depth returns None", aim.estimate_depth(rgb, (64, 96)) is None)
    finally:
        aim._SESSIONS.clear()
        aim._SESSIONS.update(saved)


def test_postprocess_helpers():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1.0
    feathered = aim.feather_alpha(alpha, 3)
    check("feather stays in range", feathered.min() >= 0.0 and feathered.max() <= 1.0)
    check("feather softens edge", 0.0 < feathered[7, 16] < 1.0, f"got {feathered[7, 16]}")
    check("feather no-op at radius 0", aim.feather_alpha(alpha, 0) is alpha)

    soft = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    hard = aim.binarize_alpha(soft, 0.5)
    check("binarize is 0/1", set(np.unique(hard)).issubset({0.0, 1.0}))


def test_op_download_mb_is_per_model():
    """The wait dialog used to claim every first use was 214 MB."""
    check("subject is the large one", aim.op_download_mb("subject") == 214)
    check("sky is the compact one", aim.op_download_mb("sky") == 84)
    check("depth is its own size", aim.op_download_mb("depth") == 94)
    check(
        "click sums encoder and decoder",
        aim.op_download_mb("click") == 27 + 16,
    )
    check("unknown op is zero", aim.op_download_mb("nope") == 0)


def main():
    print("AI mask generation (raw_ai_masks)")
    test_float_rgb()
    test_preprocess()
    test_to_alpha()
    test_matte_model_contract()
    test_graph_input_size_override()
    test_missing_model_degrades()
    test_sam_embedding_cache()
    test_sam_decoder_picks_best_iou()
    test_sam_thresholds_logits_not_sigmoids_them()
    test_depth_feed_size()
    test_depth_normalization()
    test_postprocess_helpers()
    test_op_download_mb_is_per_model()

    print("")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("All AI mask checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
