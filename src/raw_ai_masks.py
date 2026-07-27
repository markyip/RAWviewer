"""
AI mask generation for raw_mask_layers.MaskLayer.

Every entry point here returns exactly one thing: a float32 (H, W) alpha
buffer in [0, 1] at the caller's requested resolution -- i.e. a drop-in
``MaskLayer.alpha``. Nothing in this module knows about adjustments,
compositing or XMP; the mask stack already handles all of that, so an AI
mask is indistinguishable from a hand-painted one once created (it can be
brushed, erased, inverted, and round-trips through mask_layers_xmp for
free).

Three models, all ONNX, all commercially licensed:

    subject  BiRefNet (MIT)        1024x1024  one-click foreground matte
    sky      U^2-Net skyseg (MIT)   320x320   one-click sky matte
    click    MobileSAM (Apache 2)  1024x1024  point-prompted segmentation

Weights live in ``<repo>/models`` and are fetched on first use with
SHA-256 verification, like the denoise models. Unlike them, they are
fetched from their upstream publishers rather than mirrored in this
repo's Git LFS -- see the note above ``_MODELS``. On Windows the pixi
installer payload already ships that directory, so the download path is
macOS/Linux-only in practice.

MobileSAM is split encoder/decoder on purpose. The encoder is the
expensive half (~0.3-1.5s on CPU) but depends only on the image, so
``SamPredictor.set_image`` runs it once and caches the 256x64x64
embedding; each subsequent click runs only the decoder (~5-15ms), which
is what makes click-to-mask feel interactive rather than modal. The
embedding cache is keyed on the base image identity the same way
raw_mask_layers' composite cache is keyed on ``id(img)``.

Inference deliberately runs at each model's native input resolution
(1024 or 320) rather than the edit base's working resolution: these are
fully-convolutional segmentation nets trained at a fixed scale, and the
resulting alpha is then bilinearly resized up to the mask resolution.
Masks tolerate that upsample in a way pixel output never could -- which
is the whole reason segmentation fits this app and generative editing
does not.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Weights come straight from their upstream publishers rather than being
# re-hosted in this repo's Git LFS. 445 MB of mirrored model is real LFS
# quota and bandwidth for files we do not produce and never modify; the
# SHA-256 below is what makes a third-party host safe to depend on, since a
# substituted file fails verification and is discarded rather than loaded.
#
# The tradeoff is that upstream can move a file. That is a 404 on first use
# with a clear log line, not a corrupt install -- and every URL here is
# re-checkable with scripts/fetch_ai_mask_models.py.
#
# "archive_member" means the publisher ships the model inside a zip rather
# than as a loose file; the download path extracts that one member and
# verifies its hash, not the archive's.

# Both MobileSAM halves live in one archive from the same publisher, so a
# single download serves both.
_MOBILESAM_ZIP = (
    "https://huggingface.co/vietanhdev/segment-anything-onnx-models"
    "/resolve/main/mobile_sam_20230629.zip"
)

# SHA-256 of each published .onnx. Empty string = "not yet published";
# _download_verified() then warns and accepts the download unverified so
# development builds work before the weights are committed. Fill these in
# (scripts/fetch_ai_mask_models.py prints them) before shipping.
_MODELS = {
    "subject": {
        "filename": "birefnet.onnx",
        "url": "https://huggingface.co/onnx-community/BiRefNet_lite/resolve/main/onnx/model.onnx",
        "sha256": "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
        "input_size": 1024,
        # ImageNet mean/std, the normalization BiRefNet was trained with.
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "label": "Subject",
    },
    "sky": {
        "filename": "skyseg.onnx",
        # fp16 export of the same U^2-Net: 84 MB against the fp32's 168, for
        # output that is pixel-identical (measured IoU 1.0000, mean absolute
        # difference 0.00000) and 6% slower -- 30ms on a 320px input.
        #
        # The equivalent trade is NOT available for the subject model above.
        # Both ways of shrinking it were measured and both cost more time than
        # they save space, on the same principle: reduced precision only pays
        # where the runtime has native kernels for it, and Apple Silicon's CPU
        # has neither fp16 nor int8 kernels that beat its fp32 ones.
        #
        #   BiRefNet_lite fp16    214 -> 109 MB, IoU 0.9997, 32% SLOWER
        #                         (10.1s vs 7.6s per press)
        #   BiRefNet_lite int8    214 -> 147 MB, mean IoU 0.9917 over four real
        #                         photographs, 42% SLOWER (10.5s vs 7.4s)
        #
        # Quality was never the objection -- neither is visible in a mask. The
        # objection is that Smart Object is already the slowest thing in the
        # app, so seconds cost more than megabytes here.
        #
        # If this is revisited: a naive int8 quantise only reaches 193 MB,
        # because 61% of this export's weights sit behind Identity
        # pass-through nodes (alongside 7296 Constant nodes) that the
        # quantiser cannot see through. Folding the graph first
        # (ORT_ENABLE_ALL) and quantising that reaches 147 MB. Worth doing
        # only on a platform whose runtime is fast at int8 -- x86 with VNNI,
        # or DirectML -- which would mean a per-platform model choice here.
        "url": "https://huggingface.co/voyagerfromeast/skyseg/resolve/main/skyseg_fp16.onnx",
        "sha256": "74d87f4a69378a610a6be662f859c38cfbdfdd75ff74bbfc54842965ed6fc9f7",
        "input_size": 320,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "label": "Sky",
    },
    "sam_encoder": {
        "filename": "mobilesam_encoder.onnx",
        "url": _MOBILESAM_ZIP,
        "archive_member": "mobile_sam.encoder.onnx",
        "sha256": "20deef402855b31222b528f52b04807e41ebe47216ac0e39a0729f43491a0209",
        "input_size": 1024,
        # SAM normalizes with these 0-255-scale constants, not ImageNet's.
        "mean": (123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0),
        "std": (58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0),
        "label": "Click",
    },
    "sam_decoder": {
        "filename": "mobilesam_decoder.onnx",
        "url": _MOBILESAM_ZIP,
        # MobileSAM replaces only SAM's encoder and reuses its mask decoder
        # verbatim, which is why the decoder member carries a vit_h name.
        "archive_member": "sam_vit_h_4b8939.decoder.onnx",
        "sha256": "22cf85e35d14182f4b4712364264c06b22edbef63f065189586f080ef4e2f325",
        "input_size": 0,  # decoder takes an embedding, not an image
        "label": "Click",
    },
}

# Which model files each user-facing operation needs present.
_OP_REQUIREMENTS = {
    "subject": ("subject",),
    "sky": ("sky",),
    "click": ("sam_encoder", "sam_decoder"),
}


def _models_dir() -> str:
    """Same directory the denoise models use -- see onnx_scunet._models_dir."""
    try:
        from onnx_scunet import _models_dir as _scunet_models_dir

        return _scunet_models_dir()
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def model_path(kind: str) -> str:
    """Absolute path to a model file (may not exist yet)."""
    spec = _MODELS[kind]
    return os.path.join(_models_dir(), spec["filename"])


def _sha256_of_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, dest: str, sha256: str, archive_member: str = "") -> bool:
    """Download to a temp file, verify, then move into place.

    Verifying before the rename means a failed or tampered download never
    leaves a loadable-but-corrupt model behind -- same contract as
    onnx_scunet.ensure_scunet_model_downloaded.

    archive_member: when the publisher ships the model inside a zip, name the
    member to extract. The hash is checked against that member's bytes, not
    the archive's, so re-compression upstream does not spuriously fail while
    a substituted model still does.
    """
    import urllib.request

    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)

        if archive_member:
            import zipfile

            try:
                with zipfile.ZipFile(tmp) as archive:
                    payload = archive.read(archive_member)
            except (zipfile.BadZipFile, KeyError):
                logger.warning(
                    "[AIMASK] %s does not contain member %s", url, archive_member, exc_info=True
                )
                os.remove(tmp)
                return False
            with open(tmp, "wb") as fh:
                fh.write(payload)

        if sha256:
            actual = _sha256_of_file(tmp)
            if actual.lower() != sha256.lower():
                logger.warning(
                    "[AIMASK] Download from %s failed SHA-256 verification", url
                )
                os.remove(tmp)
                return False
        else:
            logger.warning(
                "[AIMASK] No SHA-256 recorded for %s -- accepting unverified download",
                os.path.basename(dest),
            )
        os.replace(tmp, dest)
        return True
    except Exception:
        logger.warning("[AIMASK] Failed to download %s", url, exc_info=True)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def ensure_model_downloaded(kind: str) -> bool:
    path = model_path(kind)
    if os.path.exists(path):
        return True
    spec = _MODELS[kind]
    return _download_verified(
        spec["url"], path, spec["sha256"], spec.get("archive_member", "")
    )


def ai_masks_enabled() -> bool:
    """False in the Standard edition, where AI masks are not offered.

    No model is bundled in either edition -- they all download on first use
    -- so this saves no installer size. What it guarantees is that a Standard
    user never triggers a 214 MB download by clicking a mask button. The
    edition line is drawn at "does pressing this fetch a large model", not at
    install footprint.

    Unaffected, because they need no model at all: brush, linear and radial
    gradients, and dodge/burn Edge Assist (a guided filter, not a
    segmentation net).
    """
    if os.environ.get("RAWVIEWER_FORCE_AI_MASKS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    try:
        from rawviewer_profile import is_lite_build

        return not is_lite_build()
    except Exception:
        # Unknown profile (dev checkout, tests): behave like Plus rather than
        # silently removing working features.
        return True


def ensure_op_available(op: str) -> bool:
    """Download every model the given operation needs. True if all present."""
    if not ai_masks_enabled():
        return False
    return all(ensure_model_downloaded(k) for k in _OP_REQUIREMENTS.get(op, ()))


def op_is_ready(op: str) -> bool:
    """True if the operation's models are already on disk (no download)."""
    if not ai_masks_enabled():
        return False
    return all(os.path.exists(model_path(k)) for k in _OP_REQUIREMENTS.get(op, ()))


def available_ops() -> dict:
    if not ai_masks_enabled():
        return {op: False for op in _OP_REQUIREMENTS}
    return {op: op_is_ready(op) for op in _OP_REQUIREMENTS}


# ----------------------------------------------------------------------
# ONNX session handling
# ----------------------------------------------------------------------

_SESSIONS: dict = {}
_SESSION_LOCK = threading.Lock()


def _make_session(path: str):
    """Build an InferenceSession, mirroring SCUNetONNX._init_session.

    Providers are intersected with what this onnxruntime build actually
    reports, so a CPU-only wheel doesn't raise on a missing CUDA/DirectML
    EP -- the same guard onnx_scunet uses.
    """
    import onnxruntime as ort

    requested = []
    if sys.platform.startswith("win"):
        requested = ["DmlExecutionProvider", "CUDAExecutionProvider"]
    requested.append("CPUExecutionProvider")

    available = set(ort.get_available_providers())
    providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3  # silence provider chatter (see commit 64c3090)
    return ort.InferenceSession(path, sess_options=options, providers=providers)


def _get_session(kind: str):
    """Lazily create and cache one session per model kind."""
    with _SESSION_LOCK:
        session = _SESSIONS.get(kind)
        if session is not None:
            return session
    if not ensure_model_downloaded(kind):
        return None
    try:
        session = _make_session(model_path(kind))
    except Exception:
        logger.warning("[AIMASK] Failed to load %s", kind, exc_info=True)
        return None
    with _SESSION_LOCK:
        _SESSIONS[kind] = session
    return session


def release_sessions() -> None:
    """Drop cached sessions (frees model memory when leaving the Masks tool)."""
    with _SESSION_LOCK:
        _SESSIONS.clear()


def _graph_input_layout(session, fallback: int) -> tuple:
    """(layout, size) for a model's image input, read off the ONNX graph.

    Exports differ in both respects, so this is detected rather than
    configured -- swapping in a different export should not need a code
    change:

    * ``"nchw"`` -- ``[1, 3, H, W]``. We resize and normalize (BiRefNet,
      U^2-Net skyseg). A static H is preferred over our configured default.
    * ``"hwc"``  -- ``[h, w, 3]``, dynamic, no batch axis. The graph does
      its own resize/normalize/pad, so it wants a plain 0-255 image
      (the AnyLabeling-style MobileSAM encoder export).

    Reading shape[2] unconditionally would read the HWC channel count (3)
    as the spatial size.
    """
    try:
        shape = session.get_inputs()[0].shape
        if len(shape) == 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                return "nchw", int(shape[2])
            return "nchw", fallback
        if len(shape) == 3:
            return "hwc", fallback
    except Exception:
        pass
    return "nchw", fallback


def _graph_input_size(session, fallback: int) -> int:
    return _graph_input_layout(session, fallback)[1]


# ----------------------------------------------------------------------
# Pre / post processing
# ----------------------------------------------------------------------


def _to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    """Accept uint8 or float RGB; return float32 (H, W, 3) in [0, 1]."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    else:
        out = arr.astype(np.float32, copy=False)
        # Edit-base buffers are scene-linear-ish and can exceed 1.0 at
        # highlights; segmentation nets expect display-referred input, so
        # clamp rather than let specular pixels dominate the normalization.
        out = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(out)


def _preprocess(rgb: np.ndarray, size: int, mean, std) -> np.ndarray:
    """Resize to (size, size), normalize, return NCHW float32."""
    import cv2

    img = _to_float_rgb(rgb)
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    normalized = (resized - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return np.ascontiguousarray(
        normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _to_alpha(raw: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Squeeze a model output to (H, W) float32 [0, 1] at the target size.

    Some exports emit probabilities, others logits, and multi-stage nets
    like U^2-Net emit a list whose first entry is the fused output. Detect
    rather than assume: anything already inside [0, 1] is taken as-is, and
    anything outside gets a sigmoid.
    """
    import cv2

    arr = np.asarray(raw, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0] if arr.shape[0] == 1 else arr[0:1][0]
    if arr.ndim != 2:
        raise ValueError(f"unexpected mask output shape {np.asarray(raw).shape}")

    lo, hi = float(arr.min()), float(arr.max())
    if lo < 0.0 or hi > 1.0:
        arr = _sigmoid(arr)
    elif hi > 0.0 and hi <= 1.0 and lo >= 0.0:
        # Already probabilities. Some U^2-Net exports emit a min-max
        # normalized map; leave it alone either way.
        pass

    th, tw = int(target_hw[0]), int(target_hw[1])
    if arr.shape != (th, tw):
        arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------
# One-click matte models (BiRefNet / sky)
# ----------------------------------------------------------------------


def _run_matte_model(kind: str, rgb: np.ndarray, target_hw: tuple) -> Optional[np.ndarray]:
    session = _get_session(kind)
    if session is None:
        return None
    spec = _MODELS[kind]
    size = _graph_input_size(session, int(spec["input_size"]))
    try:
        tensor = _preprocess(rgb, size, spec["mean"], spec["std"])
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
    except Exception:
        logger.warning("[AIMASK] %s inference failed", kind, exc_info=True)
        return None
    if not outputs:
        return None
    # Multi-stage nets (U^2-Net) list the fused/highest-quality map first.
    return _to_alpha(outputs[0], target_hw)


def segment_subject(rgb: np.ndarray, target_hw: tuple) -> Optional[np.ndarray]:
    """One-click foreground matte (BiRefNet). None if unavailable."""
    return _run_matte_model("subject", rgb, target_hw)


def segment_sky(rgb: np.ndarray, target_hw: tuple) -> Optional[np.ndarray]:
    """One-click sky matte. None if unavailable."""
    return _run_matte_model("sky", rgb, target_hw)


# ----------------------------------------------------------------------
# MobileSAM click-to-mask
# ----------------------------------------------------------------------


def _sam_resize_longest(rgb: np.ndarray, size: int) -> tuple:
    """SAM's preprocessing: scale longest side to ``size``, pad to square.

    Returns (padded_float_rgb, scale, (new_h, new_w)). The scale and
    unpadded extent are needed later to map click coordinates into the
    model's frame and to crop the decoder's square output back.
    """
    import cv2

    img = _to_float_rgb(rgb)
    h, w = img.shape[:2]
    scale = float(size) / float(max(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros((size, size, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized
    return padded, scale, (new_h, new_w)


class SamPredictor:
    """Point-prompted segmentation with a cached per-image embedding.

    ``set_image`` runs the expensive encoder once; ``predict`` runs only
    the small decoder, so each click costs milliseconds. Reusing an
    instance across clicks on the same photo is the entire point -- create
    one when the Click tool is armed and drop it when the image changes.
    """

    def __init__(self) -> None:
        self._embedding: Optional[np.ndarray] = None
        self._image_key = None
        self._scale = 1.0
        self._unpadded_hw = (0, 0)
        self._source_hw = (0, 0)
        self._encoder_size = 1024

    def set_image(self, rgb: np.ndarray, *, image_key=None) -> bool:
        """Encode ``rgb``. No-op (True) if ``image_key`` matches the cache."""
        key = image_key if image_key is not None else id(rgb)
        if self._embedding is not None and key == self._image_key:
            return True

        session = _get_session("sam_encoder")
        if session is None:
            return False
        spec = _MODELS["sam_encoder"]
        layout, size = _graph_input_layout(session, int(spec["input_size"]))
        try:
            if layout == "hwc":
                # The graph resizes/normalizes/pads internally and wants a
                # plain 0-255 image. We still downscale to the longest side
                # ourselves: it is the same result far cheaper than pushing
                # a 40MP frame through the graph, and it makes ``scale``
                # (needed to map clicks) explicit rather than implied.
                import cv2

                src = _to_float_rgb(rgb) * 255.0
                h, w = src.shape[:2]
                scale = float(size) / float(max(h, w))
                nh, nw = int(round(h * scale)), int(round(w * scale))
                resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
                unpadded = (nh, nw)
                feed = np.ascontiguousarray(resized.astype(np.float32))
            else:
                padded, scale, unpadded = _sam_resize_longest(rgb, size)
                normalized = (
                    padded - np.asarray(spec["mean"], dtype=np.float32)
                ) / np.asarray(spec["std"], dtype=np.float32)
                feed = np.ascontiguousarray(
                    normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
                )
            outputs = session.run(None, {session.get_inputs()[0].name: feed})
        except Exception:
            logger.warning("[AIMASK] SAM encoder failed", exc_info=True)
            return False
        if not outputs:
            return False

        self._embedding = np.asarray(outputs[0], dtype=np.float32)
        self._image_key = key
        self._scale = scale
        self._unpadded_hw = unpadded
        self._encoder_size = size
        self._source_hw = (int(np.asarray(rgb).shape[0]), int(np.asarray(rgb).shape[1]))
        return True

    def is_ready(self) -> bool:
        return self._embedding is not None

    def reset(self) -> None:
        self._embedding = None
        self._image_key = None

    def predict(
        self,
        points: Sequence[Sequence[float]],
        labels: Sequence[int],
        target_hw: tuple,
    ) -> Optional[np.ndarray]:
        """Decode a mask from click points.

        ``points`` are (x, y) in the *source* image coordinate space that
        was passed to set_image; ``labels`` are 1 for include, 0 for
        exclude. Returns float32 (H, W) alpha at ``target_hw``.
        """
        if self._embedding is None or not len(points):
            return None
        session = _get_session("sam_decoder")
        if session is None:
            return None

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) * self._scale
        lbls = np.asarray(labels, dtype=np.float32).reshape(-1)
        # SAM's exported decoder expects a trailing padding point when no
        # box prompt is supplied (coord 0,0 with label -1).
        pts = np.concatenate([pts, np.zeros((1, 2), np.float32)], axis=0)
        lbls = np.concatenate([lbls, np.array([-1.0], np.float32)], axis=0)

        src_h, src_w = self._source_hw
        feeds = {
            "image_embeddings": self._embedding,
            "point_coords": pts[np.newaxis, ...].astype(np.float32),
            "point_labels": lbls[np.newaxis, ...].astype(np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array([src_h, src_w], dtype=np.float32),
        }
        # Decoder exports differ in which of these they declare (some omit
        # orig_im_size and return a fixed-size low-res mask). Feed only
        # what this graph actually asks for.
        declared = {i.name for i in session.get_inputs()}
        feeds = {k: v for k, v in feeds.items() if k in declared}
        missing = declared - set(feeds)
        if missing:
            logger.warning("[AIMASK] SAM decoder wants unknown inputs: %s", sorted(missing))
            return None

        try:
            outputs = session.run(None, feeds)
        except Exception:
            logger.warning("[AIMASK] SAM decoder failed", exc_info=True)
            return None
        if not outputs:
            return None

        masks = np.asarray(outputs[0], dtype=np.float32)
        # When multiple candidate masks come back, take the highest-IoU one
        # rather than index 0 -- SAM's first slot is not the best slot.
        if masks.ndim == 4 and masks.shape[1] > 1:
            iou = None
            for extra in outputs[1:]:
                cand = np.asarray(extra, dtype=np.float32).reshape(-1)
                if cand.size == masks.shape[1]:
                    iou = cand
                    break
            best = int(np.argmax(iou)) if iou is not None else 0
            masks = masks[:, best : best + 1]

        alpha = _to_alpha(masks, target_hw)
        if "orig_im_size" not in feeds:
            # Fixed-size output covers the padded square, so crop the
            # letterbox away before it gets stretched over the frame.
            alpha = self._crop_padding(alpha, target_hw)
        return alpha

    def _crop_padding(self, alpha: np.ndarray, target_hw: tuple) -> np.ndarray:
        import cv2

        size = float(self._encoder_size)
        new_h, new_w = self._unpadded_hw
        if not new_h or not new_w:
            return alpha
        h, w = alpha.shape[:2]
        cy = max(1, int(round(h * (new_h / size))))
        cx = max(1, int(round(w * (new_w / size))))
        cropped = alpha[:cy, :cx]
        th, tw = int(target_hw[0]), int(target_hw[1])
        return np.clip(
            cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_LINEAR), 0.0, 1.0
        ).astype(np.float32)


# ----------------------------------------------------------------------
# Mask post-processing shared by all three
# ----------------------------------------------------------------------


def feather_alpha(alpha: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian-soften a mask edge. radius <= 0 returns the input."""
    if radius <= 0:
        return alpha
    import cv2

    k = int(radius) * 2 + 1
    return np.clip(
        cv2.GaussianBlur(alpha, (k, k), radius), 0.0, 1.0
    ).astype(np.float32)


def binarize_alpha(alpha: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Hard-threshold a matte. Useful for SAM output, which is already crisp."""
    return (alpha >= threshold).astype(np.float32)
