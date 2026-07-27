"""Local instruction-based editing: InstructPix2Pix on ONNX Runtime.

The provider layer's other implementation uploads the photograph to a server
the user nominates. This one does not: the model runs on the user's own
machine, so ``requires_consent`` is False and nothing leaves the computer.
That is the whole point of it.

Why InstructPix2Pix and why ONNX
--------------------------------
The app's entire ML stack is ONNX Runtime plus numpy -- the AI masks, the
denoiser and the upscaler all go through it, with CoreML / DirectML / CUDA
selected per platform. A modern editing model (FLUX Kontext, Qwen-Image-Edit)
would mean torch, diffusers and 12-24 GB of weights, which is a different app.
InstructPix2Pix is ~2.2 GB across four graphs, runs on the accelerators
already wired up, and needs no new dependency at all.

The cost is honesty about quality. This is a 2022-era model: it is good at
global and stylistic instructions ("make it winter", "give it a sunset
mood") and weak at precise object removal, which is exactly the instruction
a photographer reaches for first. The UI says so rather than letting the
user discover it one 30-second generation at a time.

Sampling
--------
InstructPix2Pix conditions on both text and image, so it uses three-way
classifier-free guidance and the UNet takes 8 input channels (4 noise latent
+ 4 image latent) rather than the usual 4. Both are handled in _denoise; the
three-branch batch is [text+image, uncond+image, uncond+no-image] and the
guidance combination follows the reference pipeline exactly.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Callable, Optional

import numpy as np

from raw_generative_edit import (
    CancelToken,
    GenerativeEditError,
    GenerativeProvider,
    GenerativeRequest,
    GenerativeResult,
    build_provenance,
)

logger = logging.getLogger(__name__)

MODEL_KIND = "instruct_pix2pix"
MODEL_NAME = "instruct-pix2pix-onnx"

# The four graphs plus the tokenizer and schedule that must match them.
# Published as a diffusers ONNX export; see docs for the licence note.
_BASE_URL = "https://huggingface.co/ForserX/instruct-pix2pix-onnx/resolve/main"

#
# PROVENANCE, read this before changing the URL
# --------------------------------------------
# These are a community ONNX conversion of timbrooks/instruct-pix2pix (31k
# downloads, the canonical publication). The conversion repo itself is a
# one-person upload from March 2023 with TWO downloads and one like, made for
# an unrelated GUI project. That is a weak trust signal, so the bytes are
# pinned below by SHA-256: the hashes are of the exact files that were
# downloaded, loaded and verified end-to-end here. A re-upload, a substituted
# graph or a truncated download now fails loudly instead of running.
#
# ONNX is protobuf, not pickle, so loading one does not execute code the way
# a .ckpt can -- the exposure is a model that behaves differently from the
# one claimed, not arbitrary code execution. The hashes close that gap for
# the files we have actually exercised.
#
# What is NOT verified: that this conversion faithfully matches timbrooks'
# weights. Confirming that needs torch to re-export and compare, which is the
# dependency this provider exists to avoid.
#
_FILES = {
    "text_encoder": (
        "text_encoder/model.onnx",
        "text_encoder_model.onnx",
        "8f34e5ee561cba4d0624a1845f7e06d7fda6c8ca27741c1e75f291ae6991bbc7",
    ),
    "unet": (
        "unet/model.onnx",
        "unet_model.onnx",
        "e756e0716b999c89c32312ea43eb5ba4458e89e9c61aa4cb5ff724a2a7515a6e",
    ),
    "vae_encoder": (
        "vae_encoder/model.onnx",
        "vae_encoder_model.onnx",
        "2a8dd0b1a179446d5902fc35a8c8cac082820cb0530422134f9143fd08f8040a",
    ),
    "vae_decoder": (
        "vae_decoder/model.onnx",
        "vae_decoder_model.onnx",
        "9eb64b3179d4301df2932423797053917e1e32a0e2292521bf6a537377f51d9b",
    ),
    "vocab": (
        "tokenizer/vocab.json",
        "tokenizer_vocab.json",
        "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    ),
    "merges": (
        "tokenizer/merges.txt",
        "tokenizer_merges.txt",
        "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    ),
    "scheduler": (
        "scheduler/scheduler_config.json",
        "scheduler_scheduler_config.json",
        "e199bc7f03bed345f780139e988dcf4b2ecfe4a3d1de1df9f3b21ce654a26e22",
    ),
}

# Latents are 1/8 the image size, and the model was trained at 512. Larger
# inputs are tiled by resizing rather than by patching: a patched diffusion
# edit produces visible seams because each patch is denoised independently.
_TRAIN_SIZE = 512
_VAE_SCALE = 0.18215
_LATENT_DOWNSCALE = 8

_SESSION_LOCK = threading.Lock()
_SESSIONS: dict = {}


def models_dir() -> str:
    from raw_ai_masks import _models_dir

    return os.path.join(_models_dir(), "ip2p")


def file_path(key: str) -> str:
    entry = _FILES.get(key)
    if entry is None:
        raise KeyError(key)
    return os.path.join(models_dir(), entry[1])


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_downloaded(strict: bool = True) -> list:
    """Names of files whose contents do not match the pinned hash.

    Empty list means every present file is byte-identical to what was tested.
    Missing files are reported too when ``strict``.
    """
    bad = []
    for key, entry in _FILES.items():
        path = os.path.join(models_dir(), entry[1])
        if not os.path.isfile(path):
            if strict:
                bad.append(key)
            continue
        if _sha256_of_file(path) != entry[2]:
            bad.append(key)
    return bad


def is_downloaded() -> bool:
    """Whether every file the provider needs is already on disk."""
    return all(os.path.isfile(file_path(k)) for k in _FILES)


def total_download_bytes() -> int:
    """Rough size, for telling the user what they are agreeing to."""
    return 2_310_000_000


def download(
    progress: Optional[Callable[[str], None]] = None,
    cancel: Optional[CancelToken] = None,
) -> bool:
    """Fetch the model set. Returns True when everything is present.

    Every file is checked against its pinned SHA-256 and a mismatch is
    discarded rather than kept: the source repo is a low-traffic third-party
    conversion (see the provenance note above), so "the bytes we tested" is a
    much better guarantee than "whatever that URL serves today".
    """
    import urllib.request

    os.makedirs(models_dir(), exist_ok=True)
    for key, (remote, local, expected_sha) in _FILES.items():
        dest = os.path.join(models_dir(), local)
        if os.path.isfile(dest):
            continue
        if cancel is not None:
            cancel.raise_if_cancelled()
        if progress:
            progress(f"Downloading {key}…")
        url = f"{_BASE_URL}/{remote}"
        partial = dest + ".part"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                total = int(response.headers.get("content-length") or 0)
                done = 0
                with open(partial, "wb") as fh:
                    while True:
                        if cancel is not None:
                            cancel.raise_if_cancelled()
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if progress and total:
                            pct = int(done * 100 / total)
                            if pct % 5 == 0:
                                progress(f"Downloading {key}… {pct}%")
            actual = _sha256_of_file(partial)
            if actual != expected_sha:
                os.remove(partial)
                logger.error(
                    "[IP2P] %s failed verification (expected %s, got %s)",
                    key, expected_sha[:12], actual[:12],
                )
                if progress:
                    progress(f"{key} failed verification — download rejected.")
                return False
            os.replace(partial, dest)
        except Exception:
            try:
                os.remove(partial)
            except OSError:
                pass
            logger.warning("[IP2P] download failed for %s", key, exc_info=True)
            return False
    return is_downloaded()


def _make_session(path: str):
    """ORT session with the platform's accelerator preferred.

    Mirrors onnx_realesrgan's ordering rather than onnx_scunet's: the UNet is
    run 20+ times per edit, so a provider that falls back to CPU silently is
    the difference between 30 seconds and several minutes.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = [
        p
        for p in (
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        )
        if p in available
    ]
    # ORT_ENABLE_ALL crashes on this export: SimplifiedLayerNormFusion in
    # ORT 1.27 rewrites the text encoder's layer norms into a graph it then
    # cannot resolve ("Attempting to get index by a name which does not
    # exist"). EXTENDED keeps every optimisation that matters and loads
    # cleanly, so the ladder starts there and degrades rather than failing.
    levels = [
        ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    ]
    last_error: Optional[Exception] = None
    for level in levels:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = level
        opts.log_severity_level = 3
        try:
            return ort.InferenceSession(path, sess_options=opts, providers=preferred)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[IP2P] %s failed to load at %s, trying a lower level",
                os.path.basename(path),
                level,
            )
    raise GenerativeEditError(
        f"Could not load {os.path.basename(path)}: {last_error}"
    )


def _session(key: str):
    with _SESSION_LOCK:
        hit = _SESSIONS.get(key)
        if hit is not None:
            return hit
    session = _make_session(file_path(key))
    with _SESSION_LOCK:
        _SESSIONS[key] = session
    return session


def release_sessions() -> None:
    """Drop the graphs; ~2 GB of resident memory."""
    with _SESSION_LOCK:
        _SESSIONS.clear()


def _input_names(session) -> list:
    return [i.name for i in session.get_inputs()]


def _fit_to_model(rgb: np.ndarray, max_edge: int = _TRAIN_SIZE) -> tuple:
    """Resize so both sides are multiples of 8 and the long edge is sane.

    Returns (resized, original_hw). The model was trained at 512; running far
    above that degrades coherence rather than adding detail, so the edit
    happens at model scale and is resized back afterwards.
    """
    import cv2

    h, w = rgb.shape[:2]
    scale = float(max_edge) / float(max(h, w))
    if scale < 1.0:
        nh, nw = int(round(h * scale)), int(round(w * scale))
    else:
        nh, nw = h, w
    nh = max(_LATENT_DOWNSCALE, (nh // _LATENT_DOWNSCALE) * _LATENT_DOWNSCALE)
    nw = max(_LATENT_DOWNSCALE, (nw // _LATENT_DOWNSCALE) * _LATENT_DOWNSCALE)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, (h, w)


class LocalInstructPix2PixProvider(GenerativeProvider):
    """InstructPix2Pix on ONNX Runtime, entirely on this machine."""

    name = "local-ip2p"
    model_name = MODEL_NAME
    # Nothing leaves the computer, so there is nothing to consent to.
    requires_consent = False

    def __init__(
        self,
        steps: int = 10,
        text_guidance: float = 7.5,
        image_guidance: float = 1.5,
        seed: Optional[int] = None,
        max_edge: int = _TRAIN_SIZE,
    ):
        # 10 steps, not the reference 20. Measured on an M-series Mac through
        # the CoreML EP, one UNet evaluation of the three-branch batch costs
        # ~10.7 s at 384x512 and ~18.6 s at 512x512 -- so 20 steps is 3.6-6.2
        # minutes. That is a long time to wait to find out an instruction did
        # not land. 10 steps roughly halves it at a modest quality cost, and
        # both knobs are settings for anyone who wants to trade the other way.
        self.steps = max(1, int(steps))
        self.text_guidance = float(text_guidance)
        self.image_guidance = float(image_guidance)
        self.seed = seed
        self.max_edge = max(64, int(max_edge))

    def is_configured(self) -> bool:
        return is_downloaded()

    def describe(self) -> str:
        if not is_downloaded():
            gb = total_download_bytes() / 1e9
            return f"Local model not downloaded yet (~{gb:.1f} GB)"
        return "InstructPix2Pix, running on this machine"

    # -- pipeline stages ----------------------------------------------

    def _embed(self, tokenizer, text: str) -> np.ndarray:
        session = _session("text_encoder")
        ids = np.array([tokenizer.tokenize(text)], dtype=np.int32)
        name = _input_names(session)[0]
        out = session.run(None, {name: ids})[0]
        return np.asarray(out, dtype=np.float32)

    def _encode_image(self, rgb01: np.ndarray) -> np.ndarray:
        """VAE-encode to the image-conditioning latent.

        Note the absence of the 0.18215 scaling that the noise latent gets:
        InstructPix2Pix conditions on the *unscaled* image latent, and
        scaling it here washes the conditioning out.
        """
        session = _session("vae_encoder")
        # [-1, 1], NCHW
        x = (rgb01.astype(np.float32) * 2.0 - 1.0).transpose(2, 0, 1)[None, ...]
        name = _input_names(session)[0]
        out = session.run(None, {name: x.astype(np.float32)})[0]
        out = np.asarray(out, dtype=np.float32)
        if out.shape[1] == 8:
            # Some exports return the full distribution; take the mean.
            out = out[:, :4]
        return out

    def _decode_latents(self, latents: np.ndarray) -> np.ndarray:
        session = _session("vae_decoder")
        name = _input_names(session)[0]
        out = session.run(None, {name: (latents / _VAE_SCALE).astype(np.float32)})[0]
        img = np.asarray(out, dtype=np.float32)[0].transpose(1, 2, 0)
        return np.clip((img + 1.0) / 2.0, 0.0, 1.0)

    def _denoise(
        self,
        image_latents: np.ndarray,
        cond: np.ndarray,
        uncond: np.ndarray,
        cancel: Optional[CancelToken],
        progress: Optional[Callable[[str], None]],
    ) -> np.ndarray:
        from onnx_ip2p_scheduler import EulerAncestralScheduler

        scheduler = EulerAncestralScheduler.from_config_file(file_path("scheduler"))
        scheduler.set_timesteps(self.steps)

        rng = np.random.default_rng(self.seed)
        shape = (1, 4, image_latents.shape[2], image_latents.shape[3])
        latents = rng.standard_normal(size=shape).astype(np.float32)
        latents *= scheduler.init_noise_sigma

        # Three branches: text+image, uncond+image, uncond+blank-image.
        embeddings = np.concatenate([cond, uncond, uncond], axis=0)
        blank = np.zeros_like(image_latents)
        image_cond = np.concatenate([image_latents, image_latents, blank], axis=0)

        session = _session("unet")
        names = _input_names(session)
        sample_name, timestep_name, hidden_name = names[0], names[1], names[2]

        for i, t in enumerate(scheduler.timesteps):
            if cancel is not None:
                cancel.raise_if_cancelled()
            if progress:
                progress(f"Generating… step {i + 1}/{len(scheduler.timesteps)}")

            scaled = scheduler.scale_model_input(latents, i)
            batch = np.concatenate([scaled] * 3, axis=0)
            # 8-channel input: noise latent alongside image latent.
            model_input = np.concatenate([batch, image_cond], axis=1)

            noise = session.run(
                None,
                {
                    sample_name: model_input.astype(np.float32),
                    timestep_name: np.array([t], dtype=np.float32),
                    hidden_name: embeddings.astype(np.float32),
                },
            )[0]
            noise = np.asarray(noise, dtype=np.float32)
            n_text, n_image, n_uncond = noise[0:1], noise[1:2], noise[2:3]
            guided = (
                n_uncond
                + self.text_guidance * (n_text - n_image)
                + self.image_guidance * (n_image - n_uncond)
            )
            latents = scheduler.step(guided, i, latents, generator=rng)
        return latents

    # -- provider API -------------------------------------------------

    def edit(
        self,
        request: GenerativeRequest,
        cancel: Optional[CancelToken] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> GenerativeResult:
        if not is_downloaded():
            raise GenerativeEditError(
                "The local model is not downloaded yet — press Setup…"
            )
        instruction = (request.instruction or "").strip()
        if not instruction:
            raise GenerativeEditError("Describe the edit you want first.")

        from clip_bpe_tokenizer import load_tokenizer

        tokenizer = load_tokenizer(file_path("vocab"), file_path("merges"))
        if tokenizer is None:
            raise GenerativeEditError("Could not load the tokenizer files.")

        if progress:
            progress("Preparing…")
        rgb = np.asarray(request.image)
        if rgb.dtype == np.uint8:
            rgb01 = rgb.astype(np.float32) / 255.0
        else:
            rgb01 = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        if rgb01.ndim != 3 or rgb01.shape[2] < 3:
            raise GenerativeEditError("Expected an RGB image.")
        rgb01 = rgb01[:, :, :3]
        small, (orig_h, orig_w) = _fit_to_model(rgb01, self.max_edge)

        if cancel is not None:
            cancel.raise_if_cancelled()
        if progress:
            progress("Reading the instruction…")
        cond = self._embed(tokenizer, instruction)
        uncond = self._embed(tokenizer, "")

        if progress:
            progress("Encoding the image…")
        image_latents = self._encode_image(small)

        latents = self._denoise(image_latents, cond, uncond, cancel, progress)

        if progress:
            progress("Decoding…")
        out01 = self._decode_latents(latents)

        # Back to the source resolution. The edit happened at model scale, so
        # this is an upscale; the user's original pixel count is preserved
        # rather than silently reduced to 512.
        if (out01.shape[0], out01.shape[1]) != (orig_h, orig_w):
            import cv2

            out01 = cv2.resize(
                out01, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
            )
        out = np.clip(out01 * 255.0 + 0.5, 0, 255).astype(np.uint8)

        return GenerativeResult(
            image=out,
            provenance=build_provenance(
                request,
                self.name,
                self.model_name,
                parent_provenance=(request.options or {}).get("parent_provenance"),
            ),
        )
