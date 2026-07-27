"""
Instruction-based generative editing -- provider layer.

This is the deliberately-outside-the-pipeline half of the editor. Every
other edit in RAWviewer is a number in a dict that gets replayed against
the RAW; a generative edit is not, and pretending otherwise would break
the non-destructive guarantee. So the contract here is narrow and
explicit:

    baked RGB pixels + an instruction  ->  different RGB pixels

Nothing in this module touches the adjustment dict, the mask stack, or a
sidecar. The caller bakes the current parametric render, sends it here,
and writes whatever comes back as a NEW derived file that then gets its
own ordinary parametric stack (see generative_derived_file.py). The RAW
is never modified and the original edit stack is never lost.

Provider strategy (see also the platform notes in build.py):
  - ``HttpEndpointProvider`` is the default and the only one that works
    on macOS, which cannot run any current editing model locally. The
    user supplies the URL; we ship no weights and take no license or
    bundle-size risk.
  - A local provider is a later addition for Windows+NVIDIA, where
    Mage-Flow-Edit-Turbo (MIT, 4B) is the smallest credible option.

PRIVACY: a remote provider uploads the user's photograph to a third
party. That is a materially different act from every other operation in
this app, and the module reflects it -- there is no default endpoint, no
implicit fallback to a hosted service, and ``requires_consent`` is True
for anything off-machine. Callers must gate on it. Photographers shoot
under NDA and model release; silently uploading their work would be a
serious breach, not a UX wrinkle.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0

# Bump when the provenance record's shape changes.
PROVENANCE_VERSION = 1

# Adjustment-dict key carrying the provenance JSON. Persisted as the
# crs:RVGenerativeProvenance child element, same pattern as RVMaskLayers.
# Present in the dict => the file is AI-generated, which also forces
# is_default_adjustments to False so the sidecar is never cleared away.
PROVENANCE_KEY = "_generative_provenance"


class GenerativeEditError(Exception):
    """Provider failed. Message is safe to show the user."""


class CancelledError(GenerativeEditError):
    """The user cancelled before the provider returned."""


class CancelToken:
    """Cooperative cancellation for a multi-second remote call.

    Threading-only by design: providers poll ``cancelled`` at their own
    checkpoints rather than being killed, so a half-written response is
    never handed back as a result.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("Generative edit cancelled.")


@dataclass
class GenerativeRequest:
    """One edit. ``image`` is display-referred RGB, uint8 or float [0, 1]."""

    image: np.ndarray
    instruction: str
    seed: Optional[int] = None
    source_path: str = ""
    # Free-form provider knobs (steps, guidance, ...). Kept opaque so a
    # provider can expose its own without changing this dataclass.
    options: dict = field(default_factory=dict)


@dataclass
class GenerativeResult:
    image: np.ndarray  # uint8 RGB
    provenance: dict


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Bake to 8-bit sRGB-ish, the only thing these models accept.

    This is a real loss of latitude and the UI must say so -- a scene-
    linear edit base carries highlight headroom that does not survive the
    trip. Clipping here rather than at the far end at least makes it
    happen once, visibly, on our side.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise GenerativeEditError("Expected an RGB image.")
    arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(
        np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5
    ).astype(np.uint8)


def _encode_png_b64(image: np.ndarray) -> str:
    import cv2

    rgb = _to_uint8_rgb(image)
    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])  # cv2 wants BGR
    if not ok:
        raise GenerativeEditError("Could not encode the image to send.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_png_b64(serial: str) -> np.ndarray:
    import cv2

    try:
        raw = base64.b64decode(serial.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        raise GenerativeEditError("Provider returned malformed image data.") from exc
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise GenerativeEditError("Provider returned an image that could not be decoded.")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def build_provenance(
    request: GenerativeRequest,
    provider_name: str,
    model_name: str,
    *,
    parent_provenance: Optional[dict] = None,
) -> dict:
    """Record what generated this image, for the derived file's sidecar.

    Written so a file can be identified as AI-generated a year later, by
    someone who was not in the room. ``parent_provenance`` chains: editing
    an already-generated file appends rather than overwrites, so the whole
    lineage back to the RAW stays legible.
    """
    entry = {
        "version": PROVENANCE_VERSION,
        "provider": provider_name,
        "model": model_name,
        "instruction": request.instruction,
        "seed": request.seed,
        "source": request.source_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "generated": True,
    }
    chain = []
    if parent_provenance:
        chain = list(parent_provenance.get("chain") or [])
        parent_entry = dict(parent_provenance)
        parent_entry.pop("chain", None)
        chain.append(parent_entry)
    entry["chain"] = chain
    return entry


class GenerativeProvider:
    """Base class. Subclasses implement ``edit``."""

    name = "base"
    model_name = "unknown"
    requires_consent = True  # True whenever pixels leave the machine

    def is_configured(self) -> bool:
        return False

    def describe(self) -> str:
        """One line for the UI: where is this going?"""
        return self.name

    def edit(
        self,
        request: GenerativeRequest,
        cancel: Optional[CancelToken] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> GenerativeResult:
        raise NotImplementedError


class HttpEndpointProvider(GenerativeProvider):
    """POST the image to a user-configured HTTP endpoint.

    Wire format, kept deliberately trivial so a user can put a ~30-line
    shim in front of ComfyUI, a diffusers script, or a hosted API:

        POST <endpoint>
        Content-Type: application/json
        {"instruction": str, "image": "<base64 PNG>",
         "seed": int|null, "options": {...}}

        200 {"image": "<base64 PNG>", "model": "<optional name>"}
        4xx/5xx {"error": "<message shown to the user>"}

    There is no default endpoint and never should be -- an empty setting
    means the feature is off, not that it silently falls back to someone
    else's server.
    """

    name = "http"
    requires_consent = True

    def __init__(
        self,
        endpoint: str = "",
        *,
        api_key: str = "",
        model_name: str = "remote",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        allow_insecure: bool = False,
    ) -> None:
        self.endpoint = (endpoint or "").strip()
        self.api_key = api_key or ""
        self.model_name = model_name
        self.timeout_s = float(timeout_s)
        # Plain http leaks the photo to anything on the path. Allowed only
        # when the user opts in, and always allowed for loopback (running
        # a model on your own machine is the privacy-preserving case).
        self.allow_insecure = bool(allow_insecure)

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def describe(self) -> str:
        return f"Remote endpoint: {self.endpoint}" if self.endpoint else "No endpoint configured"

    def _validate_endpoint(self) -> None:
        from urllib.parse import urlparse

        if not self.endpoint:
            raise GenerativeEditError("No generative endpoint is configured.")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https"):
            raise GenerativeEditError("Endpoint must be an http:// or https:// URL.")
        host = (parsed.hostname or "").lower()
        is_loopback = host in ("localhost", "127.0.0.1", "::1")
        if parsed.scheme == "http" and not (is_loopback or self.allow_insecure):
            raise GenerativeEditError(
                "Refusing to send photos over plain http. Use https, or enable "
                "the insecure-endpoint option if this is a trusted local network."
            )

    def edit(
        self,
        request: GenerativeRequest,
        cancel: Optional[CancelToken] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> GenerativeResult:
        import urllib.error
        import urllib.request

        self._validate_endpoint()
        if not (request.instruction or "").strip():
            raise GenerativeEditError("Describe the edit you want before generating.")
        if cancel is not None:
            cancel.raise_if_cancelled()

        if progress:
            progress("Preparing image...")
        payload = json.dumps(
            {
                "instruction": request.instruction,
                "image": _encode_png_b64(request.image),
                "seed": request.seed,
                "options": dict(request.options or {}),
            }
        ).encode("utf-8")

        if cancel is not None:
            cancel.raise_if_cancelled()
        if progress:
            progress("Sending to endpoint...")

        req = urllib.request.Request(self.endpoint, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise GenerativeEditError(self._http_error_message(exc)) from exc
        except urllib.error.URLError as exc:
            raise GenerativeEditError(
                f"Could not reach the endpoint: {getattr(exc, 'reason', exc)}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise GenerativeEditError(f"Request failed: {exc}") from exc

        # Cancellation cannot interrupt urlopen, so a cancel that lands
        # mid-flight is honoured here: the response is discarded rather
        # than turned into a file the user no longer wants.
        if cancel is not None:
            cancel.raise_if_cancelled()
        if progress:
            progress("Decoding result...")

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise GenerativeEditError("Endpoint returned a non-JSON response.") from exc
        if not isinstance(data, dict):
            raise GenerativeEditError("Endpoint returned an unexpected response.")
        if data.get("error"):
            raise GenerativeEditError(str(data["error"]))
        if not data.get("image"):
            raise GenerativeEditError("Endpoint returned no image.")

        image = _decode_png_b64(str(data["image"]))
        model_name = str(data.get("model") or self.model_name)
        return GenerativeResult(
            image=image,
            provenance=build_provenance(
                request,
                self.name,
                model_name,
                parent_provenance=request.options.get("parent_provenance"),
            ),
        )

    @staticmethod
    def _http_error_message(exc) -> str:
        """Prefer the endpoint's own error text over a bare status code."""
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            if isinstance(detail, dict) and detail.get("error"):
                return str(detail["error"])
        except Exception:
            pass
        return f"Endpoint returned HTTP {getattr(exc, 'code', '?')}."


class StubProvider(GenerativeProvider):
    """Deterministic local provider for tests and UI development.

    Applies a visible, reversible transform (channel roll) so the whole
    round-trip -- bake, send, receive, write a derived file, stack it --
    can be exercised end-to-end with no model, no GPU and no network.
    """

    name = "stub"
    model_name = "stub-v1"
    requires_consent = False

    def __init__(self, delay_s: float = 0.0, fail_with: str = "") -> None:
        self.delay_s = float(delay_s)
        self.fail_with = fail_with

    def is_configured(self) -> bool:
        return True

    def describe(self) -> str:
        return "Test stub (no model, stays on this machine)"

    def edit(
        self,
        request: GenerativeRequest,
        cancel: Optional[CancelToken] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> GenerativeResult:
        if self.fail_with:
            raise GenerativeEditError(self.fail_with)
        if progress:
            progress("Generating...")
        # Poll rather than sleep through, so cancellation is testable.
        deadline = time.monotonic() + self.delay_s
        while time.monotonic() < deadline:
            if cancel is not None:
                cancel.raise_if_cancelled()
            time.sleep(0.005)
        if cancel is not None:
            cancel.raise_if_cancelled()

        rgb = _to_uint8_rgb(request.image)
        out = np.roll(rgb, 1, axis=2)
        return GenerativeResult(
            image=out,
            provenance=build_provenance(
                request,
                self.name,
                self.model_name,
                parent_provenance=(request.options or {}).get("parent_provenance"),
            ),
        )


def make_provider(settings: Optional[dict] = None) -> GenerativeProvider:
    """Build the configured provider. Never raises; may be unconfigured.

    Callers should check ``is_configured()`` and ``requires_consent``
    before showing the feature as available.
    """
    settings = settings or {}
    kind = str(settings.get("provider", "http") or "http").strip().lower()
    if kind == "stub":
        return StubProvider()
    if kind == "local":
        # Imported lazily: it pulls in onnxruntime and the scheduler, which a
        # user who never touches generative editing should not pay for.
        from raw_generative_local import LocalInstructPix2PixProvider

        return LocalInstructPix2PixProvider(
            steps=int(settings.get("local_steps", 20) or 20),
            text_guidance=float(settings.get("local_text_guidance", 7.5) or 7.5),
            image_guidance=float(settings.get("local_image_guidance", 1.5) or 1.5),
        )
    return HttpEndpointProvider(
        endpoint=str(settings.get("endpoint", "") or ""),
        api_key=str(settings.get("api_key", "") or ""),
        model_name=str(settings.get("model_name", "remote") or "remote"),
        timeout_s=float(settings.get("timeout_s", DEFAULT_TIMEOUT_S) or DEFAULT_TIMEOUT_S),
        allow_insecure=bool(settings.get("allow_insecure", False)),
    )
