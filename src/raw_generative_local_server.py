"""Generative editing against a local server that speaks the OpenAI image-edit API.

The motivating case is mlx-serve (https://github.com/ddalcu/mlx-serve), a
native Apple Silicon server that runs Mage-Flow-Edit-Turbo and exposes
``POST /v1/images/edits`` on ``http://localhost:11234``. Anything else with
that endpoint shape works too -- a llama.cpp-style server, a small FastAPI
shim in front of ComfyUI, or a hosted OpenAI-compatible API.

Why this rather than embedding the model
----------------------------------------
Mage-Flow-Edit-Turbo is 17.5 GB in bf16 and 9.7 GB quantised to 8-bit, and
ships as safetensors in diffusers layout -- torch or MLX, neither of which
this app has. Embedding it would mean owning a second ML runtime, a
multi-gigabyte download, and the memory management for a 4 GB transformer,
all for one feature.

Pointing at a server the user already runs costs none of that and gives a
strictly better model than anything that fits in-process. The photograph
still never leaves the machine, which is the property that actually matters:
``requires_consent`` is False for loopback for exactly that reason, and True
for anything else, because "OpenAI-compatible" says nothing about where the
host is.

This is the difference between local *inference* and local *process*. The
user gets the former without this app having to become the latter.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from typing import Callable, Optional
from urllib.parse import urlparse

import numpy as np

from raw_generative_edit import (
    CancelToken,
    GenerativeEditError,
    GenerativeProvider,
    GenerativeRequest,
    GenerativeResult,
    _encode_png_b64,
    build_provenance,
)

logger = logging.getLogger(__name__)

# mlx-serve's default. Not a fallback the app ever picks on its own -- the
# user chooses the local-server provider explicitly; this only spares them
# typing a URL they would otherwise have to look up.
DEFAULT_ENDPOINT = "http://localhost:11234/v1/images/edits"
DEFAULT_MODEL = "mage-flow-edit-turbo-8bit"

# Turbo models use a distilled 4-step schedule, so a local edit is seconds,
# not minutes. The timeout still has to cover a cold model load, which for a
# ~10 GB weight set is not fast.
DEFAULT_TIMEOUT_S = 600.0

_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def is_loopback(endpoint: str) -> bool:
    """Whether this URL points at the user's own machine."""
    try:
        host = (urlparse(endpoint or "").hostname or "").lower()
    except Exception:
        return False
    return host in _LOOPBACK_HOSTS


def _multipart(fields: dict, file_field: str, filename: str, data: bytes) -> tuple:
    """Encode multipart/form-data by hand.

    ``requests`` would do this in a line but is not a dependency here, and
    the format is simple enough that adding one for it would be silly.
    """
    boundary = f"----RAWviewer{uuid.uuid4().hex}"
    out = bytearray()
    for key, value in fields.items():
        if value is None:
            continue
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    out += f"--{boundary}\r\n".encode()
    out += (
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\n'
    ).encode()
    out += b"Content-Type: image/png\r\n\r\n"
    out += data
    out += f"\r\n--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


class LocalServerProvider(GenerativeProvider):
    """POST to an OpenAI-compatible ``/v1/images/edits`` endpoint."""

    name = "local-server"

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        model_name: str = DEFAULT_MODEL,
        api_key: str = "",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).strip()
        self.model_name = model_name or DEFAULT_MODEL
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)

    @property
    def requires_consent(self) -> bool:
        """Loopback needs no consent; the image never leaves the machine.

        Deliberately computed from the endpoint rather than fixed, because a
        user can point this provider at a remote OpenAI-compatible host, and
        that is a completely different act from running a model locally.
        """
        return not is_loopback(self.endpoint)

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def describe(self) -> str:
        where = "on this machine" if is_loopback(self.endpoint) else self.endpoint
        return f"{self.model_name} — {where}"

    def _validate(self) -> None:
        if not self.endpoint:
            raise GenerativeEditError("No local server endpoint is configured.")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https"):
            raise GenerativeEditError("Endpoint must be an http:// or https:// URL.")
        if parsed.scheme == "http" and not is_loopback(self.endpoint):
            raise GenerativeEditError(
                "Refusing to send photos over plain http to a non-local host."
            )

    def edit(
        self,
        request: GenerativeRequest,
        cancel: Optional[CancelToken] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> GenerativeResult:
        import urllib.error
        import urllib.request

        self._validate()
        instruction = (request.instruction or "").strip()
        if not instruction:
            raise GenerativeEditError("Describe the edit you want before generating.")
        if cancel is not None:
            cancel.raise_if_cancelled()

        if progress:
            progress("Preparing image…")
        png = base64.b64decode(_encode_png_b64(request.image))

        fields = {
            "prompt": instruction,
            "model": self.model_name,
            "n": "1",
            "response_format": "b64_json",
        }
        if request.seed is not None:
            fields["seed"] = str(int(request.seed))
        body, content_type = _multipart(fields, "image", "source.png", png)

        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if cancel is not None:
            cancel.raise_if_cancelled()
        if progress:
            progress("Generating on this machine…" if is_loopback(self.endpoint) else "Generating…")

        req = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise GenerativeEditError(_http_error_text(exc)) from exc
        except urllib.error.URLError as exc:
            hint = ""
            if is_loopback(self.endpoint):
                # By far the most likely cause, and the least obvious from
                # a bare "connection refused".
                hint = " Is the local server running?"
            raise GenerativeEditError(f"Could not reach {self.endpoint}.{hint}") from exc

        if cancel is not None:
            cancel.raise_if_cancelled()
        if progress:
            progress("Decoding…")
        image = _decode_edit_response(raw)

        return GenerativeResult(
            image=image,
            provenance=build_provenance(
                request,
                self.name,
                self.model_name,
                parent_provenance=(request.options or {}).get("parent_provenance"),
            ),
        )


def _http_error_text(exc) -> str:
    """Surface the server's own message; it is usually the useful one."""
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        err = payload.get("error")
        if isinstance(err, dict):
            err = err.get("message")
        if err:
            return str(err)
    except Exception:
        pass
    return f"Server returned {exc.code}."


def _decode_edit_response(raw: bytes) -> np.ndarray:
    """Pull the image out of an OpenAI image-edit response."""
    import cv2

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise GenerativeEditError("Server did not return JSON.") from exc

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        err = payload.get("error")
        if isinstance(err, dict):
            err = err.get("message")
        raise GenerativeEditError(str(err) if err else "Server returned no image.")

    entry = data[0] or {}
    b64 = entry.get("b64_json")
    if not b64:
        if entry.get("url"):
            # Deliberately not followed: fetching a returned URL turns a
            # local edit into a second network request to who-knows-where.
            raise GenerativeEditError(
                "Server returned a URL rather than image data. Ask it for "
                "response_format=b64_json."
            )
        raise GenerativeEditError("Server returned no image data.")

    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise GenerativeEditError("Could not decode the returned image.") from exc
    if bgr is None:
        raise GenerativeEditError("Could not decode the returned image.")
    return np.ascontiguousarray(bgr[:, :, ::-1])
