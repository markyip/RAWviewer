"""
Persisted settings and the consent gate for generative editing.

Split out of raw_generative_edit so that module stays Qt-free and unit
testable; this one owns the QSettings keys and is the single authority on
whether the user has agreed to upload their photographs.

The consent gate is GLOBAL (one decision, remembered) rather than
per-folder. It is stored alongside the endpoint it was granted for:
consenting to your own loopback server is not consent to send the same
photographs to a different company's API, so changing the endpoint
revokes it and the user is asked again. That is the one piece of
per-context strictness worth keeping in an otherwise global gate.
"""

from __future__ import annotations

from typing import Optional

ORG = "RAWviewer"
APP = "RAWviewer"

_K_PROVIDER = "generative/provider"
_K_ENDPOINT = "generative/endpoint"
_K_API_KEY = "generative/api_key"
_K_MODEL = "generative/model_name"
_K_INSECURE = "generative/allow_insecure"
_K_CONSENT_FOR = "generative/consent_endpoint"
_K_SERVER_ENDPOINT = "generative/server_endpoint"
_K_SERVER_MODEL = "generative/server_model"
_K_LOCAL_STEPS = "generative/local_steps"
_K_LOCAL_TEXT_G = "generative/local_text_guidance"
_K_LOCAL_IMAGE_G = "generative/local_image_guidance"


def _settings():
    from PyQt6.QtCore import QSettings

    return QSettings(ORG, APP)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_settings() -> dict:
    """Everything raw_generative_edit.make_provider needs."""
    s = _settings()
    return {
        "provider": str(s.value(_K_PROVIDER, "http") or "http"),
        "endpoint": str(s.value(_K_ENDPOINT, "") or ""),
        "api_key": str(s.value(_K_API_KEY, "") or ""),
        "model_name": str(s.value(_K_MODEL, "remote") or "remote"),
        "allow_insecure": _as_bool(s.value(_K_INSECURE), False),
        # Local provider (raw_generative_local). Defaults chosen for time,
        # not fidelity: one UNet step costs ~10-19 s on Apple Silicon, so the
        # reference 20 steps would be a 3-6 minute wait per edit.
        "local_steps": int(s.value(_K_LOCAL_STEPS, 10) or 10),
        "local_text_guidance": float(s.value(_K_LOCAL_TEXT_G, 7.5) or 7.5),
        "local_image_guidance": float(s.value(_K_LOCAL_IMAGE_G, 1.5) or 1.5),
        # Local-server provider (raw_generative_local_server).
        "server_endpoint": str(s.value(_K_SERVER_ENDPOINT, "") or ""),
        "server_model": str(s.value(_K_SERVER_MODEL, "") or ""),
    }


def save_server_settings(
    *, endpoint: str, model_name: str = "", api_key: str = ""
) -> None:
    """Select the local-server provider and remember where it listens.

    Kept apart from save_settings deliberately. That one revokes upload
    consent whenever the endpoint changes, which is right for a remote
    destination and meaningless here -- a loopback server uploads nothing.
    LocalServerProvider derives requires_consent from the URL itself, so a
    user who later points this at a remote host is still asked.
    """
    s = _settings()
    s.setValue(_K_PROVIDER, "local_server")
    s.setValue(_K_SERVER_ENDPOINT, endpoint or "")
    s.setValue(_K_SERVER_MODEL, model_name or "")
    s.setValue(_K_API_KEY, api_key or "")


def save_local_settings(
    *, steps: int = 10, text_guidance: float = 7.5, image_guidance: float = 1.5
) -> None:
    """Select the local provider and persist its sampling knobs.

    No consent is revoked or granted here: the local model never sends the
    photograph anywhere, so there is nothing to agree to.
    """
    s = _settings()
    s.setValue(_K_PROVIDER, "local")
    s.setValue(_K_LOCAL_STEPS, int(steps))
    s.setValue(_K_LOCAL_TEXT_G, float(text_guidance))
    s.setValue(_K_LOCAL_IMAGE_G, float(image_guidance))


def save_settings(
    *,
    endpoint: str,
    api_key: str = "",
    model_name: str = "remote",
    allow_insecure: bool = False,
    provider: str = "http",
) -> None:
    s = _settings()
    previous = str(s.value(_K_ENDPOINT, "") or "")
    s.setValue(_K_PROVIDER, provider)
    s.setValue(_K_ENDPOINT, endpoint or "")
    s.setValue(_K_API_KEY, api_key or "")
    s.setValue(_K_MODEL, model_name or "remote")
    s.setValue(_K_INSECURE, bool(allow_insecure))
    if (endpoint or "") != previous:
        # Consent was granted for a specific destination, not in general.
        revoke_consent()


def has_consent(endpoint: Optional[str] = None) -> bool:
    """True if the user agreed to upload photos to this exact endpoint."""
    s = _settings()
    granted_for = str(s.value(_K_CONSENT_FOR, "") or "")
    if not granted_for:
        return False
    target = endpoint if endpoint is not None else str(s.value(_K_ENDPOINT, "") or "")
    return bool(target) and granted_for == target


def grant_consent(endpoint: str) -> None:
    _settings().setValue(_K_CONSENT_FOR, endpoint or "")


def revoke_consent() -> None:
    _settings().setValue(_K_CONSENT_FOR, "")


def consent_prompt_text(endpoint: str) -> str:
    """Wording for the gate. Concrete about what leaves the machine.

    Deliberately plain: a photographer needs to know their client's
    images are being uploaded to a third party, in the terms they would
    use themselves, not in the language of a EULA.
    """
    return (
        "Generative editing sends a copy of your photograph to:\n\n"
        f"    {endpoint}\n\n"
        "The image leaves this computer. Depending on who runs that "
        "server, it may be stored, logged, or used to train models.\n\n"
        "Do not use this for images under NDA, images of people who have "
        "not agreed to it, or any work you are not free to share.\n\n"
        "Your RAW files are never modified or uploaded — only the "
        "rendered copy you are editing.\n\n"
        "Continue?"
    )
