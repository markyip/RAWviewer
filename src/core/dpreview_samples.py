"""DPReview search URLs for ColorChecker camera calibration.

No network, no download, no HTML scraping. The calibrate dialog opens a
browser search so the user can pick a studio/sample file themselves, then
upload it with the existing file picker.
"""

from __future__ import annotations

import re
import urllib.parse

DPREVIEW_SEARCH_URL = "https://www.dpreview.com/search/?q="
# Studio shot comparison: pick a camera body and download a test scene.
DPREVIEW_STUDIO_HUB_URL = "https://www.dpreview.com/reviews/image-comparison"

_MAKE_NOISE = re.compile(
    r"\b(corporation|corp\.?|incorporated|inc\.?|ltd\.?|limited|co\.?|company)\b",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def _collapse_ws(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def normalize_make(make: str) -> str:
    """Drop Inc/Corp suffixes; title-case ALL-CAPS EXIF makes."""
    text = _MAKE_NOISE.sub("", make or "")
    text = _collapse_ws(text)
    if text.isupper() and any(ch.isalpha() for ch in text):
        text = text.title()
    return text


def camera_search_terms(make: str, model: str) -> str:
    """Human camera name for DPReview search, without 'Canon Canon EOS R5'."""
    make_n = normalize_make(make)
    model_n = _collapse_ws(model or "")
    if make_n and model_n:
        mk = make_n.lower()
        md = model_n.lower()
        if md.startswith(mk):
            rest = model_n[len(make_n) :].strip(" -_/")
            return _collapse_ws(f"{make_n} {rest}" if rest else make_n)
        first = model_n.split()[0].lower()
        if first == mk.split()[0]:
            return model_n if mk in md else _collapse_ws(f"{make_n} {model_n}")
        return _collapse_ws(f"{make_n} {model_n}")
    return make_n or model_n


def dpreview_search_query(make: str, model: str) -> str:
    terms = camera_search_terms(make, model)
    if terms:
        return f"{terms} studio scene"
    return ""


def dpreview_open_url(make: str = "", model: str = "") -> str:
    """Browser URL: per-camera site search, or the studio-comparison hub.

    Empty make/model must not invent a camera query — open the generic
    DPReview studio/cameras page instead.
    """
    query = dpreview_search_query(make, model)
    if not query:
        return DPREVIEW_STUDIO_HUB_URL
    return f"{DPREVIEW_SEARCH_URL}{urllib.parse.quote_plus(query)}"

