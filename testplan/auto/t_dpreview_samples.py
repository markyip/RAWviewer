"""DPReview calibration search URLs — no network.

The calibrate dialog only opens a browser URL. These tests lock the query
built from EXIF make/model and the empty-EXIF fallback hub.
"""
from __future__ import annotations

import os
import sys
from urllib.parse import parse_qs, unquote_plus, urlparse

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from dpreview_samples import (  # noqa: E402
    DPREVIEW_SEARCH_URL,
    DPREVIEW_STUDIO_HUB_URL,
    camera_search_terms,
    dpreview_open_url,
    dpreview_search_query,
)


def _query_q(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    raw = (qs.get("q") or [""])[0]
    return unquote_plus(raw)


def test_nikon_z8_search_url() -> None:
    url = dpreview_open_url("Nikon", "Z8")
    assert url.startswith(DPREVIEW_SEARCH_URL), url
    q = _query_q(url)
    assert "Nikon" in q and "Z8" in q, q
    assert "studio scene" in q, q
    assert "Nikon Nikon" not in q, q


def test_nikon_exif_corporation_dedupes() -> None:
    terms = camera_search_terms("NIKON CORPORATION", "NIKON Z 8")
    assert terms.lower().count("nikon") == 1, terms
    assert "Z 8" in terms or "Z8" in terms, terms
    q = dpreview_search_query("NIKON CORPORATION", "NIKON Z 8")
    assert q.endswith("studio scene"), q


def test_canon_eos_r5_no_double_make() -> None:
    url = dpreview_open_url("Canon", "Canon EOS R5")
    q = _query_q(url)
    assert q == "Canon EOS R5 studio scene", q
    assert "Canon Canon" not in q, q
    assert url.startswith("https://www.dpreview.com/search/"), url


def test_empty_exif_opens_studio_hub() -> None:
    assert dpreview_search_query("", "") == ""
    assert camera_search_terms("", "") == ""
    url = dpreview_open_url("", "")
    assert url == DPREVIEW_STUDIO_HUB_URL, url
    assert "search/?q=" not in url
    assert dpreview_open_url() == DPREVIEW_STUDIO_HUB_URL


def test_whitespace_only_exif_is_empty() -> None:
    assert dpreview_open_url("  ", "\t") == DPREVIEW_STUDIO_HUB_URL


def main() -> int:
    test_nikon_z8_search_url()
    test_nikon_exif_corporation_dedupes()
    test_canon_eos_r5_no_double_make()
    test_empty_exif_opens_studio_hub()
    test_whitespace_only_exif_is_empty()
    print("PASS t_dpreview_samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
