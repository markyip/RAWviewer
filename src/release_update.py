"""Compare the running app version against the latest GitHub release."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

from app_version import APP_VERSION
GITHUB_REPO = "markyip/RAWviewer"
GITHUB_LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASE_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"


def parse_version(label: str) -> Tuple[int, ...]:
    """Extract numeric version parts from a tag or version string (e.g. v2.3.0 -> (2, 3, 0))."""
    text = (label or "").strip().lstrip("vV")
    parts = re.findall(r"\d+", text)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts[:4])


def compare_version_labels(current: str, latest: str) -> int:
    """Return -1 if current < latest, 0 if equal, 1 if current > latest."""
    a = parse_version(current)
    b = parse_version(latest)
    width = max(len(a), len(b))
    a = a + (0,) * (width - len(a))
    b = b + (0,) * (width - len(b))
    for x, y in zip(a, b):
        if x < y:
            return -1
        if x > y:
            return 1
    return 0


def _is_connectivity_error(exc: BaseException) -> bool:
    """True when the release check failed due to network reachability, not app logic."""
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        # Windows: 10051 network unreachable, 10060/10061 timeout; POSIX often errno 101/110/111
        errno = getattr(exc, "errno", None)
        if errno in (10051, 10060, 10061, 101, 110, 111, 113):
            return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "network error" in msg or "timed out" in msg:
            return True
    return False


def fetch_latest_release(*, timeout: float = 8.0) -> Dict[str, str]:
    """Fetch GitHub /releases/latest metadata. Raises on network or parse failure."""
    req = urllib.request.Request(
        GITHUB_LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"RAWviewer/{APP_VERSION}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub API HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid GitHub API response") from exc

    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError("GitHub release has no tag_name")
    return {
        "tag_name": tag,
        "name": str(payload.get("name") or tag).strip(),
        "html_url": str(payload.get("html_url") or RELEASE_PAGE_URL).strip(),
        "published_at": str(payload.get("published_at") or "").strip(),
    }


def mock_update_result(
    mock_latest: str,
    current_version: Optional[str] = None,
    *,
    release_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a successful update-check result for UI/dev testing (no network)."""
    current = (current_version or APP_VERSION).strip() or APP_VERSION
    latest = (mock_latest or "").strip()
    result: Dict[str, Any] = {
        "current": current,
        "latest": latest,
        "is_latest": True,
        "update_available": False,
        "release_url": (release_url or RELEASE_PAGE_URL).strip() or RELEASE_PAGE_URL,
        "release_name": f"RAWviewer {latest} (preview)",
        "published_at": "",
        "offline": False,
        "error": "",
    }
    if not latest:
        result["error"] = "mock_latest is empty"
        return result
    cmp = compare_version_labels(current, latest)
    result["is_latest"] = cmp >= 0
    result["update_available"] = cmp < 0
    return result


def check_for_update(
    current_version: Optional[str] = None,
    *,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """
    Check whether *current_version* is the latest GitHub release.

    Returns a dict with keys: current, latest, is_latest, update_available,
    release_url, release_name, published_at, offline, error (empty when check succeeded).
    On network failure, ``offline`` is True and the caller should exit silently.
    """
    current = (current_version or APP_VERSION).strip() or APP_VERSION
    mock_latest = os.environ.get("RAWVIEWER_MOCK_UPDATE_VERSION", "").strip()
    if mock_latest:
        logger.info("[UPDATE] Mock mode: simulating latest=%s", mock_latest)
        return mock_update_result(mock_latest, current)

    result: Dict[str, Any] = {
        "current": current,
        "latest": "",
        "is_latest": True,
        "update_available": False,
        "release_url": RELEASE_PAGE_URL,
        "release_name": "",
        "published_at": "",
        "offline": False,
        "error": "",
    }
    try:
        release = fetch_latest_release(timeout=timeout)
    except Exception as exc:
        if _is_connectivity_error(exc):
            logger.debug("[UPDATE] Release check skipped (offline/unreachable): %s", exc)
            result["offline"] = True
        else:
            logger.debug("[UPDATE] Release check failed: %s", exc)
            result["error"] = str(exc)
        return result

    latest = release["tag_name"]
    result["latest"] = latest
    result["release_name"] = release.get("name") or latest
    result["release_url"] = release.get("html_url") or RELEASE_PAGE_URL
    result["published_at"] = release.get("published_at") or ""
    cmp = compare_version_labels(current, latest)
    result["is_latest"] = cmp >= 0
    result["update_available"] = cmp < 0
    logger.info(
        "[UPDATE] current=%s latest=%s update_available=%s",
        current,
        latest,
        result["update_available"],
    )
    return result
