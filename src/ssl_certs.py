"""HTTPS certificate setup for bundled Python (PyInstaller) and urllib downloads."""

from __future__ import annotations

import os
import shutil
import ssl
import urllib.request

_CONFIGURED = False


def configure_ssl_certificates() -> bool:
    """Point stdlib SSL and common HTTP clients at certifi's CA bundle."""
    global _CONFIGURED
    if _CONFIGURED:
        return True
    try:
        import certifi
    except ImportError:
        return False

    cafile = certifi.where()
    if not cafile or not os.path.isfile(cafile):
        return False

    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    os.environ.setdefault("CURL_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cafile)
    _CONFIGURED = True
    return True


def urlretrieve(
    url: str,
    dest_path: str,
    *,
    timeout: int = 120,
    user_agent: str = "RAWviewer",
) -> None:
    """Download *url* to *dest_path* using certifi when available."""
    configure_ssl_certificates()
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(resp, out)
