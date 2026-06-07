"""PyInstaller runtime hook: configure certifi CA bundle before any HTTPS."""
import os
import ssl

try:
    import certifi

    _cafile = certifi.where()
    if _cafile and os.path.isfile(_cafile):
        os.environ.setdefault("SSL_CERT_FILE", _cafile)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", _cafile)
        os.environ.setdefault("CURL_CA_BUNDLE", _cafile)
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=_cafile)
except Exception:
    pass
