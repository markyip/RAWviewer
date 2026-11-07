"""_activate_macos_foreground_app: opt-in NSApplication foreground."""
from __future__ import annotations

import sys

from rawviewer_app.env import _env_true


def _activate_macos_foreground_app() -> None:
    """Opt-in NSApplication foreground (RAWVIEWER_MACOS_FORCE_FOREGROUND=1). No-op elsewhere."""
    if sys.platform != "darwin" or not _env_true("RAWVIEWER_MACOS_FORCE_FOREGROUND"):
        return
    try:
        import objc

        ns_app = objc.lookUpClass("NSApplication").sharedApplication()
        if ns_app is not None:
            ns_app.setActivationPolicy_(0)
            ns_app.activateIgnoringOtherApps_(True)
    except Exception:
        pass
