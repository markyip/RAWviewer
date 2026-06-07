#!/usr/bin/env python3
"""Preview the RAWviewer 'Update Available' dialog without launching the full app."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from PyQt6.QtWidgets import QApplication

from rawviewer_ui.release_update_dialog import show_release_update_dialog
from release_update import APP_VERSION


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("RAWviewer Update Preview")

    current = os.environ.get("RAWVIEWER_MOCK_CURRENT_VERSION", APP_VERSION).strip()
    latest = os.environ.get("RAWVIEWER_MOCK_UPDATE_VERSION", "2.4.0").strip()
    dismissed = show_release_update_dialog(
        current=current,
        latest=latest,
    )

    if dismissed:
        print("[preview] Not Now / dismissed (14-day snooze would apply in the full app)")
    else:
        print("[preview] Open Download Page clicked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
