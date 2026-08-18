"""Splash card includes version; status helper is a no-op without a widget."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


class SplashCopyTests(unittest.TestCase):
    def test_version_matches_app_version(self):
        from app_version import APP_VERSION
        from startup_splash import app_version_for_splash, splash_subtitle

        self.assertEqual(app_version_for_splash(), APP_VERSION)
        sub = splash_subtitle()
        self.assertIn(APP_VERSION, sub)
        self.assertTrue("Plus" in sub or "Standard" in sub)

    def test_status_helper_tolerates_none(self):
        from startup_splash import splash_show_status

        splash_show_status(None, "Starting…")

    def test_splash_card_has_footer(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(["splash-test"])
        from startup_splash import (
            SPLASH_CARD_H,
            SPLASH_CARD_W,
            SPLASH_FOOTER_H,
            make_splash_pixmap,
        )

        pm = make_splash_pixmap()
        self.assertFalse(pm.isNull())
        self.assertEqual(pm.width(), SPLASH_CARD_W)
        self.assertEqual(pm.height(), SPLASH_CARD_H)
        self.assertGreater(SPLASH_FOOTER_H, 40)
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
