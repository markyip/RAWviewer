"""PyInstaller runtime hook: disable LibRaw process pool by default on macOS."""
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("RAWVIEWER_USE_PROCESS_POOL", "0")
