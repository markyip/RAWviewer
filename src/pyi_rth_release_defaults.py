"""PyInstaller runtime hook: release defaults (opt-out via environment variables)."""
import os

os.environ.setdefault("RAWVIEWER_GPU_VIEW", "1")
