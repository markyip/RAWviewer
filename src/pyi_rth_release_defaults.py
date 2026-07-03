"""PyInstaller runtime hook: release defaults (opt-out via environment variables)."""
import os

os.environ.setdefault("RAWVIEWER_GPU_VIEW", "1")
# Match launch_dev_full.sh so packaged Full builds use perf-v2 search/index paths.
os.environ.setdefault("RAWVIEWER_PERF_V2", "1")
