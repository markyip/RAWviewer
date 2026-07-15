"""PyInstaller runtime hook: release defaults (opt-out via environment variables)."""
import os

os.environ.setdefault("RAWVIEWER_GPU_VIEW", "1")
# Match launch_dev_full.sh so packaged Full builds use perf-v2 search/index paths.
os.environ.setdefault("RAWVIEWER_PERF_V2", "1")
# Match pixi.toml so packaged builds use the same stable/tested demosaic and threading defaults
os.environ.setdefault("RAWVIEWER_PREFER_GPU_DECODE", "1")
os.environ.setdefault("RAWVIEWER_USE_PROCESS_POOL", "0")
