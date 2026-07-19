"""PyInstaller runtime hook: release defaults (opt-out via environment variables)."""
import os

os.environ.setdefault("RAWVIEWER_GPU_VIEW", "1")
# Match launch_dev_full.sh so packaged Full builds use perf-v2 search/index paths.
os.environ.setdefault("RAWVIEWER_PERF_V2", "1")
# Full: prefer CUDA demosaic when the backend is present. Lite already
# setdefault'd RAWVIEWER_PREFER_GPU_DECODE=0 in pyi_rth_profile_defaults
# (runs before this hook), so this only applies to Full / unset profiles.
os.environ.setdefault("RAWVIEWER_PREFER_GPU_DECODE", "1")
# Do NOT force RAWVIEWER_USE_PROCESS_POOL=0 here. When GPU demosaic is actually
# in use, common_image_loader.use_raw_process_pool() / ImageLoadManager already
# disable the LibRaw process pool. Forcing "0" previously killed multi-core CPU
# Fast RAW (Lite / DirectML / PREFER_GPU=0) even on 8+ core Windows boxes.
