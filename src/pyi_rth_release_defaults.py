"""PyInstaller runtime hook: release defaults (opt-out via environment variables)."""
import os
import sys

os.environ.setdefault("RAWVIEWER_GPU_VIEW", "1")
# Match launch_dev.sh / launch_dev_full.sh packaged ↔ dev search/index paths.
os.environ.setdefault("RAWVIEWER_PERF_V2", "1")
os.environ.setdefault("RAWVIEWER_AUTO_METADATA_INDEX", "1")

# GPU demosaic preference must match platform launch scripts:
# - Windows Full: prefer CUDA/CuPy when present (launch_dev.bat sets 1 for CUDA editions).
# - macOS: keep OFF — MPS demosaic serializes heavy RAW (raw_limit=1) and starves
#   gallery tile fill; launch_dev.sh / pixi.toml default to 0, and macOS .app
#   builds also exclude torch so forcing 1 is misleading.
# Lite already setdefault'd RAWVIEWER_PREFER_GPU_DECODE=0 in pyi_rth_profile_defaults
# (runs before this hook).
if sys.platform == "darwin":
    os.environ.setdefault("RAWVIEWER_PREFER_GPU_DECODE", "0")
else:
    os.environ.setdefault("RAWVIEWER_PREFER_GPU_DECODE", "1")

# Do NOT force RAWVIEWER_USE_PROCESS_POOL=0 here. When GPU demosaic is actually
# in use, common_image_loader.use_raw_process_pool() / ImageLoadManager already
# disable the LibRaw process pool. Forcing "0" previously killed multi-core CPU
# Fast RAW (Lite / DirectML / PREFER_GPU=0) even on 8+ core Windows boxes.
