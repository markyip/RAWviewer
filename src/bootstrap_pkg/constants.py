"""Constants and imports for the bootstrap/installer package."""
import os
import sys

# Pin the Windows installer pixi binary so a future upstream release cannot
# break installs mid-flight (do not use .../releases/latest/...).
PIXI_VERSION = "0.73.0"
PIXI_DOWNLOAD_URL = (
    f"https://github.com/prefix-dev/pixi/releases/download/v{PIXI_VERSION}/"
    "pixi-x86_64-pc-windows-msvc.zip"
)
DOWNLOAD_RETRIES = 3
RETRY_DELAY_SEC = 3
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024  # ~2 GiB for pixi env + models (Plus)
MIN_FREE_BYTES_LITE = 1024 * 1024 * 1024  # ~1 GiB for lite (no AI models)
INSTALLER_PROGRESS_PREFIX = "@RAWVIEWER_PROGRESS"
MODEL_DOWNLOAD_PROGRESS_START = 5
MODEL_DOWNLOAD_PROGRESS_END = 100

# Bundle directory — IMPORTANT: Keep PyInstaller bundle dir
# __file__ is .../src/bootstrap_pkg/constants.py, so we need
# dirname(dirname(dirname())) to reach the repo root.
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    EXE_DIR = BUNDLE_DIR

FILE_ASSOC_PROGID = "RAWviewer.Image"
