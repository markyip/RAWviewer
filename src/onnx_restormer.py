"""Backward-compatibility wrapper forwarding RestormerONNX calls to SCUNetONNX."""

import os
from onnx_scunet import (
    DENOISE_MODEL_URL,
    SCUNetONNX,
    ensure_scunet_model_downloaded,
    scunet_model_path,
)

# Backward compatibility aliases
RestormerONNX = SCUNetONNX
restormer_model_path = scunet_model_path
ensure_restormer_model_downloaded = ensure_scunet_model_downloaded

__all__ = [
    "DENOISE_MODEL_URL",
    "RestormerONNX",
    "restormer_model_path",
    "ensure_restormer_model_downloaded",
]
