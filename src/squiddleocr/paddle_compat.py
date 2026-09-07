"""One place to create PaddleX predictors quietly (``paddle`` extra).

PaddleX prints INFO lines about cached models and checks its model hosters on every start; ONNX
Runtime reports unused initialisers in the official ONNX exports. None of that is actionable for a
user, so predictors are created with PaddleX's logger at WARNING, the hoster check disabled and
ONNX Runtime's log level at ERROR (``SQUIDDLE_VERBOSE=1`` restores PaddleX's INFO output).
"""
from __future__ import annotations

import logging
import os

from .runtime import paddlex_device

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if not os.environ.get("SQUIDDLE_VERBOSE"):
        import onnxruntime as ort
        from paddlex.utils import logging as pdx_logging

        pdx_logging._logger.setLevel(logging.WARNING)
        ort.set_default_logger_severity(3)   # PaddleX's sessions trigger a harmless "plugin EP device" warning
    _configured = True


def create_predictor(model_name: str, device: str = "auto", **kwargs):
    """``paddlex.inference.create_predictor`` on the ONNX Runtime engine for ``device``, quietly."""
    _configure()
    from paddlex.inference import create_predictor as _create

    engine_config = {"log_severity_level": 3, **kwargs.pop("engine_config", {})}
    return _create(model_name, engine="onnxruntime", device=paddlex_device(device), engine_config=engine_config, **kwargs)
