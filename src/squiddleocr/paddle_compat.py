"""One place to create PaddleX predictors quietly (``paddle`` extra).

Two engines: ``onnxruntime`` for the exported vision models (layout, text detection, tables) and
``transformers`` for the models PaddleX ships as safetensors for Hugging Face ``transformers`` on
torch (PP-FormulaNet). Paddle Inference itself is never used.

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
        try:
            from transformers.utils import logging as tf_logging

            tf_logging.set_verbosity_error()
            tf_logging.disable_progress_bar()   # the "Loading weights" bar; model downloads still show
        except ImportError:
            pass
    _configured = True


def paddle_image(image):
    """An RGB page array as PaddleX wants it: PaddleX reads arrays as cv2 does (BGR) and converts them
    itself for models trained on RGB, so an RGB array has to go in with its channels swapped."""
    import numpy as np

    return np.ascontiguousarray(image[:, :, ::-1]) if image.ndim == 3 and image.shape[2] == 3 else np.ascontiguousarray(image)


def create_predictor(model_name: str, device: str = "auto", **kwargs):
    """``paddlex.inference.create_predictor`` on the ONNX Runtime engine for ``device``, quietly."""
    _configure()
    from paddlex.inference import create_predictor as _create

    engine_config = {"log_severity_level": 3, **kwargs.pop("engine_config", {})}
    return _create(model_name, engine="onnxruntime", device=paddlex_device(device), engine_config=engine_config, **kwargs)


def create_transformers_predictor(model_name: str, device: str = "auto", **kwargs):
    """``paddlex.inference.create_predictor`` on the transformers engine (torch), on the GPU when there is one."""
    _configure()
    import torch
    from paddlex.inference import create_predictor as _create

    cuda = device != "cpu" and torch.cuda.is_available()
    engine_config = {"device_type": "cuda" if cuda else "cpu", **({"device_id": 0} if cuda else {}),
                     **kwargs.pop("engine_config", {})}
    return _create(model_name, engine="transformers", device="gpu:0" if cuda else "cpu", engine_config=engine_config, **kwargs)


def create_pipeline(config: dict, device: str = "auto", **kwargs):
    """``paddlex.create_pipeline`` from a config dict on the ONNX Runtime engine for ``device``, quietly."""
    _configure()
    from paddlex import create_pipeline as _create

    return _create(config=config, engine="onnxruntime", device=paddlex_device(device), **kwargs)
