"""ONNX Runtime sessions with automatic execution-provider selection.

Device names accepted everywhere in SquiddleOCR: ``auto`` (best available), ``cpu``, ``cuda``,
``tensorrt``, ``coreml``. ``onnxruntime-gpu`` provides CUDA and TensorRT on Linux and Windows,
``onnxruntime`` provides CoreML on macOS; every build has the CPU provider.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROVIDER_FOR_DEVICE = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}
_AUTO_ORDER = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]


def _preload_cuda_libraries() -> None:
    """Let ONNX Runtime find the CUDA/cuDNN pip packages (``nvidia-*``) without LD_LIBRARY_PATH."""
    import onnxruntime as ort

    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as e:  # pragma: no cover - depends on the installed CUDA packages
        logger.debug("onnxruntime.preload_dlls failed: %s", e)


def available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def resolve_providers(device: str = "auto") -> list[str]:
    """Map a device name to an ordered provider list, keeping only providers this build offers."""
    device = (device or "auto").lower()
    avail = set(available_providers())
    if device == "auto":
        wanted = _AUTO_ORDER
    elif device in _PROVIDER_FOR_DEVICE:
        wanted = _PROVIDER_FOR_DEVICE[device]
    else:
        raise ValueError(f"Unknown device {device!r}; use one of auto, {', '.join(_PROVIDER_FOR_DEVICE)}")
    providers = [p for p in wanted if p in avail]
    if device != "auto" and device != "cpu" and providers[:1] == ["CPUExecutionProvider"]:
        raise RuntimeError(
            f"Device {device!r} requested but its execution provider is not available; "
            f"available: {sorted(avail)}. Install onnxruntime-gpu (Linux/Windows) or onnxruntime (macOS)."
        )
    return providers or ["CPUExecutionProvider"]


def create_session(model_path: str | Path, device: str = "auto", *, intra_op_threads: int | None = None):
    """Create an ``InferenceSession`` for ``model_path`` on ``device``."""
    import onnxruntime as ort

    providers = resolve_providers(device)
    if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
        _preload_cuda_libraries()
    opts = ort.SessionOptions()
    opts.log_severity_level = int(os.environ.get("SQUIDDLE_ORT_LOG_LEVEL", "3"))
    if intra_op_threads:
        opts.intra_op_num_threads = intra_op_threads
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    logger.info("%s on %s", Path(model_path).name, session.get_providers()[0])
    return session


def paddlex_device(device: str = "auto") -> str:
    """Translate a SquiddleOCR device name into what PaddleX's onnxruntime engine accepts (cpu/gpu)."""
    providers = resolve_providers(device)
    return "gpu" if providers[0] in ("CUDAExecutionProvider", "TensorrtExecutionProvider") else "cpu"
