"""Device names and ONNX Runtime providers for the PaddleX models.

Device names accepted everywhere in SquiddleOCR: ``auto`` (CUDA if available, else CPU), ``cpu``,
``cuda``, ``tensorrt``, ``coreml``. The PaddleX models (layout, text detection, tables) run on
ONNX Runtime: ``onnxruntime-gpu`` provides CUDA and TensorRT on Linux and Windows, ``onnxruntime``
provides CoreML on macOS; every build has the CPU provider. kraken's models run on torch and map
the same names to ``accelerator="cpu"`` or ``"auto"``.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PROVIDER_FOR_DEVICE = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}
_AUTO_ORDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def preload_cuda_libraries() -> None:
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


def paddlex_device(device: str = "auto") -> str:
    """Translate a device name into what PaddleX's onnxruntime engine accepts (``cpu``/``gpu``)."""
    providers = resolve_providers(device)
    if providers[0] in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        preload_cuda_libraries()
        return "gpu"
    return "cpu"
