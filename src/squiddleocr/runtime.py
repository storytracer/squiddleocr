"""ONNX Runtime sessions with automatic execution-provider selection and CPU fallback.

Device names accepted everywhere in SquiddleOCR: ``auto`` (CUDA if available, else CPU), ``cpu``,
``cuda``, ``tensorrt``, ``coreml``. ``onnxruntime-gpu`` provides CUDA and TensorRT on Linux and
Windows, ``onnxruntime`` provides CoreML on macOS; every build has the CPU provider. CoreML is
opt-in only: it partitions the recogniser's dynamic-width graph and has failed at run time on
Apple Silicon, so ``auto`` does not select it. A session whose accelerated provider fails while
running is rebuilt once on the CPU and the call retried (``Session``).
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
_AUTO_ORDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_PROVIDER_OPTIONS = {
    # MLProgram is CoreML's newer format with better dynamic-shape support than NeuralNetwork.
    "CoreMLExecutionProvider": {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
}


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


def _raw_session(model_path: Path, providers: list[str], intra_op_threads: int | None):
    import onnxruntime as ort

    if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
        _preload_cuda_libraries()
    opts = ort.SessionOptions()
    opts.log_severity_level = int(os.environ.get("SQUIDDLE_ORT_LOG_LEVEL", "3"))
    if intra_op_threads:
        opts.intra_op_num_threads = intra_op_threads
    options = [_PROVIDER_OPTIONS.get(p, {}) for p in providers]
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers, provider_options=options)
    logger.info("%s on %s", model_path.name, session.get_providers()[0])
    return session


class Session:
    """An ``InferenceSession`` that falls back to the CPU provider once if its accelerator fails at run time."""

    def __init__(self, model_path: str | Path, device: str = "auto", *, intra_op_threads: int | None = None):
        self.model_path = Path(model_path)
        self.intra_op_threads = intra_op_threads
        self._session = _raw_session(self.model_path, resolve_providers(device), intra_op_threads)

    @property
    def provider(self) -> str:
        return self._session.get_providers()[0]

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def get_providers(self):
        return self._session.get_providers()

    def run(self, output_names, feeds):
        try:
            return self._session.run(output_names, feeds)
        except Exception as e:  # noqa: BLE001 - provider-specific failure classes vary
            if self.provider == "CPUExecutionProvider":
                raise
            logger.warning("%s failed on %s (%s); falling back to the CPU provider.",
                           self.model_path.name, self.provider, str(e).splitlines()[0][:160])
            self._session = _raw_session(self.model_path, ["CPUExecutionProvider"], self.intra_op_threads)
            return self._session.run(output_names, feeds)


def create_session(model_path: str | Path, device: str = "auto", *, intra_op_threads: int | None = None) -> Session:
    """Create a ``Session`` for ``model_path`` on ``device`` (see the module docstring for the device names)."""
    return Session(model_path, device, intra_op_threads=intra_op_threads)


def paddlex_device(device: str = "auto") -> str:
    """Translate a SquiddleOCR device name into what PaddleX's onnxruntime engine accepts (cpu/gpu)."""
    providers = resolve_providers(device)
    return "gpu" if providers[0] in ("CUDAExecutionProvider", "TensorrtExecutionProvider") else "cpu"
