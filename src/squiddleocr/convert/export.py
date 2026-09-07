"""ONNX export of the wrapped PP-OCRv6 network and parity checks against kraken."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .wrapper import PaddleRecWrapper

INPUT_NAME = "x"        # same as PaddleOCR's own recognisers
OUTPUT_NAME = "probs"
OPSET = 18


def export_onnx(
    wrapper: PaddleRecWrapper,
    path: str | Path,
    height: int,
    example_width: int = 640,
    opset: int = OPSET,
    embed_weights: bool = True,
) -> Path:
    """Export ``wrapper`` to ONNX with dynamic batch and width.

    Uses the dynamo-based exporter; the legacy TorchScript exporter fails on the
    backbone's dynamic padding. With ``embed_weights`` the weights are stored in
    the ``.onnx`` file itself, otherwise in ``<name>.onnx.data`` next to it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper.eval().cpu().float()
    x = torch.rand(1, 3, height, example_width) * 2 - 1
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch", 3: "width"},
                OUTPUT_NAME: {0: "batch", 1: "seq"},
            },
            opset_version=opset,
            dynamo=True,
            external_data=not embed_weights,
        )
    import onnx

    onnx.checker.check_model(str(path))
    return path


def make_session(path: str | Path, providers: list[str] | None = None):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])


def run_onnx(session, x: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    return session.run(None, {name: np.ascontiguousarray(x, dtype=np.float32)})[0]


def check_parity(
    wrapper: PaddleRecWrapper,
    session,
    height: int,
    widths: tuple[int, ...] = (640, 1000),
    batch: int = 1,
    seed: int = 0,
) -> dict[int, float]:
    """Compare ONNX Runtime with the PyTorch wrapper on random inputs.

    Returns ``{width: max_abs_diff}`` of the output probabilities.
    """
    g = torch.Generator().manual_seed(seed)
    diffs = {}
    wrapper = wrapper.eval().cpu().float()
    for w in widths:
        x = torch.rand(batch, 3, height, w, generator=g) * 2 - 1
        with torch.no_grad():
            ref = wrapper(x).numpy()
        out = run_onnx(session, x.numpy())
        if out.shape != ref.shape:
            raise RuntimeError(f"Shape mismatch at width {w}: onnx {out.shape} vs torch {ref.shape}")
        diffs[w] = float(np.abs(out - ref).max())
    return diffs
