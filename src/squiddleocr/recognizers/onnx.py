"""The SquiddleOCR recogniser: a converted kraken PP-OCRv6 model run with ONNX Runtime."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml

from ..runtime import create_session
from ..types import Recognition
from .ctc import ctc_greedy_decode

MAX_WIDTH = 3200  # PaddleOCR's cap; kept so results match the PaddleOCR drop-in exactly


def _read_model_config(model_dir: Path) -> tuple[int, list[str]]:
    cfg = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))
    shape = next(op["RecResizeImg"]["image_shape"] for op in cfg["PreProcess"]["transform_ops"] if "RecResizeImg" in op)
    return int(shape[1]), list(cfg["PostProcess"]["character_dict"])


def preprocess_line(image: np.ndarray, height: int, max_width: int = MAX_WIDTH) -> np.ndarray:
    """RGB line image -> ``(3, height, W)`` float32 in ``[-1, 1]`` (the contract of the converted model).

    Same arithmetic as PaddleX's ``OCRReisizeNormImg``: bilinear resize to ``height`` keeping the
    aspect ratio, ``x/255``, ``-0.5``, ``/0.5``. Everything kraken-specific (inversion, white
    padding, batch masking) lives inside the ONNX graph.
    """
    h, w = image.shape[:2]
    new_w = min(max(int(math.ceil(height * w / float(h))), 1), max_width)
    resized = cv2.resize(image, (new_w, height))
    x = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (x - 0.5) / 0.5


def batch_lines(tensors: Sequence[np.ndarray]) -> np.ndarray:
    """Right-pad ``(3, H, W_i)`` tensors with zeros to the widest and stack; the graph detects the padding."""
    width = max(t.shape[2] for t in tensors)
    batch = np.zeros((len(tensors), 3, tensors[0].shape[1], width), dtype=np.float32)
    for i, t in enumerate(tensors):
        batch[i, :, :, : t.shape[2]] = t
    return batch


class OnnxRecognizer:
    """Runs ``inference.onnx`` of a SquiddleOCR model directory.

    Lines are sorted by width so that each batch pads as little as possible, then decoded greedily.
    ``batch_size=1`` reproduces kraken's single-line results exactly; larger batches reproduce
    kraken's own batched behaviour (see NOTES.md).
    """

    def __init__(self, model_dir: str | Path, device: str = "auto", batch_size: int = 8):
        self.model_dir = Path(model_dir)
        self.height, self.chars = _read_model_config(self.model_dir)
        self.batch_size = max(int(batch_size), 1)
        self.session = create_session(self.model_dir / "inference.onnx", device)
        self._input = self.session.get_inputs()[0].name

    @property
    def device_provider(self) -> str:
        return self.session.provider

    def recognize(self, lines: Sequence[np.ndarray]) -> list[Recognition]:
        if not lines:
            return []
        tensors = [preprocess_line(np.asarray(im), self.height) for im in lines]
        order = sorted(range(len(tensors)), key=lambda i: tensors[i].shape[2])
        out: list[Recognition | None] = [None] * len(tensors)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            probs = self.session.run(None, {self._input: batch_lines([tensors[i] for i in idx])})[0]
            for j, i in enumerate(idx):
                text, score = ctc_greedy_decode(probs[j], self.chars)
                out[i] = Recognition(text, score)
        return out  # type: ignore[return-value]
