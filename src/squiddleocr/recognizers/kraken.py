"""kraken's own recogniser, unchanged: ``RecognitionTaskModel`` on kraken's PP-OCRv6 weights.

Lines are extracted, transformed, batched, decoded and turned into ``ocr_record``s (text,
per-character cuts and confidences) by kraken itself; the weights are kraken's safetensors from
Zenodo, running on torch. Our part is only ``segmentation.py``: turning the detector's lines into
the kraken ``Segmentation`` that ``predict`` takes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from ..segmentation import lines_to_segmentation
from ..types import Page, Recognition, TextLine


def record_to_recognition(rec) -> Recognition:
    conf = [float(c) for c in getattr(rec, "confidences", []) or []]
    return Recognition(rec.prediction, sum(conf) / len(conf) if conf else 0.0)


class KrakenRecognizer:
    """``kraken.tasks.recognition.RecognitionTaskModel`` with a ``RecognitionInferenceConfig``."""

    def __init__(self, model_path: str | Path, device: str = "auto", batch_size: int = 8,
                 num_line_workers: int | None = None, text_direction: str = "horizontal-lr"):
        from kraken.configs import RecognitionInferenceConfig
        from kraken.tasks.recognition import RecognitionTaskModel

        self.model_path = Path(model_path)
        self.task = RecognitionTaskModel.load_model(str(self.model_path))
        self.text_direction = text_direction
        kwargs = {"batch_size": max(int(batch_size), 1), "accelerator": "cpu" if device == "cpu" else "auto"}
        if num_line_workers is not None:
            kwargs["num_line_workers"] = num_line_workers
        self.config = RecognitionInferenceConfig(**kwargs)
        self.task.net.prepare_for_inference(self.config)   # kraken does this per predict; doing it now places the model

    @property
    def device(self) -> str:
        try:
            return str(next(self.task.net.parameters()).device)
        except StopIteration:  # pragma: no cover
            return "cpu"

    def predict(self, im: Image.Image, segmentation) -> list:
        """kraken's ``RecognitionTaskModel.predict`` on a kraken ``Segmentation``; records in line order."""
        return list(self.task.predict(im, segmentation, self.config))

    def recognize_lines(self, page: Page, lines: Sequence[TextLine]) -> list:
        if not lines:
            return []
        return self.predict(Image.fromarray(page.image), lines_to_segmentation(page, lines, self.text_direction))
