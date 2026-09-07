"""PP-OCRv6 text detection through PaddleX's ONNX Runtime engine (``paddle`` extra)."""
from __future__ import annotations

import numpy as np

from ..crops import crop_bbox
from ..runtime import paddlex_device
from ..types import Page, Region, TextLine

DETECTORS = ("PP-OCRv6_medium_det", "PP-OCRv6_small_det", "PP-OCRv6_tiny_det")


class PaddleTextDetector:
    """DBNet-style detector; ``unclip_ratio`` expands boxes (historical print wants ~2.0, PaddleOCR's default is 1.5)."""

    def __init__(self, model_name: str = "PP-OCRv6_medium_det", device: str = "auto", unclip_ratio: float = 2.0,
                 box_thresh: float = 0.6, thresh: float = 0.3, pad: int = 0):
        from paddlex.inference import create_predictor

        self.model_name = model_name
        self.unclip_ratio, self.box_thresh, self.thresh, self.pad = unclip_ratio, box_thresh, thresh, pad
        self.predictor = create_predictor(model_name, engine="onnxruntime", device=paddlex_device(device))

    def detect(self, page: Page, region: Region | None = None) -> list[TextLine]:
        if region is None:
            image, dx, dy, rid = page.image, 0.0, 0.0, ""
        else:
            b = region.bbox
            image = crop_bbox(page.image, b, self.pad)
            dx, dy, rid = max(b.x0 - self.pad, 0.0), max(b.y0 - self.pad, 0.0), region.id
            if image.size == 0:
                return []
        res = next(iter(self.predictor.predict(np.ascontiguousarray(image), thresh=self.thresh,
                                                box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio)))
        lines = []
        for poly, score in zip(res["dt_polys"], res["dt_scores"]):
            pts = np.asarray(poly, dtype=float).reshape(-1, 2) + np.array([int(dx), int(dy)], dtype=float)
            lines.append(TextLine(pts, float(score), region_id=rid))
        return lines
