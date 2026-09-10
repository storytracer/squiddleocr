"""PP-OCRv6 text detection through PaddleX's ONNX Runtime engine (``paddle`` extra)."""
from __future__ import annotations

import numpy as np

from ..crops import crop_bbox, crop_polygon
from ..paddle_compat import create_predictor, paddle_image
from ..types import Page, Region, TextLine

DETECTORS = ("PP-OCRv6_medium_det", "PP-OCRv6_small_det", "PP-OCRv6_tiny_det")


class PaddleTextDetector:
    """DBNet-style detector; ``unclip_ratio`` expands boxes (historical print wants ~2.0, PaddleOCR's default is 1.5).

    Given a region, detection runs on its crop: the bounding box with ``pad`` pixels around it and,
    with ``mask``, everything outside the region polygon painted white (a layout analyser's regions
    in front of the detector, so a neighbouring column cannot leak in)."""

    def __init__(self, model_name: str = "PP-OCRv6_medium_det", device: str = "auto", unclip_ratio: float = 2.0,
                 box_thresh: float = 0.6, thresh: float = 0.3, pad: int = 0, mask: bool = False):
        self.model_name = model_name
        self.unclip_ratio, self.box_thresh, self.thresh, self.pad, self.mask = unclip_ratio, box_thresh, thresh, pad, mask
        self.predictor = create_predictor(model_name, device)

    def detect(self, page: Page, region: Region | None = None) -> list[TextLine]:
        if region is None:
            image, dx, dy, rid = page.image, 0.0, 0.0, ""
        elif self.mask:
            image, dx, dy = crop_polygon(page.image, region.polygon, self.pad)
            rid = region.id
            if image.size == 0:
                return []
        else:
            b = region.bbox
            image = crop_bbox(page.image, b, self.pad)
            dx, dy, rid = max(b.x0 - self.pad, 0.0), max(b.y0 - self.pad, 0.0), region.id
            if image.size == 0:
                return []
        res = next(iter(self.predictor.predict(paddle_image(image), thresh=self.thresh,
                                                box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio)))
        lines = []
        for poly, score in zip(res["dt_polys"], res["dt_scores"]):
            pts = np.asarray(poly, dtype=float).reshape(-1, 2) + np.array([int(dx), int(dy)], dtype=float)
            lines.append(TextLine(pts, float(score), region_id=rid))
        return lines
