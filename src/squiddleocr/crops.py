"""Cutting region images out of a page (line images are kraken's business)."""
from __future__ import annotations

import numpy as np

from .types import BBox


def crop_bbox(image: np.ndarray, bbox: BBox, pad: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    b = BBox(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad).clipped(w, h)
    return image[int(b.y0):int(np.ceil(b.y1)), int(b.x0):int(np.ceil(b.x1))]
