"""Cutting region images out of a page (line images are kraken's business)."""
from __future__ import annotations

import numpy as np

from .types import BBox


def crop_bbox(image: np.ndarray, bbox: BBox, pad: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    b = BBox(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad).clipped(w, h)
    return image[int(b.y0):int(np.ceil(b.y1)), int(b.x0):int(np.ceil(b.x1))]


def crop_polygon(image: np.ndarray, polygon: np.ndarray, pad: int = 0, fill: int = 255) -> tuple[np.ndarray, int, int]:
    """The padded bounding box of ``polygon`` with everything outside the polygon painted ``fill`` (white),
    so a segmenter run on the crop sees only this region. Returns the crop and its offset in the page."""
    import cv2

    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    b = BBox.of(poly)
    h, w = image.shape[:2]
    c = BBox(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad).clipped(w, h)
    x0, y0, x1, y1 = int(c.x0), int(c.y0), int(np.ceil(c.x1)), int(np.ceil(c.y1))
    crop = image[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return crop, x0, y0
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(poly - (x0, y0)).astype(np.int32)], 1)
    crop[mask == 0] = fill
    return crop, x0, y0
