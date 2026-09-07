"""Cutting line and region images out of a page."""
from __future__ import annotations

import cv2
import numpy as np

from .types import BBox, Page


def crop_bbox(image: np.ndarray, bbox: BBox, pad: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    b = BBox(bbox.x0 - pad, bbox.y0 - pad, bbox.x1 + pad, bbox.y1 + pad).clipped(w, h)
    return image[int(b.y0):int(np.ceil(b.y1)), int(b.x0):int(np.ceil(b.x1))]


def crop_quad(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Perspective-crop a 4-point (possibly rotated) box to an upright strip, as PaddleOCR does."""
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect).astype(np.float32)
        pts = _order_quad(pts)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    out = cv2.warpPerspective(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h / max(w, 1) >= 1.5:  # vertical text: rotate to horizontal
        out = np.rot90(out)
    return out


def crop_polygon(image: np.ndarray, polygon: np.ndarray, fill: int = 255) -> np.ndarray:
    """Crop the bounding box of ``polygon`` and paint everything outside the polygon ``fill`` (kraken style)."""
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    bbox = BBox.of(pts)
    sub = crop_bbox(image, bbox).copy()
    mask = np.zeros(sub.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts - np.array([int(bbox.x0), int(bbox.y0)])], 1)
    sub[mask == 0] = fill
    return sub


def line_image(page: Page, polygon: np.ndarray) -> np.ndarray:
    """Cut a text line: perspective crop for 4-point boxes, polygon mask for longer polygons."""
    pts = np.asarray(polygon).reshape(-1, 2)
    return crop_quad(page.image, pts) if len(pts) == 4 else crop_polygon(page.image, pts)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)
