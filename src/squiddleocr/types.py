"""Plain data types shared by all components. Coordinates are pixels in the page image, origin top-left."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def of(cls, polygon: np.ndarray) -> "BBox":
        p = np.asarray(polygon, dtype=float).reshape(-1, 2)
        return cls(float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def polygon(self) -> np.ndarray:
        return np.array([[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]], dtype=float)

    def intersection_area(self, other: "BBox") -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(w, 0.0) * max(h, 0.0)

    def clipped(self, width: float, height: float) -> "BBox":
        return BBox(max(self.x0, 0.0), max(self.y0, 0.0), min(self.x1, width), min(self.y1, height))


@dataclass
class Page:
    """A page image as an RGB uint8 array."""

    image: np.ndarray
    path: Path | None = None
    number: int = 1

    @classmethod
    def load(cls, path: str | Path, number: int = 1) -> "Page":
        from PIL import Image

        p = Path(path)
        with Image.open(p) as im:
            return cls(np.asarray(im.convert("RGB")), p, number)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class Region:
    """A layout region: a label from the analyser's own vocabulary, a polygon and a reading-order rank."""

    label: str
    polygon: np.ndarray                 # (N, 2)
    score: float = 1.0
    order: int | None = None            # reading order rank within the page, 0-based; None = unknown
    id: str = ""

    @property
    def bbox(self) -> BBox:
        return BBox.of(self.polygon)


@dataclass
class TextLine:
    """A text line found by a detector or segmenter, in page coordinates."""

    polygon: np.ndarray                 # (N, 2); 4 points for box detectors, more for polygon segmenters
    score: float = 1.0
    baseline: np.ndarray | None = None  # (M, 2) for baseline segmenters
    region_id: str = ""

    @property
    def bbox(self) -> BBox:
        return BBox.of(self.polygon)


@dataclass
class Recognition:
    text: str
    score: float


@dataclass
class TableCellResult:
    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    bbox: BBox | None = None
    header: bool = False


@dataclass
class TableResult:
    """A recognised table: its cells (already filled with text) and the source HTML if the recogniser made one."""

    region_id: str
    cells: list[TableCellResult] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    html: str | None = None
