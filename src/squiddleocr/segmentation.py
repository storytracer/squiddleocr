"""Bridge from SquiddleOCR's lines to kraken's containers (``kraken`` extra).

kraken's recogniser and serialiser both take a ``Segmentation`` of line records. This builds one
from ``TextLine``s: a ``BaselineLine`` (boundary polygon + baseline) when the detector gave a
baseline, else a ``BBoxLine`` with the axis-aligned bounds of the box. Line ids are
``<region id>_l<n>`` and each record names its region, which is how kraken's serialiser groups
lines into regions.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import BBox, Page, TextLine


def clip_points(poly: np.ndarray, page: Page) -> list[tuple[int, int]]:
    pts = np.rint(np.asarray(poly, dtype=float)).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, page.width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, page.height - 1)
    return [(int(x), int(y)) for x, y in pts]


def clip_box(b: BBox, page: Page) -> tuple[int, int, int, int]:
    (x0, y0), (x1, y1) = clip_points(np.array([[b.x0, b.y0], [b.x1, b.y1]]), page)
    return x0, y0, x1, y1


def line_record(ln: TextLine, line_id: str, region_id: str | None, page: Page, text_direction: str = "horizontal-lr"):
    """A kraken ``BaselineLine`` or ``BBoxLine`` for one ``TextLine``."""
    from kraken.containers import BaselineLine, BBoxLine

    regions = [region_id] if region_id else None
    if ln.baseline is not None:
        return BaselineLine(id=line_id, baseline=clip_points(ln.baseline, page), boundary=clip_points(ln.polygon, page),
                            regions=regions)
    return BBoxLine(id=line_id, bbox=clip_box(ln.bbox, page), regions=regions, text_direction=text_direction)


def lines_to_segmentation(page: Page, lines: Sequence[TextLine], text_direction: str = "horizontal-lr"):
    """A kraken ``Segmentation`` of ``lines`` in the given order (ids ``<region>_l<n>``, per region)."""
    from kraken.containers import Segmentation

    counters: dict[str, int] = {}
    records = []
    for ln in lines:
        rid = ln.region_id or "line"
        n = counters.get(rid, 0)
        counters[rid] = n + 1
        records.append(line_record(ln, f"{rid}_l{n}", ln.region_id or None, page, text_direction))
    seg_type = "baselines" if any(r.type == "baselines" for r in records) else "bbox"
    return Segmentation(type=seg_type, imagename=page.path.name if page.path else "", text_direction=text_direction,
                        script_detection=False, lines=records)
