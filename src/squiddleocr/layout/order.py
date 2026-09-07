"""Reading order by recursive XY-cut over region boxes (top-to-bottom, then left-to-right)."""
from __future__ import annotations

from typing import Sequence

from ..types import BBox


def xy_cut_order(boxes: Sequence[BBox], min_gap: float = 1.0) -> list[int]:
    """Return the indices of ``boxes`` in reading order.

    Recursively splits the set at the largest whitespace gap: horizontally (a gap spanning the full
    width, top group first) whenever one exists, otherwise vertically (left group first); groups that
    cannot be split are ordered by their top edge, then left edge.
    """
    idx = list(range(len(boxes)))
    return _order(idx, boxes, min_gap)


def _order(idx: list[int], boxes: Sequence[BBox], min_gap: float) -> list[int]:
    if len(idx) <= 1:
        return idx
    for axis in ("y", "x"):
        groups = _split(idx, boxes, axis, min_gap)
        if len(groups) > 1:
            return [i for g in groups for i in _order(g, boxes, min_gap)]
    return sorted(idx, key=lambda i: (boxes[i].y0, boxes[i].x0))


def _split(idx: list[int], boxes: Sequence[BBox], axis: str, min_gap: float) -> list[list[int]]:
    lo = (lambda b: b.y0) if axis == "y" else (lambda b: b.x0)
    hi = (lambda b: b.y1) if axis == "y" else (lambda b: b.x1)
    ordered = sorted(idx, key=lambda i: lo(boxes[i]))
    groups, current, reach = [], [ordered[0]], hi(boxes[ordered[0]])
    for i in ordered[1:]:
        if lo(boxes[i]) - reach >= min_gap:
            groups.append(current)
            current, reach = [i], hi(boxes[i])
        else:
            current.append(i)
            reach = max(reach, hi(boxes[i]))
    groups.append(current)
    return groups
