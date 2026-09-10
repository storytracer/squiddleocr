"""Text lines from eynollah's PAGE-XML (``eynollah`` extra).

eynollah gives each line a polygon and no baseline. kraken's recogniser reads ``BaselineLine``s
along their baseline inside the boundary polygon (dewarping, the way ``--pipeline kraken`` reads
blla's lines), so a straight baseline is synthesised from the polygon: a least-squares line through
the vertices at or below the polygon's median y (its lower edge), spanning xmin..xmax, raised by
``descender`` x the line height (default 0.18) to sit above the descenders. With
``baselines=False`` the lines are plain boxes (``BBoxLine``). Line order is the XML's.
"""
from __future__ import annotations

import numpy as np

from ..eynollah.source import EynollahSource
from ..types import Page, Region, TextLine


def synthesize_baseline(polygon: np.ndarray, descender: float = 0.18) -> np.ndarray:
    """A two-point baseline for a line polygon (see the module docstring); y is clipped to the polygon's
    vertical extent."""
    p = np.asarray(polygon, dtype=float).reshape(-1, 2)
    xmin, xmax = p[:, 0].min(), p[:, 0].max()
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    lower = p[p[:, 1] >= np.median(p[:, 1])]
    if len(np.unique(lower[:, 0])) >= 2 and xmax > xmin:
        slope, intercept = np.polyfit(lower[:, 0], lower[:, 1], 1)
    else:
        slope, intercept = 0.0, float(lower[:, 1].mean())
    lift = descender * (ymax - ymin)
    ys = np.clip(np.array([xmin, xmax]) * slope + intercept - lift, ymin, ymax)
    return np.array([[xmin, ys[0]], [xmax, ys[1]]], dtype=float)


class EynollahLines:
    def __init__(self, source: EynollahSource, baselines: bool = True, descender: float = 0.18):
        self.source, self.baselines, self.descender = source, baselines, descender

    def detect(self, page: Page, region: Region | None = None) -> list[TextLine]:
        parsed = self.source.page(page)
        if region is None:
            polys = [(r.id, ln) for r in parsed.regions if r.label is not None for ln in r.lines]
        else:
            r = parsed.region(region.id)
            polys = [(region.id, ln) for ln in (r.lines if r is not None else [])]
        out = []
        for rid, poly in polys:
            base = synthesize_baseline(poly, self.descender) if self.baselines else None
            out.append(TextLine(np.asarray(poly, dtype=float), 1.0, base, rid))
        return out
