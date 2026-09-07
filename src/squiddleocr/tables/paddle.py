"""Table structure recognition with PaddleX's SLANeXt / SLANet models (``paddle`` extra).

The structure model returns an HTML token sequence and one box per cell; the pipeline reads the
cells afterwards.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from ..crops import crop_bbox
from ..paddle_compat import create_predictor
from ..types import BBox, Page, Region, TableCellResult, TableResult

_ATTR = re.compile(r'(colspan|rowspan)="(\d+)"')


def parse_structure(tokens: Sequence[str]) -> list[tuple[int, int, int, int]]:
    """Turn the model's token list into ``(row, col, row_span, col_span)`` per cell, in cell order."""
    cells, grid, row, col = [], set(), -1, 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "<tr>":
            row, col = row + 1, 0
        elif t.startswith("<td"):
            attrs = ""
            if t in ("<td", "<td "):  # attributes follow as separate tokens until '>'
                i += 1
                while i < len(tokens) and tokens[i] != ">":
                    attrs += tokens[i]
                    i += 1
            spans = dict(_ATTR.findall(attrs + t))
            rs, cs = int(spans.get("rowspan", 1)), int(spans.get("colspan", 1))
            while (row, col) in grid:
                col += 1
            for r in range(row, row + rs):
                for c in range(col, col + cs):
                    grid.add((r, c))
            cells.append((row, col, rs, cs))
            col += cs
        i += 1
    return cells


def cell_boxes(raw_boxes, crop_shape: tuple[int, ...], offset: tuple[float, float]) -> list[BBox]:
    """Cell boxes in page coordinates.

    SLANeXt reports its boxes scaled by the crop *width* on both axes (a PaddleX post-processing
    quirk that PP-StructureV3 never hits because it takes cell boxes from a separate detector), so
    when the boxes overshoot the crop height the y axis is rescaled by height/width.
    """
    h, w = crop_shape[:2]
    boxes = [np.asarray(raw, dtype=float).reshape(-1, 2) for raw in raw_boxes]
    if boxes and max(float(b[:, 1].max()) for b in boxes) > 1.05 * h and w > 0:
        boxes = [b * np.array([1.0, h / w]) for b in boxes]
    return [BBox.of(b + np.array(offset)) for b in boxes]


class PaddleTableRecognizer:
    """SLANet_plus by default for every table: its cell boxes are accurate enough to read cells from.

    Pass ``wired_model="SLANeXt_wired"`` with the classifier to use PaddleX's wired/wireless split;
    SLANeXt's own cell boxes are imprecise (PaddleX pairs it with a separate cell detector), so
    expect clipped cells until that detector is integrated.
    """

    def __init__(self, device: str = "auto", wired_model: str = "SLANet_plus", wireless_model: str | None = None,
                 classifier: str | None = None):
        self.wired = create_predictor(wired_model, device)
        self.wireless = create_predictor(wireless_model, device) if wireless_model else None
        self.classifier = create_predictor(classifier, device) if classifier and self.wireless else None

    def _structure_model(self, crop: np.ndarray):
        if self.classifier is None:
            return self.wired
        res = next(iter(self.classifier.predict(crop)))
        label = res["label_names"][0] if res.get("label_names") else "wired_table"
        return self.wireless if label == "wireless_table" else self.wired

    def structure(self, page: Page, region: Region) -> TableResult:
        b = region.bbox
        crop = np.ascontiguousarray(crop_bbox(page.image, b))
        res = next(iter(self._structure_model(crop).predict(crop)))
        tokens = list(res["structure"])
        boxes = cell_boxes(res["bbox"], crop.shape, (int(b.x0), int(b.y0)))
        cells = [TableCellResult("", row, col, rs, cs, cb, header=row == 0)
                 for (row, col, rs, cs), cb in zip(parse_structure(tokens), boxes)]
        num_rows = max((c.row + c.row_span for c in cells), default=0)
        num_cols = max((c.col + c.col_span for c in cells), default=0)
        return TableResult(region.id, cells, num_rows, num_cols, html="".join(tokens))
