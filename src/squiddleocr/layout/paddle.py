"""Layout analysis with PaddleX's PP-DocLayout models (``paddle`` extra)."""
from __future__ import annotations

import numpy as np

from ..runtime import paddlex_device
from ..types import BBox, Page, Region
from .order import suppress_contained, xy_cut_order

#: PP-DocLayout_plus-L labels -> Docling labels. ``number`` is a page number and is placed by position.
LABEL_MAP = {
    "doc_title": "title",
    "paragraph_title": "section_header",
    "text": "text",
    "abstract": "text",
    "content": "text",
    "aside_text": "text",
    "algorithm": "code",
    "reference": "reference",
    "reference_content": "reference",
    "footnote": "footnote",
    "header": "page_header",
    "footer": "page_footer",
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "formula": "formula",
    "formula_number": "text",
    "table": "table",
    "image": "picture",
    "chart": "picture",
    "header_image": "picture",
    "footer_image": "picture",
    "seal": "seal",
}


#: PaddleX's PP-StructureV3 per-class score thresholds for PP-DocLayout_plus-L (class index -> threshold).
THRESHOLDS = {0: 0.3, 1: 0.5, 2: 0.4, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.3, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5,
              12: 0.5, 13: 0.5, 14: 0.5, 15: 0.45, 16: 0.5, 17: 0.5, 18: 0.5, 19: 0.5}


class PaddleLayout:
    """PP-DocLayout_plus-L (default) region detection with XY-cut reading order."""

    def __init__(self, model_name: str = "PP-DocLayout_plus-L", device: str = "auto",
                 threshold: float | dict[int, float] | None = None,
                 header_zone: float = 0.15, nms: bool = True, merge_mode: str = "large", max_containment: float = 0.8):
        from paddlex.inference import create_predictor

        self.model_name, self.header_zone = model_name, header_zone
        self.threshold = THRESHOLDS if threshold is None else threshold
        self.nms, self.merge_mode, self.max_containment = nms, merge_mode, max_containment
        self.predictor = create_predictor(model_name, engine="onnxruntime", device=paddlex_device(device))

    def analyze(self, page: Page) -> list[Region]:
        res = next(iter(self.predictor.predict(np.ascontiguousarray(page.image), threshold=self.threshold,
                                               layout_nms=self.nms, layout_merge_bboxes_mode=self.merge_mode)))
        regions = []
        for i, box in enumerate(res["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in box["coordinate"])
            raw = str(box["label"])
            label = LABEL_MAP.get(raw, "text")
            if raw == "number":
                label = "page_header" if (y0 + y1) / 2 < self.header_zone * page.height else "page_footer"
            regions.append(Region(label, BBox(x0, y0, x1, y1).polygon, float(box["score"]), None, f"{raw}_{i}",
                                  raw_label=raw))
        regions = suppress_contained(regions, self.max_containment)
        for rank, i in enumerate(xy_cut_order([r.bbox for r in regions])):
            regions[i].order = rank
        return regions
