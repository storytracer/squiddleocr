"""Layout analysis with PaddleX's PP-DocLayout models (``paddle`` extra)."""
from __future__ import annotations

import numpy as np

from ..runtime import paddlex_device
from ..types import BBox, Page, Region
from .order import xy_cut_order

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


class PaddleLayout:
    """PP-DocLayout_plus-L (default) region detection with XY-cut reading order."""

    def __init__(self, model_name: str = "PP-DocLayout_plus-L", device: str = "auto", threshold: float = 0.5,
                 header_zone: float = 0.15):
        from paddlex.inference import create_predictor

        self.model_name, self.threshold, self.header_zone = model_name, threshold, header_zone
        self.predictor = create_predictor(model_name, engine="onnxruntime", device=paddlex_device(device))

    def analyze(self, page: Page) -> list[Region]:
        res = next(iter(self.predictor.predict(np.ascontiguousarray(page.image), threshold=self.threshold)))
        regions = []
        for i, box in enumerate(res["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in box["coordinate"])
            raw = str(box["label"])
            label = LABEL_MAP.get(raw, "text")
            if raw == "number":
                label = "page_header" if (y0 + y1) / 2 < self.header_zone * page.height else "page_footer"
            regions.append(Region(label, BBox(x0, y0, x1, y1).polygon, float(box["score"]), None, f"{raw}_{i}",
                                  raw_label=raw))
        for rank, i in enumerate(xy_cut_order([r.bbox for r in regions])):
            regions[i].order = rank
        return regions
