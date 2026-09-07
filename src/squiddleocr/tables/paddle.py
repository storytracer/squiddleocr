"""Table structure with PaddleX's table pipeline and the pipeline's own text (``paddle`` extra).

PaddleX's ``table_recognition_v2`` pipeline is what PP-StructureV3 runs on a table: a wired/wireless
classifier, SLANeXt / SLANet structure tokens, an RT-DETR cell detector and a matching step that
places OCR boxes into the cells. Its ``predict`` accepts an external OCR result, so it gets the
text lines SquiddleOCR detected on the page and kraken's transcriptions instead of running its own
OCR; the HTML it returns carries our text in its cells. The table crop is grown by ``crop_pad``
pixels: with the layout box exactly, the cell detector can flip to a worse cell set (2026-09-07,
NOTES), from 8 px on the result is stable.
"""
from __future__ import annotations

import html
import re
from typing import Sequence

import numpy as np

from ..crops import crop_bbox
from ..paddle_compat import create_pipeline, paddle_image
from ..types import BBox, Page, Region, TableCellResult, TableResult, TextLine

_ATTR = re.compile(r'(colspan|rowspan)="?(\d+)"?')
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t([dh])([^>]*)>(.*?)</t[dh]>", re.S)


def table_pipeline_config(wired: str = "SLANeXt_wired", wireless: str = "SLANet_plus", classifier: str = "PP-LCNet_x1_0_table_cls",
                          wired_cells: str = "RT-DETR-L_wired_table_cell_det", wireless_cells: str = "RT-DETR-L_wireless_table_cell_det") -> dict:
    """PaddleX's ``table_recognition_v2`` config without layout, preprocessing or its own OCR
    (PP-StructureV3's model choice: SLANeXt for wired, SLANet_plus for wireless tables)."""
    mod = lambda module, name: {"module_name": module, "model_name": name, "model_dir": None}  # noqa: E731
    return {"pipeline_name": "table_recognition_v2", "use_doc_preprocessor": False, "use_layout_detection": False, "use_ocr_model": False,
            "SubModules": {"TableClassification": mod("table_classification", classifier),
                           "WiredTableStructureRecognition": mod("table_structure_recognition", wired),
                           "WirelessTableStructureRecognition": mod("table_structure_recognition", wireless),
                           "WiredTableCellsDetection": mod("table_cells_detection", wired_cells),
                           "WirelessTableCellsDetection": mod("table_cells_detection", wireless_cells),
                           "TableOrientationClassify": mod("doc_text_orientation", "PP-LCNet_x1_0_doc_ori")}}


def parse_structure(tokens: Sequence[str]) -> list[tuple[int, int, int, int]]:
    """Turn a structure model's token list into ``(row, col, row_span, col_span)`` per cell, in cell order."""
    cells = []
    i, row = 0, -1
    while i < len(tokens):
        t = tokens[i]
        if t == "<tr>":
            row += 1
            cells.append(("row", None))
        elif t.startswith("<td"):
            attrs = ""
            if t in ("<td", "<td "):  # attributes follow as separate tokens until '>'
                i += 1
                while i < len(tokens) and tokens[i] != ">":
                    attrs += tokens[i]
                    i += 1
            cells.append(("cell", attrs + t))
        i += 1
    return [c for c, _ in _place([(kind, attrs, "") for kind, attrs in cells])]


def _place(items: Sequence[tuple[str, str, str]]) -> list[tuple[tuple[int, int, int, int], str]]:
    """Grid placement: ``items`` are ("row", _, _) markers and ("cell", attrs, text); returns
    ``((row, col, row_span, col_span), text)`` per cell, spans occupying the grid like HTML does."""
    out, grid, row, col = [], set(), -1, 0
    for kind, attrs, text in items:
        if kind == "row":
            row, col = row + 1, 0
            continue
        spans = dict(_ATTR.findall(attrs))
        rs, cs = int(spans.get("rowspan", 1)), int(spans.get("colspan", 1))
        while (row, col) in grid:
            col += 1
        for r in range(row, row + rs):
            for c in range(col, col + cs):
                grid.add((r, c))
        out.append(((row, col, rs, cs), text))
        col += cs
    return out


def cells_from_html(table_html: str) -> list[TableCellResult]:
    """``TableCellResult``s (row, col, spans, text, header flag) from a ``<table>`` HTML string."""
    items: list[tuple[str, str, str]] = []
    headers: list[bool] = []
    for row in _ROW.findall(table_html):
        items.append(("row", "", ""))
        for tag, attrs, inner in _CELL.findall(row):
            text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", inner)).split())
            items.append(("cell", attrs, text))
            headers.append(tag == "h")
    return [TableCellResult(text, r, c, rs, cs, None, header=h) for ((r, c, rs, cs), text), h in zip(_place(items), headers)]


class PaddleTableRecognizer:
    """PaddleX's table pipeline on ONNX Runtime, fed with the page's own lines and kraken's text."""

    def __init__(self, device: str = "auto", crop_pad: int = 8, config: dict | None = None):
        self.crop_pad = crop_pad
        self.pipeline = create_pipeline(config or table_pipeline_config(), device)

    def structure(self, page: Page, region: Region, lines: Sequence[TextLine] = (), texts: Sequence[str] = ()) -> TableResult:
        from paddlex.inference.pipelines.ocr.result import OCRResult

        b = BBox(region.bbox.x0 - self.crop_pad, region.bbox.y0 - self.crop_pad,
                 region.bbox.x1 + self.crop_pad, region.bbox.y1 + self.crop_pad).clipped(page.width, page.height)
        dx, dy = int(b.x0), int(b.y0)
        crop = paddle_image(crop_bbox(page.image, b))
        if crop.size == 0:
            return TableResult(region.id)
        texts = [str(t) for t in texts]
        polys = np.array([(ln.bbox.polygon - (dx, dy)).astype(int) for ln in lines]).reshape(-1, 4, 2)
        boxes = np.array([[int(ln.bbox.x0) - dx, int(ln.bbox.y0) - dy, int(ln.bbox.x1) - dx, int(ln.bbox.y1) - dy] for ln in lines]).reshape(-1, 4)
        ocr = OCRResult({"input_path": None, "page_index": None, "input_img": crop, "doc_preprocessor_res": {"output_img": crop},
                         "dt_polys": polys, "rec_polys": polys, "rec_boxes": boxes, "rec_texts": texts,
                         "rec_scores": [1.0] * len(texts), "rec_labels": ["text"] * len(texts),
                         "model_settings": {"use_doc_preprocessor": False, "use_textline_orientation": False},
                         "text_det_params": {}, "text_type": "general", "textline_orientation_angles": [-1] * len(texts)})
        res = next(iter(self.pipeline.predict(crop, use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=False,
                                              use_ocr_model=False, overall_ocr_res=ocr, use_ocr_results_with_table_cells=False,
                                              use_table_orientation_classify=False)))
        tables = res.get("table_res_list") or []
        if not tables:
            return TableResult(region.id)
        t = tables[0]
        pred_html = str(t.get("pred_html", ""))
        cells = cells_from_html(pred_html)
        cell_boxes = [np.asarray(cb, dtype=float).reshape(-1, 2) for cb in (t.get("cell_box_list") or [])]
        if len(cell_boxes) == len(cells):
            for cell, cb in zip(cells, cell_boxes):
                cell.bbox = BBox.of(cb + np.array([dx, dy]))
        num_rows = max((c.row + c.row_span for c in cells), default=0)
        num_cols = max((c.col + c.col_span for c in cells), default=0)
        return TableResult(region.id, cells, num_rows, num_cols, html=pred_html)
