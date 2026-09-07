"""The orchestrator: layout -> text lines -> recognition -> tables -> DoclingDocument."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from docling_core.types.doc import DoclingDocument

from .crops import crop_bbox, line_image
from .detectors.base import TextDetector
from .document import DocumentBuilder, RegionContent
from .layout.base import LayoutAnalyzer
from .recognizers.base import Recognizer
from .tables.base import TableRecognizer
from .types import BBox, Page, Region, TableCellResult, TextLine


@dataclass
class Pipeline:
    """Any ``LayoutAnalyzer`` + any ``TextDetector`` + the recogniser (+ optional ``TableRecognizer``).

    Lines from all regions of a page are recognised in one call so the recogniser can batch them.
    Regions whose label is in ``skip_labels`` are kept as pictures without OCR. With
    ``detect_per_region`` the detector runs on each region's crop (tighter boxes, no bleed across
    regions); otherwise it runs once on the page and lines are assigned to the region they overlap most.
    Table regions get their cell structure from ``tables`` and are then read cell by cell.
    """

    recognizer: Recognizer
    detector: TextDetector
    layout: LayoutAnalyzer
    tables: TableRecognizer | None = None
    skip_labels: frozenset[str] = frozenset({"picture", "chart"})
    detect_per_region: bool = True
    min_cell_size: int = 6
    cell_pad: float = 0.15          # cell boxes are grown by this fraction of their height before reading

    def process_page(self, page: Page) -> list[RegionContent]:
        contents = [RegionContent(r) for r in self.layout.analyze(page)]
        self._detect(page, contents)
        self._recognize(page, contents)
        self._tables(page, contents)
        return contents

    def run(self, pages: Iterable[Page], name: str = "document") -> DoclingDocument:
        builder = DocumentBuilder(name)
        for page in pages:
            builder.add_page(page, self.process_page(page))
        return builder.build()

    def run_files(self, paths: Sequence[str | Path], name: str | None = None) -> DoclingDocument:
        paths = [Path(p) for p in paths]
        return self.run((Page.load(p, number=i + 1) for i, p in enumerate(paths)), name or paths[0].stem)

    # ------------------------------------------------------------ stages
    def _wants_lines(self, region: Region) -> bool:
        if region.label in self.skip_labels:
            return False
        return not (region.label == "table" and self.tables is not None)   # tables are read per cell

    def _detect(self, page: Page, contents: list[RegionContent]) -> None:
        ocr = [c for c in contents if self._wants_lines(c.region)]
        if self.detect_per_region:
            for c in ocr:
                c.lines = self.detector.detect(page, c.region)
        else:
            by_id = {c.region.id: c for c in ocr}
            for ln in self.detector.detect(page):
                owner = _owner(ln, [c.region for c in ocr])
                if owner is not None:
                    by_id[owner.id].lines.append(ln)
        for c in ocr:
            c.lines = order_lines(c.lines)

    def _recognize(self, page: Page, contents: list[RegionContent]) -> None:
        flat = [(c, ln) for c in contents for ln in c.lines]
        if not flat:
            return
        results = self.recognizer.recognize([line_image(page, ln.polygon) for _, ln in flat])
        for (c, _), r in zip(flat, results):
            c.texts.append(r)

    def _tables(self, page: Page, contents: list[RegionContent]) -> None:
        if self.tables is None:
            return
        for c in contents:
            if c.region.label != "table":
                continue
            c.table = self.tables.structure(page, c.region)
            self._read_cells(page, c.table.cells)

    def _read_cells(self, page: Page, cells: list[TableCellResult]) -> None:
        """Detect lines inside every cell box and recognise them in one batch."""
        jobs: list[tuple[TableCellResult, np.ndarray]] = []
        for cell in cells:
            if cell.bbox is None or cell.bbox.width < self.min_cell_size or cell.bbox.height < self.min_cell_size:
                continue
            pad_y = self.cell_pad * cell.bbox.height
            pad_x = 2 * pad_y   # cell boxes tend to clip the first and last glyph
            box = BBox(cell.bbox.x0 - pad_x, cell.bbox.y0 - pad_y, cell.bbox.x1 + pad_x, cell.bbox.y1 + pad_y).clipped(page.width, page.height)
            lines = order_lines(self.detector.detect(page, Region("text", box.polygon, 1.0, None, "cell")))
            if not lines:  # nothing detected: read the whole cell if it has ink
                crop = crop_bbox(page.image, box)
                if crop.size and crop.min() < 128:
                    jobs.append((cell, crop))
                continue
            for ln in lines:
                jobs.append((cell, line_image(page, ln.polygon)))
        if not jobs:
            return
        texts = self.recognizer.recognize([im for _, im in jobs])
        for (cell, _), r in zip(jobs, texts):
            cell.text = f"{cell.text} {r.text}".strip() if cell.text else r.text


def order_lines(lines: list[TextLine], overlap: float = 0.5) -> list[TextLine]:
    """Sort lines into visual rows (boxes whose vertical extents overlap) and left-to-right within a row."""
    if not lines:
        return lines
    by_top = sorted(lines, key=lambda ln: ln.bbox.y0)
    rows: list[list[TextLine]] = [[by_top[0]]]
    for ln in by_top[1:]:
        row = rows[-1]
        ref = min(row, key=lambda r: r.bbox.y0).bbox
        inter = min(ref.y1, ln.bbox.y1) - max(ref.y0, ln.bbox.y0)
        if inter >= overlap * min(ref.height, ln.bbox.height):
            row.append(ln)
        else:
            rows.append([ln])
    out = []
    for i, row in enumerate(rows):
        for ln in sorted(row, key=lambda r: r.bbox.x0):
            ln.row = i
            out.append(ln)
    return out


def _owner(line: TextLine, regions: Sequence[Region]) -> Region | None:
    lb: BBox = line.bbox
    best, best_area = None, 0.0
    for r in regions:
        a = lb.intersection_area(r.bbox)
        if a > best_area:
            best, best_area = r, a
    return best
