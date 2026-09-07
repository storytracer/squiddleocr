"""The orchestrator: layout -> text lines -> kraken recognition -> tables -> DoclingDocument."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from docling_core.types.doc import DoclingDocument

from .crops import crop_bbox
from .detectors.base import TextDetector
from .document import DocumentBuilder, RegionContent
from .formulas.base import FormulaRecognizer
from .layout.base import LayoutAnalyzer
from .recognizers.base import Recognizer
from .recognizers.kraken import record_to_recognition
from .tables.base import TableRecognizer
from .types import BBox, Page, Region, TextLine


@dataclass
class Pipeline:
    """Any ``LayoutAnalyzer`` + any ``TextDetector`` + kraken's recogniser (+ optional ``TableRecognizer``).

    The detector runs once on the whole page; each line goes to the region it overlaps most, and
    lines outside every region (a page number the layout model missed, marginalia) become their own
    ``text`` regions so nothing is lost. With ``detect_per_region`` the detector runs on each region's
    crop instead. All lines of a page are recognised in one call (kraken batches them); the
    ``ocr_record``s are kept per region for the line-level exports. Regions whose label is in
    ``skip_labels`` are pictures without OCR. Table regions get their cell structure from ``tables``
    and are read cell by cell; formula regions are read as LaTeX by ``formulas`` instead of as text. With ``keep_line_order`` the lines stay in the detector's order (one
    row each) instead of being de-duplicated and sorted into visual rows.
    """

    recognizer: Recognizer
    detector: TextDetector
    layout: LayoutAnalyzer
    tables: TableRecognizer | None = None
    formulas: FormulaRecognizer | None = None
    skip_labels: frozenset[str] = frozenset({"picture", "chart"})
    detect_per_region: bool = False
    keep_line_order: bool = False   # trust the detector's line order (a segmenter that orders lines itself)
    min_cell_size: int = 6
    cell_pad: float = 0.15          # cell boxes are grown by this fraction of their height before reading

    def process_page(self, page: Page) -> list[RegionContent]:
        contents = [RegionContent(r) for r in self.layout.analyze(page)]
        self._detect(page, contents)
        self._recognize(page, contents)
        self._tables(page, contents)
        self._formulas(page, contents)
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
        if region.label == "formula" and self.formulas is not None:        # read as LaTeX
            return False
        return not (region.label == "table" and self.tables is not None)   # tables are read per cell

    def _detect(self, page: Page, contents: list[RegionContent]) -> None:
        ocr = [c for c in contents if self._wants_lines(c.region)]
        if self.detect_per_region:
            for c in ocr:
                c.lines = self.detector.detect(page, c.region)
                for ln in c.lines:
                    ln.region_id = c.region.id
        else:
            by_id = {c.region.id: c for c in ocr}
            orphans = []
            for ln in self.detector.detect(page):
                owner = _owner(ln, [c.region for c in ocr])
                if owner is not None:
                    ln.region_id = owner.id
                    by_id[owner.id].lines.append(ln)
                elif _owner(ln, [c.region for c in contents]) is None:   # not inside a picture either
                    orphans.append(ln)
            if orphans:
                contents.extend(self._orphan_regions(orphans))
                self._reorder(contents)
                ocr = [c for c in contents if self._wants_lines(c.region)]
        for c in ocr:
            if self.keep_line_order:
                for i, ln in enumerate(c.lines):
                    ln.row = i
            else:
                c.lines = order_lines(c.lines)

    @staticmethod
    def _orphan_regions(lines: list[TextLine]) -> list[RegionContent]:
        """One ``text`` region per visual row of lines that no layout region claimed."""
        out = []
        for i, row in enumerate(_rows(lines)):
            pts = np.concatenate([ln.polygon for ln in row])
            region = Region("text", BBox.of(pts).polygon, 1.0, None, f"orphan_{i}", raw_label="orphan")
            c = RegionContent(region)
            c.lines = list(row)
            for ln in c.lines:
                ln.region_id = region.id
            out.append(c)
        return out

    @staticmethod
    def _reorder(contents: list[RegionContent]) -> None:
        """Give orphan regions a place in the reading order without disturbing the layout model's.

        An orphan goes after the last ordered region that ends above its centre and overlaps it
        horizontally (falling back to any region whose centre is above it), so a page number at the
        top comes first and a missed line follows the paragraph above it. When no region carries
        an order the whole page is ordered with XY-cut.
        """
        from .layout.order import xy_cut_order

        ordered = sorted((c for c in contents if c.region.order is not None), key=lambda c: c.region.order)
        if not ordered:
            for rank, i in enumerate(xy_cut_order([c.region.bbox for c in contents])):
                contents[i].region.order = rank
            return
        orphans = sorted((c for c in contents if c.region.order is None), key=lambda c: (c.region.bbox.y0, c.region.bbox.x0))
        seq = list(ordered)
        for o in orphans:
            ob = o.region.bbox
            cy = (ob.y0 + ob.y1) / 2
            above = [i for i, c in enumerate(seq)
                     if c.region.bbox.y1 <= cy and min(ob.x1, c.region.bbox.x1) > max(ob.x0, c.region.bbox.x0)]
            if not above:
                above = [i for i, c in enumerate(seq) if (c.region.bbox.y0 + c.region.bbox.y1) / 2 < cy]
            seq.insert(max(above) + 1 if above else 0, o)
        for rank, c in enumerate(seq):
            c.region.order = rank

    def _recognize(self, page: Page, contents: list[RegionContent]) -> None:
        flat = [(c, ln) for c in contents for ln in c.lines]
        if not flat:
            return
        records = self.recognizer.recognize_lines(page, [ln for _, ln in flat])
        for (c, _), rec in zip(flat, records):
            c.records.append(rec)
            c.texts.append(record_to_recognition(rec))

    def _tables(self, page: Page, contents: list[RegionContent]) -> None:
        if self.tables is None:
            return
        for c in contents:
            if c.region.label != "table":
                continue
            c.table = self.tables.structure(page, c.region)
            self._read_cells(page, c)

    def _formulas(self, page: Page, contents: list[RegionContent]) -> None:
        if self.formulas is None:
            return
        targets = [c for c in contents if c.region.label == "formula"]
        if targets:
            for c, latex in zip(targets, self.formulas.recognize(page, [c.region for c in targets])):
                c.formula = latex

    def _read_cells(self, page: Page, c: RegionContent) -> None:
        """Detect lines inside every cell box (or take the whole cell when nothing is detected but
        there is ink) and recognise them in one batch; the records stay on the table region."""
        jobs: list[tuple[int, TextLine]] = []
        for i, cell in enumerate(c.table.cells):
            if cell.bbox is None or cell.bbox.width < self.min_cell_size or cell.bbox.height < self.min_cell_size:
                continue
            pad_y = self.cell_pad * cell.bbox.height
            pad_x = 2 * pad_y   # cell boxes tend to clip the first and last glyph
            box = BBox(cell.bbox.x0 - pad_x, cell.bbox.y0 - pad_y, cell.bbox.x1 + pad_x, cell.bbox.y1 + pad_y).clipped(page.width, page.height)
            lines = order_lines(self.detector.detect(page, Region("text", box.polygon, 1.0, None, "cell")))
            if not lines:
                crop = crop_bbox(page.image, box)
                if crop.size and crop.min() < 128:
                    lines = [TextLine(box.polygon, 1.0, None)]
            for ln in lines:
                ln.region_id = c.region.id
                jobs.append((i, ln))
        if not jobs:
            return
        records = self.recognizer.recognize_lines(page, [ln for _, ln in jobs])
        for (i, ln), rec in zip(jobs, records):
            c.lines.append(ln)
            c.records.append(rec)
            c.texts.append(record_to_recognition(rec))
            cell = c.table.cells[i]
            cell.text = f"{cell.text} {rec.prediction}".strip() if cell.text else rec.prediction


def dedupe_lines(lines: list[TextLine], max_containment: float = 0.7) -> list[TextLine]:
    """Drop a line box that lies mostly inside another (detectors sometimes emit a fragment twice)."""
    keep: list[TextLine] = []
    for ln in sorted(lines, key=lambda l: -(l.bbox.width * l.bbox.height)):
        area = max(ln.bbox.width * ln.bbox.height, 1e-6)
        if not any(ln.bbox.intersection_area(k.bbox) / area > max_containment for k in keep):
            keep.append(ln)
    return keep


def _rows(lines: list[TextLine], overlap: float = 0.5) -> list[list[TextLine]]:
    """Group lines into visual rows: a line joins the current row when it vertically overlaps the row's top line."""
    rows: list[list[TextLine]] = []
    by_top = sorted(lines, key=lambda ln: ln.bbox.y0)
    for ln in by_top:
        if rows:
            ref = min(rows[-1], key=lambda r: r.bbox.y0).bbox
            inter = min(ref.y1, ln.bbox.y1) - max(ref.y0, ln.bbox.y0)
            if inter >= overlap * min(ref.height, ln.bbox.height):
                rows[-1].append(ln)
                continue
        rows.append([ln])
    return [sorted(row, key=lambda r: r.bbox.x0) for row in rows]


def order_lines(lines: list[TextLine]) -> list[TextLine]:
    """De-duplicate, then order lines row by row (top to bottom, left to right) and record the row index."""
    out = []
    for i, row in enumerate(_rows(dedupe_lines(lines))):
        for ln in row:
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
