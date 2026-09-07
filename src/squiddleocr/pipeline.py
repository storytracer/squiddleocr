"""The orchestrator: layout -> text lines -> recognition -> tables -> DoclingDocument."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from docling_core.types.doc import DoclingDocument

from .crops import line_image
from .detectors.base import TextDetector
from .document import DocumentBuilder, RegionContent
from .layout.base import LayoutAnalyzer
from .recognizers.base import Recognizer
from .tables.base import TableRecognizer
from .types import Page, Region, TextLine


@dataclass
class Pipeline:
    """Any ``LayoutAnalyzer`` + any ``TextDetector`` + the recogniser (+ optional ``TableRecognizer``).

    Lines from all regions of a page are recognised in one call so the recogniser can batch them.
    Regions whose label is in ``skip_labels`` are kept as pictures without OCR. With
    ``detect_per_region`` the detector runs on each region's crop (tighter boxes, no bleed across
    regions); otherwise it runs once on the page and lines are assigned to the region they overlap most.
    """

    recognizer: Recognizer
    detector: TextDetector
    layout: LayoutAnalyzer
    tables: TableRecognizer | None = None
    skip_labels: frozenset[str] = frozenset({"picture", "chart"})
    detect_per_region: bool = True

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
    def _detect(self, page: Page, contents: list[RegionContent]) -> None:
        ocr = [c for c in contents if c.region.label not in self.skip_labels]
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
            c.lines.sort(key=lambda ln: (ln.bbox.y0, ln.bbox.x0))

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
            if c.region.label == "table":
                c.table = self.tables.recognize(page, c.region, c.lines, c.texts)


def _owner(line: TextLine, regions: Sequence[Region]) -> Region | None:
    lb = line.bbox
    best, best_area = None, 0.0
    for r in regions:
        a = lb.intersection_area(r.bbox)
        if a > best_area:
            best, best_area = r, a
    return best
