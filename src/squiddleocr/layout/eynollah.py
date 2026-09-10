"""Layout analysis from eynollah's PAGE-XML (``eynollah`` extra; the analyser itself runs as a subprocess).

Regions come from the XML with Docling labels (``eynollah.pagexml``); the ones eynollah's
``ReadingOrder`` references keep its order, the rest (drop capitals, images, tables) are slotted in
by position (``place_unordered``). Separators are dropped.

Page numbers come from a second model when one is given (``numbers``, PP-DocLayoutV3 through
``PaddleLayout``): eynollah has no page-number class and draws a page number together with the
ornaments around it, PP-DocLayoutV3 has a ``number`` class with a tight box. An eynollah region that
is nothing but a number box (one row around it) is replaced by that box, label ``page_header`` or
``page_footer`` by position, reading-order place kept; a number box on which eynollah has no region is
added as a new region; a number box inside a larger eynollah region is left alone and counted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eynollah.pagexml import regions_from_page
from ..eynollah.source import EynollahSource
from ..types import BBox, Page, Region
from .order import place_unordered

HEADER_ZONE = 0.15      # a page number above this fraction of the page height is a header, else a footer


@dataclass
class NumberStats:
    replaced: int = 0
    added: int = 0
    left: int = 0

    def describe(self) -> str:
        return f"page numbers from PP-DocLayoutV3: {self.replaced} eynollah regions replaced, {self.added} added, {self.left} inside larger regions left"


class EynollahLayout:
    def __init__(self, source: EynollahSource, numbers=None):
        self.source = source
        self.numbers = numbers          # a LayoutAnalyzer whose regions carry raw_label "number" (PaddleLayout)
        self.number_stats = NumberStats()

    def prepare(self, paths) -> None:
        self.source.prepare(paths)

    def close(self) -> None:
        self.source.close()

    def analyze(self, page: Page) -> list[Region]:
        regions = regions_from_page(self.source.page(page))
        if self.numbers is not None and regions:
            regions = self.take_page_numbers(page, regions, self.numbers.analyze(page))
        if regions:
            place_unordered(regions)
        return sorted(regions, key=lambda r: r.order)

    def take_page_numbers(self, page: Page, regions: list[Region], oracle: list[Region]) -> list[Region]:
        """Apply the ``number`` boxes of ``oracle`` to ``regions`` (see the module docstring)."""
        out = list(regions)
        for box in (r for r in oracle if r.raw_label == "number"):
            b = box.bbox
            area = max(b.width * b.height, 1e-6)
            label = "page_header" if (b.y0 + b.y1) / 2 < HEADER_ZONE * page.height else "page_footer"
            holders = [r for r in out if r.label in ("text", "section_header", "page_header", "page_footer")
                       and r.bbox.intersection_area(b) / area >= 0.8]
            if not holders:
                out.append(Region(label, b.polygon, box.score, None, f"number_{len(out)}", raw_label="number"))
                self.number_stats.added += 1
                continue
            holder = min(holders, key=lambda r: r.bbox.width * r.bbox.height)
            if holder.bbox.height <= 1.6 * b.height:      # one row: the number and whatever surrounds it
                holder.polygon = b.polygon
                holder.label, holder.raw_label = label, holder.raw_label + "/number"
                self.number_stats.replaced += 1
            else:
                self.number_stats.left += 1
        return out
