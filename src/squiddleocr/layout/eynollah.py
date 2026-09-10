"""Layout analysis from eynollah's PAGE-XML (``eynollah`` extra; the analyser itself runs as a subprocess).

Regions come from the XML with Docling labels (``eynollah.pagexml``); the ones eynollah's
``ReadingOrder`` references keep its order, the rest (drop capitals, images, tables) are slotted in
by position (``place_unordered``). Separators are dropped.
"""
from __future__ import annotations

from ..eynollah.pagexml import regions_from_page
from ..eynollah.source import EynollahSource
from ..types import Page, Region
from .order import place_unordered


class EynollahLayout:
    def __init__(self, source: EynollahSource):
        self.source = source

    def prepare(self, paths) -> None:
        self.source.prepare(paths)

    def close(self) -> None:
        self.source.close()

    def analyze(self, page: Page) -> list[Region]:
        regions = regions_from_page(self.source.page(page))
        if regions:
            place_unordered(regions)
        return sorted(regions, key=lambda r: r.order)
