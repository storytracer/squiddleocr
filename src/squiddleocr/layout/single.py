from __future__ import annotations

from ..types import BBox, Page, Region


class SingleRegionLayout:
    """No layout analysis: the whole page is one text region (plain OCR)."""

    def analyze(self, page: Page) -> list[Region]:
        return [Region("text", BBox(0, 0, page.width, page.height).polygon, 1.0, 0, "page")]
