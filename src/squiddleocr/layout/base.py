from __future__ import annotations

from typing import Protocol

from ..types import Page, Region


class LayoutAnalyzer(Protocol):
    """Splits a page into labelled regions in reading order (``Region.order`` set, 0-based).

    Labels use the Docling vocabulary where possible (``text``, ``section_header``, ``title``,
    ``caption``, ``footnote``, ``page_header``, ``page_footer``, ``table``, ``picture``,
    ``formula``, ``list_item``); analysers translate their own label sets in ``analyze``.
    """

    def analyze(self, page: Page) -> list[Region]: ...
