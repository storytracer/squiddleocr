from __future__ import annotations

from typing import Protocol

from ..types import Page, Region, TableResult


class TableRecognizer(Protocol):
    """Recovers the cell structure (rows, columns, spans, cell boxes) of a ``table`` region.

    Cells come back with empty ``text``; the pipeline reads each cell with the detector and the
    recogniser, so table text is SquiddleOCR's and no second OCR engine is involved.
    """

    def structure(self, page: Page, region: Region) -> TableResult: ...
