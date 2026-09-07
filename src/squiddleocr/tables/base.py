from __future__ import annotations

from typing import Protocol, Sequence

from ..types import Page, Region, TableResult, TextLine


class TableRecognizer(Protocol):
    """Recovers the cell structure (rows, columns, spans, cell boxes) of a ``table`` region and
    places the already recognised text lines of that region into the cells.

    ``lines`` are the region's text lines in reading order and ``texts`` their transcriptions, one
    per line; the recogniser fills each cell's ``text`` from them, so table text is the pipeline's
    own recognition and no second OCR engine is involved.
    """

    def structure(self, page: Page, region: Region, lines: Sequence[TextLine] = (), texts: Sequence[str] = ()) -> TableResult: ...
