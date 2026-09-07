from __future__ import annotations

from typing import Protocol, Sequence

from ..types import Page, Recognition, Region, TableResult, TextLine


class TableRecognizer(Protocol):
    """Recovers the cell structure of a ``table`` region and fills the cells with the recognised lines."""

    def recognize(self, page: Page, region: Region, lines: Sequence[TextLine],
                  texts: Sequence[Recognition]) -> TableResult: ...
