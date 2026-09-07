from __future__ import annotations

from typing import Protocol, Sequence

from ..types import Page, Region


class FormulaRecognizer(Protocol):
    """Reads display formulas: one LaTeX string per region, in the order given ("" when unreadable)."""

    def recognize(self, page: Page, regions: Sequence[Region]) -> list[str]: ...
