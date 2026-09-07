from __future__ import annotations

from typing import Protocol, Sequence

from ..types import Page, TextLine


class Recognizer(Protocol):
    """Reads ``lines`` off the page image and returns one kraken ``ocr_record`` per line, in order:
    text, per-character cuts and confidences. The records go into the DoclingDocument (text) and
    unchanged into kraken's serialiser (hOCR, ALTO, PAGE)."""

    def recognize_lines(self, page: Page, lines: Sequence[TextLine]) -> list: ...
