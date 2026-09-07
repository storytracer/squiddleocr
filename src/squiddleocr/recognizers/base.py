from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..types import Page, TextLine, Recognition


class Recognizer(Protocol):
    """Reads text-line images (RGB uint8 arrays of any width) and returns one ``Recognition`` per line."""

    def recognize(self, lines: Sequence[np.ndarray]) -> list[Recognition]: ...


class LineRecognizer(Protocol):
    """Recognises lines on the page image itself and returns kraken ``ocr_record``s (text, character
    cuts, confidences), one per line in order. The pipeline prefers this over ``recognize`` when a
    recogniser offers it (the kraken level); the records reach the ALTO / PAGE / hOCR exports as they are."""

    def recognize_lines(self, page: Page, lines: Sequence[TextLine]) -> list: ...
