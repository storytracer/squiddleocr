from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..types import Recognition


class Recognizer(Protocol):
    """Reads text-line images (RGB uint8 arrays of any width) and returns one ``Recognition`` per line."""

    def recognize(self, lines: Sequence[np.ndarray]) -> list[Recognition]: ...
