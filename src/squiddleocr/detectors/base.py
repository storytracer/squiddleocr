from __future__ import annotations

from typing import Protocol

from ..types import Page, Region, TextLine


class TextDetector(Protocol):
    """Finds text lines on a page, optionally restricted to one region; returns page coordinates.

    A detector may also define ``line_images(page, lines) -> list[np.ndarray]`` to cut the line
    images its lines should be recognised from (the kraken segmenter uses kraken's own polygon
    extraction); without it the pipeline uses ``crops.line_image``.
    """

    def detect(self, page: Page, region: Region | None = None) -> list[TextLine]: ...
