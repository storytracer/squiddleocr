from __future__ import annotations

from typing import Protocol

from ..types import Page, Region, TextLine


class TextDetector(Protocol):
    """Finds text lines on a page, optionally restricted to one region; returns page coordinates."""

    def detect(self, page: Page, region: Region | None = None) -> list[TextLine]: ...
