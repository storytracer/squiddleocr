"""kraken's blla baseline segmenter as a line detector (``kraken`` extra)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..crops import crop_bbox, crop_polygon
from ..models import KRAKEN_SEGMENTER_DOI, kraken_model_file
from ..types import Page, Region, TextLine


class KrakenSegmenter:
    """Baseline segmentation; lines come back as boundary polygons (with baselines) in page coordinates.

    Given a region, blla runs on the region's crop: its bounding box with ``pad`` pixels around it and,
    with ``mask``, everything outside the region polygon painted white, so lines of a neighbouring
    column cannot leak in (a layout analyser's regions in front of blla: kraken's own advice for
    complex pages such as newspapers, where blla on the whole page merges lines across columns)."""

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", text_direction: str = "horizontal-lr",
                 pad: int = 0, mask: bool = False):
        from kraken.configs import SegmentationInferenceConfig
        from kraken.tasks.segmentation import SegmentationTaskModel

        path = Path(model_path) if model_path else kraken_model_file(KRAKEN_SEGMENTER_DOI, (".mlmodel", ".safetensors"))
        if not path.is_file():
            raise FileNotFoundError(f"kraken segmentation model not found: {path}")
        self.model = SegmentationTaskModel.load_model(str(path))
        accelerator = "cpu" if device == "cpu" else "auto"
        self.text_direction = text_direction
        self.pad, self.mask = pad, mask
        self.config = SegmentationInferenceConfig(accelerator=accelerator, text_direction=text_direction)

    def detect(self, page: Page, region: Region | None = None) -> list[TextLine]:
        if region is None:
            image, dx, dy, rid = page.image, 0, 0, ""
        elif self.mask:
            image, dx, dy = crop_polygon(page.image, region.polygon, self.pad)
            rid = region.id
            if image.size == 0:
                return []
        else:
            b = region.bbox
            image = crop_bbox(page.image, b, self.pad)
            dx, dy, rid = int(max(b.x0 - self.pad, 0)), int(max(b.y0 - self.pad, 0)), region.id
            if image.size == 0:
                return []
        seg = self.model.predict(im=Image.fromarray(image), config=self.config)
        lines = []
        for line in seg.lines:
            if not line.boundary:
                continue
            poly = np.asarray(line.boundary, dtype=float) + (dx, dy)
            base = np.asarray(line.baseline, dtype=float) + (dx, dy) if line.baseline else None
            lines.append(TextLine(poly, 1.0, base, rid))
        return lines
