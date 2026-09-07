"""kraken's blla baseline segmenter as a line detector (``kraken`` extra)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..crops import crop_bbox, line_image
from ..types import Page, Region, TextLine

HTRMOPO_BLLA = Path.home() / ".local/share/htrmopo/97665cf3-f83d-5594-8855-f28d3af9df7a/blla.mlmodel"


class KrakenSegmenter:
    """Baseline segmentation; lines come back as boundary polygons (with baselines) in page coordinates."""

    def __init__(self, model_path: str | Path | None = None, device: str = "auto", text_direction: str = "horizontal-lr"):
        from kraken.configs import SegmentationInferenceConfig
        from kraken.tasks.segmentation import SegmentationTaskModel

        path = Path(model_path) if model_path else HTRMOPO_BLLA
        if not path.is_file():
            raise FileNotFoundError(f"kraken segmentation model not found: {path} (kraken get 10.5281/zenodo.10592716)")
        self.model = SegmentationTaskModel.load_model(str(path))
        accelerator = "cpu" if device == "cpu" else "auto"
        self.text_direction = text_direction
        self.config = SegmentationInferenceConfig(accelerator=accelerator, text_direction=text_direction)

    def detect(self, page: Page, region: Region | None = None) -> list[TextLine]:
        if region is None:
            image, dx, dy, rid = page.image, 0, 0, ""
        else:
            b = region.bbox
            image, dx, dy, rid = crop_bbox(page.image, b), int(b.x0), int(b.y0), region.id
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

    def line_images(self, page: Page, lines: list[TextLine]) -> list[np.ndarray]:
        """Cut lines exactly as kraken does (``kraken.lib.segmentation.extract_polygons``: boundary
        mask, white background, baseline dewarping), so the recogniser sees what it was trained on.
        Lines without a baseline, or that kraken rejects (baseline under 5 px), fall back to
        ``crops.line_image``."""
        from kraken.containers import BaselineLine, Segmentation
        from kraken.lib.segmentation import extract_polygons

        im = Image.fromarray(page.image)
        out = []
        for i, ln in enumerate(lines):
            if ln.baseline is None:
                out.append(line_image(page, ln.polygon))
                continue
            rec = BaselineLine(id=f"line_{i}", baseline=[tuple(p) for p in ln.baseline.tolist()],
                               boundary=[tuple(p) for p in ln.polygon.tolist()])
            seg = Segmentation(type="baselines", imagename="", text_direction=self.text_direction,
                               script_detection=False, lines=[rec])
            try:
                line_im, _ = next(extract_polygons(im, seg))
            except (ValueError, StopIteration):
                out.append(line_image(page, ln.polygon))
                continue
            if line_im is None or 0 in line_im.size:
                out.append(line_image(page, ln.polygon))
                continue
            out.append(np.asarray(line_im.convert("RGB")))
        return out
