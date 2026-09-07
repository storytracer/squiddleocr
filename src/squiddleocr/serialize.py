"""Line-level page exports (ALTO, PAGE-XML) through kraken's serialiser (``kraken`` extra).

The pipeline keeps every line's polygon, baseline (kraken segmenter) and text per region; the
Docling exports drop that to one text item per region. This module hands the same data to
``kraken.serialization.serialize`` so the XML is written by kraken's own templates, not by us.

What goes where: each region becomes a kraken ``Region`` tagged with its Docling label; each
recognised line becomes a baseline record (polygon + baseline) when the detector gave a
baseline, else a bbox record (the axis-aligned bounds of a four-point box); table cells become
bbox records inside the table region, one per cell. hOCR is not offered: kraken's hOCR template
only carries text through per-character cuts, which the recogniser does not produce.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .types import BBox, Page

if TYPE_CHECKING:
    from .document import RegionContent

#: export format name -> file suffix (kraken template name in ``TEMPLATES``)
PAGE_FORMATS = {"alto": ".alto.xml", "page": ".page.xml"}
TEMPLATES = {"alto": "alto", "page": "pagexml"}


def available() -> bool:
    try:
        import kraken.serialization  # noqa: F401
    except ImportError:
        return False
    return True


def _points(poly: np.ndarray, page: Page) -> list[tuple[int, int]]:
    pts = np.rint(np.asarray(poly, dtype=float)).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, page.width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, page.height - 1)
    return [(int(x), int(y)) for x, y in pts]


def _ibox(b: BBox, page: Page) -> tuple[int, int, int, int]:
    (x0, y0), (x1, y1) = _points(np.array([[b.x0, b.y0], [b.x1, b.y1]]), page)
    return x0, y0, x1, y1


def to_segmentation(page: Page, contents: Sequence["RegionContent"], text_direction: str = "horizontal-lr"):
    """Build a kraken ``Segmentation`` (regions + recognised line records) from one page's results."""
    from kraken.containers import BaselineLine, BaselineOCRRecord, BBoxLine, BBoxOCRRecord, Region, Segmentation

    regions: dict[str, list] = {}
    records = []
    ordered = sorted(contents, key=lambda c: (c.region.order if c.region.order is not None else 1 << 30))
    for c in ordered:
        r = c.region
        regions.setdefault(r.label, []).append(
            Region(id=r.id, boundary=_points(r.polygon, page), tags={"type": [{"type": r.label}]}))
        if c.table is not None:
            for cell in c.table.cells:
                if cell.bbox is None:
                    continue
                line = BBoxLine(id=f"{r.id}_r{cell.row}c{cell.col}", bbox=_ibox(cell.bbox, page), regions=[r.id],
                                text_direction=text_direction)
                records.append(BBoxOCRRecord(cell.text, [], [], line))
            continue
        texts = c.texts if len(c.texts) == len(c.lines) else [None] * len(c.lines)
        for i, (ln, rec) in enumerate(zip(c.lines, texts)):
            text = rec.text if rec is not None else ""
            lid = f"{r.id}_l{i}"
            if ln.baseline is not None:
                line = BaselineLine(id=lid, baseline=_points(ln.baseline, page), boundary=_points(ln.polygon, page),
                                    regions=[r.id])
                records.append(BaselineOCRRecord(text, [], [], line))
            else:
                line = BBoxLine(id=lid, bbox=_ibox(ln.bbox, page), regions=[r.id], text_direction=text_direction)
                records.append(BBoxOCRRecord(text, [], [], line))
    seg_type = "baselines" if any(rec.type == "baselines" for rec in records) else "bbox"
    return Segmentation(type=seg_type, imagename=page.path.name if page.path else "", text_direction=text_direction,
                        script_detection=False, lines=records, regions=regions)


def serialize_page(page: Page, contents: Sequence["RegionContent"], fmt: str, settings: dict | None = None) -> str:
    """Render one page as ``alto`` or ``page`` XML with kraken's templates.

    ``settings`` (recogniser, detector, layout, ...) are recorded as a processing step where the
    format has room for it (ALTO).
    """
    from kraken.containers import ProcessingStep
    from kraken.serialization import serialize

    if fmt not in TEMPLATES:
        raise ValueError(f"Unknown page export format {fmt!r}; choose from {tuple(TEMPLATES)}")
    steps = None
    if settings:
        steps = [ProcessingStep(id="squiddleocr", category="processing", description="SquiddleOCR layout, detection and recognition",
                                settings={k: v for k, v in settings.items() if isinstance(v, (str, int, float, bool))})]
    seg = to_segmentation(page, contents)
    return serialize(seg, image_size=(page.width, page.height), template=TEMPLATES[fmt], processing_steps=steps,
                     sub_line_segmentation=False)
