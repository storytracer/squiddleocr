"""Line-level page exports (hOCR, ALTO, PAGE-XML) through kraken's serialiser.

The pipeline holds kraken ``ocr_record``s (text, character cuts, confidences) per layout region.
This module puts them into a kraken ``Segmentation`` together with the regions and lets
``kraken.serialization.serialize`` render kraken's own templates: lines with polygons or boxes,
baselines where the segmenter gave them, words and glyphs derived from the cuts. We write no XML.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .segmentation import clip_points
from .types import Page

if TYPE_CHECKING:
    from .document import RegionContent

#: export format name -> file suffix; kraken template name in ``TEMPLATES``
PAGE_FORMATS = {"hocr": ".hocr", "alto": ".alto.xml", "page": ".page.xml"}
TEMPLATES = {"hocr": "hocr", "alto": "alto", "page": "pagexml"}


def _has_cuts(rec) -> bool:
    try:
        return len(rec.cuts) > 0
    except (AttributeError, TypeError):
        return False


def to_segmentation(page: Page, contents: Sequence["RegionContent"], text_direction: str = "horizontal-lr"):
    """A kraken ``Segmentation`` of the page: one ``Region`` per layout region (tagged with its Docling
    label) and the records of every region in reading order.

    Returns ``(segmentation, has_cuts)``. kraken's serialiser derives words and glyphs from character
    cuts and cannot mix cut and cut-less records with text, so ``has_cuts`` is only True when every
    record with text carries cuts (a rejected line comes back from kraken empty and without cuts).
    """
    from kraken.containers import Region, Segmentation

    regions: dict[str, list] = {}
    records = []
    for c in sorted(contents, key=lambda c: (c.region.order if c.region.order is not None else 1 << 30)):
        r = c.region
        regions.setdefault(r.label, []).append(
            Region(id=r.id, boundary=clip_points(r.polygon, page), tags={"type": [{"type": r.label}]}))
        records.extend(c.records)
    seg_type = "baselines" if any(rec.type == "baselines" for rec in records) else "bbox"
    seg = Segmentation(type=seg_type, imagename=page.path.name if page.path else "", text_direction=text_direction,
                       script_detection=False, lines=records, regions=regions)
    with_text = [rec for rec in records if rec.prediction]
    return seg, bool(with_text) and all(_has_cuts(rec) for rec in with_text)


def serialize_page(page: Page, contents: Sequence["RegionContent"], fmt: str, settings: dict | None = None) -> str:
    """Render one page as ``hocr``, ``alto`` or ``page`` with kraken's templates. ``settings``
    (recogniser, segmentation, layout, ...) become a processing step where the format records one (ALTO)."""
    from kraken.containers import ProcessingStep
    from kraken.serialization import serialize

    if fmt not in TEMPLATES:
        raise ValueError(f"Unknown page export format {fmt!r}; choose from {tuple(TEMPLATES)}")
    seg, has_cuts = to_segmentation(page, contents)
    steps = None
    if settings:
        steps = [ProcessingStep(id="squiddleocr", category="processing", description="SquiddleOCR layout, segmentation and recognition",
                                settings={k: v for k, v in settings.items() if isinstance(v, (str, int, float, bool))})]
    return serialize(seg, image_size=(page.width, page.height), template=TEMPLATES[fmt], processing_steps=steps,
                     sub_line_segmentation=has_cuts)
