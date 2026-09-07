"""Line-level page exports (ALTO, PAGE-XML, hOCR) through kraken's serialiser (``kraken`` extra).

The pipeline keeps every line's geometry and text per region; the Docling exports drop that to
one text item per region. This module hands the same data to ``kraken.serialization.serialize``
so the XML is written by kraken's own templates, not by us.

Kraken level: the pipeline already holds kraken ``ocr_record``s (text, character cuts,
confidences) per region; they are passed through unchanged and kraken renders words and glyphs
from the cuts, which is what hOCR needs. Paddle level: records are built from the lines and
texts (bbox records without cuts), so ALTO and PAGE carry line text and geometry only, and hOCR is
not available. Table cells become one bbox record per cell in both cases.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .segmentation import clip_box, clip_points, line_record
from .types import Page

if TYPE_CHECKING:
    from .document import RegionContent

#: export format name -> file suffix; kraken template name in ``TEMPLATES``
PAGE_FORMATS = {"alto": ".alto.xml", "page": ".page.xml", "hocr": ".hocr"}
TEMPLATES = {"alto": "alto", "page": "pagexml", "hocr": "hocr"}


def available() -> bool:
    try:
        import kraken.serialization  # noqa: F401
    except ImportError:
        return False
    return True


def _has_cuts(rec) -> bool:
    try:
        return len(rec.cuts) > 0
    except (AttributeError, TypeError):
        return False


def to_segmentation(page: Page, contents: Sequence["RegionContent"], text_direction: str = "horizontal-lr"):
    """A kraken ``Segmentation`` (regions + recognised line records) for one page, in reading order.

    Returns ``(segmentation, has_cuts)``; ``has_cuts`` is True when every record with text carries
    kraken's character cuts (kraken level), which lets the serialiser emit words and glyphs. kraken's
    serialiser cannot mix cut and cut-less records, so one cut-less text record disables words for the page.
    """
    from kraken.containers import BBoxLine, BBoxOCRRecord, BaselineOCRRecord, Region, Segmentation

    regions: dict[str, list] = {}
    records = []
    ordered = sorted(contents, key=lambda c: (c.region.order if c.region.order is not None else 1 << 30))
    for c in ordered:
        r = c.region
        regions.setdefault(r.label, []).append(
            Region(id=r.id, boundary=clip_points(r.polygon, page), tags={"type": [{"type": r.label}]}))
        if c.table is not None and not c.records:
            for cell in c.table.cells:
                if cell.bbox is None:
                    continue
                line = BBoxLine(id=f"{r.id}_r{cell.row}c{cell.col}", bbox=clip_box(cell.bbox, page), regions=[r.id],
                                text_direction=text_direction)
                records.append(BBoxOCRRecord(cell.text, [], [], line))
            continue
        if c.records and len(c.records) == len(c.lines):     # kraken level: kraken's own records
            records.extend(c.records)
            continue
        texts = c.texts if len(c.texts) == len(c.lines) else [None] * len(c.lines)
        for i, (ln, rec) in enumerate(zip(c.lines, texts)):
            text = rec.text if rec is not None else ""
            line = line_record(ln, f"{r.id}_l{i}", r.id, page, text_direction)
            records.append(BaselineOCRRecord(text, [], [], line) if line.type == "baselines"
                           else BBoxOCRRecord(text, [], [], line))
    seg_type = "baselines" if any(rec.type == "baselines" for rec in records) else "bbox"
    seg = Segmentation(type=seg_type, imagename=page.path.name if page.path else "", text_direction=text_direction,
                       script_detection=False, lines=records, regions=regions)
    with_text = [rec for rec in records if rec.prediction]
    return seg, bool(with_text) and all(_has_cuts(rec) for rec in with_text)


def serialize_page(page: Page, contents: Sequence["RegionContent"], fmt: str, settings: dict | None = None) -> str:
    """Render one page as ``alto``, ``page`` or ``hocr`` with kraken's templates.

    ``settings`` (recogniser, detector, layout, ...) are recorded as a processing step where the
    format has room for it (ALTO). ``hocr`` needs records with character cuts (kraken level).
    """
    from kraken.containers import ProcessingStep
    from kraken.serialization import serialize

    if fmt not in TEMPLATES:
        raise ValueError(f"Unknown page export format {fmt!r}; choose from {tuple(TEMPLATES)}")
    seg, has_cuts = to_segmentation(page, contents)
    if fmt == "hocr" and not has_cuts:
        raise ValueError("hOCR needs kraken-level recognition (--level kraken): kraken's hOCR template writes text "
                         "as words with boxes derived from character cuts, which the paddle level does not produce")
    steps = None
    if settings:
        steps = [ProcessingStep(id="squiddleocr", category="processing", description="SquiddleOCR layout, detection and recognition",
                                settings={k: v for k, v in settings.items() if isinstance(v, (str, int, float, bool))})]
    return serialize(seg, image_size=(page.width, page.height), template=TEMPLATES[fmt], processing_steps=steps,
                     sub_line_segmentation=has_cuts)
