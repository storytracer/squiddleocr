"""Reading eynollah's PAGE-XML: regions with labels, the reading order and each region's text lines.

eynollah writes ``TextRegion`` (``type`` paragraph, heading, drop-capital or marginalia),
``ImageRegion``, ``SeparatorRegion`` and ``TableRegion``; a ``ReadingOrder`` that references the
marginalia and the text regions (headings included) but not drop capitals, images, separators or
tables; and ``TextLine``s with ``Coords`` only, top to bottom within their region. Namespace and
version of the schema are ignored.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..types import Region

#: PAGE region element -> Docling label (``TextRegion`` by its ``type`` attribute; absent = paragraph).
TEXT_TYPE_MAP = {
    "paragraph": "text",
    "": "text",
    "marginalia": "text",
    "drop-capital": "text",
    "heading": "section_header",
    "header": "page_header",
    "caption": "caption",
    "footnote": "footnote",
    "page-number": "page_header",
}
REGION_MAP = {
    "ImageRegion": "picture",
    "GraphicRegion": "picture",
    "TableRegion": "table",
    "MathsRegion": "formula",
}
DROPPED = {"SeparatorRegion", "NoiseRegion", "UnknownRegion", "Border", "PrintSpace"}


@dataclass
class EynollahRegion:
    id: str
    kind: str                        # PAGE element name
    type: str                        # TextRegion type attribute, "" for others
    polygon: np.ndarray
    lines: list[np.ndarray] = field(default_factory=list)
    order: int | None = None
    conf: float = 1.0

    @property
    def label(self) -> str | None:
        if self.kind == "TextRegion":
            return TEXT_TYPE_MAP.get(self.type, "text")
        return REGION_MAP.get(self.kind)


@dataclass
class EynollahPage:
    path: Path
    width: int
    height: int
    regions: list[EynollahRegion]

    def region(self, region_id: str) -> EynollahRegion | None:
        return next((r for r in self.regions if r.id == region_id), None)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_points(text: str | None) -> np.ndarray:
    pts = [tuple(float(v) for v in p.split(",")) for p in (text or "").split() if "," in p]
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def parse_page_xml(path: str | Path) -> EynollahPage:
    path = Path(path)
    root = ET.parse(path).getroot()
    page = next((el for el in root.iter() if _local(el.tag) == "Page"), None)
    if page is None:
        raise ValueError(f"{path}: no Page element")
    width, height = int(page.get("imageWidth", 0)), int(page.get("imageHeight", 0))
    order: dict[str, int] = {}
    for el in page.iter():
        if _local(el.tag) in ("RegionRefIndexed", "RegionRef") and el.get("regionRef"):
            order.setdefault(el.get("regionRef"), len(order))
    regions: list[EynollahRegion] = []
    for el in page:
        kind = _local(el.tag)
        if kind in DROPPED or not kind.endswith("Region"):
            continue
        coords = next((c for c in el if _local(c.tag) == "Coords"), None)
        polygon = parse_points(coords.get("points") if coords is not None else "")
        if len(polygon) < 3:
            continue
        conf = coords.get("conf") if coords is not None else None
        region = EynollahRegion(el.get("id", f"region_{len(regions)}"), kind, el.get("type", "") or "", polygon,
                                order=order.get(el.get("id", "")), conf=float(conf) if conf else 1.0)
        for line in el:
            if _local(line.tag) != "TextLine":
                continue
            lc = next((c for c in line if _local(c.tag) == "Coords"), None)
            pts = parse_points(lc.get("points") if lc is not None else "")
            if len(pts) >= 3:
                region.lines.append(pts)
        regions.append(region)
    return EynollahPage(path, width, height, regions)


def regions_from_page(page: EynollahPage) -> list[Region]:
    """``Region``s with Docling labels for the regions SquiddleOCR keeps (separators and noise are
    dropped); ``order`` is the rank in eynollah's ``ReadingOrder`` for the regions it references, else
    None."""
    kept = [r for r in page.regions if r.label is not None]
    ranked = sorted((r.order for r in kept if r.order is not None))
    rank = {o: i for i, o in enumerate(ranked)}
    return [Region(r.label, r.polygon.copy(), r.conf, rank[r.order] if r.order is not None else None, r.id,
                   raw_label=f"{r.kind}/{r.type}" if r.type else r.kind)
            for r in kept]
