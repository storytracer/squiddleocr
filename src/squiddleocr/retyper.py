"""Retyper: what the typesetter's sizes and initials say about a page's regions (the page-level half of
the retyper layer; ``reflow`` is the row-level half).

Runs on a page's ``RegionContent``s after recognition and before the document is built. All rules are
ratios to the page's own body type, measured from the row heights the line stage delivered, so they
hold across scan resolutions and scripts:

- **Type-size ladder** (``heading_level``): the body size is the median row height of the page's text
  regions. A heading region whose rows are body-sized (below ``DEMOTE``, 1.2×) is a kicker, a
  sub-head or an advertisement line and becomes text; otherwise its level follows the ratio:
  ``LEVELS`` = 2.5× and above level 1, 1.7× level 2, else level 3 (Markdown ``##``, ``###``,
  ``####``; Docling's title stays ``#``).
- **Headline split** (``split_leading_heading``): a text region whose first one to three rows are at
  least ``SPLIT`` (1.6×) taller than the rest of its rows carries a headline the layout analyser
  merged into the body; those rows become a heading region of their own, placed just before it.
- **Drop capitals** (``merge_drop_capitals``): a region the layout analyser labelled as a drop
  capital is one letter set large at a paragraph start. Its text is glued onto the first row of
  the text region it sits in (the one overlapping it most, else the nearest below-right), and the
  region disappears.

Regions keep their labels in Docling's vocabulary; the level lands in ``Region.heading_level``.
Out of scope for now, planned: separators as article boundaries, alignment (datelines,
signatures), boxed regions as advertisements, emphasis from letter-spacing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .types import BBox, Page, Region

DEMOTE = 1.2           # a "heading" below this ratio to the body size is body text
SPLIT = 1.6            # leading rows at least this much taller than the rest are a headline
LEVELS = ((2.5, 1), (1.7, 2), (0.0, 3))
HEADING_LABELS = ("section_header", "title")
MAX_HEADLINE_ROWS = 3


@dataclass
class RetypeStats:
    body_size: float = 0.0
    demoted: int = 0
    split: int = 0
    levelled: int = 0
    drop_capitals: int = 0
    examples: list[str] = field(default_factory=list)

    def note(self, kind: str, text: str) -> None:
        if len(self.examples) < 200:
            self.examples.append(f"{kind}: {text[:60]}")

    def __iadd__(self, other: "RetypeStats") -> "RetypeStats":
        self.demoted += other.demoted
        self.split += other.split
        self.levelled += other.levelled
        self.drop_capitals += other.drop_capitals
        self.examples = (self.examples + other.examples)[:200]
        return self

    def describe(self) -> str:
        return (f"headings levelled {self.levelled}, demoted {self.demoted}, split off {self.split}, "
                f"drop capitals merged {self.drop_capitals}")


def heading_level(ratio: float) -> int:
    for threshold, level in LEVELS:
        if ratio >= threshold:
            return level
    return LEVELS[-1][1]


def row_heights(content) -> list[float]:
    return [r.bbox.height for r in content.rows() if r.text.strip()]


def body_size(contents: Sequence) -> float:
    """The page's body type: the median row height over its text regions (all rows, so long regions weigh more)."""
    heights = [h for c in contents if c.region.label == "text" for h in row_heights(c)]
    if not heights:
        heights = [h for c in contents for h in row_heights(c)]
    return float(np.median(heights)) if heights else 0.0


def region_size(content) -> float:
    heights = row_heights(content)
    return float(np.median(heights)) if heights else 0.0


# ------------------------------------------------------------------------------------------ rules
def level_headings(contents: list, body: float, stats: RetypeStats) -> None:
    for c in contents:
        if c.region.label not in HEADING_LABELS or body <= 0:
            continue
        size = region_size(c)
        if size <= 0:
            continue
        ratio = size / body
        if ratio < DEMOTE and c.region.label == "section_header":
            c.region.label = "text"
            c.region.heading_level = None
            stats.demoted += 1
            stats.note("demoted", c.text)
        else:
            c.region.heading_level = heading_level(ratio)
            stats.levelled += 1


def split_leading_heading(content, body: float, stats: RetypeStats):
    """Cut a headline off the top of a text region; returns the new heading ``RegionContent`` or None."""
    from .document import RegionContent

    rows = [r for r in content.rows() if r.text.strip()]
    if content.region.label != "text" or len(rows) < 2 or len(content.lines) != len(content.texts):
        return None
    heights = [r.bbox.height for r in rows]
    tall = 0
    while tall < min(MAX_HEADLINE_ROWS, len(rows) - 1):
        rest = heights[tall + 1:]
        if heights[tall] >= SPLIT * float(np.median(rest)) and heights[tall] >= SPLIT * body:
            tall += 1
        else:
            break
    if tall == 0:
        return None
    head_rows = {rows[i].index for i in range(tall)}
    keep = [i for i, ln in enumerate(content.lines) if ln.row not in head_rows]
    move = [i for i, ln in enumerate(content.lines) if ln.row in head_rows]
    if not keep or not move:
        return None
    pts = np.concatenate([content.lines[i].polygon for i in move])
    region = Region("section_header", BBox.of(pts).polygon, content.region.score, content.region.order,
                    f"{content.region.id}_head", raw_label=content.region.raw_label + "/split")
    head = RegionContent(region)
    head.lines = [content.lines[i] for i in move]
    head.texts = [content.texts[i] for i in move]
    head.records = [content.records[i] for i in move] if len(content.records) == len(content.lines) else []
    for ln in head.lines:
        ln.region_id = region.id
    content.lines = [content.lines[i] for i in keep]
    content.texts = [content.texts[i] for i in keep]
    if len(content.records) == len(keep) + len(move):
        content.records = [content.records[i] for i in keep]
    body_pts = np.concatenate([ln.polygon for ln in content.lines])
    content.region.polygon = BBox.of(body_pts).polygon
    stats.split += 1
    stats.note("split", head.text)
    return head


def merge_drop_capitals(contents: list, stats: RetypeStats) -> list:
    """Glue each drop-capital region's text onto the first row of the paragraph it opens."""
    drops = [c for c in contents if c.region.raw_label.endswith("drop-capital")]
    if not drops:
        return contents
    drop_ids = {id(d) for d in drops}
    texts = [c for c in contents if c.region.label == "text" and id(c) not in drop_ids and c.texts]
    removed = set()
    for d in drops:
        letter = d.text.strip()
        if not letter or not texts:
            continue
        db = d.region.bbox
        target = max(texts, key=lambda c: c.region.bbox.intersection_area(db))
        if target.region.bbox.intersection_area(db) <= 0:
            below = [c for c in texts if c.region.bbox.y1 > db.y0 and c.region.bbox.x1 > db.x0]
            if not below:
                continue
            target = min(below, key=lambda c: (abs(c.region.bbox.y0 - db.y0), c.region.bbox.x0 - db.x1))
        first = target.texts[0]
        target.texts[0] = type(first)(letter + first.text.lstrip(), first.score)
        removed.add(id(d))
        stats.drop_capitals += 1
        stats.note("drop capital", target.texts[0].text)
    return [c for c in contents if id(c) not in removed]


def retype_page(page: Page, contents: list, stats: RetypeStats | None = None) -> list:
    """Apply the rules to a page's contents; returns the new list (regions may be added or removed)."""
    stats = stats if stats is not None else RetypeStats()
    body = body_size(contents)
    stats.body_size = body
    out: list = []
    for c in contents:
        head = split_leading_heading(c, body, stats) if body > 0 else None
        if head is not None:
            out.append(head)
        out.append(c)
    level_headings(out, body, stats)
    out = merge_drop_capitals(out, stats)
    ordered = [c for c in out if c.region.order is not None]
    ordered.sort(key=lambda c: (c.region.order, 0 if c.region.id.endswith("_head") else 1))
    for rank, c in enumerate(ordered):
        c.region.order = rank
    return out
