"""Layout plots: the page image with the regions, rows and labels the pipeline ended up with.

One JPEG per page (``--plots``): region polygons coloured by Docling label, a tag with the label,
reading-order rank and heading level, the detected rows as thin boxes, furniture (page numbers,
running heads) in grey. Large pages are scaled down to ``max_width`` pixels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOURS = {
    "text": (30, 100, 230), "section_header": (220, 40, 40), "title": (220, 40, 40),
    "page_header": (120, 120, 120), "page_footer": (120, 120, 120),
    "picture": (150, 60, 200), "table": (240, 140, 20), "caption": (0, 150, 150), "footnote": (0, 150, 150),
    "formula": (200, 120, 0), "list_item": (30, 100, 230),
}
DEFAULT = (200, 170, 0)
ROW = (0, 170, 60)


def plot_page(page, contents: Sequence, path: str | Path, max_width: int = 2400) -> Path:
    scale = min(1.0, max_width / max(page.width, 1))
    im = Image.fromarray(page.image).convert("RGB")
    if scale < 1.0:
        im = im.resize((int(page.width * scale), int(page.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    size = max(12, int(im.width / 90))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    line_w = max(2, im.width // 800)
    for c in contents:                        # rows first, regions on top
        for ln in c.lines:
            b = ln.bbox
            d.rectangle((b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale), outline=ROW + (255,), width=max(1, line_w // 2))
    for c in contents:
        r = c.region
        colour = COLOURS.get(r.label, DEFAULT)
        pts = [(float(x) * scale, float(y) * scale) for x, y in np.asarray(r.polygon, dtype=float).reshape(-1, 2)]
        if len(pts) >= 3:
            d.polygon(pts, outline=colour + (255,), fill=colour + (28,))
            d.line(pts + [pts[0]], fill=colour + (255,), width=line_w)
        tag = r.label + (f" h{r.heading_level}" if r.heading_level else "") + (f" #{r.order}" if r.order is not None else "")
        if r.raw_label and r.raw_label.split("/")[-1] not in ("", r.label):
            tag += f" ({r.raw_label.split('/')[-1]})"
        b = r.bbox
        x, y = b.x0 * scale, b.y0 * scale
        tw = d.textlength(tag, font=font) + 8
        d.rectangle((x, y - size - 6, x + tw, y), fill=colour + (230,))
        d.text((x + 4, y - size - 4), tag, fill=(255, 255, 255, 255), font=font)
    path = Path(path)
    im.save(path, quality=88) if path.suffix.lower() in (".jpg", ".jpeg") else im.save(path)
    return path
