"""Retyper's page-level rules on synthetic regions: the type-size ladder, headline splits, drop capitals."""
import numpy as np

from squiddleocr.document import DocumentBuilder, RegionContent
from squiddleocr.retyper import RetypeStats, body_size, heading_level, retype_page
from squiddleocr.types import BBox, Page, Recognition, Region, TextLine


def content(label, rid, order, rows, x0=20, y0=20, raw=""):
    """rows: (text, height) pairs stacked with a 4 px gap."""
    y = y0
    lines, texts = [], []
    for t, h in rows:
        ln = TextLine(BBox(x0, y, x0 + 400, y + h).polygon)
        ln.row = len(lines)
        lines.append(ln)
        texts.append(Recognition(t, 0.9))
        y += h + 4
    c = RegionContent(Region(label, BBox(x0, y0, x0 + 400, y).polygon, 1.0, order, rid, raw_label=raw))
    c.lines, c.texts = lines, texts
    return c


def page():
    return Page(np.full((600, 500, 3), 255, dtype=np.uint8), None, 1)


def test_ladder():
    assert heading_level(3.0) == 1 and heading_level(2.0) == 2 and heading_level(1.3) == 3 and heading_level(1.0) == 3


def test_levels_and_demotion():
    body = content("text", "b", 2, [("body row", 20)] * 6)
    masthead = content("section_header", "m", 0, [("MASTHEAD", 60)], y0=0, raw="TextRegion/heading")
    headline = content("section_header", "h", 1, [("Headline", 40), ("over two rows", 40)], raw="TextRegion/heading")
    kicker = content("section_header", "k", 3, [("a body-sized kicker", 21)], raw="TextRegion/heading")
    stats = RetypeStats()
    out = retype_page(page(), [body, masthead, headline, kicker], stats)
    assert stats.body_size == 20 and body_size([body]) == 20
    by_id = {c.region.id: c for c in out}
    assert by_id["m"].region.heading_level == 1 and by_id["h"].region.heading_level == 2
    assert by_id["k"].region.label == "text" and by_id["k"].region.heading_level is None
    assert stats.levelled == 2 and stats.demoted == 1 and stats.split == 0
    assert [c.region.id for c in sorted(out, key=lambda c: c.region.order)] == ["m", "h", "b", "k"]


def test_headline_split_off_a_body_region():
    merged = content("text", "r", 0, [("Big headline", 40), ("first body row", 20), ("second body row", 20), ("third", 20)])
    other = content("text", "s", 1, [("more body", 20)] * 3)
    stats = RetypeStats()
    out = retype_page(page(), [merged, other], stats)
    assert [c.region.id for c in sorted(out, key=lambda c: c.region.order)] == ["r_head", "r", "s"]
    head = next(c for c in out if c.region.id == "r_head")
    assert head.region.label == "section_header" and head.text == "Big headline" and head.region.heading_level == 2
    body = next(c for c in out if c.region.id == "r")
    assert body.text.splitlines() == ["first body row", "second body row", "third"]
    assert body.region.bbox.y0 == 20 + 40 + 4 and head.region.bbox.y1 == 60
    assert stats.split == 1
    # a uniform region is left alone
    plain = content("text", "p", 0, [("row", 20)] * 4)
    assert len(retype_page(page(), [plain, other], RetypeStats())) == 2


def test_drop_capital_is_glued_to_its_paragraph():
    para = content("text", "p", 1, [("er Cavalier ritt", 20), ("weiter fort.", 20)], x0=60)
    drop = content("text", "d", None, [("D", 44)], x0=20, raw="TextRegion/drop-capital")
    stats = RetypeStats()
    out = retype_page(page(), [para, drop], stats)
    assert [c.region.id for c in out] == ["p"] and out[0].texts[0].text == "Der Cavalier ritt" and stats.drop_capitals == 1


def test_builder_uses_levels(tmp_path):
    body = content("text", "b", 2, [("body row", 20)] * 4)
    masthead = content("section_header", "m", 0, [("MASTHEAD", 60)], y0=0)
    sub = content("section_header", "s", 1, [("Sub-head", 28)])
    out = retype_page(page(), [body, masthead, sub], RetypeStats())
    b = DocumentBuilder("t")
    b.add_page(page(), out)
    md = b.build().export_to_markdown()
    assert "## MASTHEAD" in md and "#### Sub-head" in md and "body row" in md


def test_prose_or_display():
    from squiddleocr.retyper import classify

    prose = content("text", "p", 0, [("row", 20)] * 5)
    assert classify(prose.rows()) == "prose"
    two = content("text", "t", 0, [("first row of a short paragraph", 20), ("end.", 20)])
    two.lines[1].polygon = BBox(20, 44, 120, 64).polygon                     # a short last row is fine
    assert classify(two.rows()) == "prose"
    mixed = content("text", "a", 0, [("SOOWIN OSTA", 44), ("lapsewankrit", 20), ("Teatada: Aleksandri t. 32", 20)])
    assert classify(mixed.rows()) == "display"                               # mixed type sizes
    centred = content("text", "c", 0, [("Wanemuise teater", 20), ("Laupäewal", 20), ("kl. 8 öhtul", 20)])
    for ln, w in zip(centred.lines, (300, 120, 180)):
        b = ln.bbox
        ln.polygon = BBox(220 - w / 2, b.y0, 220 + w / 2, b.y1).polygon      # centred, ragged rows
    assert classify(centred.rows()) == "display"
    assert classify(content("text", "s", 0, [("one row", 20)]).rows()) == "display"
    rtl = content("text", "r", 0, [("א" * 10, 20)] * 4)
    assert classify(rtl.rows(), rtl=True) == "prose"


def test_display_regions_keep_their_rows_and_stop_continuation():
    prose = content("text", "p", 0, [("erster Absatz ohne", 20), ("Ende und weiter", 20), ("und weiter noch", 20)])
    ad = content("text", "a", 1, [("SOOWIN OSTA", 44), ("lapsewankrit", 20), ("Teatada: t. 32", 20)])
    for ln, w in zip(ad.lines, (300, 120, 180)):                            # centred advertisement lines
        b = ln.bbox
        ln.polygon = BBox(220 - w / 2, b.y0, 220 + w / 2, b.y1).polygon
    after = content("text", "q", 2, [("geht es hier weiter.", 20), ("Und aus.", 20)])
    out = retype_page(page(), [prose, ad, after], RetypeStats())
    assert [c.region.role for c in out] == ["prose", "display", "prose"]
    b = DocumentBuilder("t")
    b.add_page(page(), out)
    texts = [t.text for t in b.build().texts]
    assert texts == ["erster Absatz ohne Ende und weiter und weiter noch", "SOOWIN OSTA\nlapsewankrit\nTeatada: t. 32",
                     "geht es hier weiter. Und aus."]
    assert b.stats.continuations == 0
    assert len(retype_page(page(), [content("text", "x", 0, [("Big line", 40), ("SMALL", 20), ("mixed", 30)])], RetypeStats())) == 1   # no split off a display block
