import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from squiddleocr.document import RegionContent
from squiddleocr.segmentation import lines_to_segmentation
from squiddleocr.serialize import serialize_page, to_segmentation
from squiddleocr.types import BBox, Page, Recognition, Region, TextLine

kraken = pytest.importorskip("kraken.serialization")

from kraken.containers import BaselineLine, BaselineOCRRecord, BBoxLine, BBoxOCRRecord  # noqa: E402

TEXT = "ab cd"


def _record(rid, i, y, baseline=True, cuts=True):
    """A kraken record as the recogniser returns it: line geometry, text, one cut per character."""
    cut = [(k * 60, (k + 1) * 60) for k in range(len(TEXT))] if cuts else []
    if baseline:
        line = BaselineLine(id=f"{rid}_l{i}", baseline=[(12, y + 24), (380, y + 26)],
                            boundary=[(12, y), (380, y + 2), (380, y + 28), (12, y + 26)], regions=[rid])
        return BaselineOCRRecord(TEXT, cut, [0.9] * len(TEXT), line)
    line = BBoxLine(id=f"{rid}_l{i}", bbox=(12, y, 380, y + 28), regions=[rid])
    boxes = [[[12 + k * 60, y], [12 + (k + 1) * 60, y], [12 + (k + 1) * 60, y + 28], [12 + k * 60, y + 28]] for k in range(len(TEXT))]
    return BBoxOCRRecord(TEXT, boxes if cuts else [], [0.9] * len(TEXT), line)


def _contents(baseline=True, cuts=True):
    head = RegionContent(Region("section_header", BBox(10, 100, 390, 140).polygon, 0.9, 0, "paragraph_title_1"))
    head.lines = [TextLine(BBox(12, 112, 380, 140).polygon, 1.0, region_id="paragraph_title_1")]
    head.records = [_record("paragraph_title_1", 0, 112, baseline, cuts)]
    body = RegionContent(Region("text", BBox(10, 150, 390, 230).polygon, 0.9, 1, "text_0"))
    body.lines = [TextLine(BBox(12, 152, 380, 180).polygon, 1.0, region_id="text_0"),
                  TextLine(BBox(12, 190, 380, 218).polygon, 1.0, region_id="text_0")]
    body.records = [_record("text_0", 0, 152, baseline, cuts), _record("text_0", 1, 190, baseline, cuts)]
    for c in (head, body):
        c.texts = [Recognition(TEXT, 0.9) for _ in c.records]
    pic = RegionContent(Region("picture", BBox(10, 240, 100, 290).polygon, 0.9, 2, "image_3"))
    return [head, body, pic]


def test_lines_to_segmentation_ids_and_regions():
    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    lines = [TextLine(BBox(0, 0, 10, 10).polygon, 1.0, region_id="text_0"),
             TextLine(np.array([[0, 20], [10, 20], [10, 30], [0, 30]]), 1.0, np.array([[0, 28], [10, 28]]), "text_0"),
             TextLine(BBox(0, 40, 10, 50).polygon, 1.0, region_id="title_1")]
    seg = lines_to_segmentation(page, lines)
    assert [ln.id for ln in seg.lines] == ["text_0_l0", "text_0_l1", "title_1_l0"]
    assert [ln.regions for ln in seg.lines] == [["text_0"], ["text_0"], ["title_1"]]
    assert seg.type == "baselines" and seg.lines[0].type == "bbox" and seg.lines[1].type == "baselines"


def test_segmentation_keeps_reading_order_regions_and_cuts():
    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    seg, has_cuts = to_segmentation(page, _contents())
    assert seg.type == "baselines" and has_cuts
    assert [ln.regions[0] for ln in seg.lines] == ["paragraph_title_1", "text_0", "text_0"]
    assert set(seg.regions) == {"section_header", "text", "picture"}
    _, has_cuts = to_segmentation(page, _contents(cuts=False))
    assert not has_cuts


@pytest.mark.parametrize("fmt", ["hocr", "alto", "page"])
@pytest.mark.parametrize("baseline", [True, False])
def test_kraken_templates_render_lines_words_and_glyphs(fmt, baseline):
    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    out = serialize_page(page, _contents(baseline=baseline), fmt, {"pipeline": "test", "squiddleocr": "0"})
    if fmt == "hocr":
        words = [w for w in re.findall(r'<span class="ocrx_word"[^>]*>([^<]*)</span>', out) if w.strip()]
        assert words == ["ab", "cd"] * 3 and out.count('class="ocr_line"') == 3
        return
    ET.fromstring(out)
    if fmt == "page":
        assert out.count("<TextLine") == 3 and out.index("paragraph_title_1") < out.index("text_0_l0")
        assert ('<Baseline points="12,136 380,138"/>' in out) == baseline
        assert "type {type:section_header;}" in out
    else:
        assert out.count("<String ") >= 6 and "<Glyph" in out
        assert ('BASELINE="12 136 380 138"' in out) == baseline


@pytest.mark.parametrize("fmt", ["hocr", "alto", "page"])
@pytest.mark.parametrize("baseline", [True, False])
def test_detail_levels(fmt, baseline):
    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    outs = {d: serialize_page(page, _contents(baseline=baseline), fmt, None, detail=d) for d in ("line", "word", "glyph")}
    if fmt == "hocr":
        assert outs["word"] == outs["glyph"] and 'class="ocrx_word"' in outs["word"]     # kraken's hOCR has no glyphs
        assert 'class="ocrx_word"' not in outs["line"] and outs["line"].count('class="ocr_line"') == 3
        assert outs["line"].count("ab cd") == 3          # our hocr_line template; kraken's renders no text here
        return
    for out in outs.values():
        ET.fromstring(out)
    word_tag, glyph_tag = ("<String ID=", "<Glyph") if fmt == "alto" else ("<Word", "<Glyph")
    assert glyph_tag in outs["glyph"] and word_tag in outs["glyph"]
    assert glyph_tag not in outs["word"] and outs["word"].count(word_tag) == outs["glyph"].count(word_tag)
    assert word_tag not in outs["line"] and glyph_tag not in outs["line"] and outs["line"].count("ab cd") == 3
    # the word level is the glyph level minus the Glyph elements: same words, same boxes
    assert re.findall(word_tag + r"[^>]*>", outs["word"]) == re.findall(word_tag + r"[^>]*>", outs["glyph"])
    with pytest.raises(ValueError):
        serialize_page(page, _contents(), fmt, None, detail="char")
