import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from squiddleocr.document import RegionContent
from squiddleocr.types import BBox, Page, Recognition, Region, TableCellResult, TableResult, TextLine

kraken = pytest.importorskip("kraken.serialization")

from squiddleocr.serialize import serialize_page, to_segmentation  # noqa: E402
from squiddleocr.segmentation import lines_to_segmentation  # noqa: E402


def _contents():
    body = RegionContent(Region("text", BBox(10, 10, 390, 100).polygon, 0.9, 1, "text_0", raw_label="text"))
    body.lines = [TextLine(np.array([[12, 12], [380, 14], [380, 40], [12, 38]]), 1.0, np.array([[12, 36], [380, 38]])),
                  TextLine(BBox(12, 50, 200, 80).polygon, 1.0)]
    body.texts = [Recognition("first line", 0.9), Recognition("second", 0.8)]
    head = RegionContent(Region("section_header", BBox(10, 110, 390, 140).polygon, 0.9, 0, "paragraph_title_1"))
    head.lines = [TextLine(BBox(12, 112, 300, 138).polygon, 1.0)]
    head.texts = [Recognition("Heading", 0.9)]
    table = RegionContent(Region("table", BBox(10, 150, 390, 250).polygon, 0.9, 2, "table_2"))
    table.table = TableResult("table_2", cells=[TableCellResult("a", 0, 0, bbox=BBox(12, 152, 100, 180)),
                                                TableCellResult("b", 0, 1, bbox=BBox(110, 152, 200, 180))], num_rows=1, num_cols=2)
    pic = RegionContent(Region("picture", BBox(10, 260, 100, 290).polygon, 0.9, 3, "image_3"))
    return [body, head, table, pic]


def test_segmentation_keeps_order_geometry_and_text():
    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    seg, has_cuts = to_segmentation(page, _contents())
    assert seg.type == "baselines" and not has_cuts
    assert [ln.regions[0] for ln in seg.lines] == ["paragraph_title_1", "text_0", "text_0", "table_2", "table_2"]
    assert seg.lines[1].type == "baselines" and seg.lines[1].baseline == [(12, 36), (380, 38)]
    assert seg.lines[2].type == "bbox" and seg.lines[2].bbox == (12, 50, 200, 80)
    assert [ln.prediction for ln in seg.lines] == ["Heading", "first line", "second", "a", "b"]
    assert set(seg.regions) == {"text", "section_header", "table", "picture"}


@pytest.mark.parametrize("fmt", ["alto", "page"])
def test_kraken_templates_render_valid_xml_with_text_and_geometry(fmt):
    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    xml = serialize_page(page, _contents(), fmt, {"detector": "test", "squiddleocr": "0"})
    root = ET.fromstring(xml)
    assert root is not None
    assert "first line" in xml and "Heading" in xml and 'a' in xml
    if fmt == "page":
        assert '<Baseline points="12,36 380,38"/>' in xml
        assert xml.index("Heading") < xml.index("first line")
        assert 'type {type:section_header;}' in xml
    else:
        assert 'BASELINE="12 36 380 38"' in xml and '<String CONTENT="first line" />' in xml


def test_lines_to_segmentation_ids_and_regions():
    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    lines = [TextLine(BBox(0, 0, 10, 10).polygon, 1.0, region_id="text_0"),
             TextLine(np.array([[0, 20], [10, 20], [10, 30], [0, 30]]), 1.0, np.array([[0, 28], [10, 28]]), "text_0"),
             TextLine(BBox(0, 40, 10, 50).polygon, 1.0, region_id="title_1")]
    seg = lines_to_segmentation(page, lines)
    assert [ln.id for ln in seg.lines] == ["text_0_l0", "text_0_l1", "title_1_l0"]
    assert [ln.regions for ln in seg.lines] == [["text_0"], ["text_0"], ["title_1"]]
    assert seg.type == "baselines" and seg.lines[0].type == "bbox" and seg.lines[1].type == "baselines"


def test_hocr_needs_cuts_and_renders_words_with_kraken_records():
    from kraken.containers import BaselineLine, BaselineOCRRecord

    page = Page(np.full((300, 400, 3), 255, dtype=np.uint8), None, 1)
    contents = _contents()
    with pytest.raises(ValueError, match="kraken"):
        serialize_page(page, contents, "hocr")
    text = "ab cd"

    def rec(rid, i, y):
        line = BaselineLine(id=f"{rid}_l{i}", baseline=[(12, y + 24), (380, y + 26)],
                            boundary=[(12, y), (380, y + 2), (380, y + 28), (12, y + 26)], regions=[rid])
        return BaselineOCRRecord(text, [(k * 60, (k + 1) * 60) for k in range(len(text))], [0.9] * len(text), line)

    body, head, table = contents[0], contents[1], contents[2]
    body.records = [rec("text_0", i, 12 + 38 * i) for i in range(len(body.lines))]
    head.records = [rec("paragraph_title_1", 0, 112)]
    table.records = [rec("table_2", 0, 152)]     # a cell read as a kraken record
    table.lines = [TextLine(BBox(12, 152, 380, 180).polygon, 1.0, region_id="table_2")]
    html = serialize_page(page, contents, "hocr")
    words = re.findall(r'<span class="ocrx_word"[^>]*>([^<]*)</span>', html)
    assert [w for w in words if w.strip()] == ["ab", "cd"] * 4      # kraken also emits the whitespace segments
    assert html.count('class="ocr_line"') == 4
    xml = serialize_page(page, contents, "alto")
    assert xml.count("<String ") >= 8 and "<Glyph" in xml
