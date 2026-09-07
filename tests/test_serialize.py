import xml.etree.ElementTree as ET

import numpy as np
import pytest

from squiddleocr.document import RegionContent
from squiddleocr.types import BBox, Page, Recognition, Region, TableCellResult, TableResult, TextLine

kraken = pytest.importorskip("kraken.serialization")

from squiddleocr.serialize import serialize_page, to_segmentation  # noqa: E402


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
    seg = to_segmentation(page, _contents())
    assert seg.type == "baselines"
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
