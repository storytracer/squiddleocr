import numpy as np
import pytest

from squiddleocr.document import DocumentBuilder, RegionContent, export
from squiddleocr.layout import SingleRegionLayout
from squiddleocr.pipeline import Pipeline
from squiddleocr.runtime import resolve_providers
from squiddleocr.types import BBox, Recognition, Region, TableCellResult, TableResult, TextLine


def test_bbox_of_polygon_and_intersection():
    b = BBox.of(np.array([[10, 20], [50, 20], [50, 40], [10, 40]]))
    assert (b.x0, b.y0, b.x1, b.y1) == (10, 20, 50, 40)
    assert b.intersection_area(BBox(30, 30, 100, 100)) == 20 * 10
    assert b.intersection_area(BBox(60, 60, 70, 70)) == 0


def test_resolve_providers_cpu():
    assert resolve_providers("cpu") == ["CPUExecutionProvider"]
    assert resolve_providers("auto")[-1] == "CPUExecutionProvider"
    with pytest.raises(ValueError):
        resolve_providers("gpu")


def test_document_builder_and_exports(tmp_path, page):
    regions = [Region("section_header", BBox(20, 40, 380, 70).polygon, 1.0, 0, "r0"),
               Region("text", BBox(20, 120, 200, 150).polygon, 1.0, 1, "r1"),
               Region("table", BBox(20, 200, 380, 280).polygon, 1.0, 2, "r2"),
               Region("picture", BBox(300, 100, 380, 180).polygon, 1.0, 3, "r3")]
    contents = [RegionContent(r) for r in regions]
    contents[0].lines = [TextLine(BBox(20, 40, 380, 70).polygon, 1.0)]
    contents[0].texts = [Recognition("Heading", 0.9)]
    contents[1].lines = [TextLine(BBox(20, 120, 200, 135).polygon, 1.0), TextLine(BBox(20, 135, 200, 150).polygon, 1.0)]
    contents[1].texts = [Recognition("first line", 0.9), Recognition("second line", 0.9)]
    contents[2].table = TableResult("r2", cells=[TableCellResult("a", 0, 0, bbox=BBox(20, 200, 200, 240)),
                                                 TableCellResult("b", 0, 1, bbox=BBox(200, 200, 380, 240))], num_rows=1, num_cols=2)
    b = DocumentBuilder("t")
    b.add_page(page, contents)
    doc = b.build()
    md = doc.export_to_markdown()
    assert md.index("Heading") < md.index("first line") < md.index("| a")
    assert "second line" in md and "<!-- image -->" in md
    paths = export(doc, tmp_path, "p", ["doclang", "md", "html", "json", "txt"])
    assert [p.name for p in paths] == ["p.doclang.xml", "p.md", "p.html", "p.json", "p.txt"]
    xml = (tmp_path / "p.doclang.xml").read_text()
    assert "Heading" in xml and "<table" in xml
    with pytest.raises(ValueError):
        export(doc, tmp_path, "p", ["docx"])


def test_pipeline_single_region_keeps_records_and_texts(page, fake_detector, fake_recognizer):
    pipe = Pipeline(recognizer=fake_recognizer, detector=fake_detector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=SingleRegionLayout())
    contents = pipe.process_page(page)
    c = contents[0]
    assert len(contents) == 1 and c.region.label == "text"
    assert [t.text for t in c.texts] == ["page:360x30", "page:180x30"] and c.texts[0].score == 0.75
    assert len(c.records) == 2 and [ln.region_id for ln in c.lines] == ["page", "page"]
    doc = pipe.run([page], "t")
    assert doc.export_to_text().splitlines() == ["page:360x30", "page:180x30"]


def test_pipeline_page_level_detection_assigns_lines_to_regions(page, fake_detector, fake_recognizer):
    regions = [Region("text", BBox(0, 0, 400, 100).polygon, 1.0, 0, "top"),
               Region("text", BBox(0, 100, 400, 300).polygon, 1.0, 1, "bottom"),
               Region("picture", BBox(300, 200, 400, 300).polygon, 1.0, 2, "pic")]

    class Layout:
        def analyze(self, page):
            return regions

    pipe = Pipeline(recognizer=fake_recognizer, detector=fake_detector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=Layout(), detect_per_region=False)
    contents = {c.region.id: c for c in pipe.process_page(page)}
    assert len(contents["top"].lines) == 1 and len(contents["bottom"].lines) == 1 and contents["pic"].lines == []
    assert contents["top"].texts[0].text == "top:360x30" and contents["bottom"].texts[0].text == "bottom:180x30"


def test_pipeline_orphan_lines_become_regions_in_order(page, fake_detector, fake_recognizer):
    class Layout:
        def analyze(self, page):
            return [Region("text", BBox(0, 100, 400, 300).polygon, 1.0, 0, "body")]

    pipe = Pipeline(recognizer=fake_recognizer, detector=fake_detector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=Layout())
    contents = sorted(pipe.process_page(page), key=lambda c: c.region.order)
    assert [c.region.id for c in contents] == ["orphan_0", "body"]
    assert contents[0].texts[0].text == "orphan_0:360x30"


def test_pipeline_reads_table_cells_as_records(page, fake_detector, fake_recognizer):
    class Layout:
        def analyze(self, page):
            return [Region("table", BBox(0, 0, 400, 300).polygon, 1.0, 0, "table_0")]

    class Tables:
        def structure(self, page, region):
            return TableResult("table_0", cells=[TableCellResult("", 0, 0, bbox=BBox(20, 40, 380, 70)),
                                                 TableCellResult("", 1, 0, bbox=BBox(20, 120, 200, 150))], num_rows=2, num_cols=1)

    class CellDetector:
        def detect(self, page, region=None):   # one line per cell crop, none on the page
            return [TextLine(region.polygon.copy(), 1.0, None, "cell")] if region is not None and region.id == "cell" else []

    pipe = Pipeline(recognizer=fake_recognizer, detector=CellDetector(), layout=Layout(), tables=Tables())
    c = pipe.process_page(page)[0]
    assert [cell.text for cell in c.table.cells] == ["table_0:378x39", "table_0:198x39"]   # cell boxes padded by 15 % / 30 %
    assert len(c.records) == 2 and all(ln.region_id == "table_0" for ln in c.lines)


def test_pipeline_keep_line_order_trusts_the_detector(page, fake_detector, fake_recognizer):
    # bottom line first, as a segmenter that orders lines itself might; kept as is, one row per line
    pipe = Pipeline(recognizer=fake_recognizer, detector=fake_detector([(20, 120, 200, 150), (20, 40, 380, 70)]),
                    layout=SingleRegionLayout(), keep_line_order=True)
    c = pipe.process_page(page)[0]
    assert [ln.bbox.y0 for ln in c.lines] == [120, 40] and [ln.row for ln in c.lines] == [0, 1]
    assert c.text == "page:180x30\npage:360x30"


def test_pipeline_reads_formula_regions_as_latex(page, fake_detector, fake_recognizer):
    class Layout:
        def analyze(self, page):
            return [Region("formula", BBox(0, 30, 400, 80).polygon, 1.0, 0, "display_formula_0", raw_label="display_formula"),
                    Region("text", BBox(0, 100, 400, 160).polygon, 1.0, 1, "text_1")]

    class Formulas:
        def recognize(self, page, regions):
            return [f"\\frac{{{int(r.bbox.width)}}}{{{int(r.bbox.height)}}}" for r in regions]

    pipe = Pipeline(recognizer=fake_recognizer, detector=fake_detector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=Layout(), formulas=Formulas())
    contents = pipe.process_page(page)
    assert contents[0].formula == "\\frac{400}{50}" and contents[0].lines == []     # no text OCR inside a formula
    assert contents[1].formula is None and contents[1].texts[0].text == "text_1:180x30"
    builder = DocumentBuilder("t")
    builder.add_page(page, contents)
    md = builder.build().export_to_markdown()
    assert "$$\\frac{400}{50}$$" in md and "180x30" in md


def test_cli_formats_follow_the_pipeline():
    from squiddleocr.cli import resolve_formats

    assert resolve_formats("auto", "paddle") == ["md"] and resolve_formats("auto", "kraken") == ["hocr"]
    assert resolve_formats("md, hocr,json", "paddle") == ["md", "hocr", "json"]
    assert resolve_formats("txt,alto", "kraken") == ["txt", "alto"]
    with pytest.raises(ValueError, match="need --pipeline paddle"):
        resolve_formats("md", "kraken")
    with pytest.raises(ValueError, match="unknown export format"):
        resolve_formats("pdf", "paddle")
