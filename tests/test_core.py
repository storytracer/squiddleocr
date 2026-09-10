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
    b = DocumentBuilder("t", text="lines")
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
    doc = pipe.run([page], "t", text="lines")
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


def test_pipeline_reads_table_lines_and_hands_them_to_the_table_recognizer(page, fake_detector, fake_recognizer):
    class Layout:
        def analyze(self, page):
            return [Region("table", BBox(0, 0, 400, 300).polygon, 1.0, 0, "table_0")]

    class Tables:
        def structure(self, page, region, lines=(), texts=()):   # one cell per line, text from the pipeline's recognition
            cells = [TableCellResult(t, i, 0, bbox=ln.bbox) for i, (ln, t) in enumerate(zip(lines, texts))]
            return TableResult(region.id, cells, len(cells), 1)

    pipe = Pipeline(recognizer=fake_recognizer, detector=fake_detector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=Layout(), tables=Tables())
    c = pipe.process_page(page)[0]
    assert [ln.region_id for ln in c.lines] == ["table_0", "table_0"] and len(c.records) == 2     # lines stay on the table region
    assert [cell.text for cell in c.table.cells] == ["table_0:360x30", "table_0:180x30"]          # kraken's text, page-level boxes
    builder = DocumentBuilder("t", text="lines")
    builder.add_page(page, c and [c])
    assert "<table" in builder.build().export_to_html()


def test_cells_from_html_places_spans():
    from squiddleocr.tables.paddle import cells_from_html

    cells = cells_from_html('<table><tr><td colspan="2">Eos &amp; ♂</td><td>x</td></tr>'
                            '<tr><td rowspan="2">Forceps</td><td>Apodemen</td><td></td></tr><tr><td>Corpus</td><th>h</th></tr></table>')
    grid = {(c.row, c.col): (c.text, c.row_span, c.col_span, c.header) for c in cells}
    assert grid[(0, 0)] == ("Eos & ♂", 1, 2, False) and grid[(0, 2)] == ("x", 1, 1, False)
    assert grid[(1, 0)] == ("Forceps", 2, 1, False) and grid[(1, 1)] == ("Apodemen", 1, 1, False)
    assert grid[(2, 1)] == ("Corpus", 1, 1, False) and grid[(2, 2)] == ("h", 1, 1, True)   # the rowspan pushes row 2 to col 1
    assert max(c.row + c.row_span for c in cells) == 3 and max(c.col + c.col_span for c in cells) == 3


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
    builder = DocumentBuilder("t", text="lines")
    builder.add_page(page, contents)
    md = builder.build().export_to_markdown()
    assert "$$\\frac{400}{50}$$" in md and "180x30" in md


def test_cli_formats_follow_the_pipeline():
    from squiddleocr.cli import resolve_formats

    assert resolve_formats("auto", "paddle") == ["md"] and resolve_formats("auto", "kraken") == ["hocr"]
    assert resolve_formats("md, hocr,json", "paddle") == ["md", "hocr", "json"]
    assert resolve_formats("txt,alto", "kraken") == ["txt", "alto"]
    with pytest.raises(ValueError, match="need layout regions"):
        resolve_formats("md", "kraken")
    with pytest.raises(ValueError, match="unknown export format"):
        resolve_formats("pdf", "paddle")


def test_markdown_tables_keep_spans_as_html():
    from docling_core.types.doc import DoclingDocument, TableCell, TableData

    from squiddleocr.document import export_markdown

    plain = DoclingDocument(name="p")
    plain.add_table(data=TableData(num_rows=1, num_cols=2, table_cells=[
        TableCell(text="a", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1),
        TableCell(text="b", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2)]))
    assert "| a" in export_markdown(plain) and "<table" not in export_markdown(plain)
    spanned = DoclingDocument(name="s")
    spanned.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=[
        TableCell(text="Forceps", start_row_offset_idx=0, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1, row_span=2),
        TableCell(text="Apodemen", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2),
        TableCell(text="Corpus", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2)]))
    md = export_markdown(spanned)
    assert '<td rowspan="2">Forceps</td>' in md and md.count("Forceps") == 1


def test_sections_group_heading_with_what_follows(tmp_path, page):
    from docling_core.types.doc import DoclingDocument, GroupItem

    def region(label, y, i):
        return Region(label, BBox(10, y, 390, y + 20).polygon, 1.0, i, f"r{i}")

    def content(label, y, i, text):
        c = RegionContent(region(label, y, i))
        c.lines, c.texts = [TextLine(BBox(10, y, 390, y + 20).polygon)], [Recognition(text, 0.9)]
        return c

    contents = [content("text", 10, 0, "masthead"), content("section_header", 40, 1, "Headline one"),
                content("text", 70, 2, "body 1"), RegionContent(region("picture", 100, 3)),
                content("section_header", 130, 4, "Headline two"), content("text", 160, 5, "body 2")]
    plain = DocumentBuilder("t", text="lines")
    plain.add_page(page, contents)
    grouped = DocumentBuilder("t", sections=True)
    grouped.add_page(page, contents)
    doc = grouped.build()
    assert doc.export_to_markdown() == plain.build().export_to_markdown()
    assert doc.export_to_text() == plain.build().export_to_text()
    body = [ref.resolve(doc) for ref in doc.body.children]
    assert [type(x).__name__ for x in body] == ["TextItem", "GroupItem", "GroupItem"]
    first, second = body[1], body[2]
    assert isinstance(first, GroupItem) and first.name == "Headline one" and first.label.value == "section"
    kids = [ref.resolve(doc) for ref in first.children]
    assert [k.label.value for k in kids] == ["section_header", "text", "picture"]
    assert [ref.resolve(doc).label.value for ref in second.children] == ["section_header", "text"]
    paths = export(doc, tmp_path, "s", ["json", "doclang", "html"])
    assert (tmp_path / "s.json").stat().st_size > 0 and paths[0].name == "s.json"
    xml = (tmp_path / "s.doclang.xml").read_text()
    assert xml.count("<group") == 2 and 'name="Headline one"' in xml and xml.index("<group") < xml.index("Headline one")
    reloaded = DoclingDocument.load_from_json(tmp_path / "s.json")
    assert sum(1 for r in reloaded.body.children if isinstance(r.resolve(reloaded), GroupItem)) == 2
