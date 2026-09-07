"""Tests for the pluggable core: types, crops, recogniser, document builder, pipeline (no PaddleX needed)."""
import numpy as np
import pytest

from squiddleocr.crops import crop_polygon, crop_quad, line_image
from squiddleocr.document import DocumentBuilder, RegionContent, export
from squiddleocr.layout import SingleRegionLayout
from squiddleocr.pipeline import Pipeline
from squiddleocr.recognizers import OnnxRecognizer
from squiddleocr.recognizers.onnx import batch_lines, preprocess_line
from squiddleocr.runtime import resolve_providers
from squiddleocr.types import BBox, Page, Recognition, Region, TableCellResult, TableResult, TextLine

H = 96


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tiny_model_dir(tiny_net, tmp_path_factory):
    """A complete SquiddleOCR model directory built from the random tiny network."""
    from squiddleocr.convert.config import build_inference_config, dump_yaml
    from squiddleocr.convert.export import export_onnx
    from squiddleocr.convert.wrapper import PaddleRecWrapper

    d = tmp_path_factory.mktemp("model")
    export_onnx(PaddleRecWrapper(tiny_net), d / "inference.onnx", H)
    chars = list(" abcdefghij")  # 11 chars -> 12 classes incl. blank
    dump_yaml(build_inference_config(chars, H, "PP-OCRv6_tiny_rec"), d / "inference.yml")
    return d


@pytest.fixture
def page():
    rng = np.random.default_rng(0)
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    img[40:70, 20:380] = rng.integers(0, 80, (30, 360, 3), dtype=np.uint8)
    img[120:150, 20:200] = rng.integers(0, 80, (30, 180, 3), dtype=np.uint8)
    return Page(img, None, 1)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, page, region=None):
        return [TextLine(BBox(*b).polygon, 0.9, region_id=region.id if region else "") for b in self.boxes]


class FakeRecognizer:
    device_provider = "FakeExecutionProvider"

    def recognize(self, lines):
        return [Recognition(f"line{i}:{im.shape[1]}x{im.shape[0]}", 0.5) for i, im in enumerate(lines)]


# -------------------------------------------------------------------- types
def test_bbox_of_polygon_and_intersection():
    b = BBox.of(np.array([[10, 20], [50, 20], [50, 40], [10, 40]]))
    assert (b.x0, b.y0, b.x1, b.y1) == (10, 20, 50, 40)
    assert b.intersection_area(BBox(30, 30, 100, 100)) == 20 * 10
    assert b.intersection_area(BBox(60, 60, 70, 70)) == 0


def test_crops(page):
    quad = crop_quad(page.image, BBox(20, 40, 380, 70).polygon)
    assert quad.shape == (30, 360, 3)
    poly = crop_polygon(page.image, np.array([[20, 40], [380, 40], [380, 70], [200, 70]]))
    assert poly.shape == (30, 360, 3)
    assert poly[-1, 10].tolist() == [255, 255, 255]       # outside the polygon -> white
    assert line_image(page, BBox(20, 40, 380, 70).polygon).shape == (30, 360, 3)


# --------------------------------------------------------------- recogniser
def test_preprocess_and_batch(page):
    x = preprocess_line(page.image[40:70, 20:380], H)
    assert x.shape == (3, H, 1152) and -1 <= x.min() and x.max() <= 1
    b = batch_lines([x, preprocess_line(page.image[120:150, 20:200], H)])
    assert b.shape == (2, 3, H, 1152) and b[1, :, :, -1].max() == 0


def test_onnx_recognizer_runs_and_batches(tiny_model_dir, page):
    rec = OnnxRecognizer(tiny_model_dir, device="cpu", batch_size=2)
    assert rec.height == H and len(rec.chars) == 11
    lines = [page.image[40:70, 20:380], page.image[120:150, 20:200], page.image[40:70, 20:100]]
    out = rec.recognize(lines)
    assert len(out) == 3 and all(isinstance(r.text, str) and 0 <= r.score <= 1 for r in out)
    single = rec.recognize(lines[1:2])[0]
    assert single.text == out[1].text          # order of results follows the input order, whatever the batching


def test_resolve_providers_cpu():
    assert resolve_providers("cpu") == ["CPUExecutionProvider"]
    assert resolve_providers("auto")[-1] == "CPUExecutionProvider"
    with pytest.raises(ValueError):
        resolve_providers("gpu")


# ------------------------------------------------------------------ document
def test_document_builder_and_exports(tmp_path, page):
    regions = [Region("section_header", BBox(20, 40, 380, 70).polygon, 1.0, 0, "r0"),
               Region("text", BBox(20, 120, 200, 150).polygon, 1.0, 1, "r1"),
               Region("picture", BBox(20, 200, 200, 280).polygon, 1.0, 2, "r2"),
               Region("table", BBox(210, 200, 390, 280).polygon, 1.0, 3, "r3")]
    contents = [RegionContent(regions[0], texts=[Recognition("Heading", 0.9)]),
                RegionContent(regions[1], texts=[Recognition("first line", 0.9), Recognition("second line", 0.8)]),
                RegionContent(regions[2]),
                RegionContent(regions[3], table=TableResult("r3", [TableCellResult("a", 0, 0), TableCellResult("b", 0, 1),
                                                                    TableCellResult("c", 1, 0, col_span=2)], 2, 2))]
    b = DocumentBuilder("doc")
    b.add_page(page, contents[::-1])       # out of order on purpose; the builder sorts by Region.order
    doc = b.build()
    md = doc.export_to_markdown()
    assert md.index("Heading") < md.index("first line") < md.index("| a")
    assert "second line" in md and "<!-- image -->" in md
    paths = export(doc, tmp_path, "p", ["doclang", "md", "html", "json", "txt"])
    assert [p.name for p in paths] == ["p.doclang.xml", "p.md", "p.html", "p.json", "p.txt"]
    xml = paths[0].read_text(encoding="utf-8")
    assert "Heading" in xml and "<table" in xml
    with pytest.raises(ValueError):
        export(doc, tmp_path, "p", ["docx"])


# ------------------------------------------------------------------ pipeline
def test_pipeline_single_region(page):
    pipe = Pipeline(recognizer=FakeRecognizer(), detector=FakeDetector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=SingleRegionLayout())
    contents = pipe.process_page(page)
    assert len(contents) == 1 and contents[0].region.label == "text"
    assert [t.text for t in contents[0].texts] == ["line0:360x30", "line1:180x30"]
    doc = pipe.run([page], "t")
    assert doc.export_to_text().splitlines() == ["line0:360x30", "line1:180x30"]


def test_pipeline_page_level_detection_assigns_lines_to_regions(page):
    regions = [Region("text", BBox(0, 0, 400, 100).polygon, 1.0, 0, "top"),
               Region("text", BBox(0, 100, 400, 300).polygon, 1.0, 1, "bottom"),
               Region("picture", BBox(300, 200, 400, 300).polygon, 1.0, 2, "pic")]

    class Layout:
        def analyze(self, page):
            return regions

    pipe = Pipeline(recognizer=FakeRecognizer(), detector=FakeDetector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=Layout(), detect_per_region=False)
    contents = {c.region.id: c for c in pipe.process_page(page)}
    assert len(contents["top"].lines) == 1 and len(contents["bottom"].lines) == 1 and contents["pic"].lines == []


def test_session_falls_back_to_cpu_on_runtime_failure(tiny_model_dir):
    from squiddleocr import runtime

    s = runtime.create_session(tiny_model_dir / "inference.onnx", "cpu")

    class Broken:
        def get_providers(self):
            return ["CoreMLExecutionProvider"]

        def run(self, *a):
            raise RuntimeError("Unable to compute the prediction (simulated)")

    s._session = Broken()
    x = np.random.rand(1, 3, H, 120).astype(np.float32) * 2 - 1
    out = s.run(None, {"x": x})
    assert s.provider == "CPUExecutionProvider" and out[0].shape[0] == 1


def test_pipeline_record_level_uses_recognize_lines_and_keeps_records(page):
    class Rec:
        def __init__(self, text):
            self.prediction, self.confidences, self.cuts = text, [0.5, 1.0], [(0, 1), (1, 2)]

    class KrakenLike:
        device_provider = "torch cpu"

        def recognize_lines(self, page, lines):
            return [Rec(f"rec{i}:{ln.region_id}") for i, ln in enumerate(lines)]

        def recognize(self, images):
            raise AssertionError("record level must not crop")

    pipe = Pipeline(recognizer=KrakenLike(), detector=FakeDetector([(20, 40, 380, 70), (20, 120, 200, 150)]),
                    layout=SingleRegionLayout())
    contents = pipe.process_page(page)
    c = contents[0]
    assert [t.text for t in c.texts] == ["rec0:page", "rec1:page"] and c.texts[0].score == 0.75
    assert len(c.records) == 2 and [ln.region_id for ln in c.lines] == ["page", "page"]
