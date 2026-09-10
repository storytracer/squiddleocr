"""The eynollah stages on a synthetic PAGE-XML, and the launcher's patch (no subprocess, no models)."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from squiddleocr.detectors.eynollah import EynollahLines, synthesize_baseline
from squiddleocr.eynollah.pagexml import parse_page_xml, regions_from_page
from squiddleocr.eynollah.source import EynollahOptions, EynollahSource
from squiddleocr.layout.eynollah import EynollahLayout
from squiddleocr.pipeline import Pipeline
from squiddleocr.types import BBox, Page

PAGE_XML = """<?xml version="1.0" encoding="utf-8"?>
<pc:PcGts xmlns:pc="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <pc:Metadata><pc:Creator>SBB_QURATOR</pc:Creator></pc:Metadata>
  <pc:Page imageFilename="p.jpg" imageWidth="400" imageHeight="300">
    <pc:ReadingOrder><pc:OrderedGroup id="ro1">
      <pc:RegionRefIndexed index="0" regionRef="region_0003"/>
      <pc:RegionRefIndexed index="1" regionRef="region_0001"/>
      <pc:RegionRefIndexed index="2" regionRef="region_0002"/>
    </pc:OrderedGroup></pc:ReadingOrder>
    <pc:TextRegion id="region_0001" type="paragraph">
      <pc:Coords points="20,60 380,60 380,140 20,140" conf="0.9"/>
      <pc:TextLine id="region_0001_line_0001"><pc:Coords points="22,62 378,64 378,96 22,94"/></pc:TextLine>
      <pc:TextLine id="region_0001_line_0002"><pc:Coords points="22,100 200,100 200,138 22,138"/></pc:TextLine>
    </pc:TextRegion>
    <pc:TextRegion id="region_0002">
      <pc:Coords points="20,160 380,160 380,200 20,200"/>
      <pc:TextLine id="region_0002_line_0001"><pc:Coords points="20,160 380,170 380,200 20,190"/></pc:TextLine>
    </pc:TextRegion>
    <pc:TextRegion id="region_0003" type="heading">
      <pc:Coords points="20,10 380,10 380,50 20,50"/>
      <pc:TextLine id="region_0003_line_0001"><pc:Coords points="20,10 380,10 380,50 20,50"/></pc:TextLine>
    </pc:TextRegion>
    <pc:TextRegion id="region_0004" type="drop-capital">
      <pc:Coords points="20,62 60,62 60,96 20,96"/>
      <pc:TextLine id="region_0004_line_0001"><pc:Coords points="20,62 60,62 60,96 20,96"/></pc:TextLine>
    </pc:TextRegion>
    <pc:ImageRegion id="region_0005"><pc:Coords points="20,210 380,210 380,290 20,290"/></pc:ImageRegion>
    <pc:SeparatorRegion id="region_0006"><pc:Coords points="20,150 380,150 380,152 20,152"/></pc:SeparatorRegion>
  </pc:Page>
</pc:PcGts>
"""


@pytest.fixture
def xml_dir(tmp_path):
    (tmp_path / "p.xml").write_text(PAGE_XML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def source(xml_dir):
    return EynollahSource(EynollahOptions(xml_dir=xml_dir))


@pytest.fixture
def eyn_page(tmp_path):
    return Page(np.full((300, 400, 3), 255, dtype=np.uint8), tmp_path / "p.jpg", 1)


def test_parse_labels_order_and_lines(xml_dir):
    parsed = parse_page_xml(xml_dir / "p.xml")
    assert (parsed.width, parsed.height) == (400, 300)
    assert [r.id for r in parsed.regions] == ["region_0001", "region_0002", "region_0003", "region_0004", "region_0005"]
    by_id = {r.id: r for r in parsed.regions}
    assert by_id["region_0001"].conf == 0.9 and len(by_id["region_0001"].lines) == 2
    regions = regions_from_page(parsed)
    labels = {r.id: r.label for r in regions}
    assert labels == {"region_0001": "text", "region_0002": "text", "region_0003": "section_header",
                      "region_0004": "text", "region_0005": "picture"}
    orders = {r.id: r.order for r in regions}
    assert orders["region_0003"] == 0 and orders["region_0001"] == 1 and orders["region_0002"] == 2
    assert orders["region_0004"] is None and orders["region_0005"] is None
    assert regions[0].raw_label == "TextRegion/paragraph"


def test_layout_keeps_reading_order_and_places_the_rest(source, eyn_page):
    regions = EynollahLayout(source).analyze(eyn_page)
    assert [r.id for r in regions] == ["region_0003", "region_0004", "region_0001", "region_0002", "region_0005"]
    assert [r.order for r in regions] == [0, 1, 2, 3, 4]
    assert "region_0006" not in {r.id for r in regions}


def test_lines_per_region_with_baselines_inside_the_polygon(source, eyn_page):
    det = EynollahLines(source)
    layout = EynollahLayout(source)
    regions = {r.id: r for r in layout.analyze(eyn_page)}
    lines = det.detect(eyn_page, regions["region_0001"])
    assert len(lines) == 2 and all(ln.region_id == "region_0001" for ln in lines)
    assert lines[0].bbox == BBox(22, 62, 378, 96) and lines[1].bbox.y0 == 100      # XML order kept
    for ln in lines:
        b = ln.baseline
        assert b.shape == (2, 2) and b[0, 0] == ln.bbox.x0 and b[1, 0] == ln.bbox.x1
        assert all(ln.bbox.y0 <= y <= ln.bbox.y1 for y in b[:, 1])
        assert all(y > (ln.bbox.y0 + ln.bbox.y1) / 2 for y in b[:, 1])            # in the lower half
    assert det.detect(eyn_page, regions["region_0005"]) == []
    assert len(det.detect(eyn_page)) == 5
    assert EynollahLines(source, baselines=False).detect(eyn_page, regions["region_0002"])[0].baseline is None


def test_synthesized_baseline_follows_a_sloped_bottom_edge():
    poly = np.array([[0, 0], [100, 10], [100, 50], [0, 40]], dtype=float)    # bottom edge from (0,40) to (100,50)
    base = synthesize_baseline(poly, descender=0.0)
    assert np.allclose(base, [[0, 40], [100, 50]])
    raised = synthesize_baseline(poly, descender=0.18)
    assert np.allclose(raised[:, 1], base[:, 1] - 0.18 * 50)
    tiny = synthesize_baseline(np.array([[5, 5], [5, 5], [5, 5]], dtype=float))
    assert tiny.shape == (2, 2)


def test_pipeline_recognises_every_line_per_region(source, eyn_page, fake_recognizer):
    pipe = Pipeline(recognizer=fake_recognizer, detector=EynollahLines(source), layout=EynollahLayout(source),
                    detect_per_region=True, keep_line_order=True)
    contents = pipe.process_page(eyn_page)
    assert [c.region.id for c in contents] == ["region_0003", "region_0004", "region_0001", "region_0002", "region_0005"]
    texts = {c.region.id: [t.text for t in c.texts] for c in contents}
    assert texts["region_0001"] == ["region_0001:356x34", "region_0001:178x38"]
    assert texts["region_0005"] == []                                       # a picture gets no lines
    assert [ln.row for ln in contents[2].lines] == [0, 1]
    doc = pipe.run([eyn_page], "t")
    text = doc.export_to_text()
    assert text.index("region_0003") < text.index("region_0001") < text.index("region_0002")
    pipe.close()


def test_source_uses_existing_xml_without_a_runner(source, eyn_page):
    assert source._runner is None
    parsed = source.page(eyn_page)
    assert parsed is source.page(eyn_page) and source._runner is None
    source.close()


def test_missing_xml_is_a_clear_error_when_eynollah_is_absent(tmp_path, monkeypatch, eyn_page):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    src = EynollahSource(EynollahOptions(xml_dir=tmp_path))
    with pytest.raises(RuntimeError, match="squiddleocr\\[eynollah\\]"):
        src.page(eyn_page)


def test_options_policy_adds_r2l_and_parses_args():
    pol = EynollahOptions(args="-fl -romb -tab", rtl=True, vram_margin="20%", jobs=3).policy()
    assert pol.args == ("-fl", "-romb", "-tab", "-r2l") and pol.vram_margin == 0.2 and pol.jobs == 3
    assert EynollahOptions(args="-r2l", rtl=True).policy().args == ("-r2l",)


def test_runner_helpers(tmp_path):
    from squiddleocr.eynollah.runner import EynollahRunner, _by_stem, _drop_partial_xml, _valid_xml

    good, bad = tmp_path / "a.xml", tmp_path / "b.xml"
    good.write_text("<a/>")
    bad.write_text("<a>")
    assert _valid_xml(good) and not _valid_xml(bad) and not _valid_xml(tmp_path / "c.xml")
    _drop_partial_xml({"a": Path("a.jpg"), "b": Path("b.jpg")}, tmp_path)
    assert good.exists() and not bad.exists()
    assert list(_by_stem([Path("x/a.jpg"), Path("x/b.png")])) == ["a", "b"]
    with pytest.raises(ValueError):
        _by_stem([Path("x/a.jpg"), Path("y/a.png")])
    from squiddleocr.eynollah.resources import Plan
    from squiddleocr.eynollah.resources import Policy

    runner = EynollahRunner(tmp_path / "models", Policy(args=("-fl", "-romb"), device="GPU0"))
    plan = Plan(jobs=3, threads=2, reserved_cores=2, device="GPU0", providers=["CUDA", "CPU"], use_gpu=True, models=(),
                vram_limits={"page": 1236})
    cmd = runner.command(tmp_path / "in", tmp_path, plan, 3)
    assert cmd[1:3] == ["-m", "squiddleocr.eynollah.launch"] and "-D" in cmd and cmd[cmd.index("-D") + 1] == "GPU0"
    assert cmd[cmd.index("-j") + 1] == "3" and cmd[-2:] == ["-fl", "-romb"] and "layout" in cmd


def test_launcher_patches_vram_limits_and_threads(monkeypatch):
    pytest.importorskip("eynollah")
    import cv2

    from eynollah.model_zoo import model_zoo

    from squiddleocr.eynollah import launch

    before = dict(model_zoo.MODEL_VRAM_LIMITS)
    threads = cv2.getNumThreads()
    try:
        applied = launch.apply_settings({"SQUIDDLE_EYNOLLAH_VRAM": '{"page": 4096, "textline": 6000}', "SQUIDDLE_EYNOLLAH_THREADS": "3"})
        assert applied == {"page": 4096, "textline": 6000}
        assert model_zoo.MODEL_VRAM_LIMITS["page"] == 4096 and model_zoo.MODEL_VRAM_LIMITS["textline"] == 6000
        assert model_zoo.MODEL_VRAM_LIMITS["region_1_2"] == before["region_1_2"]
        assert cv2.getNumThreads() == 3
    finally:
        model_zoo.MODEL_VRAM_LIMITS.update(before)
        cv2.setNumThreads(threads)
