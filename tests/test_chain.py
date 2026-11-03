"""The squiddle pipeline on linecaster: chain shape, per-page files, goldens comparison, CLI.
The GPU goldens run is opt-in (``SQUIDDLE_GOLDENS=1``; see test_goldens_gpu)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from datatrove.data import Document

from linecaster.blocks.base import set_page
from linecaster.page import Page, PageImage
from linecaster.transcription import Line, Region, Transcription, Word, from_alto, from_hocr, from_page

from squiddleocr.chain import MODELS, blocks
from squiddleocr.cli import main
from linecaster.blocks.writers import PageFilesWriter
from squiddleocr.goldens import compare

# goldens per machine (GPU numerics differ): spark (aarch64, GB10) and forge (x86_64, RTX 4000 Ada)
GOLDENS = Path(os.path.expanduser(os.environ.get(
    "SQUIDDLE_GOLDENS_DIR",
    f"~/data/goldens/squiddleocr-7c8a052/{'spark' if platform.machine() == 'aarch64' else 'forge'}")))
IMAGES = Path(os.path.expanduser("~/data/bhl-ocr-eval"))


def test_chain_shape_both_backends():
    for kind, describe in (("in-process", "in-process"), ("served", "served http://h:1/v1 (layout)")):
        steps = blocks(kind, endpoint="http://h:1/v1")
        assert [type(s).__name__ for s in steps] == ["LayoutStep", "LinesStep", "AssignLinesToRegions", "RecognitionStep",
                                                 "TablesStep"]
        assert steps[0].backend.describe() == describe
    assert blocks("in-process", recognition_model="small")[3].backend.kwargs["model"] == "small"
    assert MODELS[2]["kwargs"]["model"] == "medium"            # not mutated by the override
    with pytest.raises(ValueError):
        blocks("elsewhere")


def _page() -> Document:
    t = Transcription(regions=[Region(id="text_0", label="text", polygon=[[10, 10], [200, 10], [200, 60], [10, 60]],
                                      bbox=[10, 10, 190, 50], reading_order=0)],
                      lines=[Line(id="text_0_l0", region_id="text_0", bbox=[12, 12, 180, 40],
                                  polygon=[[12, 12], [192, 12], [192, 52], [12, 52]], text="Weſen der Dinge",
                                  confidence=0.9, words=[Word(id="w0", text="Weſen", bbox=[12, 12, 60, 40], confidence=0.9),
                                                         Word(id="w1", text="der", bbox=[80, 12, 40, 40], confidence=0.9),
                                                         Word(id="w2", text="Dinge", bbox=[130, 12, 62, 40], confidence=0.9)])])
    page = Page(page_id="p1", image=PageImage(uri="/x/p1.webp", width=300, height=400))
    page.transcriptions["ocr"] = t
    d = Document(text="", id="p1", metadata={})
    set_page(d, page)
    return d


def test_page_files(tmp_path: Path):
    out = list(PageFilesWriter(str(tmp_path), ("hocr", "alto", "page", "txt", "transcription"),
                               bucket_by_item=False).run(iter([_page()])))
    assert len(out) == 1
    assert (tmp_path / "p1.txt").read_text() == "Weſen der Dinge"
    for name, reader in (("p1.hocr", from_hocr), ("p1.alto.xml", from_alto), ("p1.page.xml", from_page)):
        t = reader((tmp_path / name).read_bytes())
        assert t.text == "Weſen der Dinge" and [w.text for w in t.lines[0].words] == ["Weſen", "der", "Dinge"]
    assert json.loads((tmp_path / "p1.transcription.json").read_text())["lines"][0]["id"] == "text_0_l0"
    with pytest.raises(ValueError):
        PageFilesWriter(str(tmp_path), ("pdf",))


def test_goldens_compare_on_a_synthetic_page(tmp_path: Path):
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "p1.paddle.hocr").write_text(
        "<html><head><meta charset='utf-8'></head><body><div class='ocr_page'><span class='ocr_line' id='text_0_l0' title='bbox 12 12 192 52'>"
        "<span class='ocrx_word' title='bbox 12 12 72 52'>Weſen</span> <span class='ocrx_word' title='bbox 80 12 120 52'>der</span> "
        "<span class='ocrx_word' title='bbox 130 12 192 52'>Dinge</span></span></div></body></html>")
    (tmp_path / "run").mkdir()
    d = _page()
    (tmp_path / "run" / "0.jsonl").write_text(json.dumps({"id": d.id, "text": "", "page": d.metadata["page"]}) + "\n")
    tot = compare(tmp_path / "run", gold)
    assert tot == {"pages": 1, "lines_gold": 1, "lines_ours": 1, "common": 1, "box": 1, "words": 1, "pages_exact": 1}


def test_cli_lists_commands():
    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0 and "ocr" in r.output and "goldens" in r.output and "legacy" in r.output
    assert CliRunner().invoke(main, ["legacy", "--help"]).exit_code == 0


@pytest.mark.skipif(os.environ.get("SQUIDDLE_GOLDENS") != "1", reason="GPU goldens run: set SQUIDDLE_GOLDENS=1")
@pytest.mark.parametrize("subset", ["sample-60", "tables-7"])
def test_goldens_gpu(tmp_path: Path, subset: str):
    """The chain reproduces squiddleocr 0.3's lines exactly (in-process, or SQUIDDLE_BACKEND=served).
    tables-7 page 12342174 has one extra line: a page number read at 0.74 that 0.3's retyper dropped
    (below its 0.75 page-number confidence); the chain has no retyper (NOTES)."""
    backend = os.environ.get("SQUIDDLE_BACKEND", "in-process")
    cmd = [sys.executable, "-m", "squiddleocr.cli", "ocr", str(IMAGES / subset), "-o", str(tmp_path), "--backend", backend,
           "--window", "4", "-f", "txt"]
    subprocess.run(cmd, check=True, capture_output=True)
    tot = compare(tmp_path, GOLDENS / subset)
    assert tot["common"] == tot["lines_gold"] == tot["box"] == tot["words"]
    assert tot["lines_ours"] - tot["lines_gold"] == (1 if subset == "tables-7" else 0)
