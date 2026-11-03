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

from squiddleocr.chain import MODELS, blocks, models
from squiddleocr.cli import main
from linecaster.blocks.writers import PageFilesWriter
from squiddleocr.goldens import compare

# goldens per machine (GPU numerics differ): spark (aarch64, GB10) and forge (x86_64, RTX 4000 Ada)
GOLDENS = Path(os.path.expanduser(os.environ.get(
    "SQUIDDLE_GOLDENS_DIR",
    f"~/data/goldens/squiddleocr-7c8a052/{'spark' if platform.machine() == 'aarch64' else 'forge'}")))
IMAGES = Path(os.path.expanduser("~/data/bhl-ocr-eval"))


def _names(steps):
    return [type(s).__name__ for s in steps]


def test_chain_shape_both_backends():
    full = ["LayoutStep", "LinesStep", "AssignLinesToRegions", "RecognitionStep", "NormalizeUnicode", "TablesStep",
            "FormulasStep", "Typography", "AssembleDocument"]
    for kind, describe in (("in-process", "in-process"), ("served", "served http://h:1/v1 (layout)")):
        steps = blocks(kind, endpoint="http://h:1/v1")
        assert _names(steps) == full
        assert steps[0].backend.describe() == describe
        assert steps[2].formulas_separately and steps[-1].text == "reflow" and not steps[-1].sections
    assert blocks("in-process", recognition_model="small")[3].backend.kwargs["model"] == "small"
    assert MODELS[2]["kwargs"]["model"] == "medium"            # not mutated by the override
    steps = blocks(layout="none", typography=False, nfc=False, document=False)
    assert _names(steps) == ["SingleRegionLayout", "LinesStep", "AssignLinesToRegions", "RecognitionStep"]
    steps = blocks(formulas=False, tables=False, text="lines", sections=True)
    assert "FormulasStep" not in _names(steps) and "TablesStep" not in _names(steps)
    assert not steps[2].formulas_separately                    # formula regions are read as text
    assert steps[-1].text == "lines" and steps[-1].sections
    with pytest.raises(ValueError):
        blocks("elsewhere")
    with pytest.raises(ValueError):
        blocks(layout="eynollah")


def test_model_options_map_to_engine_settings():
    assert models() == MODELS
    m = {x["name"]: x["kwargs"] for x in models(recognition_model="small", det_model="PP-OCRv6_small_det",
                                                unclip_ratio=2.0, layout_model="PP-DocLayout_plus-L",
                                                formula_model="PP-FormulaNet-L", batch_size=4, precision="32-true",
                                                device="cpu", rtl=True)}
    assert m["layout"] == {"model_name": "PP-DocLayout_plus-L", "device": "cpu"}
    assert m["lines"] == {"model_name": "PP-OCRv6_small_det", "unclip_ratio": 2.0, "device": "cpu"}
    assert m["recognition"] == {"model": "small", "batch_size": 4, "precision": "32-true",
                                "text_direction": "horizontal-rl", "device": "cpu"}
    assert m["formulas"] == {"batch_size": 4, "model_name": "PP-FormulaNet-L", "device": "cpu"}
    assert m["tables"] == {"device": "cpu"}
    # the served config has the same shape as linecaster's examples/squiddle_models.json
    shipped = Path(__file__).resolve().parents[2] / "linecaster" / "examples" / "squiddle_models.json"
    if shipped.exists():
        assert json.loads(shipped.read_text())["models"] == MODELS


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
    h = CliRunner().invoke(main, ["ocr", "--help"]).output
    for opt in ("--per-document", "--text", "--typography", "--furniture", "--dividers", "--sections", "--detail",
                "--plots", "--layout", "--layout-model", "--tables", "--formulas", "--formula-model", "--det-model",
                "--unclip-ratio", "--batch-size", "--precision", "--device", "--rtl", "--suffix", "--backend"):
        assert opt in h, opt
    assert "--no-nfc" not in h                                 # parity checks only


# ------------------------------------------------------------------ the CLI on CPU stand-ins
@pytest.fixture
def fake_models(monkeypatch):
    import fake_engines

    from squiddleocr import chain

    monkeypatch.setattr(chain, "MODELS", fake_engines.MODELS)
    return fake_engines


def _images(folder: Path, names=("p2", "p1")) -> Path:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.new("RGB", (400, 300), "white").save(folder / f"{n}.png")
    (folder / "notes.txt").write_text("not an image")
    (folder / "sub").mkdir(exist_ok=True)
    Image.new("RGB", (400, 300), "white").save(folder / "sub" / "deeper.png")     # subfolders are not read
    return folder


def _run(args):
    r = CliRunner().invoke(main, ["ocr", *map(str, args)], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    return r


def test_cli_defaults_write_reflowed_markdown_next_to_the_images(tmp_path: Path, fake_models):
    scans = _images(tmp_path / "scans")
    _run([scans])
    assert sorted(p.name for p in scans.iterdir() if p.suffix == ".md") == ["p1.paddle.md", "p2.paddle.md"]
    md = (scans / "p1.paddle.md").read_text()
    assert md == "# DE NUMERIS\n\nWeſen der Dinge und Zahlen."          # reflow joins the division
    assert not (scans / "rows").exists() and not (scans / "logs").exists() and not (scans / "sub" / "deeper.paddle.md").exists()


def test_cli_every_format_into_an_output_folder(tmp_path: Path, fake_models):
    scans, out = _images(tmp_path / "scans"), tmp_path / "out"
    _run([scans, "-o", out, "-f", "md,txt,json,html,doclang,hocr,alto,page,transcription", "--plots", "--text", "lines",
          "--suffix", "none", "--detail", "line", "--sections", "--dividers", "--furniture"])
    names = sorted(p.name for p in out.iterdir() if p.name.startswith("p1."))
    assert names == ["p1.alto.xml", "p1.doclang.xml", "p1.hocr", "p1.html", "p1.json", "p1.layout.jpg", "p1.md",
                     "p1.page.xml", "p1.transcription.json", "p1.txt"]
    assert (out / "p1.md").read_text() == "# DE NUMERIS\n\n---\n\nWeſen der Din-  \nge und Zahlen."
    assert json.loads((out / "p1.json").read_text())["groups"][0]["label"] == "section"
    t = from_alto((out / "p1.alto.xml").read_bytes())
    assert [ln.text for ln in t.ordered_lines()][1] == "Weſen der Din-" and not t.lines[0].words   # --detail line
    rows = [json.loads(line) for line in (out / "rows" / "00000.jsonl").read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == ["p1", "p2"]


def test_cli_without_typography_tables_formulas_and_layout(tmp_path: Path, fake_models):
    scans, out = _images(tmp_path / "scans", ("p1",)), tmp_path / "out"
    _run([scans / "p1.png", "-o", out, "-f", "txt", "--layout", "none", "--no-typography"])
    t = json.loads((out / "rows" / "00000.jsonl").read_text())["page"]["transcriptions"]["ocr"]
    assert [r["id"] for r in t["regions"]] == ["page"] and all(ln["region_id"] == "page" for ln in t["lines"])
    assert [m["step"] for m in t["model"]] == ["lines", "recognition", "unicode"]     # no layout, typography, tables


def test_cli_per_document(tmp_path: Path, fake_models):
    book, out = _images(tmp_path / "book", ("0002", "0001")), tmp_path / "out"
    _run([book, "--per-document", "-f", "md,txt,json,alto", "-o", out])
    assert (out / "book.paddle.md").read_text().count("# DE NUMERIS") == 2
    doc = json.loads((out / "book.paddle.json").read_text())
    assert doc["name"] == "book" and len(doc["pages"]) == 2
    assert (out / "0001.paddle.alto.xml").exists() and not (out / "0001.paddle.md").exists()
    _run([book / "0001.png", "--per-document"])                       # one file: named after it, next to it
    assert (book / "0001.paddle.md").exists()


def test_cli_rejects_unknown_formats_and_duplicate_names(tmp_path: Path, fake_models):
    scans = _images(tmp_path / "scans", ("p1",))
    r = CliRunner().invoke(main, ["ocr", str(scans), "-f", "md,pdf"])
    assert r.exit_code != 0 and "pdf" in r.output
    other = _images(tmp_path / "other", ("p1",))
    r = CliRunner().invoke(main, ["ocr", str(scans), str(other)])
    assert r.exit_code != 0 and "p1" in r.output


def test_models_config_for_the_server():
    r = CliRunner().invoke(main, ["models-config", "--det-model", "PP-OCRv6_small_det"])
    cfg = json.loads(r.output)
    assert [m["name"] for m in cfg["models"]] == ["layout", "lines", "recognition", "tables", "formulas"]
    assert cfg["models"][1]["kwargs"]["model_name"] == "PP-OCRv6_small_det"


@pytest.mark.skipif(os.environ.get("SQUIDDLE_GOLDENS") != "1", reason="GPU goldens run: set SQUIDDLE_GOLDENS=1")
@pytest.mark.parametrize("subset", ["sample-60", "tables-7"])
def test_goldens_gpu(tmp_path: Path, subset: str):
    """The chain reproduces squiddleocr 0.3's lines exactly after NFC (in-process, or SQUIDDLE_BACKEND=served),
    including the page number 0.3's retyper drops on tables-7 (12342174, read at 0.74 < 0.75)."""
    backend = os.environ.get("SQUIDDLE_BACKEND", "in-process")
    cmd = [sys.executable, "-m", "squiddleocr.cli", "ocr", str(IMAGES / subset), "-o", str(tmp_path), "--backend", backend,
           "--window", "4", "-f", "txt"]
    subprocess.run(cmd, check=True, capture_output=True)
    tot = compare(tmp_path, GOLDENS / subset)
    assert tot["common"] == tot["lines_gold"] == tot["lines_ours"] == tot["box"] == tot["words"]
    assert tot["pages_exact"] == tot["pages"]


PARITY_SETS = {"scans": "~/data/squiddletest/scans", "formulas": "~/data/squiddletest/formulas",
               "tables-7": "~/data/bhl-ocr-eval/tables-7"}


@pytest.mark.skipif(os.environ.get("SQUIDDLE_PARITY") != "1", reason="GPU parity with 0.3: set SQUIDDLE_PARITY=1")
@pytest.mark.parametrize("subset", sorted(PARITY_SETS))
@pytest.mark.parametrize("flags", [[], ["--text", "lines", "--sections", "--dividers"]], ids=["defaults", "lines"])
def test_document_formats_match_legacy_gpu(tmp_path: Path, subset: str, flags: list):
    """`squiddle ocr` against `squiddle legacy ocr --pipeline paddle` on the same pages: md, txt, html and doclang
    byte-exact after NFC of 0.3's text, the DoclingDocument JSON byte-exact without NFC (--no-nfc)."""
    import unicodedata

    images = Path(os.path.expanduser(PARITY_SETS[subset]))
    fmts = ["-f", "md,txt,html,doclang,json"]
    run = lambda *args: subprocess.run([sys.executable, "-m", "squiddleocr.cli", *map(str, args)], check=True,
                                       capture_output=True)
    run("legacy", "ocr", images, "-o", tmp_path / "legacy", *fmts, *flags)
    run("ocr", images, "-o", tmp_path / "new", *fmts, *flags)
    run("ocr", images, "-o", tmp_path / "raw", *fmts, *flags, "--no-nfc")
    legacy = sorted(p for p in (tmp_path / "legacy").iterdir() if p.is_file())
    assert legacy
    nfc = lambda p: unicodedata.normalize("NFC", p.read_text(encoding="utf-8"))
    for p in legacy:
        assert p.read_bytes() == (tmp_path / "raw" / p.name).read_bytes(), p.name
        if not p.name.endswith(".json") and flags:          # reflow decisions move with NFC (NOTES); lines do not
            assert nfc(p) == (tmp_path / "new" / p.name).read_text(encoding="utf-8"), p.name
