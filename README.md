# SquiddleOCR

Document OCR for historical print, built on [kraken](https://kraken.re)'s PP-OCRv6 recogniser, with
two pipelines:

- **paddle**: PaddleX layout analysis with learned reading order, PP-OCRv6 text detection, table
  structure and formula recognition, out as a `DoclingDocument` (Markdown, HTML, DocLang, JSON,
  text) and as hOCR, ALTO or PAGE-XML written by kraken's serialiser.
- **kraken**: kraken's own pipeline, blla segmentation on the whole page and kraken's line order,
  out as hOCR, ALTO, PAGE-XML or text, exactly what the `kraken` command writes.

```
pip install "squiddleocr[paddle]"
squiddle ocr scans/
```

That reads every image in `scans/` and writes `scans/<name>.paddle.md` next to it: Markdown with
reading order, tables and formulas. `-o out/` collects the outputs elsewhere, `-f md,hocr,page`
adds formats. Models are downloaded on first use: kraken's from Zenodo into kraken's model cache,
PaddleX's into PaddleX's.

The recognisers are Benjamin Kiessling's kraken PP-OCRv6 models (Apache-2.0). Nothing is converted
or retrained: recognition is kraken's own code on kraken's own weights, in both pipelines.

## Contents

1. [Install](#1-install)
2. [Command line](#2-command-line)
3. [Python API](#3-python-api)
4. [How it works](#4-how-it-works)
5. [Exports](#5-exports)
6. [GPU and devices](#6-gpu-and-devices)
7. [Known limitations](#7-known-limitations)
8. [Licence, credit and citation](#8-licence-credit-and-citation)

## 1. Install

Python 3.11 or 3.12. The core package brings kraken with torch, docling-core and ONNX Runtime and
runs the kraken pipeline.

| extra | adds | you need it for |
|---|---|---|
| `paddle` | paddlex, paddlepaddle, transformers | the paddle pipeline: layout analysis, PP-OCRv6 text detection, tables, formulas |
| `test` | pytest | the test-suite |

On Linux and Windows the ONNX Runtime GPU build and the CUDA 13 runtime libraries are installed
as pip packages; nothing else is needed for the GPU. macOS gets the CPU/CoreML build.

Development checkout with [uv](https://docs.astral.sh/uv/):

```
uv sync --extra paddle --extra test
.venv/bin/python -m pytest -q
```

## 2. Command line

```
squiddle ocr INPUTS... [options]
```

`INPUTS` are image files (PNG, JPEG, TIFF, WebP, BMP) or folders of them.

| option | default | meaning |
|---|---|---|
| `--pipeline paddle\|kraken` | `paddle` | `paddle` = PaddleX layout analysis, PP-OCRv6 text detection, tables and formulas; every format. `kraken` = blla on the whole page (polygons and baselines, kraken's own line order; no layout, tables or formulas); formats `hocr`, `alto`, `page`, `txt` |
| `-m, --model SIZE\|FILE` | `medium` | kraken PP-OCRv6 recogniser: `tiny` (0.7M parameters), `small` (3.2M), `medium` (15.8M, most accurate), fetched by DOI; or a path to a kraken model file |
| `-o, --output DIR` | next to each image | where the exports go, one file set per image, named after it |
| `-f, --formats LIST` | `auto` | `md` for paddle, `hocr` for kraken. Document level: `md` (Markdown; a table with spanning cells is an HTML table), `doclang` (DocLang XML), `html`, `json` (lossless DoclingDocument), `txt`. Line level, one file per image: `hocr`, `alto`, `page` (PAGE-XML) |
| `--detail line\|word\|glyph` | `glyph` | depth of `hocr`, `alto`, `page`: `glyph` = words and glyphs from kraken's character cuts (kraken's default), `word` = words without glyphs, `line` = text per line. See [Exports](#5-exports) |
| `--layout paddle\|none` | `paddle` | `none` treats the page as one text block (no layout model, tables or formulas) |
| `--layout-model NAME` | `PP-DocLayoutV3` | `PP-DocLayout_plus-L` (PP-StructureV3's layout model, XY-cut reading order) |
| `--det-model NAME` | `PP-OCRv6_medium_det` | `PP-OCRv6_small_det` or `PP-OCRv6_tiny_det` for speed |
| `--unclip-ratio X` | `2.0` | expansion of PP-OCRv6 line boxes; PaddleOCR's default 1.5 clips ascenders and line-final hyphens on old print |
| `--tables / --no-tables` | on | cell structure of table regions with PaddleX's table pipeline, fed with our text lines and kraken's readings; cell text is kraken's |
| `--formulas / --no-formulas` | on | formula regions as LaTeX with PP-FormulaNet_plus-L (`--formula-model` picks PP-FormulaNet-L) |
| `--batch-size N` | `8` | lines per kraken forward pass |
| `--device auto\|cpu\|cuda\|tensorrt\|coreml` | `auto` | ONNX Runtime provider for the PaddleX models; `cpu` or `auto` for kraken's torch models |
| `--per-document` | off | one document for all inputs (a book) instead of one per image |
| `--suffix TAG` | `auto` | tag between name and extension, `<name>.<tag>.md`; `auto` is the pipeline name, so paddle and kraken runs sit side by side; `none` gives `<name>.md` |

Library chatter is off by default (PaddleX's model notes, ONNX Runtime's initialiser warnings,
kraken's per-line polygonizer warnings and the PIL warning that follows them); `SQUIDDLE_VERBOSE=1`
shows all of it.

```
squiddle ocr page.jpg -f md,doclang,json                 # one page, Markdown + DocLang + JSON
squiddle ocr book/ --per-document -f doclang             # whole book as one DocLang file, book/book.paddle.doclang.xml
squiddle ocr scans/ --pipeline kraken -f hocr,page       # kraken's own pipeline: polygons, baselines, words, glyphs
squiddle ocr scans/ -f alto --detail word                # ALTO with Strings but no Glyphs
squiddle ocr scans/ --layout none -m tiny                # fastest: plain OCR with the tiny recogniser
```

## 3. Python API

```python
from squiddleocr.factory import build_pipeline
from squiddleocr.document import export

pipe = build_pipeline("medium", pipeline="paddle", layout="paddle", tables=True, formulas=True, device="auto")
doc = pipe.run_files(["scans/0001.jpg", "scans/0002.jpg"], name="book")   # a DoclingDocument
export(doc, "out/", "book", ["md", "doclang", "json"])
```

Or assemble the components yourself:

```python
from squiddleocr.pipeline import Pipeline
from squiddleocr.recognizers.kraken import KrakenRecognizer
from squiddleocr.models import resolve_kraken_model
from squiddleocr.detectors.paddle import PaddleTextDetector
from squiddleocr.layout.paddle import PaddleLayout
from squiddleocr.tables.paddle import PaddleTableRecognizer
from squiddleocr.formulas.paddle import PaddleFormulaRecognizer
from squiddleocr.types import Page

pipe = Pipeline(
    recognizer=KrakenRecognizer(resolve_kraken_model("medium")),
    detector=PaddleTextDetector("PP-OCRv6_medium_det", unclip_ratio=2.0),
    layout=PaddleLayout("PP-DocLayoutV3"),
    tables=PaddleTableRecognizer(),
    formulas=PaddleFormulaRecognizer(),
)
page = Page.load("scans/0001.jpg", number=1)
contents = pipe.process_page(page)        # per-region lines, kraken records, table cells, LaTeX
```

`KrakenRecognizer.predict(image, segmentation)` is kraken's `RecognitionTaskModel.predict` and can
be used on its own with any kraken `Segmentation`.

## 4. How it works

```
page ─► LayoutAnalyzer (PP-DocLayoutV3) ─► regions: label, polygon, reading order
          │                                (kraken pipeline: the page is the one region)
          ├─► TextDetector, once per page ─► lines, assigned to the region they overlap most;
          │     paddle: PP-OCRv6 detection boxes       unclaimed lines become text regions
          │     kraken: blla polygons + baselines, blla's own line order
          │
          ├─► segmentation.py: lines ─► kraken Segmentation (BBoxLine / BaselineLine, ids <region>_l<n>)
          │
          ├─► kraken RecognitionTaskModel.predict ─► one ocr_record per line: text, character cuts, confidences
          │
          ├─► table regions ─► TableRecognizer places the region's lines and texts into the cells it finds
          └─► formula regions ─► FormulaRecognizer ─► LaTeX
                                                            │
                          DocumentBuilder ─► DoclingDocument ─► md / html / doclang / json / txt
                          serialize.py ─► kraken.serialization.serialize ─► hocr / alto / page
```

Each stage is a `typing.Protocol` with one method, in `squiddleocr/<stage>/base.py`:

| protocol | method | implementations |
|---|---|---|
| `Recognizer` | `recognize_lines(page, lines) -> [ocr_record]` | `KrakenRecognizer` (kraken's `RecognitionTaskModel`) |
| `TextDetector` | `detect(page, region=None) -> [TextLine]` | `PaddleTextDetector` (PP-OCRv6 det), `KrakenSegmenter` (blla) |
| `LayoutAnalyzer` | `analyze(page) -> [Region]` | `PaddleLayout` (PP-DocLayoutV3 with learned reading order; PP-DocLayout_plus-L with XY-cut), `SingleRegionLayout` |
| `TableRecognizer` | `structure(page, region, lines, texts) -> TableResult` | `PaddleTableRecognizer` (PaddleX's `table_recognition_v2` with our OCR result) |
| `FormulaRecognizer` | `recognize(page, regions) -> [latex]` | `PaddleFormulaRecognizer` (PP-FormulaNet_plus-L on PaddleX's transformers engine, torch) |

**Tables.** PaddleX's table pipeline is what PP-StructureV3 runs on a table: a wired/wireless
classifier, SLANet structure models, RT-DETR cell detectors and a step that matches OCR boxes into
cells. It accepts an external OCR result, so it gets the lines we detected on the page and kraken's
transcriptions and returns HTML with our text in its cells. The default is SLANet_plus end to end
(structure and cell boxes from its own prediction), which keeps the sub-rows of old ruled tables
apart where PP-StructureV3's SLANeXt plus cell detector merges them; `PaddleTableRecognizer(e2e=False)`
and its config give PP-StructureV3's choice back.

**Formulas.** Display formula regions skip text detection and go to PP-FormulaNet as crops; the
LaTeX becomes a Docling formula item. PP-FormulaNet ships as safetensors for PaddleX's transformers
engine, so it runs on the same torch and GPU as kraken.

**Line handling.** One detection pass per page, lines assigned to the region they overlap most,
orphan lines turned into `text` regions slotted into the reading order so page numbers survive a
layout miss, row-wise line ordering with de-duplication of detector fragments (the kraken pipeline
keeps blla's order untouched), one batched kraken recognition call per page including table lines,
and a `DoclingDocument` whose body order is the reading order and whose provenance boxes are the
regions.

Adding a layout model means returning `Region`s with Docling labels (`text`, `title`,
`section_header`, `caption`, `footnote`, `page_header`, `page_footer`, `table`, `picture`, `formula`,
`list_item`, `code`, `reference`) and an `order`, or leaving the order to
`squiddleocr.layout.order.xy_cut_order`. Adding a segmenter means returning `TextLine` polygons in
page coordinates, with baselines when the model has them. `build_pipeline` and the CLI learn the
new name in `factory.py`.

The rule behind the design: reuse kraken at the highest level it offers and never reimplement a
slice of it. Line extraction, batching, decoding, character positions and XML serialisation are
kraken's; SquiddleOCR adds layout, detection, tables, formulas, assignment and the Docling side.
The kraken pipeline is the check: its hOCR must agree with the `kraken` command line for line.

## 5. Exports

Two levels, from the same results:

- **Document level** (`md`, `html`, `doclang`, `json`, `txt`), paddle pipeline only: the
  `DoclingDocument`. Docling has no line level, so each region is one text item with the region's
  bounding box, in reading order, tables with their cells and spans, formulas as LaTeX, pictures
  as placeholders. Markdown writes a table with spanning cells as an HTML table, since Docling's
  pipe table would repeat a spanned cell's text in every position it covers.
- **Line level** (`hocr`, `alto`, `page`), both pipelines: kraken's serialiser, given the regions
  and kraken's records. Every line carries its geometry and text; with `--pipeline kraken` also the
  boundary polygon and the baseline. Regions are tagged with their Docling label (`custom="type
  {type:text;}"` in PAGE, `TAGREFS` in ALTO). A table is one region holding its cell lines; cell
  structure is not yet expressed in these formats.

`--detail` sets how far below the line these files go, per format:

| level | hOCR | ALTO | PAGE |
|---|---|---|---|
| `glyph` (default) | kraken's template: word spans with per-character `x_bboxes` and `x_confs` | String and Glyph | Word and Glyph |
| `word` | word spans with `x_wconf`, no character properties | String, no Glyph | Word, no Glyph |
| `line` | line spans with the text | one String per line | TextEquiv per line |

`glyph` and `line` are kraken's own sub-line switch; `word`, and `line` for hOCR, use kraken's
templates minus the character level, through kraken's custom-template mechanism (`templates/`).
Word and glyph boxes are derived from CTC time steps, about 8 input pixels of resolution; they are
not a trained word detector.

## 6. GPU and devices

`--device auto` uses CUDA when ONNX Runtime and torch find it. kraken and PP-FormulaNet run on
torch, the other PaddleX models on ONNX Runtime; both use the GPU. On aarch64 (DGX Spark, Jetson)
everything runs: torch ships cu130 wheels, ONNX Runtime GPU builds, and the PaddlePaddle CPU wheel
is only imported for PaddleX's bookkeeping (its inference engine is never used). macOS gets torch
MPS/CPU for kraken and CoreML/CPU for PaddleX; untested from here.

On a GB10 the paddle pipeline runs at 1 to 2.5 s per page depending on tables and formulas, the
kraken pipeline at about 5 s per page (blla's vectorisation is CPU work), sequentially.

## 7. Known limitations

- **Crops decide a lot.** With `--pipeline paddle`, kraken reads the axis-aligned box of each
  detected line; boxes that cut ascenders, descenders or the line-final `⸗` make the recogniser
  read `-`, and tiny single-digit cells are misread more often. `--unclip-ratio` mitigates this.
  `--pipeline kraken` reads blla's polygons along the baseline, as kraken itself does.
- **Layout, detection, table and formula models are modern-document models.** Text they miss is
  recovered by the page-level pass, but region labels, reading order on unusual pages and table
  detection are not guaranteed on historical layouts. Tables with vertical rules and no horizontal
  ones come back as one cell per column: every PaddleX table model reads them that way.
- **Inline formulas are read as text.** Only display formula regions go to PP-FormulaNet; a formula
  inside a text line is kraken's to read. Display lines that mix words and formulas produce poor
  LaTeX for the words.
- **Docling exports are region-level and box-based**; use `hocr`, `alto` or `page` for line
  geometry. Pictures and other line-less regions are appended after the text regions in those
  files (kraken's serialiser does that), not at their reading-order position.
- Seals and stamps are exported as pictures.

## 8. Licence, credit and citation

SquiddleOCR is Apache-2.0. The recognition and segmentation models are kraken's, by Benjamin
Kiessling, Apache-2.0, fetched from Zenodo:

| model | DOI |
|---|---|
| PP-OCRv6 tiny | 10.5281/zenodo.21788403 |
| PP-OCRv6 small | 10.5281/zenodo.21788405 |
| PP-OCRv6 medium | 10.5281/zenodo.21788410 |
| blla segmentation | 10.5281/zenodo.14602569 |

Please cite kraken and the DOI of the model you use. The PaddleX models (PP-DocLayoutV3, PP-OCRv6
detection, the table pipeline's SLANet and RT-DETR models, PP-FormulaNet) are Apache-2.0 from
PaddlePaddle.
