# SquiddleOCR

Document OCR for historical print: PaddleX layout analysis, text detection and table structure in
front of [kraken](https://kraken.re)'s PP-OCRv6 recogniser, out as a `DoclingDocument` (Markdown,
HTML, DocLang, JSON) and as hOCR, ALTO or PAGE-XML written by kraken's serialiser.

```
pip install "squiddleocr[paddle]"
squiddle ocr scans/
```

That reads every image in `scans/` and writes `scans/<name>.paddle.md` next to it: Markdown with
reading order and tables. `-o out/` collects the outputs elsewhere, `-f md,hocr,page` adds formats.
Models are downloaded on first use: kraken's from Zenodo into kraken's model cache, PaddleX's into
PaddleX's.

The recognisers are Benjamin Kiessling's kraken PP-OCRv6 models (Apache-2.0). Nothing is converted
or retrained: recognition is kraken's own code on kraken's own weights.

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

Python 3.11 or 3.12. The core package brings kraken with torch, docling-core and ONNX Runtime.

| extra | adds | you need it for |
|---|---|---|
| `paddle` | paddlex, paddlepaddle (CPU build) | layout analysis, PP-OCRv6 text detection and tables; without it use `--pipeline kraken` |
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
| `-m, --model SIZE\|FILE` | `medium` | kraken PP-OCRv6 recogniser: `tiny` (0.7M parameters), `small` (3.2M), `medium` (15.8M, most accurate), fetched by DOI; or a path to a kraken model file |
| `-o, --output DIR` | next to each image | where the exports go, one file set per image, named after it |
| `-f, --formats LIST` | `md` | document level: `md` (Markdown, tables as HTML), `doclang` (DocLang XML), `html`, `json` (lossless DoclingDocument), `txt`. Line level, one file per image: `hocr`, `alto`, `page` (PAGE-XML) |
| `--pipeline paddle\|kraken` | `paddle` | `paddle` = PaddleX layout analysis, PP-OCRv6 text detection (line boxes), tables and formulas; every format. `kraken` = kraken's blla segmenter on the whole page (polygons and baselines, kraken's own line order; no layout, tables or formulas), what the `kraken` command does; formats `hocr`, `alto`, `page`, `txt`. Recognition is kraken's either way |
| `--det-model NAME` | `PP-OCRv6_medium_det` | `PP-OCRv6_small_det` or `PP-OCRv6_tiny_det` for speed (`--pipeline paddle`) |
| `--unclip-ratio X` | `2.0` | expansion of PP-OCRv6 line boxes; PaddleOCR's default 1.5 clips ascenders and line-final hyphens on old print |
| `--layout paddle\|none` | `paddle` | `none` treats the page as one text block (no layout models, tables or formulas) |
| `--layout-model NAME` | `PP-DocLayoutV3` | `PP-DocLayout_plus-L` (PP-StructureV3's layout model, XY-cut reading order) |
| `--tables / --no-tables` | on | recognise table structure in table regions (SLANet_plus) |
| `--detail line\|word\|glyph` | `glyph` | depth of `hocr`, `alto`, `page`: `glyph` = words and glyphs from kraken's character cuts (kraken's default), `word` = words without glyphs (ALTO `String`, PAGE `Word`; kraken's templates minus the `Glyph` elements), `line` = text per line (kraken's `--no-subline-segmentation`). hOCR has no glyph elements, so `word` and `glyph` coincide there |
| `--formulas / --no-formulas` | on | read formula regions as LaTeX (PP-FormulaNet_plus-L on torch; `--formula-model` picks PP-FormulaNet-L instead) |
| `--batch-size N` | `8` | lines per kraken forward pass |
| `--device auto\|cpu\|cuda\|tensorrt\|coreml` | `auto` | ONNX Runtime provider for the PaddleX models; `cpu` or `auto` for kraken's torch models |
| `--per-document` | off | one document for all inputs (a book) instead of one per image |
| `--suffix TAG` | `auto` | tag between name and extension, `<name>.<tag>.md`; `auto` is the pipeline name, so paddle and kraken runs sit side by side as `<name>.paddle.md` and `<name>.kraken.md`; `none` gives `<name>.md` |

Library chatter is off by default (PaddleX's model notes, ONNX Runtime's initialiser warnings, kraken's per-line polygonizer warnings and the PIL warning that follows them); `SQUIDDLE_VERBOSE=1` shows all of it.

```
squiddle ocr page.jpg -f md,doclang,json                 # one page, Markdown + DocLang + JSON
squiddle ocr book/ --per-document -f doclang             # whole book as one DocLang file, book/book.paddle.doclang.xml
squiddle ocr scans/ --pipeline kraken -f hocr,page       # kraken's own pipeline: blla on the whole page, polygons, baselines, words, glyphs
squiddle ocr scans/ --layout none -m tiny                # fastest: plain OCR with the tiny recogniser
```

## 3. Python API

```python
from squiddleocr.factory import build_pipeline
from squiddleocr.document import export

pipe = build_pipeline("medium", pipeline="paddle", layout="paddle", tables=True, formulas=True, device="auto")
doc = pipe.run_files(["scans/0001.jpg", "scans/0002.jpg"], name="book")   # a DoclingDocument
print(doc.export_to_markdown())
export(doc, "out/", "book", ["doclang", "json"])
```

Or assemble the components yourself:

```python
from squiddleocr.pipeline import Pipeline
from squiddleocr.recognizers.kraken import KrakenRecognizer
from squiddleocr.models import resolve_kraken_model
from squiddleocr.detectors.paddle import PaddleTextDetector
from squiddleocr.layout.paddle import PaddleLayout
from squiddleocr.tables.paddle import PaddleTableRecognizer

pipe = Pipeline(
    recognizer=KrakenRecognizer(resolve_kraken_model("medium")),
    detector=PaddleTextDetector("PP-OCRv6_medium_det", unclip_ratio=2.0),
    layout=PaddleLayout("PP-DocLayoutV3"),
    tables=PaddleTableRecognizer(),
)
contents = pipe.process_page(page)        # per-region lines, kraken records and table cells
```

`KrakenRecognizer.predict(image, segmentation)` is kraken's `RecognitionTaskModel.predict` and can
be used on its own with any kraken `Segmentation`.

## 4. How it works

```
page ─► LayoutAnalyzer (PaddleX PP-DocLayoutV3) ─► regions: label, polygon, reading order
          │
          ├─► TextDetector, once per page ─► lines, assigned to the region they overlap most;
          │     paddle: PP-OCRv6 detection boxes       unclaimed lines become text regions
          │     kraken: blla polygons + baselines on the whole page, blla's line order,
          │             the page is the one region (no layout analysis, no tables)
          │
          ├─► segmentation.py: lines ─► kraken Segmentation (BBoxLine / BaselineLine, ids <region>_l<n>)
          │
          ├─► kraken RecognitionTaskModel.predict ─► one ocr_record per line: text, character cuts, confidences
          │
          ├─► table regions ─► TableRecognizer (SLANet_plus cells) ─► cell lines read the same way
          └─► formula regions ─► FormulaRecognizer (PP-FormulaNet) ─► LaTeX
                                                            │
                          DocumentBuilder ─► DoclingDocument ─► md / html / doclang / json / txt
                          serialize.py ─► kraken.serialization.serialize ─► hocr / alto / page
```

Each stage is a `typing.Protocol` with one method, in `squiddleocr/<stage>/base.py`:

| protocol | method | implementations |
|---|---|---|
| `Recognizer` | `recognize_lines(page, lines) -> [ocr_record]` | `KrakenRecognizer` (kraken's `RecognitionTaskModel`) |
| `TextDetector` | `detect(page, region=None) -> [TextLine]` | `PaddleTextDetector` (PP-OCRv6 det), `KrakenSegmenter` (blla) |
| `LayoutAnalyzer` | `analyze(page) -> [Region]` | `PaddleLayout` (PP-DocLayoutV3 with learned reading order; PP-DocLayout_plus-L + XY-cut), `SingleRegionLayout` |
| `TableRecognizer` | `structure(page, region) -> TableResult` | `PaddleTableRecognizer` (SLANet_plus) |
| `FormulaRecognizer` | `recognize(page, regions) -> [latex]` | `PaddleFormulaRecognizer` (PP-FormulaNet_plus-L, PaddleX's transformers engine on torch) |

Adding a layout model means returning `Region`s with Docling labels (`text`, `title`,
`section_header`, `caption`, `footnote`, `page_header`, `page_footer`, `table`, `picture`, `formula`,
`list_item`, `code`, `reference`) and an `order`, or leaving the order to
`squiddleocr.layout.order.xy_cut_order`. Adding a segmenter means returning `TextLine` polygons in
page coordinates, with baselines when the model has them. `build_pipeline` and the CLI learn the
new name in `factory.py`.

What the pipeline guarantees: one detection pass per page (or per region with
`detect_per_region=True`), lines assigned to the region they overlap most, orphan lines turned into
`text` regions slotted into the reading order so page numbers survive a layout miss, row-wise line
ordering with de-duplication of detector fragments, one batched kraken recognition call per page,
cell-by-cell table reading, and a `DoclingDocument` whose body order is the reading order and whose
provenance boxes are the regions.

The rule behind the design: reuse kraken at the highest level it offers and never reimplement a
slice of it. Line extraction, batching, decoding, character positions and XML serialisation are
kraken's; SquiddleOCR adds layout, detection, tables, assignment and the Docling side.

## 5. Exports

Two levels, from the same results:

- **Document level** (`md`, `html`, `doclang`, `json`, `txt`): the `DoclingDocument`. Docling has no
  line level, so each region is one text item with the region's bounding box, in reading order,
  tables with their cells, pictures as placeholders.
- **Line level** (`hocr`, `alto`, `page`): kraken's serialiser, given the regions and kraken's records.
  Every line carries its geometry, its text, and (`--detail`) words and glyphs derived from the
  character cuts; with `--pipeline kraken` also the boundary polygon and the baseline. Regions are
  tagged with their Docling label (`custom="type {type:text;}"` in PAGE, `TAGREFS` in ALTO).
  With `--pipeline kraken` these files are what the `kraken` command itself writes: blla on
  the whole page, kraken's records, kraken's line order. With `--pipeline paddle` the lines are
  grouped by our layout regions.

Word and glyph boxes are derived from CTC time steps, about 8 input pixels of resolution, in both
kraken and PaddleOCR; they are not a trained word detector.

## 6. GPU and devices

`--device auto` uses CUDA when ONNX Runtime and torch find it. kraken's models run on torch,
PaddleX's on ONNX Runtime; the two do not share memory but both use the GPU. On aarch64 (DGX
Spark, Jetson) everything runs: torch ships cu130 wheels, ONNX Runtime GPU builds, and the
PaddlePaddle CPU wheel is only imported for PaddleX's bookkeeping (its inference engine is never
used). macOS gets torch MPS/CPU for kraken and CoreML/CPU for PaddleX; untested from here.

## 7. Known limitations

- **Crops decide a lot.** With `--pipeline paddle`, kraken reads the axis-aligned box of each
  detected line; boxes that cut ascenders, descenders or the line-final `⸗` make the recogniser read
  `-`, and very short table cells can flip a Latin word to Cyrillic. `--unclip-ratio` and the
  pipeline's cell padding mitigate this. `--pipeline kraken` reads blla's polygons along the
  baseline, as kraken itself does.
- **Layout and detection models are modern-document models.** PP-DocLayoutV3 and the PP-OCRv6
  detector were trained on modern material. Text they miss is recovered by the page-level pass,
  but region labels, reading order on unusual pages and table detection are not guaranteed on
  historical layouts.
- **Docling exports are region-level and box-based**; use `hocr`, `alto` or `page` for line geometry.
  Pictures and other line-less regions are appended after the text regions in those files
  (kraken's serialiser does that), not at their reading-order position.
- No handling of seals and stamps yet (PP-DocLayoutV3 detects `seal` regions; they are exported as
  pictures).

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
detection, SLANet_plus, PP-FormulaNet) are Apache-2.0 from PaddlePaddle.
