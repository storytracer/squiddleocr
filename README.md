# SquiddleOCR

**Historical documents, modern pipeline.** SquiddleOCR combines [kraken](https://kraken.re)'s
PP-OCRv6 text recognisers, trained on historical print and handwriting in 44 languages and 10
scripts, with pluggable layout analysis, text detection and table recognition models, and writes
the result as a [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/):
DocLang, Markdown, HTML or JSON, with reading order, region labels, boxes and tables.

```
uv tool install "squiddleocr[paddle] @ git+https://github.com/storytracer/squiddleocr"
squiddle ocr scans/
```

That reads every image in `scans/` and writes `scans/<name>.paddle.md` next to it, Markdown with reading
order and tables; `-o out/` collects the outputs elsewhere and `-f doclang,json,html` adds formats.
The recogniser (about 64 MB) and the layout, detection and table models are downloaded on first use.

The recognisers are Benjamin Kiessling's kraken PP-OCRv6 models (Apache-2.0), converted to ONNX
with kraken's line preprocessing folded into the graph and published at
[huggingface.co/storytracer/squiddleocr](https://huggingface.co/storytracer/squiddleocr). Nothing is
retrained. Each converted directory is also a standard PaddleX recognition model.

## Contents

1. [Install](#1-install)
2. [Command line](#2-command-line)
3. [Python API](#3-python-api)
4. [Architecture and extending it](#4-architecture-and-extending-it)
5. [Models](#5-models)
6. [Using the recogniser in PaddleX](#6-using-the-recogniser-in-paddlex)
7. [GPU and devices](#7-gpu-and-devices)
8. [Known limitations](#8-known-limitations)
9. [Licence, credit and citation](#9-licence-credit-and-citation)

## 1. Install

Python 3.11 or 3.12. As a command-line tool, with [uv](https://docs.astral.sh/uv/):

```
uv tool install "squiddleocr[paddle] @ git+https://github.com/storytracer/squiddleocr"
squiddle ocr scans/
```

This gives `squiddle` its own isolated environment and puts it on your PATH; `uv tool upgrade
squiddleocr` updates it. As a library, or into an existing environment:

```
uv pip install "squiddleocr[paddle] @ git+https://github.com/storytracer/squiddleocr"    # or: pip install ...
```

For development, from a checkout: `uv sync --extra paddle --extra test` and `source .venv/bin/activate`.

The core package brings ONNX Runtime (`onnxruntime-gpu` with the CUDA 13 runtime libraries on
Linux and Windows, `onnxruntime` with CoreML on macOS), `docling-core`, NumPy, Pillow, OpenCV and
`huggingface_hub`. Extras add optional components:

| extra | adds | you need it for |
|---|---|---|
| `paddle` | paddlex, paddlepaddle (CPU build) | layout analysis, text detection and tables (PP-DocLayout, PP-OCRv6 det, SLANet) |
| `kraken` | kraken, torch | kraken's blla segmenter as the line detector (`--detector kraken`) |
| `convert` | torch, kraken, onnx, onnxscript | converting kraken safetensors models yourself (`squiddle convert`) |

Everyday use needs `paddle` only. The `kraken` and `convert` extras install kraken from its git
repository (the PP-OCRv6 code is not in a PyPI release yet), which needs `git`.

## 2. Command line

`squiddle --help` lists the commands; every command has `--help`.

### `squiddle ocr`: read images

```
squiddle ocr INPUTS... [options]
```

`INPUTS` are image files (PNG, JPEG, TIFF, WebP, BMP) or folders of them. Defaults: medium
recogniser, PaddleX layout, PP-OCRv6 detector, tables on, Markdown written next to each image.

| option | default | meaning |
|---|---|---|
| `-o, --output DIR` | next to each image | where the exports go, one file set per image, named after it |
| `--suffix TAG` | `auto` | tag between name and extension, `<name>.<tag>.md`; `auto` is the detector name, so `--detector paddle` and `--detector kraken` runs sit side by side as `<name>.paddle.md` and `<name>.kraken.md`; `none` gives `<name>.md` |
| `-f, --formats LIST` | `md` | any of `md` (Markdown, tables as HTML), `doclang` (DocLang XML), `html`, `json` (lossless DoclingDocument), `txt` |
| `-m, --model SIZE\|DIR` | `medium` | recogniser: `tiny` (0.7M parameters, 3 MB), `small` (3.2M, 14 MB), `medium` (15.8M, 64 MB, most accurate), or a model directory |
| `--models SOURCE` | `storytracer/squiddleocr` | where sizes come from: a Hub repo or a local folder from `squiddle convert` (env `SQUIDDLE_MODELS`) |
| `--layout paddle\|none` | `paddle` | `none` treats the page as one text block (plain OCR, no layout models) |
| `--detector paddle\|kraken` | `paddle` | text line detector: PP-OCRv6 detection, or kraken's blla segmenter (`kraken` extra); kraken lines are cut with kraken's own polygon extraction, byte-identical to what kraken itself recognises |
| `--det-model NAME` | `PP-OCRv6_medium_det` | `PP-OCRv6_small_det` or `PP-OCRv6_tiny_det` for speed |
| `--layout-model NAME` | `PP-DocLayoutV3` | `PP-DocLayout_plus-L` (PP-StructureV3's layout model, XY-cut reading order) |
| `--unclip-ratio X` | `2.0` | how much the detector's boxes are expanded; PaddleOCR's default 1.5 clips ascenders and line-final hyphens on old print |
| `--tables / --no-tables` | on | recognise table structure in table regions |
| `--batch-size N` | `8` | lines per recogniser call; `1` reproduces kraken's single-line output exactly, larger is faster |
| `--device auto\|cpu\|cuda\|tensorrt\|coreml` | `auto` | ONNX Runtime execution provider for every model; `auto` is CUDA when available, otherwise CPU (CoreML is opt-in) |
| `--per-document` | off | one document for all inputs (a book) instead of one per image |

Output text is kraken's diplomatic transcription: NFD Unicode, long s (ſ), combining diacritics,
`⸗` hyphens, historical orthography. Normalise afterwards if you need NFC or modern letterforms.

Examples:

```
squiddle ocr page.jpg -f md,doclang,json             # one page, Markdown + DocLang + JSON
squiddle ocr book/ --per-document -f doclang         # whole book as one DocLang file, book/book.paddle.doclang.xml
squiddle ocr scans/ --detector kraken -f md,txt      # scans/<name>.kraken.md beside <name>.paddle.md for comparison
squiddle ocr scans/ --layout none -m tiny            # fastest: plain OCR with the tiny recogniser
squiddle ocr scans/ --detector kraken --layout none  # kraken segmentation, SquiddleOCR recognition
squiddle ocr scans/ --models ./squiddleocr-models    # recognisers from a local folder
```

### `squiddle models`: recognisers

```
squiddle models list [--models SOURCE]        # sizes available locally (folder, or the cache of a repo)
squiddle models pull [SIZE...] [--models REPO] # download ahead of time (all sizes by default)
```

The cache lives in `~/.cache/squiddleocr/` (`SQUIDDLE_HOME` overrides it).

### `squiddle convert` and `squiddle upload`: make and publish recognisers

```
squiddle convert [tiny|small|medium|FILE.safetensors ...] [-o squiddleocr-models] [--repo you/squiddleocr]
squiddle upload squiddleocr-models [--repo you/squiddleocr] [--private]
```

`convert` (needs the `convert` extra) turns kraken PP-OCRv6 models into a *model source folder*:
all three sizes from kraken's Hub mirror when nothing is named, or the sizes or `.safetensors`
files you name. The folder has the Hub repo layout, `README.md` (a model card) and
`models/squiddle_PP-OCRv6_<size>_rec/`, so it can be used directly (`--models FOLDER`) or
published with `upload`, which creates the repo if needed. Each model directory contains
`inference.onnx`, `inference.yml`, `dict.txt`, the original kraken `MODEL_CARD.md`, `NOTICE`
(attribution, DOI), `LICENSE` and `squiddle.json` (source hash, versions, parity check).

### `squiddle verify` and `squiddle extract-lines`: check a conversion

```
squiddle extract-lines page1.jpg page2.jpg -o lines/     # kraken segmentation -> one PNG per line
squiddle verify MODEL_DIR lines/ [--batch-size 8] [--paddle] [--report r.json]
```

`verify` runs the lines through kraken and through the ONNX model (plus PaddleX's own predictor
with `--paddle`) and reports exact-match rate and character error rate against kraken. Exact
agreement at batch size 1 is the acceptance criterion for a conversion.

## 3. Python API

```python
from squiddleocr.factory import build_pipeline
from squiddleocr.document import export

pipe = build_pipeline("medium")                      # same defaults as the CLI; models="folder-or-repo" to change the source
doc = pipe.run_files(["page1.jpg", "page2.jpg"])     # a DoclingDocument
print(doc.export_to_markdown())
export(doc, "out/", "book", ["doclang", "json"])
```

Assembling a pipeline from components:

```python
from squiddleocr.models import resolve_model
from squiddleocr.pipeline import Pipeline
from squiddleocr.recognizers import OnnxRecognizer
from squiddleocr.detectors.paddle import PaddleTextDetector
from squiddleocr.layout.paddle import PaddleLayout
from squiddleocr.tables.paddle import PaddleTableRecognizer
from squiddleocr.types import Page

pipe = Pipeline(
    recognizer=OnnxRecognizer(resolve_model("medium"), device="cuda", batch_size=8),
    detector=PaddleTextDetector("PP-OCRv6_medium_det", unclip_ratio=2.0),
    layout=PaddleLayout(),
    tables=PaddleTableRecognizer(),
)
page = Page.load("page.jpg")
for content in pipe.process_page(page):   # per region: content.region (label, polygon, order), .lines, .texts, .table
    print(content.region.order, content.region.label, content.text[:60])
```

`OnnxRecognizer.recognize(list_of_line_images)` can be used on its own for line-level OCR.

## 4. Architecture and extending it

```
Page ─► LayoutAnalyzer ─► regions (label, polygon, reading order)
          │
          ├─► TextDetector (once per page) ─► lines, assigned to regions; unclaimed lines become regions
          │                                      │
          │                                      └─► Recognizer ─► text, batched across regions
          │
          └─► table regions ─► TableRecognizer (cells) ─► each cell read by detector + recogniser
                                                            │
                                            DocumentBuilder ─► DoclingDocument ─► DocLang / Markdown / HTML / JSON
```

Each stage is a `typing.Protocol` with one method, in `squiddleocr/<stage>/base.py`:

| protocol | method | implementations |
|---|---|---|
| `Recognizer` | `recognize(line_images) -> [Recognition]` | `OnnxRecognizer` (the converted kraken model) |
| `TextDetector` | `detect(page, region=None) -> [TextLine]` | `PaddleTextDetector` (PP-OCRv6 det), `KrakenSegmenter` (blla, cuts lines with kraken's `extract_polygons`) |
| `LayoutAnalyzer` | `analyze(page) -> [Region]` | `PaddleLayout` (PP-DocLayoutV3 with its learned reading order and polygons; PP-DocLayout_plus-L + XY-cut order), `SingleRegionLayout` |
| `TableRecognizer` | `structure(page, region) -> TableResult` | `PaddleTableRecognizer` (SLANet_plus) |

**Adding a layout model**: implement `analyze`, return `Region`s with Docling labels (`text`,
`title`, `section_header`, `caption`, `footnote`, `page_header`, `page_footer`, `table`, `picture`,
`formula`, `list_item`, `code`, `reference`) and set `order`, or leave it to
`squiddleocr.layout.order.xy_cut_order`; `suppress_contained` removes overlapping duplicates.
**Adding a segmenter**: return `TextLine` polygons in page coordinates (baselines optional); define
`line_images(page, lines)` if the recogniser should see the segmenter's own cuts instead of a
perspective crop (four points) or polygon mask.
**Adding a table model**: return cells with row, column, spans and boxes; the pipeline reads them.
Nothing else changes, and `build_pipeline` / the CLI can be taught the new name in `factory.py`.

What `Pipeline` guarantees: one detection pass per page (or per region with
`detect_per_region=True`), lines assigned to the region they overlap most, orphan lines turned
into `text` regions so page numbers survive a layout miss (slotted into the layout model's
reading order after the region above them), row-wise line ordering with
de-duplication of detector fragments, one batched recognition call, cell-by-cell table reading,
and a `DoclingDocument` whose body order is the reading order and whose provenance boxes are the
regions.

## 5. Models

| size | parameters | ONNX | kraken test CER (macro) | Zenodo DOI |
|---|---|---|---|---|
| tiny | 0.7M | 3 MB | 8.7 % (11.0 %) | 10.5281/zenodo.21788403 |
| small | 3.2M | 14 MB | 5.4 % (6.9 %) | 10.5281/zenodo.21788405 |
| medium | 15.8M | 64 MB | 3.9 % (4.9 %) | 10.5281/zenodo.21788410 |

All three read 44 languages in 10 scripts (Arabic, Armenian, Cyrillic, Ethiopic, Georgian, Greek,
Hebrew, Latin, Malayalam, Syriac), printed and handwritten, with the same 1622-character
alphabet. The CER column is kraken's own evaluation on its held-out test set (from the model
cards).

**The conversion.** kraken feeds its network 96 px lines scaled to 0..1 and inverted (ink
bright), with 16 px of white padding at both ends, and batches lines with white padding plus an
attention mask on the padded time steps. `squiddle convert` folds all of that into the ONNX graph:
the graph takes PaddleOCR's `[-1, 1]` input (`0.5 - 0.5·x` equals `1 - x/255`), pads, detects
batch padding (trailing all-zero columns, a value real pixels cannot take), masks it as kraken
does, forces those time steps to the CTC blank and emits `(batch, time, classes)` probabilities.
`OnnxRecognizer` therefore only resizes and normalises. `squiddle verify` shows the export
reproducing kraken line for line at batch size 1 and kraken's own batched behaviour at larger
batches.

**Model sources.** A source is a folder or Hub repo laid out as `README.md` +
`models/squiddle_PP-OCRv6_<size>_rec/`. The default is `storytracer/squiddleocr`; `--models`
or `SQUIDDLE_MODELS` selects another. When a source lacks a size and the `convert` extra is
installed, kraken's weights are fetched from their Hub mirror (`small-models-for-glam/kraken-ppocrv6-<size>`)
and converted locally.

## 6. Using the recogniser in PaddleX

Each converted directory is a PaddleX text recognition model: `inference.onnx` plus an
`inference.yml` registered under the PP-OCRv6 name PaddleX knows. It loads with PaddleX's ONNX
Runtime engine and nothing else:

```python
from paddlex.inference import create_predictor
rec = create_predictor("PP-OCRv6_medium_rec", model_dir="models/squiddle_PP-OCRv6_medium_rec", engine="onnxruntime")
```

`squiddle verify --paddle` checks that this path gives the same text as the SquiddleOCR
recogniser. SquiddleOCR's own pipeline is the supported way to read whole pages; the PaddleOCR
PP-StructureV3 drop-in it once shipped was removed because the native pipeline does everything it
did (and uses PP-DocLayoutV3, which PP-StructureV3 cannot).

## 7. GPU and devices

`--device auto` uses CUDA when `onnxruntime-gpu` finds an NVIDIA GPU (TensorRT with
`--device tensorrt`) and the CPU otherwise. The CUDA 13 runtime, cuBLAS and cuDNN 9 come from the
`nvidia-*` pip packages and are found automatically. Every model of the pipeline runs on the
chosen provider; if an accelerated provider fails while running a model, that model falls back
to the CPU with a warning instead of aborting. On a DGX Spark (GB10, aarch64) a text page takes
about 0.7 s and a table page 2 to 3 s; the recogniser alone is 13x faster than on the 20 CPU cores.

**macOS**: the default is the CPU. Apple's CoreML provider is available as `--device coreml`
but is experimental: it partitions the recogniser's dynamic-width graph and has failed at run
time on Apple Silicon, which the fallback now catches. A page takes a few seconds on the CPU of
an Apple Silicon Mac.

## 8. Known limitations

- **Crops decide a lot.** Detector boxes and table cells that cut ascenders, descenders or the
  line-final `⸗` make the recogniser read `-`, and very short cell crops can flip a Latin word
  to Cyrillic. `--unclip-ratio` and the pipeline's cell padding mitigate this; both are measured in `NOTES.md`.
  `--detector kraken` sidesteps it: its lines are cut exactly as kraken cuts them for its own
  recognition, which is what the recogniser was trained on.
- **Layout and detection models are modern-document models.** PP-DocLayout and the PP-OCRv6
  detector were trained on modern material. Text they miss is recovered by the page-level
  detection pass, but region labels, reading order on unusual pages and table detection are
  not guaranteed on historical layouts.
- **Batching changes a few lines** (long s versus s at line ends, as in kraken's own batch
  mode); `--batch-size 1` gives kraken's exact single-line output.
- **Preprocessing mismatch degrades accuracy silently.** Anything that feeds the recogniser
  differently from kraken still yields plausible text; `squiddle verify` is the check.
- No handling of seals and stamps yet (PP-DocLayoutV3 detects `seal` regions; they are exported
  as pictures).

## 9. Licence, credit and citation

Recognition models: Apache-2.0, © Benjamin Kiessling (ALMAnaCH, Inria Paris), trained with
support of the ATRIUM and MiDRASH projects; the original model cards, with their dataset
credits, ship in every converted directory. When you use SquiddleOCR, cite kraken and the
Zenodo DOI of the model size you used (table above). The `squiddle_` naming is this project's
and not the author's. Code: Apache-2.0. Bundled PaddleX pipeline templates: Apache-2.0
(PaddlePaddle).
