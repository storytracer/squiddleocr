# SquiddleOCR

Document OCR for historical print, built on [kraken](https://kraken.re)'s PP-OCRv6 recogniser, with
three pipelines:

- **paddle**: PaddleX layout analysis with learned reading order, PP-OCRv6 text detection, table
  structure and formula recognition, out as a `DoclingDocument` (Markdown, HTML, DocLang, JSON,
  text) and as hOCR, ALTO or PAGE-XML written by kraken's serialiser.
- **kraken**: kraken's own pipeline, blla segmentation on the whole page and kraken's line order,
  out as hOCR, ALTO, PAGE-XML or text, exactly what the `kraken` command writes.
- **eynollah**: [eynollah](https://github.com/qurator-spk/eynollah), SBB's layout analyser for
  historical print, for regions and reading order (run as a subprocess with a resource plan for the
  machine), PP-OCRv6 text detection inside each of its text regions, kraken reading the lines; the
  same exports as the paddle pipeline.

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
7. [eynollah](#7-eynollah)
8. [Known limitations](#8-known-limitations)
9. [Licence, credit and citation](#9-licence-credit-and-citation)

## 1. Install

Python 3.11 or 3.12. The core package brings kraken with torch, docling-core and ONNX Runtime and
runs the kraken pipeline.

| extra | adds | you need it for |
|---|---|---|
| `paddle` | paddlex, paddlepaddle, transformers | the paddle pipeline: layout analysis, PP-OCRv6 text detection, tables, formulas |
| `eynollah` | eynollah (with ocrd), psutil | the eynollah pipeline: `--layout eynollah`, see [eynollah](#7-eynollah) |
| `test` | pytest | the test-suite |

On Linux and Windows the ONNX Runtime GPU build and the CUDA 13 runtime libraries are installed
as pip packages; nothing else is needed for the GPU. macOS gets the CPU/CoreML build.

Development checkout with [uv](https://docs.astral.sh/uv/):

```
uv sync --extra paddle --extra eynollah --extra test
.venv/bin/python -m pytest -q
```

## 2. Command line

```
squiddle ocr INPUTS... [options]
```

`INPUTS` are image files (PNG, JPEG, TIFF, WebP, BMP) or folders of them.

| option | default | meaning |
|---|---|---|
| `--pipeline paddle\|kraken\|eynollah` | `paddle` | `paddle` = PaddleX layout analysis, PP-OCRv6 text detection, tables and formulas; every format. `kraken` = blla on the whole page (polygons and baselines, kraken's own line order; no layout, tables or formulas); formats `hocr`, `alto`, `page`, `txt`. `eynollah` = eynollah's regions, lines and reading order (same as `--layout eynollah`); every format |
| `-m, --model SIZE\|FILE` | `medium` | kraken PP-OCRv6 recogniser: `tiny` (0.7M parameters), `small` (3.2M), `medium` (15.8M, most accurate), fetched by DOI; or a path to a kraken model file |
| `-o, --output DIR` | next to each image | where the exports go, one file set per image, named after it |
| `-f, --formats LIST` | `auto` | `md` for paddle, `hocr` for kraken. Document level: `md` (Markdown; a table with spanning cells is an HTML table), `doclang` (DocLang XML), `html`, `json` (lossless DoclingDocument), `txt`. Line level, one file per image: `hocr`, `alto`, `page` (PAGE-XML) |
| `--detail line\|word\|glyph` | `glyph` | depth of `hocr`, `alto`, `page`: `glyph` = words and glyphs from kraken's character cuts (kraken's default), `word` = words without glyphs, `line` = text per line. See [Exports](#5-exports) |
| `--layout paddle\|none\|eynollah` | `paddle` | `none` treats the page as one text block (no layout model, tables or formulas); `eynollah` runs eynollah for regions and order, see [eynollah](#7-eynollah) |
| `--layout-model NAME` | `PP-DocLayoutV3` | `PP-DocLayout_plus-L` (PP-StructureV3's layout model, XY-cut reading order) |
| `--det-model NAME` | `PP-OCRv6_medium_det` | `PP-OCRv6_small_det` or `PP-OCRv6_tiny_det` for speed |
| `--unclip-ratio X` | `2.0` | expansion of PP-OCRv6 line boxes; PaddleOCR's default 1.5 clips ascenders and line-final hyphens on old print |
| `--tables / --no-tables` | on | cell structure of table regions with PaddleX's table pipeline, fed with our text lines and kraken's readings; cell text is kraken's |
| `--formulas / --no-formulas` | on | formula regions as LaTeX with PP-FormulaNet_plus-L (`--formula-model` picks PP-FormulaNet-L) |
| `--batch-size N` | `8` | lines per kraken forward pass |
| `--device auto\|cpu\|cuda\|tensorrt\|coreml` | `auto` | ONNX Runtime provider for the PaddleX models; `cpu` or `auto` for kraken's torch models |
| `--per-document` | off | one document for all inputs (a book) instead of one per image |
| `--rtl` | off | right-to-left script: kraken reads lines right to left, eynollah orders regions right to left (`-r2l`) |
| `--eynollah-lines paddle\|eynollah\|blla` | `paddle` | line stage of the eynollah pipeline: PP-OCRv6 detection on each eynollah text region (`--det-model`, `--unclip-ratio` apply), eynollah's own line polygons, or blla per region |
| `--baselines / --no-baselines` | on | with `--eynollah-lines eynollah`: read along a baseline synthesised inside the polygon (kraken dewarps by the polygon), or as boxes |
| `--eynollah-jobs N` | auto | parallel eynollah page jobs; auto = from cores, RAM and page size, at most 8 |
| `--eynollah-device SPEC` | auto | eynollah's `-D`: `GPU`, `GPU0`, `CPU` or per-model globs `col*:CPU,*:GPU0` |
| `--eynollah-vram-margin X` | `20%` or 4 GB | GPU memory kept free of eynollah's models: a fraction (`0.2`, `20%`) or gigabytes (`4`, `4G`) |
| `--eynollah-tensorrt` | off | let eynollah use ONNX Runtime's TensorRT provider when `libnvinfer` is loadable |
| `--eynollah-args ARGS` | `-fl -romb` | flags for `eynollah layout`, see the table in [eynollah](#7-eynollah) |
| `--eynollah-xml DIR` | temporary | directory of eynollah PAGE-XML: pages that have a file there are consumed without running eynollah, the rest are written into it |
| `--suffix TAG` | `auto` | tag between name and extension, `<name>.<tag>.md`; `auto` is the pipeline name, so paddle, kraken and eynollah runs sit side by side; `none` gives `<name>.md` |

Library chatter is off by default (PaddleX's model notes, ONNX Runtime's initialiser warnings,
kraken's per-line polygonizer warnings and the PIL warning that follows them); `SQUIDDLE_VERBOSE=1`
shows all of it.

```
squiddle ocr page.jpg -f md,doclang,json                 # one page, Markdown + DocLang + JSON
squiddle ocr book/ --per-document -f doclang             # whole book as one DocLang file, book/book.paddle.doclang.xml
squiddle ocr scans/ --pipeline kraken -f hocr,page       # kraken's own pipeline: polygons, baselines, words, glyphs
squiddle ocr scans/ -f alto --detail word                # ALTO with Strings but no Glyphs
squiddle ocr scans/ --layout none -m tiny                # fastest: plain OCR with the tiny recogniser
squiddle ocr scans/ --layout eynollah -f md,page         # eynollah regions and order, PP-OCRv6 lines, kraken text
squiddle models pull eynollah medium                     # fetch models ahead of a run (models path NAME prints where they are)
```

`squiddle models pull NAME...` downloads `eynollah` (the layout bundle), `tiny`/`small`/`medium`
(kraken's recognisers) or `blla` (kraken's segmenter) ahead of a run; `squiddle models path NAME`
prints where a model lives.

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

The eynollah pipeline takes its run options as an `EynollahOptions`; `prepare` runs eynollah on the
whole batch up front (its page jobs run in parallel), otherwise each page is run when first asked
for, and `close` removes the temporary XML directory:

```python
from squiddleocr.eynollah import EynollahOptions

pipe = build_pipeline("medium", pipeline="eynollah", eynollah=EynollahOptions(args="-fl -romb -tab", xml_dir="work/eyn_xml"))
pipe.prepare(["scans/0001.jpg", "scans/0002.jpg"])
doc = pipe.run_files(["scans/0001.jpg", "scans/0002.jpg"], name="book")
pipe.close()
```

## 4. How it works

```
page ─► LayoutAnalyzer (PP-DocLayoutV3) ─► regions: label, polygon, reading order
          │                                (kraken pipeline: the page is the one region;
          │                                 eynollah pipeline: eynollah's PAGE-XML regions and order)
          ├─► TextDetector, once per page ─► lines, assigned to the region they overlap most;
          │     paddle: PP-OCRv6 detection boxes       unclaimed lines become text regions
          │     kraken: blla polygons + baselines, blla's own line order
          │     eynollah: PP-OCRv6 detection on each region's masked crop (or eynollah's polygons, or blla)
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
| `TextDetector` | `detect(page, region=None) -> [TextLine]` | `PaddleTextDetector` (PP-OCRv6 det), `KrakenSegmenter` (blla), `EynollahLines` (eynollah's PAGE-XML) |
| `LayoutAnalyzer` | `analyze(page) -> [Region]` | `PaddleLayout` (PP-DocLayoutV3 with learned reading order; PP-DocLayout_plus-L with XY-cut), `EynollahLayout` (eynollah's PAGE-XML), `SingleRegionLayout` |
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

## 7. eynollah

[eynollah](https://github.com/qurator-spk/eynollah) (Staatsbibliothek zu Berlin, Apache-2.0) is a
layout analyser trained on historical print: text regions with headings, drop capitals and
marginalia, images, separators, tables, text line polygons and a machine-based reading order. In
SquiddleOCR it is a layout stage and a line stage: eynollah writes PAGE-XML, SquiddleOCR reads the
regions (`paragraph`, `marginalia`, `drop-capital` → `text`, `heading` → `section_header`,
`ImageRegion` → `picture`, `TableRegion` → `table`; separators dropped), keeps its reading order
(regions it does not order, drop capitals, images and tables, are slotted in by position), then
runs PP-OCRv6 text detection on each text region's crop with everything outside the polygon painted
white, groups the boxes into visual rows and lets kraken read them, as in the paddle pipeline.
There is no table or formula stage (eynollah's table regions carry no lines).

The line stage is a choice, `--eynollah-lines`:

| value | lines | cost | notes |
|---|---|---|---|
| `paddle` (default) | PP-OCRv6 detection boxes per region | ~40 ms per region | whole lines; eynollah's text line mask breaks a justified line at wide word gaps, the detector does not. `--det-model` and `--unclip-ratio` apply |
| `eynollah` | eynollah's line polygons | none | fragments at wide gaps; no baselines in the XML, so one is fitted per line (a least-squares line through the polygon's lower vertices, raised by 18 % of the line height; `--no-baselines` reads boxes) and it tends to sit high in eynollah's generously padded polygons |
| `blla` | kraken's blla on each region's masked crop | ~4 s per region | real baselines and polygons, but blla's CPU post-processing (ridge filter and polygonisation) costs seconds per call whatever the crop size, and at native newspaper scale it fragments lines too |

kraken's own documentation warns that blla on a whole complex page (a newspaper) merges lines
across columns; eynollah's regions in front of any line detector is the answer to that.

**Install and models.**

```
pip install "squiddleocr[eynollah]"        # eynollah 0.9.x with ocrd; no TensorFlow, no TensorRT
squiddle models pull eynollah              # 1.8 GB ONNX bundle from Zenodo, once
squiddle ocr scans/ --layout eynollah -f md,page
```

The inference models are ONNX files distributed on Zenodo only (record 21381102,
`models_inference_layout_v0_9_1.zip`; the Hugging Face `SBB/eynollah-*` repos hold the old Keras
models and are not used). `squiddle models pull eynollah` downloads the bundle with resume, checks
its size and MD5 against the Zenodo record, unpacks it into SquiddleOCR's data directory
(`~/.local/share/squiddleocr/eynollah/<version>/`, next to kraken's `~/.local/share/htrmopo`;
`$SQUIDDLE_HOME` moves it), verifies the model files and writes `squiddle.json` (source URL,
version, hash, date, file list). The first `--layout eynollah` run does the same. A bundle you
already have is used through `SQUIDDLE_EYNOLLAH_MODELS=/path/to/models_eynollah` (or its parent).
The bundle is pinned in one place, `squiddleocr.models.EYNOLLAH_BUNDLE`.

eynollah pins `tensorrt_cu12` (CUDA 12 TensorRT, an sdist without an aarch64 build) and
`onnxruntime-gpu[cuda,cudnn]`. The extra keeps the second (it resolves to the CUDA 13 `nvidia-*`
packages SquiddleOCR already uses, on x86_64 and aarch64) and drops the first with a uv dependency
override (`[tool.uv] override-dependencies` in `pyproject.toml`), so TensorRT is opt-in. With plain
pip, install eynollah with `--no-deps` and its other requirements (`ocrd>=3.3`, `scikit-learn`,
`scikit-image`, `tabulate`) yourself if the TensorRT pin fails. eynollah declares Python ≤ 3.11 in
its classifiers; it runs on 3.12.

**How it runs.** eynollah forks one worker process per page job and spawns one process per model,
so it must not run inside the process that holds kraken's CUDA context: it runs as a subprocess,
`python -m squiddleocr.eynollah.launch`, a launcher of ours that applies the run plan (below) and
then calls eynollah's own CLI (`eynollah layout -di ... -o ... -j N`). eynollah is never vendored or
forked; the one runtime patch, the per-model VRAM caps, lives in that launcher module. The
subprocess runs at `nice 10` with idle-class I/O priority in its own process group, its log goes
to `eynollah.log` in the XML directory, warnings and errors are forwarded (the harmless ONNX Runtime
`GPU device discovery failed ... /sys/class/drm/card0` line is dropped), and it is ended as soon as
it reports all jobs done (its own teardown takes 15 to 25 s). Every page is run once: pages whose
XML exists in the XML directory are skipped, so `--eynollah-xml DIR` both keeps the raw PAGE-XML
and lets a later run (a different recogniser, other formats) reuse it without launching eynollah.

**The resource plan.** Before the launch SquiddleOCR reads the machine (logical and physical
cores with `sched_getaffinity` and cgroup limits, total and available RAM, the GPU's name and
free memory through torch or `nvidia-smi`, the ONNX Runtime providers, whether `libnvinfer` loads)
and prints one `resources` line with what it decided:

- **Provider.** `EYNOLLAH_ONNX_EP=CUDA,CPU` by default. eynollah itself prefers TensorRT > CUDA >
  CPU from what ONNX Runtime lists, and an ONNX Runtime GPU build lists TensorRT even when
  `libnvinfer` is missing, in which case it silently runs on the CPU; restricting the list avoids
  that. `--eynollah-tensorrt` adds TensorRT only when `libnvinfer` is loadable (engines are built
  on first use, minutes per model, cached under `XDG_CONFIG_HOME`). Without a CUDA provider or a
  GPU eynollah runs on the CPU with a warning. The provider eynollah used for each model is in
  the `eynollah` status line after the run and in `eynollah.log`.
- **VRAM caps.** eynollah caps each model's ONNX Runtime arena with a hard-coded table
  (`MODEL_VRAM_LIMITS`, 200 MB to 1.9 GB, calibrated for small GPUs). On Blackwell and on unified
  memory cuDNN chooses larger convolution workspaces and a run dies with
  `BFCArena ... Available memory of 0 is smaller than requested bytes`. There is no override in
  eynollah, so the launcher replaces the table in memory: free GPU memory minus a margin
  (`--eynollah-vram-margin`, default 20 % of the GPU or 4 GB, whichever is larger) minus 2 GB for
  kraken's torch model, split across the resident models (col_classifier, page, textline,
  region_1_2, plus region_fl_np for `-fl`, reading_order for `-romb`, table for `-tab`,
  binarization for `-ib`) in the ratio of eynollah's defaults, each at least twice its default
  and at most 8 GB. The caps are limits, not reservations.
- **Jobs and threads.** eynollah's `-j 0` means one job per CPU core, which overloads a machine:
  each page job is single-threaded OpenCV/NumPy work at full resolution holding several copies of
  the page. The number of jobs is
  `min(physical cores − reserved, RAM budget / RAM per job, 8)` with `reserved = max(2, 10 %)`,
  RAM per job from the largest input (pixels × 3 bytes × 8 copies, at least 1.5 GB) and a RAM
  budget of available RAM minus a margin of 25 % or 8 GB, whichever is larger; never below 1.
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `cv2.setNumThreads` are set so
  that jobs × threads stays within the unreserved cores. `--eynollah-jobs N` overrides.
- **Unified memory** (DGX Spark, Jetson: GPU memory is system RAM) is detected from the GPU's
  size and name. Then GPU and RAM are one budget: the sum of the VRAM caps comes out of the RAM
  budget instead of being counted twice, and the free figure is the RAM's (CUDA's own "free"
  leaves out the page cache).
- **Watchdog.** While eynollah runs, available RAM is polled every second. Below the margin a
  warning is printed; if it keeps falling (or drops below half the margin) eynollah is stopped and
  restarted with one job fewer on the pages that are still missing (a half-written XML is removed
  first). With one job left it only warns.

On the DGX Spark (20 cores, 128 GB unified, GB10) the plan is 8 jobs × 2 threads with caps of 2.8
to 8 GB per model; 17 test scans (1100×2100 to 2500×3800 px) take 22 s inside eynollah (44 s
with model loading and teardown before the early stop, about 25 s after), then 1.1 s per page for
kraken; available RAM never dropped by more than 17 GB. With `--eynollah-jobs 1` eynollah takes
43 s for the same pages (2.5 s per page). On aarch64 the ONNX Runtime GPU wheel has no TensorRT
provider at all, so the provider list is `CUDA,CPU` regardless.

**eynollah flags worth passing** (`--eynollah-args "..."`, default `-fl -romb`):

| flag | effect |
|---|---|
| `-fl` | full layout: headings and drop capitals as their own regions (`region_fl_np` model) |
| `-romb` | machine-based reading order (`reading_order` model) instead of the heuristic |
| `-tab` | detect table regions (`table` model); they become Docling tables without cell structure |
| `-cl` | curved line polygons: deskews and detects lines per region, much slower |
| `-as` | check the scale and rescale for better region detection |
| `-ae` | check whether the image needs enhancement and enhance it (`enhancement` model) |
| `-ib` | binarise the input first (`binarization` model), for very dark or bright scans |
| `-ncu N`, `-ncl N` | upper/lower bound on the number of columns |
| `-ipe` | skip page-frame cropping |
| `-slro` | no layout or reading order: one region with all lines |

`-r2l` is added by `--rtl`. Plotting and OCR flags of eynollah are not useful here.

## 8. Known limitations

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
- **eynollah's regions are plain text blocks.** eynollah has no caption, footnote or page-number
  class; page numbers and catchwords come out as small `text` regions in its reading order, and
  a table region has no lines (`-tab` gives an empty table item). With `-fl` newspaper sub-heads
  and kickers are all `heading`, so a third of a newspaper page's regions can be `##` in Markdown.

## 9. Licence, credit and citation

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
