# SquiddleOCR

SquiddleOCR reads historical print and handwriting with kraken's PP-OCRv6 recogniser inside a
modern document-parsing pipeline. Any layout-analysis, structure or line-segmentation model can be
plugged in front of the recogniser; the result is a `DoclingDocument` that exports to DocLang,
Markdown, HTML and JSON.

The recogniser is [kraken](https://kraken.re)'s PP-OCRv6 model family (Benjamin Kiessling,
ALMAnaCH / Inria, Apache-2.0), converted to ONNX with its line preprocessing folded into the graph
and run with ONNX Runtime on CUDA, TensorRT, CoreML or CPU. Nothing is retrained. The converted
model directory is also a drop-in text recognition model for PaddleOCR / PP-StructureV3.

## 1. Quick start

### Install

```
git clone https://github.com/storytracer/squiddleocr.git && cd squiddleocr
uv sync --extra paddle                  # runtime + PaddleX models (layout, detector, tables)
source .venv/bin/activate
```

or with pip: `pip install "squiddleocr[paddle] @ git+https://github.com/storytracer/squiddleocr"`.

The core install brings ONNX Runtime (`onnxruntime-gpu` on Linux and Windows, `onnxruntime` with
CoreML on macOS), `docling-core`, NumPy, Pillow and OpenCV. Extras:

| extra | adds | needed for |
|---|---|---|
| `paddle` | paddlex, paddleocr, paddlepaddle (CPU) | PP-DocLayout layout, PP-OCRv6 text detection, SLANet tables, the PaddleOCR drop-in |
| `kraken` | kraken, torch | kraken's blla segmenter as the line detector |
| `convert` | torch, kraken, onnx, onnxscript | converting kraken safetensors models to ONNX |

Python 3.11 or 3.12.

### OCR an image or a folder

```
squiddle ocr page.jpg
squiddle ocr scans/ -o out/ -f doclang,md,html,json
```

That is all. The first run fetches the medium recogniser (about 64 MB) into
`~/.cache/squiddleocr/models/` and the PaddleX layout, detection and table models into
`~/.paddlex/official_models/`. Per image you get `<name>.doclang.xml` and `<name>.md` (tables as
HTML inside the Markdown) in `out/`, plus whatever else `-f` lists (`html`, `json` = lossless
DoclingDocument, `txt`). `--per-document` writes one document for a whole folder instead.

Options that matter:

- `-m tiny|small|medium` picks the recogniser size (medium is the default and the most accurate;
  tiny is 3 MB). `-m <directory>` uses a model directory you converted yourself.
- `--layout none` skips layout analysis (the whole page is one text region: plain OCR).
- `--detector kraken` uses kraken's blla baseline segmenter instead of the PP-OCRv6 detector
  (`kraken` extra).
- `--unclip-ratio 2.0` (default) expands the detector's boxes; historical print wants a bit more
  than PaddleOCR's 1.5, see "Known risks".
- `--batch-size 1` reproduces kraken's single-line results exactly; larger batches are faster and
  reproduce kraken's own batched behaviour.
- `--device auto|cpu|cuda|tensorrt|coreml`.

Output text is kraken's diplomatic transcription: NFD Unicode, long s, combining diacritics, `⸗`
hyphens. Normalise afterwards if you need NFC.

### Models

The recognisers come from a *model source*: the Hugging Face repo
[`storytracer/squiddleocr`](https://huggingface.co/storytracer/squiddleocr) by default, or any
folder or repo with the same layout (`README.md` plus `models/squiddle_PP-OCRv6_<size>_rec/`).
Sizes are downloaded into `~/.cache/squiddleocr/` on first use (`SQUIDDLE_HOME` moves the cache).

```
squiddle models pull                        # fetch all sizes ahead of time
squiddle models list                        # what is available locally
squiddle ocr scans/ --models ./my-models    # or --models someone/their-repo; env SQUIDDLE_MODELS
```

To convert kraken's models yourself and, if you like, publish them to your own repo:

```
uv sync --extra convert                     # torch + kraken
squiddle convert -o squiddleocr-models      # all three sizes from kraken's Hub mirror; or: squiddle convert small
squiddle convert my-model.safetensors -o squiddleocr-models     # a kraken file of your own
squiddle ocr scans/ --models squiddleocr-models
squiddle upload squiddleocr-models --repo you/squiddleocr       # creates the repo, uploads folder + model card
```

Each model directory holds `inference.onnx`, `inference.yml`, `dict.txt`, the kraken model card,
NOTICE, LICENSE and `squiddle.json` with provenance and the ONNX/PyTorch parity check; the folder's
`README.md` is a Hub model card listing them.

## 2. Python API

```python
from squiddleocr.factory import build_pipeline
from squiddleocr.document import export

pipe = build_pipeline("medium", layout="paddle", detector="paddle", tables=True)   # or models="folder-or-repo"
doc = pipe.run_files(["page1.jpg", "page2.jpg"])         # a DoclingDocument
print(doc.export_to_markdown())
export(doc, "out/", "book", ["doclang", "json"])
```

Or assemble the pipeline yourself:

```python
from squiddleocr.pipeline import Pipeline
from squiddleocr.models import resolve_model
from squiddleocr.recognizers import OnnxRecognizer
from squiddleocr.detectors.paddle import PaddleTextDetector
from squiddleocr.layout.paddle import PaddleLayout
from squiddleocr.tables.paddle import PaddleTableRecognizer

pipe = Pipeline(
    recognizer=OnnxRecognizer(resolve_model("medium"), device="cuda", batch_size=8),
    detector=PaddleTextDetector("PP-OCRv6_medium_det", unclip_ratio=2.0),
    layout=PaddleLayout(),
    tables=PaddleTableRecognizer(),
)
for content in pipe.process_page(page):        # per-region access: content.region, .lines, .texts, .table
    ...
```

## 3. Architecture

```
Page ──► LayoutAnalyzer ──► regions (label, polygon, reading order)
           │
           ├─► TextDetector ──► text lines (page-level, assigned to regions; orphans become regions)
           │                        │
           │                        └─► Recognizer ──► text per line, batched across regions
           │
           └─► table regions ──► TableRecognizer (cell structure) ──► cells read by detector + recogniser
                                                                    │
                                                    DocumentBuilder ──► DoclingDocument ──► DocLang / Markdown / HTML / JSON
```

Four small protocols in `squiddleocr/{recognizers,detectors,layout,tables}/base.py`:

| protocol | method | implementations |
|---|---|---|
| `Recognizer` | `recognize(line_images) -> [Recognition]` | `OnnxRecognizer` (the converted kraken model) |
| `TextDetector` | `detect(page, region=None) -> [TextLine]` | `PaddleTextDetector` (PP-OCRv6 det), `KrakenSegmenter` (blla) |
| `LayoutAnalyzer` | `analyze(page) -> [Region]` with `order` | `PaddleLayout` (PP-DocLayout_plus-L + XY-cut order), `SingleRegionLayout` |
| `TableRecognizer` | `structure(page, region) -> TableResult` | `PaddleTableRecognizer` (SLANet_plus) |

To add a layout model, implement `analyze` and translate its labels to the Docling vocabulary
(`text`, `title`, `section_header`, `caption`, `footnote`, `page_header`, `page_footer`, `table`,
`picture`, `formula`, `list_item`, ...); `squiddleocr.layout.order` provides overlap suppression and
XY-cut ordering for models without a reading-order head. To add a segmenter, return `TextLine`
polygons in page coordinates. Nothing else changes.

`Pipeline` orchestrates: layout, one detection pass on the page, assignment of lines to regions
(lines outside every region become their own regions so page numbers are not lost), row-wise
ordering and de-duplication of line boxes, one batched recognition call, cell-by-cell reading of
tables, and the `DoclingDocument` build. `DocumentBuilder` maps labels, keeps boxes as provenance
and preserves reading order; exports are docling-core's own serialisers.

## 4. Results

**Fraktur** (every 10th page of a 230-page 19th-century German novel, 23 pages, GPU, compared
with kraken's transcription of the same pages; `scripts/eval_fraktur_squiddle.py`):

| pipeline | 21 regular pages | all 23 pages | s/page |
|---|---|---|---|
| SquiddleOCR, PP-DocLayout + PP-OCRv6 det | 0.45 % | 0.49 % | 0.67 |
| SquiddleOCR, no layout | 0.45 % | 0.49 % | 0.64 |
| PaddleOCR PP-StructureV3 drop-in, same models | 0.61 % | 1.48 % | 0.77 |

The remaining errors are the line-final `⸗` read as `-` and quote glyphs; long s and combining
diacritics are read correctly throughout. The gap on "all pages" is one advertisement page whose
irregular layout PP-StructureV3's reading order reshuffles and SquiddleOCR's XY-cut orders correctly.

**Tables** (the seven BHL IMPACT pages with a table region): the same five tables found as with
PP-StructureV3; a six-row eight-column morphology table with column and row spans is recovered
with correct structure and cell text, except that the recogniser reads "Hypopharynx" in Cyrillic
letters on the tight cell crops. Details in `NOTES.md`.

## 5. The recogniser and the conversion

kraken's PP-OCRv6 models take 96 px lines, scaled to 0..1 and inverted (ink bright, paper dark),
with 16 px of white padding at both ends, and are batched by white padding plus an attention mask
on the padded time steps. `squiddle convert` folds all of that into the ONNX graph: the graph takes
PaddleOCR's `[-1, 1]` input (`x = 0.5 - 0.5 * x_paddle` equals `1 - x/255`), pads, detects
batch padding (trailing all-zero columns, a value real pixels cannot take), masks it as kraken
does, forces those time steps to blank and emits `(batch, time, classes)` softmax probabilities.
`OnnxRecognizer` only resizes and normalises; `squiddle verify` proves the export reproduces kraken
line for line on real line strips (`squiddle extract-lines` cuts them with kraken's segmenter).

| kraken model (Zenodo DOI) | directory | `model_name` in `inference.yml` |
|---|---|---|
| PP-OCRv6 tiny, `10.5281/zenodo.21788403` | `squiddle_PP-OCRv6_tiny_rec` | `PP-OCRv6_tiny_rec` |
| PP-OCRv6 small, `10.5281/zenodo.21788405` | `squiddle_PP-OCRv6_small_rec` | `PP-OCRv6_small_rec` |
| PP-OCRv6 medium, `10.5281/zenodo.21788410` | `squiddle_PP-OCRv6_medium_rec` | `PP-OCRv6_medium_rec` |

The `squiddle_` prefix follows PaddleOCR's naming scheme and identifies this packager; it is not
the model author's naming. The `model_name` inside `inference.yml` must be a name PaddleX
registers, hence the PP-OCRv6 names.

## 6. Using the recogniser inside PaddleOCR / PP-StructureV3

The converted directory is a PaddleOCR text recognition model. `squiddle pipeline-config` writes
PaddleX's PP-StructureV3 (or OCR) YAML with the recogniser and the PP-OCRv6 detector set:

```
squiddle pipeline-config squiddle_PP-OCRv6_medium_rec -o PP-StructureV3_squiddle.yaml
paddleocr pp_structurev3 -i scans/ --paddlex_config PP-StructureV3_squiddle.yaml --engine onnxruntime \
    --text_det_unclip_ratio 2.0 --save_path out/
```

or with keyword arguments: `PPStructureV3(text_recognition_model_dir=..., text_recognition_model_name="PP-OCRv6_medium_rec", engine="onnxruntime")`.
The ONNX model needs PaddleX's `onnxruntime` engine. For bulk runs use the Python API and skip
`save_to_img`: writing the visualisation images costs about 18 s per page.

## 7. GPU

`onnxruntime-gpu` is installed on Linux and Windows and provides the CUDA and TensorRT
providers; on macOS `onnxruntime` provides CoreML. `--device auto` picks the best available
provider. ONNX Runtime needs the CUDA 13 runtime, cuBLAS and cuDNN 9 libraries; the `nvidia-*`
pip packages that `onnxruntime-gpu` or torch pull in provide them and are found automatically
(`onnxruntime.preload_dlls`). On a DGX Spark (GB10, aarch64) a text page takes about 0.7 s,
a table page 2 to 3 s; the recogniser alone runs 13x faster than on the 20 CPU cores.

## 8. Known risks

- **Preprocessing mismatch degrades accuracy silently.** Anything that feeds the recogniser
  differently from kraken still produces plausible text. `squiddle verify` compares against
  kraken on real line strips; run it after changing anything in the recognition path.
- **Tight crops.** Detector boxes and table cells cut ascenders, descenders and line-final `⸗`;
  the recogniser then reads `-`, or, on very short cell crops, flips a Latin word to Cyrillic.
  `--unclip-ratio` and the pipeline's cell padding mitigate this; both are measured in `NOTES.md`.
- **Modern-document layout models.** PP-DocLayout and the PP-OCRv6 detector are trained on
  modern documents; their behaviour on historical pages is only measured on the material above.
  Text they miss is recovered by the page-level detection pass, but region labels, reading order
  and table detection on unusual pages are not guaranteed.
- **Batching changes a few lines** (long s versus s at line ends); `--batch-size 1` for kraken's
  exact single-line output.

## Licence and credit

Model weights: Apache-2.0, Benjamin Kiessling (ALMAnaCH, Inria Paris). Cite the Zenodo DOI of
the model you use (in `NOTICE` and `squiddle.json` of every converted directory). Code:
Apache-2.0. The bundled PaddleX pipeline templates are PaddleX's (Apache-2.0).
