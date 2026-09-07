# SquiddleOCR

SquiddleOCR lets you read historical print and handwriting with PaddleOCR and
PP-StructureV3. It converts [kraken](https://kraken.re)'s PP-OCRv6 text
recognition models (Benjamin Kiessling, ALMAnaCH / Inria, Apache-2.0) into
model directories that PaddleOCR 3.x accepts as a text recognition model, so
you get PaddleOCR's layout analysis, tables as HTML and Markdown/JSON output
with a recogniser trained for old documents.

Nothing is retrained. The converter loads the kraken model, folds kraken's
line preprocessing into the network graph, exports ONNX and writes the PaddleX
config and dictionary. The model card, licence and Zenodo DOI of the source
model travel with every converted directory.

## 1. Quick start: OCR an image or a folder from the command line

### Install

```
git clone https://github.com/storytracer/squiddleocr.git && cd squiddleocr
uv sync --extra paddle
source .venv/bin/activate
```

This installs the converter (torch, kraken, onnx, onnxruntime) and the
PaddleOCR side (paddlepaddle CPU, paddleocr, paddlex). Python 3.11 or 3.12.

### Get and convert a model

```
kraken get 10.5281/zenodo.21788410      # medium model, ~64 MB, lands in ~/.local/share/htrmopo/<id>/
squiddle convert ~/.local/share/htrmopo/<id>/medium.safetensors -o squiddle_PP-OCRv6_medium_rec
```

`squiddle convert` takes a few seconds and prints the ONNX/PyTorch parity
check. The result is the directory `squiddle_PP-OCRv6_medium_rec/`.

### OCR one image

```
paddleocr ocr -i page.jpg \
    --text_recognition_model_dir squiddle_PP-OCRv6_medium_rec \
    --text_recognition_model_name PP-OCRv6_medium_rec \
    --engine onnxruntime \
    --text_det_unclip_ratio 2.0 \
    --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False \
    --save_path out/
```

`out/` receives `page_res.json` (boxes, texts, scores) and
`page_ocr_res_img.jpg` (visualisation); the result is also printed. The three `--use_* False` flags
switch off document preprocessing models that are meant for photographed
modern documents; leave them on if your scans are rotated or warped.

### OCR a folder of images

`-i` accepts a directory; every image in it is processed and the results land
in `--save_path` named after each file:

```
paddleocr ocr -i scans/ \
    --text_recognition_model_dir squiddle_PP-OCRv6_medium_rec \
    --text_recognition_model_name PP-OCRv6_medium_rec \
    --engine onnxruntime --text_det_unclip_ratio 2.0 \
    --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False \
    --save_path out/
```

The first run downloads the other models of the pipeline (text detector, in
ONNX form because of `--engine onnxruntime`) into `~/.paddlex/official_models/`.

Three things to know:

- Both `--text_recognition_model_dir` and `--text_recognition_model_name`
  are needed. The name has to be a name PaddleX knows; `PP-OCRv6_medium_rec`
  (or `_small_`, `_tiny_`) is written into the directory's `inference.yml`
  by the converter.
- `--engine onnxruntime` is needed because the converted model is ONNX.
- `--text_det_unclip_ratio 2.0` makes the detector's boxes a little larger
  than the default 1.5. Tight boxes cut ascenders, descenders and line-final
  hyphens on historical print; see "Known risks".

Output text is kraken's diplomatic transcription: NFD Unicode, long s,
combining diacritics, `⸗` hyphens. Normalise afterwards if you need NFC.

## 2. Layout, tables and Markdown with PP-StructureV3

PP-StructureV3 adds layout detection, reading order, tables as HTML and
Markdown/JSON export. Generate a pipeline config that plugs the recogniser in,
then run the pipeline with it:

```
squiddle pipeline-config squiddle_PP-OCRv6_medium_rec -o PP-StructureV3_squiddle.yaml

paddleocr pp_structurev3 -i scans/ \
    --paddlex_config PP-StructureV3_squiddle.yaml \
    --engine onnxruntime --text_det_unclip_ratio 2.0 \
    --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False \
    --save_path out/
```

`out/` receives, per page, `<name>.md` (page text in reading order, tables
as HTML), `<name>_res.json` (every block, box and table), `<name>.docx`,
`<name>.tex` and visualisations of layout, reading order and OCR.
Formula and chart recognition can be switched off with
`--use_formula_recognition False --use_chart_recognition False` to save time.

The generated YAML is PaddleX's own `PP-StructureV3.yaml` with the general
text recognisers replaced (`model_name`, `model_dir`, `engine: onnxruntime`);
the seal recogniser keeps its stock model. `--pipeline OCR` produces the same
for the plain OCR pipeline. Other pipeline settings (thresholds, batch sizes,
which sub-models to use) can be edited in that file.

With `--paddlex_config`, `--engine onnxruntime` can be dropped if the other
models should run on Paddle Inference; the recogniser keeps its own engine
setting from the YAML.

## 3. Python API

Plain OCR:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_recognition_model_dir="squiddle_PP-OCRv6_medium_rec",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    engine="onnxruntime",
    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
)
for res in ocr.predict("page.jpg", text_det_unclip_ratio=2.0):
    for text, score in zip(res["rec_texts"], res["rec_scores"]):
        print(text, score)
    res.save_to_json("out/")
```

PP-StructureV3:

```python
from paddleocr import PPStructureV3

pipe = PPStructureV3(paddlex_config="PP-StructureV3_squiddle.yaml",
                     use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
for res in pipe.predict("scans/", text_det_unclip_ratio=2.0):
    res.save_to_markdown("out/")
    res.save_to_json("out/")
```

Text recognition of line images only (no detection):

```python
from paddlex.inference import create_predictor

rec = create_predictor("PP-OCRv6_medium_rec", model_dir="squiddle_PP-OCRv6_medium_rec",
                       engine="onnxruntime", batch_size=1)
for r in rec.predict("line.png"):
    print(r["rec_text"], r["rec_score"])
```

### GPU

Install `onnxruntime-gpu` in place of `onnxruntime` (same import name) and add
`--device gpu` (Python: `device="gpu"`). With `--engine onnxruntime` every
model of the pipeline then runs on ONNX Runtime's CUDA provider. ONNX Runtime
needs the CUDA 13 runtime, cuBLAS and cuDNN 9 libraries; the pip packages torch
installs into the venv provide them, so put them on the library path:

```
uv pip install onnxruntime-gpu
SP=.venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$SP/cu13/lib:$SP/cudnn/lib
```

On a DGX Spark (GB10, aarch64) this took PP-StructureV3 from 10-16 s per page
on CPU to under a second per page with identical output; the recogniser alone
runs 13x faster. Numbers in `NOTES.md`.

## 4. Tuning

- **Exact kraken results.** Batched recognition (`batch_size: 8` in the pipeline
  YAML, the PaddleOCR default) changes a small share of lines, mostly long s
  versus round s, because the model sees padding beyond the line end. kraken's
  own batch mode behaves identically. For kraken's default single-line results
  set the recogniser's `batch_size: 1` in the YAML, at a speed cost.
- **Box expansion.** `text_det_unclip_ratio` 2.0 to 2.5 recovered most
  line-final `⸗` hyphens and lowered page CER on the Fraktur test pages from
  about 0.7 % to 0.2-0.4 %; larger values eventually merge neighbouring lines.
- **Very long lines.** PaddleX caps the recogniser input at 3200 px width
  (33:1 aspect at 96 px) and squashes wider lines. kraken has no cap.
- **Other sizes.** The tiny (`10.5281/zenodo.21788403`) and small
  (`10.5281/zenodo.21788405`) models convert the same way; the converter reads
  the variant from the file.

## 5. Model names

| kraken model (Zenodo DOI) | SquiddleOCR directory | `model_name` in `inference.yml` |
|---|---|---|
| PP-OCRv6 tiny, `10.5281/zenodo.21788403` | `squiddle_PP-OCRv6_tiny_rec` | `PP-OCRv6_tiny_rec` |
| PP-OCRv6 small, `10.5281/zenodo.21788405` | `squiddle_PP-OCRv6_small_rec` | `PP-OCRv6_small_rec` |
| PP-OCRv6 medium, `10.5281/zenodo.21788410` | `squiddle_PP-OCRv6_medium_rec` | `PP-OCRv6_medium_rec` |

The `squiddle_` prefix follows PaddleOCR's `<prefix>_PP-OCRv<N>_<size>_rec`
scheme and identifies this packager. It is not the model author's naming and
does not imply an official PaddleOCR variant.

The `model_name` inside `inference.yml` must be a name registered in PaddleX,
which rejects unknown names and refuses a directory whose `Global.model_name`
differs from the requested name. PaddleX 3.7 registers
`PP-OCRv6_{tiny,small,medium}_rec` (Baidu's own PP-OCRv6 recognisers, same
architecture family), so those are used. `squiddle convert --model-name
PP-OCRv5_server_rec` registers under PP-StructureV3's default name instead,
which lets a pipeline pick the directory up without a name argument.

## 6. What is in a converted directory

| file | purpose |
|---|---|
| `inference.onnx` | the recogniser, weights embedded (64 MB for medium), opset 18, dynamic batch and width |
| `inference.yml` | PaddleX model config: `Global.model_name`, preprocessing (RGB, height 96), `CTCLabelDecode` with the dictionary |
| `dict.txt` | the same dictionary, one entry per line, for humans and other tools |
| `MODEL_CARD.md` | the kraken model card, unchanged, with a note on the conversion |
| `NOTICE`, `LICENSE` | attribution, DOI and the Apache-2.0 licence of the weights |
| `squiddle.json` | provenance: source file hash, kraken/torch versions, export parity numbers |

`squiddle convert --help` lists the options (`--model-name`, `--padding`,
`--external-weights` to put the weights in `inference.onnx.data`).

## 7. How the conversion works

PaddleOCR's own recognisers take 48 px lines normalised to [-1, 1]. kraken's
PP-OCRv6 models take 96 px lines, scaled to 0..1 and inverted (ink bright,
paper dark), with 16 px of white padding at both ends, and are batched by white
padding plus an attention mask on the padded time steps. PaddleX's recognition
preprocessing hardcodes the [-1, 1] normalisation, resizes with OpenCV
bilinear, pads batches with mid-grey and only lets `inference.yml` choose the
height and colour order. SquiddleOCR therefore:

- sets `RecResizeImg.image_shape: [3, 96, 96]` and `DecodeImage.img_mode: RGB`;
- folds the rest into the ONNX graph: `x_kraken = 0.5 - 0.5 * x_paddle`
  (identical to `1 - x/255`), the white padding, and detection of PaddleOCR's
  batch padding (trailing all-zero columns, a value real pixels cannot take),
  which is turned into white, masked exactly as kraken masks it and forced to
  the CTC blank in the output;
- emits `(batch, time, classes)` softmax probabilities with the blank at index
  0, as PaddleX's `CTCLabelDecode` expects.

The dictionary is the kraken codec in label order (1622 single NFD codepoints
for medium; space is label 1). PaddleX prepends `blank` itself and, because
`use_space_char` is always on for CTC, appends one extra space at an index the
network never emits.

## 8. Verifying a conversion

```
squiddle extract-lines page1.jpg page2.jpg -o lines/     # kraken segmentation -> line PNGs
squiddle verify squiddle_PP-OCRv6_medium_rec lines/ --report verify.json
```

`verify` runs every line through kraken (the reference), through ONNX Runtime
on kraken's own tensor, through ONNX Runtime with an exact re-implementation of
PaddleX's preprocessing (batch 1 and batched), through kraken's native batch
mode and, when `paddlex` is importable, through the real PaddleX predictor. It
reports exact-match rate and character error rate against kraken and prints
the differing lines. On 68 kraken-extracted lines of 19th-century German
Fraktur the export reproduces kraken line for line at batch size 1, and batched
runs reproduce kraken's own batched output line for line (details in
`NOTES.md`). Unit tests: `pytest`.

## 9. Results on a Fraktur book

Every 10th page of a 230-page 19th-century German Fraktur novel (23 pages),
PP-StructureV3 on a DGX Spark GPU with the medium model, box expansion 2.0,
compared with kraken's transcription of the same pages
(`scripts/eval_fraktur_pages.py`):

| | CER |
|---|---|
| 21 regular text pages | 0.41 % |
| same, ignoring `⸗`/`-` and quote-glyph differences | 0.06 % |
| all 23 pages (one advertisement page with reordered layout, one page-number-only page) | 1.30 % |

0.83 s per page. Long s and combining diacritics were read correctly on every
page; the residual errors are line-final `⸗` hyphens cut by the detector's
boxes and quote glyphs. Details and per-page numbers in `NOTES.md`.

## 10. Known risks

- **Preprocessing mismatch degrades accuracy silently.** Anything that feeds
  the model differently from kraken (another height, un-inverted input, grey
  padding) still produces plausible text. Run `squiddle verify` on real line
  strips after any change to PaddleX or to the config.
- **Tight detection boxes.** PP-Structure's DBNet detector produces tight
  rectangular crops. Earlier tests showed this model switching from diplomatic
  to normalised transcription on tight crops and reading correctly when the
  crop had a normal ascender/descender allowance. On two Fraktur test pages the
  default crops kept every long s and combining mark but lost most line-final
  `⸗` hyphens (read as `-`); `text_det_unclip_ratio` 2.0 to 2.5 recovered most
  of them.
- **Layout and detection models are modern-document models.** PP-DocLayout and
  DBNet are trained on modern documents. Their behaviour on historical pages,
  and thus the reading order, tables and crops SquiddleOCR's recogniser
  receives, is unmeasured. SquiddleOCR cannot fix that.
- **Batching changes a few lines** (see "Tuning").
- **No native Paddle inference format.** X2Paddle 1.6 cannot convert the
  graph (opset 18, dynamic width; see `NOTES.md`). PaddleX's ONNX Runtime
  engine is the supported path; Paddle's high-performance inference (`use_hpip`)
  on x86 has not been validated.

## Licence and credit

Model weights: Apache-2.0, Benjamin Kiessling (ALMAnaCH, Inria Paris). Cite
the Zenodo DOI of the model you use (in `NOTICE` and `squiddle.json` of every
converted directory). The `squiddle_` naming is this project's and not the
author's. Converter code: Apache-2.0. The bundled pipeline templates are
PaddleX's (Apache-2.0).
