# SquiddleOCR

SquiddleOCR packages [kraken](https://kraken.re)'s PP-OCRv6 text recognition
models (Benjamin Kiessling, ALMAnaCH / Inria, Apache-2.0) as model directories
that PaddleOCR 3.x and PP-StructureV3 accept as a text recognition model.
Point `text_recognition_model_dir` at a SquiddleOCR directory and PP-Structure's
layout analysis, table-to-HTML and Markdown/JSON export run with a recogniser
trained for historical print and handwriting.

Nothing is retrained. The converter loads the kraken safetensors file, folds
kraken's line preprocessing into the network graph, exports ONNX and writes the
PaddleX `inference.yml` and dictionary. The model card, licence and Zenodo DOI
of the source model are reproduced in every output directory.

## Model names

| kraken model (Zenodo DOI) | SquiddleOCR directory | `model_name` in `inference.yml` |
|---|---|---|
| PP-OCRv6 tiny, `10.5281/zenodo.21788403` | `squiddle_PP-OCRv6_tiny_rec` | `PP-OCRv6_tiny_rec` |
| PP-OCRv6 small, `10.5281/zenodo.21788405` | `squiddle_PP-OCRv6_small_rec` | `PP-OCRv6_small_rec` |
| PP-OCRv6 medium, `10.5281/zenodo.21788410` | `squiddle_PP-OCRv6_medium_rec` | `PP-OCRv6_medium_rec` |

The `squiddle_` prefix follows PaddleOCR's `<prefix>_PP-OCRv<N>_<size>_rec`
scheme and identifies this packager. It is not the model author's naming and
does not imply an official PaddleOCR variant.

The `model_name` inside `inference.yml` has to be a name registered in PaddleX,
which rejects unknown names ("No engine bindings registered") and refuses a
directory whose `Global.model_name` differs from the requested model name.
PaddleX 3.7 registers `PP-OCRv6_{tiny,small,medium}_rec` (Baidu's own PP-OCRv6
recognisers, which share the architecture family), so those are used. This
means you must pass **both** `text_recognition_model_dir` and
`text_recognition_model_name` (or use the generated pipeline YAML).

## Installation

```
git clone <this repo> squiddleocr && cd squiddleocr
uv sync                       # converter: torch, kraken, onnx, onnxruntime
uv sync --extra paddle        # + paddlepaddle, paddleocr, paddlex (verification, CPU)
```

`kraken>=7.1` with the `kraken.lib.ppocr` package is required (the version in
`/home/seb/dev/kraken` is used as a path dependency in `pyproject.toml`; change
`[tool.uv.sources]` to a released kraken when one ships the PP-OCRv6 code).

## Converting

```
kraken get 10.5281/zenodo.21788410          # downloads medium.safetensors + model card
squiddle convert ~/.local/share/htrmopo/<id>/medium.safetensors -o squiddle_PP-OCRv6_medium_rec
```

Output directory:

| file | purpose |
|---|---|
| `inference.onnx` | the recogniser, weights embedded (64 MB for medium), opset 18, dynamic batch and width |
| `inference.yml` | PaddleX model config: `Global.model_name`, preprocessing (RGB, height 96), `CTCLabelDecode` with the dictionary |
| `dict.txt` | the same dictionary, one entry per line, for humans and other tools |
| `MODEL_CARD.md` | the kraken model card, unchanged, with a note on the conversion |
| `NOTICE`, `LICENSE` | attribution, DOI and the Apache-2.0 licence of the weights |
| `squiddle.json` | provenance: source file hash, kraken/torch versions, export parity numbers |

`squiddle convert --help` lists the options (`--model-name` to register under
another PaddleX name such as `PP-OCRv5_server_rec`, `--padding`, `--external-weights`).

## Using with PP-StructureV3 / PaddleOCR

The exported model is ONNX, so the text recognition sub-module has to run on
PaddleX's `onnxruntime` engine (the default `paddle` engine refuses a directory
without Paddle files). Two ways:

**1. Generated pipeline YAML (recommended).** It keeps every other model on
its default engine and only switches the recogniser:

```
squiddle pipeline-config squiddle_PP-OCRv6_medium_rec -o PP-StructureV3_squiddle.yaml
```

```python
from paddleocr import PPStructureV3

pipe = PPStructureV3(paddlex_config="PP-StructureV3_squiddle.yaml")
for res in pipe.predict("page.jpg", text_det_unclip_ratio=2.0):
    res.save_to_markdown("out/")
    res.save_to_json("out/")
```

Use `--pipeline OCR` for the plain OCR pipeline. The generated file is
PaddleX's own `PP-StructureV3.yaml` / `OCR.yaml` with `model_name`, `model_dir`
and `engine: onnxruntime` set on the general text recognisers (the seal
recogniser is left alone).

**2. Keyword arguments.** Set the engine for the whole pipeline; PaddleX then
downloads the ONNX variants of the other official models automatically:

```python
pipe = PPStructureV3(
    text_recognition_model_dir="squiddle_PP-OCRv6_medium_rec",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    engine="onnxruntime",
)
```

The CLI equivalent is `paddleocr pp_structurev3 --text_recognition_model_dir ...
--text_recognition_model_name PP-OCRv6_medium_rec --engine onnxruntime`.

For plain text recognition of line images:

```python
from paddlex.inference import create_predictor

rec = create_predictor("PP-OCRv6_medium_rec", model_dir="squiddle_PP-OCRv6_medium_rec",
                       engine="onnxruntime")
for r in rec.predict("line.png"):
    print(r["rec_text"], r["rec_score"])
```

GPU: `engine="onnxruntime", device="gpu"` uses ONNX Runtime's CUDA provider if
`onnxruntime-gpu` is installed. On x86 with Paddle's high-performance inference
(`use_hpip=True`) the HPI runner can also pick the ONNX file up; see NOTES.md
for what has and has not been validated.

## What differs from stock PaddleOCR recognisers

PaddleOCR's own recognisers take 48 px lines normalised to [-1, 1]. kraken's
PP-OCRv6 models take **96 px** lines, scaled to 0..1 and **inverted** (ink
bright, paper dark), with 16 px of white padding on both ends, and are batched
by white padding plus an attention mask on the padded time steps. PaddleX's
recognition preprocessing (`OCRReisizeNormImg`) hardcodes the `[-1, 1]`
normalisation, resizes with OpenCV bilinear, pads batches with mid-grey and only
lets `inference.yml` choose the height and colour order. SquiddleOCR therefore:

- sets `RecResizeImg.image_shape: [3, 96, 96]` and `DecodeImage.img_mode: RGB`;
- folds the rest into the ONNX graph: `x_kraken = 0.5 - 0.5 * x_paddle`
  (identical to `1 - x/255`), white padding, and detection of PaddleOCR's
  batch padding (trailing all-zero columns, a value real pixels cannot take),
  which is turned into white, masked exactly as kraken masks it and forced to
  the CTC blank in the output;
- emits `(batch, time, classes)` softmax probabilities with the blank at 0, as
  PaddleX's `CTCLabelDecode` expects.

The dictionary is the kraken codec in label order (1622 single NFD codepoints
for medium; space is label 1). PaddleX prepends `blank` itself and, because
`use_space_char` is always on for CTC, appends one extra space at an index the
network never emits.

Output text is exactly kraken's: NFD, diplomatic (long s, combining marks,
`⸗` hyphens). Normalise afterwards if you need NFC.

## Verification

```
squiddle extract-lines page1.jpg page2.jpg -o lines/     # kraken segmentation -> line PNGs
squiddle verify squiddle_PP-OCRv6_medium_rec lines/ --report verify.json
```

`verify` runs every line through kraken (the reference), through ONNX Runtime
on kraken's own tensor, through ONNX Runtime with an exact re-implementation of
PaddleX's preprocessing (batch 1 and batched), through kraken's native batch
mode, and, when `paddlex` is importable, through the real PaddleX predictor.
It reports exact-match rate and character error rate against kraken, and
prints the differing lines. Results on 68 kraken-extracted Fraktur lines
(19th-century German print) are in NOTES.md: the export reproduces kraken
line-for-line at batch size 1, and batched runs reproduce kraken's own batched
output line-for-line.

## Known risks

- **Preprocessing mismatch degrades accuracy silently.** Anything that feeds
  the model differently from kraken (another height, un-inverted input, grey
  padding) still produces plausible text. Run `squiddle verify` on real line
  strips after any change to PaddleX or to the config.
- **Batching changes a few lines.** With `batch_size > 1` the SVTR neck sees
  white padding beyond the line end and a small share of lines change, mostly
  between long s and round s or quote styles. This is inherent to the model:
  kraken's own `-b 8` output is identical to the batched ONNX output. Set the
  recogniser's `batch_size: 1` in the pipeline YAML (or `text_rec_batch_size=1`)
  to reproduce kraken's default single-line results exactly, at a speed cost.
- **Tight detection boxes.** PP-Structure's DBNet detector produces tight
  rectangular crops. Earlier tests showed this model switching from diplomatic
  to normalised transcription on tight crops and reading correctly when the
  crop had a normal ascender/descender allowance. The pipeline's
  `text_det_unclip_ratio` (default 1.5 in both PP-StructureV3 and OCR) expands
  detected boxes. On two Fraktur test pages the default crops kept every long s
  and combining mark but lost most line-final `⸗` hyphens (read as `-`);
  2.0-2.5 recovered most of them and lowered the page CER from about 0.7 % to
  0.2-0.4 %. See NOTES.md.
- **Lines wider than 3200 px are squashed.** PaddleX caps the recogniser input
  width at 3200 px (a 33:1 aspect at 96 px). kraken has no such cap.
- **Layout and detection models are modern-document models.** PP-DocLayout and
  DBNet are trained on modern documents. Their behaviour on historical pages,
  and thus the quality of the reading order, tables and crops SquiddleOCR's
  recogniser receives, is unmeasured. SquiddleOCR cannot fix that.
- **Native Paddle inference format is not provided.** X2Paddle 1.6 cannot
  convert the graph (opset 18, dynamic width; see NOTES.md). The ONNX engine of
  PaddleX is the supported path.

## Licence and credit

Model weights: Apache-2.0, Benjamin Kiessling (ALMAnaCH, Inria Paris). Cite the
Zenodo DOI of the model you use (in `NOTICE` and `squiddle.json` of every
converted directory). The `squiddle_` naming is this project's and not the
author's. Converter code: Apache-2.0. The bundled pipeline templates are
PaddleX's (Apache-2.0).
