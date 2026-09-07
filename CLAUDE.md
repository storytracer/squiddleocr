# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. Read README.md for what
the tool does and NOTES.md for decisions, measurements and open validation.

## What this is

Document OCR for historical print with two pipelines behind kraken's PP-OCRv6
recogniser. `--pipeline paddle`: PaddleX layout analysis, PP-OCRv6 text
detection, SLANet tables and PP-FormulaNet formulas, a `DoclingDocument` in the
middle (Markdown, HTML, DocLang, JSON, txt) and kraken's serialiser for hOCR /
ALTO / PAGE. `--pipeline kraken`: blla on the whole page, kraken's records and
line order, hOCR / ALTO / PAGE / txt only, what the `kraken` command does.
Nothing is converted: kraken's models run on torch through kraken's own code.
Package `squiddleocr`, CLI `squiddle` with one command (`ocr`), src layout,
`uv` project.

Layout: `types.py` (Page, Region, TextLine, ...), `runtime.py` (device names,
ONNX Runtime providers for PaddleX), `paddle_compat.py` (quiet PaddleX
predictors: ONNX Runtime, and the transformers engine for PP-FormulaNet), `models.py` (kraken models by DOI via htrmopo), `recognizers/`
(`KrakenRecognizer` = kraken's `RecognitionTaskModel`), `detectors/` (PP-OCRv6
det, blla), `layout/` (PP-DocLayout, XY-cut, single region), `tables/`
(SLANet_plus), `formulas/` (PP-FormulaNet, LaTeX), `segmentation.py` (our lines -> kraken containers),
`pipeline.py` (orchestration), `document.py` (DoclingDocument builder +
exports), `serialize.py` (kraken's serialiser), `factory.py` (names ->
pipeline), `cli.py`. Each stage is one protocol in `<stage>/base.py`.

## The rule

Reuse kraken at the highest abstraction it offers and never reimplement a slice
of it: `RecognitionTaskModel.predict` for recognition (extraction, transforms,
batching, CTC decoding, character cuts, records), `kraken.serialization.serialize`
for XML, `htrmopo.get_model` for model files. The two segmentations are kraken's
own segmentation types: `bbox` (PP-OCRv6 detection) and `baselines` (blla).
Detail follows native capability, not what we happen to wire through. Check:
`squiddle ocr --pipeline kraken -f hocr` must agree with
`kraken -i page out.hocr -h segment -bl -i blla.mlmodel ocr -m medium.safetensors -B 8`
line for line in text, line boxes and word boxes.

## Environment (do not reinstall torch)

- `.venv`: `uv sync --extra paddle --extra test` (the paddle extra also brings
  transformers for PP-FormulaNet; it runs on the same torch). torch is pinned to `2.14.0` /
  torchvision `0.29.0` (the aarch64 PyPI wheel is the cu130 build, in the uv
  cache). Never upgrade or reinstall torch.
- kraken comes from upstream git (`[tool.uv.sources]`, pinned rev with
  `kraken.lib.ppocr`). To develop against the local fork:
  `uv pip install -e /home/seb/dev/kraken`.
- kraken models live in `~/.local/share/htrmopo/<uuid5(DOI)>/`; the medium
  recogniser and blla are cached there.

## This machine (DGX Spark, aarch64, GB10, CUDA 13)

- kraken runs on torch cu130 on the GPU. PaddleX models run through
  `engine="onnxruntime"` on the CUDA provider (`runtime.paddlex_device` preloads
  the nvidia pip libraries). Paddle Inference segfaults here and the official
  aarch64 `paddlepaddle-gpu` wheel is sm_100 only; decision: never use the Paddle
  engine, no custom build.
- Test images: `~/data/squiddletest/scans/` (7 mixed scans; `12342041.jpg` has
  tables). Use these for probes and smoke runs. The Fraktur book under
  `~/data/nls/fraktur_test/` (kraken reference transcriptions) is only for CER
  measurements, and only when asked.

## Commands

```
uv sync --extra paddle --extra test
.venv/bin/python -m pytest -q
squiddle ocr scans/ -f md,doclang,json                     # <name>.paddle.md next to the images unless -o
squiddle ocr scans/ --pipeline kraken -f hocr,page         # blla on the whole page, kraken records, <name>.kraken.hocr
```

Scratch outputs, experiments and evaluation scripts go to `work/` (git-ignored).
Keep one-off scripts out of the repo; record their results in NOTES.md.

## Working style

Small commits with plain messages; do not push unless asked. Record anything
that cannot be validated on this machine in NOTES.md instead of guessing.
