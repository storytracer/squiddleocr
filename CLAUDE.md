# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. Read README.md for what
the tool does and NOTES.md for decisions, measurements and open validation.

## What this is

A small document-OCR framework: any layout / structure / line-segmentation
model in front, kraken's PP-OCRv6 recogniser (converted to ONNX) behind, a
`DoclingDocument` in the middle, exported as DocLang / Markdown / HTML / JSON.
Package `squiddleocr`, CLI `squiddle`, src layout, `uv` project.

Layout: `types.py` (Page, Region, TextLine, ...), `runtime.py` (ONNX Runtime
providers), `recognizers/`, `detectors/`, `layout/`, `tables/` (one `base.py`
protocol + implementations each), `pipeline.py` (orchestration), `document.py`
(DoclingDocument builder + exports), `factory.py` (names -> pipeline), `convert/`
(kraken -> ONNX), `models.py` (model sources: folder / Hub repo / cache / convert
fallback), `hub.py` (source folder + model card + upload), `integrations/`
(verify against kraken/PaddleX, extract-lines).
Adding a model = one class implementing one protocol; keep it that way.

## Environments (do not reinstall torch)

- `.venv` — everything: `uv sync --extra convert --extra paddle --extra kraken
  --extra test`. Core deps bring `onnxruntime-gpu` (Linux/Windows) or
  `onnxruntime` (macOS) and `docling-core`. torch is pinned to `2.14.0` /
  torchvision `0.29.0` (the aarch64 PyPI wheel is the cu130 build, in the uv
  cache). Never upgrade or reinstall torch in any venv.
- kraken comes from upstream git (`[tool.uv.sources]`, pinned rev with the
  `kraken.lib.ppocr` package; PyPI kraken does not have it yet). To develop
  against the local fork instead: `uv pip install -e /home/seb/dev/kraken`
  after `uv sync` (it leaves torch alone).
- End users install with `uv tool install "squiddleocr[paddle] @ git+..."`;
  test that path after dependency changes (the tool env has no torch, so the
  CUDA libraries must come from our own `nvidia-*` dependencies).

## This machine (DGX Spark, aarch64, GB10, CUDA 13)

- Paddle Inference (`engine="paddle"`) segfaults here; PaddleX models are
  always created with `engine="onnxruntime"` (`runtime.paddlex_device`).
  GPU works out of the box (`--device auto`); `runtime.create_session` calls
  `onnxruntime.preload_dlls()` so no `LD_LIBRARY_PATH` is needed.
- No `paddlepaddle-gpu` for this GPU (the official aarch64 wheel is sm_100
  only). Decision: no custom Paddle build; do not propose one.
- Test data (read-only, never modify): the Fraktur book under
  `~/data/nls/fraktur_test/images/.../images/` (`NNNN.jpg` + `NNNN.txt`
  kraken reference transcriptions, no XML). kraken models in
  `~/.local/share/htrmopo/`.

## Commands

```
uv sync --extra convert --extra paddle --extra kraken --extra test
.venv/bin/python -m pytest -q                 # SQUIDDLE_SKIP_SLOW=1 skips the real-model tests
squiddle ocr scans/ -f md,doclang,json             # outputs next to the images unless -o; model sizes come from storytracer/squiddleocr (Hub) or --models FOLDER
squiddle convert -o squiddleocr-models             # all sizes -> model source folder (Hub layout); squiddle upload publishes it
squiddle extract-lines page.jpg -o lines/     # kraken segmentation -> line PNGs
squiddle verify <model_dir> lines/ --paddle
```

Scratch outputs, experiments and evaluation scripts go to `work/` (git-ignored): the converted
medium model (`work/squiddle_PP-OCRv6_medium_rec`), extracted lines, verify reports. Keep eval
and one-off scripts out of the repo; record their results in NOTES.md.

## Invariants to keep

- kraken's line contract is folded into the ONNX graph (`wrapper.py`):
  `x = 0.5 - 0.5 * x_paddle`, 16 px white padding, batch-padding detection
  with kraken-style masking, softmax `(N, W', C)` with blank at 0. Any change
  there must keep `squiddle verify` at 100 % exact vs kraken at batch size 1.
- `inference.yml` `Global.model_name` must be a PaddleX-registered name
  (`PP-OCRv6_<size>_rec`); the directory name `squiddle_PP-OCRv6_<size>_rec`
  is ours. Dictionary = codec in label order; PaddleX prepends `blank` and
  appends a space itself.
- With `--detector kraken` the line images the recogniser sees are byte-identical to
  kraken's `extract_polygons` output (`KrakenSegmenter.line_images`); `squiddle extract-lines`
  writes that output, so the two can be compared directly.
- Model card, NOTICE, LICENSE and DOI must be emitted with every conversion;
  weights are Benjamin Kiessling's (Apache-2.0).

## Working style

Small commits with plain messages; do not push unless asked. Record anything
that cannot be validated on this machine in NOTES.md instead of guessing.
