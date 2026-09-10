# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. Read README.md for what
the tool does and NOTES.md for decisions, measurements and open validation.

## What this is

Document OCR for historical print with two pipelines behind kraken's PP-OCRv6
recogniser. `--pipeline paddle`: PaddleX layout analysis (PP-DocLayoutV3),
PP-OCRv6 text detection, PaddleX's table pipeline fed with our lines and
kraken's text, PP-FormulaNet formulas, a `DoclingDocument` in the middle
(Markdown, HTML, DocLang, JSON, txt) and kraken's serialiser for hOCR / ALTO /
PAGE. `--pipeline kraken`: blla on the whole page, kraken's records and line
order, hOCR / ALTO / PAGE / txt only, what the `kraken` command does. Nothing
is converted: kraken's models run on torch through kraken's own code. Package
`squiddleocr`, CLI `squiddle` with `ocr` and `models pull|path`, src layout, `uv`
project.

Layout of `src/squiddleocr/`: `types.py` (Page, Region, TextLine, TableResult,
...), `runtime.py` (device names, ONNX Runtime providers), `paddle_compat.py`
(quiet PaddleX predictors and pipelines: ONNX Runtime engine, transformers
engine for PP-FormulaNet, BGR handoff), `models.py` (kraken models by DOI via
htrmopo, eynollah's ONNX bundle from Zenodo into `~/.local/share/squiddleocr`), `recognizers/`
(`KrakenRecognizer` = kraken's `RecognitionTaskModel`), `detectors/` (PP-OCRv6 det, blla,
eynollah lines), `layout/` (PP-DocLayout, XY-cut, single region, eynollah regions),
`eynollah/` (machine detection and run plan, `python -m squiddleocr.eynollah.launch`, subprocess
runner with RAM watchdog, PAGE-XML parser), `tables/` (PaddleX's
`table_recognition_v2` with our OCR result, SLANet_plus end to end by default),
`formulas/` (PP-FormulaNet, LaTeX), `segmentation.py` (our lines -> kraken
containers), `pipeline.py` (orchestration), `document.py` (DoclingDocument
builder, exports, HTML tables in Markdown for spanned cells), `serialize.py`
(kraken's serialiser; `--detail` through `templates/`: kraken's templates
minus the character level), `factory.py` (names -> pipeline), `cli.py`. Each
stage is one protocol in `<stage>/base.py`.

## The rule

Reuse kraken at the highest abstraction it offers and never reimplement a slice
of it: `RecognitionTaskModel.predict` for recognition (extraction, transforms,
batching, CTC decoding, character cuts, records), `kraken.serialization.serialize`
for XML (custom templates are its own mechanism), `htrmopo.get_model` for model
files. The two segmentations are kraken's own types: `bbox` (PP-OCRv6
detection) and `baselines` (blla). The same goes for PaddleX: use its pipelines
and predictors, never its internals (the table pipeline takes an external OCR
result by design; PP-StructureV3 as a whole does not, so we compose its parts). And for
eynollah: never vendored or forked, configured from outside (env, CLI flags) with one in-memory
patch of `MODEL_VRAM_LIMITS` in `eynollah/launch.py`; it runs as a subprocess because it forks
and spawns worker processes.
Check: `squiddle ocr --pipeline kraken -f hocr` must agree with
`kraken -i page out.hocr -h segment -bl -i blla.mlmodel ocr -m medium.safetensors -B 8`
line for line in text, line boxes and word boxes (`iiif_page_8.jpg`, 34 lines).

## Environment (do not reinstall torch)

- `.venv`: `uv sync --extra paddle --extra eynollah --extra test --inexact`. torch is pinned to `2.14.0` /
  torchvision `0.29.0` (the aarch64 PyPI wheel is the cu130 build, in the uv
  cache). Never upgrade or reinstall torch. The paddle extra brings paddlex
  (`ocr-core,ocr`), transformers and ftfy.
- kraken comes from upstream git (`[tool.uv.sources]`, pinned rev with
  `kraken.lib.ppocr`). To develop against the local fork:
  `uv pip install -e /home/seb/dev/kraken`. The eynollah extra drops eynollah's `tensorrt_cu12`
  pin through `[tool.uv] override-dependencies`.
- kraken models live in `~/.local/share/htrmopo/<uuid5(DOI)>/`, PaddleX's in
  `~/.paddlex/official_models/` (the formula model is 700 MB), eynollah's 1.8 GB ONNX bundle in
  `~/.local/share/squiddleocr/eynollah/v0_9_1/models_eynollah/` (`squiddle models pull eynollah`).

## This machine (DGX Spark, aarch64, GB10, CUDA 13)

- kraken and PP-FormulaNet run on torch cu130 on the GPU. The other PaddleX
  models run through `engine="onnxruntime"` on the CUDA provider
  (`runtime.paddlex_device` preloads the nvidia pip libraries). Paddle
  Inference segfaults here and the official aarch64 `paddlepaddle-gpu` wheel is
  sm_100 only; decision: never use the Paddle engine, no custom build.
- PaddleX reads numpy arrays as BGR (cv2 order); every array goes through
  `paddle_compat.paddle_image`. The test scans are sepia, so it matters.
- Test images: `~/data/squiddletest/scans/` (17 mixed scans; `12342041.jpg`
  and `9739692.jpg` have tables, `iiif_page_8.jpg` is the kraken parity page),
  `~/data/squiddletest/formulas/` (12 Internet Archive pages: Euler, Gauss, two
  19th-century physics textbooks), `work/bhl_tables/` (7 BHL table pages with
  ABBYY table outlines, no cell truth). Use these for probes and smoke runs.
  The Fraktur book under `~/data/nls/fraktur_test/` (kraken reference
  transcriptions) is only for CER measurements, and only when asked.
- eynollah on the GB10: 8 jobs x 2 threads by the plan, ~25 s for 17 pages plus 1 s/page for
  kraken; the runner ends eynollah 2 s after "All jobs done" (its own teardown is 15-25 s).
- A page takes 1 to 5 s. Cap probes at 30 to 40 s with a traceback dump
  (`faulthandler.dump_traceback_later`) and debug; never wait a stall out.

## Commands

```
uv sync --extra paddle --extra eynollah --extra test
.venv/bin/python -m pytest -q
squiddle ocr scans/ -f md,doclang,json                     # <name>.paddle.md next to the images unless -o
squiddle ocr scans/ --pipeline kraken -f hocr,page         # blla on the whole page, kraken records, <name>.kraken.hocr
squiddle ocr scans/ --layout eynollah -f md,page --eynollah-xml work/eyn_xml   # eynollah subprocess, XML kept and reused
squiddle ocr scans/ -f alto --detail word                  # Strings without Glyphs
SQUIDDLE_VERBOSE=1 squiddle ocr page.jpg                   # library warnings back on
```

Scratch outputs, experiments and evaluation scripts go to `work/` (git-ignored;
`work/ocrscout-handover.md` is the brief for integrating SquiddleOCR into
ocrscout, the next step). Keep one-off scripts out of the repo; record their
results in NOTES.md.

## Open items (details in NOTES.md)

- Table structure in hOCR / ALTO / PAGE (PAGE `TableRegion`/`TableCellRole`,
  ALTO `ComposedBlock`, hOCR `ocr_table` with an HTML table): postponed, plan in
  NOTES.
- Inline formulas: not handled; only display formula regions reach
  PP-FormulaNet.
- Tables with unruled rows come back as one cell per column (every PaddleX
  table model); a text-box grid could split them.
- PaddleOCR-VL as an alternative for formula crops (and tables) is untested.
- eynollah: not exercised on an x86 discrete GPU, on CPU only, with TensorRT, with `-tab` or
  with `--rtl`; the watchdog restart is unit-tested only.

## Reflow and the diplomatic / reading distinction

`reflow.py` (`--text reflow`) turns the visual rows of a region into paragraphs for the document
exports: joins, division marks, paragraph starts from geometry, continuation across regions and
pages. Rules use Unicode character properties, geometry and the document's own words only; never
a language name, dictionary or model. It is the first module of a larger topic: the line-level
exports are a diplomatic transcription, and users differ in how far towards a reading
transcription they want to go (reflow -> glyph normalisation such as long s and ligatures ->
emphasis from letter-spacing -> spelling modernisation). Only reflow lives here; the plan is to
grow it to maturity in this repo and then extract it as a library and CLI of its own (NOTES
"Reflow"). Do not add normalisation or modernisation to the OCR pipeline.

## Reflow and the diplomatic / reading distinction

`reflow.py` (`--text reflow`) turns the visual rows of a region into paragraphs for the document
exports: joins, division marks, paragraph starts from geometry, continuation across regions and
pages. Rules use Unicode properties and the Line Breaking Algorithm (`uniseg`), geometry and the
run's own words only; never a language name, dictionary or model. It is the first module of a
larger topic: the line-level exports are a diplomatic transcription, and users differ in how far
towards a reading transcription they want to go (reflow -> glyph normalisation such as long s and
ligatures -> emphasis from letter-spacing -> spelling modernisation). Only reflow lives here; the
plan is to grow it to maturity in this repo and then extract it as a library and CLI of its own
(NOTES "Reflow"). Do not add normalisation or modernisation to the OCR pipeline.

## Working style

Small commits with plain messages; do not push unless asked. Record anything
that cannot be validated on this machine in NOTES.md instead of guessing.
