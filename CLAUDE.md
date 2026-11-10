# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. Read README.md for what
the tool does and NOTES.md for decisions, measurements and open validation.

## What this is

The squiddle recipe as a linecaster chain, with its command line interface. PaddleX layout analysis
(PP-DocLayoutV3), PP-OCRv6 line detection, PaddleX's table structure filled with kraken's text and
PP-FormulaNet formulas in front of kraken's PP-OCRv6 recogniser, then linecaster's typography,
recomposition and composition into a `DoclingDocument` (Markdown, HTML, DocLang, JSON, txt) and linecaster's
serialisers for hOCR / ALTO / PAGE. The engines are linecaster's (`linecaster-paddlex`, `linecaster-kraken`);
squiddleocr adds the combination and the `squiddle` command. Package `squiddleocr`, src layout, `uv` project,
version 0.4.1 (`squiddle --version` reads it from the installed distribution).

Layout of `src/squiddleocr/`:

- `chain.py`: the blocks in 0.3's order, 0.3's model options mapped onto the engine settings (`DEFAULTS`,
  `models()` for the server config), `composition` and `item_blocks`, `FailedPagesDropped`;
- `cli.py`: `squiddle ocr` (reader, writers, naming), `squiddle models-config`, `squiddle goldens`;
- `goldens.py`: a run's page records against the 0.3 goldens (hOCR), line by line, after NFC.

## The chain

`squiddle ocr` has every feature of 0.3's paddle pipeline: layout (or `--layout none`:
linecaster's `single-region@1` preset), lines (`LineDetector`), `LineAssigner`, recognition (`LineRecognizer`),
`UnicodeNormalizer` (NFC; 0.3 kept kraken's decomposed text), tables (`TableRecognizer`), formulas
(`FormulaRecognizer`), `Typographer`, `Recomposer` (`--text reflow`; `--rtl`), `Composer` (`--sections`); the
rule-based steps on the recipe's presets (`ratios@1`, `unicode-recomposition@1`, `docling-composition@1`;
`chain.composition`), `PageFileWriter` (md, txt, html,
doclang, json, hocr, alto, page, transcription, plot; `--furniture`, `--dividers`, `--detail word|line`;
flat naming, `bucket_by_item=False`) or, with `--per-document`, `Binder` + item-level `Recomposer` and
`Composer` (`chain.item_blocks`) + `ItemDocumentWriter`. The
transcription is linecaster's default, `ocr_new`. The CLI runs its pipeline through linecaster's builder
(`linecaster.runtime.builder.build`, `cli._run`): the blocks' contracts are checked before anything runs (exit
code 2), and the run is traced. A page a block fails on carries the error and goes on: `RejectWriter` writes
it to `rejects/`, the page records keep it, `FailedPagesDropped` (`chain.py`) keeps it out of the files and the
document, and the CLI lists the failed pages and exits with 1.
Naming is 0.3's (`<name>.paddle.<ext>` next to each image unless `-o`; with `-o` also `pages/` (page records),
`logs/`, `traces/` and, when a page failed, `rejects/`; without `-o` these go to a temporary folder).
Both backends: `in-process`, and `served` against `linecaster serve --config` (the server's model settings;
`squiddle models-config` writes them for the given options). Defaults are 0.3's except `--detail` (`word`:
linecaster has no glyph level).

The typography and reflow rules (0.3's `retyper.py` and `reflow.py`) live in linecaster as
`linecaster.text.typography` and `linecaster.text.recomposition`; change them there. Models are fetched by
linecaster's engines on first use from the catalogue's pinned Hugging Face revisions (sha256-checked;
`linecaster.catalog.fetch`); there is no separate pull command. kraken's models never come from Zenodo, which
throttles downloads.

linecaster comes from `../linecaster` (path sources). Check: the goldens test
(`SQUIDDLE_GOLDENS=1 .venv/bin/python -m pytest -q tests/test_chain.py -k goldens`, exact after NFC).
The 0.3 code (`squiddle legacy`: its own pipeline, kraken's blla on the whole page) and eynollah were removed
on 2026-11-10 (NOTES); git history keeps them. blla is discarded; eynollah would come back as a linecaster engine.

## Environment (do not reinstall torch)

- `.venv`: Python 3.12 (`.python-version`; moved from 3.11 on 2026-11-02, outputs byte-identical on
  the test scans).
  `uv sync --extra test --inexact`. torch is pinned to `2.14.0` /
  torchvision `0.29.0`: the PyPI wheels are the CUDA 13 build on Linux (aarch64 and x86_64), which
  the GB10 needs, and 2.14 is kraken's upper bound. torch never changes as a side effect of a
  resolve (a new dependency with its own torch bound, another package index, `uv run --with`): a
  change of build is deliberate, applies to one named environment and is recorded in NOTES.md,
  and parity is then compared at CER level, not against the goldens. linecaster-paddlex brings PaddleX
  (`ocr-core,ocr`), paddlepaddle, ONNX Runtime GPU with the CUDA 13 libraries,
  transformers and ftfy.
- OpenCV: PaddleX pins `opencv-contrib-python` 4.10.0.84; linecaster's stub (`../linecaster/stubs/`) swaps in
  the headless build of the same version. `cv2.__version__` must be 4.10.0: with OpenCV 5 the goldens fail
  (NOTES, 2026-11-06).
- kraken comes from our fork storytracer/kraken (`[tool.uv.sources]`, rev `48eecb78` = upstream main
  bb487219 + `fix/inference-autocast` (3d235ee8, makes `--precision` take effect at inference) + recognition
  lines prepared ahead of the GPU; `~/dev/kraken` is that fork). To develop against the local checkout:
  `uv pip install -e /home/seb/dev/kraken`.
- Models live in the Hugging Face cache (`~/.cache/huggingface/hub`), kraken's and PaddleX's alike, at the
  revisions linecaster's catalogue pins (the formula model is 700 MB); PaddleX's own download into
  `~/.paddlex/official_models/` is left for a PaddleX model name the catalogue does not know.

## This machine (DGX Spark, aarch64, GB10, CUDA 13)

- kraken and PP-FormulaNet run on torch cu130 on the GPU. The other PaddleX
  models run through `engine="onnxruntime"` on the CUDA provider
  (`linecaster_paddlex.runtime` preloads the nvidia pip libraries). Paddle
  Inference segfaults here and the official aarch64 `paddlepaddle-gpu` wheel is
  sm_100 only; decision: never use the Paddle engine, no custom build.
- PaddleX reads numpy arrays as BGR (cv2 order); linecaster-paddlex hands every array over through
  `linecaster_paddlex.compat.paddle_image`. The test scans are sepia, so it matters.
- Test images: `~/data/squiddletest/scans/` (17 mixed scans; `12342041.jpg`
  and `9739692.jpg` have tables), `~/data/squiddletest/formulas/` (12 Internet Archive pages: Euler, Gauss,
  two 19th-century physics textbooks), `work/bhl_tables/` (7 BHL table pages with
  ABBYY table outlines, no cell truth). Use these for probes and smoke runs.
  The goldens test reads `~/data/bhl-ocr-eval/{sample-60,tables-7}` and
  `~/data/goldens/squiddleocr-7c8a052/spark`.
  The Fraktur book under `~/data/nls/fraktur_test/` (kraken reference
  transcriptions) is only for CER measurements, and only when asked.
- A page takes 1 to 5 s. Cap probes at 30 to 40 s with a traceback dump
  (`faulthandler.dump_traceback_later`) and debug; never wait a stall out.

## Commands

```
uv sync --extra test --inexact
.venv/bin/python -m pytest -q
squiddle ocr scans/ -f md,doclang,json                     # <name>.paddle.md next to the images unless -o
squiddle ocr book/ --per-document -f md,txt -o out/        # one document for the book
squiddle ocr scans/ -o out/ --backend served               # against linecaster serve (squiddle models-config)
squiddle goldens OUT ~/data/goldens/squiddleocr-7c8a052/spark/sample-60
```

Scratch outputs, experiments and evaluation scripts go to `work/` (git-ignored;
`work/ocrscout-handover.md` is the brief for integrating SquiddleOCR into
ocrscout). Keep one-off scripts out of the repo; record their
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
- Right-to-left pages and CPU-only runs are unchecked with `squiddle ocr`.

## Working style

Small commits with plain messages; do not push unless asked. Record anything
that cannot be validated on this machine in NOTES.md instead of guessing.
