# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. README.md says what it does; NOTES.md has decisions,
measurements and open validation.

## What this is

SquiddleOCR is the `squiddle` recipe of linecaster, run by linecaster's own recipe script,
`recipes/squiddle.py`, straight from its raw GitHub URL on linecaster's `main` branch (the user's decision,
2026-11-12). This repository holds no code: README.md (what SquiddleOCR does and how to run it), NOTES.md, LICENSE.
No package, no script, no lock, no tests. Everything is linecaster's: the script and its environment (git sources on
`main`; linecaster-kraken brings the kraken fork and torch 2.14.0, linecaster-paddlex the OpenCV stub), the blocks,
the recipe and its presets (the catalog), the command line (`IMAGES OUT`, `--on`, `--tasks`, `--workers`, `--set`,
`--backend`, `serve`, ...), the run folder (`pages/`, `rejects/`, `logs/`, `traces/`, `run.yaml`) and the
failed-page handling.

- **Exactly linecaster's recipe, no options or defaults of its own** (the user's decision, 2026-11-10). Change
  behavior in linecaster (the chain, its blocks, the rules, the recipe, the script), never here.
- **No versions, tags or releases** (the user's decision, 2026-11-12), and nothing to move when linecaster makes a
  release: every push to linecaster's `main` reaches SquiddleOCR's users at once. The README follows when an option
  it documents changes in linecaster.
- **torch never changes as a side effect of a resolve**; it is linecaster-kraken's pin (2.14.0), and OpenCV must be
  4.10.0 (linecaster's stub). Both are linecaster's business; a run here only checks them.

## This machine (DGX Spark, aarch64, GB10, CUDA 13)

- kraken and PP-FormulaNet run on torch cu130 on the GPU; the other PaddleX models on ONNX Runtime's CUDA provider.
  Paddle Inference segfaults here; never use the Paddle engine.
- Test images: `~/data/squiddletest/scans/` (17 mixed scans; `12342041.jpg` and `9739692.jpg` have tables),
  `~/data/squiddletest/formulas/` (12 Internet Archive pages), the BHL samples in
  `~/data/bhl-ocr-eval/{sample-60,tables-7}`. The Fraktur book under `~/data/nls/fraktur_test/`
  is only for CER measurements, and only when asked.
- A page takes 1 to 5 s. Cap probes at 30 to 40 s with a traceback dump and debug; never wait a stall out.

## Commands

```
uv run https://raw.githubusercontent.com/storytracer/linecaster/main/recipes/squiddle.py --help
uv run https://raw.githubusercontent.com/storytracer/linecaster/main/recipes/squiddle.py IMAGES OUT --files OUT/files
uv run https://raw.githubusercontent.com/storytracer/linecaster/main/recipes/squiddle.py serve
```

The URL runs what is pushed to linecaster's `main`; unpushed linecaster changes are tried in the linecaster checkout
(its `pipelines/` and tests). Use a fresh OUT for each run: a run into an existing OUT resumes it.
Scratch outputs and experiments go to `work/` (git-ignored); keep one-off scripts out of the repo and record their
results in NOTES.md.

## Open items (details in NOTES.md)

- `--on ray` and `--on hf-jobs` with the recipe script: not tried.
- Table structure in hOCR / ALTO / PAGE; inline formulas; tables with unruled rows; PaddleOCR-VL for formulas and
  tables (linecaster's work now).
- Right-to-left pages and CPU-only runs: unchecked.

## Working style

Small commits with plain messages; do not push unless asked. Record anything that cannot be validated on this machine
in NOTES.md instead of guessing.
