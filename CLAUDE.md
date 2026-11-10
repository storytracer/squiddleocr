# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. README.md says what it does; NOTES.md has decisions,
measurements and open validation.

## What this is

SquiddleOCR is the `squiddle` recipe of linecaster, run by one PEP 723 uv script, `squiddle.py`, with its lock
`squiddle.py.lock`. There is no package, no `pyproject.toml` and no tests here. The script imports linecaster's
modular pipeline shape (`linecaster.pipelines.transcribe_modular`) and runs it with `linecaster.runtime.run.main`,
with SquiddleOCR's defaults where the command line leaves them open: `--recipe squiddle`, `--formats md`,
`--naming squiddleocr` (`<page>.paddle.<ext>`, flat) and `--files OUT` (`options` sets the first three with
`ap.set_defaults`, `pipeline` the last). Everything else is linecaster's: the blocks, the recipe and its presets
(the catalog), the command line (`IMAGES OUT`, `--on`, `--tasks`, `--workers`, `--set`, `--backend`, `serve`, ...),
the run folder (`pages/`, `rejects/`, `logs/`, `traces/`, `run.yaml`) and the failed-page handling (no files for a
page a block failed on: `PageFileWriter(skip_failed=True)`; `rejects/` lists it).

- **Change behavior in linecaster, not here.** The chain, its blocks, the typography and recomposition rules and the
  recipe live in linecaster; a project's choice made here would fork them. `squiddle.py` holds only defaults.
- **Defaults only where they are SquiddleOCR's.** The script maps no 0.4 flag; README's table says what each became.
  A new default must be a one-liner over the modular pipeline's options.

## The environment

- `squiddle.py`'s metadata mirrors linecaster's `pipelines/transcribe_modular.py`, restricted to the squiddle
  recipe's engines: `linecaster`, `linecaster-kraken`, `linecaster-paddlex`, `kraken` (from the fork
  storytracer/kraken, rev `48eecb78`, listed directly because sources only apply to a script's own dependencies)
  and the OpenCV contrib stub, with the same `constraint-dependencies` list, which holds the goldens' numerics. When
  linecaster's list changes, copy it.
- **Sources point at `../linecaster` until linecaster's release 0.3.0**; then they switch to the git tag `v0.3.0`
  (the engines and the stub by `subdirectory`); the lines are in the script's comment. The user makes that switch.
- **After changing the metadata: `uv lock --script squiddle.py`.** Seed it from linecaster's lock
  (`cp ../linecaster/pipelines/transcribe_modular.py.lock squiddle.py.lock` first) so every shared package keeps the
  same version, then check that torch is still 2.14.0.
- **torch never changes as a side effect of a resolve** (a new dependency with its own torch bound, another package
  index, `uv run --with`). torch 2.14.0 / torchvision 0.29.0 come from linecaster-kraken's pins: the PyPI wheels are
  the CUDA 13 build on Linux (aarch64 and x86_64), which the GB10 needs, and 2.14 is kraken's upper bound. A change
  of build is deliberate, recorded in NOTES.md, and compared at CER level, not against the goldens.
- OpenCV must be 4.10.0 (`cv2.__version__`): PaddleX pins `opencv-contrib-python` 4.10.0.84 and linecaster's stub
  swaps in the headless build of the same version; with OpenCV 5 the goldens fail.
- The script's environment is uv's (`uv python find --script squiddle.py` names its Python). The `.venv` of the
  former package is not used.
- Models come from the Hugging Face cache at the revisions linecaster's catalog pins, fetched on first use; kraken's
  models always from their Hugging Face mirrors, never from Zenodo, which throttles.

## This machine (DGX Spark, aarch64, GB10, CUDA 13)

- kraken and PP-FormulaNet run on torch cu130 on the GPU; the other PaddleX models on ONNX Runtime's CUDA provider.
  Paddle Inference segfaults here; never use the Paddle engine.
- Test images: `~/data/squiddletest/scans/` (17 mixed scans; `12342041.jpg` and `9739692.jpg` have tables),
  `~/data/squiddletest/formulas/` (12 Internet Archive pages). The goldens: `~/data/goldens/squiddleocr-7c8a052/spark`
  with their images in `~/data/bhl-ocr-eval/{sample-60,tables-7}`. The Fraktur book under `~/data/nls/fraktur_test/`
  is only for CER measurements, and only when asked.
- A page takes 1 to 5 s. Cap probes at 30 to 40 s with a traceback dump and debug; never wait a stall out.

## Commands

```
uv run squiddle.py --help
uv run squiddle.py IMAGES OUT                                   # <page>.paddle.md in OUT
uv run squiddle.py IMAGES OUT --formats md,hocr,alto --dry-run  # the plan, nothing run
uv run squiddle.py serve                                        # linecaster serve with the squiddle models
uv lock --script squiddle.py
```

## The check: the goldens

After any change to the script, its lock or linecaster, the goldens must stay exact (line boxes, text after NFC,
every word box), on both samples:

1. `uv run squiddle.py ~/data/bhl-ocr-eval/sample-60 work/goldens/sample-60`
2. `uv run --no-project --with lxml --with numpy python ../linecaster/tests/parity/compare_goldens.py work/goldens/sample-60/pages ~/data/goldens/squiddleocr-7c8a052/spark/sample-60`
3. The same with `tables-7`.

Use a fresh OUT each time: a run into an existing OUT resumes it. Record the results in NOTES.md.

Scratch outputs and experiments go to `work/` (git-ignored); keep one-off scripts out of the repo and record their
results in NOTES.md.

## Open items (details in NOTES.md)

- The switch of the sources to linecaster's tag `v0.3.0` at the release.
- `--on ray` and `--on hf-jobs` with `squiddle.py`: not tried.
- Table structure in hOCR / ALTO / PAGE; inline formulas; tables with unruled rows; PaddleOCR-VL for formulas and
  tables (linecaster's work now).
- Right-to-left pages and CPU-only runs: unchecked.

## Working style

Small commits with plain messages; do not push unless asked. Record anything that cannot be validated on this machine
in NOTES.md instead of guessing.
