# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. README.md says what it does; NOTES.md has decisions,
measurements and open validation.

## What this is

SquiddleOCR is the `squiddle` recipe of linecaster, run by one PEP 723 uv script, `squiddle.py`, with its lock
`squiddle.py.lock`. There is no package, no `pyproject.toml` and no tests here. The script imports linecaster's
modular pipeline shape (`linecaster.pipelines.transcribe_modular`) and runs it with `linecaster.runtime.run.main`,
with one default of its own: `--recipe squiddle`. Everything else is linecaster's: the blocks, the recipe and its presets
(the catalog), the command line (`IMAGES OUT`, `--on`, `--tasks`, `--workers`, `--set`, `--backend`, `serve`, ...),
the run folder (`pages/`, `rejects/`, `logs/`, `traces/`, `run.yaml`) and the failed-page handling (no files for a
page a block failed on: `PageFileWriter(skip_failed=True)`; `rejects/` lists it).

- **Exactly linecaster's recipe, no options or defaults of its own** (the user's decision, 2026-11-10): the script
  names the recipe and nothing else, so `squiddle.py IMAGES OUT` is `transcribe_modular.py IMAGES OUT --recipe
  squiddle`. Change behavior in linecaster (the chain, its blocks, the rules, the recipe), never here.

## Releases

SquiddleOCR follows linecaster's [versioning policy][policy] with git tags `v<version>` (no package, so the tag is the
version); its interface is `squiddle.py`'s command line and the recipe it runs. The agent decides and makes the
releases:

- **After every linecaster release**: move the four tags in `squiddle.py` to it, relock, run a smoke test on two
  scans, and release SquiddleOCR with a bump of the same kind (a linecaster minor while 0.x is a SquiddleOCR minor,
  a linecaster patch a SquiddleOCR patch); NOTES.md says which linecaster release it runs.
- **A change of `squiddle.py` itself** bumps by the policy's table (its command line or recipe changed so that a
  command stops working or means something else: minor while 0.x; otherwise patch).
- **A commit that changes only the README is no release.**
- **How:** an annotated tag `v<version>` on `main`, pushed. **Never 1.0.0 without the maintainer's word.**

[policy]: https://github.com/storytracer/linecaster/blob/main/docs/reference/compatibility.md#versions-and-releases

## The environment

- `squiddle.py`'s metadata mirrors linecaster's `pipelines/transcribe_modular.py`, restricted to the squiddle
  recipe's engines: `linecaster`, `linecaster-kraken`, `linecaster-paddlex`, `kraken` (from the fork
  storytracer/kraken, rev `48eecb78`, listed directly because sources only apply to a script's own dependencies)
  and the OpenCV contrib stub.
- **Sources point at a linecaster release tag** (now `v0.3.2`; the engines and the stub by `subdirectory`). Moving
  to a new release: change the four tags, relock and run a smoke test (Releases below). To develop against a local checkout, point
  them at `../linecaster` by path for the time being, never in a commit.
- **After changing the metadata: `uv lock --script squiddle.py`.** Seed it from linecaster's lock
  (`cp ../linecaster/pipelines/transcribe_modular.py.lock squiddle.py.lock` first) so every shared package keeps the
  same version, then check that torch is still 2.14.0.
- **torch never changes as a side effect of a resolve** (a new dependency with its own torch bound, another package
  index, `uv run --with`). torch 2.14.0 / torchvision 0.29.0 come from linecaster-kraken's pins: the PyPI wheels are
  the CUDA 13 build on Linux (aarch64 and x86_64), which the GB10 needs, and 2.14 is kraken's upper bound. A change
  of build is deliberate and recorded in NOTES.md.
- OpenCV must be 4.10.0 (`cv2.__version__`): PaddleX pins `opencv-contrib-python` 4.10.0.84 and linecaster's stub
  swaps in the headless build of the same version; OpenCV 5 moved the line detection (NOTES.md).
- The script's environment is uv's (`uv python find --script squiddle.py` names its Python). The `.venv` of the
  former package is not used.
- Models come from the Hugging Face cache at the revisions linecaster's catalog pins, fetched on first use; kraken's
  models always from their Hugging Face mirrors, never from Zenodo, which throttles.

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
uv run squiddle.py --help
uv run squiddle.py IMAGES OUT                                   # the page records in OUT/pages
uv run squiddle.py IMAGES OUT --files OUT/files --dry-run       # the plan, nothing run
uv run squiddle.py serve                                        # linecaster serve with the squiddle models
uv lock --script squiddle.py
```

Use a fresh OUT for each run: a run into an existing OUT resumes it. Scratch outputs and experiments go to `work/`
(git-ignored); keep one-off scripts out of the repo and record their results in NOTES.md.

## Open items (details in NOTES.md)

- `--on ray` and `--on hf-jobs` with `squiddle.py`: not tried.
- Table structure in hOCR / ALTO / PAGE; inline formulas; tables with unruled rows; PaddleOCR-VL for formulas and
  tables (linecaster's work now).
- Right-to-left pages and CPU-only runs: unchecked.

## Working style

Small commits with plain messages; do not push unless asked. Record anything that cannot be validated on this machine
in NOTES.md instead of guessing.
