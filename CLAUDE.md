# CLAUDE.md

Guidance for working on SquiddleOCR in this checkout. Read README.md for what
the tool does and NOTES.md for decisions, measurements and open validation.

## What this is

A converter that packages kraken's PP-OCRv6 recognisers (safetensors) as
PaddleOCR / PP-StructureV3 text recognition model directories (ONNX +
`inference.yml` + dictionary). Package `squiddleocr`, CLI `squiddle`, src
layout, `uv` project.

## Environments (do not reinstall torch)

- `.venv` — converter + tests (`uv sync --extra test`); `paddle` extra also
  installs paddlepaddle CPU, paddleocr, paddlex. torch is pinned to
  `2.14.0` / torchvision `0.29.0`: the aarch64 PyPI wheel is the cu130 build
  and is in the uv cache. Never upgrade or reinstall torch in any venv.
- `.venv-paddle` — PaddleOCR/PaddleX venv with `onnxruntime-gpu`; used for
  `squiddle verify --paddle` and pipeline runs. Not tracked.
- kraken comes from `/home/seb/dev/kraken` as an editable path dependency
  (`[tool.uv.sources]`); it has the `kraken.lib.ppocr` package that PyPI
  kraken may not have yet.

## This machine (DGX Spark, aarch64, GB10, CUDA 13)

- Paddle Inference (`engine="paddle"`) segfaults here; always run PaddleX
  with `engine="onnxruntime"`. GPU: `device="gpu"` with `onnxruntime-gpu`
  and `LD_LIBRARY_PATH=$SP/cu13/lib:$SP/cudnn/lib` where
  `SP=<venv>/lib/python3.11/site-packages/nvidia`.
- No `paddlepaddle-gpu` for this GPU (the official aarch64 wheel is sm_100
  only). Decision: no custom Paddle build; do not propose one.
- Test data (read-only, never modify): the Fraktur book under
  `~/data/nls/fraktur_test/images/.../images/` (`NNNN.jpg` + `NNNN.txt`
  kraken reference transcriptions, no XML). kraken models in
  `~/.local/share/htrmopo/`.

## Commands

```
uv sync --extra test
.venv/bin/python -m pytest -q                 # 32 tests; SQUIDDLE_SKIP_SLOW=1 skips the real-model tests
squiddle convert <model.safetensors> -o squiddle_PP-OCRv6_<size>_rec
squiddle extract-lines page.jpg -o lines/     # kraken segmentation -> line PNGs
squiddle verify <model_dir> lines/ --paddle   # run from .venv-paddle for the PaddleX rows
squiddle pipeline-config <model_dir> -o PP-StructureV3_squiddle.yaml
scripts/eval_fraktur_pages.py 2.0             # every 10th test page, CER vs reference (needs .venv-paddle, GPU)
```

Scratch outputs go to `work/` (git-ignored): the converted medium model,
extracted lines, verify reports, `eval_every10/`.

## Invariants to keep

- kraken's line contract is folded into the ONNX graph (`wrapper.py`):
  `x = 0.5 - 0.5 * x_paddle`, 16 px white padding, batch-padding detection
  with kraken-style masking, softmax `(N, W', C)` with blank at 0. Any change
  there must keep `squiddle verify` at 100 % exact vs kraken at batch size 1.
- `inference.yml` `Global.model_name` must be a PaddleX-registered name
  (`PP-OCRv6_<size>_rec`); the directory name `squiddle_PP-OCRv6_<size>_rec`
  is ours. Dictionary = codec in label order; PaddleX prepends `blank` and
  appends a space itself.
- Model card, NOTICE, LICENSE and DOI must be emitted with every conversion;
  weights are Benjamin Kiessling's (Apache-2.0).

## Working style

Small commits with plain messages; do not push unless asked. Record anything
that cannot be validated on this machine in NOTES.md instead of guessing.
