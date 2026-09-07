# NOTES

Decisions, measurements and what still has to be validated elsewhere.
Machine: DGX Spark (aarch64, Grace CPU, GB10, CUDA 13), PaddlePaddle 3.2.2 CPU,
PaddleX 3.7.2, PaddleOCR 3.7.0, torch 2.14.0+cu130, kraken from
`/home/seb/dev/kraken` (commit 17a6952, 2026-09-07).

## Facts re-checked (corrections to the brief)

- **kraken inverts the input.** `ImageInputTransforms` ends with
  `tensor_invert` (`im.max() - im`) after `ToDtype(scale=True)`. The network sees
  ink as 1 and paper as 0. Because kraken always pads 16 px of pure white,
  `im.max()` is 1.0 and the inversion equals `1 - x/255`. The export folds
  exactly that in (`wrapper.py`).
- Metadata: input `(1, 3, 96, 0)`, 1623 classes, variant medium, `one_channel_mode`
  `RGB`, `seg_type` baselines, 1622 graphemes, all single NFD codepoints,
  labels 1..1622, space = label 1. Confirmed with `load_safetensors`.
- The test folder has `NNNN.jpg` and `NNNN.txt` only; there are no `.xml`
  files. Line strips for verification are produced by `squiddle extract-lines`
  with kraken's `blla.mlmodel` segmenter.
- torch 2.14.0 on aarch64 from PyPI is the cu130 build; pinning `torch==2.14.0`
  reuses the uv cache.

## PaddleX facts that shaped the design (PaddleX 3.7.2 source)

- `create_predictor` requires `Global.model_name` to equal the requested name
  and to have an engine binding; unregistered names raise. Registered CTC
  recogniser names include `PP-OCRv6_{tiny,small,medium}_rec` and
  `PP-OCRv5_server_rec`. Those `PP-OCRv6_*` entries are Baidu's own PP-OCRv6
  models (48 px, HF-format safetensors, `paddle_dynamic`/`transformers`
  engines); the runner-side preprocessing is identical for all CTC names, so
  the name only has to be *a* registered CTC name. `squiddle convert
  --model-name PP-OCRv5_server_rec` is available if a pipeline should work
  without passing `text_recognition_model_name`.
- Runner preprocessing `OCRReisizeNormImg`: cv2 bilinear resize to the
  configured height, `x/255 - 0.5) / 0.5`, right pad with 0 to
  `int(h * max(imgW/imgH, w/h))`, hard cap `max_imgW = 3200`; `ToBatch` pads to
  the widest line with 0. None of this is configurable, hence the in-graph
  adaptation. `RecResizeImg.image_shape` width acts as a minimum width; set to
  96 (= no padding for lines with aspect > 1) instead of PaddleOCR's 320.
- `CTCLabelDecode(character_list, use_space_char=True)`: prepends `blank`,
  appends `" "`. With space already at label 1 the appended index 1623 is never
  produced. Decoding reads every time step, so the graph forces padded steps to
  blank.
- ONNX engine: `inference.onnx` in the directory, `engine="onnxruntime"`,
  single input fed by name, `device_type` cpu/gpu. Per-sub-module `engine:` keys
  are honoured in pipeline YAMLs (`_resolve_child_engine`). Official models have
  `_onnx` download variants, so a pipeline-wide `engine="onnxruntime"` also works.
- Default engine `paddle` with a model dir resolves the engine from the files
  present and raises "No Paddle model files were found" for an ONNX-only
  directory. HPI (`use_hpip`) needs `ultra-infer`, unavailable on aarch64.
- Colour: `DecodeImage.img_mode: RGB` makes `ReadImage` convert both files and
  in-pipeline BGR crops to RGB.

## Export

- `torch.onnx.export(..., dynamo=True, opset_version=18)`; the legacy exporter
  fails on the backbone's dynamic padding. Weights are embedded in the single
  `inference.onnx` (64 MB medium) by default; `--external-weights` writes
  `inference.onnx.data`.
- Requests for opset 15/17 are silently ignored by the exporter (the ONNX
  version converter fails on `ReduceMean` axes inputs), so the file is opset 18.
- Time steps: `W' = ((w - 1) // 4 - 1) // 2 + 1` for a backbone input of width
  `w` (verified for widths 64..3232). kraken's own length mapping
  (`floor(len * W'/W)`) is not exportable (`float(SymInt)` under
  `torch.compiler.disable`), so the wrapper computes lengths with the formula
  and calls backbone / neck / head directly with the mask.
- ONNX Runtime vs PyTorch on random inputs: max |Δp| 5e-6 at widths 640 and
  1000, 6e-6 for batch 3 at width 320. On real lines (CPU) 2.6e-4.

## Verification results (68 lines from pages 0010, 0011, 0050, medium model)

`squiddle verify work/squiddle_PP-OCRv6_medium_rec work/lines --paddle`:

| comparison | exact | CER vs kraken |
|---|---|---|
| ONNX on kraken's tensor | 68/68 | 0 |
| ONNX, PaddleX preprocessing re-implemented, batch 1 | 67/68 | 0.038 % |
| ONNX, PaddleX preprocessing, batch 8 | 65/68 | 0.114 % |
| kraken native batch 8 (`kraken ocr -b 8` semantics) | 65/68 | 0.114 % |
| PaddleX `create_predictor` (onnxruntime), batch 1 | 67/68 | 0.038 % |
| PaddleX `create_predictor` (onnxruntime), batch 8 | 65/68 | 0.114 % |

Discrepancies, all explained:

- Batch 1, one line (`0010_0000.png`, a 16x56 px fragment): kraken returns
  "", PaddleX returns "2". PaddleX resizes it to 27 px wide and pads to the
  96 px minimum width; the padding is detected and masked, but the SVTR neck's
  local convolution still sees the boundary differently from an unpadded run.
  Real text lines are unaffected (exact match on the other 67).
- Batch 8: three lines change (`statten`/`ſtatten`, `entsagt`/`entſagt`, the
  fragment above). The batched ONNX output is identical, line for line, to
  kraken's own batched output, so this is the model's sensitivity to the
  white padding beyond the line end, not a conversion error. Use
  `batch_size: 1` for kraken's default behaviour.
- The PaddleX-style cv2 bilinear resize (vs kraken's Lanczos) changed no line.

## PP-StructureV3 end to end (CPU, aarch64)

`PPStructureV3(paddlex_config=<generated YAML>, engine="onnxruntime", device="cpu")`
with doc orientation/unwarping, formula, chart and seal recognition off, table
recognition and region detection on. With the default `paddle` engine Paddle
Inference segfaults in `AnalysisPredictor::Init` on this aarch64 CPU (also
with `enable_mkldnn=False`), so every model ran on ONNX Runtime (PaddleX
downloads the official `_onnx` variants). Pipeline init 31 s, 10-16 s per page on
the 20-core CPU. Page text = `overall_ocr_res.rec_texts` joined, compared NFD
against the reference transcription (`NNNN.txt`, whole page, reading order as
produced). Pages 0010 and 0050, medium model:

| page | `text_det_unclip_ratio` | lines found / ref | CER | `⸗` kept / ref | `ſ` / ref | combining marks / ref |
|---|---|---|---|---|---|---|
| 0010 | 1.5 (default) | 23 / 23 | 0.66 % | 1 / 7 | 21 / 21 | 15 / 15 |
| 0010 | 2.0 | 23 / 23 | 0.44 % | 3 / 7 | 21 / 21 | 15 / 15 |
| 0010 | 2.5 | 23 / 23 | 0.22 % | 5 / 7 | 21 / 21 | 15 / 15 |
| 0050 | 1.5 (default) | 23 / 23 | 0.79 % | 0 / 3 | 25 / 25 | 11 / 11 |
| 0050 | 2.0 | 23 / 23 | 0.68 % | 1 / 3 | 25 / 25 | 11 / 11 |
| 0050 | 2.5 | 23 / 23 | 0.68 % | 2 / 3 | 25 / 25 | 11 / 11 |

Reading: long s, umlaut-e combining marks and the diplomatic transcription
survive DBNet's crops on these pages; the one systematic loss is the
line-final double hyphen `⸗`, which the tight boxes cut and the model then reads
as `-`. Expanding the boxes (`text_det_unclip_ratio` 2.0-2.5) recovers most of
them and lowers CER monotonically on both pages. Remaining errors are
single-character. Layout, reading order and tables were not evaluated (no
reference).

## Native Paddle format (X2Paddle): not feasible

Attempted with x2paddle 1.6.0 (needs `onnx<1.17`; 1.16.1 installed):

1. opset 18 rejected (`LayerNormalization` unsupported, opset must be <= 15);
2. after rewriting `LayerNormalization` to primitives, folding constants,
   converting `ReduceMean/ReduceMax` axes to attributes and stamping opset 15
   (`scripts/x2paddle_attempt.py`; ONNX Runtime output unchanged), the mapper
   fails on the dynamic-shape subgraph: `convert failed node: sym_size_int_84,
   op_type is Squeeze`.

X2Paddle has no support for symbolic shapes; a fixed-width export would lose
the free-width contract. PaddleX's `paddle_dynamic` engine for
`PP-OCRv6_*_rec` loads HF-format safetensors into PaddleX's own Paddle
implementation, but its preprocessing is the fixed `[-1, 1]` path, so kraken
weights would run un-inverted at the wrong scale. Not pursued.

## Not done / deferred

- **transformers route.** transformers 5.16.1 ships `pp_ocrv6_small_rec` /
  `pp_ocrv6_tiny_rec` with a configurable image processor (mean/std/size), and
  PaddleX has a `transformers` engine for those names. A weight-name mapping
  from kraken to that layout plus `image_mean=[1,1,1], image_std=[-1,-1,-1]`,
  `size.height=96` might give a native PyTorch/GPU path, but the 16 px white
  padding and the batch masking are not expressible there, and the layout
  parity is unverified.
- tiny and small models were not converted here (not downloaded); the code
  paths are variant-agnostic (`build_recognizer` reads the variant from the
  file; the tiny head has the extra guide layer and `fc1/fc2`).

## Prebuilt PaddlePaddle GPU wheels for aarch64 (researched 2026-09-07)

- PyPI `paddlepaddle` 3.3.1 has no aarch64 wheel at all; 3.2.2 has a CPU one.
  `pyproject.toml` therefore pins `paddlepaddle>=3.2.2,<3.3` and declares
  aarch64 + x86_64 as `tool.uv.required-environments` so `uv sync --extra paddle`
  resolves on both.
- Paddle's own index `https://www.paddlepaddle.org.cn/packages/stable/cu130/`
  (and `cu132/`) carries exactly one aarch64 GPU wheel,
  `paddlepaddle_gpu-3.4.0.post20260612+99af93f9466-cp312-cp312-linux_aarch64.whl`
  (255 MB, Python 3.12 only; the nightly index has a 3.3.1.post20260403 one).
  Installed and tested here: it is compiled for **sm_100 only** (GB200-class
  Grace-Blackwell). On the GB10 (compute capability 12.1) Paddle aborts with
  "Mismatched GPU Architecture: compiled for 100, but your current GPU is 121"
  before running a single kernel. Not usable on the DGX Spark.
- GitHub: PaddlePaddle/Paddle#76215 (DGX Spark support request, open since
  2025-11, no maintainer reply), PaddleOCR#17077 (closed stale), PaddleOCR
  discussion #17328. The discussion has one community wheel for GB10 / CUDA
  13.0 / Python 3.12 shared through a private ProtonMail link (not a
  reproducible or auditable source) and a build recipe.
- Build recipe (news.metaparadigma.de, April 2026, ~40 min on a DGX Spark):
  Paddle 3.3.0 or develop, CUDA 13.0, cuDNN + NCCL installed by hand, CMake with
  `-DWITH_GPU=ON -DCUDA_ARCH_BIN="12.1" -DWITH_ARM=ON -DWITH_AVX=OFF` and
  `CMAKE_CXX_FLAGS="-U__ARM_NEON -DEIGEN_DONT_VECTORIZE=1"` (Eigen fails to
  compile otherwise), Ninja. A build of v3.3.1 was started here (the
  configure step additionally needed unshallowed submodules, pip in the build
  venv and explicit Python 3.11 include/library paths) and then abandoned:
  **decision 2026-09-07: GPU inference on this machine goes through
  `onnxruntime-gpu`** (next section), no custom Paddle build. All build files
  were removed.

## GPU through ONNX Runtime on the DGX Spark (2026-09-07)

PyPI `onnxruntime-gpu==1.29.0` ships an aarch64 wheel built against CUDA 13
and runs on the GB10 as is. The CUDA 13 runtime, cuBLAS and cuDNN 9 it needs
are already in the venv as the pip packages torch pulled in; they only have
to be on `LD_LIBRARY_PATH` (`site-packages/nvidia/cu13/lib` and
`nvidia/cudnn/lib`) or loaded with `onnxruntime.preload_dlls()`.

Recogniser alone, batch of 8 lines at 1200 px: 2330 ms on the 20 Grace cores,
180 ms on the GB10.

PP-StructureV3 with every model on `engine="onnxruntime"`, `device="gpu"`
(same script and pages as the CPU table above; outputs and CER identical):

| page | unclip | CPU (s) | GPU (s) |
|---|---|---|---|
| 0010 | 1.5 | 16.2 | 2.3 (first page, includes warm-up) |
| 0010 | 2.0 | 10.7 | 0.9 |
| 0010 | 2.5 | 9.6 | 0.8 |
| 0050 | 1.5 | 11.9 | 0.9 |
| 0050 | 2.0 | 10.5 | 0.8 |
| 0050 | 2.5 | 9.6 | 0.7 |

`onnxruntime-gpu` replaces the `onnxruntime` package (same import name), so it
is installed by hand rather than as an extra.

## Every 10th Fraktur test page, PP-StructureV3 on GPU (2026-09-07)

`scripts/eval_fraktur_pages.py 2.0`: pages 0010, 0020, ... 0230 (23 pages),
PP-StructureV3 with the squiddle medium recogniser, all models on ONNX Runtime
CUDA, `text_det_unclip_ratio=2.0`, doc preprocessing / formula / chart / seal
off. Page text = `overall_ocr_res.rec_texts` joined in the pipeline's reading
order, compared NFD against `NNNN.txt`. The reference files are kraken
transcriptions (kraken's own page output from `output/kraken/txt` scores
0.06 % against them), so the numbers measure the whole PP-Structure pipeline
(detector, crops, reading order) around an identical recogniser.

| | pages | ref chars | CER |
|---|---|---|---|
| all 23 pages | 23 | 20427 | 1.30 % |
| 21 regular text pages | 21 | 19059 | 0.41 % |
| same, with `⸗`→`-` and quote glyphs normalised | 21 | 19059 | 0.06 % |

- 0.83 s per page on the GB10 (first page 2.1 s with warm-up).
- On the 21 regular pages almost every remaining error is a line-final
  `⸗` read as `-` (27 of 78 kept, even at unclip 2.0) or a quotation mark
  variant (`„ “` vs `"`), plus an occasional character at a line edge (a
  trailing `“` or `—"` appended, page number `2` read as `42`). Long s and
  combining diacritics were read correctly throughout.
- Page 0220 (an advertisement page with title, price and indented review
  block): CER 13.6 %, 38 lines instead of 33. The layout/reading-order stage
  reordered the price and title lines and split one line into three; the
  recogniser's text on the lines it saw is correct. This is the
  modern-document layout model risk from the README, measured.
- Page 0230 has 3 reference characters (a page number); both pipelines miss it.
- Output folder `work/eval_every10/`: per page `NNNN.squiddle.txt` (pipeline
  text), `NNNN.squiddle.md` (PP-StructureV3 Markdown), `NNNN.kraken.txt`
  (reference), plus `summary.json` with per-page CER and timings.

## Must be validated on an x86 GPU machine

- `engine="onnxruntime", device="gpu"` with `onnxruntime-gpu` (CUDA provider):
  parity with the CPU numbers above.
- `use_hpip=True` (HPI, `ultra-infer`) picking up `inference.onnx` for the
  recogniser while the other models run on Paddle Inference / TensorRT; the
  `Hpi.backend_configs` block in `inference.yml` is copied from the official
  layout with height 96 and has not been exercised.
- PaddlePaddle GPU builds (`paddlepaddle-gpu` cu130 exists only for x86) for the
  other PP-StructureV3 models at full speed; on aarch64 everything Paddle-side
  ran on CPU.
- `paddleocr pp_structurev3 ... --engine onnxruntime` CLI path end to end.
