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

See the table at the end of this file (filled from `work/ppstruct/summary.json`).

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
