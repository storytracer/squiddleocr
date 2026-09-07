# NOTES

Decisions, measurements and what still has to be validated elsewhere.

## Model sources (2026-09-07)

A model source is a folder or a Hub model repo with `README.md` + `models/squiddle_PP-OCRv6_<size>_rec/`.
`squiddleocr.models.resolve_model(size, source)`: local folder -> cache of the repo
(`~/.cache/squiddleocr/<owner>--<repo>/`) -> Hub download of just that size's subfolder ->
local conversion from kraken's Hub mirror (`small-models-for-glam/kraken-ppocrv6-<size>`,
byte-identical to Zenodo) when the `convert` extra is present. Default source
`storytracer/squiddleocr` (env `SQUIDDLE_MODELS`, flag `--models`).
`squiddle convert -o FOLDER` builds a source folder (all sizes, or given sizes / kraken files)
with a Hub model card; `squiddle upload FOLDER --repo ...` publishes it.
All three sizes converted cleanly (tiny 3 MB, small 14 MB, medium 64 MB; parity < 2e-6) and are
published at https://huggingface.co/storytracer/squiddleocr (public, Apache-2.0, cards and
NOTICE files included). Verified: an empty cache + `squiddle ocr page.jpg -m small` downloads
only the small model and runs.

## SquiddleOCR 0.2: the pluggable pipeline (2026-09-07)

Decision: SquiddleOCR's heart is a small framework that combines any layout / structure /
line-segmentation model with the kraken PP-OCRv6 recogniser and produces a `DoclingDocument`
(docling-core 2.95, which serialises to DocLang 0.7, Markdown, HTML, DocTags and lossless JSON).
DocLang is the wire format; DoclingDocument is the object we compute with (versioned schema,
serialisers, the harness's ground-truth format). The converter stays.

- Protocols (`recognizers/base.py`, `detectors/base.py`, `layout/base.py`, `tables/base.py`)
  are `typing.Protocol` classes; implementations are plain classes, PaddleX models are used one
  by one through `paddlex.inference.create_predictor(engine="onnxruntime")` inside the `paddle`
  extra, never through PaddleX pipelines.
- ONNX Runtime provider selection lives in `runtime.py` (`auto` = CUDA > CoreML > CPU;
  `tensorrt` on request). `onnxruntime.preload_dlls()` makes the `nvidia-*` pip packages
  visible, so no `LD_LIBRARY_PATH` is needed any more.
- Detection runs once per page (PP-StructureV3 does the same); lines go to the region they
  overlap most, and lines no region claims become `text` regions of their own (a page number the
  layout model missed would otherwise be lost). Per-region detection is available
  (`Pipeline(detect_per_region=True)`).
- Layout post-processing: PaddleX's `layout_nms` and `layout_merge_bboxes_mode="large"` plus our
  own containment suppression (drop a region >80 % inside a larger one). Without it page 0070
  was read twice (CER 100 %). PaddleX's per-class thresholds are copied from its template.
- Reading order: PP-DocLayoutV3's own (see the section below); recursive XY-cut
  (`layout/order.py`, horizontal cuts first) for models without one. On the Fraktur
  advertisement page XY-cut orders correctly where PP-StructureV3's enhanced XY-cut did not
  (0.9 % vs 13.6 % CER).
- Lines are grouped into visual rows and de-duplicated (a box >70 % inside another on the same
  row is a detector duplicate; the v6 detector emits those on Fraktur).
- Tables: `SLANet_plus` for structure and cell boxes; cells are grown by 15 % of their height
  (30 % horizontally) and read with the detector + recogniser, so table text is SquiddleOCR's.
  `SLANeXt_wired` gives better structure tokens but its own cell boxes are scaled by the crop
  width on both axes and stay imprecise after correction (PaddleX pairs it with
  `RT-DETR-L_wired_table_cell_det` instead); integrating that cell detector is the next step
  for tables.
- Evaluation of the new pipeline (every 10th Fraktur page, script since removed from the repo, text export
  incl. furniture, blank lines dropped): 0.445 % CER on the 21 regular pages with or without
  layout, 0.49 % on all 23; 0.67 s/page on the GB10. Seven BHL table pages: 5/7 tables found,
  as with PP-StructureV3.

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
  the name only has to be *a* registered CTC name; SquiddleOCR uses the
  PP-OCRv6 names only.
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

## PP-StructureV3 end to end (CPU, aarch64) — historical, drop-in removed

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
   (a one-off script, since removed from the repo; ONNX Runtime output unchanged), the mapper
   fails on the dynamic-shape subgraph: `convert failed node: sym_size_int_84,
   op_type is Squeeze`.

X2Paddle has no support for symbolic shapes; a fixed-width export would lose
the free-width contract. PaddleX's `paddle_dynamic` engine for
`PP-OCRv6_*_rec` loads HF-format safetensors into PaddleX's own Paddle
implementation, but its preprocessing is the fixed `[-1, 1]` path, so kraken
weights would run un-inverted at the wrong scale. Not pursued.

## PP-DocLayoutV3 replaces PP-DocLayout_plus-L as the default layout model (2026-09-07)

- PaddleX 3.7.2 registers `PP-DocLayoutV3` (PaddleOCR-VL-1.5's layout model) as ONNX-supported;
  `create_predictor(engine="onnxruntime")` downloads `PP-DocLayoutV3_onnx` and runs on the GB10
  (CUDA provider). Only TensorRT is blocklisted for it. 25 classes (`label_list` in its
  `inference.yml`): plus-L's 20 plus `display_formula`, `inline_formula`, `vertical_text`,
  `vision_footnote`, and `header`/`footer` split from `header_image`/`footer_image`.
- Output per box: `cls_id`, `label`, `score`, `coordinate`, `order`, `polygon_points`. PaddleX
  sorts the boxes by the model's predicted order before returning them
  (`layout_analysis/processors.py`, `boxes.shape[1] == 8` branch), then numbers non-furniture
  boxes 1..n in `order` and sets it to `None` for `SKIP_ORDER_LABELS` (figure_title, image, chart,
  table, header, footer, footnote, aside_text, ...). So the list position is the complete reading
  order, including furniture; `regions_from_boxes` uses that and ignores the `order` numbers.
- Post-processing defaults copied from PaddleOCR-VL-1.5's pipeline YAML: threshold 0.3 for all
  classes, `layout_nms`, per-class `layout_merge_bboxes_mode` (`large` for chart, display and inline
  formula, doc and paragraph title, `union` elsewhere). Our containment suppression stays on.
- The pipeline keeps the layout model's order: orphan regions are inserted after the last ordered
  region that ends above them and overlaps them horizontally (`Pipeline._reorder`); a full XY-cut
  is only done when no region carries an order. Previously every page with orphans was re-cut,
  which would have discarded the learned order.
- Comparison on the 7 scans in `~/data/squiddletest/scans` (`work/v3/`): identical region
  sequences on 5 pages; table boxes differ slightly on 3 (cell counts 33/98 vs 25/107, 16 vs 15,
  17 vs 16); on `iiif_page_9` V3 labels the heading `doc_title` (plus-L: `paragraph_title`) and
  puts the woodcut initial before the paragraph next to it. 2.78 vs 2.69 s/page with tables on.
- Not re-run: the Fraktur CER eval (script since removed from the repo); layout affects it only
  through ordering, and the last measurement was 0.445 % with or without layout.
- PaddleX's PP-StructureV3 pipeline has no handling for the V2/V3 order output
  (`inference/pipelines/layout_parsing/` does not mention them); only PaddleOCR-VL uses it.

## PaddleOCR drop-in removed (2026-09-07)

Removed `squiddle pipeline-config`, the bundled `PP-StructureV3.yaml` / `OCR.yaml` templates,
`integrations/paddleocr.py`, `scripts/eval_fraktur_pages.py` and the `paddleocr` dependency
(the pipeline only ever imported `paddlex`). Reason: the native pipeline does everything the
drop-in did, with learned reading order, per-cell table reading and Docling export, and the
second route doubled the README and the model card. What remains is the format underneath:
every converted directory is a standard PaddleX recognition model (`verify --paddle` checks
it loads and agrees). The two PP-StructureV3 sections above are kept as the historical
baseline. The published model card on the Hub still shows the old section until the next
`squiddle upload`.

## kraken detector: line crops byte-identical to kraken (2026-09-07)

`Pipeline` used `crops.line_image` for every detector: a perspective crop for four-point boxes,
a polygon mask for longer polygons. For the kraken segmenter that was close to but not what
kraken feeds its recogniser (`kraken.lib.segmentation.extract_polygons`: boundary mask, white
background, baseline dewarping). Detectors can now define `line_images(page, lines)`;
`KrakenSegmenter` implements it by wrapping each line in a one-line `Segmentation` and calling
`extract_polygons` on the page image (lines without a baseline, or that kraken rejects for a
baseline under 5 px, fall back to `line_image`). Recognition and table-cell reading go through it.

Check on `~/data/squiddletest/scans/iiif_page_8.jpg`: `squiddle extract-lines` PNGs vs the
pipeline's crops from the same segmentation, 34 lines, 34 byte-identical (`work/v3/byte_check.py`).
One remaining difference to kraken's CLI: `Page.load` converts every image to RGB, while kraken
opens 1-bit images as mode `1` and extracts them with nearest-neighbour interpolation; for RGB
and greyscale scans the pixels are the same. Not measured: the CER effect versus the PP-OCRv6
detector's box crops (needs the Fraktur references).

## Output naming (2026-09-07)

Exports carry a tag between the image name and the extension: `<name>.<tag>.md`,
`<name>.<tag>.doclang.xml`, `<name>.<tag>.json`, ... The default tag is the detector name, so the
two line pipelines can be run into the same folder and compared file by file
(`0112.paddle.md` vs `0112.kraken.md`). `--suffix` sets any other tag (e.g. `--suffix kraken-nolayout`
for a variant) and `--suffix none` restores bare `<name>.md`. Layout and recogniser size are not
in the default tag; add them by hand when they vary.

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

## Every 10th Fraktur test page, PP-StructureV3 on GPU (2026-09-07) — historical, drop-in removed

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
  (reference), `NNNN.jpg` (the page image), plus `summary.json` with per-page CER and timings.

## Text detector: PP-OCRv6 medium replaces PP-OCRv5 server (2026-09-07)

PaddleX 3.7.2's `PP-StructureV3.yaml` (also on the develop branch, and in the
PaddleX 3.7 / PaddleOCR docs) still pairs the structure pipeline with
`PP-OCRv5_server_det`; only the plain `OCR.yaml` moved to `PP-OCRv6_medium_det`.
PaddleOCR's `PPStructureV3` shortcut accepts `ocr_version` up to v5. No
official or community PP-StructureV3 template with PP-OCRv6 exists on GitHub,
the PaddleX docs or Hugging Face (HF has the `PaddlePaddle/PP-OCRv6_*_det`
model repos, incl. `_onnx` variants, and `small-models-for-glam/kraken-ppocrv6-*`,
a byte-identical mirror of the kraken weights for the kraken CLI). Decision:
the bundled template and `squiddle pipeline-config` (both since removed) used
`PP-OCRv6_medium_det` for the general and table OCR (`--det-model` overrides);
the seal detector stays `PP-OCRv4_server_seal_det` (no v6 seal detector exists).

Measured, same 23 Fraktur pages and 7 BHL table pages, everything else equal:

| | PP-OCRv5_server_det | PP-OCRv6_medium_det |
|---|---|---|
| 21 regular Fraktur pages, CER | 0.41 % | 0.61 % |
| same, `⸗`/quotes normalised | 0.06 % | 0.28 % |
| `⸗` kept | 27/78 | 26/78 |
| all 23 pages | 1.30 % | 1.48 % |
| BHL table pages: tables found | 5/7 | 5/7 |
| German table header cells filled | 4/7 | 7/7 |

The v6 detector splits some Fraktur lines at wide gaps (e.g. around an em
dash: one line became `predigen. — Ich` / `aber` / `mußz zuvor den`), which
costs recognition accuracy on the fragments; the same finer boxes let the
table cell matcher fill the header cells that v5's whole-row boxes left
empty. Both detectors run the pages at native resolution here
(`limit_side_len` 736 vs 64 with `limit_type: min` makes no difference for
1100 px wide scans). Not evaluated yet: the two detectors on the full BHL
Antiqua set.

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
