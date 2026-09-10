# NOTES

> **2026-09-07, reset to kraken-native recognition.** Everything below about converting kraken's
> PP-OCRv6 models to ONNX, PaddleX/PaddleOCR compatibility of the converted directories, `verify`,
> `extract-lines`, the Hub model repo and the ONNX recogniser is historical: that toolchain was
> removed (see the section right below). It stays here as the record of what was measured.

## Reset: kraken recognises, nothing is converted (2026-09-07)

Decision (the user's): the lowest-level coordinate both the original PP-OCRv6 and kraken's version
produce is the character (CTC time steps); words are a grouping convention on top and lines come
from the segmenter. So the only honest way to offer "levels of detail" is to run kraken's own
recognition on either of kraken's two segmentation types and let kraken serialise. Consequences:

- `KrakenRecognizer` (kraken's `RecognitionTaskModel` on kraken's safetensors from Zenodo, torch,
  GPU here) is the only recogniser. `--segmentation paddle` = PP-OCRv6 detection boxes as a kraken
  `bbox` segmentation; `--segmentation kraken` = blla as a `baselines` segmentation. `--level` is gone.
- Removed: `convert/` (wrapper, ONNX export, codec, config, model card), `integrations/` (`verify`,
  `extract-lines`), `hub.py`, the ONNX recogniser and CTC decoder, the Hub model source logic in
  `models.py`, the `convert` and `kraken` extras (kraken + torch are core now), `huggingface_hub`,
  `pyyaml`, `onnx`, `onnxscript`, `safetensors` as direct deps. The Hub repo `storytracer/squiddleocr`
  with the ONNX directories is no longer referenced by the code.
- Kept: `runtime.py` (device names -> ONNX Runtime providers for PaddleX, CUDA preload),
  `crops.py` (region crops only), everything PaddleX-side, the Docling builder, `serialize.py`
  (records only, no cut-less fallback), `segmentation.py`.
- `paddle` extra is `paddlex[ocr-core]` + paddlepaddle: PaddleX's `create_predictor` needs
  `pypdfium2` and `opencv-contrib-python` (its `PDFReaderBackend`), which used to arrive through
  `paddleocr`.
- Speed: the first kraken-native build ran the user's 17-page folder at 2.73 s/page against 1.53 for
  the ONNX path. Two causes, both in how kraken's API was called, neither in kraken's inference:
  `RecognitionTaskModel.predict` runs `prepare_for_inference` on every call (a new Lightning Fabric,
  the module converted and moved again), and kraken's default of two line-extraction worker
  processes pickles the whole page image into the pool for every line. `KrakenRecognizer` now
  prepares once and calls `PPOCRv6Model.predict` directly, with `num_line_workers=0` (kraken's
  in-process mode): 1.43 s/page on the same folder, kraken-CLI parity unchanged (34/34 lines).
- Smoke runs (`~/data/squiddletest/scans`): `12342041.jpg` (two tables) with paddle segmentation,
  all formats, 365 lines of which 358 are table cells; `iiif_page_8.jpg` with kraken segmentation: 34/34 lines
  identical to `kraken -h segment -bl ocr -B 8` in text, line boxes and word boxes; the recogniser
  reports `cuda:0`, PaddleX runs on the CUDA provider (the "No registered plugin EP device" warning
  is harmless).
- Not measured: CER of kraken's bbox path on PP-OCRv6 boxes versus the old perspective-cropped ONNX
  path (needs the Fraktur references).


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

## Two levels, each native: the kraken level is kraken itself (2026-09-07)

Decision (the user's rule): reuse kraken at the highest abstraction possible, and let each level's
detail be what its stack natively provides. Two earlier steps were superseded the same day and
removed again: a `line_images` hook that fed kraken's `extract_polygons` crops to the ONNX
recogniser (crops were byte-identical to `squiddle extract-lines`, 34/34 lines, but the ONNX
decoder still dropped kraken's character cuts), and a second ONNX export with kraken's calling
convention so the ONNX net could sit under kraken's inference (the dynamo exporter made the width
static once `seq_lens` entered the trace; abandoned rather than debugged).

- `--level kraken` = `KrakenSegmenter` (blla) + `KrakenRecognizer` = kraken's
  `RecognitionTaskModel.load_model(<size>.safetensors)` + `predict(im, Segmentation,
  RecognitionInferenceConfig)`. Line extraction, transforms, batching, CTC decoding, cut scaling and
  record building are kraken's; the weights are kraken's from Zenodo on torch (GPU here, cu130).
  Our part is `segmentation.py`: `TextLine`s -> `BaselineLine`/`BBoxLine` with ids `<region>_l<n>`
  and the region id, so kraken's serialiser groups lines by our layout regions.
- Models are fetched by DOI with `htrmopo.get_model` (what `kraken get` does) into
  `~/.local/share/htrmopo`: recognisers 10.5281/zenodo.21788403/405/410 (tiny/small/medium),
  blla 10.5281/zenodo.14602569 (from its README; the DOI in the old error message was another
  model). `--model` at the kraken level is a size or a kraken model file.
- Two pipelines, `--pipeline paddle|kraken` (2026-09-07, was `--segmentation`). `paddle`: PaddleX
  layout, PP-OCRv6 lines, SLANet tables, PP-FormulaNet formulas, every export (the document formats
  from the `DoclingDocument`, hOCR/ALTO/PAGE from kraken's serialiser with the records grouped by
  layout region). `kraken`: blla on the whole page, kraken's records and line order; only hOCR, ALTO,
  PAGE and txt, since there are no regions or tables to put in the document formats (`-f md` is
  refused with a message). `-f auto` = md for paddle, hocr for kraken (kraken's own default).
- Formulas (2026-09-07): `formulas/paddle.py` runs PP-FormulaNet_plus-L on PaddleX's `transformers`
  engine (PaddleX 3.7 ships `PP-FormulaNet_plus-L_safetensors` for Hugging Face transformers on
  torch; `create_transformers_predictor` in `paddle_compat.py`, on cuda when torch has it). The
  official model source has no ONNX package for any formula model (PP-FormulaNet-S/L, plus-S/M/L,
  UniMERNet, LaTeX_OCR_rec all refuse `engine="onnxruntime"`), and the `paddle_dynamic` path needs
  Paddle, so transformers is the only engine here. Probe on PaddleX's `general_formula_rec_001.png`:
  load 17.6 s with download, 1.1 s per formula, LaTeX correct. In the pipeline, `formula` regions
  (PP-DocLayoutV3 `formula`/`display_formula`; `inline_formula` boxes inside text are already
  dropped by `suppress_contained`) skip line detection and get `RegionContent.formula`; the
  document builder writes a Docling `FORMULA` item (`$$...$$` in Markdown). The paddle extra gained
  `transformers` and `ftfy`.
- PaddleX reads numpy arrays as cv2 does, BGR, and converts to RGB itself for the models that want
  RGB (`ReadImage(format="RGB")`: layout, text detection, formulas; tables take BGR as is). Our
  `Page.image` is RGB and used to go in unchanged, so those models saw swapped channels. Fixed
  with `paddle_compat.paddle_image` (2026-09-07). The test scans are sepia (mean |R-B| 26-57), so
  it matters; runs are deterministic (a repeated run gives byte-identical Markdown). Before/after
  on three scans, `-f md --no-formulas`: `0010.jpg` 137/137 words, two line-final `⸗` recovered
  that were read as `-` before (detection boxes moved slightly), no other change; `iiif_page_8.jpg`
  222/219 words, the heading lost its `section_header` label and three words changed, mixed;
  `12342041.jpg` (tables) 672/919 words, SLANet_plus now returns fewer, larger cells with the first
  column merged where it was repeated per row before; both structures are poor (the cell boxes
  are the known weak point, see above), so this page decides nothing. Kept because PaddleX's own
  pipelines feed `cv2.imread` (BGR) arrays and the text-page changes were improvements.
- Smoke on PaddleX's `demo_paper.png`, paddle pipeline with formulas, `-f md,hocr,json`: 3 display
  formulas as `$$...$$` in Markdown and as `formula` items in the JSON; hOCR has the 58 text lines,
  the formula regions carry no lines (a formula has no kraken record). 4.1 s/page, 9 s to load.
- `--detail line|word|glyph` for hOCR/ALTO/PAGE (2026-09-07). kraken has one switch,
  `serialize(sub_line_segmentation=...)` = the CLI's `--subline-segmentation/--no-subline-segmentation`
  (a general flag, default on, not tied to a model): on = words and glyphs from the character cuts,
  off = text per line. `glyph` and `line` map onto it. `word` (ALTO Strings / PAGE Words without
  Glyphs, the level hOCR and PDF want) is not native: `templates/alto_word` and
  `templates/pagexml_word` are kraken's templates with the Glyph loop removed, rendered through
  kraken's own custom-template path (`template_source="custom"`, the CLI's `-t`). kraken's hOCR
  has no glyph element but keeps the character level in properties (`x_bboxes` on every line,
  `x_confs` on every word; all optional in the hOCR spec, only the word `bbox` is required), so
  `templates/hocr_word` drops both and writes the spec's word confidence `x_wconf` (0-100) instead.
  The word boxes and confidences are still computed by kraken's serialiser from the cuts; only the
  rendering differs. Keep the files in step with kraken's templates when the pinned rev moves (the
  header of each says so). Parity:
  `kraken -i img out.xml -t templates/alto_word segment -bl ocr -m medium.safetensors -B 8` on
  `iiif_page_8.jpg` vs `squiddle ocr --pipeline kraken -f alto --detail word`: 34 TextLines and 216
  Strings each, identical in CONTENT, HPOS, VPOS, WIDTH, HEIGHT and WC, 0 Glyphs. Line level needs a
  third file: kraken's hOCR template renders only word spans, so with the switch off (`kraken -h
  --no-subline-segmentation`) the lines have no text at all; `templates/hocr_line` adds
  `{{ line.text }}` and drops the per-character `x_bboxes` from the line title. Sizes on that page
  (glyph / word / line): ALTO 460 / 121 / 39 KB, PAGE 401 / 121 / 34 KB, hOCR 147 / 85 / 30 KB.
- Tables through PaddleX's table pipeline with our text (2026-09-07). The per-cell stage (SLANet_plus
  cell boxes, PP-OCRv6 re-run on every padded cell crop, kraken per cell) sliced words and dropped
  text whenever a cell box was off; on `12342041.jpg` the Markdown was unusable. The 15:14 run of
  PP-StructureV3 was better because it matches the whole page's OCR boxes into the cells it finds.
  `table_recognition_v2.predict` takes that OCR result from outside (`use_ocr_model=False,
  overall_ocr_res=OCRResult(...)` with `rec_boxes`, `rec_texts`, `doc_preprocessor_res.output_img`),
  so `tables/paddle.py` now feeds it the table region's page-level lines (reading order) and
  kraken's transcriptions; `use_ocr_results_with_table_cells=False` and
  `use_table_orientation_classify=False` keep PaddleX's own OCR out (the assertion on the OCR
  sub-config is only reached with those on). Pipeline flow: table regions get lines from the page
  detection like text regions and are recognised in the same kraken batch; `_tables` then hands
  lines and texts to the recogniser; cell text comes back in the HTML (`cells_from_html`), cell
  boxes from `cell_box_list` when the counts match. Needs `paddlex[ocr]` (13 small packages).
  Seven BHL table pages (`work/bhl_tables`, ABBYY ground truth has table outlines only):
  layout finds tables on 5 (the other two are two-column index lists ABBYY tagged as tables;
  PP-StructureV3 finds none there either). On all 5 the pipeline route gives the same structure as
  PP-StructureV3 with kraken's text: 12342041 t1 2x9 / t2 3x9 (every number in its column, totals
  row right), 9739675 6x7, 9739677 9x2, 9739678 5x5, 9739692 6x4 (header colspans and the
  Forceps/Perigynium rowspans right, three middle rows merged). The old stage's higher word counts
  were duplicates and fragments from overlapping cell crops. Row split on ruled-column tables
  (12342041) is a limit of every PaddleX table model here: each column is one cell.
  Crop margin: with the exact layout box on 12342041 t1 the cell detector returned 25 cells and the
  matching kept 44 words; padded by 8/16/32 px it returned 17 cells and 121/120/119 words, so
  `crop_pad=8`. PP-StructureV3 itself never hits this because its layout box is looser.
  Second round (same day): the first version copied PP-StructureV3's model choice (SLANeXt_wired +
  RT-DETR cells for tables the classifier calls wired) and merged rows that the old per-cell stage,
  which always used SLANet_plus, had kept apart (9739692: three body rows in one cell). Four
  variants on the five tables, all with the pipeline's box matching and kraken's text:
  A SLANeXt_wired + cell detector (PP-StructureV3), B SLANet_plus + cell detector, C SLANet_plus
  end to end (`use_e2e_*_table_rec_model=True`: structure and cell boxes from SLANet_plus's own
  prediction), D SLANeXt end to end. 9739692: A/B 6-7 rows with 8 filled cells, C 7 rows with 16
  filled (every body row separate, Gleitschiene and Acrogynium recovered), D scrambled. 9739675:
  C 28 filled / 49 words vs 26 / 42. 9739677 and 9739678: A/B/C identical, D loses cells. 12342041:
  A 2 rows, B 4 rows with a clean totals row, C 4 rows with a partial split row and the totals, D
  garbage. C is the default now (`PaddleTableRecognizer(e2e=True)`, `table_pipeline_config` with
  SLANet_plus for both classes); `e2e=False` and the config give PP-StructureV3's behaviour back.
  PP-StructureV3's other flags (`use_ocr_results_with_table_cells` with box splitting and re-reading
  by its own recogniser, table orientation) changed three header cells on 9739675 and nothing else.
  Not chosen: plugging kraken into PP-StructureV3 as a whole (no external-OCR parameter on its
  predict; only a monkeypatch of `general_ocr_pipeline.text_rec_model`, kraken's cuts would not
  come back, Docling and the line-level exports would have to be rebuilt from PaddleX's blocks).
- Markdown tables (2026-09-07): Docling's Markdown serialiser writes a spanning cell's text into
  every grid position it covers, so a rowspan became a column of repeated text ("Forceps
  (Perifallo)" three times on 9739692) and looked like a recognition fault. `document.export_markdown`
  swaps in Docling's own `HTMLTableSerializer` for any table with a `row_span`/`col_span` > 1
  (`<table>` with `rowspan`/`colspan`, what PP-StructureV3 writes) and keeps the pipe table for
  plain grids. HTML and JSON always carried the spans.
- Formula test set (2026-09-07): `~/data/squiddletest/formulas/`, 12 pages at 1600 px from the
  Internet Archive's IIIF service (`https://iiif.archive.org/iiif/<id>$<leaf>/full/1600,/0/default.jpg`):
  Euler, Introductio in analysin infinitorum 1748 (`introductioanaly00eule`, leaves 120/160/200),
  Gauss, Disquisitiones arithmeticae 1801 (`disquisitionesa00gaus`, 150/300/450), Lehrbuch der
  Physik 1897 (`11881024bsb`, 160/260/360) and Lehrbuch der Physik und Meteorologie 1876
  (`11763637bsb`, 200/320/440). bhl-impact-gt is useless for this: no maths regions in its ABBYY
  ground truth and none of 12 sampled pages has a formula (2165 pages would take 40-55 min).
  Paddle pipeline, 1.84 s/page: formulas on 5 of 12 pages, 29 in all. Physics 1876 p320: 5 display
  formulas correct (`D=\frac{v\cdot P}{g}`, `D=P\frac{4\pi^{2}r}{g t^{2}}`, a leader-dot row
  becomes `\quad.` repeats); 1897 p360: two geometry statements right. Euler p160: series correct
  (`Q=1+\frac{r}{9^2}+...`, `&c.` kept); p200: 14 regions, the pure formulas right, but display
  lines that mix Latin words and formulas ("posito", "fit", long s) come out as `\mathrm` letter
  soup, PP-FormulaNet's failure mode on text inside a formula box. Gauss: 0 on all three pages,
  every formula is inline in the prose; PP-DocLayoutV3 gives no display region there and its
  `inline_formula` boxes inside text are suppressed, so kraken reads them as text. Inline
  formulas in old mathematics are the open gap; nothing in the pipeline handles them.
- Warnings (2026-09-07): `cli.quiet_libraries` sets kraken's logger to ERROR and ignores PIL's
  numpy `RuntimeWarning` unless `SQUIDDLE_VERBOSE` is set; the polygonizer warning is per line
  (kraken falls back to the line's bounding box) and PIL's divide-by-zero is the zero-width crop
  that follows. Remaining log records go through `tqdm.write`, so the progress bar survives them.
- kraken pipeline on `0010.jpg`, `12342041.jpg`, `iiif_page_8.jpg`: 24 / 204 / 35 lines, 5.0 s/page
  with the table page (blla's own polygonizer warning on one line there, as the kraken CLI shows).
- `--pipeline kraken` implies `--layout none` and no tables (2026-09-07). Reason: blla is a
  whole-page segmenter; running it on every table cell crop (what the table stage did) took
  minutes per table page (`12342041.jpg` never finished: sato ridge filter per cell, polygonizer
  warnings on the tiny crops) and is not what the `kraken` command does. Now blla runs once on the
  page, its line order is kept (`Pipeline.keep_line_order`, no row sorting or de-duplication), and
  the page is the one region, so the outputs are kraken's own pipeline's. Layout analysis and
  tables belong to `--segmentation paddle`.
- Records with cuts are kept on `RegionContent.records` and passed to `serialize` unchanged with
  `sub_line_segmentation=True`, so hOCR, ALTO and PAGE get words and glyphs. kraken's serialiser
  cannot mix cut and cut-less records with text (`max_bbox([])`), so at the kraken level table
  cells are read as kraken records too (cell lines from blla on the cell, or the whole cell box as a
  bbox line) and kept on the table region; `has_cuts` is only true when every text record has cuts.
- `--level paddle` is unchanged: PP-OCRv6 detection + the ONNX recogniser; ALTO/PAGE carry
  line boxes and text, hOCR raises a clear error.
- Check on `~/data/squiddletest/scans/iiif_page_8.jpg`: `squiddle ocr --level kraken -f hocr` vs
  `kraken -i ... -h segment -bl -i blla.mlmodel ocr -m medium.safetensors -B 8`: 34 lines each,
  34 identical in text, line bbox and word text+bbox. ALTO 216 Strings / 1036 Glyphs, PAGE 34
  TextLines / 398 Words. 5.9 s/page including layout and tables. The only difference to the CLI
  is line order (ours follows the layout regions' reading order).
- `KrakenRecognizer.recognize(images)` also exists (kraken's `--no-segmentation` mode, one bbox
  line per image) for callers that only have crops; the pipeline does not use it at this level.

## Output naming (2026-09-07)

Exports carry a tag between the image name and the extension: `<name>.<tag>.md`,
`<name>.<tag>.doclang.xml`, `<name>.<tag>.json`, ... The default tag is the level name, so the
two line pipelines can be run into the same folder and compared file by file
(`0112.paddle.md` vs `0112.kraken.md`). `--suffix` sets any other tag (e.g. `--suffix kraken-nolayout`
for a variant) and `--suffix none` restores bare `<name>.md`. Layout and recogniser size are not
in the default tag; add them by hand when they vary.

## Line-level exports through kraken's serialiser (2026-09-07)

Decision: ALTO and PAGE-XML are not written by our own code. `serialize.py` converts a page's
`RegionContent`s into a kraken `Segmentation` (regions tagged `{'type': [{'type': <Docling
label>}]}` as blla does; `BaselineOCRRecord` for lines with a baseline, `BBoxOCRRecord` for
four-point boxes and for table cells, one record per cell; empty `cuts`/`confidences`) and calls
`kraken.serialization.serialize(..., sub_line_segmentation=False)`, which renders kraken's
`alto` / `pagexml` templates with `line.text`. Checked in PaddleX 3.7.2 / kraken checkout
17a6952: both templates take the `line.text` branch, output is well-formed XML with `Coords`,
`Baseline`, `TextEquiv` (PAGE) and `Polygon`, `BASELINE`, `String CONTENT` (ALTO).

- hOCR is not offered: kraken's hOCR template only emits text inside `ocrx_word` spans built
  from per-character `cuts`; with empty cuts `serialize` fails in `max_bbox`, and the recogniser
  gives no character geometry. Inventing cuts was rejected.
- Line-less regions (pictures, empty regions) are appended after the lined regions by
  `serialize`, so they lose their reading-order position in the XML. Reading order of text
  regions is preserved (records are emitted in region order, grouped by region id).
- `BaselineOCRRecord` requires a baseline; Paddle quads therefore go out as bbox records
  (axis-aligned bounds), not as polygons.
- Page-level formats are per image also in `--per-document` mode (`<image>.<tag>.page.xml`
  next to `<doc>.<tag>.md`). The ALTO `Processing` block records recogniser, detector, layout,
  unclip ratio and tables from the CLI settings.
- Not validated against the ALTO / PAGE XSDs here (no schema files offline); kraken's own
  outputs use the same templates.

## eynollah stage (2026-09-10)

`--layout eynollah` / `--pipeline eynollah`: eynollah 0.9.2 for regions, line polygons and
reading order, kraken's recogniser on the lines, the paddle pipeline's exports. Code:
`squiddleocr/eynollah/` (`resources.py` plan, `launch.py` launcher, `runner.py` subprocess and
watchdog, `pagexml.py`, `source.py`), `layout/eynollah.py`, `detectors/eynollah.py`,
`models.py` (bundle), `squiddle models pull|path`. README section 7 has the user-facing story.

Facts checked in the installed 0.9.2 source that differ from the brief:

- Predictors (one per model) are `spawn`ed, not forked (`Predictor(mp.context.SpawnProcess)`);
  only the page jobs' `ProcessPoolExecutor` uses `fork`. Consequence: the `MODEL_VRAM_LIMITS`
  patch must run at import time of the launcher module, because a spawned child re-imports the
  `-m` main module as `__mp_main__` and creates its ONNX session there; a patch under
  `if __name__ == "__main__"` would not reach it. The env var `SQUIDDLE_EYNOLLAH_VRAM` carries the
  dict so the child applies the same patch.
- The writer emits `TextRegion` types paragraph, heading, drop-capital, marginalia only (no
  `header`), plus `ImageRegion`, `SeparatorRegion`, `TableRegion`; never `GraphicRegion`. The
  reading order lists left marginalia, then the ordered text regions (headings included), then
  right marginalia; drop capitals, images, separators and tables are unreferenced. `Coords` carry
  a `conf` attribute (kept as `Region.score`). Lines have `Coords` only. Output is
  `<dir_out>/<stem>.xml`, skipped when it exists (`-O` overwrites).
- `-di` reads `.jpg .jpeg .png .tif .tiff` only; the runner stages inputs as symlinks in a
  temporary directory (WebP/BMP are written as PNG) so a subset of a folder can be run.
- `onnxruntime-gpu[cuda,cudnn]` is not an aarch64 problem: 1.29 has aarch64 wheels and the extras
  are the CUDA 13 `nvidia-*` packages this project pins anyway. Only `tensorrt_cu12<11`
  (sdist meta-package, CUDA 12) needed the override; uv's
  `override-dependencies = ["tensorrt_cu12; sys_platform == 'never'"]` keeps it in the lock with
  a never-true marker.
- The aarch64 ONNX Runtime GPU build lists no `TensorrtExecutionProvider`, so the TensorRT
  fallback trap (listed provider, missing `libnvinfer`, silent CPU) is an x86 story here;
  `libnvinfer` is not on this machine either. `--eynollah-tensorrt` is coded, not exercised.
- The ORT `GPU device discovery failed ... /sys/class/drm/card0` warning did not appear in
  eynollah's log on this machine (eynollah sets the ORT logger to ERROR before creating
  sessions); the filter stays.
- eynollah's `Predictor` loop polls its task queue with a 4.5 s timeout before checking the stop
  flag, and shutdown joins the six predictors one after another: 15 to 25 s of teardown after
  "All jobs done". The runner ends the process group 2 s after that log line (all XML is on disk
  by then). 3 pages: 18 s → 11 s wall for the eynollah phase.
- There was no `squiddle models` command before; `pull` and `path` were added for eynollah and,
  for symmetry, kraken's sizes and blla.
- Unified memory: `torch.cuda.mem_get_info` on the GB10 reports 56 GB free of 122 while `free`
  shows 115 GB available (the page cache is reclaimable but CUDA does not count it). The plan
  uses psutil's available RAM as the one pool on unified machines.

Measurements (DGX Spark, 17 scans in `~/data/squiddletest/scans`, `-fl -romb`, kraken medium):

| run | eynollah phase | inside eynollah | kraken | RAM available min (of 114 GB) |
|---|---|---|---|---|
| default plan: 8 jobs × 2 threads, caps 2.8 G / 8 G | 44.2 s (before the early stop) | 22.2 s, per page 2.8 to 21 s wall | 18.8 s, 1.1 s/page | 97 GB |
| `--eynollah-jobs 1` (18 threads) | 70.2 s | 43.2 s, per page 1.1 to 5.7 s | 19 s | 100 GB |
| `--eynollah-xml` of the first run | not run | | 20 s | |

All six models on `ONNX provider CUDA`, no BFCArena error, every eynollah text line has a
`TextEquiv`, region counts equal eynollah's non-separator counts on every page, Markdown order
equals the `ReadingOrder` (checked by region sequence), the XML-reuse run reproduces the Markdown
and PAGE-XML byte for byte (timestamps aside), every eynollah process ran at nice 10. The RAM
margin (29 GB) was never approached, so the watchdog's restart path is unit-tested with a fake
process only. The user's own eynollah test pages (`00675290` and friends, outputs in
`~/data/squiddletest/eynollah/out`) were not found as images; the smoke runs used the scans.

Not done: an x86 discrete-GPU run (the caps' split below the 8 GB ceiling, the TensorRT path), a
CPU-only run of the whole pipeline, eynollah's `-tab` on the BHL table pages, Hebrew pages with
`--rtl` (kraken's `BBoxLine` takes the direction; `BaselineLine`s carry none).

### Line stage of the eynollah pipeline (2026-09-10, six newspaper pages in `~/data/squiddletest/eynollah/pages`)

The pages are 1852×2295 to 7072×8416 px, 11 to 157 regions, 92 to 546 lines; eynollah took 8 to
41 s per page (41.5 s for the batch with 8 jobs, the slowest page sets the wall time), kraken
1.5 s per hundred lines. Region counts, `TextEquiv`s and reading order checked as for the scans.

- eynollah's text line mask breaks justified lines at wide word gaps: 31 "lines" in a 16-row
  paragraph of 00675290, single words among them; kraken then reads words without their
  neighbours and the Markdown showed one word per row. The synthesised baselines sit near the
  x-height middle because eynollah pads its polygons far beyond the descenders (58 px polygons for
  35 px type). Kept as `--eynollah-lines eynollah`.
- blla per region (`--eynollah-lines blla`, `KrakenSegmenter(pad, mask)` on a polygon-masked crop):
  correct baselines, but 4.2 to 4.8 s per call on an 815×950 crop and no better on fragmentation
  at that scale (45 lines for the same paragraph). Profile: the network is ~0.3 s; `vec_lines`
  is the rest, `skimage.filters.sato` 2.1 s and `calculate_polygonal_environment` 1.6 s, CPU
  work that does not shrink with the crop (kraken resizes to the model's `(1, 3, 1800, 0)`
  input). `SegmentationTaskModel.predict` calls `prepare_for_inference` per call, but that is
  free (0.00 s). Eight crops in a `ThreadPoolExecutor(8)`: 28 s against 40 s sequential, so the
  GIL is mostly held. 157 regions would be 10 minutes per page. Kept as an option, not viable as
  a default.
- PP-OCRv6 detection per region (`--eynollah-lines paddle`, `PaddleTextDetector(pad=20, mask=True)`):
  41 ms per region, 8.5 s per page overall against 8.0 s with eynollah's lines; 25 whole-line
  boxes for the paragraph above, rows joined in the Markdown by the pipeline's row grouping
  (`keep_line_order=False`, as in the paddle pipeline). Boxes per page: 95 / 91 / 369 / 264 /
  424 / 528 against eynollah's 100 / 92 / 375 / 302 / 410 / 546 lines. Now the default. Note
  that PaddleX gives PP-OCRv6 det `limit_type=min, 736`: big pages are never downscaled, so
  per-region detection gains nothing in resolution over page-level detection; the gains are the
  polygon mask (no leaks from the next column) and the exact box-to-region assignment.
- The pipeline therefore is layout → line detection → recognition in all three cases: PP-DocLayoutV3
  → PP-OCRv6 det (page level) → kraken; blla (both) → kraken; eynollah → PP-OCRv6 det (per region)
  → kraken.
- Sections (`--sections`, `DocumentBuilder(sections=True)`, default on for eynollah): heading +
  following items up to the next heading = a Docling `GroupLabel.SECTION` group named after the
  heading. Markdown/text identical (checked on the six pages), JSON carries the groups,
  DocLang only with `export_to_doclang(add_named_groups=True)` (`<group label="section" name=...>`;
  the default drops groups silently). On 00675290: 5 sections (three one-item masthead
  sections, then 33 items under the lead headline, 3 under the next); on the Yiddish page
  00761908: 31 sections of mostly one item because eynollah's `-fl` calls 31 of 92 regions
  headings. Article separation would need its own model; this is the reading order folded at
  headings and is documented as such.

## Retyper: the type-size ladder, headline splits, drop capitals (2026-09-10)

`retyper.py`, `--typography` (default on), README "Retyper". Motivation: eynollah's `-fl` labels
every kicker, sub-head and bold advertisement line a heading (a third of the regions on the
Yiddish and Polish pages), and without `-fl` the pages are flat. Row heights are on the page, so
the level is decided by a ladder of ratios to the page's median body row height (2.5 / 1.7 /
below; below 1.2 demoted to text), a headline merged into a body region is split off when its
first rows are 1.6 × taller than the rest, and a drop-capital region is glued onto its paragraph.

First run, six newspaper pages with `-fl -romb`: 101 headings levelled, 13 demoted, 10 split
off, 2 drop capitals merged. The Polish page went from 45 flat `##` to 16 `##` / 19 `###` /
9 `####` with the article headlines at `##`; the Estonian page keeps its lead headline at `##`,
"Moodne auto." at `###`, and its running head ("Nr. 2. Lhk. 8 · Tallinna Post · 20. oktoober
1929", three small regions) is demoted to text; the 1691 Utrecht page's drop capital gives
"PArijs" (the "A" is the recogniser's, the old print sets the letters after an initial as
capitals). Note that the body size is measured on the detector's boxes (PP-OCRv6 with unclip
2.0: 78 px on a page whose type is about 58 px), which is fine because everything is a ratio.
Splits in advertisement regions ("Risti-", "Tüdrukut") were real display lines but not headlines,
and a large-type product list became `##`. Rather than an advertisement rule, the general fix
(same day): every text region is classified **prose** (rows of one type size within 35 % spread,
one start edge within 0.6 em, one leading within 35 %, all rows but the last at least 80 % of the
widest, at least two rows) or **display** (everything else). Display rows stay lines in the
document, are never split, joined or continued; the headline split requires the rows below the
headline to be prose. Six pages: 138 prose and 240 display regions, splits down from 10 to 2,
reflow paragraphs 250 (from 529, the display rows are no longer paragraphs), continuations 4
(from 28). The Batavia advertisement page now reads as advertisements, line by line, with the
firm names as headings; the Estonian feuilleton is unchanged. MAD was tried for the height spread
and is useless on three-row blocks (one tall row has MAD 0); the range over the median is used.

## Reflow (2026-09-10)

`--text reflow` / `DocumentBuilder(text="reflow")`, module `reflow.py`; README "Reflow" has the
rules. Design background: a Claude Desktop research report on dehyphenation and line joining
(dehyphen/pd3f: Flair char-LM perplexity over three candidates per hyphen, German-centric,
GPL, unmaintained; pyphen: Hunspell hyphenation points, ~40 languages, no Hebrew/Arabic/CJK;
Impresso and Living-with-Machines: project-internal rules for two or three languages; OCR-D,
kraken, Tesseract, hocr-tools: nothing; ALTO carries `SUBS_CONTENT`, PAGE nothing). Nobody
packaged a language-free version, so it is written from a typology of scripts: bicameral
spaced hyphenating (Latin, Cyrillic, Greek, Armenian: the full problem, casing cue available),
unicameral spaced hyphenating (Yiddish in Hebrew script, Georgian, Indic: no casing cue),
spaced non-dividing (Arabic script, modern Hebrew: a line-end hyphen is the author's), spaced
dividing without a mark (Korean: unsupported), unspaced (Han, kana, Thai, ...: no separator,
only paragraphs). The "document as its own dictionary" (`Lexicon`, word counts over the run)
is the only language-free evidence for the compound case.

Facts from the first runs (six newspaper pages, 17 Fraktur scans):

- kraken's PP-OCRv6 model writes `¬` (U+00AC) for a line-end hyphen on Antiqua pages and `⸗`
  on Fraktur; both are in `DIVISION_MARKS`. 265 marks removed and 8 kept on the six pages, 66
  removed and 0 kept on the scans; the kept ones are capital continuations (`tütarlast- |
  Kaupa`, `Paris- | Soir`).
- The blank-line rule must be relative to the rows' own heights, not the region's median
  leading: a two-row centred subtitle in larger type was split by a gap rule based on the
  body leading.
- A single-row region is its own margin, so "the last row reaches the end margin" is no
  evidence there; without that guard every one-line advertisement continued into the next
  (103 continuations instead of 32 on the six pages, most of them wrong).
- Continuation after a division mark into a region starting with a capital is refused
  (`pod- | Wéród` was a wrong pairing; `Paris- | Soir` is lost with it).
- Justified columns: 60 % of rows within 0.6 em of the end margin; ragged text never uses the
  short-line rule.

Not measured yet: hyphen-decision accuracy against a reference (the Fraktur book's line
transcriptions can give it for the pairs with a mark), paragraph boundaries (no ground truth
here; ALTO corpora with `SUBS_CONTENT` and PAGE corpora with paragraph structure are the
candidates). Planned: once the rules settle, move `reflow.py` with readers for PAGE, ALTO and
hOCR and a Docling writer into a library and CLI of its own.

## Retyper: the type-size ladder, headline splits, drop capitals (2026-09-10)

`retyper.py`, `--typography` (default on), README "Retyper". Motivation: eynollah's `-fl` labels
every kicker, sub-head and bold advertisement line a heading (a third of the regions on the
Yiddish and Polish pages), and without `-fl` the pages are flat. Row heights are on the page, so
the level is decided by a ladder of ratios to the page's median body row height (2.5 / 1.7 /
below; below 1.2 demoted to text), a headline merged into a body region is split off when its
first rows are 1.6 × taller than the rest, and a drop-capital region is glued onto its paragraph.

First run, six newspaper pages with `-fl -romb`: 101 headings levelled, 13 demoted, 10 split
off, 2 drop capitals merged. The Polish page went from 45 flat `##` to 16 `##` / 19 `###` /
9 `####` with the article headlines at `##`; the Estonian page keeps its lead headline at `##`,
"Moodne auto." at `###`, and its running head ("Nr. 2. Lhk. 8 · Tallinna Post · 20. oktoober
1929", three small regions) is demoted to text; the 1691 Utrecht page's drop capital gives
"PArijs" (the "A" is the recogniser's, the old print sets the letters after an initial as
capitals). Note that the body size is measured on the detector's boxes (PP-OCRv6 with unclip
2.0: 78 px on a page whose type is about 58 px), which is fine because everything is a ratio.
Splits in advertisement regions ("Risti-", "Tüdrukut") were real display lines but not headlines,
and a large-type product list became `##`. Rather than an advertisement rule, the general fix
(same day): every text region is classified **prose** (rows of one type size within 35 % spread,
one start edge within 0.6 em, one leading within 35 %, all rows but the last at least 80 % of the
widest, at least two rows) or **display** (everything else). Display rows stay lines in the
document, are never split, joined or continued; the headline split requires the rows below the
headline to be prose. Six pages: 138 prose and 240 display regions, splits down from 10 to 2,
reflow paragraphs 250 (from 529, the display rows are no longer paragraphs), continuations 4
(from 28). The Batavia advertisement page now reads as advertisements, line by line, with the
firm names as headings; the Estonian feuilleton is unchanged. MAD was tried for the height spread
and is useless on three-row blocks (one tall row has MAD 0); the range over the median is used.

## Reflow (2026-09-10)

`--text reflow` / `DocumentBuilder(text="reflow")`, module `reflow.py`; README "Reflow" has the
rules. Background: a Claude Desktop research report on dehyphenation and line joining (dehyphen /
pd3f: Flair char-LM perplexity over three candidates per hyphen, two words of context, German-
centric, GPL, unmaintained; pyphen: Hunspell hyphenation points for ~40 languages, none for
Hebrew, Arabic or CJK; Impresso and Living-with-Machines: project-internal rules for two or
three languages; OCR-D, kraken, Tesseract, hocr-tools: nothing; ALTO carries `SUBS_CONTENT`,
PAGE nothing; Paragraph2Graph is a line-to-block layout GNN without released weights). Nobody
packaged a language-free version, so it is written from a typology of scripts: bicameral spaced
hyphenating (Latin, Cyrillic, Greek, Armenian: the full problem, casing cue available),
unicameral spaced hyphenating (Yiddish in Hebrew script, Georgian, Indic: no casing cue), spaced
non-dividing (Arabic script, modern Hebrew), spaced dividing without a mark (Korean: unsupported),
unspaced (Han, kana, Thai: no separator, only paragraphs). "Text flow" in the literature means
reading order, so the module is called reflow.

Unicode does most of the classification (`uniseg` 0.10.1, UAX #14 and #29, conformance-tested):

- Marks: Line_Break HY, or BA/HH/GL with HYPHEN in the character name. uniseg's tables are
  Unicode 15 (`‐ ⸗ ⹀ ֊ ־` are BA there); Unicode 16 moved them to the new class HH "unambiguous
  hyphen" (checked with the `regex` module's tables). The name test keeps the maqaf out in both.
  `¬` (class AL) is the one convention mark: kraken's PP-OCRv6 model writes it for a line-end
  hyphen on Antiqua pages and `⸗` on Fraktur.
- Separator: a break opportunity at the junction of the boundary words = no space. Thai/Lao/
  Khmer/Myanmar (SA) get no space by an explicit rule: UAX #14 leaves breaks inside SA runs to
  dictionaries and reports none.
- Sentence end: Sentence_Break STerm/ATerm after stripping Close.
- LB21a (no break after a Hebrew letter + hyphen) is Unicode's statement that Hebrew does not
  divide words; the Yiddish page divides with `⸗`, so after HL a mark counts only before HL.
- The lexicon must count line by line: counted over the region text as a whole, "Bin-\ndung"
  attested the compound "Bin-dung" and kept its own hyphen.

Facts from the first runs (six newspaper pages, 17 Fraktur scans): 265 marks removed and 5
kept on the six pages, 66 removed and 0 kept on the scans; the kept ones are capital
continuations (`Laulatuse- | SormUSEid`, `Paris- | Soir`, Polish `przestrzen- | Niemal`, all
across regions). The blank-line rule must be relative to the rows' own heights (a two-row
centred subtitle in larger type was split by a rule on the body leading). A single-row region
is its own margin, so "the last row reaches the end margin" is no evidence there: without that
guard every one-line advertisement continued into the next (103 continuations instead of 29).
Continuation after a mark into a region starting with a capital is refused (`pod- | Wéród` was a
wrong pairing; `Paris- | Soir` is lost with it). Justified: 60 % of rows within 0.6 em of the
end margin; ragged text never uses the short-line rule.

Reflow became the default for the five document formats on 2026-09-10, by decision, before any
measurement: the failure modes are mild and visible, `--text lines` switches back, and the
line-level formats are untouched either way.

Not measured yet: hyphen decisions against a reference (the Fraktur book's line transcriptions
can give it for pairs with a mark), paragraph boundaries (no ground truth here; ALTO corpora with
`SUBS_CONTENT` and PAGE corpora with paragraph structure are the candidates). Planned: once the
rules settle, move `reflow.py` (and the page-level `typography.py` that is to follow: heading
levels from type size, separators as article boundaries, drop capitals, advertisements) with
readers for PAGE, ALTO and hOCR and a Docling writer into a library and CLI of their own, named
`retyper` (decided 2026-09-10; `retype` and `reflow` are taken on PyPI, `retyper`, `resetter`
and `retypeset` were free). Archaic marks beyond Unicode's hyphens can be added to
`CONVENTION_MARKS` as they turn up.

## Not done / deferred

- **Tables in the line-level exports (postponed 2026-09-07).** Today a table region is one flat
  block of cell lines in hOCR, ALTO and PAGE (kraken's templates know one region kind), although
  SLANet gives the full cell grid. Specs checked 2026-09-07: PAGE (2019-07-15, unchanged in the
  newest 2024-07-15) has `TableRegion` + `rows`/`columns` and per-cell `TextRegion` with
  `Roles/TableCellRole` (`rowIndex`, `columnIndex`, `rowSpan`, `colSpan`, `header`); ALTO 4.4
  (March 2023) has no table element, the convention is `ComposedBlock TYPE="table"` with one
  `TextBlock` per cell (row/column only via nesting or IDs); hOCR 1.2 has `ocr_table` (bbox only)
  and says to use an HTML `table`/`tr`/`td`/`th` inside. Tesseract detects tables internally
  (`PT_TABLE`) but its hOCR/ALTO/PAGE writers emit plain lines; ocrmypdf's parser reads only
  `ocr_page > ocr_par > ocr_line|ocr_header|ocr_footer|ocr_caption|ocr_textfloat > ocrx_word`.
  Plan when picked up: cells become their own kraken `Region`s tagged with table id, row, column,
  spans and header (kraken passes `tags` through untouched), cell records point at the cell id,
  and templates of ours (every detail level) render PAGE `TableRegion`/`TableCellRole`, ALTO
  `ComposedBlock` of cell `TextBlock`s, hOCR `ocr_table` with an HTML table. Do PAGE, ALTO, hOCR
  in that order. Alongside the hOCR templates: wrap block lines in `ocr_par` (kraken's template
  has none, so its hOCR is invisible to ocrmypdf); a documented divergence from kraken.
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
