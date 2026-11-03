# SquiddleOCR

SquiddleOCR reads scanned pages of historical print and turns them into text and standard OCR
files. It finds the layout and the text lines with PaddleX models, reads each line with kraken's
PP-OCRv6 recogniser, and runs as a pipeline built from [linecaster](https://github.com/storytracer/linecaster) blocks.

## What you get

For every page image:

| File | Contents |
|---|---|
| `<page>.txt` | The page text in reading order |
| `<page>.md` | The page text, with tables as Markdown (HTML when cells span rows or columns) |
| `<page>.hocr`, `<page>.alto.xml`, `<page>.page.xml` | hOCR 1.2, ALTO 4.4 and PAGE-XML 2019, with coordinates for regions, lines and words |
| `<page>.transcription.json` | linecaster's page record: regions, lines, words and tables |

A run also writes one JSON line per page to `rows/`, with everything the pipeline found. A later
pipeline can start from these rows.

Tables are found by the layout model. PaddleX's table model then works out the cell structure,
and the cells are filled with kraken's readings of the lines inside them.

Some features are only in the older pipelines for now, under `squiddle legacy ocr`:

- formulas as LaTeX (PP-FormulaNet),
- headings and heading levels from type size, and other typography rules,
- dehyphenation and reflow of lines into paragraphs, also across pages,
- one document per book, as Markdown, HTML, DocLang or Docling JSON,
- kraken's own pipeline (blla segmentation) and the eynollah layout analyser.

linecaster already has blocks for formulas, typography, dehyphenation, reflow and document
assembly. They come to `squiddle ocr` next.

## Quick start

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU. The tested setups are
Linux on x86_64 and on aarch64. SquiddleOCR installs linecaster from a checkout next to it, so
clone both into the same folder:

```bash
git clone https://github.com/storytracer/linecaster
git clone -b linecaster-derivative https://github.com/storytracer/squiddleocr
cd squiddleocr
uv sync --extra paddle
```

Read a folder of page images (JPEG, PNG, TIFF, WebP or JPEG 2000):

```bash
uv run squiddle ocr scans/ -o out/
```

The models download on first use. kraken's come from Hugging Face, PaddleX's from PaddleX.

Choose the files with `-f`. The default is `hocr,alto,page,txt`:

```bash
uv run squiddle ocr scans/ -o out/ -f txt,md,alto
```

Other options:

| Option | Default | What it does |
|---|---|---|
| `-m, --model` | `medium` | kraken recogniser: `tiny`, `small` or `medium` |
| `--tasks`, `--workers` | `1`, `1` | split the images into datatrove tasks and run several at once |
| `--backend` | `in-process` | `served` sends each step to a model server instead |
| `--endpoint`, `--window` | `http://127.0.0.1:8100/v1`, `32` | server address, and pages in flight per step |

### With a model server

linecaster can keep the models loaded on a GPU behind a small HTTP server. Start it from the
linecaster checkout, in its engine environment:

```bash
cd ../linecaster
(cd envs/kraken-paddlex && uv sync)
envs/kraken-paddlex/.venv/bin/python -m linecaster.cli serve --config examples/squiddle_models.json --port 8100
```

Then point SquiddleOCR at it:

```bash
uv run squiddle ocr scans/ -o out/ --backend served
```

Each model works on a different page at the same time, so this is faster than in-process.

### The older pipelines

```bash
uv run squiddle legacy ocr scans/ -f md,txt               # Markdown with tables, formulas and reflowed text
uv run squiddle legacy ocr book/ --per-document -f md     # one document for a whole book
uv run squiddle legacy ocr scans/ --pipeline kraken -f hocr   # kraken's own pipeline
uv run squiddle legacy ocr --help                         # every option
```

The README on `main` documents these options in full.

## How it relates to linecaster

linecaster is a modular OCR suite. Each step (reading images, layout, line detection, recognition,
tables, writing files) is a [datatrove](https://github.com/huggingface/datatrove) block. A pipeline
is a list of blocks, so any layout model can be combined with any recogniser. The models run
in-process or behind `linecaster serve`, which the blocks reach through
[saturate](https://github.com/davanstrien/saturate).

SquiddleOCR is one such combination, the "squiddle" chain:

1. layout with PP-DocLayoutV3 (linecaster-paddlex),
2. text lines with PP-OCRv6 detection (linecaster-paddlex),
3. lines assigned to layout regions (linecaster),
4. recognition with kraken's PP-OCRv6 medium model (linecaster-kraken),
5. table structure with PaddleX (linecaster-paddlex).

The chain is defined in `src/squiddleocr/chain.py`. The data model, the file writers and the
model engines all come from linecaster. SquiddleOCR adds the `squiddle` command and the checks
against its own earlier outputs.

The chain reproduces SquiddleOCR 0.3 line for line. On a 60-page sample of the BHL IMPACT ground
truth, all 1,560 lines have the same text, line box and word boxes, in-process and through the
server, on x86_64 and on aarch64. On seven table pages every line matches. One extra line remains:
a page number that 0.3's typography step removes. Run the check with
`SQUIDDLE_GOLDENS=1 uv run pytest tests/test_chain.py -k goldens` (it needs the golden files
recorded with 0.3).

## Results

All scores come from the frozen [bhl-ocr-eval](https://github.com/finebooks/bhl-ocr-eval) harness
on all 2,165 pages of the BHL IMPACT ground truth (six natural-history books in English, French,
German, Latin and Russian). CER is the character error rate in reading order. Lower is better.

| Text | CER | Recall |
|---|---|---|
| BHL's published OCR | 0.0749 | 0.892 |
| Common Pile v0.1 BHL text | 0.0689 | 0.891 |
| SquiddleOCR 0.3 | 0.0432 | 0.976 |
| squiddle chain on linecaster, with tables | 0.0419 | 0.975 |
| kraken's own pipeline (blla segmentation), leaderboard row | 0.0338 | 0.976 |

Speed, in seconds per page:

| Setup | GPU | s/page | Source |
|---|---|---|---|
| kraken's own pipeline, one page at a time | A10G | 12.3 | bhl-ocr-eval `RESULTS.md` |
| squiddle chain, served | A10G | 0.72 | linecaster `NOTES.md` |
| squiddle chain, served | L4 | 0.76 | linecaster `NOTES.md` |
| squiddle chain, served | L40S | 0.44 | linecaster `NOTES.md` |

The leaderboard figure is wall time per page. In the served chain the models work on different
pages at the same time and recognition sets the pace, so the chain figures are the recognition
step's seconds per page. On an RTX 4000 SFF Ada workstation, 60 pages took 42 s of wall time
(0.70 s per page). Most of the kraken pipeline's time is CPU-side segmentation; the recogniser
itself is fast.

Sources: the BHL and Common Pile rows are the dataset cards of `finebooks/bhl-ocr-baseline` and
`finebooks/bhl-commonpile-baseline` on Hugging Face. The kraken row and the 12.3 s figure are in
bhl-ocr-eval's `RESULTS.md`. The SquiddleOCR and squiddle chain figures are in linecaster's
`NOTES.md` (sections "Full ground truth" and "GPUs compared").

## Status

Version 0.4.0.dev0, a development version. `squiddle ocr` is the squiddle chain on linecaster. The 0.3
pipelines (PaddleX, kraken and eynollah, with tables, formulas, typography, reflow and Docling documents)
live on as `squiddle legacy`.

It depends on a linecaster checkout next to it and on a fork of kraken (`storytracer/kraken`) that makes
the `--precision` setting take effect at inference.

Open work:

- formulas, typography, reflow and document assembly in `squiddle ocr` (the blocks exist in
  linecaster),
- eynollah as a linecaster layout engine,
- removing the 0.3 modules once `squiddle legacy` is no longer needed.

`NOTES.md` records the decisions and measurements. `CLAUDE.md` has the working notes for the code.

## Licence and credit

SquiddleOCR is Apache-2.0. The recognition models are kraken's PP-OCRv6 models by Benjamin
Kiessling (Apache-2.0), released on Zenodo and fetched from pinned Hugging Face mirrors:

| Model | DOI | Hugging Face mirror |
|---|---|---|
| PP-OCRv6 tiny | 10.5281/zenodo.21788403 | `small-models-for-glam/kraken-ppocrv6-tiny` |
| PP-OCRv6 small | 10.5281/zenodo.21788405 | `small-models-for-glam/kraken-ppocrv6-small` |
| PP-OCRv6 medium | 10.5281/zenodo.21788410 | `small-models-for-glam/kraken-ppocrv6-medium` |

Please cite kraken and the DOI of the model you use. The PaddleX models (PP-DocLayoutV3, PP-OCRv6
detection, the table models, PP-FormulaNet) are Apache-2.0 from PaddlePaddle.
