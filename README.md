# SquiddleOCR

### Fast OCR for historical print, built from small models

SquiddleOCR turns scanned pages of old books into text, tables and the file formats libraries use. You
point it at a folder of page images and get, for every page, the text in reading order, a Markdown
version with tables, and ALTO, PAGE-XML and hOCR files with the position of every word.

It is a ready-made pipeline built on [Linecaster](https://github.com/storytracer/linecaster), so it runs
wherever Linecaster runs: on a laptop, on a GPU server, on a cluster or on Hugging Face Jobs.

SquiddleOCR is in development (version 0.4). The commands below work; options may still change.

## Why SquiddleOCR

**Good line recognisers exist, but the steps around them are slow or incomplete.** kraken's PP-OCRv6
recogniser reads lines of historical print very well, and it is small: it runs on modest GPUs and even on
a laptop. kraken's own way of finding the lines on a page, however, takes most of the time per page, runs
on the CPU, and tends to skip decorated initials and captions.

**Large vision-language models read well, but at a price.** They need big GPUs, cost much more per page,
and can invent or repeat text when a page confuses them. They also give no word positions, which
libraries need to highlight search hits on the scan.

SquiddleOCR keeps kraken's recogniser and replaces the steps around it:

- **Layout from PaddleX.** A fast layout model finds text blocks, headings, tables, pictures and page
  numbers.
- **Lines from PaddleX.** A fast line detector finds every line of text on the whole page.
- **Reading with kraken.** kraken's PP-OCRv6 recogniser reads each line and reports where each word sits.
- **Tables rebuilt.** Table regions get rows and columns, filled with kraken's readings.

The result is a pipeline of small models that gives the text, the tables and the word positions in one
run. With a model server, it reads a page in under a second on a common GPU such as an Nvidia L4.

## What you get

For every page image:

| File | Contents |
|---|---|
| `<page>.txt` | the page text in reading order |
| `<page>.md` | the page text, with tables as Markdown (HTML when cells span rows or columns) |
| `<page>.hocr`, `<page>.alto.xml`, `<page>.page.xml` | hOCR, ALTO and PAGE-XML, with the positions of regions, lines and words |
| `<page>.transcription.json` | Linecaster's page record: regions, lines, words and tables |

A run also writes one record per page to `rows/`, with everything the pipeline found, so a later run can
start from there.

## Design principles

- **Reuse, don't reimplement.** Recognition is kraken's own code with kraken's own models; layout, lines
  and tables are PaddleX's. SquiddleOCR only connects them.
- **Small models first.** Every model in the chain runs on a modest GPU, so a whole collection can be read
  at low cost.
- **The same output every time.** A page's result depends only on the page. Each change to the pipeline
  is checked line by line against earlier outputs before it is accepted.
- **Built on Linecaster.** The data model, the file formats, the model engines and the ways of running at
  scale all come from Linecaster. SquiddleOCR adds the combination and the `squiddle` command.

## Under the hood

### The chain

| Step | Model | From |
|---|---|---|
| 1. Layout | PP-DocLayoutV3 | `linecaster-paddlex` |
| 2. Text lines | PP-OCRv6 detection, on the whole page | `linecaster-paddlex` |
| 3. Lines into regions | rules | Linecaster |
| 4. Recognition | kraken PP-OCRv6 medium (tiny and small also available) | `linecaster-kraken` |
| 5. Table structure | PaddleX table models | `linecaster-paddlex` |

The chain is defined in `src/squiddleocr/chain.py`. kraken runs from a fork (`storytracer/kraken`) that
makes kraken's precision setting take effect at inference.

### Checked against earlier outputs

The chain reproduces SquiddleOCR 0.3 line for line: the same text, line boxes and word boxes, run inside
the pipeline or through a model server, on x86 and on ARM machines. The check runs with
`SQUIDDLE_GOLDENS=1 uv run pytest tests/test_chain.py -k goldens` and needs the golden files recorded
with 0.3.

## How it works

A page is read in stages. The layout model first marks where the text blocks, headings, tables and
pictures are. The line detector then finds every line on the page, and each line is assigned to the
block it sits in. kraken reads the lines, block by block in reading order. For table blocks, PaddleX
works out the rows and columns, and the cells are filled with the lines kraken read inside them. Finally
the text and the files are written.

Each of these steps is a Linecaster block, and each page carries one record that every step fills in
further. The models can run inside the pipeline or on a model server that keeps them loaded on the GPU;
with a server, the steps work on different pages at the same time.

## Getting started

### Install

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (Linux, x86_64 or aarch64).
SquiddleOCR installs Linecaster from a checkout next to it, so clone both into the same folder:

```bash
git clone https://github.com/storytracer/linecaster
git clone https://github.com/storytracer/squiddleocr
cd squiddleocr
uv sync --extra paddle
```

### Read a folder of page images

JPEG, PNG, TIFF, WebP and JPEG 2000 work:

```bash
uv run squiddle ocr scans/ -o out/
```

The models download on first use: kraken's from Hugging Face, PaddleX's from PaddleX. Choose the files
with `-f` (the default is `hocr,alto,page,txt`):

```bash
uv run squiddle ocr scans/ -o out/ -f txt,md,alto
```

| Option | Default | What it does |
|---|---|---|
| `-m, --model` | `medium` | kraken recogniser: `tiny`, `small` or `medium` |
| `--tasks`, `--workers` | `1`, `1` | split the images into tasks and run several at once |
| `--backend` | `in-process` | `served` sends each step to a model server instead |
| `--endpoint`, `--window` | `http://127.0.0.1:8100/v1`, `32` | server address, and pages in flight per step |

### Faster, with a model server

A model server keeps the models loaded on the GPU, and the steps work on different pages at the same
time. Start it from the Linecaster checkout, in its engine environment:

```bash
cd ../linecaster
(cd envs/kraken-paddlex && uv sync)
envs/kraken-paddlex/.venv/bin/python -m linecaster.cli serve --config examples/squiddle_models.json --port 8100
```

Then point SquiddleOCR at it:

```bash
uv run squiddle ocr scans/ -o out/ --backend served
```

### Whole collections, on many GPUs

For runs over many machines, use the same chain through Linecaster's recipes: `recipes/ray/squiddle_chain.py`
for a Ray cluster (one task per GPU, each with its own model server) and `recipes/hf-jobs/squiddle_chain.py`
for Hugging Face Jobs. Both are described in Linecaster's README.

### The older pipelines

Some features are still only in the 0.3 pipelines, under `squiddle legacy ocr`: formulas as LaTeX,
headings and other typography rules, dehyphenation and reflow into paragraphs, one document per book, and
the alternative layout engines (kraken's own segmentation, eynollah).

```bash
uv run squiddle legacy ocr scans/ -f md,txt               # Markdown with tables, formulas and reflowed text
uv run squiddle legacy ocr book/ --per-document -f md     # one document for a whole book
uv run squiddle legacy ocr scans/ --pipeline kraken -f hocr   # kraken's own pipeline
uv run squiddle legacy ocr --help                         # every option
```

Linecaster already has blocks for formulas, typography, dehyphenation, reflow and document assembly;
they come to `squiddle ocr` next.

## How well it works

On the BHL IMPACT ground truth (2,165 pages from six natural-history books in five languages), the chain
makes about 40 % fewer character errors than the text BHL publishes today. On the same kind of GPU it
reads a page more than ten times faster than kraken's own pipeline. `NOTES.md` and Linecaster's
`NOTES.md` have the measurements and how they were made.

## Status

Version 0.4.0.dev0. `squiddle ocr` is the chain on Linecaster; the 0.3 pipelines live on as
`squiddle legacy`.

Open work:

- formulas, typography, reflow and document assembly in `squiddle ocr`,
- eynollah as a Linecaster layout engine,
- removing the 0.3 code once `squiddle legacy` is no longer needed.

`NOTES.md` records decisions and measurements; `CLAUDE.md` has working notes for the code.

## Licence and credit

SquiddleOCR is Apache-2.0. The recognition models are kraken's PP-OCRv6 models by Benjamin Kiessling
(Apache-2.0), released on Zenodo and fetched from pinned Hugging Face mirrors:

| Model | DOI | Hugging Face mirror |
|---|---|---|
| PP-OCRv6 tiny | 10.5281/zenodo.21788403 | `small-models-for-glam/kraken-ppocrv6-tiny` |
| PP-OCRv6 small | 10.5281/zenodo.21788405 | `small-models-for-glam/kraken-ppocrv6-small` |
| PP-OCRv6 medium | 10.5281/zenodo.21788410 | `small-models-for-glam/kraken-ppocrv6-medium` |

Please cite kraken and the DOI of the model you use. The PaddleX models (PP-DocLayoutV3, PP-OCRv6
detection, the table models, PP-FormulaNet) are Apache-2.0 from PaddlePaddle.
