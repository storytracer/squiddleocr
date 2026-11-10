# SquiddleOCR

### Fast OCR for historical print, built from small models

Point SquiddleOCR at a folder of scanned book pages. You get the text in reading order, Markdown with
headings, tables and formulas, and the [ALTO], [PAGE-XML] and [hOCR] files libraries use.

SquiddleOCR is the `squiddle` recipe of [Linecaster], run by one self-contained [uv] script, `squiddle.py`.

**Jump to:** [Why](#-why-squiddleocr) · [Quick start](#-quick-start) · [What you get](#-what-you-get) ·
[Options](#%EF%B8%8F-options) · [How it works](#-how-it-works) · [Checks](#-checked-against-the-goldens) ·
[Status](#%EF%B8%8F-status) · [Credit](#-licence-and-credit)

## 🤔 Why SquiddleOCR

- 📜 **[kraken] reads old print very well.** Its [PP-OCRv6 line recognizer][kraken-medium] is small enough for a modest
  GPU or even a laptop.
- 🐢 **But finding the lines is slow.** kraken's own page segmentation takes most of the time per page,
  runs on the CPU and tends to miss decorated initials and captions.
- 💸 **Large vision-language models are no way out.** They need big GPUs, can invent or repeat text, and
  give no word positions, which libraries need to highlight search hits on the scan.

SquiddleOCR keeps kraken's recognizer and replaces everything around it with small, fast models. It
gives text, tables, formulas and word positions in one run, in under a second per page on a common GPU
such as an Nvidia L4.

## 🚀 Quick start

You need [uv] and an NVIDIA GPU (Linux, x86_64 or aarch64). Clone Linecaster and SquiddleOCR into the same
folder (until the release, `squiddle.py` takes Linecaster from the checkout next to it):

```bash
git clone https://github.com/storytracer/linecaster
```

```bash
git clone https://github.com/storytracer/squiddleocr
```

```bash
cd squiddleocr
```

Read a folder of page images (JPEG, PNG, TIFF, WebP or JPEG 2000) into `out/`:

```bash
uv run squiddle.py scans/ out/
```

The first run installs the script's environment from `squiddle.py.lock`; the models download on first use, from
their pinned Hugging Face revisions. Each page gets `out/<page>.paddle.md`.

More examples:

```bash
# Markdown, text, ALTO and PAGE-XML
uv run squiddle.py scans/ out/ --formats md,txt,alto,page
```

```bash
# page numbers and running heads kept, the text grouped under its headings
uv run squiddle.py scans/ out/ --furniture --set composition.sections=true
```

```bash
# kraken's small recognizer in place of the recipe's medium one
uv run squiddle.py scans/ out/ --recognition kraken-ppocrv6-small
```

```bash
# every option
uv run squiddle.py --help
```

## 📦 What you get

For every page, one file per format you ask for with `--formats`, flat in OUT, named `<page>.paddle.<ext>`:

| File | Contents |
|---|---|
| 📝 `.md` (default) | headings, paragraphs, tables, formulas as LaTeX |
| 📄 `.txt` | the same as plain text |
| 🌐 `.html`, `.doclang.xml`, `.json` | the same as HTML, DocLang and [Docling] JSON |
| 🏛️ `.hocr`, `.alto.xml`, `.page.xml` | positions of regions, lines and words |
| 🗂️ `.transcription.json` | Linecaster's transcription of the page |
| 🖼️ `.layout.jpg` (`plot`) | regions, reading order and lines drawn on the page |

Next to them, Linecaster's run folder:

- `pages/`: one record per page (everything read from it), which later runs can start from;
- `rejects/`: the pages a step failed on, and why. They get no files; their records keep the error;
- `logs/`, `traces/`: the run's logs and traces;
- `run.yaml`: what ran: the recipe, every step with its model and settings, the software.

A second run into the same OUT resumes the first (finished tasks are skipped); use a new OUT for a new run.

## ⚙️ Options

`squiddle.py` has the command line of Linecaster's modular pipeline, with SquiddleOCR's defaults where it is left
open: `--recipe squiddle`, `--formats md`, `--naming squiddleocr` (`<page>.paddle.<ext>`, flat) and `--files OUT`.

- **Input**: `IMAGES` is a folder of images (`--glob-pattern` picks some, `--paths-file` lists them); `--pages`
  reads the page records of an earlier run.
- **Output**: `--formats` (`md`, `txt`, `html`, `doclang`, `json`, `hocr`, `alto`, `page`, `transcription`, `plot`),
  `--furniture` (keep page numbers and running heads in md, txt and html), `--files DIR` (the files elsewhere),
  `--item-texts DIR` (one text per book).
- **Models**: each step is a slot the recipe fills; give another preset to change one (`--layout`, `--lines`,
  `--recognition`, `--tables`, `--formulas`). `--set STEP.OPTION=VALUE` changes one setting
  (`--set recognition.batch_size=16`, `--set lines.unclip_ratio=1.8`). `uv run ../linecaster/launch.py` and
  [Linecaster's catalog][catalog] list the presets.
- **Running**: `--tasks`, `--workers`, `--backend in-process|served`, `--endpoint`, `--on local|ray|hf-jobs`.

What became of the options of `squiddle ocr` (0.4):

| `squiddle ocr` 0.4 | `squiddle.py` |
|---|---|
| `squiddle ocr IMAGES... -o OUT` | `uv run squiddle.py IMAGES OUT` (one folder; OUT is required) |
| `-f`, `--formats` | `--formats` (`auto` is gone; the default is `md`) |
| `--furniture` | `--furniture` |
| `--sections` | `--set composition.sections=true` |
| `-m`, `--model tiny\|small\|medium` | `--recognition kraken-ppocrv6-tiny\|small\|medium` |
| `--det-model PP-OCRv6_small_det` | `--lines pp-ocrv6-small-det` |
| `--layout-model PP-DocLayout_plus-L` | `--layout pp-doclayout-plus-l` |
| `--formula-model PP-FormulaNet-L` | `--formulas pp-formulanet-l` |
| `--layout none` | `--layout single-region` |
| `--unclip-ratio`, `--batch-size`, `--precision`, `--device` | `--set lines.unclip_ratio=…`, `--set recognition.batch_size=…`, `--set recognition.precision=…`, `--set <step>.device=…` |
| `--plots` | `plot` in `--formats` |
| `--backend`, `--endpoint`, `--window`, `--tasks`, `--workers` | the same |
| `--per-document` | gone: `--item-texts DIR` for one text per book, or Linecaster's `pipelines/bind.py` on `OUT/pages` |
| `--text lines`, `--no-tables`, `--no-formulas`, `--no-typography` | gone (the recipe fills those steps) |
| `--rtl` | `--set recognition.text_direction=horizontal-rl --set recomposition.rtl=true` (unchecked) |
| `--detail line`, `--dividers`, `--suffix` | gone |
| output next to the images | gone: files go to OUT |
| `squiddle models-config` | `uv run squiddle.py serve` (with the same model options) |
| `squiddle goldens` | Linecaster's `tests/parity/compare_goldens.py` ([below](#-checked-against-the-goldens)) |

## ⚡ Faster, with a model server

A model server keeps the models loaded on the GPU, and the steps work on different pages at the same time.
`--backend served` starts one in the run:

```bash
uv run squiddle.py scans/ out/ --backend served
```

Or keep one running, with the squiddle models, and point runs at it:

```bash
uv run squiddle.py serve --port 8100
```

```bash
uv run squiddle.py scans/ out/ --backend served --endpoint http://127.0.0.1:8100/v1
```

## 🔍 How it works

A page is read in nine steps. Each step adds its part to the page's record.

| Step | What happens |
|---|---|
| 1. 🗺️ Layout | [PP-DocLayoutV3] marks text blocks, headings, tables, pictures and page numbers |
| 2. 📏 Lines | [PP-OCRv6 medium detection][PP-OCRv6_medium_det] finds every line on the page |
| 3. 🧲 Lines into blocks | each line is assigned to the block it sits in |
| 4. 👁️ Recognition | [kraken PP-OCRv6 medium][kraken-medium] reads the lines, block by block, with word positions |
| 5. 🔤 Unicode | the text in NFC |
| 6. 📊 Tables | [SLANet_plus] finds rows and columns; cells get kraken's text |
| 7. ➗ Formulas | [PP-FormulaNet_plus-L] reads formulas as LaTeX |
| 8. 🔠 Typography | headings get their level from the type size |
| 9. 📖 Document | lines joined into paragraphs, line-end hyphens removed |

These steps are Linecaster's modular pipeline (`linecaster.pipelines.transcribe_modular`), and the models and rules
in them are the recipe `squiddle@1` in [Linecaster's catalog][squiddle recipe], each at an exact edition.
`squiddle.py` imports that pipeline and runs it with the recipe and SquiddleOCR's file names; every file and the run
record name the recipe. A slot given on the command line is recorded as an override of it.

<details>
<summary><b>Where each part comes from</b></summary>

- Recognition is kraken's own code and models, from the fork [storytracer/kraken][kraken fork], which makes kraken's
  precision setting take effect.
- Layout, lines, tables and formulas are [PaddleX] models, through `linecaster-paddlex`.
- The pipeline, the data model, the file formats, the typography and recomposition rules and the ways of running at
  scale are Linecaster's.
- SquiddleOCR is the recipe's command: `squiddle.py`, its environment (`squiddle.py.lock`) and its defaults.

</details>

<details>
<summary><b>🧭 Design principles</b></summary>

- **Reuse, don't reimplement.** [kraken], [PaddleX] and [Docling] do the work; Linecaster connects them.
- **Small models first.** Every model runs on a modest GPU, so a whole collection can be read cheaply.
- **The same output every time.** A page's result depends only on the page. Every change is checked
  line by line against earlier outputs.
- **Built on Linecaster.** Anything beyond the recipe's command goes into Linecaster.

</details>

## 🎯 Checked against the goldens

The outputs of SquiddleOCR 0.3 (the goldens, recorded per machine) are the reference: `squiddle.py` must reproduce
them line for line (line boxes, text after NFC, every word box). To check, on a machine with the goldens:

1. Read the goldens' sample:

   ```bash
   uv run squiddle.py ~/data/bhl-ocr-eval/sample-60 work/goldens/sample-60
   ```

2. Compare the page records with the goldens:

   ```bash
   uv run --no-project --with lxml --with numpy python ../linecaster/tests/parity/compare_goldens.py work/goldens/sample-60/pages ~/data/goldens/squiddleocr-7c8a052/spark/sample-60
   ```

3. The same for the table pages: `tables-7` in place of `sample-60` in both commands.

It prints the totals and exits with 1 unless every page is exact. `NOTES.md` has the latest results.

## 📊 How well it works

On the [BHL IMPACT ground truth][bhl-impact-gt] (2,165 pages from six natural-history books in five languages):

- ✅ about **40 % fewer character errors** than the text the [Biodiversity Heritage Library][BHL] publishes today;
- 🚀 **more than ten times faster** than kraken's own pipeline on the same GPU.

`NOTES.md` here and in Linecaster has the measurements and how they were made.

## 🗺️ Status

Since 2026-11-10 SquiddleOCR is one uv script on Linecaster; the Python package and its `squiddle` command (0.4.1)
are gone, and the git history keeps them. Until Linecaster's release 0.3.0, `squiddle.py` takes Linecaster from the
checkout next to it; at the release it takes the tag `v0.3.0`.

<details>
<summary><b>🚧 Open work</b></summary>

- Running `squiddle.py` on Ray and Hugging Face Jobs (`--on ray`, `--on hf-jobs`): not tried yet.
- Right-to-left pages and CPU-only runs: unchecked.

`NOTES.md` records decisions and measurements; `CLAUDE.md` has working notes.

</details>

## 📄 Licence and credit

SquiddleOCR is Apache-2.0.

The recognition models are [kraken]'s PP-OCRv6 models by Benjamin Kiessling (Apache-2.0), released on
Zenodo and fetched from pinned Hugging Face mirrors. Please cite kraken and the DOI of the model you
use.

| Model | DOI |
|---|---|
| [kraken PP-OCRv6 tiny][kraken-tiny] | [10.5281/zenodo.21788403](https://doi.org/10.5281/zenodo.21788403) |
| [kraken PP-OCRv6 small][kraken-small] | [10.5281/zenodo.21788405](https://doi.org/10.5281/zenodo.21788405) |
| [kraken PP-OCRv6 medium][kraken-medium] | [10.5281/zenodo.21788410](https://doi.org/10.5281/zenodo.21788410) |

The model names link to the Hugging Face mirrors SquiddleOCR downloads from.

The layout, line, table and formula models are Apache-2.0 from [PaddlePaddle] ([PaddleX]).

<!-- links -->
[Linecaster]: https://github.com/storytracer/linecaster
[uv]: https://docs.astral.sh/uv/
[catalog]: https://storytracer.github.io/linecaster/latest/catalog/
[squiddle recipe]: https://storytracer.github.io/linecaster/latest/catalog/recipes/squiddle/
[kraken]: https://github.com/mittagessen/kraken
[kraken fork]: https://github.com/storytracer/kraken
[PaddleX]: https://github.com/PaddlePaddle/PaddleX
[PaddlePaddle]: https://huggingface.co/PaddlePaddle
[Docling]: https://github.com/docling-project/docling
[ALTO]: https://www.loc.gov/standards/alto/
[PAGE-XML]: https://github.com/PRImA-Research-Lab/PAGE-XML
[hOCR]: https://kba.github.io/hocr-spec/1.2/
[BHL]: https://www.biodiversitylibrary.org/
[bhl-impact-gt]: https://huggingface.co/datasets/finebooks/bhl-impact-gt
[kraken-tiny]: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-tiny
[kraken-small]: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-small
[kraken-medium]: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-medium
[PP-DocLayoutV3]: https://huggingface.co/PaddlePaddle/PP-DocLayoutV3
[PP-OCRv6_medium_det]: https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det
[SLANet_plus]: https://huggingface.co/PaddlePaddle/SLANet_plus
[PP-FormulaNet_plus-L]: https://huggingface.co/PaddlePaddle/PP-FormulaNet_plus-L
