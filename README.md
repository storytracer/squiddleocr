# SquiddleOCR

### Fast OCR for historical print, built from small models

Point SquiddleOCR at a folder of scanned book pages. You get the text in reading order, Markdown with
headings, tables and formulas, and the [ALTO], [PAGE-XML] and [hOCR] files libraries use.

**Jump to:** [Why](#-why-squiddleocr) · [Quick start](#-quick-start) · [Options](#%EF%B8%8F-options) ·
[Faster](#-faster-with-a-model-server) · [How it works](#-how-it-works) · [Status](#%EF%B8%8F-status) ·
[Credit](#-licence-and-credit)

## 🤔 Why SquiddleOCR

- 📜 **[kraken] reads old print very well.** Its [PP-OCRv6 line recogniser][kraken-medium] is small enough for a modest
  GPU or even a laptop.
- 🐢 **But finding the lines is slow.** kraken's own page segmentation takes most of the time per page,
  runs on the CPU and tends to miss decorated initials and captions.
- 💸 **Large vision-language models are no way out.** They need big GPUs, can invent or repeat text, and
  give no word positions, which libraries need to highlight search hits on the scan.

SquiddleOCR keeps kraken's recogniser and replaces everything around it with small, fast models. It
gives text, tables, formulas and word positions in one run, in under a second per page on a common GPU
such as an Nvidia L4.

It is a ready-made pipeline on [Linecaster], so it runs wherever Linecaster runs: a laptop, a GPU
server, a [Ray] cluster or [Hugging Face Jobs][HF Jobs].

## 🚀 Quick start

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU (Linux, x86_64 or aarch64).
Clone Linecaster and SquiddleOCR into the same folder:

```bash
git clone https://github.com/storytracer/linecaster
git clone https://github.com/storytracer/squiddleocr
cd squiddleocr
uv sync --extra paddle
```

Read a folder of page images (JPEG, PNG, TIFF, WebP or JPEG 2000):

```bash
uv run squiddle ocr scans/
```

Each image gets a `.paddle.md` file next to it. The models download on first use.

More examples:

```bash
# Markdown, text and ALTO into out/
uv run squiddle ocr scans/ -o out/ \
  -f md,txt,alto

# one document for the whole book
uv run squiddle ocr book/ -o out/ \
  --per-document

# every option
uv run squiddle ocr --help
```

## 📦 What you get

For every page, as many of these as you ask for with `-f`:

| File | Contents |
|---|---|
| 📝 `.md` (default) | headings, paragraphs, tables, formulas as LaTeX |
| 📄 `.txt` | the same as plain text |
| 🌐 `.html`, `.doclang.xml`, `.json` | the same as HTML, DocLang and [Docling] JSON |
| 🏛️ `.hocr`, `.alto.xml`, `.page.xml` | positions of regions, lines and words |
| 🗂️ `.transcription.json` | Linecaster's full page record |
| 🖼️ `.layout.jpg` | regions, reading order and lines drawn on the page (`--plots`) |

- Files are named `<page>.paddle.<ext>`, next to each image or in the folder given with `-o`.
- With `--per-document`, the text formats hold the whole book in one file, with paragraphs continued
  across page breaks.
- With `-o`, a run also writes one record per page to `rows/`, so a later run can start from there.

## ⚙️ Options

<details>
<summary><b>Output</b></summary>

- `-o`, `--output`: output folder; default: next to each image
- `-f`, `--formats`: `md` (default), `txt`, `html`, `doclang`, `json`, `hocr`, `alto`, `page`, `transcription`
- `--per-document`: one document for all images, such as a book
- `--text`: `reflow` (default) joins lines into paragraphs; `lines` keeps them
- `--detail`: `word` (default); `line` leaves words out of hOCR, ALTO and PAGE-XML
- `--suffix`: the name part before the extension; default: `paddle`
- `--plots`: a picture of the layout per page

</details>

<details>
<summary><b>Document</b></summary>

- `--typography`: heading levels from type size, drop capitals, headlines (on)
- `--tables`: rows and columns of tables (on)
- `--formulas`: formulas as LaTeX (on)
- `--furniture`: keep page numbers and running heads (off)
- `--sections`: group the text under its headings (off)
- `--dividers`: `---` between blocks (off)

Each on/off option also has a `--no-…` form.

</details>

<details>
<summary><b>Models</b></summary>

- `-m`, `--model`: kraken recogniser: [`tiny`][kraken-tiny], [`small`][kraken-small],
  [`medium`][kraken-medium] (default), or a model file
- `--det-model`: line detector: [`PP-OCRv6_medium_det`][PP-OCRv6_medium_det] (default),
  [`PP-OCRv6_small_det`][PP-OCRv6_small_det], [`PP-OCRv6_tiny_det`][PP-OCRv6_tiny_det]
- `--unclip-ratio`: how far line boxes grow (1.5)
- `--layout`: `paddle` (default); `none` treats the page as one text block
- `--layout-model`: [`PP-DocLayoutV3`][PP-DocLayoutV3] (default) or
  [`PP-DocLayout_plus-L`][PP-DocLayout_plus-L]
- `--formula-model`: [`PP-FormulaNet_plus-L`][PP-FormulaNet_plus-L] (default) or
  [`PP-FormulaNet-L`][PP-FormulaNet-L]
- `--rtl`: right-to-left scripts such as Hebrew and Arabic: one text block, no tables or formulas

</details>

<details>
<summary><b>Hardware and speed</b></summary>

- `--device`: `auto` (default), `cpu`, `cuda`, `tensorrt`, `coreml`
- `--precision`: kraken's arithmetic: `auto` (default), `32-true`, `bf16-mixed`, `16-mixed`, `bf16-true`
- `--batch-size`: lines read at once (8)
- `--tasks`, `--workers`: split the images into tasks and run several at once (1, 1)
- `--backend`: `in-process` (default) or `served`
- `--endpoint`: the model server; default `http://127.0.0.1:8100/v1`
- `--window`: pages in flight per step with a server (32)

</details>

## ⚡ Faster, with a model server

A model server keeps the models loaded on the GPU, and the steps work on different pages at the same
time. `squiddle models-config` writes SquiddleOCR's models as Linecaster presets; Linecaster's chain
recipe serves them (uv installs the engines on the first run):

```bash
uv run squiddle models-config > models.json
cd ../linecaster
uv run recipes/chain.py serve \
  --config ../squiddleocr/models.json
```

Then, back in `squiddleocr`:

```bash
uv run squiddle ocr scans/ -o out/ \
  --backend served
```

> [!TIP]
> The server decides which models run. `squiddle models-config` takes the same model options as
> `squiddle ocr`, so other models or settings go into the configuration the same way.

<details>
<summary><b>🌍 Whole collections, on many GPUs</b></summary>

Linecaster's chain pipeline runs the same combination over many machines with SquiddleOCR's recipe:
`uv run pipelines/chain.py IMAGES OUT --recipe squiddle`,
plus

- 🧮 Ray cluster: `--on ray --tasks 64`
- ☁️ Hugging Face Jobs: `--on hf-jobs --commit <sha>`

Each task gets one GPU with its own model server. [Linecaster's README](https://github.com/storytracer/linecaster#-run-it-anywhere) explains both.

</details>

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

SquiddleOCR's combination is the Linecaster recipe `squiddle@1`
in Linecaster's catalogue ([`squiddle`](https://github.com/storytracer/linecaster/blob/main/docs/catalog/recipes/squiddle.md)): the five presets, each at an
exact edition, and reflowed text. `squiddle ocr` at its defaults runs exactly that recipe, and every file it
writes names it; other model options make a variation, which the files then don't call the recipe.

<details>
<summary><b>Where each part comes from</b></summary>

- Recognition is kraken's own code and models, from the fork [storytracer/kraken][kraken fork], which makes kraken's
  precision setting take effect.
- Layout, lines, tables and formulas are [PaddleX] models, through `linecaster-paddlex`.
- The data model, file formats, typography and reflow rules and the ways of running at scale come from
  Linecaster.
- SquiddleOCR adds the combination and the `squiddle` command. The chain is in
  `src/squiddleocr/chain.py`.

</details>

<details>
<summary><b>🎯 Checked against earlier outputs</b></summary>

- The chain reproduces SquiddleOCR 0.3 line for line: the same text, line boxes and word boxes, inside
  the pipeline or through a server, on x86 and ARM.
- `SQUIDDLE_GOLDENS=1 uv run pytest tests/test_chain.py -k goldens` runs this check. It needs the golden
  files recorded with 0.3.
- Each document option was compared with `squiddle legacy ocr` on the same pages;
  `SQUIDDLE_PARITY=1` runs a short version. `NOTES.md` explains the differences that remain.

</details>

<details>
<summary><b>🧭 Design principles</b></summary>

- **Reuse, don't reimplement.** [kraken], [PaddleX] and [Docling] do the work; SquiddleOCR connects them.
- **Small models first.** Every model runs on a modest GPU, so a whole collection can be read cheaply.
- **The same output every time.** A page's result depends only on the page. Every change is checked
  line by line against earlier outputs.
- **Built on Linecaster.** Anything useful beyond this chain goes into Linecaster.

</details>

## 📊 How well it works

On the [BHL IMPACT ground truth][bhl-impact-gt] (2,165 pages from six natural-history books in five languages):

- ✅ about **40 % fewer character errors** than the text the [Biodiversity Heritage Library][BHL] publishes today;
- 🚀 **more than ten times faster** than kraken's own pipeline on the same GPU.

`NOTES.md` here and in Linecaster has the measurements and how they were made.

## 🗺️ Status

Version 0.4.0.dev0. `squiddle ocr` has every feature of the 0.3 paddle pipeline, with two changes:

- Words are the finest level in hOCR, ALTO and PAGE-XML; 0.3 also had glyphs.
- Text is now Unicode NFC, which changes a few line-end hyphenation decisions after accented letters.

<details>
<summary><b>🚧 Open work</b></summary>

- [eynollah] and kraken's blla segmentation as Linecaster engines;
- removing the 0.3 code once `squiddle legacy` is no longer needed;
- checking right-to-left pages and CPU-only runs with the new command.

`NOTES.md` records decisions and measurements; `CLAUDE.md` has working notes for the code.

</details>

<details>
<summary><b>🧓 The older pipelines</b></summary>

Two layout engines are only in the 0.3 command, `squiddle legacy ocr`: kraken's own segmentation
(blla) and [eynollah].

```bash
# kraken's own pipeline
uv run squiddle legacy ocr scans/ \
  --pipeline kraken -f hocr

# eynollah's regions and reading order
uv run squiddle legacy ocr scans/ \
  --layout eynollah -f md,page
```

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
[kraken]: https://github.com/mittagessen/kraken
[kraken fork]: https://github.com/storytracer/kraken
[PaddleX]: https://github.com/PaddlePaddle/PaddleX
[PaddlePaddle]: https://huggingface.co/PaddlePaddle
[Docling]: https://github.com/docling-project/docling
[eynollah]: https://github.com/qurator-spk/eynollah
[Ray]: https://www.ray.io/
[HF Jobs]: https://huggingface.co/docs/huggingface_hub/guides/jobs
[ALTO]: https://www.loc.gov/standards/alto/
[PAGE-XML]: https://github.com/PRImA-Research-Lab/PAGE-XML
[hOCR]: https://kba.github.io/hocr-spec/1.2/
[BHL]: https://www.biodiversitylibrary.org/
[bhl-impact-gt]: https://huggingface.co/datasets/finebooks/bhl-impact-gt
[kraken-tiny]: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-tiny
[kraken-small]: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-small
[kraken-medium]: https://huggingface.co/small-models-for-glam/kraken-ppocrv6-medium
[PP-DocLayoutV3]: https://huggingface.co/PaddlePaddle/PP-DocLayoutV3
[PP-DocLayout_plus-L]: https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L
[PP-OCRv6_tiny_det]: https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det
[PP-OCRv6_small_det]: https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det
[PP-OCRv6_medium_det]: https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det
[SLANet_plus]: https://huggingface.co/PaddlePaddle/SLANet_plus
[PP-FormulaNet_plus-L]: https://huggingface.co/PaddlePaddle/PP-FormulaNet_plus-L
[PP-FormulaNet-L]: https://huggingface.co/PaddlePaddle/PP-FormulaNet-L
