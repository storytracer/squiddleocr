# SquiddleOCR

### Fast OCR for historical print, built from small models

Point SquiddleOCR at a folder of scanned book pages. You get the text in reading order, Markdown with
headings, tables and formulas, and the [ALTO], [PAGE-XML] and [hOCR] files libraries use.

SquiddleOCR is the `squiddle` recipe of [Linecaster], run by one self-contained [uv] script, `squiddle.py`.

**Jump to:** [Why](#-why-squiddleocr) · [Quick start](#-quick-start) · [What you get](#-what-you-get) ·
[Options](#%EF%B8%8F-options) · [How it works](#-how-it-works) ·
[Credit](#-licence-and-credit)

## 🤔 Why SquiddleOCR

OCR (optical character recognition) turns a scanned page into text you can search, read aloud, quote and analyze.
Old printed books are hard for it: worn type, long s, ligatures, faded ink, uneven lines.

- 📜 **[kraken] reads old print very well.** It is an open-source OCR system made for historical documents, and its
  [PP-OCRv6 recognizer][kraken-medium] is small enough to run on an ordinary GPU.
- 🐢 **But kraken is slow at finding the lines.** Before it can read, it has to work out where each line of text is.
  That step takes most of the time per page and tends to miss decorated initials and captions.
- 💸 **The large AI models are no answer.** Vision-language models, the kind behind chat assistants, can read pages,
  but they need expensive hardware, sometimes invent or repeat text, and don't say where on the page each word is.
  Libraries need those word positions to highlight search hits on the scan.

SquiddleOCR keeps kraken's reader and replaces everything around it with small, fast models: one finds the parts of
the page (text, headings, tables, formulas, page numbers), another the lines, others read tables and formulas. In one
run you get the text, the page's structure and the position of every word, in under a second per page on a common
GPU such as an Nvidia L4.

## 🚀 Quick start

You need [uv] and an NVIDIA GPU (Linux, x86_64 or aarch64). The script runs on Python 3.12, which uv installs if
it is missing, and takes Linecaster from its release tag `v0.4.0`:

```bash
git clone https://github.com/storytracer/squiddleocr
```

```bash
cd squiddleocr
```

Read a folder of page images (JPEG, PNG, TIFF, WebP or JPEG 2000) into `out/`, with one file per page and format
in `out/files/`:

```bash
uv run squiddle.py scans/ out/ --files out/files
```

The first run installs the script's environment from `squiddle.py.lock`; the models download on first use, from
their pinned Hugging Face revisions.

Only Markdown:

```bash
uv run squiddle.py scans/ out/ --files out/files --formats md
```

```bash
# every option
uv run squiddle.py --help
```

## 📦 What you get

With `--files DIR`, one file per page and format (`--formats`, default `hocr,alto,page,txt,md,json`) in DIR:

| File | Contents |
|---|---|
| 📝 `.md` | headings, paragraphs, tables, formulas as LaTeX |
| 📄 `.txt` | the same as plain text |
| 🌐 `.html`, `.doclang.xml`, `.json` | the same as HTML, DocLang and [Docling] JSON |
| 🏛️ `.hocr`, `.alto.xml`, `.page.xml` | positions of regions, lines and words |
| 🗂️ `.transcription.json` | Linecaster's transcription of the page |
| 🖼️ `.layout.jpg` (`plot`) | regions, reading order and lines drawn on the page |

In OUT, Linecaster's run folder:

- `pages/`: one record per page (everything read from it), which later runs can start from;
- `rejects/`: the pages a step failed on, and why. They get no files; their records keep the error;
- `logs/`, `traces/`: the run's logs and traces;
- `run.yaml`: what ran: the recipe, every step with its model and settings, the software.

A second run into the same OUT resumes the first (finished tasks are skipped); use a new OUT for a new run.

## ⚙️ Options

`squiddle.py` takes the options of Linecaster's modular pipeline, with the same defaults; it only names the recipe.
`uv run squiddle.py --help` lists them all. The ones you need most:

**📥 Input**

| Option | What it does |
|---|---|
| `IMAGES` | the folder of page images |
| `--glob-pattern PATTERN` | only the matching files |
| `--paths-file FILE` | only the files listed, one per line |
| `--pages DIR` | the page records of an earlier run instead of images |

**📤 Output**

| Option | What it does |
|---|---|
| `--files DIR` | one file per page and format in DIR |
| `--formats LIST` | `md`, `txt`, `html`, `doclang`, `json`, `hocr`, `alto`, `page`, `transcription`, `plot` (default `hocr,alto,page,txt,md,json`) |
| `--naming item\|squiddleocr` | `<item>/<page>.<ext>` (default) or flat `<page>.paddle.<ext>` |
| `--furniture` | keep page numbers and running heads in md, txt and html |
| `--item-texts DIR` | one text per book |

**🧠 Models**

| Option | What it does |
|---|---|
| `--layout`, `--lines`, `--recognition`, `--tables`, `--formulas` | another preset for that step, e.g. `--recognition kraken-ppocrv6-small` |
| `--set STEP.OPTION=VALUE` | one setting, e.g. `--set recognition.batch_size=16` |

[Linecaster's catalog][catalog] lists the presets and their settings.

**🏃 Running**

| Option | What it does |
|---|---|
| `--tasks N`, `--workers N` | split the images into N tasks; run N at a time |
| `--backend served`, `--endpoint URL` | use a model server ([below](#-faster-with-a-model-server)) |
| `--on local\|ray\|hf-jobs` | where the run happens |
| `--dry-run` | print the plan, run nothing |

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
`squiddle.py` imports that pipeline and runs it with the recipe, nothing else; every file and the run record name
the recipe. A slot given on the command line is recorded as an override of it.

<details>
<summary><b>Where each part comes from</b></summary>

- Recognition is kraken's own code and models, from the fork [storytracer/kraken][kraken fork], which makes kraken's
  precision setting take effect.
- Layout, lines, tables and formulas are [PaddleX] models, through `linecaster-paddlex`.
- The pipeline, the data model, the file formats, the typography and recomposition rules and the ways of running at
  scale are Linecaster's.
- SquiddleOCR is the recipe's command: `squiddle.py` and its environment (`squiddle.py.lock`).

</details>

<details>
<summary><b>🧭 Design principles</b></summary>

- **Reuse, don't reimplement.** [kraken], [PaddleX] and [Docling] do the work; Linecaster connects them.
- **Small models first.** Every model runs on a modest GPU, so a whole collection can be read cheaply.
- **The same output every time.** A page's result depends only on the page. Every change is checked
  line by line against earlier outputs.
- **Built on Linecaster.** Anything beyond the recipe's command goes into Linecaster.

</details>

## 📊 How well it works

On the [BHL IMPACT ground truth][bhl-impact-gt] (2,165 pages from six natural-history books in five languages):

- ✅ about **40 % fewer character errors** than the text the [Biodiversity Heritage Library][BHL] publishes today;
- 🚀 **more than ten times faster** than kraken's own pipeline on the same GPU.

`NOTES.md` here and in Linecaster has the measurements and how they were made.

## 📄 Licence and credit

SquiddleOCR is licensed under the Apache License 2.0, see [LICENSE](LICENSE).

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
