"""Building a model source folder (the Hub repo layout) and uploading it."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from . import __version__, model_dir_name
from .models import DEFAULT_SOURCE, KRAKEN_REPOS, MODELS_SUBDIR, SIZES, convert_model, list_models

Log = Callable[[str], None]

CARD_TEMPLATE = """---
license: apache-2.0
library_name: squiddleocr
pipeline_tag: image-to-text
language:
- multilingual
tags:
- ocr
- text-recognition
- historical-documents
- handwritten-text-recognition
- kraken
- pp-ocrv6
- onnx
- paddleocr
base_model:
{base_models}
---

# SquiddleOCR recognisers

ONNX conversions of [kraken](https://kraken.re)'s **PP-OCRv6** text-line recognisers by
Benjamin Kiessling (ALMAnaCH, Inria Paris), packaged by
[SquiddleOCR](https://github.com/storytracer/squiddleocr) {version}. The models read printed and
handwritten text lines in 44 languages and 10 scripts (Arabic, Armenian, Cyrillic, Ethiopic,
Georgian, Greek, Hebrew, Latin, Malayalam, Syriac), historical and modern, and transcribe them
diplomatically: long s, ligatures, combining diacritics and historical orthography are kept.

Nothing is retrained. Each model directory contains the weights as `inference.onnx`, the
PaddleOCR-style `inference.yml` and `dict.txt`, the **original kraken model card**
(`MODEL_CARD.md`, with training data credits), a `NOTICE` with author, licence and Zenodo DOI,
the Apache-2.0 `LICENSE`, and `squiddle.json` recording the source file hash, tool versions and the
ONNX-vs-PyTorch parity check of the conversion.

| model | parameters | size | kraken test CER (macro) | directory | source |
|---|---|---|---|---|---|
{rows}

CER figures are kraken's own evaluation on its held-out test set, from the model cards.

## Use with SquiddleOCR

```
uv tool install "squiddleocr[paddle] @ git+https://github.com/storytracer/squiddleocr"
squiddle ocr scans/                    # medium recogniser from this repo, layout analysis, tables; Markdown
squiddle ocr scans/ -m small           # a smaller size
squiddle ocr scans/ --models {repo}    # this repo explicitly (it is the default)
```

SquiddleOCR combines these recognisers with layout, text detection and table models and writes
DoclingDocuments (DocLang, Markdown, HTML, JSON). Sizes are downloaded into
`~/.cache/squiddleocr/` on first use.

## Use with PaddleOCR / PP-StructureV3

Each directory is a drop-in text recognition model for PaddleOCR 3.x. Download it (for example
with `squiddle models pull medium`, or `huggingface_hub.snapshot_download("{repo}", allow_patterns=["models/squiddle_PP-OCRv6_medium_rec/*"])`)
and pass it with its registered name and the ONNX Runtime engine:

```python
from paddleocr import PPStructureV3
pipe = PPStructureV3(text_recognition_model_dir="models/squiddle_PP-OCRv6_medium_rec",
                     text_recognition_model_name="PP-OCRv6_medium_rec", engine="onnxruntime")
```

## What the conversion does

kraken feeds its network 96 px lines scaled to 0..1 and inverted, with 16 px of white padding at
both ends, and batches lines with white padding plus an attention mask. The ONNX graph takes
PaddleOCR's `[-1, 1]` input and folds all of that in (inversion, padding, detection and masking
of batch padding, blank forcing on padded time steps), emitting `(batch, time, classes)` softmax
probabilities with the CTC blank at index 0. Verified line for line against kraken on real line
strips (see the SquiddleOCR repository, `squiddle verify`).

## Input contract

RGB line images of any width, resized to 96 px height, normalised `(x/255 - 0.5) / 0.5`; batches
right-padded with zeros. Output: `(batch, time, 1623)` probabilities, `dict.txt` entry *i* is
class *i + 1*, class 0 is the blank; time steps correspond to 8 input pixels each.

## Licence and citation

The weights are Apache-2.0, © Benjamin Kiessling, trained with support of the ATRIUM and MiDRASH
projects; see `MODEL_CARD.md` in each directory for the training data and their licences. Please
cite kraken and the Zenodo DOI of the model you use. The `squiddle_` naming identifies the
packager and is not the author's.
"""

ZENODO = {"tiny": "10.5281/zenodo.21788403", "small": "10.5281/zenodo.21788405", "medium": "10.5281/zenodo.21788410"}
#: parameters / ONNX size / kraken test CER (from the model cards), per size
FACTS = {"tiny": ("0.7M", "3 MB", "8.7 % (11.0 %)"), "small": ("3.2M", "14 MB", "5.4 % (6.9 %)"),
         "medium": ("15.8M", "64 MB", "3.9 % (4.9 %)")}


def build_source(out_dir: str | Path, sizes: Sequence[str] = SIZES, repo: str = DEFAULT_SOURCE,
                 log: Log = print) -> list[Path]:
    """Convert ``sizes`` into ``out_dir/models/...`` (skipping ones already there) and write the model card."""
    out = Path(out_dir)
    done = []
    present = dict(list_models(out))
    for size in sizes:
        target = out / MODELS_SUBDIR / model_dir_name(size)
        if size in present:
            log(f"{target} already converted")
        else:
            convert_model(size, target, log)
        done.append(target)
    write_card(out, repo)
    return done


def write_card(out_dir: str | Path, repo: str = DEFAULT_SOURCE) -> Path:
    out = Path(out_dir)
    sizes = [s for s, _ in list_models(out)]
    rows = "\n".join(
        f"| PP-OCRv6 {s} | {FACTS[s][0]} | {FACTS[s][1]} | {FACTS[s][2]} | `{MODELS_SUBDIR}/{model_dir_name(s)}` "
        f"| [kraken, DOI {ZENODO[s]}](https://doi.org/{ZENODO[s]}) |" for s in sizes)
    base = "\n".join(f"- {KRAKEN_REPOS[s]}" for s in sizes)
    card = out / "README.md"
    card.write_text(CARD_TEMPLATE.format(base_models=base, rows=rows, repo=repo, version=__version__), encoding="utf-8")
    return card


def upload_command(folder: str | Path, repo: str = DEFAULT_SOURCE) -> str:
    return f"squiddle upload {folder} --repo {repo}"


def upload(folder: str | Path, repo: str = DEFAULT_SOURCE, private: bool = False, log: Log = print) -> str:
    """Create ``repo`` if needed and upload the folder; returns the repo URL."""
    from huggingface_hub import HfApi

    folder = Path(folder)
    if not list_models(folder):
        raise FileNotFoundError(f"{folder} holds no converted models (run `squiddle convert -o {folder}` first).")
    if not (folder / "README.md").is_file():
        write_card(folder, repo)
    api = HfApi()
    url = api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    log(f"Uploading {folder} -> {url}")
    api.upload_folder(repo_id=repo, folder_path=str(folder), repo_type="model",
                      commit_message=f"SquiddleOCR {__version__}: converted models")
    return str(url)
