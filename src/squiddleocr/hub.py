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
tags:
- ocr
- text-recognition
- historical-documents
- kraken
- pp-ocrv6
- onnx
base_model:
{base_models}
---

# SquiddleOCR recognisers

kraken's PP-OCRv6 text recognition models (Benjamin Kiessling, ALMAnaCH / Inria Paris, Apache-2.0)
converted to ONNX by [SquiddleOCR](https://github.com/storytracer/squiddleocr) {version}, with
kraken's line preprocessing folded into the graph. Nothing is retrained; each directory carries the
original model card (`MODEL_CARD.md`), a `NOTICE` with the Zenodo DOI, and `squiddle.json` with
the source file hash and the ONNX/PyTorch parity check.

| model | directory | source |
|---|---|---|
{rows}

## Use

```
pip install "squiddleocr[paddle]"
squiddle ocr scans/                        # uses {repo}, medium by default
squiddle ocr scans/ -m small --models {repo}
```

The directories are also drop-in text recognition models for PaddleOCR / PP-StructureV3
(`text_recognition_model_dir`, `text_recognition_model_name=PP-OCRv6_<size>_rec`,
`engine=onnxruntime`). The `squiddle_` prefix identifies the packager and is not the model author's
naming; please cite the Zenodo DOI of the model you use.
"""

ZENODO = {"tiny": "10.5281/zenodo.21788403", "small": "10.5281/zenodo.21788405", "medium": "10.5281/zenodo.21788410"}


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
    rows = "\n".join(f"| PP-OCRv6 {s} | `{MODELS_SUBDIR}/{model_dir_name(s)}` | kraken, DOI {ZENODO[s]} |" for s in sizes)
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
