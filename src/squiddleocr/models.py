"""Finding, downloading and caching recogniser model directories.

``resolve_model("medium")`` returns a ready-to-use SquiddleOCR model directory:

1. a path that already is one (contains ``inference.onnx``) is returned as is;
2. otherwise the local cache (``$SQUIDDLE_HOME/models`` or ``~/.cache/squiddleocr/models``);
3. otherwise the converted ONNX directory is downloaded from the Hugging Face Hub (``MODEL_REPOS``);
4. otherwise, if the ``convert`` extra is installed, kraken's weights are downloaded from their
   Hugging Face mirror and converted locally.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

from . import PADDLE_MODEL_NAMES, model_dir_name

logger = logging.getLogger(__name__)

SIZES = tuple(PADDLE_MODEL_NAMES)                                   # tiny, small, medium
DEFAULT_SIZE = "medium"
#: Converted (ONNX) model directories on the Hugging Face Hub; overridable per size via
#: ``SQUIDDLE_MODEL_REPO_<SIZE>`` or all at once via ``SQUIDDLE_MODEL_REPO`` ("{size}" placeholder).
MODEL_REPOS = {size: f"storytracer/squiddle_PP-OCRv6_{size}_rec" for size in SIZES}
#: kraken's original weights (byte-identical Zenodo mirrors) for local conversion.
KRAKEN_REPOS = {size: f"small-models-for-glam/kraken-ppocrv6-{size}" for size in SIZES}
REQUIRED_FILES = ("inference.onnx", "inference.yml")


def cache_dir() -> Path:
    home = os.environ.get("SQUIDDLE_HOME")
    base = Path(home) if home else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "squiddleocr"
    return base / "models"


def is_model_dir(path: Path) -> bool:
    return all((path / f).is_file() for f in REQUIRED_FILES)


def model_repo(size: str) -> str:
    env = os.environ.get(f"SQUIDDLE_MODEL_REPO_{size.upper()}") or os.environ.get("SQUIDDLE_MODEL_REPO")
    return env.format(size=size) if env else MODEL_REPOS[size]


def list_cached() -> list[Path]:
    root = cache_dir()
    return sorted(p for p in root.iterdir() if p.is_dir() and is_model_dir(p)) if root.is_dir() else []


def resolve_model(spec: str | Path = DEFAULT_SIZE, log: Callable[[str], None] = logger.info) -> Path:
    """Return a model directory for ``spec`` (a size name or a path), downloading or converting if needed."""
    path = Path(spec)
    if path.is_dir():
        if is_model_dir(path):
            return path
        raise FileNotFoundError(f"{path} is not a SquiddleOCR model directory (needs {', '.join(REQUIRED_FILES)}).")
    size = str(spec).lower()
    if size not in SIZES:
        raise ValueError(f"Unknown model {spec!r}: give a model directory or one of {', '.join(SIZES)}.")
    target = cache_dir() / model_dir_name(size)
    if is_model_dir(target):
        return target
    try:
        return download_model(size, target, log)
    except Exception as e:  # noqa: BLE001 - any hub failure falls through to local conversion
        log(f"No converted model on the Hub ({e.__class__.__name__}); converting kraken's weights locally.")
    return convert_model(size, target, log)


def download_model(size: str, target: Path, log: Callable[[str], None] = logger.info) -> Path:
    from huggingface_hub import snapshot_download

    repo = model_repo(size)
    log(f"Downloading {repo} -> {target}")
    snapshot_download(repo, local_dir=str(target))
    if not is_model_dir(target):
        raise FileNotFoundError(f"{repo} does not contain {', '.join(REQUIRED_FILES)}.")
    return target


def convert_model(size: str, target: Path, log: Callable[[str], None] = logger.info) -> Path:
    try:
        from .convert.convert import convert
    except ImportError as e:
        raise RuntimeError(
            f"No converted {size} model available and the converter is not installed. "
            "Install it with `pip install \"squiddleocr[convert]\"` (needs torch and kraken), "
            "or pass a model directory with --model."
        ) from e
    from huggingface_hub import snapshot_download

    repo = KRAKEN_REPOS[size]
    log(f"Downloading kraken weights {repo}")
    src = Path(snapshot_download(repo, allow_patterns=[f"{size}.safetensors", "README.md"]))
    log(f"Converting {size}.safetensors -> {target}")
    convert(src / f"{size}.safetensors", target, log=log)
    return target
