"""Model sources: a local folder or a Hugging Face model repo holding converted recognisers.

Layout of a model source (identical on disk and on the Hub, see ``squiddle convert`` / ``squiddle upload``)::

    <source>/README.md                                   model card
    <source>/models/squiddle_PP-OCRv6_tiny_rec/          inference.onnx, inference.yml, dict.txt, MODEL_CARD.md, ...
    <source>/models/squiddle_PP-OCRv6_small_rec/
    <source>/models/squiddle_PP-OCRv6_medium_rec/

``resolve_model(size, source)`` returns the model directory for ``size``:

1. ``source`` is a local folder: ``<source>/models/<name>`` (or ``source`` itself when it is a model directory);
2. ``source`` is a Hub repo id: the ``models/<name>`` subfolder is downloaded into the cache
   (``$SQUIDDLE_HOME`` or ``~/.cache/squiddleocr``) once and reused;
3. neither has the size and the ``convert`` extra is installed: kraken's weights are downloaded from
   their Hub mirror and converted into the cache.
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
DEFAULT_SOURCE = "storytracer/squiddleocr"
MODELS_SUBDIR = "models"
#: kraken's original weights (byte-identical Zenodo mirrors) for local conversion.
KRAKEN_REPOS = {size: f"small-models-for-glam/kraken-ppocrv6-{size}" for size in SIZES}
REQUIRED_FILES = ("inference.onnx", "inference.yml")
Log = Callable[[str], None]


def cache_dir() -> Path:
    home = os.environ.get("SQUIDDLE_HOME")
    return Path(home) if home else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "squiddleocr"


def default_source() -> str:
    return os.environ.get("SQUIDDLE_MODELS", DEFAULT_SOURCE)


def is_model_dir(path: Path) -> bool:
    return all((path / f).is_file() for f in REQUIRED_FILES)


def parse_size(spec: str) -> str:
    size = str(spec).lower()
    if size not in SIZES:
        raise ValueError(f"Unknown model size {spec!r}; choose one of {', '.join(SIZES)}.")
    return size


def local_model_dir(source: Path, size: str) -> Path | None:
    """The model directory for ``size`` inside a local source folder, if present."""
    for candidate in (source / MODELS_SUBDIR / model_dir_name(size), source / model_dir_name(size)):
        if is_model_dir(candidate):
            return candidate
    return None


def list_models(source: str | Path | None = None) -> list[tuple[str, Path]]:
    """``(size, path)`` of the models available locally in ``source`` (a folder) or in the cache of a repo."""
    src = Path(source) if source and Path(str(source)).is_dir() else cache_dir() / _repo_dirname(str(source or default_source()))
    return [(s, p) for s in SIZES if (p := local_model_dir(src, s)) is not None]


def resolve_model(spec: str | Path = DEFAULT_SIZE, source: str | Path | None = None, log: Log = logger.info) -> Path:
    """Model directory for ``spec`` (a size name, or a path to a model directory), from ``source``."""
    path = Path(spec)
    if path.is_dir():
        if is_model_dir(path):
            return path
        raise FileNotFoundError(f"{path} is not a model directory (needs {', '.join(REQUIRED_FILES)}).")
    size = parse_size(str(spec))
    source = str(source or default_source())

    if Path(source).is_dir():
        found = local_model_dir(Path(source), size)
        if found is None:
            raise FileNotFoundError(f"{source} has no {model_dir_name(size)} (run `squiddle convert {size} -o {source}`).")
        return found

    target = cache_dir() / _repo_dirname(source)
    found = local_model_dir(target, size)
    if found is not None:
        return found
    try:
        return download_model(source, size, target, log)
    except Exception as e:  # noqa: BLE001 - any hub failure falls through to local conversion
        log(f"{source} has no converted {size} model ({e.__class__.__name__}); converting kraken's weights locally.")
    return convert_model(size, target / MODELS_SUBDIR / model_dir_name(size), log)


def download_model(repo: str, size: str, target: Path, log: Log = logger.info) -> Path:
    from huggingface_hub import snapshot_download

    name = model_dir_name(size)
    log(f"Downloading {repo}/{MODELS_SUBDIR}/{name} -> {target}")
    snapshot_download(repo, allow_patterns=[f"{MODELS_SUBDIR}/{name}/*", "README.md"], local_dir=str(target))
    found = local_model_dir(target, size)
    if found is None:
        raise FileNotFoundError(f"{repo} does not contain {MODELS_SUBDIR}/{name}/{REQUIRED_FILES[0]}.")
    return found


def convert_model(size: str, target: Path, log: Log = logger.info) -> Path:
    """Download kraken's weights for ``size`` and convert them into ``target``."""
    try:
        from .convert.convert import convert
    except ImportError as e:
        raise RuntimeError(
            f"No converted {size} model available and the converter is not installed. "
            "Install it with `pip install \"squiddleocr[convert]\"` (needs torch and kraken), "
            "or point --models at a folder or Hub repo that has the model."
        ) from e
    from huggingface_hub import snapshot_download

    repo = KRAKEN_REPOS[size]
    log(f"Downloading kraken weights {repo}")
    src = Path(snapshot_download(repo, allow_patterns=[f"{size}.safetensors", "README.md"]))
    convert(src / f"{size}.safetensors", target, log=log)
    return target


def _repo_dirname(repo: str) -> str:
    return repo.replace("/", "--")
