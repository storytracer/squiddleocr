"""kraken's models, fetched by DOI the way ``kraken get`` does (``htrmopo``, cached in kraken's data dir)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SIZES = ("tiny", "small", "medium")
DEFAULT_SIZE = "medium"
#: Zenodo DOIs of kraken's PP-OCRv6 recognisers (Benjamin Kiessling, Apache-2.0) and of the blla segmenter.
KRAKEN_DOIS = {"tiny": "10.5281/zenodo.21788403", "small": "10.5281/zenodo.21788405", "medium": "10.5281/zenodo.21788410"}
KRAKEN_SEGMENTER_DOI = "10.5281/zenodo.14602569"

Log = Callable[[str], None]


def kraken_model_file(doi: str, suffixes: tuple[str, ...] = (".safetensors", ".mlmodel"), log: Log = logger.info) -> Path:
    """The model file of a kraken model on Zenodo, via ``htrmopo.get_model`` (``~/.local/share/htrmopo``;
    downloaded on first use, served from the cache afterwards)."""
    from htrmopo import get_model

    log(f"kraken model {doi} (htrmopo cache or Zenodo)")
    folder = Path(get_model(doi))
    for suffix in suffixes:
        files = sorted(folder.glob(f"*{suffix}"))
        if files:
            return files[0]
    raise FileNotFoundError(f"{doi} was fetched to {folder} but holds no {'/'.join(suffixes)} file.")


def resolve_kraken_model(spec: str | Path = DEFAULT_SIZE, log: Log = logger.info) -> Path:
    """The recognition model for ``spec``: a size name (fetched by DOI) or a path to a kraken model file."""
    p = Path(spec)
    if p.is_file():
        return p
    size = str(spec).lower()
    if size not in SIZES:
        raise FileNotFoundError(f"{spec} is neither a model size ({', '.join(SIZES)}) nor a kraken model file.")
    return kraken_model_file(KRAKEN_DOIS[size], (".safetensors",), log)
