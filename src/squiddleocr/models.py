"""Model files: kraken's from Hugging Face (``huggingface_hub``).

kraken's models always come from the Hugging Face mirrors of their Zenodo releases
(``small-models-for-glam``): Zenodo throttles downloads badly. The mirrors are byte-identical to the
Zenodo records, whose DOIs stay in ``KRAKEN_MODELS`` as provenance; each one is pinned to a revision,
so every run gets the same file. They land in the Hugging Face cache (``~/.cache/huggingface/hub``,
``HF_HOME``; ``HF_HUB_OFFLINE=1`` reads it without the network). Nothing model-sized ever lives in
the checkout.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SIZES = ("tiny", "small", "medium")
DEFAULT_SIZE = "medium"


@dataclass(frozen=True)
class KrakenModel:
    """One kraken model file on a Hugging Face mirror of its Zenodo release."""

    repo: str
    filename: str
    revision: str   # the mirror's commit, so a run always gets the same file
    doi: str        # the Zenodo release the mirror copies: provenance and citation


#: kraken's PP-OCRv6 recognisers (Benjamin Kiessling, Apache-2.0) and the blla segmenter: the new default
#: BLLA base model (Sept 2026), newer than the one kraken 7.1.1 bundles (our fork bundles this file too).
KRAKEN_MODELS = {
    "tiny": KrakenModel("small-models-for-glam/kraken-ppocrv6-tiny", "tiny.safetensors",
                        "045d350d5a7664054370539b87bab0e790724ee2", "10.5281/zenodo.21788403"),
    "small": KrakenModel("small-models-for-glam/kraken-ppocrv6-small", "small.safetensors",
                         "49c249bf81d95ab2c5f699638d066f7935da6f16", "10.5281/zenodo.21788405"),
    "medium": KrakenModel("small-models-for-glam/kraken-ppocrv6-medium", "medium.safetensors",
                          "f7fd70f3fd22b615f41bada4674ac2462f938339", "10.5281/zenodo.21788410"),
    "blla": KrakenModel("small-models-for-glam/kraken-blla-base", "blla.mlmodel",
                        "3e91a744c37defc10c3714d9f0f4fa54174fe5c0", "10.5281/zenodo.22879549"),
}

Log = Callable[[str], None]


def kraken_model_file(name: str, log: Log = logger.info, download: bool = True) -> Path:
    """The file of kraken model ``name`` (a key of ``KRAKEN_MODELS``) from its pinned Hugging Face mirror:
    downloaded on first use, served from the Hugging Face cache afterwards. ``download=False`` only
    looks in the cache and raises ``FileNotFoundError`` when the file is not there."""
    from huggingface_hub import hf_hub_download

    m = KRAKEN_MODELS[name]
    log(f"kraken model {name}: {m.repo}@{m.revision[:12]} (Hugging Face cache or download)")
    return Path(hf_hub_download(m.repo, m.filename, revision=m.revision, local_files_only=not download))


def resolve_kraken_model(spec: str | Path = DEFAULT_SIZE, log: Log = logger.info, download: bool = True) -> Path:
    """The recognition model for ``spec``: a size name (from Hugging Face) or a path to a kraken model file."""
    p = Path(spec)
    if p.is_file():
        return p
    size = str(spec).lower()
    if size not in SIZES:
        raise FileNotFoundError(f"{spec} is neither a model size ({', '.join(SIZES)}) nor a kraken model file.")
    return kraken_model_file(size, log, download)

