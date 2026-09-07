"""Assemble a PaddleOCR-compatible model directory from a kraken PP-OCRv6 model."""
from __future__ import annotations

import importlib.metadata
import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from . import PADDLE_MODEL_NAMES, __version__, model_dir_name
from .codec import codec_to_dictionary, write_dict_file
from .config import build_inference_config, dump_yaml, validate_model_name
from .export import check_parity, export_onnx, make_session
from .loader import KrakenRecModel, load_kraken_model
from .modelcard import card_summary, find_model_card, write_model_card, write_notice, write_provenance
from .wrapper import DEFAULT_PADDING, PaddleRecWrapper

ONNX_FILE = "inference.onnx"
YML_FILE = "inference.yml"
DICT_FILE = "dict.txt"


@dataclass
class ConvertResult:
    out_dir: Path
    variant: str
    model_name: str
    num_chars: int
    height: int
    parity: dict[int, float]
    files: list[str]


def convert(
    source: str | Path,
    out_dir: str | Path | None = None,
    model_name: str | None = None,
    padding: int = DEFAULT_PADDING,
    embed_weights: bool = True,
    model_card: str | Path | None = None,
    parity_widths: tuple[int, ...] = (640, 1000),
    parity_tol: float = 1e-3,
    log: Callable[[str], None] = print,
) -> ConvertResult:
    """Convert ``source`` (kraken safetensors) into a PaddleOCR model directory.

    The directory contains ``inference.onnx``, ``inference.yml``, ``dict.txt``,
    ``MODEL_CARD.md`` (when a model card is found next to the source or given),
    ``NOTICE``, ``LICENSE`` and ``squiddle.json`` with provenance.
    """
    source = Path(source)
    log(f"Loading {source}")
    km: KrakenRecModel = load_kraken_model(source)
    variant = km.variant
    if model_name is None:
        model_name = PADDLE_MODEL_NAMES[variant]
    validate_model_name(model_name)
    out = Path(out_dir) if out_dir is not None else Path(model_dir_name(variant))
    out.mkdir(parents=True, exist_ok=True)
    log(f"Variant {variant}, input height {km.height}, {km.num_classes} classes -> {out}")

    chars = codec_to_dictionary(km.c2l)
    if len(chars) + 1 != km.num_classes:
        raise ValueError(
            f"Dictionary has {len(chars)} entries but the network has {km.num_classes} classes (incl. blank)."
        )
    write_dict_file(chars, out / DICT_FILE)

    cfg = build_inference_config(chars, km.height, model_name)
    dump_yaml(cfg, out / YML_FILE)

    wrapper = PaddleRecWrapper(km.net, padding=padding, adapt_input=True)
    log("Exporting ONNX (dynamo exporter)")
    onnx_path = export_onnx(wrapper, out / ONNX_FILE, km.height, embed_weights=embed_weights)
    session = make_session(onnx_path)
    parity = check_parity(wrapper, session, km.height, widths=parity_widths)
    worst = max(parity.values())
    log(f"ONNX Runtime vs PyTorch max abs diff: {parity}")
    if worst > parity_tol:
        raise RuntimeError(f"ONNX export disagrees with PyTorch by {worst:.3g} > {parity_tol:.3g}")

    card = Path(model_card) if model_card else find_model_card(source)
    summary = card_summary(card)
    write_model_card(card, out)
    write_notice(out, summary, variant)
    with importlib.resources.files("squiddleocr.data").joinpath("LICENSE-2.0.txt").open("rb") as f:
        (out / "LICENSE").write_bytes(f.read())

    try:
        kraken_version = importlib.metadata.version("kraken")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        kraken_version = None
    write_provenance(
        out,
        {
            "squiddleocr_version": __version__,
            "source_file": source.name,
            "source_sha256": km.sha256(),
            "source_model_card": card.name if card else None,
            "doi": summary.get("doi"),
            "authors": summary.get("authors"),
            "license": summary.get("license", "Apache-2.0"),
            "variant": variant,
            "paddle_model_name": model_name,
            "directory_name": model_dir_name(variant),
            "input": {"channels": 3, "height": km.height, "color": "RGB",
                      "paddle_normalisation": "(x/255 - 0.5)/0.5, folded into graph as 0.5 - 0.5*x",
                      "padding_px": padding},
            "output": {"layout": "(batch, seq, classes)", "softmax": True, "blank_index": 0,
                       "num_classes": km.num_classes, "width_subsampling": 8},
            "onnx": {"opset": 18, "exporter": "torch.onnx dynamo", "embedded_weights": embed_weights,
                     "torch_version": torch.__version__, "parity_max_abs_diff": parity},
            "kraken_version": kraken_version,
            "kraken_meta": km.metadata,
        },
    )
    files = sorted(p.name for p in out.iterdir())
    log(f"Wrote {files}")
    return ConvertResult(out, variant, model_name, len(chars), km.height, parity, files)
