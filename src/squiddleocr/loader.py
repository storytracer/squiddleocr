"""Loading kraken PP-OCRv6 recognition models from safetensors files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class KrakenRecModel:
    """A loaded kraken PP-OCRv6 recognition model plus the facts the converter needs."""

    model: torch.nn.Module            # kraken.lib.ppocr.model.PPOCRv6Model
    source: Path
    variant: str                       # tiny / small / medium
    height: int                        # input line height in px
    channels: int                      # 3 (RGB)
    num_classes: int                   # including CTC blank at index 0
    c2l: dict[str, list[int]]          # grapheme -> [label]
    metadata: dict[str, Any] = field(default_factory=dict)  # raw kraken_meta

    @property
    def net(self) -> torch.nn.Module:
        """The bare PPOCRv6Recognizer: ``net(x, seq_lens) -> (logits, olens)``."""
        return self.model.nn

    def sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.source, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


def read_kraken_meta(path: Path) -> dict[str, Any]:
    """Return the ``kraken_meta`` JSON header of a kraken safetensors file."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    raw = meta.get("kraken_meta")
    if raw is None:
        raise ValueError(f"{path} has no kraken_meta header; not a kraken model file.")
    return json.loads(raw)


def load_kraken_model(path: str | Path) -> KrakenRecModel:
    """Load a kraken PP-OCRv6 recognition model.

    Raises ``ValueError`` if the file does not contain exactly one PP-OCRv6
    recognition model.
    """
    from kraken.models.loaders import load_safetensors

    path = Path(path)
    models = load_safetensors(str(path), tasks=["recognition"])
    rec = [m for m in models if type(m).__name__ == "PPOCRv6Model"]
    if len(rec) != 1:
        found = [type(m).__name__ for m in models]
        raise ValueError(
            f"{path} must contain exactly one PP-OCRv6 recognition model, found {found}."
        )
    model = rec[0]
    model.eval()
    batch, channels, height, width = model.input
    if width != 0 or channels != 3:
        raise ValueError(
            f"Unexpected input spec {model.input}; expected (1, 3, H, 0) for RGB lines of free width."
        )
    if model.codec is None:
        raise ValueError(f"{path} has no codec; cannot build a dictionary.")
    return KrakenRecModel(
        model=model,
        source=path,
        variant=str(model.nn.variant),
        height=int(height),
        channels=int(channels),
        num_classes=int(model.num_classes),
        c2l={k: list(v) for k, v in model.codec.c2l.items()},
        metadata=read_kraken_meta(path),
    )
