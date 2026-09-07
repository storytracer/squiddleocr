"""Build a ``Pipeline`` from component names (the CLI's view of the components)."""
from __future__ import annotations

from pathlib import Path

from .layout import SingleRegionLayout
from .models import DEFAULT_SIZE, resolve_model
from .pipeline import Pipeline
from .recognizers import OnnxRecognizer

EXTRAS_HINT = {
    "paddlex": 'the layout, detection and table models need the paddle extra: pip install "squiddleocr[paddle]"',
    "kraken": 'the kraken segmenter needs the kraken extra: pip install "squiddleocr[kraken]"',
}


def _import(module: str):
    """Import an optional component module with an install hint instead of a bare ImportError."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        for dep, hint in EXTRAS_HINT.items():
            if dep in str(e) or dep in module:
                raise RuntimeError(hint) from e
        raise


def build_pipeline(model: str | Path = DEFAULT_SIZE, *, models: str | Path | None = None, layout: str = "paddle",
                   detector: str = "paddle", det_model: str = "PP-OCRv6_medium_det", tables: bool = True,
                   unclip_ratio: float = 2.0, device: str = "auto", batch_size: int = 8, log=None) -> Pipeline:
    """``model``: a size name (``tiny``/``small``/``medium``) or a model directory; ``models``: the folder or
    Hub repo the sizes come from (default ``storytracer/squiddleocr``, fetched on first use)."""
    model_dir = resolve_model(model, models, log) if log else resolve_model(model, models)
    recognizer = OnnxRecognizer(model_dir, device=device, batch_size=batch_size)

    if detector == "paddle":
        det = _import("squiddleocr.detectors.paddle").PaddleTextDetector(det_model, device=device, unclip_ratio=unclip_ratio)
    elif detector == "kraken":
        det = _import("squiddleocr.detectors.kraken").KrakenSegmenter(device=device)
    else:
        raise ValueError(f"Unknown detector {detector!r}")

    table_rec = None
    if layout == "none":
        lay = SingleRegionLayout()
    elif layout == "paddle":
        lay = _import("squiddleocr.layout.paddle").PaddleLayout(device=device)
        if tables:
            table_rec = _import("squiddleocr.tables.paddle").PaddleTableRecognizer(device=device)
    else:
        raise ValueError(f"Unknown layout {layout!r}")

    return Pipeline(recognizer=recognizer, detector=det, layout=lay, tables=table_rec)
