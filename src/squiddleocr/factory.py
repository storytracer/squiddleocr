"""Build a ``Pipeline`` from component names (the CLI's view of the components)."""
from __future__ import annotations

from pathlib import Path

from .layout import SingleRegionLayout
from .models import DEFAULT_SIZE, resolve_kraken_model, resolve_model
from .pipeline import Pipeline
from .recognizers import OnnxRecognizer

EXTRAS_HINT = {
    "paddlex": 'the layout, detection and table models need the paddle extra: pip install "squiddleocr[paddle]"',
    "kraken": 'the kraken level needs the kraken extra: pip install "squiddleocr[kraken]"',
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


def build_pipeline(model: str | Path = DEFAULT_SIZE, *, models: str | Path | None = None, level: str = "paddle",
                   layout: str = "paddle", det_model: str = "PP-OCRv6_medium_det", layout_model: str = "PP-DocLayoutV3",
                   tables: bool = True,
                   unclip_ratio: float = 2.0, device: str = "auto", batch_size: int = 8, log=None) -> Pipeline:
    """Two levels of detail, each as native as its stack allows.

    ``paddle``: PP-OCRv6 text detection (PaddleX) and the converted ONNX recogniser on ONNX Runtime;
    ``model`` is a size name or a converted model directory, ``models`` the folder or Hub repo the
    sizes come from. ``kraken``: kraken's blla segmenter and kraken's own recogniser on kraken's
    weights (fetched by DOI), with character cuts; ``model`` is a size name or a kraken model file.
    Layout (PaddleX) and tables are shared by both levels.
    """
    log = log or (lambda s: None)
    if level == "paddle":
        model_dir = resolve_model(model, models, log)
        recognizer = OnnxRecognizer(model_dir, device=device, batch_size=batch_size)
        det = _import("squiddleocr.detectors.paddle").PaddleTextDetector(det_model, device=device, unclip_ratio=unclip_ratio)
    elif level == "kraken":
        kraken_rec = _import("squiddleocr.recognizers.kraken")
        recognizer = kraken_rec.KrakenRecognizer(resolve_kraken_model(model, log), device=device, batch_size=batch_size)
        det = _import("squiddleocr.detectors.kraken").KrakenSegmenter(device=device)
    else:
        raise ValueError(f"Unknown level {level!r}; choose paddle or kraken")

    table_rec = None
    if layout == "none":
        lay = SingleRegionLayout()
    elif layout == "paddle":
        lay = _import("squiddleocr.layout.paddle").PaddleLayout(layout_model, device=device)
        if tables:
            table_rec = _import("squiddleocr.tables.paddle").PaddleTableRecognizer(device=device)
    else:
        raise ValueError(f"Unknown layout {layout!r}")

    return Pipeline(recognizer=recognizer, detector=det, layout=lay, tables=table_rec)
