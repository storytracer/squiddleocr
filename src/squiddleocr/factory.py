"""Build a ``Pipeline`` from component names (the CLI's view of the components)."""
from __future__ import annotations

from pathlib import Path

from .layout import SingleRegionLayout
from .models import DEFAULT_SIZE, resolve_kraken_model
from .pipeline import Pipeline
from .recognizers.kraken import KrakenRecognizer

PADDLE_HINT = 'PaddleX layout, text detection and tables need the paddle extra: pip install "squiddleocr[paddle]"'


def _import(module: str):
    """Import an optional PaddleX component module with an install hint instead of a bare ImportError."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        if "paddle" in str(e).lower():
            raise RuntimeError(PADDLE_HINT) from e
        raise


def build_pipeline(model: str | Path = DEFAULT_SIZE, *, pipeline: str = "paddle", layout: str = "paddle",
                   det_model: str = "PP-OCRv6_medium_det", layout_model: str = "PP-DocLayoutV3", tables: bool = True,
                   formulas: bool = True, formula_model: str = "PP-FormulaNet_plus-L",
                   unclip_ratio: float = 2.0, device: str = "auto", batch_size: int = 8, log=None) -> Pipeline:
    """kraken's recogniser (``model``: a size fetched by DOI, or a kraken model file) in one of two pipelines.

    ``paddle``: PaddleX layout analysis (``layout="paddle"``, or ``"none"`` for one text block per page),
    PP-OCRv6 text detection (line boxes, a kraken ``bbox`` segmentation), SLANet tables and
    PP-FormulaNet formulas (LaTeX) in the layout regions.
    ``kraken``: the blla segmenter on the whole page (polygons and baselines, a kraken ``baselines``
    segmentation), kraken's own line order, no layout analysis, tables or formulas: what the
    ``kraken`` command does."""
    log = log or (lambda s: None)
    recognizer = KrakenRecognizer(resolve_kraken_model(model, log), device=device, batch_size=batch_size)

    if pipeline == "paddle":
        det = _import("squiddleocr.detectors.paddle").PaddleTextDetector(det_model, device=device, unclip_ratio=unclip_ratio)
    elif pipeline == "kraken":
        from .detectors.kraken import KrakenSegmenter

        det = KrakenSegmenter(device=device)
        layout, tables, formulas = "none", False, False
    else:
        raise ValueError(f"Unknown pipeline {pipeline!r}; choose paddle or kraken")

    table_rec = formula_rec = None
    if layout == "none":
        lay = SingleRegionLayout()
    elif layout == "paddle":
        lay = _import("squiddleocr.layout.paddle").PaddleLayout(layout_model, device=device)
        if tables:
            table_rec = _import("squiddleocr.tables.paddle").PaddleTableRecognizer(device=device)
        if formulas:
            formula_rec = _import("squiddleocr.formulas.paddle").PaddleFormulaRecognizer(formula_model, device=device,
                                                                                         batch_size=batch_size)
    else:
        raise ValueError(f"Unknown layout {layout!r}; choose paddle or none")

    return Pipeline(recognizer=recognizer, detector=det, layout=lay, tables=table_rec, formulas=formula_rec,
                    keep_line_order=pipeline == "kraken")
