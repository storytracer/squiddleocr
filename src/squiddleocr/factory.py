"""Build a ``Pipeline`` from component names (the CLI's view of the components)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .layout import SingleRegionLayout
from .models import DEFAULT_SIZE, resolve_kraken_model
from .pipeline import Pipeline
from .recognizers.kraken import KrakenRecognizer

if TYPE_CHECKING:
    from .eynollah import EynollahOptions

PADDLE_HINT = 'PaddleX layout, text detection and tables need the paddle extra: pip install "squiddleocr[paddle]"'
EYNOLLAH_HINT = 'the eynollah stage needs the eynollah extra: pip install "squiddleocr[eynollah]"'
PIPELINES = ("paddle", "kraken", "eynollah")
LAYOUTS = ("paddle", "none", "eynollah")


def _import(module: str):
    """Import an optional component module with an install hint instead of a bare ImportError."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        if "paddle" in str(e).lower():
            raise RuntimeError(PADDLE_HINT) from e
        if "eynollah" in str(e).lower():
            raise RuntimeError(EYNOLLAH_HINT) from e
        raise


def build_pipeline(model: str | Path = DEFAULT_SIZE, *, pipeline: str = "paddle", layout: str = "paddle",
                   det_model: str = "PP-OCRv6_medium_det", layout_model: str = "PP-DocLayoutV3", tables: bool = True,
                   formulas: bool = True, formula_model: str = "PP-FormulaNet_plus-L",
                   unclip_ratio: float = 2.0, device: str = "auto", batch_size: int = 8,
                   eynollah: "EynollahOptions | None" = None, text_direction: str = "horizontal-lr",
                   log=None, warn=None, progress=None) -> Pipeline:
    """kraken's recogniser (``model``: a size fetched by DOI, or a kraken model file) in one of three pipelines.

    ``paddle``: PaddleX layout analysis (``layout="paddle"``, or ``"none"`` for one text block per page),
    PP-OCRv6 text detection (line boxes, a kraken ``bbox`` segmentation), SLANet tables and
    PP-FormulaNet formulas (LaTeX) in the layout regions.
    ``kraken``: the blla segmenter on the whole page (polygons and baselines, a kraken ``baselines``
    segmentation), kraken's own line order, no layout analysis, tables or formulas: what the
    ``kraken`` command does.
    ``eynollah`` (also ``layout="eynollah"``): eynollah's regions and reading order from its PAGE-XML
    (eynollah runs as a subprocess, see ``squiddleocr.eynollah``), then PP-OCRv6 detection on each text
    region's masked crop (``eynollah.lines="paddle"``, the default), or eynollah's own line polygons read
    along a synthesised baseline (``"eynollah"``), or blla's lines and baselines per region (``"blla"``,
    seconds per region); no tables or formulas stage. ``eynollah`` carries the run
    options, ``text_direction="horizontal-rl"`` for right-to-left scripts."""
    log = log or (lambda s: None)
    recognizer = KrakenRecognizer(resolve_kraken_model(model, log), device=device, batch_size=batch_size,
                                  text_direction=text_direction)
    if pipeline == "eynollah" or layout == "eynollah":
        eyn = _import("squiddleocr.eynollah")
        opts = eynollah or eyn.EynollahOptions()
        opts.rtl = opts.rtl or text_direction == "horizontal-rl"
        source = eyn.EynollahSource(opts, log=log, warn=warn, progress=progress)
        keep_order = True
        if opts.lines == "paddle":       # PP-OCRv6 boxes per region; fragments are grouped into visual rows
            det = _import("squiddleocr.detectors.paddle").PaddleTextDetector(opts.det_model, device=device,
                                                                            unclip_ratio=opts.unclip_ratio, pad=opts.crop_pad, mask=True)
            keep_order = False
        elif opts.lines == "blla":
            from .detectors.kraken import KrakenSegmenter

            det = KrakenSegmenter(device=device, text_direction=text_direction, pad=opts.crop_pad, mask=True)
        elif opts.lines == "eynollah":
            det = _import("squiddleocr.detectors.eynollah").EynollahLines(source, baselines=opts.baselines, descender=opts.descender)
        else:
            raise ValueError(f"Unknown eynollah line source {opts.lines!r}; choose eynollah, paddle or blla")
        lay = _import("squiddleocr.layout.eynollah").EynollahLayout(source)
        return Pipeline(recognizer=recognizer, detector=det, layout=lay, detect_per_region=True, keep_line_order=keep_order)

    if pipeline == "paddle":
        det = _import("squiddleocr.detectors.paddle").PaddleTextDetector(det_model, device=device, unclip_ratio=unclip_ratio)
    elif pipeline == "kraken":
        from .detectors.kraken import KrakenSegmenter

        det = KrakenSegmenter(device=device, text_direction=text_direction)
        layout, tables, formulas = "none", False, False
    else:
        raise ValueError(f"Unknown pipeline {pipeline!r}; choose {', '.join(PIPELINES)}")

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
        raise ValueError(f"Unknown layout {layout!r}; choose {', '.join(LAYOUTS)}")

    return Pipeline(recognizer=recognizer, detector=det, layout=lay, tables=table_rec, formulas=formula_rec,
                    keep_line_order=pipeline == "kraken")
