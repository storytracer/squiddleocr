"""Build a ``Pipeline`` from component names (the CLI's view of the components)."""
from __future__ import annotations

from pathlib import Path

from .layout import SingleRegionLayout
from .pipeline import Pipeline
from .recognizers import OnnxRecognizer


def build_pipeline(model_dir: str | Path, *, layout: str = "paddle", detector: str = "paddle",
                   det_model: str = "PP-OCRv6_medium_det", tables: bool = True, unclip_ratio: float = 2.0,
                   device: str = "auto", batch_size: int = 8) -> Pipeline:
    recognizer = OnnxRecognizer(model_dir, device=device, batch_size=batch_size)

    if detector == "paddle":
        from .detectors.paddle import PaddleTextDetector

        det = PaddleTextDetector(det_model, device=device, unclip_ratio=unclip_ratio)
    elif detector == "kraken":
        from .detectors.kraken import KrakenSegmenter

        det = KrakenSegmenter(device=device)
    else:
        raise ValueError(f"Unknown detector {detector!r}")

    if layout == "none":
        lay = SingleRegionLayout()
        table_rec = None
    elif layout == "paddle":
        from .layout.paddle import PaddleLayout

        lay = PaddleLayout(device=device)
        table_rec = None
        if tables:
            from .tables.paddle import PaddleTableRecognizer

            table_rec = PaddleTableRecognizer(device=device)
    else:
        raise ValueError(f"Unknown layout {layout!r}")

    return Pipeline(recognizer=recognizer, detector=det, layout=lay, tables=table_rec)
