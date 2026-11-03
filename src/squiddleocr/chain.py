"""The squiddle pipeline: a linecaster chain.

PP-DocLayoutV3 layout and PP-OCRv6 text-line detection on the whole page (linecaster-paddlex), lines
assigned to the layout regions (linecaster's ``AssignLinesToRegions``), kraken's PP-OCRv6 medium
recogniser (linecaster-kraken), PaddleX's table structure for table regions filled with the recognised
lines (linecaster-paddlex; the page text writes tables as Markdown, HTML when cells span). The same chain as linecaster's ``examples/squiddle_chain.py``, with
both backends:

* ``in-process``: the engines run in this process (needs this environment's torch/ONNX Runtime GPU
  stack; the parity reference);
* ``served``: each step goes through saturate to ``linecaster serve`` hosting the same models
  (``linecaster serve --config examples/squiddle_models.json``).

What the chain does not have yet (formulas, the retyper, the DoclingDocument with Markdown, kraken's blla
pipeline, eynollah) is still available as ``squiddle legacy ocr`` (see NOTES.md).
"""
from __future__ import annotations

from linecaster.backends import InProcess, Served
from linecaster.blocks.lines import AssignLinesToRegions
from linecaster.blocks.models import LayoutStep, LinesStep, RecognitionStep, TablesStep

ENDPOINT = "http://127.0.0.1:8100/v1"
TRANSCRIPTION = "ocr"
#: the squiddle combination (linecaster's examples/squiddle_models.json): name, engine class, settings
MODELS = [
    {"name": "layout", "spec": "linecaster_paddlex:PaddleLayoutModel", "kwargs": {"model_name": "PP-DocLayoutV3"}},
    {"name": "lines", "spec": "linecaster_paddlex:PaddleLineModel", "kwargs": {"model_name": "PP-OCRv6_medium_det"}},
    {"name": "recognition", "spec": "linecaster_kraken:KrakenRecognitionModel",
     "kwargs": {"model": "medium", "batch_size": 8}},
    {"name": "tables", "spec": "linecaster_paddlex:PaddleTableModel", "kwargs": {}},
]


def backends(kind: str = "in-process", endpoint: str = ENDPOINT, window: int = 32,
             recognition_model: str | None = None) -> dict:
    models = [dict(m, kwargs=dict(m["kwargs"])) for m in MODELS]
    if recognition_model:
        models[2]["kwargs"]["model"] = recognition_model
    if kind == "in-process":
        return {m["name"]: InProcess(m["spec"], **m["kwargs"]) for m in models}
    if kind == "served":
        return {m["name"]: Served(endpoint, model=m["name"], max_window=window) for m in models}
    raise ValueError(f"unknown backend {kind!r} (in-process or served)")


def blocks(kind: str = "in-process", endpoint: str = ENDPOINT, transcription: str = TRANSCRIPTION,
           window: int = 32, recognition_model: str | None = None) -> list:
    """The chain's five blocks; put a reader before and writers after."""
    b = backends(kind, endpoint, window, recognition_model)
    return [LayoutStep(b["layout"], transcription),
            LinesStep(b["lines"], transcription),
            AssignLinesToRegions(transcription),
            RecognitionStep(b["recognition"], transcription),
            TablesStep(b["tables"], transcription)]
