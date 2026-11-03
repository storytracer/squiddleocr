"""The squiddle pipeline: a linecaster chain.

PP-DocLayoutV3 layout (or one region per page) and PP-OCRv6 text-line detection on the whole page
(linecaster-paddlex), lines assigned to the layout regions (``AssignLinesToRegions``), kraken's PP-OCRv6
recogniser (linecaster-kraken), the as-written text in NFC (``NormalizeUnicode``), PaddleX's table
structure for table regions filled with the recognised lines and PP-FormulaNet LaTeX for formula
regions (linecaster-paddlex), the retyper (``Typography``) and the DoclingDocument of each page
(``AssembleDocument``). Writers are the CLI's (``cli.py``). The same chain as linecaster's
``examples/squiddle_chain.py``, with both backends:

* ``in-process``: the engines run in this process (needs this environment's torch/ONNX Runtime GPU
  stack; the parity reference);
* ``served``: each model step goes through saturate to ``linecaster serve`` hosting the same models
  (``linecaster serve --config <(squiddle models-config)``, or linecaster's ``examples/squiddle_models.json``
  for the defaults). The model settings are then the server's.

What stays in ``squiddle legacy ocr``: kraken's blla pipeline (``--pipeline kraken``) and eynollah.
"""
from __future__ import annotations

from linecaster.backends import InProcess, Served
from linecaster.blocks.document import AssembleDocument, Typography
from linecaster.blocks.layout import SingleRegionLayout
from linecaster.blocks.lines import AssignLinesToRegions
from linecaster.blocks.models import FormulasStep, LayoutStep, LinesStep, RecognitionStep, TablesStep
from linecaster.blocks.unicode import NormalizeUnicode

ENDPOINT = "http://127.0.0.1:8100/v1"
TRANSCRIPTION = "ocr"
#: 0.3's model defaults (``squiddle legacy ocr``)
DEFAULTS = dict(recognition_model="medium", det_model="PP-OCRv6_medium_det", unclip_ratio=1.5,
                layout_model="PP-DocLayoutV3", formula_model="PP-FormulaNet_plus-L", batch_size=8, precision="auto",
                device="auto", rtl=False)
#: the squiddle combination with the defaults (linecaster's examples/squiddle_models.json): name, engine class, settings
MODELS = [
    {"name": "layout", "spec": "linecaster_paddlex:PaddleLayoutModel", "kwargs": {"model_name": "PP-DocLayoutV3"}},
    {"name": "lines", "spec": "linecaster_paddlex:PaddleLineModel", "kwargs": {"model_name": "PP-OCRv6_medium_det"}},
    {"name": "recognition", "spec": "linecaster_kraken:KrakenRecognitionModel",
     "kwargs": {"model": "medium", "batch_size": 8}},
    {"name": "tables", "spec": "linecaster_paddlex:PaddleTableModel", "kwargs": {}},
    {"name": "formulas", "spec": "linecaster_paddlex:PaddleFormulaModel", "kwargs": {}},
]


def models(recognition_model: str | None = None, det_model: str | None = None, unclip_ratio: float | None = None,
           layout_model: str | None = None, formula_model: str | None = None, batch_size: int | None = None,
           precision: str | None = None, device: str | None = None, rtl: bool | None = None) -> list[dict]:
    """The model list with 0.3's options applied (only those that differ from the defaults are written,
    so the default list is ``MODELS``). The formula model reads ``batch_size`` crops per pass, as in 0.3."""
    o = {k: v for k, v in dict(recognition_model=recognition_model, det_model=det_model, unclip_ratio=unclip_ratio,
                               layout_model=layout_model, formula_model=formula_model, batch_size=batch_size,
                               precision=precision, device=device, rtl=rtl).items()
         if v is not None and v != DEFAULTS[k]}
    out = {m["name"]: dict(m, kwargs=dict(m["kwargs"])) for m in MODELS}
    kw = {name: out[name]["kwargs"] for name in out}
    if "layout_model" in o:
        kw["layout"]["model_name"] = o["layout_model"]
    if "det_model" in o:
        kw["lines"]["model_name"] = o["det_model"]
    if "unclip_ratio" in o:
        kw["lines"]["unclip_ratio"] = float(o["unclip_ratio"])
    if "recognition_model" in o:
        kw["recognition"]["model"] = str(o["recognition_model"])
    if "batch_size" in o:
        kw["recognition"]["batch_size"] = kw["formulas"]["batch_size"] = int(o["batch_size"])
    if "precision" in o:
        kw["recognition"]["precision"] = o["precision"]
    if o.get("rtl"):
        kw["recognition"]["text_direction"] = "horizontal-rl"
    if "formula_model" in o:
        kw["formulas"]["model_name"] = o["formula_model"]
    if "device" in o:
        for k in kw.values():
            k["device"] = o["device"]
    return list(out.values())


def backends(kind: str = "in-process", endpoint: str = ENDPOINT, window: int = 32, **model_options) -> dict:
    if kind == "in-process":
        return {m["name"]: InProcess(m["spec"], **m["kwargs"]) for m in models(**model_options)}
    if kind == "served":
        return {m["name"]: Served(endpoint, model=m["name"], max_window=window) for m in MODELS}
    raise ValueError(f"unknown backend {kind!r} (in-process or served)")


def blocks(kind: str = "in-process", endpoint: str = ENDPOINT, transcription: str = TRANSCRIPTION, window: int = 32,
           *, layout: str = "paddle", tables: bool = True, formulas: bool = True, typography: bool = True,
           nfc: bool = True, text: str = "reflow", sections: bool = False, document: bool = True,
           **model_options) -> list:
    """The chain's blocks; put a reader before and writers after.

    ``layout``: ``paddle`` (PP-DocLayout regions) or ``none`` (the page is one text region; no tables or
    formulas). ``tables`` / ``formulas`` / ``typography``: the steps 0.3 switches with the same flags;
    without formulas, formula regions are read as text. ``nfc``: the as-written layer in NFC (linecaster's
    choice; 0.3 kept kraken's decomposed text; off only for parity checks). ``document``: assemble the
    DoclingDocument of each page (``text``: ``reflow`` or ``lines``; ``sections``). ``model_options``:
    see ``models``."""
    if layout not in ("paddle", "none"):
        raise ValueError(f"unknown layout {layout!r} (paddle or none)")
    b = backends(kind, endpoint, window, **model_options)
    regions = layout == "paddle"
    tables, formulas = tables and regions, formulas and regions
    return [LayoutStep(b["layout"], transcription) if regions else SingleRegionLayout(transcription),
            LinesStep(b["lines"], transcription),
            AssignLinesToRegions(transcription, formulas_separately=formulas),
            RecognitionStep(b["recognition"], transcription),
            *([NormalizeUnicode(transcription)] if nfc else []),
            *([TablesStep(b["tables"], transcription)] if tables else []),
            *([FormulasStep(b["formulas"], transcription)] if formulas else []),
            *([Typography(transcription)] if typography else []),
            *([AssembleDocument(transcription, text=text, rtl=bool(model_options.get("rtl")), sections=sections)]
              if document else [])]
