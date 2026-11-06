"""The squiddle pipeline: a linecaster chain.

PP-DocLayoutV3 layout (or one region per page) and PP-OCRv6 text-line detection on the whole page
(linecaster-paddlex), lines assigned to the layout regions (``LineAssigner``), kraken's PP-OCRv6
recogniser (linecaster-kraken), the as-written text in NFC (``UnicodeNormalizer``), PaddleX's table
structure for table regions filled with the recognised lines and PP-FormulaNet LaTeX for formula
regions (linecaster-paddlex), the retyper (``Typographer``), the running text in paragraphs
(``Recomposer``; ``--text reflow``) and the DoclingDocument of each page (``Composer``); with ``--per-document``
the document of the item (``Binder``, item-level ``Recomposer`` and ``Composer``; ``item_blocks``). The rule-based
steps run on the recipe's presets (``ratios``, ``unicode-recomposition``, ``docling-composition``). Writers are the
CLI's (``cli.py``). The same chain as linecaster's
``pipelines/transcribe_modular.py`` with the ``squiddle`` recipe, with both backends:

* ``in-process``: the engines run in this process (needs this environment's torch/ONNX Runtime GPU
  stack; the parity reference);
* ``served``: each model step goes through saturate to ``linecaster serve`` hosting the same models
  (``uv run pipelines/transcribe_modular.py serve --config <(squiddle models-config)`` in linecaster, or ``serve`` alone
  for the defaults). The model settings are then the server's.

What stays in ``squiddle legacy ocr``: kraken's blla pipeline (``--pipeline kraken``) and eynollah.
"""
from __future__ import annotations

from linecaster.backends import InProcess, Served
from linecaster.blocks.base import PageBlock
from linecaster.blocks.base.page import failed
from linecaster.blocks.binding import Binder
from linecaster.blocks.composition import Composer, Recomposer, Typographer
from linecaster.blocks.layout import LayoutDetector
from linecaster.blocks.lines import LineAssigner, LineDetector
from linecaster.blocks.recognition import FormulaRecognizer, LineRecognizer, TableRecognizer
from linecaster.blocks.text import UnicodeNormalizer

ENDPOINT = "http://127.0.0.1:8100/v1"
TRANSCRIPTION = "ocr_new"
#: 0.3's model defaults (``squiddle legacy ocr``)
DEFAULTS = dict(recognition_model="medium", det_model="PP-OCRv6_medium_det", unclip_ratio=1.5,
                layout_model="PP-DocLayoutV3", formula_model="PP-FormulaNet_plus-L", batch_size=8, precision="auto",
                device="auto", rtl=False)
#: SquiddleOCR's recipe in linecaster's catalogue (``linecaster/catalog/data/recipes/squiddle.yaml``): its
#: modular pipeline with these preset editions; ``squiddle ocr`` at its defaults runs exactly this
RECIPE = "squiddle"
#: the recipe's slots that are models (the others are rule-based steps, in this process: typography,
#: recomposition, normalization, composition)
MODEL_STEPS = ("layout", "lines", "recognition", "tables", "formulas")
TEXT_MODES = ("reflow", "lines")


def recipe():
    """The recipe document (preset editions per slot, options)."""
    from linecaster.catalog.registry import catalog

    return catalog().recipe(RECIPE)


PRESETS = dict(recipe().presets)
MODELS = [{"name": name, "preset": preset} for name, preset in PRESETS.items() if name in MODEL_STEPS]
LAYOUT_PRESETS = {"PP-DocLayoutV3": "pp-doclayoutv3", "PP-DocLayout_plus-L": "pp-doclayout-plus-l"}
FORMULA_PRESETS = {"PP-FormulaNet_plus-L": "pp-formulanet-plus-l", "PP-FormulaNet-L": "pp-formulanet-l"}
SIZES = ("tiny", "small", "medium")


def models(recognition_model: str | None = None, det_model: str | None = None, unclip_ratio: float | None = None,
           layout_model: str | None = None, formula_model: str | None = None, batch_size: int | None = None,
           precision: str | None = None, device: str | None = None, rtl: bool | None = None) -> list[dict]:
    """The model list with 0.3's options applied: a model name picks its preset, other settings go over the
    preset's options as ``overrides`` (only options that differ from the defaults are written, so the default
    list is ``MODELS``). A recogniser given as a model file has no preset; it gets the engine and its options.
    The formula model reads ``batch_size`` crops per pass, as in 0.3."""
    o = {k: v for k, v in dict(recognition_model=recognition_model, det_model=det_model, unclip_ratio=unclip_ratio,
                               layout_model=layout_model, formula_model=formula_model, batch_size=batch_size,
                               precision=precision, device=device, rtl=rtl).items()
         if v is not None and v != DEFAULTS[k]}
    presets = {name: preset for name, preset in PRESETS.items() if name in MODEL_STEPS}
    over: dict[str, dict] = {name: {} for name in presets}
    if "layout_model" in o:
        presets["layout"] = LAYOUT_PRESETS[o["layout_model"]]
    if "det_model" in o:
        size = str(o["det_model"]).removeprefix("PP-OCRv6_").removesuffix("_det")
        if size not in SIZES:
            raise ValueError(f"unknown text detector {o['det_model']!r}")
        presets["lines"] = f"pp-ocrv6-{size}-det"
    if "unclip_ratio" in o:
        over["lines"]["unclip_ratio"] = float(o["unclip_ratio"])
    rec_file = None
    if "recognition_model" in o:
        if str(o["recognition_model"]) in SIZES:
            presets["recognition"] = f"kraken-ppocrv6-{o['recognition_model']}"
        else:
            rec_file = str(o["recognition_model"])
    if "batch_size" in o:
        over["recognition"]["batch_size"] = over["formulas"]["batch_size"] = int(o["batch_size"])
    if "precision" in o:
        over["recognition"]["precision"] = o["precision"]
    if o.get("rtl"):
        over["recognition"]["text_direction"] = "horizontal-rl"
    if "formula_model" in o:
        presets["formulas"] = FORMULA_PRESETS[o["formula_model"]]
    if "device" in o:
        for v in over.values():
            v["device"] = o["device"]
    out = []
    for name, preset in presets.items():
        if name == "recognition" and rec_file:
            out.append({"name": name, "spec": "linecaster_kraken:KrakenRecognitionEngine",
                        "kwargs": {"model": rec_file, "batch_size": DEFAULTS["batch_size"], **over[name]}})
        else:
            out.append({"name": name, "preset": preset, **({"overrides": over[name]} if over[name] else {})})
    return out


def backends(kind: str = "in-process", endpoint: str = ENDPOINT, window: int = 32, **model_options) -> dict:
    if kind == "in-process":
        from linecaster.runtime.run import preset_models

        out = {}
        for m in preset_models(models(**model_options)):
            b = out[m["name"]] = InProcess(m["spec"], **m["kwargs"])
            b.preset, b.preset_sha256, b.overrides = m.get("preset"), m.get("preset_sha256"), m.get("overrides")
        return out
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
    DoclingDocument of each page (``text``: ``reflow``, recomposed into paragraphs first, or ``lines``, as laid
    out; ``sections``: see ``composition``). ``model_options``: see ``models``."""
    if layout not in ("paddle", "none"):
        raise ValueError(f"unknown layout {layout!r} (paddle or none)")
    if text not in TEXT_MODES:
        raise ValueError(f"unknown text {text!r} (reflow or lines)")
    b = backends(kind, endpoint, window, **model_options)
    regions = layout == "paddle"
    tables, formulas = tables and regions, formulas and regions
    return [LayoutDetector(b["layout"] if regions else "single-region@1", transcription),
            LineDetector(b["lines"], transcription),
            LineAssigner(transcription, formulas_separately=formulas),
            LineRecognizer(b["recognition"], transcription),
            *([UnicodeNormalizer(transcription)] if nfc else []),
            *([TableRecognizer(b["tables"], transcription)] if tables else []),
            *([FormulaRecognizer(b["formulas"], transcription)] if formulas else []),
            *([Typographer(PRESETS["typography"], transcription)] if typography else []),
            *(composition(transcription, text=text, sections=sections, rtl=bool(model_options.get("rtl")))
              if document else [])]


def composition(transcription: str = TRANSCRIPTION, unit: str = "page", *, text: str = "reflow",
                sections: bool = False, rtl: bool = False) -> list:
    """The document of a page (``unit="page"``) or of an item (``"item"``, after a ``Binder``): with ``text="reflow"``
    a ``Recomposer`` first (``rtl``: mirrored geometry), then the ``Composer`` (``sections``: a heading and what
    follows it as a Docling section); with ``lines`` the ``Composer`` alone (one row per line). Options are given
    only when they differ from the presets' (the defaults are the recipe's)."""
    if text not in TEXT_MODES:
        raise ValueError(f"unknown text {text!r} (reflow or lines)")
    return [*([Recomposer(PRESETS["recomposition"], transcription, unit=unit, options={"rtl": True} if rtl else None)]
              if text == "reflow" else []),
            Composer(PRESETS["composition"], transcription, unit=unit, options={"sections": True} if sections else None)]


def item_blocks(transcription: str = TRANSCRIPTION, *, text: str = "reflow", sections: bool = False,
                rtl: bool = False) -> list:
    """``--per-document``: the pages gathered into one item record (``Binder``; one task sees the whole item) and its
    document composed (``composition`` with ``unit="item"``: paragraphs continued across pages), for an
    ``ItemDocumentWriter``."""
    return [Binder(), *composition(transcription, "item", text=text, sections=sections, rtl=rtl)]


class FailedPagesDropped(PageBlock):
    """Drops the pages an earlier block failed on (they carry ``doc.metadata["errors"]``); the others pass unchanged.
    The CLI puts it after the ``RejectWriter`` and the page records, which keep them, and before the files and the
    document, which do not (0.3 skipped a failed page)."""

    name = "failed pages dropped"

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            if failed(doc):
                self.stat_update("dropped")
                continue
            yield doc
