"""SquiddleOCR: combine any layout, structure or line-segmentation model with kraken's PP-OCRv6 recogniser.

The recogniser is kraken's PP-OCRv6 model converted to ONNX (``squiddleocr.convert``); layout
analysers, text detectors and table recognisers are pluggable components behind small protocols
(``squiddleocr.layout``, ``squiddleocr.detectors``, ``squiddleocr.tables``); ``squiddleocr.pipeline``
wires them into a ``DoclingDocument`` that exports to DocLang, Markdown, HTML and JSON.
"""

__version__ = "0.2.0"

#: PaddleX registry names for the PP-OCRv6 recogniser family. ``inference.yml``
#: must carry one of these in ``Global.model_name``; PaddleX rejects unknown names.
PADDLE_MODEL_NAMES = {
    "tiny": "PP-OCRv6_tiny_rec",
    "small": "PP-OCRv6_small_rec",
    "medium": "PP-OCRv6_medium_rec",
}


def model_dir_name(variant: str) -> str:
    """Directory naming scheme of this project (not the model author's)."""
    return f"squiddle_PP-OCRv6_{variant}_rec"
