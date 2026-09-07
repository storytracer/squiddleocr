"""SquiddleOCR: package kraken PP-OCRv6 recognisers as PaddleOCR recognition models."""

__version__ = "0.1.0"

#: PaddleX registry names for the PP-OCRv6 recogniser family. ``inference.yml``
#: must carry one of these in ``Global.model_name``; PaddleX rejects unknown names.
PADDLE_MODEL_NAMES = {
    "tiny": "PP-OCRv6_tiny_rec",
    "small": "PP-OCRv6_small_rec",
    "medium": "PP-OCRv6_medium_rec",
}

#: Directory naming scheme of this project (not the model author's).
def model_dir_name(variant: str) -> str:
    return f"squiddle_PP-OCRv6_{variant}_rec"
