"""SquiddleOCR's catalogue: its recipe ``squiddle`` (linecaster's chain with SquiddleOCR's presets and options)
and that recipe's evidence. Registered in the entry-point group ``linecaster.catalog``."""
from pathlib import Path


def directory() -> Path:
    return Path(__file__).resolve().parent
