"""SquiddleOCR: the squiddle recipe as a linecaster chain, with its command line interface.

PaddleX layout, PP-OCRv6 line detection, tables and formulas in front of kraken's PP-OCRv6 recogniser, then
linecaster's typography, recomposition and composition into a DoclingDocument (``squiddleocr.chain``); the
``squiddle`` command (``squiddleocr.cli``) runs it on image files.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("squiddleocr")
except PackageNotFoundError:        # a source tree on sys.path without an installed distribution
    __version__ = "0.0.0+unknown"
