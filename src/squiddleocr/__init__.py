"""SquiddleOCR: PaddleX layout, detection and tables in front of kraken's PP-OCRv6 recogniser.

Layout analysers, text detectors and table recognisers are pluggable components behind small
protocols (``squiddleocr.layout``, ``squiddleocr.detectors``, ``squiddleocr.tables``); recognition
is kraken's own (``squiddleocr.recognizers.kraken``); ``squiddleocr.pipeline`` wires them into a
``DoclingDocument`` (DocLang, Markdown, HTML, JSON) and ``squiddleocr.serialize`` hands the same
results to kraken's serialiser (hOCR, ALTO, PAGE-XML).
"""

__version__ = "0.3.0"
