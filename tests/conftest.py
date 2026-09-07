import numpy as np
import pytest

from squiddleocr.types import BBox, Page, TextLine


@pytest.fixture
def page():
    rng = np.random.default_rng(0)
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    img[40:70, 20:380] = rng.integers(0, 80, (30, 360, 3), dtype=np.uint8)
    img[120:150, 20:200] = rng.integers(0, 80, (30, 180, 3), dtype=np.uint8)
    return Page(img, None, 1)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, page, region=None):
        return [TextLine(BBox(*b).polygon, 0.9, region_id=region.id if region else "") for b in self.boxes]


class FakeRecord:
    """Looks like a kraken ``ocr_record`` to the pipeline and the document builder."""

    def __init__(self, text, cuts=()):
        self.prediction, self.confidences, self.cuts, self.type = text, [0.5, 1.0], list(cuts), "bbox"


class FakeRecognizer:
    """Names each line by its region and size, like a recogniser that reads lines off the page."""

    def recognize_lines(self, page, lines):
        return [FakeRecord(f"{ln.region_id}:{int(ln.bbox.width)}x{int(ln.bbox.height)}") for ln in lines]


@pytest.fixture
def fake_detector():
    return FakeDetector


@pytest.fixture
def fake_recognizer():
    return FakeRecognizer()
