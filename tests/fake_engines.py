"""CPU stand-ins for the squiddle chain's engine models (same interface as linecaster's engines), so the CLI
runs end to end in tests: a title and a body region, three detected lines, fixed text."""
from __future__ import annotations

from linecaster.data import Line, ModelInfo, Region, Word


def _box(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class _Fake:
    step = "fake"

    def __init__(self, **kwargs):
        self.name, self.kwargs = f"fake-{self.step}", kwargs

    def load(self) -> None:
        pass

    def __call__(self, image, t, **params):
        self.run(image, t)
        t.model.append(ModelInfo(step=self.step, name=self.name, settings={k: str(v) for k, v in self.kwargs.items()}))
        return t

    def run(self, image, t):
        pass


class Layout(_Fake):
    step = "layout"

    def run(self, image, t):
        t.regions = [Region(id="doc_title_0", label="title", raw_label="doc_title", polygon=_box(20, 20, 300, 60),
                            bbox=[20, 20, 300, 60], score=0.9, reading_order=0),
                     Region(id="text_1", label="text", raw_label="text", polygon=_box(20, 100, 300, 100),
                            bbox=[20, 100, 300, 100], score=0.9, reading_order=1)]


class Lines(_Fake):
    step = "lines"

    def run(self, image, t):
        boxes = [(25, 30, 280, 40), (25, 110, 280, 20), (25, 140, 280, 20)]
        t.lines = [Line(id=f"det_{i}", reading_order=i, polygon=_box(*b), bbox=list(b), score=0.9) for i, b in enumerate(boxes)]


TEXTS = {"doc_title_0_l0": "DE NUMERIS", "text_1_l0": "Weſen der Din-", "text_1_l1": "ge und Zahlen."}


class Recognition(_Fake):
    step = "recognition"

    def run(self, image, t):
        for ln in t.lines:
            ln.text = TEXTS.get(ln.id, "x")
            x, y, w, h = ln.bbox
            ln.confidence = 0.95
            parts = ln.text.split()
            ln.words = [Word(id=f"{ln.id}_w{i}", text=p, bbox=[x + i * w / len(parts), y, w / len(parts), h], confidence=0.95)
                        for i, p in enumerate(parts)]


class FailsOnBlackPages(Recognition):
    """Recognition that fails on an all-black page (a page a block fails on)."""

    def run(self, image, t):
        if image.max() == 0:
            raise RuntimeError("nothing to read on a black page")
        super().run(image, t)


class Tables(_Fake):
    step = "tables"


class Formulas(_Fake):
    step = "formulas"


MODELS = [{"name": n, "spec": f"fake_engines:{c}", "kwargs": {}}
          for n, c in (("layout", "Layout"), ("lines", "Lines"), ("recognition", "Recognition"), ("tables", "Tables"),
                       ("formulas", "Formulas"))]
