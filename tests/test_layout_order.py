import numpy as np

from squiddleocr.layout.order import suppress_contained, xy_cut_order
from squiddleocr.pipeline import dedupe_lines, order_lines
from squiddleocr.types import BBox, Region, TextLine


def test_xy_cut_two_columns_under_a_title():
    boxes = [BBox(0, 0, 100, 10),      # title across the page
             BBox(55, 20, 100, 100),   # right column
             BBox(0, 20, 45, 100),     # left column
             BBox(0, 110, 100, 120)]   # footer
    assert xy_cut_order(boxes) == [0, 2, 1, 3]


def test_suppress_contained_keeps_the_larger_region():
    big = Region("text", BBox(0, 0, 100, 100).polygon, 0.5, None, "big")
    inner = Region("text", BBox(10, 10, 50, 50).polygon, 0.9, None, "inner")
    apart = Region("text", BBox(200, 0, 300, 100).polygon, 0.9, None, "apart")
    kept = suppress_contained([inner, big, apart])
    assert {r.id for r in kept} == {"big", "apart"}


def test_dedupe_and_order_lines():
    a = TextLine(BBox(0, 0, 100, 10).polygon)
    frag = TextLine(BBox(2, 1, 20, 9).polygon)          # fragment inside a
    b = TextLine(BBox(0, 20, 50, 30).polygon)
    c = TextLine(BBox(60, 21, 100, 31).polygon)          # same row as b, to the right
    assert dedupe_lines([frag, a, b, c]) == [a, b, c]
    ordered = order_lines([c, frag, b, a])
    assert ordered == [a, b, c] and [ln.row for ln in ordered] == [0, 1, 1]
