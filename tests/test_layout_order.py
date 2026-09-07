
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


def test_regions_from_boxes_uses_learned_order_and_polygons():
    from squiddleocr.layout.paddle import regions_from_boxes

    boxes = [{"label": "number", "score": 0.7, "coordinate": [90, 2, 100, 8], "order": 1,
              "polygon_points": [[90, 2], [100, 2], [100, 8], [90, 8]]},
             {"label": "table", "score": 0.9, "coordinate": [0, 10, 100, 50], "order": None,
              "polygon_points": [[0, 10], [100, 12], [100, 50], [0, 48]]},
             {"label": "display_formula", "score": 0.8, "coordinate": [0, 60, 100, 70], "order": 2,
              "polygon_points": [[0, 60], [100, 60], [100, 70], [0, 70]]}]
    regions, learned = regions_from_boxes(boxes, page_height=100)
    assert learned and [r.order for r in regions] == [0, 1, 2]
    assert [r.label for r in regions] == ["page_header", "table", "formula"]
    assert regions[1].polygon.tolist() == [[0, 10], [100, 12], [100, 50], [0, 48]]

    plain = [{"label": "text", "score": 0.9, "coordinate": [0, 0, 10, 10]}]
    regions, learned = regions_from_boxes(plain, page_height=100)
    assert not learned and regions[0].order is None and regions[0].bbox == BBox(0, 0, 10, 10)


def test_reorder_slots_orphans_into_learned_order():
    from squiddleocr.pipeline import Pipeline, RegionContent

    body = RegionContent(Region("text", BBox(0, 20, 100, 60).polygon, 1.0, 0, "body"))
    footnote = RegionContent(Region("footnote", BBox(0, 80, 100, 95).polygon, 1.0, 1, "fn"))
    page_no = RegionContent(Region("text", BBox(90, 2, 100, 8).polygon, 1.0, None, "orphan_0"))
    missed = RegionContent(Region("text", BBox(0, 62, 100, 70).polygon, 1.0, None, "orphan_1"))
    contents = [body, footnote, page_no, missed]
    Pipeline._reorder(contents)
    assert sorted(contents, key=lambda c: c.region.order) == [page_no, body, missed, footnote]


def test_reorder_falls_back_to_xy_cut_without_layout_order():
    from squiddleocr.pipeline import Pipeline, RegionContent

    a = RegionContent(Region("text", BBox(0, 50, 100, 60).polygon, 1.0, None, "a"))
    b = RegionContent(Region("text", BBox(0, 0, 100, 10).polygon, 1.0, None, "b"))
    Pipeline._reorder([a, b])
    assert (b.region.order, a.region.order) == (0, 1)
