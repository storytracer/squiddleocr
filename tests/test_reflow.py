"""Reflow rules on synthetic rows, and the document builder's reflow mode."""
import numpy as np
import pytest

from squiddleocr.document import DocumentBuilder, RegionContent
from squiddleocr.reflow import (Lexicon, Margins, Paragraph, Row, Stats, continues, is_mark, paragraph_break, reflow_rows,
                                resolve_mark, sentence_open, separator, word_tokens)
from squiddleocr.types import BBox, Page, Recognition, Region, TextLine

EM = 20


def rows(texts, x0=100, x1=500, y0=100, indent=(), short=(), gap_before=(), h=EM, lead=24):
    """Rows of a justified column; ``indent`` rows start an em in, ``short`` rows end early, ``gap_before`` rows sit a
    blank line lower."""
    out, y = [], y0
    for i, t in enumerate(texts):
        if i in gap_before:
            y += lead
        out.append(Row(t, BBox(x0 + (EM if i in indent else 0), y, x1 - (6 * EM if i in short else 0), y + h), i))
        y += lead
    return out


def test_marks_and_separators():
    assert all(is_mark(c) for c in "-‐‑\xad⸗⹀֊¬") and not any(is_mark(c) for c in "־–—a.")
    assert resolve_mark("Wan-", "derer mit") == (True, "")
    assert resolve_mark("מאָס⸗", "געבנדיקע") == (True, "") and resolve_mark("מאָס⸗", "geben") == (False, "")   # Yiddish divides, Hebrew does not
    assert resolve_mark("uͤber⸗", "haͤngt") == (True, "")
    assert resolve_mark("Peatoime¬", "taja") == (True, "")
    assert resolve_mark("Nord-", "Ost") == (False, "")            # uppercase continuation: the author's hyphen
    assert resolve_mark("Do-", "24") == (False, "")
    assert resolve_mark("word-", "") == (False, "")
    lex = Lexicon()
    lex.add("die EU-Staaten und die EU-Staaten")
    assert resolve_mark("EU-", "staaten", lex) == (False, "EU-staaten")
    lex.add("Landschaften")
    assert resolve_mark("Land-", "schaften", lex) == (True, "")
    lex.add("erster Absatz mit Bin-\ndung und Ende.")
    assert lex.attests("bin-dung") == 0 and lex.attests("bin") == 1                     # no self-attestation across a line break
    assert separator("und", "die") == " " and separator("東京", "都") == "" and separator("OCR", "技術") == ""
    assert separator("word,", "next") == " " and separator("これは。", "次") == "" and separator("ไทย", "ไทย") == ""
    assert word_tokens("die EU-Staaten und Nord-Ost, 東京 ab.") == ["die", "EU-Staaten", "und", "Nord-Ost", "東", "京", "ab"]
    assert sentence_open("kein Ende") and sentence_open("ein Ende,") and not sentence_open("Ende.") and not sentence_open("Ende!“)")
    assert not sentence_open("終わり。") and not sentence_open("סוף׃") is False


def test_reflow_joins_removes_marks_and_keeps_compounds():
    stats = Stats()
    paras = reflow_rows(rows(["theils mit maͤchtigen Felſen⸗", "ſtuͤcken uͤberhaͤngt. Heut zu", "Tage wuͤrde der Nord-", "Oſten ſolcher"]),
                        stats=stats)
    assert len(paras) == 1
    assert paras[0].text == "theils mit maͤchtigen Felſenſtuͤcken uͤberhaͤngt. Heut zu Tage wuͤrde der Nord-Oſten ſolcher"
    assert stats.marks_removed == 1 and stats.marks_kept == 1
    text = paras[0].text
    assert [s.row for s in paras[0].spans] == [0, 1, 2, 3]
    assert [text[s.start:s.end] for s in paras[0].spans] == ["theils mit maͤchtigen Felſen", "ſtuͤcken uͤberhaͤngt. Heut zu",
                                                            "Tage wuͤrde der Nord-", "Oſten ſolcher"]


def test_paragraph_breaks_from_indent_short_line_and_gap():
    r = rows(["aaa bbb ccc", "ddd eee fff", "ggg hhh.", "iii jjj kkk", "lll mmm nnn", "ooo ppp", "qqq rrr"],
             indent=(3,), short=(2, 4), gap_before=(6,))
    m = Margins.of(r)
    assert m.justified and abs(m.em - EM) < 1 and m.start == 100 and m.end == 500
    breaks = [i for i in range(1, len(r)) if paragraph_break(r[i - 1], r[i], m)]
    assert breaks == [3, 5, 6]          # indent after a short line, plain short line, blank line
    paras = reflow_rows(r)
    assert [p.text for p in paras] == ["aaa bbb ccc ddd eee fff ggg hhh.", "iii jjj kkk lll mmm nnn", "ooo ppp", "qqq rrr"]
    assert paras[-1].open_end                       # no full stop, and the row reaches the margin
    assert not paras[0].open_end


def test_centred_subtitle_and_ragged_text_do_not_break():
    r = rows(["Hui jäätise müïjast saab Nobile ja", "väikesest riigiametnikust minister."], h=40, lead=48)
    assert [p.text for p in reflow_rows(r)] == ["Hui jäätise müïjast saab Nobile ja väikesest riigiametnikust minister."]
    ragged = [Row(t, BBox(100, 100 + 24 * i, 500 - 30 * (i % 3), 120 + 24 * i), i) for i, t in enumerate(["a b", "c d", "e f", "g h"])]
    assert len(reflow_rows(ragged)) == 1              # not justified: short rows mean nothing


def test_rtl_mirrors_indent_and_shortfall():
    r = [Row("א ב ג", BBox(100, 100, 500, 120), 0), Row("ד ה", BBox(100, 124, 500, 144), 1), Row("ו ז ח", BBox(100, 148, 480, 168), 2)]
    m = Margins.of(r, rtl=True)
    assert m.start == 500 and m.end == 100 and m.indent(r[2], True) == 20
    assert paragraph_break(r[1], r[2], m, rtl=True) and not paragraph_break(r[0], r[1], m, rtl=True)


def test_single_paragraph_regions_and_continuation():
    heading = reflow_rows(rows(["Moodsa ôhuhiiglase", "välimus ja sisemus."], indent=(1,)), single=True)
    assert [p.text for p in heading] == ["Moodsa ôhuhiiglase välimus ja sisemus."] and not heading[0].open_end
    tail = reflow_rows(rows(["der Cavalier bald mit einem", "bald mit dem anderen ſeiner"]))[0]
    head = rows(["Diener, ohne Zweifel, weil", "die Theilnahme."])
    assert tail.open_end and continues(tail, head[0], Margins.of(head))
    assert not continues(tail, rows(["Diener, ohne Zweifel", "x"], indent=(0,))[0], Margins.of(head))   # indented start
    assert continues(tail, rows(["Die Theilnahme."])[0], Margins.of(head))                              # a German noun may start it
    assert not continues(tail, rows(["— Die Theilnahme."])[0], Margins.of(head))                        # not a letter
    closed = reflow_rows(rows(["bald mit dem anderen ſeiner."]))[0]
    assert not closed.open_end
    hy = reflow_rows(rows(["ſolchen romantiſchen Wan-"]))[0]
    assert hy.open_end and continues(hy, rows(["derer mit"])[0], Margins.of(head))
    assert not continues(hy, rows(["Derer mit"])[0], Margins.of(head))                                 # mark + capital: refused
    assert not reflow_rows(rows(["Ein einzelner Satz ohne Punkt"]))[0].open_end                       # one row is its own margin


@pytest.fixture
def two_pages():
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    return Page(img, None, 1), Page(img, None, 2)


def content(label, rid, order, texts, x0=20, y0=20, lead=24, indent=(), short=()):
    c = RegionContent(Region(label, BBox(x0, y0, x0 + 400, y0 + lead * len(texts)).polygon, 1.0, order, rid))
    for i, t in enumerate(texts):
        ln = TextLine(BBox(x0 + (20 if i in indent else 0), y0 + lead * i, x0 + 400 - (120 if i in short else 0), y0 + lead * i + 20).polygon)
        ln.row = i
        c.lines.append(ln)
        c.texts.append(Recognition(t, 0.9))
    return c


def test_builder_reflow_makes_paragraph_items_with_row_provenance(two_pages):
    p1, p2 = two_pages
    b = DocumentBuilder("t", text="reflow")
    b.add_page(p1, [content("section_header", "h", 0, ["Ein Titel", "auf zwei Zeilen"]),
                    content("text", "a", 1, ["erster Absatz mit Bin-", "dung und Ende.", "zweiter Absatz ohne", "Ende und weiter"], indent=(2,), short=(1,)),
                    content("text", "b", 2, ["geht es hier. Schluss."])])
    b.add_page(p2, [content("text", "c", 0, ["Neue Seite."])])
    doc = b.build()
    texts = [t.text for t in doc.texts]
    assert texts == ["Ein Titel auf zwei Zeilen", "erster Absatz mit Bindung und Ende.", "zweiter Absatz ohne Ende und weiter geht es hier. Schluss.",
                     "Neue Seite."]
    joined = doc.texts[2]
    assert [pv.page_no for pv in joined.prov] == [1, 1, 1] and len(joined.prov) == 3
    assert [tuple(pv.charspan) for pv in joined.prov] == [(0, 19), (20, 35), (36, 58)]
    assert joined.text[36:58] == "geht es hier. Schluss."
    assert b.stats.paragraphs == 4 and b.stats.continuations == 1 and b.stats.marks_removed == 1
    md = doc.export_to_markdown()
    assert "Bindung" in md and "\n\n" in md and "  \n" not in md
    with pytest.raises(ValueError):
        DocumentBuilder("t", text="words")


def test_lines_mode_is_unchanged(two_pages):
    p1, _ = two_pages
    b = DocumentBuilder("t")
    b.add_page(p1, [content("text", "a", 0, ["erste Zeile", "zweite Zeile"])])
    assert b.build().texts[0].text == "erste Zeile\nzweite Zeile"
