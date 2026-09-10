"""Reflow: running text from the visual rows of a region (the reading transcription).

Typesetting turned the text into lines: it broke it at line ends, sometimes inside a word with a
division mark, justified it, indented paragraph starts and continued it across columns and pages.
The line-level exports keep that diplomatic form. This module inverts it for the document exports:
rows of a region are joined into paragraphs, division marks that the typesetter added are removed,
and a paragraph that runs on into the next region is continued there. Every decision is made from
Unicode character properties and geometry, never from a language name, plus the document's own
words as evidence for the one ambiguous case (a hyphen that may belong to the word).

Decisions at a row end, in order:

- **Paragraph boundary** (``paragraph_break``): the row starts indented from the region's start
  margin by about an em, or the gap above it exceeds the modal leading clearly, or in justified
  text the previous row stopped short of the end margin. Start and end margins follow the writing
  direction (``rtl``).
- **Division mark** (``resolve_mark``): a hyphen by Unicode's Line_Break property (HY, or a BA/HH
  character whose name says HYPHEN: hyphen, soft hyphen, the Fraktur ``⸗``, ``⹀``, the Armenian
  hyphen; the maqaf is not named so and dashes are other classes) or a convention mark, after a
  letter at the row end, is the typesetter's and removed, unless the next row starts with an
  uppercase letter or a digit (a compound split at its own hyphen) or the document attests the
  hyphenated form and not the joined one. After a Hebrew letter a mark divides a word only when the
  next row is Hebrew script too (UAX #14 LB21a: Hebrew does not divide words; Yiddish print does).
- **Separator** (``separator``): nothing when UAX #14 permits a line break between the two
  boundary words without a space (ideographs, Thai, after full-width punctuation), else a space.
- **Region continuation** (``continues``): a text region whose last row ends without sentence-final
  punctuation, at the end margin or with a division mark, continues into the next text region in
  reading order when that region's first row is not indented and starts with a letter (a capital
  too, German capitalises every noun, except after a division mark).

Out of scope, by design: glyph normalisation (long s, ligatures), spelling modernisation, emphasis
from letter-spacing, Korean's unmarked in-word breaks, vertical writing.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from uniseg.linebreak import LB, line_break, line_break_boundaries
from uniseg.sentencebreak import SB, sentence_break
from uniseg.wordbreak import words

from .types import BBox

#: Marks by convention, unknown to Unicode as hyphens: the not sign, which some transcription
#: conventions (and kraken models trained on them) write for a line-end hyphen.
CONVENTION_MARKS = "¬"


def is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def is_mark(ch: str) -> bool:
    """A character the typesetter may put at a line end when dividing a word."""
    if ch in CONVENTION_MARKS:
        return True
    lb = line_break(ch)
    return lb == LB.HY or (lb.name in ("BA", "HH", "GL") and "HYPHEN" in unicodedata.name(ch, ""))


def _hebrew(ch: str) -> bool:
    return line_break(ch) == LB.HL


def word_tokens(text: str) -> list[str]:
    """UAX #29 words that contain a letter, hyphenated compounds kept together (``EU-Staaten``)."""
    out: list[str] = []
    for w in words(text):
        if out and is_mark(w[-1:]) and len(w) == 1 and out[-1] and is_letter(out[-1][-1]):
            out[-1] += w                                    # a hyphen right after a word
            continue
        if out and is_mark(out[-1][-1:]) and len(out[-1]) > 1 and any(is_letter(c) for c in w):
            out[-1] += w                                    # the word after that hyphen
            continue
        if any(is_letter(c) for c in w):
            out.append(w)
    return [t for t in (_strip_marks(w) for w in out) if t]


def _strip_marks(w: str) -> str:
    while w and is_mark(w[-1]):
        w = w[:-1]
    return w


@dataclass
class Row:
    """One visual row of a region: its text (boxes on the row already joined) and its bounds."""

    text: str
    bbox: BBox
    index: int = 0        # position in the region's rows


@dataclass
class Span:
    row: int              # row index in the region
    start: int            # character span in the paragraph text
    end: int


@dataclass
class Paragraph:
    text: str
    spans: list[Span] = field(default_factory=list)
    bbox: BBox | None = None
    open_end: bool = False       # may continue into the next region


@dataclass
class Stats:
    marks_removed: int = 0
    marks_kept: int = 0
    paragraphs: int = 0
    continuations: int = 0
    examples: list[str] = field(default_factory=list)     # kept marks and continuations, for SQUIDDLE_VERBOSE

    def note(self, kind: str, before: str, after: str) -> None:
        if len(self.examples) < 200:
            self.examples.append(f"{kind}: …{before[-30:]} | {after[:30]}…")

    def __iadd__(self, other: "Stats") -> "Stats":
        self.marks_removed += other.marks_removed
        self.marks_kept += other.marks_kept
        self.paragraphs += other.paragraphs
        self.continuations += other.continuations
        self.examples = (self.examples + other.examples)[:200]
        return self

    def describe(self) -> str:
        return (f"{self.paragraphs} paragraphs, {self.continuations} continued across regions, "
                f"division marks removed {self.marks_removed}, kept {self.marks_kept}")


class Lexicon:
    """Word counts over everything recognised so far in a run: the document as its own dictionary."""

    def __init__(self):
        self.counts: Counter[str] = Counter()

    def add(self, text: str) -> None:
        for line in text.splitlines():      # line by line: a line-end hyphen must not attest a compound
            self.counts.update(w.casefold() for w in word_tokens(line))

    def attests(self, word: str) -> int:
        return self.counts.get(word.casefold(), 0)


# ------------------------------------------------------------------------------------------ geometry
@dataclass
class Margins:
    """The region's writing frame from its rows: start and end margins, em, leading, justification."""

    start: float          # x of the start margin (left for LTR, right for RTL)
    end: float
    em: float             # median row height
    leading: float        # median distance between consecutive row tops
    justified: bool

    @classmethod
    def of(cls, rows: Sequence[Row], rtl: bool = False) -> "Margins":
        heights = sorted(r.bbox.height for r in rows)
        em = heights[len(heights) // 2] if heights else 1.0
        starts = sorted((r.bbox.x1 if rtl else r.bbox.x0) for r in rows)
        ends = sorted((r.bbox.x0 if rtl else r.bbox.x1) for r in rows)
        # the start margin is the outermost start (rows are indented inwards, never outwards)
        start = starts[-1] if rtl else starts[0]
        end = ends[0] if rtl else ends[-1]
        tops = sorted(r.bbox.y0 for r in rows)
        gaps = sorted(b - a for a, b in zip(tops, tops[1:]) if b > a)
        leading = gaps[len(gaps) // 2] if gaps else em
        reaching = sum(1 for r in rows if abs((r.bbox.x0 if rtl else r.bbox.x1) - end) <= 0.6 * em)
        return cls(start, end, max(em, 1.0), max(leading, 1.0), justified=len(rows) >= 3 and reaching >= 0.6 * len(rows))

    def indent(self, row: Row, rtl: bool) -> float:
        return (self.start - row.bbox.x1) if rtl else (row.bbox.x0 - self.start)

    def shortfall(self, row: Row, rtl: bool) -> float:
        return (row.bbox.x0 - self.end) if rtl else (self.end - row.bbox.x1)


def paragraph_break(prev: Row, row: Row, m: Margins, rtl: bool = False) -> bool:
    """Does ``row`` start a new paragraph after ``prev``?"""
    if m.indent(row, rtl) >= 0.8 * m.em and m.indent(prev, rtl) < 0.8 * m.em:
        return True
    gap = row.bbox.y0 - prev.bbox.y1                    # blank space between the rows
    if gap > 0.6 * m.leading and gap > 0.5 * max(prev.bbox.height, row.bbox.height):   # about a blank line
        return True
    return m.justified and m.shortfall(prev, rtl) > 1.5 * m.em and m.indent(row, rtl) < 0.8 * m.em and m.shortfall(prev, rtl) > 0.25 * (m.end - m.start if not rtl else m.start - m.end)


# ---------------------------------------------------------------------------------------- characters
def _last_char(text: str) -> str:
    t = text.rstrip()
    return t[-1] if t else ""


def _first_char(text: str) -> str:
    t = text.lstrip()
    return t[0] if t else ""


def ends_with_mark(text: str) -> bool:
    t = text.rstrip()
    return len(t) >= 2 and is_mark(t[-1]) and is_letter(t[-2])


def resolve_mark(prev_text: str, next_text: str, lexicon: Lexicon | None = None) -> tuple[bool, str]:
    """For a row ending in a division mark: (removed?, the attested hyphenated form or "").

    Removed unless the next row starts with an uppercase letter or a digit, the mark follows a
    Hebrew letter and the next row is not Hebrew script, or the lexicon attests the hyphenated
    form and not the joined form."""
    nxt = _first_char(next_text)
    if not nxt or not (is_letter(nxt) or nxt.isdigit()):
        return False, ""
    if nxt.isupper() or nxt.isdigit():
        return False, ""
    stripped = prev_text.rstrip()
    if _hebrew(stripped[-2]) and not _hebrew(nxt):
        return False, ""
    a = word_tokens(stripped[:-1])
    b = word_tokens(next_text)
    if lexicon is not None and a and b:
        head, tail = a[-1], b[0]
        joined, hyphenated = head + tail, f"{head}-{tail}"
        if lexicon.attests(hyphenated) and not lexicon.attests(joined):
            return False, hyphenated
    return True, ""


def separator(prev_text: str, next_text: str) -> str:
    """"" when UAX #14 allows a line break right at the junction of the boundary words, else a space."""
    a = prev_text.rstrip().split()[-1:] or [""]
    b = next_text.lstrip().split()[:1] or [""]
    a, b = a[0], b[0]
    if not a or not b:
        return " "
    if line_break(a[-1]) == LB.SA and line_break(b[0]) == LB.SA:
        return ""      # Thai, Lao, Khmer, Myanmar: UAX #14 leaves breaks inside runs to a dictionary; no spaces in any case
    return "" if len(a) in set(line_break_boundaries(a + b)) else " "


def sentence_open(text: str) -> bool:
    """The row does not end a sentence: no sentence-terminal punctuation (closing quotes and brackets
    may follow it), by Unicode's Sentence_Break property."""
    t = text.rstrip()
    while t and sentence_break(t[-1]) == SB.CLOSE:
        t = t[:-1].rstrip()
    return bool(t) and sentence_break(t[-1]) not in (SB.STERM, SB.ATERM)


def continues(prev: Paragraph, first_row: Row, m: Margins, rtl: bool = False) -> bool:
    """Whether the last paragraph of a region runs on into a region whose first row is ``first_row``."""
    if not prev.open_end:
        return False
    if m.indent(first_row, rtl) >= 0.8 * m.em:
        return False
    ch = _first_char(first_row.text)
    if not ch or not is_letter(ch):
        return False
    # after a division mark a capital would make a compound; across regions that is more often a wrong
    # pairing of regions than a compound, so it is refused. Without a mark a capital is allowed: German
    # capitalises every noun.
    return not (ch.isupper() and ends_with_mark(prev.text))


# ---------------------------------------------------------------------------------------------- rows
def reflow_rows(rows: Sequence[Row], rtl: bool = False, lexicon: Lexicon | None = None, single: bool = False,
                stats: Stats | None = None) -> list[Paragraph]:
    """Join the rows of one region into paragraphs. ``single`` (headings) makes one paragraph."""
    stats = stats if stats is not None else Stats()
    rows = [r for r in rows if r.text.strip()]
    if not rows:
        return []
    m = Margins.of(rows, rtl)
    paragraphs: list[Paragraph] = []
    current: Paragraph | None = None
    for i, row in enumerate(rows):
        text = row.text.strip()
        if current is None or (not single and paragraph_break(rows[i - 1], row, m, rtl)):
            if current is not None:
                _close(current, rows)
            current = Paragraph(text, [Span(row.index, 0, len(text))])
            paragraphs.append(current)
            continue
        prev_text = current.text
        if ends_with_mark(prev_text):
            removed, hyphenated = resolve_mark(prev_text, text, lexicon)
            if removed:
                current.text = prev_text[:-1]
                current.spans[-1].end -= 1
                sep = ""
                stats.marks_removed += 1
            else:
                sep = "" if hyphenated or is_mark(prev_text[-1]) else " "
                stats.marks_kept += 1
                stats.note("kept", prev_text, text)
        else:
            sep = separator(prev_text, text)
        start = len(current.text) + len(sep)
        current.text = current.text + sep + text
        current.spans.append(Span(row.index, start, len(current.text)))
    _close(current, rows)
    last, last_row = paragraphs[-1], rows[-1]
    # a single row is its own margin, so only a division mark can leave it open
    last.open_end = (not single and sentence_open(last.text)
                     and (ends_with_mark(last.text) or (len(rows) >= 2 and m.shortfall(last_row, rtl) <= 0.6 * m.em)))
    stats.paragraphs += len(paragraphs)
    return paragraphs


def _close(p: Paragraph, rows: Sequence[Row]) -> None:
    by_index = {r.index: r for r in rows}
    boxes = [by_index[s.row].bbox for s in p.spans if s.row in by_index]
    if boxes:
        p.bbox = BBox(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def join_paragraphs(prev: Paragraph, nxt: Paragraph, lexicon: Lexicon | None = None, stats: Stats | None = None) -> str:
    """The separator (or mark removal) for continuing ``prev`` with ``nxt``; returns the joined text."""
    stats = stats if stats is not None else Stats()
    if ends_with_mark(prev.text):
        removed, hyphenated = resolve_mark(prev.text, nxt.text, lexicon)
        if removed:
            stats.marks_removed += 1
            return prev.text[:-1] + nxt.text
        stats.marks_kept += 1
        stats.note("kept", prev.text, nxt.text)
        return prev.text + ("" if hyphenated else " ") + nxt.text
    stats.note("continued", prev.text, nxt.text)
    return prev.text + separator(prev.text, nxt.text) + nxt.text
