"""Assemble a ``DoclingDocument`` from regions, lines and recognised text, and export it."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from docling_core.types.doc import (BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, GroupLabel, ProvenanceItem,
                                    Size, TableCell, TableData)

from .types import BBox, Page, Recognition, Region, TableResult, TextLine

#: Layout labels that carry text; anything else becomes a picture (or a table when a TableResult exists).
TEXT_LABELS = {
    "text": DocItemLabel.TEXT,
    "paragraph": DocItemLabel.TEXT,
    "title": DocItemLabel.TITLE,
    "section_header": DocItemLabel.SECTION_HEADER,
    "caption": DocItemLabel.CAPTION,
    "footnote": DocItemLabel.FOOTNOTE,
    "page_header": DocItemLabel.PAGE_HEADER,
    "page_footer": DocItemLabel.PAGE_FOOTER,
    "list_item": DocItemLabel.LIST_ITEM,
    "formula": DocItemLabel.FORMULA,
    "code": DocItemLabel.CODE,
    "reference": DocItemLabel.REFERENCE,
    "document_index": DocItemLabel.DOCUMENT_INDEX,
}
EXPORT_FORMATS = ("doclang", "md", "html", "json", "txt")


@dataclass
class RegionContent:
    """Everything the pipeline produced for one region."""

    region: Region
    lines: list[TextLine] = field(default_factory=list)
    texts: list[Recognition] = field(default_factory=list)
    records: list = field(default_factory=list)      # kraken ``ocr_record`` per line (kraken level), else empty
    table: TableResult | None = None
    formula: str | None = None                         # LaTeX from a formula recogniser, else None

    @property
    def text(self) -> str:
        """Recognised text, one visual row per line (boxes on the same row are joined with a space)."""
        if len(self.lines) != len(self.texts):        # texts without geometry: one per line
            return "\n".join(r.text for r in self.texts)
        return "\n".join(row.text for row in self.rows())

    def rows(self) -> list:
        """The visual rows: text of the boxes on each row joined with a space, and the row's bounds
        (``reflow.Row``), in row order."""
        from .reflow import Row

        if len(self.lines) != len(self.texts):
            return [Row(r.text, self.region.bbox, i) for i, r in enumerate(self.texts)]
        grouped: dict[int, list] = {}
        for ln, r in zip(self.lines, self.texts):
            grouped.setdefault(ln.row, []).append((ln, r.text))
        out = []
        for i, k in enumerate(sorted(grouped)):
            boxes = [ln.bbox for ln, _ in grouped[k]]
            bbox = BBox(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))
            out.append(Row(" ".join(t for _, t in grouped[k] if t), bbox, i))
        return out


def _bbox(b: BBox) -> BoundingBox:
    return BoundingBox(l=b.x0, t=b.y0, r=b.x1, b=b.y1, coord_origin=CoordOrigin.TOPLEFT)


def _prov(page: Page, b: BBox, text: str = "") -> ProvenanceItem:
    return ProvenanceItem(page_no=page.number, bbox=_bbox(b), charspan=(0, len(text)))


def _table_data(table: TableResult) -> TableData:
    cells = [
        TableCell(text=c.text, start_row_offset_idx=c.row, end_row_offset_idx=c.row + c.row_span,
                  start_col_offset_idx=c.col, end_col_offset_idx=c.col + c.col_span,
                  row_span=c.row_span, col_span=c.col_span, column_header=c.header,
                  bbox=_bbox(c.bbox) if c.bbox else None)
        for c in table.cells
    ]
    return TableData(table_cells=cells, num_rows=table.num_rows, num_cols=table.num_cols)


class DocumentBuilder:
    """Builds one ``DoclingDocument`` per document; call ``add_page`` once per page, in order.

    With ``sections`` every heading opens a Docling ``section`` group (named after the heading) that
    holds it and everything that follows in reading order up to the next heading, across pages;
    items before the first heading stay in the body. Markdown and text read the same, JSON and
    DocLang (``<group label="section" name="...">``) gain the tree. This folds the layout
    analyser's reading order at its headings; it is not article detection.

    ``text="reflow"`` (default) runs ``reflow``: rows joined into paragraphs, typesetter's division
    marks removed, one text item per paragraph with a provenance entry per row, and a paragraph
    that runs on into the next text region (also on the next page) continued there.
    ``text="lines"`` keeps one visual row per line in each text item (a hard line break each in
    Markdown). ``rtl``
    mirrors the geometry for right-to-left pages. ``stats`` sums the reflow decisions.
    """

    TEXT_MODES = ("reflow", "lines")

    def __init__(self, name: str = "document", sections: bool = False, text: str = "reflow", rtl: bool = False):
        from .reflow import Lexicon, Stats

        if text not in self.TEXT_MODES:
            raise ValueError(f"Unknown text mode {text!r}; choose {', '.join(self.TEXT_MODES)}")
        self.doc = DoclingDocument(name=name)
        self.sections, self.text_mode, self.rtl = sections, text, rtl
        self._section = None       # the open GroupItem, or None for the body
        self.lexicon, self.stats = Lexicon(), Stats()
        self._open = None          # (TextItem, Paragraph, Margins) of a paragraph that may continue
        self._level = 1            # heading level of the region being added (Region.heading_level)

    def add_page(self, page: Page, contents: Sequence[RegionContent]) -> None:
        self.doc.add_page(page_no=page.number, size=Size(width=page.width, height=page.height))
        ordered = sorted(contents, key=lambda c: (c.region.order if c.region.order is not None else 1 << 30))
        if self.text_mode == "reflow":
            for c in ordered:
                self.lexicon.add(c.text)
        for c in ordered:
            b = c.region.bbox
            self._level = c.region.heading_level or 1
            if self.text_mode == "reflow" and c.formula is None and c.table is None and c.region.label in TEXT_LABELS:
                self._add_reflowed(page, c)
                continue
            self._open = None
            if c.formula is not None:
                if c.formula.strip():
                    self.doc.add_text(label=DocItemLabel.FORMULA, text=c.formula, prov=_prov(page, b, c.formula), parent=self._section)
            elif c.table is not None:
                self.doc.add_table(data=_table_data(c.table), prov=_prov(page, b), parent=self._section)
            elif c.region.label in TEXT_LABELS:
                self._add_text(page, b, TEXT_LABELS[c.region.label], c.text)
            else:  # picture-like region; text read inside it is kept as a child
                pic = self.doc.add_picture(prov=_prov(page, b), parent=self._section)
                if c.text.strip():
                    self.doc.add_text(label=DocItemLabel.TEXT, text=c.text, prov=_prov(page, b, c.text), parent=pic)

    def _add_text(self, page: Page, b: BBox, label: DocItemLabel, text: str) -> None:
        if not text.strip():
            return
        prov = _prov(page, b, text)
        if label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER) and self.sections:
            name = " ".join(text.split())
            self._section = self.doc.add_group(label=GroupLabel.SECTION, name=name[:80])
        if label == DocItemLabel.TITLE:
            self.doc.add_title(text=text, prov=prov, parent=self._section)
        elif label == DocItemLabel.SECTION_HEADER:
            self.doc.add_heading(text=text, prov=prov, parent=self._section, level=self._level)
        else:
            self.doc.add_text(label=label, text=text, prov=prov, parent=self._section)

    def _add_reflowed(self, page: Page, c: RegionContent) -> None:
        from .reflow import Margins, continues, join_paragraphs, reflow_rows

        label = TEXT_LABELS[c.region.label]
        rows = c.rows()
        single = label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER)
        paragraphs = reflow_rows(rows, self.rtl, self.lexicon, single, self.stats)
        if not paragraphs:
            return
        m = Margins.of([r for r in rows if r.text.strip()], self.rtl)
        if self._open is not None and label == DocItemLabel.TEXT and continues(self._open[1], rows[paragraphs[0].spans[0].row], m, self.rtl):
            item, prev = self._open[0], self._open[1]
            first = paragraphs.pop(0)
            joined = join_paragraphs(prev, first, self.lexicon, self.stats)
            offset = len(joined) - len(first.text)
            item.text = item.orig = joined
            item.prov.extend(_row_provs(page, rows, first, offset))
            prev.text, prev.open_end = joined, first.open_end
            self.stats.continuations += 1
            self.stats.paragraphs -= 1
            if not paragraphs and first.open_end:
                self._open = (item, prev, m)
                return
        self._open = None
        for para in paragraphs:
            provs = _row_provs(page, rows, para)
            if single:
                if label == DocItemLabel.TITLE:
                    item = self.doc.add_title(text=para.text, prov=provs[0], parent=self._section)
                else:
                    self._section = self.doc.add_group(label=GroupLabel.SECTION, name=" ".join(para.text.split())[:80]) if self.sections else self._section
                    item = self.doc.add_heading(text=para.text, prov=provs[0], parent=self._section, level=self._level)
                if self.sections and label == DocItemLabel.TITLE:
                    self._section = self.doc.add_group(label=GroupLabel.SECTION, name=" ".join(para.text.split())[:80])
                    item.parent = self._section.get_ref() if hasattr(self._section, "get_ref") else item.parent
            else:
                item = self.doc.add_text(label=label, text=para.text, prov=provs[0], parent=self._section)
            item.prov.extend(provs[1:])
            if para.open_end and label == DocItemLabel.TEXT:
                self._open = (item, para, m)

    def build(self) -> DoclingDocument:
        return self.doc


def _row_provs(page: Page, rows, para, offset: int = 0) -> list[ProvenanceItem]:
    by_index = {r.index: r for r in rows}
    return [ProvenanceItem(page_no=page.number, bbox=_bbox(by_index[s.row].bbox), charspan=(s.start + offset, s.end + offset))
            for s in para.spans if s.row in by_index]


def _span_aware_table_serializer():
    """Docling's pipe table for a plain grid; its HTML table (``rowspan``/``colspan``) when a cell spans.

    Docling's Markdown table repeats a spanning cell's text in every row and column it covers, which
    turns a rowspan into a column of duplicates. HTML tables keep the spans and render in Markdown
    viewers; it is also what PP-StructureV3 writes.
    """
    from docling_core.transforms.serializer.html import HTMLTableSerializer
    from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

    class SpanAwareTableSerializer(MarkdownTableSerializer):
        def serialize(self, *, item, doc_serializer, doc, **kwargs):
            if any(c.row_span > 1 or c.col_span > 1 for c in item.data.table_cells):
                return HTMLTableSerializer().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)

        def get_header_and_body_lines(self, *, table_text, **kwargs):
            if table_text.lstrip().startswith("<table"):
                return HTMLTableSerializer().get_header_and_body_lines(table_text=table_text, **kwargs)
            return super().get_header_and_body_lines(table_text=table_text, **kwargs)

    return SpanAwareTableSerializer()


def export_markdown(doc: DoclingDocument) -> str:
    """``doc`` as Markdown with Docling's defaults, except that tables with spanning cells are HTML tables."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

    return MarkdownDocSerializer(doc=doc, table_serializer=_span_aware_table_serializer()).serialize().text


def export(doc: DoclingDocument, out_dir: str | Path, stem: str, formats: Sequence[str] = ("md",)) -> list[Path]:
    """Write ``doc`` in the requested formats (``doclang``, ``md``, ``html``, ``json``, ``txt``); returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "doclang":
            path = out / f"{stem}.doclang.xml"
            path.write_text(doc.export_to_doclang(add_named_groups=True), encoding="utf-8")   # sections survive as <group>
        elif fmt == "md":
            path = out / f"{stem}.md"
            path.write_text(export_markdown(doc), encoding="utf-8")
        elif fmt == "html":
            path = out / f"{stem}.html"
            path.write_text(doc.export_to_html(), encoding="utf-8")
        elif fmt == "json":
            path = out / f"{stem}.json"
            doc.save_as_json(path)
        elif fmt == "txt":
            path = out / f"{stem}.txt"
            path.write_text(doc.export_to_text(), encoding="utf-8")
        else:
            raise ValueError(f"Unknown export format {fmt!r}; choose from {EXPORT_FORMATS}")
        written.append(path)
    return written
