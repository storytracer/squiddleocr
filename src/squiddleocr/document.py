"""Assemble a ``DoclingDocument`` from regions, lines and recognised text, and export it."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from docling_core.types.doc import (BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, ProvenanceItem,
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
        rows: dict[int, list[str]] = {}
        for ln, r in zip(self.lines, self.texts):
            rows.setdefault(ln.row, []).append(r.text)
        return "\n".join(" ".join(t for t in rows[k] if t) for k in sorted(rows))


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
    """Builds one ``DoclingDocument`` per document; call ``add_page`` once per page, in order."""

    def __init__(self, name: str = "document"):
        self.doc = DoclingDocument(name=name)

    def add_page(self, page: Page, contents: Sequence[RegionContent]) -> None:
        self.doc.add_page(page_no=page.number, size=Size(width=page.width, height=page.height))
        ordered = sorted(contents, key=lambda c: (c.region.order if c.region.order is not None else 1 << 30))
        for c in ordered:
            b = c.region.bbox
            if c.formula is not None:
                if c.formula.strip():
                    self.doc.add_text(label=DocItemLabel.FORMULA, text=c.formula, prov=_prov(page, b, c.formula))
            elif c.table is not None:
                self.doc.add_table(data=_table_data(c.table), prov=_prov(page, b))
            elif c.region.label in TEXT_LABELS:
                self._add_text(page, b, TEXT_LABELS[c.region.label], c.text)
            else:  # picture-like region; text read inside it is kept as a child
                pic = self.doc.add_picture(prov=_prov(page, b))
                if c.text.strip():
                    self.doc.add_text(label=DocItemLabel.TEXT, text=c.text, prov=_prov(page, b, c.text), parent=pic)

    def _add_text(self, page: Page, b: BBox, label: DocItemLabel, text: str) -> None:
        if not text.strip():
            return
        prov = _prov(page, b, text)
        if label == DocItemLabel.TITLE:
            self.doc.add_title(text=text, prov=prov)
        elif label == DocItemLabel.SECTION_HEADER:
            self.doc.add_heading(text=text, prov=prov)
        else:
            self.doc.add_text(label=label, text=text, prov=prov)

    def build(self) -> DoclingDocument:
        return self.doc


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
            path.write_text(doc.export_to_doclang(), encoding="utf-8")
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
