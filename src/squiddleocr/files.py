"""``PageFiles``: one file per page and format from the page row, with linecaster's serialisers
(hOCR 1.2, ALTO 4.4, PAGE 2019) and the plain text (lines in reading order, as written, as laid out).

Files are named ``<page_id>.<ext>`` in the output folder: ``.hocr``, ``.alto.xml``, ``.page.xml``,
``.txt``, ``.json`` (the transcription as linecaster JSON). A candidate for linecaster's per-page
file writers.
"""
from __future__ import annotations

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep

from linecaster.blocks.base import page_of
from linecaster.transcription import to_alto, to_hocr, to_page

EXTENSIONS = {"hocr": "hocr", "alto": "alto.xml", "page": "page.xml", "txt": "txt", "json": "json"}


def render(t, fmt: str, width: int, height: int, uri: str | None) -> bytes:
    if fmt == "hocr":
        return to_hocr(t, image_width=width, image_height=height, image_uri=uri)
    if fmt == "alto":
        return to_alto(t, image_width=width, image_height=height, image_uri=uri)
    if fmt == "page":
        return to_page(t, image_width=width, image_height=height, image_uri=uri)
    if fmt == "txt":
        return (t.text + "\n").encode("utf-8") if t.text else b""
    if fmt == "json":
        return t.model_dump_json(indent=1).encode("utf-8")
    raise ValueError(f"unknown format {fmt!r}; known: {sorted(EXTENSIONS)}")


class PageFiles(PipelineStep):
    type = "💽 - WRITER"
    name = "🗂️ page files"

    def __init__(self, output_folder: DataFolderLike, formats=("hocr", "alto", "page", "txt"),
                 transcription: str = "ocr"):
        super().__init__()
        unknown = set(formats) - set(EXTENSIONS)
        if unknown:
            raise ValueError(f"unknown formats {sorted(unknown)}; known: {sorted(EXTENSIONS)}")
        self.output_folder, self.formats, self.transcription = output_folder, tuple(formats), transcription

    def run(self, data, rank: int = 0, world_size: int = 1):
        out = get_datafolder(self.output_folder)
        for doc in data:
            page = page_of(doc)
            t = page.transcriptions.get(self.transcription)
            if t is None:
                self.stat_update("no_transcription")
                yield doc
                continue
            uri = page.image.uri.rsplit("/", 1)[-1] if page.image and page.image.uri else None
            for fmt in self.formats:
                with self.track_time(fmt):
                    with out.open(f"{page.page_id}.{EXTENSIONS[fmt]}", "wb") as f:
                        f.write(render(t, fmt, page.image.width, page.image.height, uri))
                self.stat_update(fmt)
            yield doc
