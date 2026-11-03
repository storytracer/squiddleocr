"""``squiddle``: the squiddle pipeline (a linecaster chain) on image files.

    squiddle ocr IMAGES... -o OUT                     # in-process: this environment's GPU
    squiddle ocr IMAGES... -o OUT --backend served    # through `linecaster serve` (default :8100)
    squiddle goldens OUT ~/data/goldens/squiddleocr-7c8a052/spark/sample-60
    squiddle legacy ocr ...                           # the 0.3 pipelines (tables, formulas, Markdown, blla, eynollah)

``ocr`` writes the page rows (``OUT/rows/<rank>.jsonl``, linecaster's page row with the ``ocr``
transcription) and one file per page and format with linecaster's ``PageFilesWriter``
(``OUT/<page_id>.hocr|.alto.xml|.page.xml|.txt|.md|.transcription.json``).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import click

from . import __version__
from .legacy_cli import main as legacy_main

IMAGE_EXTS = {".webp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"}


@click.group()
@click.version_option(__version__, prog_name="squiddle")
def main():
    """SquiddleOCR: layout, line detection and kraken's PP-OCRv6 recogniser as a linecaster chain."""


def _images_folder(inputs: tuple[Path, ...]) -> tuple[str, str, Path | None]:
    """A folder and glob for ImageFolderReader; loose files are linked into a temporary folder."""
    if len(inputs) == 1 and inputs[0].is_dir():
        return str(inputs[0].resolve()), "**/*.{webp,jpg,jpeg,png,tif,tiff,jp2}", None
    tmp = Path(tempfile.mkdtemp(prefix="squiddle-"))
    for p in inputs:
        files = [p] if p.is_file() else [f for f in sorted(p.rglob("*")) if f.suffix.lower() in IMAGE_EXTS]
        for f in files:
            (tmp / f.name).symlink_to(f.resolve())
    return str(tmp), "*.{webp,jpg,jpeg,png,tif,tiff,jp2}", tmp


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), required=True, help="Output folder.")
@click.option("-f", "--formats", default="hocr,alto,page,txt", show_default=True,
              help="Per-page files: any of hocr, alto, page, txt, md (the page text; tables as Markdown/HTML), "
                   "transcription (linecaster JSON).")
@click.option("--backend", type=click.Choice(["in-process", "served"]), default="in-process", show_default=True)
@click.option("--endpoint", default="http://127.0.0.1:8100/v1", show_default=True, help="linecaster serve (served).")
@click.option("--window", default=32, show_default=True, help="max pages in flight per served block.")
@click.option("-m", "--model", default="medium", show_default=True, help="kraken PP-OCRv6 recogniser (in-process).")
@click.option("--tasks", default=1, show_default=True, help="datatrove tasks (shards of the image list).")
@click.option("--workers", default=1, show_default=True, help="tasks running at a time.")
def ocr(inputs, out_dir, formats, backend, endpoint, window, model, tasks, workers):
    """Recognise page images with the squiddle chain."""
    from datatrove.executor import LocalPipelineExecutor

    from linecaster.blocks.readers import ImageFolderReader
    from linecaster.blocks.writers import PageFilesWriter, PageJsonlWriter

    from .chain import blocks

    folder, pattern, tmp = _images_folder(inputs)
    out = out_dir.resolve()
    fmts = tuple(f.strip() for f in formats.split(",") if f.strip())
    pipeline = [ImageFolderReader(folder, glob_pattern=pattern),
                *blocks(backend, endpoint, window=window, recognition_model=model),
                PageFilesWriter(str(out), fmts, bucket_by_item=False),
                PageJsonlWriter(str(out / "rows"), output_filename="${rank}.jsonl", compression=None)]
    try:
        LocalPipelineExecutor(pipeline, tasks=tasks, workers=workers, logging_dir=str(out / "logs")).run()
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


@main.command()
@click.argument("run", type=click.Path(exists=True, path_type=Path))
@click.argument("goldens", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True)
def goldens(run, goldens, verbose):
    """Compare a run's page rows with squiddleocr 0.3 goldens (hOCR), line by line."""
    from .goldens import compare

    tot = compare(run, goldens, verbose=verbose)
    click.echo(json.dumps(tot))
    sys.exit(0 if tot["pages"] and tot["pages_exact"] == tot["pages"] else 1)


main.add_command(legacy_main, name="legacy")


if __name__ == "__main__":
    main()
