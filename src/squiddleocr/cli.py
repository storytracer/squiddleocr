"""The ``squiddle`` command line interface."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from tqdm import tqdm

from . import __version__

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")


def status(label: str, value: str = "") -> None:
    """One aligned status line on stderr: a dim label and a plain value."""
    click.echo(click.style(f"{label:>12s}  ", dim=True) + value, err=True)


def ok(message: str) -> None:
    click.secho(f"✔ {message}", fg="green", err=True)


def fail(message: str) -> "click.ClickException":
    return click.ClickException(click.style(message, fg="red"))


def _image_files(inputs: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES))
        else:
            files.append(p)
    if not files:
        raise click.UsageError("No images found.")
    return files


@click.group()
@click.version_option(__version__, prog_name="squiddle")
def main():
    """SquiddleOCR: PaddleX layout analysis in front of kraken's PP-OCRv6 recogniser.

    Start with `squiddle ocr scans/`. Models are downloaded on first use.
    """


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-m", "--model", default="medium", show_default=True,
              help="kraken PP-OCRv6 recogniser: tiny, small or medium (fetched from Zenodo into kraken's model cache), "
                   "or a path to a kraken model file.")
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Output folder [default: next to each image]; one set of files per image, named after it.")
@click.option("-f", "--formats", default="md", show_default=True,
              help="Export formats, comma-separated. Document level (regions, boxes): md (Markdown, tables as HTML), "
                   "doclang (DocLang XML), html, json (lossless DoclingDocument), txt. Line level (one file per image, "
                   "kraken's serialiser: lines, words, glyphs): hocr, alto, page (PAGE-XML).")
@click.option("--segmentation", type=click.Choice(["paddle", "kraken"]), default="paddle", show_default=True,
              help="Where the text lines come from: paddle = PP-OCRv6 text detection (line boxes), kraken = the blla "
                   "segmenter (polygons and baselines). Recognition is kraken's either way.")
@click.option("--det-model", default="PP-OCRv6_medium_det", show_default=True,
              help="Detector size for --segmentation paddle: PP-OCRv6_medium_det, PP-OCRv6_small_det or PP-OCRv6_tiny_det.")
@click.option("--unclip-ratio", default=2.0, show_default=True,
              help="Expansion of PP-OCRv6 line boxes; PaddleOCR's 1.5 clips ascenders and line-final hyphens on old print.")
@click.option("--layout", type=click.Choice(["paddle", "none"]), default="paddle", show_default=True,
              help="Layout analysis: PP-DocLayout regions with reading order, or none (the page is one text block).")
@click.option("--layout-model", default="PP-DocLayoutV3", show_default=True,
              help="PaddleX layout model for --layout paddle: PP-DocLayoutV3 (learned reading order, polygons) or "
                   "PP-DocLayout_plus-L (PP-StructureV3's model, XY-cut order).")
@click.option("--tables/--no-tables", default=True, show_default=True, help="Recognise the cell structure of table regions.")
@click.option("--batch-size", default=8, show_default=True, help="Lines per kraken forward pass.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "tensorrt", "coreml"]), default="auto",
              show_default=True, help="ONNX Runtime provider for the PaddleX models; cpu or auto for kraken's torch models.")
@click.option("--per-document/--per-page", default=False, show_default=True,
              help="One document for all inputs (a book) instead of one per image.")
@click.option("--suffix", default="auto", show_default=True,
              help="Tag between the image name and the extension (<name>.<suffix>.md). auto = the segmentation name, so "
                   "paddle and kraken runs sit side by side as <name>.paddle.md and <name>.kraken.md; any other word is "
                   "used as is; none (or empty) writes <name>.md.")
def ocr(inputs, model, out_dir, formats, segmentation, det_model, unclip_ratio, layout, layout_model, tables, batch_size,
        device, per_document, suffix):
    """Read images or folders of images; write Markdown / DocLang / HTML / JSON and hOCR / ALTO / PAGE.

    INPUTS are image files or folders. Defaults: medium recogniser, PP-OCRv6 text detection, PaddleX
    layout, tables on, <name>.paddle.md next to each image. Examples:

      squiddle ocr scans/ -f md,doclang,json

      squiddle ocr scans/ --segmentation kraken -f md,hocr,page
    """
    from .document import EXPORT_FORMATS, DocumentBuilder, export
    from .factory import build_pipeline
    from .serialize import PAGE_FORMATS, serialize_page
    from .types import Page

    suffix = segmentation if suffix == "auto" else ("" if suffix.lower() == "none" else suffix.strip("."))
    tagged = (lambda stem: f"{stem}.{suffix}") if suffix else (lambda stem: stem)

    files = _image_files(inputs)
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    unknown = [f for f in fmts if f not in EXPORT_FORMATS and f not in PAGE_FORMATS]
    if unknown:
        raise fail(f"unknown export format(s) {', '.join(unknown)}; choose from {', '.join(EXPORT_FORMATS + tuple(PAGE_FORMATS))}")
    doc_fmts = [f for f in fmts if f not in PAGE_FORMATS]
    page_fmts = [f for f in fmts if f in PAGE_FORMATS]

    click.secho(f"SquiddleOCR {__version__}", bold=True, err=True)
    t0 = time.perf_counter()
    try:
        pipe = build_pipeline(model, segmentation=segmentation, layout=layout, layout_model=layout_model, det_model=det_model,
                              tables=tables, unclip_ratio=unclip_ratio, device=device, batch_size=batch_size,
                              log=lambda s: status("models", s))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise fail(str(e)) from e
    status("recogniser", f"kraken {pipe.recognizer.model_path.name}  on {pipe.recognizer.device}  (batch {batch_size})")
    status("segmentation", f"{det_model}  ·  unclip {unclip_ratio}" if segmentation == "paddle" else "kraken blla")
    status("layout", (layout_model if layout == "paddle" else layout) + f"  ·  tables {'on' if tables and layout != 'none' else 'off'}")
    status("output", f"{out_dir or 'next to each image'}  ·  {tagged('<name>')}.{{{','.join(fmts)}}}")
    status("ready in", f"{time.perf_counter() - t0:.1f} s")
    settings = {"squiddleocr": __version__, "recogniser": pipe.recognizer.model_path.name, "segmentation": segmentation,
                "detector": det_model if segmentation == "paddle" else "kraken blla", "unclip_ratio": unclip_ratio,
                "layout": layout_model if layout == "paddle" else layout, "tables": bool(tables and layout != "none")}

    def write_page_formats(page, contents, target, stem):
        out = []
        if page_fmts:
            Path(target).mkdir(parents=True, exist_ok=True)
        for fmt in page_fmts:
            path = Path(target) / f"{stem}{PAGE_FORMATS[fmt]}"
            path.write_text(serialize_page(page, contents, fmt, settings), encoding="utf-8")
            out.append(path)
        return out

    if per_document:
        target = out_dir or files[0].parent
        stem = files[0].stem if len(files) == 1 else (Path(inputs[0]).name if Path(inputs[0]).is_dir() else "document")
        t1 = time.perf_counter()
        builder = DocumentBuilder(stem)
        written = []
        with tqdm(files, unit="page", desc="OCR", dynamic_ncols=True, leave=False) as bar:
            for i, f in enumerate(bar):
                bar.set_postfix_str(f.name, refresh=False)
                page = Page.load(f, number=i + 1)
                contents = pipe.process_page(page)
                builder.add_page(page, contents)
                written += write_page_formats(page, contents, target, tagged(f.stem))
        if doc_fmts:
            written = export(builder.build(), target, tagged(stem), doc_fmts) + written
        shown = len(doc_fmts) + len(page_fmts)
        ok(f"{len(files)} pages in {time.perf_counter() - t1:.1f} s -> " + ", ".join(str(p) for p in written[:shown])
           + (" ..." if len(written) > shown else ""))
        return

    failed = []
    t1 = time.perf_counter()
    with tqdm(files, unit="page", desc="OCR", dynamic_ncols=True, leave=False) as bar:
        for f in bar:
            bar.set_postfix_str(f.name, refresh=False)
            try:
                page = Page.load(f, number=1)
                contents = pipe.process_page(page)
                target = out_dir or f.parent
                if doc_fmts:
                    builder = DocumentBuilder(f.stem)
                    builder.add_page(page, contents)
                    export(builder.build(), target, tagged(f.stem), doc_fmts)
                write_page_formats(page, contents, target, tagged(f.stem))
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                failed.append(f)
                tqdm.write(click.style(f"! {f.name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", fg="yellow"), file=sys.stderr)
    dt = time.perf_counter() - t1
    done = len(files) - len(failed)
    ok(f"{done} of {len(files)} pages in {dt:.1f} s ({dt / max(done, 1):.2f} s/page) -> {out_dir or 'next to the images'}")
    if failed:
        raise fail(f"{len(failed)} page(s) failed: " + ", ".join(f.name for f in failed[:5]) + (" ..." if len(failed) > 5 else ""))
