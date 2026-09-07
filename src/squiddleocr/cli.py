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
    click.echo(click.style(f"{label:>11s}  ", dim=True) + value, err=True)


def ok(message: str) -> None:
    click.secho(f"✔ {message}", fg="green", err=True)


def warn(message: str) -> None:
    click.secho(f"! {message}", fg="yellow", err=True)


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
    """SquiddleOCR: read historical documents with kraken's PP-OCRv6 recognisers and pluggable layout models.

    Start with `squiddle ocr scans/`. Models are downloaded on first use.
    """


# ---------------------------------------------------------------------------------------- ocr
@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-m", "--model", default="medium", show_default=True,
              help="Recogniser: tiny, small or medium, or a model directory.")
@click.option("--models", default=None,
              help="Where the sizes come from: a local folder (from `squiddle convert`) or a Hub repo "
                   "[default: storytracer/squiddleocr, env SQUIDDLE_MODELS].")
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Output folder [default: next to each image]; one set of files per image, named after it.")
@click.option("-f", "--formats", default="md", show_default=True,
              help="Export formats, comma-separated. Document level (regions, boxes): md (Markdown, tables as HTML), "
                   "doclang (DocLang XML), html, json (lossless DoclingDocument), txt. Line level (polygons, baselines, "
                   "one file per image, written by kraken's serialiser; kraken extra): alto, page.")
@click.option("--layout", type=click.Choice(["paddle", "none"]), default="paddle", show_default=True,
              help="Layout analysis: PP-DocLayout regions with reading order, or none (the page is one text block).")
@click.option("--layout-model", default="PP-DocLayoutV3", show_default=True,
              help="PaddleX layout model for --layout paddle: PP-DocLayoutV3 (learned reading order, polygons) or "
                   "PP-DocLayout_plus-L (PP-StructureV3's model, XY-cut order).")
@click.option("--detector", type=click.Choice(["paddle", "kraken"]), default="paddle", show_default=True,
              help="Text line detector: PP-OCRv6 detection, or kraken's blla baseline segmenter (kraken extra).")
@click.option("--det-model", default="PP-OCRv6_medium_det", show_default=True,
              help="Detector size for --detector paddle: PP-OCRv6_medium_det, PP-OCRv6_small_det or PP-OCRv6_tiny_det.")
@click.option("--unclip-ratio", default=2.0, show_default=True,
              help="Expansion of detected line boxes; PaddleOCR's 1.5 clips ascenders and line-final hyphens on old print.")
@click.option("--tables/--no-tables", default=True, show_default=True, help="Recognise the cell structure of table regions.")
@click.option("--batch-size", default=8, show_default=True,
              help="Lines per recogniser call; 1 reproduces kraken's single-line output exactly, larger is faster.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "tensorrt", "coreml"]), default="auto",
              show_default=True, help="ONNX Runtime execution provider for all models.")
@click.option("--per-document/--per-page", default=False, show_default=True,
              help="One document for all inputs (a book) instead of one per image.")
@click.option("--suffix", default="auto", show_default=True,
              help="Tag between the image name and the extension (<name>.<suffix>.md). auto = the detector name, so "
                   "runs with --detector paddle and --detector kraken sit side by side as <name>.paddle.md and "
                   "<name>.kraken.md; any other word is used as is; none (or empty) writes <name>.md.")
def ocr(inputs, model, models, out_dir, layout, layout_model, detector, det_model, tables, unclip_ratio, device, batch_size,
        formats, per_document, suffix):
    """Read images or folders of images and write DocLang / Markdown / HTML / JSON documents.

    INPUTS are image files or folders. Defaults: medium recogniser, PaddleX layout analysis,
    PP-OCRv6 text detection, table recognition, <name>.paddle.md next to each image. Example:

      squiddle ocr scans/ -f md,doclang,json
    """
    from .document import DocumentBuilder, export
    from .factory import build_pipeline
    from .serialize import PAGE_FORMATS, available, serialize_page
    from .types import Page

    suffix = detector if suffix == "auto" else ("" if suffix.lower() == "none" else suffix.strip("."))
    tagged = (lambda stem: f"{stem}.{suffix}") if suffix else (lambda stem: stem)

    files = _image_files(inputs)
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    click.secho(f"SquiddleOCR {__version__}", bold=True, err=True)
    t0 = time.perf_counter()
    try:
        pipe = build_pipeline(model, models=models, layout=layout, layout_model=layout_model, detector=detector,
                              det_model=det_model, tables=tables,
                              unclip_ratio=unclip_ratio, device=device, batch_size=batch_size, log=lambda s: status("models", s))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise fail(str(e)) from e
    provider = pipe.recognizer.device_provider.replace("ExecutionProvider", "")
    status("recogniser", f"{model}  on {provider}  (batch {batch_size})")
    status("layout", (layout_model if layout == "paddle" else layout) + f"  ·  tables {'on' if tables and layout != 'none' else 'off'}")
    status("detector", f"{det_model if detector == 'paddle' else 'kraken blla'}  ·  unclip {unclip_ratio}")
    status("output", f"{out_dir or 'next to each image'}  ·  {tagged('<name>')}.{{{','.join(fmts)}}}")
    status("ready in", f"{time.perf_counter() - t0:.1f} s")

    from .document import EXPORT_FORMATS
    unknown = [f for f in fmts if f not in EXPORT_FORMATS and f not in PAGE_FORMATS]
    if unknown:
        raise fail(f"unknown export format(s) {', '.join(unknown)}; choose from {', '.join(EXPORT_FORMATS + tuple(PAGE_FORMATS))}")
    doc_fmts = [f for f in fmts if f not in PAGE_FORMATS]
    page_fmts = [f for f in fmts if f in PAGE_FORMATS]
    if page_fmts and not available():
        raise fail('alto/page exports use kraken\'s serialiser: pip install "squiddleocr[kraken]"')
    settings = {"squiddleocr": __version__, "recogniser": str(model), "layout": layout_model if layout == "paddle" else layout,
                "detector": det_model if detector == "paddle" else "kraken blla", "unclip_ratio": unclip_ratio,
                "tables": bool(tables and layout != "none")}

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
        ok(f"{len(files)} pages in {time.perf_counter() - t1:.1f} s -> " + ", ".join(str(p) for p in written[:len(doc_fmts) + len(page_fmts)])
           + (" ..." if len(written) > len(doc_fmts) + len(page_fmts) else ""))
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


# ------------------------------------------------------------------------------------- models
@main.group()
def models():
    """List and download recogniser models."""


@models.command("list")
@click.option("--models", "source", default=None, help="Local folder or Hub repo [default: storytracer/squiddleocr].")
def models_list(source):
    """Show which sizes are available locally (in a folder, or in the cache of a Hub repo)."""
    from .models import SIZES, cache_dir, default_source, list_models

    src = source or default_source()
    status("source", src + ("" if Path(src).is_dir() else f"  (cache {cache_dir()})"))
    have = dict(list_models(src))
    for size in SIZES:
        status(size, click.style(str(have[size]), fg="green") if size in have else click.style("not downloaded", dim=True))


@models.command("pull")
@click.argument("sizes", nargs=-1, type=click.Choice(["tiny", "small", "medium"]))
@click.option("--models", "source", default=None, help="Hub repo to pull from [default: storytracer/squiddleocr].")
def models_pull(sizes, source):
    """Download recognisers into the cache ahead of time (all three sizes when none is given)."""
    from .models import SIZES, resolve_model

    try:
        for size in sizes or SIZES:
            ok(f"{size}: {resolve_model(size, source, log=lambda s: status('models', s))}")
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise fail(str(e)) from e


# ------------------------------------------------------------------------------------ convert
@main.command()
@click.argument("what", nargs=-1)
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=Path("squiddleocr-models"),
              show_default=True, help="Model source folder to build (Hub repo layout: README.md + models/...).")
@click.option("--repo", default="storytracer/squiddleocr", show_default=True,
              help="Hub repo the folder is meant for (written into its README).")
def convert(what, out_dir, repo):
    """Convert kraken PP-OCRv6 models into a model source folder (needs the convert extra).

    WHAT: sizes to convert (tiny, small, medium; all three when omitted, weights fetched from
    kraken's Hub mirror) and/or paths to kraken .safetensors files. The folder has the Hub repo
    layout (README.md model card + models/squiddle_PP-OCRv6_<size>_rec/), can be used directly
    with `squiddle ocr --models FOLDER` and published with `squiddle upload FOLDER`.
    """
    from .hub import build_source, upload_command
    from .models import SIZES, parse_size

    echo = lambda s: status("convert", s)  # noqa: E731
    try:
        files = [Path(w) for w in what if Path(w).is_file()]
        sizes = [parse_size(w) for w in what if not Path(w).is_file()] or ([] if files else list(SIZES))
        if files:
            from .convert.convert import convert as _convert

            for f in files:
                res = _convert(f, out_dir / "models" / "_converting", log=echo)   # variant is read from the file
                target = out_dir / "models" / res.out_dir.name if res.out_dir.name != "_converting" \
                    else out_dir / "models" / f"squiddle_PP-OCRv6_{res.variant}_rec"
                if target.exists():
                    import shutil

                    shutil.rmtree(target)
                res.out_dir.rename(target)
                echo(f"-> {target}")
        if sizes:
            build_source(out_dir, sizes, repo, log=echo)
        else:
            from .hub import write_card

            write_card(out_dir, repo)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e
    ok(str(out_dir))
    status("use", f"squiddle ocr scans/ --models {out_dir}")
    status("publish", upload_command(out_dir, repo))


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--repo", default="storytracer/squiddleocr", show_default=True, help="Hub model repo (created if missing).")
@click.option("--private", is_flag=True, help="Create the repo as private.")
def upload(folder, repo, private):
    """Publish a model source folder (from `squiddle convert`) to a Hugging Face model repo.

    Creates the repo if it does not exist (log in first with `hf auth login`).
    """
    from .hub import upload as _upload

    try:
        ok(_upload(folder, repo, private=private, log=lambda s: status("upload", s)))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise fail(str(e)) from e


# ------------------------------------------------------------------------------------- verify
@main.command()
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("lines", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--kraken-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="kraken safetensors model used as the reference (default: located via squiddle.json / htrmopo cache).")
@click.option("--batch-size", default=8, show_default=True, help="Batch size for the batched runs.")
@click.option("--paddle/--no-paddle", default=True, show_default=True,
              help="Also run the real PaddleX predictor if paddlex is importable.")
@click.option("--device", default="cpu", show_default=True, help="Device for the kraken reference run (cpu or cuda).")
@click.option("--report", type=click.Path(path_type=Path), default=None, help="Write a JSON report here.")
@click.option("--show", default=20, show_default=True, help="How many mismatching lines to print per comparison.")
def verify(model_dir, lines, kraken_model, batch_size, paddle, device, report, show):
    """Compare a converted model with kraken on line images (needs the convert extra).

    LINES are line image files or folders (see `squiddle extract-lines`). Reports exact-match
    rate and CER against kraken for the ONNX model, batched runs and, with --paddle, PaddleX's
    own predictor. Exact agreement at batch size 1 is the acceptance criterion.
    """
    from .integrations.verify import run_verify

    ok = run_verify(model_dir, _image_files(lines), kraken_model=kraken_model, batch_size=batch_size,
                    use_paddle=paddle, device=device, report=report, show=show, echo=click.echo)
    sys.exit(0 if ok else 1)


@main.command("extract-lines")
@click.argument("pages", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), required=True)
@click.option("--seg-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="kraken segmentation model (default: blla.mlmodel from the htrmopo cache).")
@click.option("--max-lines", default=0, show_default=True, help="Stop after this many lines (0 = all).")
@click.option("--device", default="cpu", show_default=True)
def extract_lines(pages, out_dir, seg_model, max_lines, device):
    """Cut page images into line images with kraken's segmenter, for `squiddle verify` (kraken extra)."""
    from .integrations.lines import extract_lines as _extract

    n = _extract(pages, out_dir, seg_model=seg_model, max_lines=max_lines, device=device,
                 echo=lambda s: click.echo(s, err=True))
    click.echo(f"{n} lines written to {out_dir}")
