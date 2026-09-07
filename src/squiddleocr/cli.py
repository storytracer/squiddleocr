"""The ``squiddle`` command line interface."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")


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
    """SquiddleOCR: any layout / segmentation model + kraken's PP-OCRv6 recogniser -> DoclingDocument."""


# ---------------------------------------------------------------------------------------- ocr
@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-m", "--model", default="medium", show_default=True,
              help="Recogniser: tiny, small or medium (downloaded on first use) or a model directory.")
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=Path("out"), show_default=True)
@click.option("--layout", type=click.Choice(["none", "paddle"]), default="paddle", show_default=True,
              help="Layout analyser: 'none' = whole page is one text region; 'paddle' = PP-DocLayout (paddle extra).")
@click.option("--detector", type=click.Choice(["paddle", "kraken"]), default="paddle", show_default=True,
              help="Text line detector: PP-OCRv6 detection (paddle extra) or kraken's blla segmenter (kraken extra).")
@click.option("--det-model", default="PP-OCRv6_medium_det", show_default=True,
              help="PaddleX detector name when --detector paddle (PP-OCRv6_{medium,small,tiny}_det).")
@click.option("--tables/--no-tables", default=True, show_default=True, help="Recognise table structure (paddle extra).")
@click.option("--unclip-ratio", default=2.0, show_default=True, help="Detector box expansion (paddle detector).")
@click.option("--device", default="auto", show_default=True, help="auto, cpu, cuda, tensorrt or coreml.")
@click.option("--batch-size", default=8, show_default=True, help="Recogniser batch size (1 = exact kraken single-line results).")
@click.option("-f", "--formats", default="doclang,md", show_default=True,
              help="Comma-separated export formats: doclang, md, html, json, txt.")
@click.option("--per-document/--per-page", default=False, show_default=True,
              help="Write one document for all inputs instead of one per image.")
def ocr(inputs, model, out_dir, layout, detector, det_model, tables, unclip_ratio, device, batch_size, formats,
        per_document):
    """OCR images or folders of images: `squiddle ocr scans/` writes DocLang and Markdown to out/."""
    from .document import export
    from .factory import build_pipeline

    files = _image_files(inputs)
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    try:
        pipe = build_pipeline(model, layout=layout, detector=detector, det_model=det_model, tables=tables,
                              unclip_ratio=unclip_ratio, device=device, batch_size=batch_size,
                              log=lambda s: click.echo(s, err=True))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"recogniser on {pipe.recognizer.device_provider}; {len(files)} image(s) -> {out_dir}", err=True)
    if per_document:
        doc = pipe.run_files(files)
        for p in export(doc, out_dir, files[0].stem if len(files) == 1 else Path(out_dir).name, fmts):
            click.echo(str(p))
        return
    with click.progressbar(files, label="OCR", item_show_func=lambda f: f.name if f else "", file=sys.stderr) as bar:
        for f in bar:
            export(pipe.run_files([f]), out_dir, f.stem, fmts)


# ------------------------------------------------------------------------------------- models
@main.group()
def models():
    """Manage recogniser models (cache, download, conversion)."""


@models.command("list")
def models_list():
    """Show cached models and where they live."""
    from .models import cache_dir, list_cached

    click.echo(f"cache: {cache_dir()}")
    for p in list_cached():
        click.echo(f"  {p.name}")


@models.command("pull")
@click.argument("size", type=click.Choice(["tiny", "small", "medium"]))
def models_pull(size):
    """Download (or convert) a recogniser into the cache ahead of time."""
    from .models import resolve_model

    try:
        click.echo(str(resolve_model(size, log=lambda s: click.echo(s, err=True))))
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e


# ------------------------------------------------------------------------------------ convert
@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Output model directory (default: squiddle_PP-OCRv6_<size>_rec).")
@click.option("--model-name", default=None,
              help="PaddleX registry name to write into inference.yml (default: PP-OCRv6_<size>_rec).")
@click.option("--padding", default=16, show_default=True,
              help="White padding in px added on both ends inside the graph (kraken's default is 16).")
@click.option("--embed-weights/--external-weights", default=True, show_default=True,
              help="Store weights inside inference.onnx or in inference.onnx.data.")
@click.option("--model-card", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Model card to copy (default: README.md next to the source).")
def convert(source, out_dir, model_name, padding, embed_weights, model_card):
    """Convert a kraken PP-OCRv6 safetensors file into a SquiddleOCR / PaddleOCR model directory (convert extra)."""
    from .convert.convert import convert as _convert

    res = _convert(source, out_dir, model_name=model_name, padding=padding, embed_weights=embed_weights,
                   model_card=model_card, log=lambda s: click.echo(s, err=True))
    click.echo(str(res.out_dir))


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
    """Run line images through kraken and the exported model and report agreement (convert extra)."""
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
    """Segment page images with kraken and write one PNG per text line (for `verify`; kraken extra)."""
    from .integrations.lines import extract_lines as _extract

    n = _extract(pages, out_dir, seg_model=seg_model, max_lines=max_lines, device=device,
                 echo=lambda s: click.echo(s, err=True))
    click.echo(f"{n} lines written to {out_dir}")


# ------------------------------------------------------------------------------ PaddleOCR drop-in
@main.command("pipeline-config")
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Where to write the YAML.")
@click.option("--pipeline", type=click.Choice(["PP-StructureV3", "OCR"]), default="PP-StructureV3", show_default=True)
@click.option("--engine", type=click.Choice(["onnxruntime", "paddle"]), default="onnxruntime", show_default=True,
              help="Engine for the recognition sub-module (the ONNX export needs onnxruntime).")
@click.option("--det-model", default="PP-OCRv6_medium_det", show_default=True,
              help="PP-OCRv6 text detector beside the recogniser: PP-OCRv6_medium_det, PP-OCRv6_small_det or PP-OCRv6_tiny_det.")
def pipeline_config(model_dir, output, pipeline, engine, det_model):
    """Write a PaddleX pipeline YAML that plugs the model directory into PaddleOCR's PP-StructureV3 / OCR."""
    from .integrations.paddleocr import write_pipeline_config

    write_pipeline_config(model_dir, output, pipeline=pipeline, engine=engine, det_model=det_model)
    click.echo(str(output))
