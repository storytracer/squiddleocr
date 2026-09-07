"""The ``squiddle`` command line interface."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__


@click.group()
@click.version_option(__version__, prog_name="squiddle")
def main():
    """SquiddleOCR: package kraken PP-OCRv6 recognisers for PaddleOCR / PP-StructureV3."""


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
    """Convert a kraken PP-OCRv6 safetensors file into a PaddleOCR model directory."""
    from .convert import convert as _convert

    res = _convert(source, out_dir, model_name=model_name, padding=padding, embed_weights=embed_weights,
                   model_card=model_card, log=lambda s: click.echo(s, err=True))
    click.echo(str(res.out_dir))


@main.command()
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("lines", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--kraken-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="kraken safetensors model used as the reference (default: located via squiddle.json / htrmopo cache).")
@click.option("--batch-size", default=8, show_default=True, help="Batch size for the batched PaddleX-style runs.")
@click.option("--paddle/--no-paddle", default=True, show_default=True,
              help="Run the real PaddleX predictor if paddlex is importable.")
@click.option("--device", default="cpu", show_default=True, help="Device for the kraken reference run (cpu or cuda).")
@click.option("--report", type=click.Path(path_type=Path), default=None, help="Write a JSON report here.")
@click.option("--show", default=20, show_default=True, help="How many mismatching lines to print per comparison.")
def verify(model_dir, lines, kraken_model, batch_size, paddle, device, report, show):
    """Run line images through kraken and the exported model and report agreement."""
    from .verify import run_verify

    files = []
    for p in lines:
        if p.is_dir():
            files.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")))
        else:
            files.append(p)
    if not files:
        raise click.UsageError("No line images given.")
    ok = run_verify(model_dir, files, kraken_model=kraken_model, batch_size=batch_size, use_paddle=paddle,
                    device=device, report=report, show=show, echo=click.echo)
    sys.exit(0 if ok else 1)


@main.command("extract-lines")
@click.argument("pages", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), required=True)
@click.option("--seg-model", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="kraken segmentation model (default: blla.mlmodel from the htrmopo cache).")
@click.option("--max-lines", default=0, show_default=True, help="Stop after this many lines (0 = all).")
@click.option("--device", default="cpu", show_default=True)
def extract_lines(pages, out_dir, seg_model, max_lines, device):
    """Segment page images with kraken and write one PNG per text line (for `verify`)."""
    from .lines import extract_lines as _extract

    n = _extract(pages, out_dir, seg_model=seg_model, max_lines=max_lines, device=device,
                 echo=lambda s: click.echo(s, err=True))
    click.echo(f"{n} lines written to {out_dir}")


@main.command("pipeline-config")
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True, help="Where to write the YAML.")
@click.option("--pipeline", type=click.Choice(["PP-StructureV3", "OCR"]), default="PP-StructureV3", show_default=True)
@click.option("--engine", type=click.Choice(["onnxruntime", "paddle"]), default="onnxruntime", show_default=True,
              help="Engine for the recognition sub-module (the ONNX export needs onnxruntime).")
def pipeline_config(model_dir, output, pipeline, engine):
    """Write a PaddleX pipeline YAML that plugs the model directory into a pipeline."""
    from .pipeline import write_pipeline_config

    write_pipeline_config(model_dir, output, pipeline=pipeline, engine=engine)
    click.echo(str(output))
