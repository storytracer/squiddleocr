"""``squiddle``: the squiddle pipeline (a linecaster chain) on image files.

    squiddle ocr IMAGES...                            # in-process: this environment's GPU; <name>.paddle.md
    squiddle ocr IMAGES... -o OUT --backend served    # through `linecaster serve` (default :8100)
    squiddle goldens OUT ~/data/goldens/squiddleocr-7c8a052/spark/sample-60
    squiddle legacy ocr ...                           # the 0.3 CLI (kraken's blla pipeline, eynollah)

``ocr`` writes one file per page and format with linecaster's ``PageFileWriter``
(``<name>.paddle.md|.txt|.html|.doclang.xml|.json|.hocr|.alto.xml|.page.xml|.transcription.json|.layout.jpg``)
next to each image or into ``-o OUT``, with ``-o`` also the page records (``OUT/pages/<rank>.jsonl``,
linecaster's page record with the ``ocr_new`` transcription); ``--per-document`` writes one document per run
with ``Binder``, the item-level ``Recomposer`` and ``Composer`` (``chain.item_blocks``) and ``ItemDocumentWriter``.
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

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".jp2")
#: per-page formats: document exports (the DoclingDocument) and line level (linecaster's serialisers)
DOCUMENT_FORMATS = ("md", "doclang", "html", "json", "txt")
LINE_FORMATS = ("hocr", "alto", "page")
FORMATS = DOCUMENT_FORMATS + LINE_FORMATS + ("transcription",)
MODEL_OPTIONS = ("recognition_model", "det_model", "unclip_ratio", "layout_model", "formula_model", "batch_size",
                 "precision", "device")


@click.group()
@click.version_option(__version__, prog_name="squiddle")
def main():
    """SquiddleOCR: PaddleX layout, tables and formulas in front of kraken's PP-OCRv6 recogniser, as a linecaster chain."""


def _image_files(inputs: tuple[Path, ...]) -> list[Path]:
    """Image files in the order given; a folder contributes its own images (not its subfolders'), sorted by name."""
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES))
        else:
            files.append(p)
    if not files:
        raise click.UsageError("No images found.")
    stems = [f.stem for f in files]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise click.UsageError(f"several images are named {', '.join(dup[:5])}: page ids are file names without extension")
    return files


def _link_folder(files: list[Path]) -> Path:
    """A temporary folder with a link to every image (ImageFolderReader reads one folder)."""
    tmp = Path(tempfile.mkdtemp(prefix="squiddle-"))
    for f in files:
        (tmp / f.name).symlink_to(f.resolve())
    return tmp


def resolve_formats(formats: str) -> list[str]:
    fmts = ["md"] if formats == "auto" else [f.strip() for f in formats.split(",") if f.strip()]
    unknown = [f for f in fmts if f not in FORMATS]
    if unknown:
        raise click.BadParameter(f"unknown format(s) {', '.join(unknown)}; choose from {', '.join(FORMATS)}",
                                 param_hint="-f/--formats")
    return fmts


def _model_options(**kw) -> dict:
    return {k: kw[k] for k in MODEL_OPTIONS}


def _beside_writer(targets: dict[str, Path], formats, **kw):
    """linecaster's PageFileWriter, each page's files in its own target folder (next to its image)."""
    from linecaster.blocks.writers import PageFileWriter

    class FilesBeside(PageFileWriter):
        def path(self, page) -> str:
            return f"{str(targets[page.page_id]).lstrip('/')}/{page.page_id}{self.suffix}"

    return FilesBeside("/", formats, bucket_by_item=False, **kw)


def model_options(f):
    for opt in reversed([
        click.option("-m", "--model", "recognition_model", default="medium", show_default=True,
                     help="kraken PP-OCRv6 recogniser: tiny, small or medium (fetched from Hugging Face into its cache), "
                          "or a path to a kraken model file."),
        click.option("--det-model", default="PP-OCRv6_medium_det", show_default=True,
                     help="Text detector: PP-OCRv6_medium_det, PP-OCRv6_small_det or PP-OCRv6_tiny_det."),
        click.option("--unclip-ratio", default=1.5, show_default=True,
                     help="Expansion of PP-OCRv6 line boxes (PaddleOCR's default)."),
        click.option("--layout-model", default="PP-DocLayoutV3", show_default=True,
                     help="PaddleX layout model: PP-DocLayoutV3 (learned reading order, polygons) or PP-DocLayout_plus-L "
                          "(PP-StructureV3's model, XY-cut order)."),
        click.option("--formula-model", default="PP-FormulaNet_plus-L", show_default=True,
                     help="PaddleX formula model: PP-FormulaNet_plus-L or PP-FormulaNet-L (torch, on kraken's device)."),
        click.option("--batch-size", default=8, show_default=True, help="Lines (and formula crops) per forward pass."),
        click.option("--precision", type=click.Choice(["auto", "32-true", "bf16-mixed", "16-mixed", "bf16-true"]),
                     default="auto", show_default=True,
                     help="Numerical precision of kraken's recogniser: auto = bf16-mixed on a CUDA GPU, 32-true on the CPU."),
        click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "tensorrt", "coreml"]), default="auto",
                     show_default=True, help="ONNX Runtime provider for the PaddleX models; cpu or auto for the torch models."),
        click.option("--rtl", is_flag=True, default=False,
                     help="Right-to-left script (Hebrew, Arabic): kraken reads lines right to left, reflow mirrors the geometry."),
    ]):
        f = opt(f)
    return f


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Output folder [default: next to each image]; one set of files per image, named after it. With -o the "
                   "page records (OUT/pages) and datatrove's logs (OUT/logs) are written too.")
@click.option("-f", "--formats", default="auto", show_default=True,
              help="Formats, comma-separated; auto = md. Document (the DoclingDocument): md (a table with spanning cells "
                   "is an HTML table), doclang (DocLang XML), html, json, txt. Line level (linecaster's serialisers): "
                   "hocr, alto, page (PAGE-XML). transcription = linecaster's JSON of the page.")
@model_options
@click.option("--layout", type=click.Choice(["paddle", "none"]), default="paddle", show_default=True,
              help="Layout analysis: PP-DocLayout regions with reading order (paddle), or none (the page is one text "
                   "block, no tables or formulas).")
@click.option("--tables/--no-tables", default=True, show_default=True, help="Recognise the cell structure of table regions.")
@click.option("--formulas/--no-formulas", default=True, show_default=True,
              help="Read formula regions as LaTeX (without: as text).")
@click.option("--detail", type=click.Choice(["word", "line"]), default="word", show_default=True,
              help="Depth of hocr/alto/page: word = lines and words, line = text per line. (0.3's glyph level is not "
                   "there: linecaster's finest level is the word.)")
@click.option("--per-document/--per-page", default=False, show_default=True,
              help="One document for all inputs (a book) instead of one per image.")
@click.option("--suffix", default="auto", show_default=True,
              help="Tag between the image name and the extension (<name>.<suffix>.md); auto = paddle; none (or empty) "
                   "writes <name>.md.")
@click.option("--text", "text_mode", type=click.Choice(["reflow", "lines"]), default="reflow", show_default=True,
              help="Text in md/txt/html/json/doclang: reflow = rows joined into paragraphs, the typesetter's line-end "
                   "hyphens removed, paragraphs continued across regions and pages; lines = one visual row per line. "
                   "hocr/alto/page keep the lines either way.")
@click.option("--typography/--no-typography", default=True, show_default=True,
              help="Retyper: heading levels from type size, body-sized headings demoted, headlines split off, drop "
                   "capitals glued to their paragraph, uncertain page numbers dropped.")
@click.option("--plots/--no-plots", default=False, show_default=True,
              help="Write <name>.<suffix>.layout.jpg: the page with its regions (label, heading level, reading order) "
                   "and lines drawn in.")
@click.option("--furniture/--no-furniture", default=False, show_default=True,
              help="Keep page numbers and running heads in md, txt and html; json and doclang always hold them, in "
                   "Docling's furniture layer.")
@click.option("--dividers/--no-dividers", default=False, show_default=True,
              help="Markdown: a --- rule between the items of consecutive layout regions.")
@click.option("--sections/--no-sections", default=False, show_default=True,
              help="Group each heading with what follows it into a Docling section (json, doclang); md and txt read the same.")
@click.option("--backend", type=click.Choice(["in-process", "served"]), default="in-process", show_default=True,
              help="in-process = the models run here; served = through `linecaster serve` (model settings are the "
                   "server's: `squiddle models-config` writes its config).")
@click.option("--endpoint", default="http://127.0.0.1:8100/v1", show_default=True, help="linecaster serve (served).")
@click.option("--window", default=32, show_default=True, help="Most pages in flight per served block.")
@click.option("--tasks", default=1, show_default=True, help="datatrove tasks (shards of the image list; 1 with --per-document).")
@click.option("--workers", default=1, show_default=True, help="Tasks running at a time.")
@click.option("--no-nfc", "no_nfc", is_flag=True, default=False, hidden=True,
              help="Parity checks only: keep kraken's decomposed text as 0.3 did (no UnicodeNormalizer).")
def ocr(inputs, out_dir, formats, layout, tables, formulas, detail, per_document, suffix, text_mode, typography,
        plots, furniture, dividers, sections, backend, endpoint, window, tasks, workers, no_nfc, **models):
    """Read images or folders of images; write Markdown / DocLang / HTML / JSON / txt and hOCR / ALTO / PAGE.

    INPUTS are image files or folders. Defaults: PP-DocLayoutV3 layout, PP-OCRv6 text detection, tables and
    formulas, kraken's medium recogniser, reflowed text, <name>.paddle.md next to each image. Examples:

      squiddle ocr scans/ -f md,doclang,json

      squiddle ocr book/ --per-document -f md,txt -o out/

      squiddle ocr scans/ -f alto,page --detail line -o out/
    """
    from linecaster.blocks.readers import ImageFolderReader
    from linecaster.blocks.writers import ItemDocumentWriter, PageFileWriter, PageJsonlWriter, RejectWriter

    from .chain import DEFAULTS, FailedPagesDropped, blocks, item_blocks

    files = _image_files(inputs)
    fmts = resolve_formats(formats)
    if backend == "served":
        changed = [k for k in MODEL_OPTIONS if models[k] != DEFAULTS[k]]
        if changed:
            click.secho(f"! --backend served: {', '.join(changed)} are the server's settings; "
                        "start it with `squiddle models-config` and the same options", fg="yellow", err=True)
    suffix = "paddle" if suffix == "auto" else ("" if suffix.lower() == "none" else suffix.strip("."))
    tag = f".{suffix}" if suffix else ""
    doc_fmts = [f for f in fmts if f in DOCUMENT_FORMATS]
    page_fmts = [f for f in fmts if f not in DOCUMENT_FORMATS] + (["plot"] if plots else [])
    if per_document:
        tasks = workers = 1          # the Binder sees the whole item in one task
        item = files[0].stem if len(files) == 1 else (inputs[0].name if inputs[0].is_dir() else "document")
    folder = _link_folder(files)
    out = out_dir.resolve() if out_dir else None
    logs = out / "logs" if out else Path(tempfile.mkdtemp(prefix="squiddle-logs-"))
    per_page = page_fmts + ([] if per_document else doc_fmts)
    writer_kw = dict(suffix=tag, furniture=furniture, detail=detail, dividers=dividers)
    pipeline = [ImageFolderReader(str(folder), glob_pattern="*", item_id=item if per_document else None,
                                  sequence=per_document),
                *blocks(backend, endpoint, window=window, layout=layout, tables=tables, formulas=formulas,
                        typography=typography, text=text_mode, sections=sections, nfc=not no_nfc,
                        document=bool(doc_fmts) and not per_document, rtl=models["rtl"], **_model_options(**models))]
    from .chain import MODELS, RECIPE, models as model_list

    # the recipe: its model presets as they are, every step, and its rule-based presets at their options
    # (--text lines leaves out its recomposition, --sections changes its composition's option)
    if (model_list(rtl=models["rtl"], **_model_options(**models)) == MODELS and layout == "paddle" and tables
            and formulas and typography and not no_nfc and text_mode == "reflow" and not sections):
        from linecaster.blocks.validation import RecipeStamper
        from linecaster.catalog.registry import catalog

        r = catalog().resolve_recipe(RECIPE)          # the defaults are the recipe: every file names it
        pipeline.append(RecipeStamper(r.ref, r.sha256))
    run_folder = out or logs
    # pages a block failed on: listed in rejects/ (and, with -o, kept in the page records with their errors), but
    # no files and no place in the document for them (0.3 skipped them too)
    pipeline.append(RejectWriter(str(run_folder / "rejects")))
    if out:
        pipeline.append(PageJsonlWriter(str(out / "pages"), output_filename="${rank}.jsonl", compression=None))
    pipeline.append(FailedPagesDropped())
    if per_page:
        pipeline.append(PageFileWriter(str(out), per_page, bucket_by_item=False, **writer_kw) if out else
                        _beside_writer({f.stem: f.parent.resolve() for f in files}, per_page, **writer_kw))
    if per_document and doc_fmts:
        target = out or files[0].parent.resolve()
        pipeline += [*item_blocks(text=text_mode, sections=sections, rtl=models["rtl"]),
                     ItemDocumentWriter(str(target), doc_fmts, suffix=tag, furniture=furniture, dividers=dividers)]
    try:
        _run(pipeline, tasks, workers, run_folder, logs)
        failed = _rejects(run_folder / "rejects")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        if not out:
            shutil.rmtree(logs, ignore_errors=True)
    if failed:
        for page_id, why in list(failed.items())[:20]:
            click.secho(f"✘ {page_id}: {why}", fg="red", err=True)
        raise click.ClickException(f"{len(failed)} of {len(files)} page(s) failed" +
                                   (" (OUT/rejects lists them all)" if out and len(failed) > 20 else "") +
                                   "; the other pages were written")
    click.secho(f"✔ {len(files)} pages -> {out or 'next to the images'}", fg="green", err=True)


class PipelineProblems(click.ClickException):
    """A pipeline linecaster's builder refuses (its contracts do not fit): exit code 2, nothing ran."""

    exit_code = 2


def _run(pipeline: list, tasks: int, workers: int, run_folder: Path, logs: Path) -> None:
    """Run ``pipeline`` through linecaster's builder: its contracts checked first (nothing runs when they do not
    fit), the run traced into ``<run_folder>/traces``; one datatrove executor per stage (squiddle's pipelines have
    one: ``--per-document`` runs as one task). The rejects of an earlier run into the same folder go first."""
    from datatrove.executor import LocalPipelineExecutor

    from linecaster.errors import PipelineConfigError
    from linecaster.runtime.builder import build

    plan = build(pipeline, tasks=tasks, run_folder=str(run_folder))
    try:
        plan.check()
    except PipelineConfigError as e:
        raise PipelineProblems(str(e)) from e
    shutil.rmtree(run_folder / "rejects", ignore_errors=True)
    previous = None
    for stage in plan.stages:
        stage_logs = logs / stage.name if len(plan.stages) > 1 else logs
        previous = LocalPipelineExecutor(stage.pipeline, tasks=stage.tasks, workers=min(workers, stage.tasks),
                                         logging_dir=str(stage_logs), skip_completed=False, depends=previous)
    previous.run()


def _rejects(folder: Path) -> dict[str, str]:
    """Why each page a block failed on failed (its first error), from the ``RejectWriter``'s files."""
    out: dict[str, str] = {}
    for f in sorted(folder.glob("*.jsonl")) if folder.is_dir() else ():
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out.setdefault(r["key"], f"{r.get('block') or r.get('field')}: {r.get('type') or 'error'}: {r.get('error')}")
    return out


@main.command("models-config")
@model_options
def models_config(**models):
    """Print the `linecaster serve --config` file for the squiddle models with these settings (for --backend served):

      linecaster serve --port 8100 --config <(squiddle models-config --det-model PP-OCRv6_small_det)
    """
    from .chain import models as model_list

    click.echo(json.dumps({"models": model_list(rtl=models["rtl"], **_model_options(**models))}, indent=1))


@main.command()
@click.argument("run", type=click.Path(exists=True, path_type=Path))
@click.argument("goldens", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True)
def goldens(run, goldens, verbose):
    """Compare a run's page records with squiddleocr 0.3 goldens (hOCR), line by line."""
    from .goldens import compare

    tot = compare(run, goldens, verbose=verbose)
    click.echo(json.dumps(tot))
    sys.exit(0 if tot["pages"] and tot["pages_exact"] == tot["pages"] else 1)


main.add_command(legacy_main, name="legacy")


if __name__ == "__main__":
    main()
