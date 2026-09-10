"""The ``squiddle`` command line interface."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import click
from tqdm import tqdm

from . import __version__

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")
#: Export formats each pipeline can write: paddle and eynollah have layout regions for the document
#: formats, kraken has one page of lines and writes what the kraken command writes.
DOCUMENT_PIPELINE_FORMATS = ("md", "doclang", "html", "json", "txt", "hocr", "alto", "page")
PIPELINE_FORMATS = {"paddle": DOCUMENT_PIPELINE_FORMATS, "eynollah": DOCUMENT_PIPELINE_FORMATS,
                    "kraken": ("hocr", "alto", "page", "txt")}
PULLABLE = ("eynollah", "tiny", "small", "medium", "blla")


def status(label: str, value: str = "") -> None:
    """One aligned status line on stderr: a dim label and a plain value."""
    click.echo(click.style(f"{label:>12s}  ", dim=True) + value, err=True)


def warn(message: str) -> None:
    tqdm.write(click.style(f"! {message}", fg="yellow"), file=sys.stderr)


def ok(message: str) -> None:
    click.secho(f"✔ {message}", fg="green", err=True)


def fail(message: str) -> "click.ClickException":
    return click.ClickException(click.style(message, fg="red"))


def quiet_libraries() -> None:
    """Silence what the libraries say about single lines during a run, unless ``SQUIDDLE_VERBOSE`` is set.

    kraken logs a warning for every line whose boundary polygon it could not build (the line is
    still read from its bounding box), and PIL raises a numpy divide-by-zero from the degenerate
    crop that follows; both go to stderr in the middle of the progress bar. Everything else that
    still logs at WARNING or above goes through ``tqdm.write`` so the bar stays intact."""
    import logging
    import warnings

    class TqdmHandler(logging.Handler):
        def emit(self, record):
            tqdm.write(self.format(record), file=sys.stderr)

    handler = TqdmHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    if not os.environ.get("SQUIDDLE_VERBOSE"):
        logging.getLogger("kraken").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"PIL\.Image")


def resolve_formats(formats: str, pipeline: str) -> list[str]:
    """The export formats for a run: ``auto`` is md (paddle, eynollah) or hocr (kraken, its own default);
    unknown formats and formats the pipeline cannot write raise ``ValueError`` with the choices."""
    from .document import EXPORT_FORMATS
    from .serialize import PAGE_FORMATS

    allowed = PIPELINE_FORMATS[pipeline]
    if formats == "auto":
        formats = "hocr" if pipeline == "kraken" else "md"
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    unknown = [f for f in fmts if f not in EXPORT_FORMATS and f not in PAGE_FORMATS]
    if unknown:
        raise ValueError(f"unknown export format(s) {', '.join(unknown)}; choose from {', '.join(EXPORT_FORMATS + tuple(PAGE_FORMATS))}")
    refused = [f for f in fmts if f not in allowed]
    if refused:
        raise ValueError(f"format(s) {', '.join(refused)} need layout regions (--pipeline paddle or eynollah); "
                         f"--pipeline kraken writes {', '.join(allowed)}")
    return fmts


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
    """SquiddleOCR: layout analysis (PaddleX or eynollah) in front of kraken's PP-OCRv6 recogniser.

    Start with `squiddle ocr scans/`. Models are downloaded on first use; `squiddle models pull`
    fetches them ahead of time.
    """


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-m", "--model", default="medium", show_default=True,
              help="kraken PP-OCRv6 recogniser: tiny, small or medium (fetched from Zenodo into kraken's model cache), "
                   "or a path to a kraken model file.")
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Output folder [default: next to each image]; one set of files per image, named after it.")
@click.option("-f", "--formats", default="auto", show_default=True,
              help="Export formats, comma-separated; auto = md (paddle, eynollah) or hocr (kraken). Document level (regions, boxes): md (Markdown; a table with spanning cells is an HTML table), "
                   "doclang (DocLang XML), html, json (lossless DoclingDocument), txt. Line level (one file per image, "
                   "kraken's serialiser: lines, words, glyphs): hocr, alto, page (PAGE-XML).")
@click.option("--pipeline", type=click.Choice(["paddle", "kraken", "eynollah"]), default="paddle", show_default=True,
              help="paddle = PaddleX layout analysis, PP-OCRv6 text detection, tables and formulas; every format. "
                   "kraken = the blla segmenter on the whole page (polygons, baselines, kraken's line order), what the "
                   "kraken command does; formats hocr, alto, page, txt. eynollah = eynollah's regions, reading order "
                   "and reading order with PP-OCRv6 detection in its regions (same as --layout eynollah); every "
                   "format. Recognition is kraken's either way.")
@click.option("--det-model", default="PP-OCRv6_medium_det", show_default=True,
              help="Detector size for --pipeline paddle: PP-OCRv6_medium_det, PP-OCRv6_small_det or PP-OCRv6_tiny_det.")
@click.option("--unclip-ratio", default=2.0, show_default=True,
              help="Expansion of PP-OCRv6 line boxes; PaddleOCR's 1.5 clips ascenders and line-final hyphens on old print.")
@click.option("--layout", type=click.Choice(["paddle", "none", "eynollah"]), default="paddle", show_default=True,
              help="Layout analysis: PP-DocLayout regions with reading order (paddle), none (the page is one text "
                   "block, no tables or formulas), or eynollah (SBB's historical layout analyser for regions and "
                   "reading order, run as a subprocess, PP-OCRv6 detection in its regions; needs the eynollah extra "
                   "and `squiddle models pull eynollah`).")
@click.option("--layout-model", default="PP-DocLayoutV3", show_default=True,
              help="PaddleX layout model for --layout paddle: PP-DocLayoutV3 (learned reading order, polygons) or "
                   "PP-DocLayout_plus-L (PP-StructureV3's model, XY-cut order).")
@click.option("--tables/--no-tables", default=True, show_default=True, help="Recognise the cell structure of table regions.")
@click.option("--formulas/--no-formulas", default=True, show_default=True, help="Read formula regions as LaTeX.")
@click.option("--formula-model", default="PP-FormulaNet_plus-L", show_default=True,
              help="PaddleX formula model: PP-FormulaNet_plus-L or PP-FormulaNet-L (torch, on kraken's device).")
@click.option("--detail", type=click.Choice(["line", "word", "glyph"]), default="glyph", show_default=True,
              help="Depth of hocr/alto/page: glyph = words and glyphs from kraken's character cuts (kraken's default), "
                   "word = words only (ALTO Strings, PAGE Words), line = text per line (kraken's --no-subline-segmentation).")
@click.option("--batch-size", default=8, show_default=True, help="Lines per kraken forward pass.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda", "tensorrt", "coreml"]), default="auto",
              show_default=True, help="ONNX Runtime provider for the PaddleX models; cpu or auto for kraken's torch models.")
@click.option("--per-document/--per-page", default=False, show_default=True,
              help="One document for all inputs (a book) instead of one per image.")
@click.option("--suffix", default="auto", show_default=True,
              help="Tag between the image name and the extension (<name>.<suffix>.md). auto = the pipeline name, so "
                   "paddle, kraken and eynollah runs sit side by side as <name>.paddle.md and <name>.eynollah.md; any "
                   "other word is used as is; none (or empty) writes <name>.md.")
@click.option("--text", "text_mode", type=click.Choice(["reflow", "lines"]), default="reflow", show_default=True,
              help="Text in md/txt/html/json/doclang: reflow = rows joined into paragraphs, the typesetter's line-end "
                   "hyphens removed, paragraphs continued across regions and pages (language-free rules); lines = one "
                   "visual row per line with hard line breaks. hocr/alto/page keep the lines either way.")
@click.option("--sections/--no-sections", default=None,
              help="Group each heading with what follows it (up to the next heading) into a Docling section in the "
                   "json and doclang exports; md and txt read the same. [default: on for --layout eynollah, off otherwise]")
@click.option("--rtl", is_flag=True, default=False,
              help="Right-to-left script (Hebrew, Arabic): kraken reads lines right to left and eynollah orders regions "
                   "right to left (-r2l).")
@click.option("--baselines/--no-baselines", default=True, show_default=True,
              help="For --layout eynollah: read each line along a baseline synthesised inside its polygon (kraken "
                   "dewarps by the polygon), or as its bounding box.")
@click.option("--eynollah-lines", type=click.Choice(["paddle", "eynollah", "blla"]), default="paddle", show_default=True,
              help="Text lines for --layout eynollah: paddle = PP-OCRv6 detection (--det-model, --unclip-ratio) on each "
                   "eynollah text region, boxes joined into rows; eynollah = its own line polygons (baselines "
                   "synthesised; they fragment at wide word gaps); blla = kraken's segmenter on each region (real "
                   "baselines, seconds per region).")
@click.option("--eynollah-jobs", type=int, default=None, metavar="N",
              help="Parallel eynollah page jobs [default: from cores, RAM and page size; never more than 8].")
@click.option("--eynollah-device", default=None, metavar="SPEC",
              help="eynollah's -D device spec: GPU, GPU0, CPU or per-model globs like 'col*:CPU,*:GPU0' "
                   "[default: GPU when ONNX Runtime has CUDA, else CPU].")
@click.option("--eynollah-vram-margin", default=None, metavar="X",
              help="GPU memory kept free of eynollah's models: a fraction (0.2, 20%) or gigabytes (4, 4G) "
                   "[default: 20% or 4 GB, whichever is larger].")
@click.option("--eynollah-tensorrt", is_flag=True, default=False,
              help="Let eynollah use ONNX Runtime's TensorRT provider (engines are built on first use, minutes per "
                   "model, cached under XDG_CONFIG_HOME); only when libnvinfer is loadable.")
@click.option("--eynollah-args", default="-fl -romb", show_default=True, metavar="ARGS",
              help="Flags passed to `eynollah layout`: -fl full layout (headings, drop capitals), -romb machine-based "
                   "reading order, -tab tables, -cl curved lines, -as scaling check, -ib internal binarisation, ...")
@click.option("--eynollah-xml", type=click.Path(path_type=Path), default=None, metavar="DIR",
              help="Directory of eynollah PAGE-XML (<stem>.xml). Pages with a file there are consumed without "
                   "running eynollah; the rest are written into it [default: a temporary directory].")
def ocr(inputs, model, out_dir, formats, pipeline, det_model, unclip_ratio, layout, layout_model, tables, formulas,
        formula_model, detail, batch_size, device, per_document, suffix, text_mode, sections, rtl, baselines, eynollah_lines,
        eynollah_jobs, eynollah_device, eynollah_vram_margin, eynollah_tensorrt, eynollah_args, eynollah_xml):
    """Read images or folders of images; write Markdown / DocLang / HTML / JSON and hOCR / ALTO / PAGE.

    INPUTS are image files or folders. Defaults: the paddle pipeline (PaddleX layout, PP-OCRv6 text
    detection, tables and formulas, kraken recognition) with the medium recogniser, <name>.paddle.md
    next to each image. Examples:

      squiddle ocr scans/ -f md,doclang,json

      squiddle ocr scans/ --pipeline kraken -f hocr,page

      squiddle ocr scans/ --layout eynollah -f md,page
    """
    from .document import DocumentBuilder, export
    from .factory import build_pipeline
    from .serialize import PAGE_FORMATS, serialize_page
    from .types import Page

    if pipeline == "kraken":   # blla segments the whole page, as the kraken command does
        layout, tables, formulas = "none", False, False
    if pipeline == "eynollah" or layout == "eynollah":
        pipeline, layout, tables, formulas = "eynollah", "eynollah", False, False
    if sections is None:
        sections = pipeline == "eynollah"
    suffix = pipeline if suffix == "auto" else ("" if suffix.lower() == "none" else suffix.strip("."))
    tagged = (lambda stem: f"{stem}.{suffix}") if suffix else (lambda stem: stem)
    text_direction = "horizontal-rl" if rtl else "horizontal-lr"

    files = _image_files(inputs)
    try:
        fmts = resolve_formats(formats, pipeline)
    except ValueError as e:
        raise fail(str(e))
    doc_fmts = [f for f in fmts if f not in PAGE_FORMATS]
    page_fmts = [f for f in fmts if f in PAGE_FORMATS]

    click.secho(f"SquiddleOCR {__version__}", bold=True, err=True)
    quiet_libraries()
    t0 = time.perf_counter()
    eynollah = None
    if pipeline == "eynollah":
        from .eynollah import EynollahOptions

        eynollah = EynollahOptions(jobs=eynollah_jobs, device=eynollah_device, vram_margin=eynollah_vram_margin,
                                   tensorrt=eynollah_tensorrt, args=eynollah_args, xml_dir=eynollah_xml, rtl=rtl,
                                   baselines=baselines, lines=eynollah_lines, det_model=det_model, unclip_ratio=unclip_ratio)
    bar_holder: dict = {}

    def eynollah_progress(done, total):
        bar = bar_holder.get("bar")
        if bar is not None:
            bar.total = total
            bar.n = done
            bar.refresh()

    try:
        pipe = build_pipeline(model, pipeline=pipeline, layout=layout, layout_model=layout_model, det_model=det_model,
                              tables=tables, formulas=formulas, formula_model=formula_model,
                              unclip_ratio=unclip_ratio, device=device, batch_size=batch_size,
                              eynollah=eynollah, text_direction=text_direction,
                              log=lambda s: status("models", s), warn=warn, progress=eynollah_progress)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise fail(str(e)) from e
    status("recogniser", f"kraken {pipe.recognizer.model_path.name}  on {pipe.recognizer.device}  (batch {batch_size})"
           + ("  ·  right-to-left" if rtl else ""))
    if pipeline == "paddle":
        status("pipeline", "paddle  ·  PaddleX layout and lines, kraken recognition")
        status("lines", f"{det_model}  ·  unclip {unclip_ratio}")
        status("layout", (layout_model if layout == "paddle" else "none (one text block per page)")
               + f"  ·  tables {'on' if tables and layout != 'none' else 'off'}"
               + f"  ·  formulas {formula_model if formulas and layout != 'none' else 'off'}")
    elif pipeline == "eynollah":
        status("pipeline", "eynollah  ·  eynollah regions and reading order (subprocess), "
               + {"paddle": "PP-OCRv6 lines", "eynollah": "eynollah lines", "blla": "blla lines"}[eynollah_lines] + ", kraken recognition")
        lines_desc = {"blla": "blla per region (kraken's lines and baselines)",
                      "paddle": f"{det_model} per region (unclip {unclip_ratio}), rows joined",
                      "eynollah": f"eynollah's lines as {'polygons with synthesised baselines' if baselines else 'boxes'}"}[eynollah_lines]
        status("eynollah", f"eynollah layout {eynollah_args}  ·  {lines_desc}  ·  xml {eynollah_xml or 'temporary'}")
    else:
        status("pipeline", "kraken  ·  blla on the whole page, kraken's records and line order")
    status("output", f"{out_dir or 'next to each image'}  ·  {tagged('<name>')}.{{{','.join(fmts)}}}"
           + (f"  ·  detail {detail}" if page_fmts else "") + ("  ·  sections" if sections and doc_fmts else "")
           + (f"  ·  text {text_mode}" if doc_fmts else ""))
    status("ready in", f"{time.perf_counter() - t0:.1f} s")
    settings = {"squiddleocr": __version__, "recogniser": pipe.recognizer.model_path.name, "pipeline": pipeline,
                "detector": {"paddle": det_model, "kraken": "kraken blla",
                             "eynollah": {"blla": "kraken blla per eynollah region", "paddle": f"{det_model} per eynollah region",
                                          "eynollah": "eynollah"}[eynollah_lines]}[pipeline],
                "unclip_ratio": unclip_ratio,
                "layout": {"paddle": layout_model, "none": "none", "eynollah": f"eynollah layout {eynollah_args}"}[layout],
                "tables": bool(tables and layout != "none"),
                "formulas": formula_model if formulas and layout != "none" else "", "detail": detail,
                "text_direction": text_direction, "sections": bool(sections), "text": text_mode}

    if pipeline == "eynollah":
        t_eyn = time.perf_counter()
        with tqdm(total=len(files), unit="page", desc="eynollah", dynamic_ncols=True, leave=False) as bar:
            bar_holder["bar"] = bar
            try:
                # the resource plan is printed through the log callback before the subprocess starts
                pipe.layout.source.log = lambda s: tqdm.write(click.style(f"{'resources':>12s}  ", dim=True) + s, file=sys.stderr)
                pipe.prepare(files)
            except (RuntimeError, ValueError, FileNotFoundError) as e:
                pipe.close()
                raise fail(str(e)) from e
            finally:
                bar_holder.pop("bar", None)
        result = pipe.layout.source.last_result
        if result is not None and result.plan is not None:
            providers = sorted({p for p in result.providers.values()})
            status("eynollah", f"{len(result.xml)} of {len(files)} pages in {result.seconds:.1f} s"
                   + f"  ·  ONNX provider {'/'.join(providers) if providers else 'unknown'}"
                   + (f"  ·  {result.attempts} attempts" if result.attempts > 1 else "")
                   + f"  ·  xml {pipe.layout.source.xml_dir}")
        elif result is not None:
            status("eynollah", f"all {len(files)} pages have PAGE-XML in {pipe.layout.source.xml_dir}; eynollah not run")

    def write_page_formats(page, contents, target, stem):
        out = []
        if page_fmts:
            Path(target).mkdir(parents=True, exist_ok=True)
        for fmt in page_fmts:
            path = Path(target) / f"{stem}{PAGE_FORMATS[fmt]}"
            path.write_text(serialize_page(page, contents, fmt, settings, detail, text_direction), encoding="utf-8")
            out.append(path)
        return out

    try:
        if per_document:
            target = out_dir or files[0].parent
            stem = files[0].stem if len(files) == 1 else (Path(inputs[0]).name if Path(inputs[0]).is_dir() else "document")
            t1 = time.perf_counter()
            builder = DocumentBuilder(stem, sections=sections, text=text_mode, rtl=rtl)
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
                if text_mode == "reflow":
                    status("reflow", builder.stats.describe())
            shown = len(doc_fmts) + len(page_fmts)
            ok(f"{len(files)} pages in {time.perf_counter() - t1:.1f} s -> " + ", ".join(str(p) for p in written[:shown])
               + (" ..." if len(written) > shown else ""))
            return

        failed = []
        reflow_stats = None
        t1 = time.perf_counter()
        with tqdm(files, unit="page", desc="OCR", dynamic_ncols=True, leave=False) as bar:
            for f in bar:
                bar.set_postfix_str(f.name, refresh=False)
                try:
                    page = Page.load(f, number=1)
                    contents = pipe.process_page(page)
                    target = out_dir or f.parent
                    if doc_fmts:
                        builder = DocumentBuilder(f.stem, sections=sections, text=text_mode, rtl=rtl)
                        builder.add_page(page, contents)
                        export(builder.build(), target, tagged(f.stem), doc_fmts)
                        if text_mode == "reflow":
                            if reflow_stats is None:
                                reflow_stats = builder.stats
                            else:
                                reflow_stats += builder.stats
                    write_page_formats(page, contents, target, tagged(f.stem))
                except Exception as e:  # noqa: BLE001 - keep going, report at the end
                    failed.append(f)
                    tqdm.write(click.style(f"! {f.name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", fg="yellow"), file=sys.stderr)
        dt = time.perf_counter() - t1
        done = len(files) - len(failed)
        if reflow_stats is not None:
            status("reflow", reflow_stats.describe())
            if os.environ.get("SQUIDDLE_VERBOSE"):
                for ex in reflow_stats.examples:
                    status("", ex)
        ok(f"{done} of {len(files)} pages in {dt:.1f} s ({dt / max(done, 1):.2f} s/page) -> {out_dir or 'next to the images'}")
        if failed:
            raise fail(f"{len(failed)} page(s) failed: " + ", ".join(f.name for f in failed[:5]) + (" ..." if len(failed) > 5 else ""))
    finally:
        pipe.close()


@main.group()
def models():
    """Fetch and locate model files ahead of a run (they are also downloaded on first use)."""


@models.command("pull")
@click.argument("names", nargs=-1, type=click.Choice(PULLABLE), required=True)
def models_pull(names):
    """Download models: eynollah (the layout bundle from Zenodo, 1.8 GB), tiny / small / medium (kraken's
    PP-OCRv6 recognisers) or blla (kraken's segmenter). A model that is already there is not fetched again."""
    from .models import KRAKEN_SEGMENTER_DOI, eynollah_model_dir, kraken_model_file, resolve_kraken_model

    for name in names:
        try:
            if name == "eynollah":
                bars: list = []

                def progress(done, total):
                    if not bars:
                        bars.append(tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="eynollah models",
                                         dynamic_ncols=True, leave=False))
                    bars[0].n = done
                    bars[0].refresh()

                try:
                    path = eynollah_model_dir(log=lambda s: tqdm.write(s, file=sys.stderr), progress=progress)
                finally:
                    for b in bars:
                        b.close()
            elif name == "blla":
                path = kraken_model_file(KRAKEN_SEGMENTER_DOI, (".mlmodel", ".safetensors"), log=lambda s: status("models", s))
            else:
                path = resolve_kraken_model(name, log=lambda s: status("models", s))
        except (RuntimeError, FileNotFoundError, OSError) as e:
            raise fail(f"{name}: {e}") from e
        ok(f"{name}: {path}")


@models.command("path")
@click.argument("name", type=click.Choice(PULLABLE))
def models_path(name):
    """Print where a model lives (eynollah: the directory holding models_eynollah/); fails if it is not there."""
    from .models import KRAKEN_SEGMENTER_DOI, eynollah_model_dir, kraken_model_file, resolve_kraken_model

    try:
        if name == "eynollah":
            path = eynollah_model_dir(download=False)
        elif name == "blla":
            path = kraken_model_file(KRAKEN_SEGMENTER_DOI, (".mlmodel", ".safetensors"), log=lambda s: None)
        else:
            path = resolve_kraken_model(name, log=lambda s: None)
    except (RuntimeError, FileNotFoundError, OSError) as e:
        raise fail(str(e)) from e
    click.echo(str(path))
