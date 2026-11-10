# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "linecaster",
#     "linecaster-kraken",
#     "linecaster-paddlex",
#     # direct, so the fork in [tool.uv.sources] applies (sources only apply to a script's own dependencies)
#     "kraken",
#     # PaddleX pins OpenCV's GUI build; linecaster's stub swaps in the headless build of the same version
#     "opencv-contrib-python",
# ]
#
# [tool.uv]
# # the versions the goldens were recorded with (numerics: recognition, ONNX Runtime, CUDA, decoding); the same list
# # as linecaster's pipelines/transcribe_modular.py
# constraint-dependencies = [
#     "lightning==2.6.6",
#     "onnxruntime-gpu==1.29.0",
#     "pillow==12.3.0",
#     "transformers==5.12.1",
#     "tokenizers==0.22.2",
#     "nvidia-cublas==13.1.1.3",
#     "nvidia-cuda-nvrtc==13.0.88",
#     "nvidia-cuda-runtime==13.0.96",
#     "nvidia-cudnn-cu13==9.24.0.43",
#     "nvidia-cufft==12.0.0.61",
#     "nvidia-curand==10.4.0.35",
#     "scipy==1.15.3",
#     "scikit-image==0.25.2",
# ]
#
# [tool.uv.sources]
# # UNTIL THE RELEASE: linecaster from the checkout next to this one (../linecaster). At the release these four
# # switch to linecaster's tag v0.3.0 (NOTES, 2026-11-10):
# #   linecaster = { git = "https://github.com/storytracer/linecaster", tag = "v0.3.0" }
# #   linecaster-kraken = { git = "https://github.com/storytracer/linecaster", tag = "v0.3.0", subdirectory = "engines/kraken" }
# #   linecaster-paddlex = { git = "https://github.com/storytracer/linecaster", tag = "v0.3.0", subdirectory = "engines/paddlex" }
# #   opencv-contrib-python = { git = "https://github.com/storytracer/linecaster", tag = "v0.3.0", subdirectory = "stubs/opencv-contrib-python" }
# linecaster = { path = "../linecaster", editable = true }
# linecaster-kraken = { path = "../linecaster/engines/kraken", editable = true }
# linecaster-paddlex = { path = "../linecaster/engines/paddlex", editable = true }
# opencv-contrib-python = { path = "../linecaster/stubs/opencv-contrib-python" }
# # kraken with the PP-OCRv6 code from our fork: upstream main (bb487219) plus the inference precision fix and the
# # line prefetch
# kraken = { git = "https://github.com/storytracer/kraken", rev = "48eecb78" }
# ///
r"""SquiddleOCR: the squiddle recipe on linecaster's modular pipeline. PP-DocLayoutV3 finds the regions and their
reading order, PP-OCRv6 medium detection the lines, kraken's PP-OCRv6 medium recognizer reads them, SLANet_plus
rebuilds tables and PP-FormulaNet_plus-L reads formulas as LaTeX; then the typography (headings), the recomposition
(paragraphs, line-end hyphens removed) and each page's DoclingDocument. Files go flat into OUT, one per page and
format, named <page>.paddle.<ext>, next to the page records (OUT/pages), OUT/rejects (the pages that failed: they
get no files), OUT/logs, OUT/traces and the run record OUT/run.yaml.

    uv run squiddle.py IMAGES OUT                                      # <page>.paddle.md
    uv run squiddle.py IMAGES OUT --formats md,txt,hocr,alto,page,json
    uv run squiddle.py IMAGES OUT --furniture --set composition.sections=true
    uv run squiddle.py IMAGES OUT --recognition kraken-ppocrv6-small   # another preset in a slot (an override)
    uv run squiddle.py IMAGES OUT --tasks 4 --workers 2
    uv run squiddle.py IMAGES OUT --backend served                     # the models behind linecaster serve
    uv run squiddle.py serve                                           # linecaster serve with the squiddle models

The options are those of linecaster's modular pipeline (``linecaster.pipelines.transcribe_modular``) and of every
linecaster pipeline (``linecaster.runtime.run``), with SquiddleOCR's defaults where the command line leaves them
open: --recipe squiddle, --formats md, --naming squiddleocr, --files OUT.
"""
from linecaster.pipelines import transcribe_modular as shape
from linecaster.runtime import run

#: SquiddleOCR's defaults over the modular pipeline's options (``--files`` defaults to OUT in ``pipeline``)
DEFAULTS = {"recipe": "squiddle", "formats": "md", "naming": "squiddleocr"}


def options(ap):
    shape.options(ap)
    ap.set_defaults(**DEFAULTS)           # --recipe too: run adds it to the same parser before these options


def pipeline(a, script: str):
    if a.files is None:                   # the files next to the run's own folders, as squiddle ocr -o OUT wrote them
        a.files = a.out
    return shape.pipeline(a, script)


if __name__ == "__main__":
    run.main(__file__, __doc__, options, shape.models, pipeline, shape.PIPELINE, shape.SLOTS)
