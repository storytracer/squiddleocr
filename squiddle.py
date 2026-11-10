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
r"""SquiddleOCR: linecaster's squiddle recipe on its modular pipeline, exactly as linecaster runs it. PP-DocLayoutV3
finds the regions and their reading order, PP-OCRv6 medium detection the lines, kraken's PP-OCRv6 medium recognizer
reads them, SLANet_plus rebuilds tables and PP-FormulaNet_plus-L reads formulas as LaTeX; then the typography
(headings), the recomposition (paragraphs, line-end hyphens removed) and each page's DoclingDocument.

    uv run squiddle.py IMAGES OUT                         # the page records in OUT/pages
    uv run squiddle.py IMAGES OUT --files OUT/files       # also one file per page and format (--formats)
    uv run squiddle.py IMAGES OUT --tasks 4 --workers 2
    uv run squiddle.py serve                              # linecaster serve with the squiddle models

This is ``uv run pipelines/transcribe_modular.py IMAGES OUT --recipe squiddle`` in linecaster, in an environment of its
own: the options and their defaults are linecaster's (``linecaster.pipelines.transcribe_modular``,
``linecaster.runtime.run``); the script only names the recipe.
"""
from linecaster.pipelines import transcribe_modular as shape
from linecaster.runtime import run


def options(ap):
    shape.options(ap)
    ap.set_defaults(recipe="squiddle")    # run adds --recipe to the same parser before these options


if __name__ == "__main__":
    run.main(__file__, __doc__, options, shape.models, shape.pipeline, shape.PIPELINE, shape.SLOTS)
