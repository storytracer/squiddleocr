# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "linecaster",
#     # the squiddle recipe's engines; each brings what it needs (linecaster-kraken the kraken fork and torch,
#     # linecaster-paddlex the OpenCV stub)
#     "linecaster-kraken",
#     "linecaster-paddlex",
# ]
#
# [tool.uv.sources]
# # linecaster and the two engine packages from linecaster's release tag
# linecaster = { git = "https://github.com/storytracer/linecaster", tag = "v0.5.2" }
# linecaster-kraken = { git = "https://github.com/storytracer/linecaster", tag = "v0.5.2", subdirectory = "engines/kraken" }
# linecaster-paddlex = { git = "https://github.com/storytracer/linecaster", tag = "v0.5.2", subdirectory = "engines/paddlex" }
# ///
r"""SquiddleOCR: linecaster's squiddle recipe on its modular pipeline, exactly as linecaster runs it. PP-DocLayoutV3
finds the regions and their reading order, PP-OCRv6 medium detection the lines, kraken's PP-OCRv6 medium recognizer
reads them, SLANet_plus rebuilds tables and PP-FormulaNet_plus-L reads formulas as LaTeX; then the typography
(headings), the recomposition (paragraphs, line-end hyphens removed) and each page's DoclingDocument.

    uv run squiddle.py IMAGES OUT                         # the page records in OUT/pages
    uv run squiddle.py IMAGES OUT --files OUT/files       # also one file per page and format (--formats)
    uv run squiddle.py IMAGES OUT --tasks 4 --workers 2
    uv run squiddle.py serve                              # linecaster serve with the squiddle models

This is linecaster's own recipe script, ``recipes/squiddle.py``, pinned to a linecaster release instead of its
``main`` branch: the options and their defaults are linecaster's (``linecaster.pipelines.transcribe_modular``,
``linecaster.runtime.run``); the script only names the recipe.
"""
from linecaster.runtime import run

if __name__ == "__main__":
    run.recipe(__file__, "squiddle")
