"""``python -m squiddleocr.eynollah.launch <eynollah arguments>``: eynollah's own CLI, configured from outside.

Everything the run plan decided arrives in the environment (``EynollahRunner`` sets it) and is applied
here before eynollah loads anything:

- ``SQUIDDLE_EYNOLLAH_VRAM``: a JSON dict of per-model MB that replaces entries of eynollah's
  ``MODEL_VRAM_LIMITS`` (the ONNX Runtime arena cap per model; the hard-coded values are too small for
  Blackwell / unified memory, where cuDNN picks larger workspaces). This is the one place eynollah is
  patched at run time, in memory only.
- ``SQUIDDLE_EYNOLLAH_THREADS``: ``cv2.setNumThreads`` for the page jobs (OpenMP/BLAS thread counts are
  plain environment variables and need no code).
- ``EYNOLLAH_ONNX_EP``: eynollah's own provider override, read by its model zoo.

The patch runs at import time on purpose: eynollah's per-model predictors are ``spawn``ed processes,
which re-import the main module as ``__mp_main__`` and load the ONNX sessions there, so a patch
inside ``if __name__ == "__main__"`` would never reach them.
"""
from __future__ import annotations

import json
import os
import sys


def apply_settings(environ=os.environ) -> dict[str, int]:
    """Patch ``MODEL_VRAM_LIMITS`` and the OpenCV thread count from the environment; returns the caps applied."""
    limits = environ.get("SQUIDDLE_EYNOLLAH_VRAM")
    applied: dict[str, int] = {}
    if limits:
        from eynollah.model_zoo import model_zoo

        applied = {k: int(v) for k, v in json.loads(limits).items()}
        model_zoo.MODEL_VRAM_LIMITS.update(applied)
    threads = environ.get("SQUIDDLE_EYNOLLAH_THREADS")
    if threads:
        try:
            import cv2

            cv2.setNumThreads(int(threads))
        except ImportError:  # pragma: no cover
            pass
    return applied


apply_settings()


def main(argv: list[str] | None = None) -> None:
    from eynollah.cli import main as eynollah_main

    eynollah_main(args=sys.argv[1:] if argv is None else argv, prog_name="eynollah")


if __name__ == "__main__":
    main()
