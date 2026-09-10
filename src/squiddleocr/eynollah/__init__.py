"""eynollah (SBB's historical layout analyser) as a layout and line stage, run as a subprocess.

``resources`` detects the machine and computes the run plan (jobs, threads, per-model VRAM caps),
``launch`` is the ``python -m squiddleocr.eynollah.launch`` entry that applies the plan and calls
eynollah's CLI, ``runner`` starts and watches that subprocess, ``pagexml`` reads the PAGE-XML it
writes, ``source`` hands parsed pages to the ``EynollahLayout`` and ``EynollahLines`` stages.
"""
from .source import EynollahOptions, EynollahSource

__all__ = ["EynollahOptions", "EynollahSource"]
