"""CTC greedy decoding shared by the recogniser and the converter's verification."""
from __future__ import annotations

import numpy as np


def ctc_greedy_decode(probs: np.ndarray, chars: list[str], blank: int = 0) -> tuple[str, float]:
    """Greedy CTC decode of a ``(T, C)`` probability matrix into text.

    Mirrors PaddleX's ``CTCLabelDecode``: argmax per time step, collapse repeats,
    drop blanks; the score is the mean probability of the kept time steps.
    ``chars[i]`` is class ``i + 1``.
    """
    probs = np.asarray(probs)
    idx = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    keep = np.ones_like(idx, dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    keep &= idx != blank
    labels = idx[keep]
    text = "".join(chars[i - 1] for i in labels)
    score = float(conf[keep].mean()) if keep.any() else 0.0
    return text, score
