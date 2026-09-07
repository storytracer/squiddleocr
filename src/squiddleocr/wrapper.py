"""Wrap a kraken PP-OCRv6 network so it consumes PaddleOCR-preprocessed input and emits CTC probabilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: kraken's default extra white padding (px) on both ends of a line.
DEFAULT_PADDING = 16


class PaddleRecWrapper(nn.Module):
    """Adapter from PaddleOCR's recognition contract to kraken's PP-OCRv6 network.

    PaddleOCR feeds ``(N, 3, H, W)`` float32 images normalised to ``[-1, 1]``
    (``x/255 - 0.5) / 0.5``) and expects ``(N, W', C)`` softmax probabilities with
    the CTC blank at index 0.

    kraken's network was trained on ``1 - x/255`` (ink bright, paper dark) with
    16 px of white padding at both ends. This module folds that difference into
    the graph:

    1. ``x_kraken = 0.5 - 0.5 * x_paddle`` (equals ``1 - x/255``),
    2. pad ``padding`` columns of white (``0`` after inversion) left and right,
    3. run the network, drop the singleton height axis, permute to ``(N, W', C)``,
    4. softmax over classes.

    With ``adapt_input=False`` the wrapper expects kraken's own tensor (already
    inverted and padded) and only does steps 3 and 4; used for parity tests.
    """

    def __init__(self, net: nn.Module, padding: int = DEFAULT_PADDING, adapt_input: bool = True):
        super().__init__()
        self.net = net
        self.padding = int(padding)
        self.adapt_input = bool(adapt_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapt_input:
            x = 0.5 - 0.5 * x
            if self.padding > 0:
                x = F.pad(x, (self.padding, self.padding), value=0.0)
        logits, _ = self.net(x, None)          # (N, C, 1, W')
        logits = logits.squeeze(2).permute(0, 2, 1)  # (N, W', C)
        return torch.softmax(logits, dim=-1)


def kraken_to_paddle_input(x_kraken: torch.Tensor) -> torch.Tensor:
    """Invert :class:`PaddleRecWrapper`'s input adaptation (without the padding).

    Given kraken's inverted 0..1 tensor returns the ``[-1, 1]`` tensor PaddleOCR
    would produce for the same pixels: ``x_paddle = 1 - 2 * x_kraken``.
    """
    return 1.0 - 2.0 * x_kraken
