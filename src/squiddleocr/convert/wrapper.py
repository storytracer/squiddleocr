"""Wrap a kraken PP-OCRv6 network so it consumes PaddleOCR-preprocessed input and emits CTC probabilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: kraken's default extra white padding (px) on both ends of a line.
DEFAULT_PADDING = 16


def output_length(width: torch.Tensor | int) -> torch.Tensor | int:
    """Number of output time steps the PP-OCRv6 backbone produces for an input of ``width`` px.

    Stem with stride 4 followed by a stride-2 stage, both with "same"-style
    padding: ``((width - 1) // 4 - 1) // 2 + 1``. Verified against the network
    for many widths in the test-suite.
    """
    return ((width - 1) // 4 - 1) // 2 + 1


class PaddleRecWrapper(nn.Module):
    """Adapter from PaddleOCR's recognition contract to kraken's PP-OCRv6 network.

    PaddleOCR feeds ``(N, 3, H, W)`` float32 images normalised to ``[-1, 1]``
    (``(x/255 - 0.5) / 0.5``), right-pads shorter lines of a batch with ``0``
    (mid-grey), and expects ``(N, W', C)`` softmax probabilities with the CTC
    blank at index 0.

    kraken's network was trained on ``1 - x/255`` (ink bright, paper dark) with
    16 px of white padding at both ends, and kraken batches lines by padding
    with white and masking the padded time steps in the SVTR neck. This module
    folds those differences into the graph so that a PaddleOCR batch gives the
    same result as running each line alone through kraken:

    1. *batch padding detection*: trailing columns whose values are all exactly
       ``0.0`` can only be PaddleOCR's batch padding (a real pixel would need
       the value 127.5/255), so they are turned into white and excluded from
       the per-line width;
    2. ``x_kraken = 0.5 - 0.5 * x_paddle`` (equals ``1 - x/255``);
    3. pad ``padding`` columns of white (``0`` after inversion) left and right;
    4. run the backbone, derive each line's valid number of time steps with
       :func:`output_length`, run the neck with the validity mask (as kraken
       does) and the CTC head;
    5. softmax over classes; time steps beyond a line's length are forced to
       the blank so PaddleOCR's decoder, which reads every time step, ignores
       them.

    With ``adapt_input=False`` the wrapper expects kraken's own tensor (already
    inverted and padded) and skips steps 1-3 and the masking; used for parity
    tests against kraken's own forward pass.
    """

    def __init__(self, net: nn.Module, padding: int = DEFAULT_PADDING, adapt_input: bool = True):
        super().__init__()
        self.net = net
        self.padding = int(padding)
        self.adapt_input = bool(adapt_input)

    @staticmethod
    def trailing_padding(x: torch.Tensor) -> torch.Tensor:
        """Boolean ``(N, W)`` mask of trailing all-zero columns (PaddleOCR batch padding)."""
        nonzero_cols = (x != 0).any(dim=1).any(dim=1).to(torch.int64)   # (N, W)
        # a column is padding only if no column at or to its right is non-zero
        # (reverse cumulative sum; CumSum is ONNX-friendly, CumProd is not)
        return nonzero_cols.flip(-1).cumsum(-1).flip(-1) == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        if not self.adapt_input:
            logits, _ = net(x, None)                                    # (N, C, 1, W')
            return torch.softmax(logits.squeeze(2).permute(0, 2, 1), dim=-1)

        pad = self.trailing_padding(x)
        line_widths = x.shape[3] - pad.sum(-1)                          # (N,) real px per line
        x = x.masked_fill(pad[:, None, None, :], 1.0)                   # white in [-1, 1] space
        x = 0.5 - 0.5 * x                                               # kraken space
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), value=0.0)

        feat = net.backbone(x)                                          # (N, C, 1, W')
        w_out = feat.shape[3]
        out_lens = output_length(line_widths + 2 * self.padding).clamp(min=1)
        valid = torch.arange(w_out, device=x.device)[None, :] < out_lens[:, None]   # (N, W')
        seq = net.neck(feat, valid)                                     # (N, W', Cn)
        logits = net.head(seq)                                          # (N, W', classes)
        probs = torch.softmax(logits, dim=-1)
        blank = torch.zeros(probs.shape[-1], dtype=probs.dtype, device=probs.device)
        blank[0] = 1.0
        return torch.where(valid[:, :, None], probs, blank)


def kraken_to_paddle_input(x_kraken: torch.Tensor) -> torch.Tensor:
    """Invert :class:`PaddleRecWrapper`'s input adaptation (without the padding).

    Given kraken's inverted 0..1 tensor returns the ``[-1, 1]`` tensor PaddleOCR
    would produce for the same pixels: ``x_paddle = 1 - 2 * x_kraken``.
    """
    return 1.0 - 2.0 * x_kraken
