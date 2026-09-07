import pytest
import torch

from squiddleocr.convert.wrapper import PaddleRecWrapper, kraken_to_paddle_input, output_length

H = 96


@pytest.mark.parametrize("width", [64, 65, 96, 100, 127, 320, 641, 1000])
def test_output_length_matches_backbone(tiny_net, width):
    with torch.inference_mode():
        feat = tiny_net.backbone(torch.zeros(1, 3, H, width))
    assert feat.shape[3] == output_length(width)


def test_wrapper_shape_and_softmax(tiny_net):
    w = PaddleRecWrapper(tiny_net, padding=16)
    x = torch.rand(2, 3, H, 200) * 2 - 1
    with torch.inference_mode():
        y = w(x)
    assert y.shape == (2, output_length(200 + 32), 12)
    assert torch.allclose(y.sum(-1), torch.ones(2, y.shape[1]), atol=1e-5)
    assert (y >= 0).all()


def test_wrapper_matches_kraken_forward(tiny_net):
    """Paddle-space input through the wrapper == kraken-space input through the raw net."""
    torch.manual_seed(1)
    xk = torch.rand(1, 3, H, 300)               # kraken space, unpadded
    xk_padded = torch.nn.functional.pad(xk, (16, 16), value=0.0)
    with torch.inference_mode():
        logits, _ = tiny_net(xk_padded, None)
        ref = logits.squeeze(2).permute(0, 2, 1).softmax(-1)
        out = PaddleRecWrapper(tiny_net, padding=16)(kraken_to_paddle_input(xk))
    assert torch.allclose(out, ref, atol=1e-5)


def test_trailing_padding_detection():
    x = torch.rand(2, 3, H, 50) * 2 - 1
    x[0, :, :, 30:] = 0.0          # batch padding
    x[1, :, 10, 20] = 0.0          # a single zero pixel is not padding
    pad = PaddleRecWrapper.trailing_padding(x)
    assert pad[0].tolist() == [False] * 30 + [True] * 20
    assert not pad[1].any()


def test_batched_padded_lines_are_masked_to_blank(tiny_net):
    """Time steps past a line's length are forced to the blank class; valid steps use kraken's batch masking."""
    torch.manual_seed(2)
    w = PaddleRecWrapper(tiny_net, padding=16)
    a = torch.rand(1, 3, H, 120) * 2 - 1
    b = torch.rand(1, 3, H, 400) * 2 - 1
    batch = torch.zeros(2, 3, H, 400)
    batch[0, :, :, :120] = a[0]
    batch[1] = b[0]
    with torch.inference_mode():
        y = w(batch)
        yb = w(b)
    la = output_length(120 + 32)
    assert torch.equal(y[0, la:].argmax(-1), torch.zeros(y.shape[1] - la, dtype=torch.long))
    assert torch.allclose(y[0, la:, 0], torch.ones(y.shape[1] - la))
    # the full-width line is unaffected by being batched
    assert torch.allclose(y[1], yb[0], atol=1e-5)
